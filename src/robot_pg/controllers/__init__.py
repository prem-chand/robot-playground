"""Classical controller blocks, plus the closed-loop driver that serves a
scenario reference to a graph's external slot.

Each block is robot-agnostic config: gains and geometry go to the constructor,
and ports connect by name. `drive` builds the graph with the reference slot and
runs it through `robot_pg.runtime.Session`.
"""

from collections.abc import Mapping
from math import atan2, sin
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from robot_pg.contract import Array, BlockSpec, External, Graph, SignalSpec
from robot_pg.runtime import Session
from robot_pg.scenarios import Path, Reference, Targets, whole_ticks
from robot_pg.trajectory import Trajectory

if TYPE_CHECKING:
    from robot_pg.contract import Block
    from robot_pg.robots import Episode

__all__ = ["Driver", "JointPd", "PathPursuit", "Reach", "Setpoint", "drive"]


class Driver:
    """Feeds a slot that declares the reference's signal spec. Call 1 serves the
    value at tick 0, and each call steps `period` ticks.
    """

    def __init__(self, reference: Reference, dt: float, period: int = 1) -> None:
        if period < 1:
            raise ValueError(f"period {period} is not one tick or more")
        self._ref, self._dt, self._period, self._n = reference, dt, period, 0

    def step(self) -> dict[str, ArrayLike]:
        vals = self._ref.values(np.array([self._n * self._period]), self._dt)
        self._n += 1
        return {self._ref.signal.name: vals[0]}


def drive(
    model: mujoco.MjModel,
    episode: "Episode",
    blocks: Mapping[str, "Block | External"],
    command: tuple[str, ...],
    duration: float,
    seed: int = 0,
    record: tuple[str, ...] = (),
) -> Trajectory:
    """Run a graph of blocks plus the reference slot on an episode model."""
    dt = model.opt.timestep
    ref = episode.reference
    graph = Graph(
        {"ref": External(BlockSpec(1 / dt, outputs=(ref.signal,))), **blocks},
        command,
    )
    ticks = whole_ticks(duration, dt)
    session = Session(model, graph, seed, record=record, capacity=ticks)
    driver = Driver(ref, dt)
    while session.advance(ticks) is not None:
        session.set_slot(driver.step())
    return session.trajectory()


class Setpoint:
    """A fixed target. `out` names the port, so any block can wire to it."""

    def __init__(self, out: SignalSpec, value: ArrayLike) -> None:
        self.value = np.asarray(value, dtype=np.float64)
        if self.value.shape != out.shape:
            raise ValueError(f"{out.name} value is {self.value.shape}, not {out.shape}")
        self.spec = BlockSpec(500.0, outputs=(out,))
        self.digest = f"set {out.name} {self.value.tolist()}"

    def reset(self, seed: int) -> None: ...

    def step(self, inputs: Mapping[str, Array]) -> dict[str, Array]:
        return {self.spec.outputs[0].name: self.value}


class JointPd:
    """A torque command `kp * (target - q) - kd * qd`. `target` is an input, so
    a `Setpoint`, a slot, or another block can feed it."""

    def __init__(self, kp: float, kd: float, out: SignalSpec) -> None:
        if out.unit not in ("N m", "N"):
            raise ValueError(f"{out.name} unit is {out.unit!r}; JointPd outputs force")
        n = out.shape[-1]
        self.spec = BlockSpec(
            500.0,
            inputs=(
                SignalSpec("target", (n,)),
                SignalSpec("qpos_joints", (n,)),
                SignalSpec("qvel_joints", (n,)),
            ),
            outputs=(out,),
        )
        self.kp, self.kd = kp, kd
        self.digest = f"kp {kp} kd {kd}"

    def reset(self, seed: int) -> None: ...

    def step(self, inputs: Mapping[str, Array]) -> dict[str, Array]:
        err = inputs["target"] - inputs["qpos_joints"]
        return {
            self.spec.outputs[0].name: self.kp * err - self.kd * inputs["qvel_joints"]
        }


class PathPursuit:
    """Diff-drive follower of a `ref_path` polyline. It projects the base onto
    the polyline near its last arc length, commands `speed` along it, and steers
    to the point `lookahead` ahead on the path.
    """

    def __init__(
        self,
        track_m: float,
        wheel_radius_m: float,
        lookahead_m: float,
        length_m: float,
        speed_m_s: float,
    ) -> None:
        if track_m <= 0 or wheel_radius_m <= 0 or lookahead_m <= 0 or length_m <= 0:
            raise ValueError("geometry needs positive values")
        self.spec = BlockSpec(
            500.0,
            inputs=(
                Path.signal,
                SignalSpec("base_pos", (3,), "m"),
                SignalSpec("base_quat", (4,)),
            ),
            outputs=(SignalSpec("w", (2,), "rad/s"),),
        )
        self.track, self.radius = track_m, wheel_radius_m
        self.lookahead, self.length, self.speed = lookahead_m, length_m, speed_m_s
        self._s = 0.0
        self.digest = (
            f"track {track_m:g} radius {wheel_radius_m:g} "
            f"lookahead {lookahead_m:g} speed {speed_m_s:g}"
        )

    def reset(self, seed: int) -> None:
        self._s = 0.0

    def step(self, inputs: Mapping[str, Array]) -> dict[str, Array]:
        pts = inputs["ref_path"]
        spacing = self.length / (pts.shape[0] - 1)
        self._s = _project(
            inputs["base_pos"][:2], self._s, pts, spacing, self.lookahead
        )
        out = self.spec.outputs[0].name
        if self._s >= self.length - spacing:
            # The nominal point stops at the end of the path, so the base stops.
            return {out: np.zeros(2)}
        goal = _at(self._s + self.lookahead, pts, spacing)
        d = goal - inputs["base_pos"][:2]
        alpha = atan2(d[1], d[0]) - _yaw(inputs["base_quat"])
        v = self.speed
        turn = 2 * v * sin(alpha) / self.lookahead
        half = self.track / 2
        return {
            out: np.array([(v - turn * half) / self.radius, (v + turn * half) / self.radius])
        }


class Reach:
    """Damped-least-squares reach of a `ref_ee_target` position. It steps a
    position command from the joint state along the Jacobian, so both `step_gain`
    and `damping` brace the solve. `arm` is the actuator count the end-effector
    follows; a `hand` command for one slide actuator rides along.
    """

    def __init__(
        self,
        arm: int,
        hand: float | None = None,
        step_gain: float = 2.0,
        damping: float = 0.1,
    ) -> None:
        if arm <= 0 or step_gain <= 0 or damping <= 0:
            raise ValueError("arm, step_gain, and damping need positive values")
        nu = arm + (0 if hand is None else 1)
        self.spec = BlockSpec(
            500.0,
            inputs=(
                Targets.signal,
                SignalSpec("ee_pos", (3,), "m"),
                SignalSpec("ee_jacobian", (3, nu)),
                SignalSpec("qpos_joints", (nu,)),
            ),
            outputs=(
                SignalSpec("arm", (arm,), "rad"),
                *([SignalSpec("hand", (1,), "m")] if hand is not None else ()),
            ),
        )
        self.arm, self.hand, self.step_gain, self.damping = arm, hand, step_gain, damping
        self.digest = f"arm {arm} step {step_gain:g} damping {damping:g}"

    def reset(self, seed: int) -> None: ...

    def step(self, inputs: Mapping[str, Array]) -> dict[str, Array]:
        jac = inputs["ee_jacobian"][:, : self.arm]
        err = inputs["ref_ee_target"] - inputs["ee_pos"]
        step = self.step_gain * np.linalg.solve(
            jac @ jac.T + self.damping**2 * np.identity(3), err
        )
        target = inputs["qpos_joints"][: self.arm] + jac.T @ step
        out = {self.spec.outputs[0].name: target}
        if self.hand is not None:
            out[self.spec.outputs[1].name] = np.array([self.hand])
        return out


def _project(
    p: Array, s_now: float, pts: Array, spacing: float, lookahead: float
) -> float:
    """Nearest arc length to `p` on the polyline. The search window covers a bit
    back from `s_now` and one lookahead forward, so a path that crosses itself
    stays on its branch."""
    a, b = pts[:-1], pts[1:]
    seg = b - a
    sq = np.einsum("ij,ij->i", seg, seg)
    t = np.clip(np.einsum("ij,ij->i", p - a, seg) / sq, 0.0, 1.0)
    near = a + t[:, None] * seg
    dist = np.linalg.norm(p - near, axis=1)
    s = np.arange(len(seg)) * spacing + t * spacing
    visible = ((s >= s_now - spacing) & (s <= s_now + lookahead + spacing)).nonzero()[0]
    return float(s[visible[np.argmin(dist[visible])]])


def _yaw(quat: Array) -> float:
    w, x, y, z = quat
    return atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _at(s: float, pts: Array, spacing: float) -> Array:
    f = np.clip(s / spacing, 0.0, len(pts) - 1.0)
    i = int(f)
    return (1 - f + i) * pts[i] + (f - i) * pts[min(i + 1, len(pts) - 1)]
