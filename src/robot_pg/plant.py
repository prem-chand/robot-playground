"""A model opts in to two conventions by name. A site named `ee` publishes
`ee_pos` and `ee_jacobian`, and a keyframe named `home` is the reset state.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import mujoco
import numpy as np
from numpy.typing import NDArray

from robot_pg.contract import Array, Graph, SignalSpec
from robot_pg.schedule import GraphError

if TYPE_CHECKING:
    from robot_pg.trajectory import Trajectory

Kind = Literal["torque", "position", "velocity"]
UNITS: Mapping[str, tuple[Kind, bool]] = {
    "N m": ("torque", True),
    "N": ("torque", False),
    "rad": ("position", True),
    "m": ("position", False),
    "rad/s": ("velocity", True),
    "m/s": ("velocity", False),
}
EE_SITE = "ee"
HOME_KEY = "home"

_W = mujoco.mjtWarning
_FATAL = (
    ("qacc", "dof", _W.mjWARN_BADQACC),
    ("qvel", "dof", _W.mjWARN_BADQVEL),
    ("qpos", "qpos", _W.mjWARN_BADQPOS),
    ("ctrl", "actuator", _W.mjWARN_BADCTRL),
)
_FATAL_LO = min(int(w) for _, _, w in _FATAL)
_FATAL_HI = max(int(w) for _, _, w in _FATAL) + 1


class RunUnstable(RuntimeError):
    """After a bad qpos, qvel, or qacc MuJoCo resets the state, and after a bad
    ctrl it applies no actuator force. So the run no longer follows its command.

    `time` is the start of the failed tick. `Session` sets `trajectory` to every
    tick up to and including the failed one.
    """

    def __init__(self, what: str, where: str, index: int, tick: int, dt: float) -> None:
        self.what, self.index, self.tick, self.time = what, index, tick, tick * dt
        self.trajectory: Trajectory | None = None
        super().__init__(f"bad {what} at {where} {index}, t={self.time:.4f}")


@dataclass(frozen=True)
class Actuator:
    """One actuator in model order. `ctrl = clip(command * scale, ctrl_range)`."""

    name: str
    kind: Kind
    unit: str
    scale: float
    ctrl_range: tuple[float, float]
    command_range: tuple[float, float]


@dataclass(frozen=True)
class CtrlMap:
    scale: Array
    lo: Array
    hi: Array

    @classmethod
    def of(cls, actuators: tuple[Actuator, ...]) -> "CtrlMap":
        lo, hi = np.array([a.ctrl_range for a in actuators]).reshape(-1, 2).T
        return cls(np.array([a.scale for a in actuators]), lo.copy(), hi.copy())

    def apply(self, command: Array, out: Array | None = None) -> Array:
        out = np.multiply(command, self.scale, out=out)
        np.minimum(out, self.hi, out=out)
        return np.maximum(out, self.lo, out=out)

    def clipped(self, command: Array) -> NDArray[np.bool_]:
        raw = command * self.scale
        return (raw < self.lo) | (raw > self.hi)


@dataclass(frozen=True)
class PlantSpec:
    actuators: tuple[Actuator, ...]
    signals: tuple[SignalSpec, ...]
    dof_map: Array


class _Target(NamedTuple):
    actuator: str
    joints: tuple[tuple[int, float], ...]


def parse_model(model: mujoco.MjModel) -> PlantSpec:
    if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
        raise ValueError("integrator RK4 cannot split into mj_step1 and mj_step2")
    if len(_free_joints(model)) > 1:
        raise ValueError("model has more than one free joint")
    targets = [_target(model, i) for i in range(model.nu)]
    actuators = tuple(_read_actuator(model, i, t) for i, t in enumerate(targets))
    dof_map = np.zeros((model.nv, model.nu))
    for i, t in enumerate(targets):
        norm = sum(c * c for _, c in t.joints)
        for j, c in t.joints:
            dof_map[model.jnt_dofadr[j], i] = c / norm
    nu = (model.nu,)
    signals = [SignalSpec("qpos_joints", nu), SignalSpec("qvel_joints", nu)]
    if _free_joints(model):
        signals += [
            SignalSpec("base_pos", (3,), "m"),
            SignalSpec("base_quat", (4,), "", (1.0, 0.0, 0.0, 0.0)),
            SignalSpec("base_linvel_body", (3,), "m/s"),
            SignalSpec("base_angvel_body", (3,), "rad/s"),
        ]
    if _site(model) >= 0:
        signals += [
            SignalSpec("ee_pos", (3,), "m"),
            SignalSpec("ee_jacobian", (3, *nu)),
        ]
    signals.append(SignalSpec("effort", nu))
    return PlantSpec(actuators, tuple(signals), dof_map)


def _free_joints(model: mujoco.MjModel) -> list[int]:
    return [
        j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]


def _site(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)


def _target(model: mujoco.MjModel, i: int) -> _Target:
    trn, tid = model.actuator_trntype[i], int(model.actuator_trnid[i, 0])
    name, joints = None, []
    if trn == mujoco.mjtTrn.mjTRN_JOINT:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, tid)
        joints = [(tid, 1.0)]
    elif trn == mujoco.mjtTrn.mjTRN_TENDON:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tid)
        start = model.tendon_adr[tid]
        wraps = range(start, start + model.tendon_num[tid])
        if all(model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_JOINT for w in wraps):
            joints = [
                (int(model.wrap_objid[w]), float(model.wrap_prm[w])) for w in wraps
            ]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or name or ""
    if not joints:
        raise ValueError(f"actuator {name!r} must drive a joint or a fixed tendon")
    return _Target(name, tuple(joints))


def _read_actuator(model: mujoco.MjModel, i: int, target: _Target) -> Actuator:
    name = target.actuator

    def reject(rule: str) -> ValueError:
        return ValueError(f"actuator {name!r} {rule}")

    types = {int(model.jnt_type[j]) for j, _ in target.joints}
    if not types <= {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}:
        raise reject("must drive hinge or slide joints")
    if len(types) > 1:
        raise reject("drives a tendon that mixes hinge and slide joints")
    rotary = types == {mujoco.mjtJoint.mjJNT_HINGE}
    if (dyn := int(model.actuator_dyntype[i])) != mujoco.mjtDyn.mjDYN_NONE:
        raise reject(f"has dyntype {mujoco.mjtDyn(dyn).name}; only NONE is supported")
    if (gt := int(model.actuator_gaintype[i])) != mujoco.mjtGain.mjGAIN_FIXED:
        raise reject(f"has gaintype {mujoco.mjtGain(gt).name}; only FIXED is supported")
    gain, gear = float(model.actuator_gainprm[i, 0]), model.actuator_gear[i]
    bias, biastype = model.actuator_biasprm[i, :3], int(model.actuator_biastype[i])
    if np.any(gear[1:] != 0.0) or not math.isfinite(gear[0]) or gear[0] == 0.0:
        raise reject(
            f"has gear {gear.tolist()}; it needs a finite nonzero gear[0] only"
        )
    kind: Kind
    if biastype == mujoco.mjtBias.mjBIAS_NONE:
        if not math.isclose(gain, 1.0):
            raise reject(f"has gain {gain:g} and no bias; a torque motor needs gain 1")
        if model.actuator_forcelimited[i]:
            raise reject("is a motor with a force limit; limit it with ctrlrange")
        kind, scale = "torque", 1.0 / float(gear[0])
    elif biastype == mujoco.mjtBias.mjBIAS_AFFINE:
        kp, kv = -float(bias[1]), -float(bias[2])
        if gear[0] != 1.0:
            raise reject(f"is a servo with gear {gear[0]:g}; a servo needs gear 1")
        if not gain > 0.0 or bias[0] != 0.0 or kv < 0.0:
            raise reject(
                "is an affine servo that needs gain > 0, biasprm[0] 0, kv >= 0"
            )
        if kp > 0.0:
            kind, scale = "position", kp / gain
        elif kp == 0.0 and kv > 0.0:
            kind, scale = "velocity", kv / gain
        else:
            raise reject(
                f"has biasprm {bias.tolist()}; it is no position or velocity servo"
            )
    else:
        raise reject(
            f"has biastype {mujoco.mjtBias(biastype).name}; use NONE or AFFINE"
        )
    unit = next(u for u, (k, r) in UNITS.items() if k == kind and r == rotary)
    lo, hi = (
        model.actuator_ctrlrange[i]
        if model.actuator_ctrllimited[i]
        else (-np.inf, np.inf)
    )
    a, b = sorted((lo / scale, hi / scale))
    return Actuator(
        name, kind, unit, scale, (float(lo), float(hi)), (float(a), float(b))
    )


def check_command(graph: Graph, actuators: tuple[Actuator, ...]) -> None:
    producers = {
        graph.signal_of(b, o.name): o
        for b, block in graph.blocks.items()
        for o in block.spec.outputs
    }
    at = 0
    for name in graph.command:
        spec = producers.get(name)
        if spec is None or spec.unit not in UNITS:
            unit = "" if spec is None else spec.unit
            units = ", ".join(UNITS)
            raise GraphError(
                f"command signal {name} has unit {unit!r}, not one of {units}"
            )
        kind = UNITS[spec.unit][0]
        for a in actuators[at : at + spec.size]:
            if a.unit != spec.unit:
                raise GraphError(
                    f"command signal {name} drives actuator {a.name} as {kind} "
                    f"({spec.unit}), but {a.name} is a {a.kind} actuator ({a.unit})"
                )
        at += spec.size


class MujocoPlant:
    """`sense` runs `mj_step1` and `act` runs `mj_step2`. Together they equal
    `mj_step` bitwise under Euler, implicit, and implicitfast. MuJoCo never moves
    `MjData` arrays, so views stay valid.
    """

    def __init__(self, model: mujoco.MjModel, spec: PlantSpec) -> None:
        self.model, self.spec = model, spec
        self.data = d = mujoco.MjData(model)
        self.tick = 0
        self._ctrl_map = CtrlMap.of(spec.actuators)
        self._gear = model.actuator_gear[:, 0].copy()
        self._key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY)
        self._site = _site(model)
        self._jacp = np.zeros((3, model.nv))
        free = _free_joints(model)
        self._base = None
        if free:
            q, v = model.jnt_qposadr[free[0]], model.jnt_dofadr[free[0]]
            xmat = d.xmat[model.jnt_bodyid[free[0]]].reshape(3, 3)
            self._base = (
                d.qpos[q : q + 3],
                d.qpos[q + 3 : q + 7],
                d.qvel[v : v + 3],
                d.qvel[v + 3 : v + 6],
                xmat,
            )
        self._fatal = memoryview(d.warning.number)[_FATAL_LO:_FATAL_HI]
        self._bound: Mapping[str, Array] | None = None
        self.reset()

    def reset(self) -> None:
        if self._key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.tick = 0

    def sense(self, signals: Mapping[str, Array]) -> None:
        m, d = self.model, self.data
        if signals is not self._bound:
            self._bound, self._effort = signals, signals["effort"]
        mujoco.mj_step1(m, d)
        np.divide(d.actuator_length, self._gear, out=signals["qpos_joints"])
        np.divide(d.actuator_velocity, self._gear, out=signals["qvel_joints"])
        if self._base is not None:
            pos, quat, linvel, angvel, xmat = self._base
            signals["base_pos"][:] = pos
            signals["base_quat"][:] = quat
            np.matmul(linvel, xmat, out=signals["base_linvel_body"])
            signals["base_angvel_body"][:] = angvel
        if self._site >= 0:
            signals["ee_pos"][:] = d.site_xpos[self._site]
            mujoco.mj_jacSite(m, d, self._jacp, None, self._site)
            np.matmul(self._jacp, self.spec.dof_map, out=signals["ee_jacobian"])

    def act(self, command: Array) -> None:
        d = self.data
        self._ctrl_map.apply(command, out=d.ctrl)
        mujoco.mj_step2(self.model, d)
        np.multiply(d.actuator_force, self._gear, out=self._effort)
        self.tick += 1
        if any(self._fatal):
            self._raise_unstable()

    def warnings(self) -> dict[str, int]:
        n = self.data.warning.number
        return {
            _W(w).name.removeprefix("mjWARN_"): int(n[w]) for w in range(len(n)) if n[w]
        }

    def _raise_unstable(self) -> None:
        for what, where, w in _FATAL:
            if self.data.warning[w].number:
                index = int(self.data.warning[w].lastinfo)
                raise RunUnstable(
                    what, where, index, self.tick - 1, self.model.opt.timestep
                )
