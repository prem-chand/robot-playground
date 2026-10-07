"""Everything that touches a robot model: the registry, the compiled model
cache, the episode (robot, scenario, seed), and the fail rules.

The compiled model of each robot is built once per process and never handed
out. `RobotSpec.model()` and `Episode.model()` return copies, and
`Episode.apply` writes every field it owns from the cached values, so applying
twice equals applying once and no edit compounds across runs.
"""

import copy
import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from robot_pg.contract import Array
from robot_pg.plant import _FATAL, EE_SITE, HOME_KEY, parse_model
from robot_pg.scenarios import Path as PathRef
from robot_pg.scenarios import Reference, Scenario, Targets
from robot_pg.trajectory import Trajectory, TrajectoryError

ASSETS = Path(__file__).parent
FATAL_WARNINGS = frozenset(
    mujoco.mjtWarning(w).name.removeprefix("mjWARN_") for *_, w in _FATAL
)


@dataclass(frozen=True)
class Site:
    """A site added at load time with `MjSpec`, so the vendored file stays unchanged."""

    body: str
    pos: tuple[float, float, float]
    name: Literal["ee"] = EE_SITE


@dataclass(frozen=True)
class Sample:
    """What one seed drew. A fixed base records a zero base offset.

    `base_offset` is dx and dy in m and yaw in rad, in the world frame.
    `joint_offsets` holds one value in rad per actuated hinge, in actuator order,
    after the clip into the joint range. `target_order` permutes a `Targets`
    reference and is empty for any other reference.
    """

    friction_factor: float
    base_offset: tuple[float, float, float]
    joint_offsets: tuple[float, ...]
    target_order: tuple[int, ...]


@dataclass(frozen=True)
class Failure:
    tick: int
    time: float
    reason: str


@dataclass(frozen=True)
class FailRule:
    """Each threshold that is set is one check, and a run also fails on the
    warnings that raise `RunUnstable`. `trunk_contact` fails when a collision
    geom of the base body touches the geom named `floor`.
    """

    min_base_height_m: float | None = None
    trunk_contact: bool = False
    max_cross_track_m: float | None = None
    max_hold_error_m: float | None = None

    def describe(self) -> str:
        checks = []
        if self.min_base_height_m is not None:
            checks.append(f"base height < {self.min_base_height_m:g} m")
        if self.trunk_contact:
            checks.append("trunk touches floor")
        if self.max_cross_track_m is not None:
            checks.append(f"cross-track > {self.max_cross_track_m:g} m")
        if self.max_hold_error_m is not None:
            checks.append(
                f"ee error > {self.max_hold_error_m:g} m at the end of a hold"
            )
        return ", or ".join([*checks, "RunUnstable"])

    def __call__(
        self, trajectory: Trajectory, episode: "Episode", since: int = 0
    ) -> Failure | None:
        """The first failure at a tick at or after `since`, or None. The rule
        reads only recorded signals, so it scores saved and synthetic runs."""
        dt, n = trajectory.dt, len(trajectory)
        found: list[tuple[int, str]] = []
        if self.min_base_height_m is not None:
            z = _need(trajectory.signals, "base_pos")[since:, 2]
            if (low := np.flatnonzero(z < self.min_base_height_m)).size:
                found.append(
                    (
                        since + int(low[0]),
                        f"base height {z[low[0]]:.3f} m < {self.min_base_height_m:g} m",
                    )
                )
        if (
            self.trunk_contact
            and (tick := _trunk_contact(trajectory, episode.robot, since)) is not None
        ):
            found.append((tick, "trunk touches floor"))
        ref = episode.reference
        if self.max_cross_track_m is not None and isinstance(ref, PathRef):
            d = ref.cross_track(_need(trajectory.signals, "base_pos")[since:, :2])
            if (far := np.flatnonzero(d > self.max_cross_track_m)).size:
                found.append(
                    (
                        since + int(far[0]),
                        f"cross-track {d[far[0]]:.3f} m > {self.max_cross_track_m:g} m",
                    )
                )
        if self.max_hold_error_m is not None and isinstance(ref, Targets):
            ends = ref.hold_ends(dt, n)
            ends = ends[ends >= since]
            err = np.linalg.norm(
                _need(trajectory.signals, "ee_pos")[ends] - ref.values(ends, dt), axis=1
            )
            if (bad := np.flatnonzero(err > self.max_hold_error_m)).size:
                i = int(bad[0])
                hold = int(ref.hold_of(ends[i : i + 1], dt)[0])
                found.append(
                    (
                        int(ends[i]),
                        (
                            f"ee error {err[i]:.3f} m > {self.max_hold_error_m:g} m "
                            f"at the end of hold {hold + 1}/{len(ref.points)}"
                        ),
                    )
                )
        if (
            unstable := sorted(FATAL_WARNINGS & trajectory.warnings.keys())
        ) and n > since:
            found.append((n - 1, "RunUnstable: bad " + ", ".join(unstable)))
        if not found:
            return None
        tick, reason = min(found)
        return Failure(tick, tick * dt, reason)


@dataclass(frozen=True)
class RobotSpec:
    """`mjcf` is relative to the robots folder. `base_body` takes the base
    offset and the pushes, and `payload_body` takes `payload_kg`."""

    name: str
    mjcf: str
    keyframe: Literal["home"]
    base_body: str
    ee_site: Site | None
    payload_body: str
    fail_rule: FailRule

    def model(self) -> mujoco.MjModel:
        """A copy of the compiled model with no scenario applied. For the Panda
        this is the payload-free model."""
        return copy.copy(_compiled(self))

    def episode(self, scenario: Scenario, seed: int) -> "Episode":
        if scenario.robot != self.name:
            raise ValueError(f"scenario {scenario.name} is not for robot {self.name}")
        if seed < 0:
            raise ValueError(f"seed must be at least 0, got {seed}")
        if scenario.pushes and _free_joint(_compiled(self)) is None:
            raise ValueError(
                f"scenario {scenario.name} pushes the fixed base of {self.name}"
            )
        return Episode(self, scenario, seed, _draw(self, scenario, seed))

    def qpos(self, signals: Mapping[str, Array]) -> Array:
        """(T, nq) qpos rebuilt from recorded signals: the free joint from
        `base_pos` and `base_quat`, and each actuated joint from `qpos_joints`.
        A tendon sets each of its joints to the tendon length, which is exact for
        the Panda fingers. Any other joint takes its `home` value."""
        m = _compiled(self)
        q_act = _need(signals, "qpos_joints")
        q = np.tile(m.key_qpos[_key(m)], (len(q_act), 1))
        if (free := _free_joint(m)) is not None:
            a = m.jnt_qposadr[free]
            q[:, a : a + 3] = _need(signals, "base_pos")
            q[:, a + 3 : a + 7] = _need(signals, "base_quat")
        dof_map = parse_model(m).dof_map
        for j in range(m.njnt):
            weights = dof_map[m.jnt_dofadr[j]]
            if j != free and weights.any():
                q[:, m.jnt_qposadr[j]] = q_act @ weights
        return q


@dataclass(frozen=True)
class Episode:
    """One robot, one scenario, one seed, and what the seed drew."""

    robot: RobotSpec
    scenario: Scenario
    seed: int
    sample: Sample

    @property
    def reference(self) -> Reference:
        """The reference as this seed runs it, with the targets in drawn order."""
        ref = self.scenario.reference
        if isinstance(ref, Targets):
            return ref.permuted(self.sample.target_order)
        return ref

    @property
    def friction(self) -> float:
        return self.scenario.friction_scale * self.sample.friction_factor

    def model(self) -> mujoco.MjModel:
        m = self.robot.model()
        self.apply(m)
        return m

    def apply(self, model: mujoco.MjModel) -> None:
        """Write friction, payload mass, and the `home` keyframe onto a model of
        this robot, each from the compiled values. `MujocoPlant.reset` loads
        `home`, so the offsets reach tick 0. Inertia stays unchanged, as
        Appendix A measured the payload."""
        p = _compiled(self.robot)
        if (model.nq, model.ngeom, model.nbody) != (p.nq, p.ngeom, p.nbody):
            raise ValueError(f"model is not a {self.robot.name} model")
        model.geom_friction[:] = p.geom_friction * self.friction
        b = p.body(self.robot.payload_body).id
        model.body_mass[b] = p.body_mass[b] + self.scenario.payload_kg
        key = _key(p)
        q = p.key_qpos[key].copy()
        if (free := _free_joint(p)) is not None:
            a = p.jnt_qposadr[free]
            dx, dy, yaw = self.sample.base_offset
            q[a : a + 2] += (dx, dy)
            turned = np.empty(4)
            mujoco.mju_mulQuat(
                turned,
                np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]),
                q[a + 3 : a + 7],
            )
            q[a + 3 : a + 7] = turned
        for j, offset in zip(
            _hinges(self.robot), self.sample.joint_offsets, strict=True
        ):
            q[p.jnt_qposadr[j]] += offset
        model.key_qpos[key] = q


@functools.cache
def _compiled(robot: RobotSpec) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ASSETS / robot.mjcf))
    if (site := robot.ee_site) is not None:
        spec.body(site.body).add_site(name=site.name, pos=list(site.pos))
    return spec.compile()


def _key(m: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY)


def _free_joint(m: mujoco.MjModel) -> int | None:
    free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    return int(free[0]) if free.size else None


@functools.cache
def _hinges(robot: RobotSpec) -> tuple[int, ...]:
    """Hinge joints driven by an actuator, in actuator order."""
    m = _compiled(robot)
    return tuple(
        int(m.actuator_trnid[i, 0])
        for i in range(m.nu)
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
        and m.jnt_type[m.actuator_trnid[i, 0]] == mujoco.mjtJoint.mjJNT_HINGE
    )


def _draw(robot: RobotSpec, scenario: Scenario, seed: int) -> Sample:
    """Seed 0 is the nominal sample and touches no generator, because
    `default_rng(0)` does not draw zeros. Any other seed draws one vector from
    `default_rng(seed)`, in the order friction, dx, dy, yaw, then one value per
    hinge, followed by the target permutation."""
    hinges, m = _hinges(robot), _compiled(robot)
    ref = scenario.reference
    n_targets = len(ref.points) if isinstance(ref, Targets) else 0
    if seed == 0:
        return Sample(
            1.0, (0.0, 0.0, 0.0), (0.0,) * len(hinges), tuple(range(n_targets))
        )
    space = scenario.seed_space
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, 4 + len(hinges))
    lo, hi = space.friction
    friction = (lo + hi) / 2 + u[0] * (hi - lo) / 2
    base = (0.0, 0.0, 0.0)
    if _free_joint(m) is not None:
        xy = space.base_m / math.sqrt(2)
        base = (float(u[1] * xy), float(u[2] * xy), float(u[3] * space.base_yaw_rad))
    home = m.key_qpos[_key(m)]
    joints = []
    for j, uj in zip(hinges, u[4:], strict=True):
        q = home[m.jnt_qposadr[j]] + uj * space.joint_rad
        if m.jnt_limited[j]:
            q = np.clip(q, *m.jnt_range[j])
        joints.append(float(q - home[m.jnt_qposadr[j]]))
    order = tuple(int(i) for i in rng.permutation(n_targets)) if n_targets else ()
    return Sample(float(friction), base, tuple(joints), order)


def _need(signals: Mapping[str, Array], name: str) -> Array:
    if name not in signals:
        raise TrajectoryError(
            f"the fail rule needs signal {name}, which the run did not record"
        )
    return signals[name]


def _trunk_contact(trajectory: Trajectory, robot: RobotSpec, since: int) -> int | None:
    """Contact is not recorded, so it is rebuilt from the recorded state: set
    qpos, run `mj_kinematics` and `mj_collision`, and look for a base geom
    touching the floor. A tick runs that only when some trunk geom's bounding
    sphere reaches the floor, so a standing robot checks no tick."""
    m = _compiled(robot)
    base, floor = m.body(robot.base_body).id, m.geom("floor").id
    trunk = np.flatnonzero(
        (m.geom_bodyid == base) & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))
    )
    pos = _need(trajectory.signals, "base_pos")[since:]
    w, x, y, z = _need(trajectory.signals, "base_quat")[since:].T
    z_row = np.stack(
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1
    )
    center_z = pos[:, 2:3] + z_row @ m.geom_pos[trunk].T
    reach = m.geom_rbound[trunk] + m.geom_margin[trunk] + m.geom_margin[floor]
    near = np.flatnonzero((center_z - reach < m.geom_pos[floor, 2]).any(axis=1))
    if not near.size:
        return None
    rows = {k: v[since:][near] for k, v in trajectory.signals.items()}
    qpos = robot.qpos(rows)
    d = mujoco.MjData(m)
    for i, q in zip(near, qpos, strict=True):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        for c in d.contact[: d.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if floor in pair and pair & set(trunk.tolist()):
                return since + int(i)
    return None


ROBOTS: Mapping[str, RobotSpec] = {
    "diffdrive": RobotSpec(
        "diffdrive",
        "diffdrive/diffdrive.xml",
        "home",
        "base",
        None,
        "base",
        FailRule(max_cross_track_m=0.5),
    ),
    "panda": RobotSpec(
        "panda",
        "franka_emika_panda/scene.xml",
        "home",
        "link0",
        Site("hand", (0.0, 0.0, 0.1034)),
        "hand",
        FailRule(max_hold_error_m=0.2),
    ),
}
