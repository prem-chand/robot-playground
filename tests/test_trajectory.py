from pathlib import Path

import mujoco
import numpy as np

from robot_pg.contract import BlockSpec, Graph, SignalSpec
from robot_pg.runtime import run
from robot_pg.trajectory import Trajectory

PENDULUM = Path(__file__).parent / "models/pendulum.xml"
TAU = SignalSpec("command", (1,), "N m")


class Fn:
    def __init__(self, spec: BlockSpec, f) -> None:
        self.spec, self.f, self.n = spec, f, 0

    def reset(self, seed: int) -> None:
        self.n = 0

    def step(self, inputs):
        self.n += 1
        return self.f(self.n)


def pendulum() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(PENDULUM))


def counter(*extra: SignalSpec) -> Graph:
    outs = (TAU, SignalSpec("count", (1,)), *extra)
    spec = BlockSpec(500, outputs=outs)
    block = Fn(spec, lambda n: {o.name: np.full(o.shape, float(n)) for o in outs})
    return Graph({"count": block}, ("command",))


def test_rows_copy_each_tick():
    traj = run(pendulum(), counter(), 0.01, 0)
    assert traj.signals["command"][:5, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert traj.signals["count"][:5, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_record_names_a_large_signal():
    small, large = SignalSpec("small", (64,)), SignalSpec("large", (65,))
    default = run(pendulum(), counter(small, large), 0.01, 0)
    assert default.signals["small"].shape == (5, 64)
    assert "large" not in default.signals
    named = run(pendulum(), counter(small, large), 0.01, 0, record=("large",))
    assert named.signals["large"][:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_save_and_load_keep_every_field(tmp_path):
    traj = run(pendulum(), counter(), 0.01, 0)
    back = Trajectory.load(traj.save(tmp_path / "run"))
    want = ["qpos_joints", "qvel_joints", "effort", "command", "count"]
    assert list(traj.signals) == want and list(back.signals) == want
    for name, a in traj.signals.items():
        assert np.array_equal(back.signals[name], a), name
    assert np.array_equal(back.timing, traj.timing)
    assert (back.dt, back.blocks, back.command_names) == (
        traj.dt,
        traj.blocks,
        traj.command_names,
    )
    assert (back.actuators, back.warnings) == (traj.actuators, traj.warnings)


def test_clip_marks_each_tick_and_holds_ctrl_at_the_limit():
    spec = BlockSpec(500, outputs=(TAU,))
    graph = Graph(
        {"push": Fn(spec, lambda n: {"command": np.array([100.0])})}, ("command",)
    )
    traj = run(pendulum(), graph, 0.01, 0)
    assert traj.clipped.all() and traj.ctrl[:, 0].tolist() == [5.0] * 5
    assert traj.signals["effort"][:, 0].tolist() == [5.0] * 5


def test_timing_has_one_column_per_block():
    traj = run(pendulum(), counter(), 0.01, 0)
    assert traj.blocks == ("count",) and traj.timing.shape == (5, 1)
    assert (traj.timing > 0).all()
