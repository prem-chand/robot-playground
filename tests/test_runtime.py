import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robot_pg.contract import BlockSpec, External, Graph, SignalSpec
from robot_pg.runtime import RunUnstable, Session, run
from robot_pg.schedule import GraphError

ROOT = Path(__file__).parent.parent
TAU = SignalSpec("command", (1,), "N m")
Q, QD = SignalSpec("qpos_joints", (1,)), SignalSpec("qvel_joints", (1,))


def load(path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path))


class Pd:
    spec = BlockSpec(500, inputs=(Q, QD), outputs=(TAU,))

    def __init__(self, target: float) -> None:
        self.target = target

    def reset(self, seed: int) -> None: ...

    def step(self, inputs):
        q, qd = inputs["qpos_joints"], inputs["qvel_joints"]
        return {"command": 1000.0 * (self.target - q) - 5.0 * qd}


class Hold:
    def __init__(self, out: SignalSpec, value: list[float]) -> None:
        self.spec, self.value = BlockSpec(500, outputs=(out,)), np.array(value)

    def reset(self, seed: int) -> None: ...

    def step(self, inputs):
        return {self.spec.outputs[0].name: self.value}


def pendulum() -> mujoco.MjModel:
    return load(ROOT / "tests/models/pendulum.xml")


def test_torque_pd_holds_the_pendulum():
    traj = run(pendulum(), Graph({"pd": Pd(0.5)}, ("command",)), 2.0, 0)
    assert traj.signals["qpos_joints"].shape == (1000, 1)
    assert abs(traj.signals["qpos_joints"][-1, 0] - 0.5) < 1e-3


def test_floating_base_signals_and_actuated_joints_only():
    graph = Graph({"hold": Hold(TAU, [0.0])}, ("command",))
    traj = run(load(ROOT / "tests/models/floating.xml"), graph, 2.0, 0)
    base = traj.signals["base_pos"]
    assert base.shape == (1000, 3) and traj.signals["qpos_joints"].shape == (1000, 1)
    assert base[0, 2] == 0.5 and 0.04 < base[-1, 2] < 0.06


def test_base_linear_velocity_is_in_the_body_frame():
    graph = Graph({"hold": Hold(TAU, [0.0])}, ("command",))
    session = Session(load(ROOT / "tests/models/floating.xml"), graph, 0)
    session.plant.data.qpos[3:7] = [math.cos(0.4), 0.0, 0.0, math.sin(0.4)]
    session.plant.data.qvel[:3] = [1.0, 0.0, 0.0]
    session.advance(1)
    np.testing.assert_allclose(
        session.signal("base_linvel_body"),
        [math.cos(0.8), -math.sin(0.8), 0.0],
        atol=1e-12,
    )


def test_instability_raises_with_the_first_bad_dof(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = load(ROOT / "docs/plan-evidence/proto/diffdrive_euler.xml")
    graph = Graph(
        {"hold": Hold(SignalSpec("command", (2,), "rad/s"), [5.0, 5.0])}, ("command",)
    )
    with pytest.raises(RunUnstable) as e:
        run(model, graph, 1.0, 0)
    assert str(e.value) == "bad qacc at dof 6, t=0.0100"
    assert e.value.trajectory is not None and len(e.value.trajectory) == 6


def test_session_refuses_to_advance_after_instability(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = load(ROOT / "docs/plan-evidence/proto/diffdrive_euler.xml")
    graph = Graph(
        {"hold": Hold(SignalSpec("command", (2,), "rad/s"), [5.0, 5.0])}, ("command",)
    )
    session = Session(model, graph, 0)
    with pytest.raises(RunUnstable):
        session.advance(100)
    with pytest.raises(RunUnstable):
        session.advance(100)
    session.reset(0)
    assert session.advance(3) is None and len(session.trajectory()) == 3


def test_session_slot_replays_run_bitwise():
    model = pendulum()
    want = run(model, Graph({"pd": Pd(0.5)}, ("command",)), 0.2, 0)
    session = Session(model, Graph({"pd": External(Pd.spec)}, ("command",)), 0)
    block = Pd(0.5)
    while (inputs := session.advance(100)) is not None:
        session.set_slot(block.step(inputs))
    got = session.trajectory()
    assert list(got.signals) == ["qpos_joints", "qvel_joints", "effort", "command"]
    for name, a in want.signals.items():
        assert np.array_equal(got.signals[name], a), name


def test_run_rejects_a_graph_with_a_slot():
    graph = Graph({"pd": External(Pd.spec)}, ("command",))
    with pytest.raises(GraphError, match="external slot"):
        run(pendulum(), graph, 0.1, 0)
