from pathlib import Path

import mujoco
import numpy as np
import pytest

from robot_pg.contract import BlockSpec, Graph, SignalSpec
from robot_pg.plant import CtrlMap, check_command, parse_model
from robot_pg.runtime import run
from robot_pg.schedule import GraphError

MODELS = Path(__file__).parent / "models"


def load(name: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MODELS / f"{name}.xml"))


class Hold:
    def __init__(self, out: SignalSpec, value: list[float]) -> None:
        self.spec, self.value = BlockSpec(500, outputs=(out,)), np.array(value)

    def reset(self, seed: int) -> None: ...

    def step(self, inputs):
        return {self.spec.outputs[0].name: self.value}


@pytest.mark.parametrize(
    ("model", "kind", "unit"),
    [
        ("pendulum", "torque", "N m"),
        ("floating", "torque", "N m"),
        ("tendon_servo", "position", "m"),
        ("wheel", "velocity", "rad/s"),
    ],
)
def test_actuator_kind_comes_from_the_compiled_model(model, kind, unit):
    (a,) = parse_model(load(model)).actuators
    assert (a.kind, a.unit) == (kind, unit)


def test_tendon_servo_converts_a_length_to_ctrl():
    (a,) = parse_model(load("tendon_servo")).actuators
    assert a.command_range == (0.0, 0.040000000000500004)
    ctrl = CtrlMap.of((a,)).apply(np.array([0.02]))
    assert ctrl.tolist() == [127.49999999840624]


def test_kind_mismatch_names_the_actuator_and_both_kinds():
    tau = SignalSpec("command", (1,), "N m")
    graph = Graph({"hold": Hold(tau, [0.0])}, ("command",))
    with pytest.raises(GraphError) as e:
        check_command(graph, parse_model(load("wheel")).actuators)
    assert str(e.value) == (
        "command signal command drives actuator wheel as torque (N m), "
        "but wheel is a velocity actuator (rad/s)"
    )


def test_command_without_a_unit_fails():
    graph = Graph({"hold": Hold(SignalSpec("command", (1,)), [0.0])}, ("command",))
    with pytest.raises(GraphError, match="has unit '', not one of N m, N, rad, m"):
        check_command(graph, parse_model(load("pendulum")).actuators)


def test_rk4_fails():
    xml = (
        (MODELS / "pendulum.xml")
        .read_text()
        .replace(
            '<option timestep="0.002"/>', '<option timestep="0.002" integrator="RK4"/>'
        )
    )
    with pytest.raises(ValueError, match="RK4"):
        parse_model(mujoco.MjModel.from_xml_string(xml))


def test_ee_jacobian_is_the_hinge_axis_cross_the_site():
    xml = (
        (MODELS / "pendulum.xml")
        .read_text()
        .replace("</body>", '<site name="ee" pos="0 0 -0.25"/></body>')
    )
    model = mujoco.MjModel.from_xml_string(xml)
    graph = Graph(
        {"hold": Hold(SignalSpec("command", (1,), "N m"), [0.3])}, ("command",)
    )
    traj = run(model, graph, 0.2, 0)
    pos, jac = traj.signals["ee_pos"], traj.signals["ee_jacobian"]
    assert pos.shape == (100, 3) and jac.shape == (100, 3, 1)
    assert abs(pos[-1, 0]) > 0.01
    np.testing.assert_allclose(jac[:, :, 0], np.cross([0.0, 1.0, 0.0], pos), atol=1e-12)


def test_ee_jacobian_tendon_column_moves_both_fingers():
    xml = (
        (MODELS / "tendon_servo.xml")
        .read_text()
        .replace('mass="0.02"/>', 'mass="0.02"/><site name="ee"/>', 1)
    )
    model = mujoco.MjModel.from_xml_string(xml)
    graph = Graph(
        {"hold": Hold(SignalSpec("command", (1,), "m"), [0.02])}, ("command",)
    )
    traj = run(model, graph, 0.01, 0)
    assert traj.signals["ee_jacobian"][0].tolist() == [[0.0], [1.0], [0.0]]
