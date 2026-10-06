import numpy as np
import pytest

from robot_pg.contract import BlockSpec, External, Graph, SignalSpec
from robot_pg.schedule import GraphError, compile_graph

DT = 0.002
REF, CMD = SignalSpec("ref", (1,)), SignalSpec("command", (1,))


class Fn:
    def __init__(self, spec: BlockSpec) -> None:
        self.spec = spec

    def reset(self, seed: int) -> None:
        pass

    def step(self, inputs):
        return {o.name: np.zeros(o.shape) for o in self.spec.outputs}


def block(rate=500.0, latency=0.0, inputs=(), outputs=(CMD,)) -> Fn:
    return Fn(BlockSpec(rate, latency, inputs, outputs))


def error(blocks, command=("command",), wires=None, n_actuators=1) -> str:
    graph = Graph(blocks, command, wires or {})
    with pytest.raises(GraphError) as caught:
        compile_graph(graph, (), DT, n_actuators)
    return str(caught.value)


def test_rate_must_divide_the_physics_rate():
    blocks = {"outer": block(rate=300, outputs=(REF,)), "pd": block(inputs=(REF,))}
    assert error(blocks) == "block outer: 300 Hz does not divide 500 Hz"


def test_latency_must_be_whole_ticks():
    blocks = {
        "outer": block(rate=50, latency=0.007, outputs=(REF,)),
        "pd": block(inputs=(REF,)),
    }
    want = "block outer: latency 0.007 s is 3.5 ticks, not a whole number of ticks"
    assert error(blocks) == want


def test_signal_has_one_producer():
    blocks = {"pd": block(), "residual": block()}
    assert error(blocks) == "signal command has two producers: pd, residual"


def test_input_needs_a_producer():
    assert error({"pd": block(inputs=(REF,))}) == (
        "block pd: input ref reads signal ref, which has no producer"
    )


def test_input_shape_matches_its_producer():
    wide = SignalSpec("ref", (2,))
    blocks = {
        "outer": block(outputs=(wide,)),
        "pd": block(inputs=(SignalSpec("ref", (3,)),)),
    }
    assert error(blocks) == (
        "block pd: input ref has shape (3,) but signal ref from outer has shape (2,)"
    )


def test_no_cycle_inside_one_tick():
    a = SignalSpec("a", (1,))
    blocks = {"pd": block(inputs=(a,)), "obs": block(inputs=(CMD,), outputs=(a,))}
    assert (
        error(blocks)
        == "cycle inside one tick: pd reads a from obs, obs reads command from pd"
    )


def test_a_delayed_edge_is_not_a_cycle():
    a = SignalSpec("a", (1,))
    blocks = {
        "pd": block(inputs=(a,)),
        "obs": block(latency=DT, inputs=(CMD,), outputs=(a,)),
    }
    sched = compile_graph(Graph(blocks, ("command",)), (), DT, 1)
    assert repr(sched).splitlines()[1:] == [
        "  1. pd every 1, latency 0",
        "  2. obs every 1, latency 1",
    ]


def test_command_width_matches_the_actuator_count():
    arm, grip = SignalSpec("arm_pos_cmd", (7,)), SignalSpec("gripper_cmd", (1,))
    blocks = {"ik": block(outputs=(arm,)), "gripper": block(outputs=(grip,))}
    graph = Graph(blocks, ("arm_pos_cmd", "gripper_cmd"))
    compile_graph(graph, (), DT, 8)
    assert error(blocks, ("arm_pos_cmd", "gripper_cmd"), n_actuators=9) == (
        "command has 8 values (arm_pos_cmd 7, gripper_cmd 1) but the plant has 9 actuators"
    )


def test_a_wire_names_a_real_port():
    blocks = {"outer": block(outputs=(REF,)), "pd": block(inputs=(REF,))}
    assert (
        error(blocks, wires={"pd.refs": "ref"})
        == "wire pd.refs names no port of a block"
    )


def test_a_graph_has_one_external_slot():
    spec = BlockSpec(50, outputs=(REF,))
    blocks = {"a": External(spec), "b": External(spec), "pd": block()}
    assert error(blocks) == "graph has two external slots: a, b"


def test_initial_has_one_value_or_one_per_element():
    with pytest.raises(ValueError, match="signal q: initial needs 3 values"):
        SignalSpec("q", (3,), initial=(1.0, 2.0))
