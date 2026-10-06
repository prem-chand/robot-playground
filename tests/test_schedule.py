import math

import numpy as np
import pytest

from robot_pg.contract import BlockSpec, External, Graph, SignalSpec
from robot_pg.schedule import Schedule, compile_graph

DT = 0.002
Q = SignalSpec("qpos_joints", (1,))
REF, CMD = SignalSpec("ref", (1,)), SignalSpec("command", (1,))


class Fn:
    def __init__(self, spec: BlockSpec, f) -> None:
        self.spec, self.f, self.n = spec, f, 0

    def reset(self, seed: int) -> None:
        self.n = 0

    def step(self, inputs):
        self.n += 1
        return self.f(inputs, self.n)


class Pendulum:
    def __init__(self, sched: Schedule) -> None:
        self.q, self.qd = 0.0, 0.0
        self.sched, self.frames = sched, []

    def sense(self, signals) -> None:
        signals["qpos_joints"][:] = self.q

    def act(self, command) -> None:
        self.frames.append(self.sched.frame.copy())
        self.qd += (-9.81 * math.sin(self.q) + command[0]) * DT
        self.q += self.qd * DT


def pd() -> Fn:
    spec = BlockSpec(500, inputs=(Q, REF), outputs=(CMD,))
    return Fn(spec, lambda i, n: {"command": 40.0 * (i["ref"] - i["qpos_joints"])})


def run(blocks: dict, ticks: int, slot_values=None) -> tuple[Schedule, np.ndarray]:
    sched = compile_graph(Graph(blocks, ("command",)), (Q,), DT, 1)
    plant = Pendulum(sched)
    sched.reset(0)
    values = iter(slot_values or ())
    while sched.advance(plant, ticks) is not None:
        sched.set_slot({"ref": np.array([next(values)])})
    return sched, np.array(plant.frames)


def column(sched: Schedule, frames: np.ndarray, name: str) -> list[float]:
    return frames[:, sched.columns[name][0]][:, 0].tolist()


def test_held_output_changes_at_exact_ticks():
    count = Fn(
        BlockSpec(50, 4 * DT, outputs=(REF,)), lambda i, n: {"ref": np.array([n])}
    )
    sched, frames = run({"outer": count, "pd": pd()}, 40)
    ref = column(sched, frames, "ref")
    changes = [k for k in range(1, 40) if ref[k] != ref[k - 1]]
    assert changes == [4, 14, 24, 34]
    assert [ref[k] for k in changes] == [1.0, 2.0, 3.0, 4.0]


def test_latency_longer_than_the_period_keeps_every_output_in_flight():
    count = Fn(
        BlockSpec(50, 25 * DT, outputs=(REF,)), lambda i, n: {"ref": np.array([n])}
    )
    sched, frames = run({"outer": count, "pd": pd()}, 60)
    ref = column(sched, frames, "ref")
    changes = [k for k in range(1, 60) if ref[k] != ref[k - 1]]
    assert changes == [25, 35, 45, 55]
    assert [ref[k] for k in changes] == [1.0, 2.0, 3.0, 4.0]


def test_the_same_graph_run_twice_gives_equal_bytes():
    spec = BlockSpec(50, 4 * DT, outputs=(REF,))
    sine = Fn(spec, lambda i, n: {"ref": np.array([0.8 * np.sin(n / 8)])})
    sched = compile_graph(Graph({"outer": sine, "pd": pd()}, ("command",)), (Q,), DT, 1)
    runs = []
    for _ in range(2):
        plant = Pendulum(sched)
        sched.reset(0)
        sched.advance(plant, 500)
        runs.append(np.array(plant.frames).tobytes())
    assert runs[0] == runs[1]


def test_blocks_with_periods_4_and_6_step_on_their_own_ticks():
    a = Fn(BlockSpec(500 / 4, outputs=(CMD,)), lambda i, n: {"command": np.array([n])})
    b = Fn(BlockSpec(500 / 6, outputs=(REF,)), lambda i, n: {"ref": np.array([n])})
    sched, frames = run({"a": a, "b": b}, 13)
    for name, want in (("command", [4, 8, 12]), ("ref", [6, 12])):
        col = column(sched, frames, name)
        assert [k for k in range(1, 13) if col[k] != col[k - 1]] == want


def test_external_slot_equals_a_block_with_the_same_values():
    values = [0.1 * n for n in range(1, 11)]
    spec = BlockSpec(50, 4 * DT, outputs=(REF,))
    constant = Fn(spec, lambda i, n: {"ref": np.array([values[n - 1]])})
    _, by_block = run({"outer": constant, "pd": pd()}, 100)
    _, by_slot = run({"outer": External(spec), "pd": pd()}, 100, values)
    assert by_slot.tobytes() == by_block.tobytes()


def test_feedback_reads_the_previous_step_without_a_cycle():
    count = SignalSpec("count", (1,))
    spec = BlockSpec(500, outputs=(CMD, count), feedback=("count",))
    counter = Fn(
        spec, lambda i, n: {"count": i["count_prev"] + 1, "command": np.zeros(1)}
    )
    sched, frames = run({"counter": counter}, 3)
    assert column(sched, frames, "count") == [1.0, 2.0, 3.0]


def test_feedback_reads_the_step_output_before_the_latency_hold():
    count = SignalSpec("count", (1,))
    spec = BlockSpec(500, 2 * DT, outputs=(CMD, count), feedback=("count",))
    counter = Fn(
        spec, lambda i, n: {"count": i["count_prev"] + 1, "command": np.zeros(1)}
    )
    sched, frames = run({"counter": counter}, 6)
    assert column(sched, frames, "count") == [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]


def test_slot_feedback_is_the_last_set_slot_value():
    count = SignalSpec("count", (1,))
    spec = BlockSpec(500, outputs=(count,), feedback=("count",))
    graph = Graph(
        {"policy": External(spec), "pd": pd()}, ("command",), {"pd.ref": "count"}
    )
    sched = compile_graph(graph, (Q,), DT, 1)
    plant = Pendulum(sched)
    sched.reset(0)
    seen = []
    for value in (5.0, 7.0, 9.0):
        inputs = sched.advance(plant, 3)
        assert inputs is not None
        seen.append(inputs["count_prev"][0])
        sched.set_slot({"count": np.array([value])})
    assert seen == [0.0, 5.0, 7.0]


def test_a_block_cannot_write_its_inputs_in_place():
    def residual(i, n):
        i["command_nominal"] += 1.0
        return {"command": i["command_nominal"]}

    nominal = SignalSpec("command_nominal", (1,))
    blocks = {
        "pd": pd(),
        "residual": Fn(BlockSpec(500, inputs=(nominal,), outputs=(CMD,)), residual),
    }
    wires = {"pd.command": "command_nominal", "pd.ref": "qpos_joints"}
    sched = compile_graph(Graph(blocks, ("command",), wires), (Q,), DT, 1)
    sched.reset(0)
    with pytest.raises(ValueError, match="read-only"):
        sched.advance(Pendulum(sched), 1)
