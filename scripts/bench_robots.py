"""Time each robot's model load, a bare `mj_step` loop, and `run()` on the same
constant hold, interleaved. Exit 1 when `run` adds more than 60 us per step or a
load takes more than 2 s.

A position actuator holds its `home` ctrl, and a torque or velocity actuator
holds 0 in both loops alike.
"""

import itertools
import statistics
import sys
import time
from collections.abc import Mapping

import mujoco
import numpy as np

from robot_pg.contract import Array, BlockSpec, Graph, SignalSpec
from robot_pg.plant import CtrlMap, parse_model
from robot_pg.robots import ROBOTS
from robot_pg.runtime import run

STEPS = 5000
ROUNDS = 3
MAX_OVERHEAD_US = 60.0
MAX_LOAD_S = 2.0


class Hold:
    def __init__(self, outputs: Mapping[SignalSpec, Array]) -> None:
        self.spec = BlockSpec(500.0, outputs=tuple(outputs))
        self._out = {s.name: v for s, v in outputs.items()}

    def reset(self, seed: int) -> None:
        pass

    def step(self, inputs: Mapping[str, Array]) -> Mapping[str, Array]:
        return self._out


def hold(model: mujoco.MjModel) -> tuple[Graph, Array]:
    """One output per run of actuators with the same unit, and the hold command."""
    actuators = parse_model(model).actuators
    key = model.key_ctrl[model.key("home").id]
    command = np.array(
        [
            k / a.scale if a.kind == "position" else 0.0
            for a, k in zip(actuators, key, strict=True)
        ]
    )
    outputs, start = {}, 0
    for i, (unit, group) in enumerate(itertools.groupby(actuators, lambda a: a.unit)):
        n = len(list(group))
        outputs[SignalSpec(f"hold{i}", (n,), unit)] = command[start : start + n]
        start += n
    names = tuple(s.name for s in outputs)
    return Graph({"hold": Hold(outputs)}, names), command


def bare(model: mujoco.MjModel, ctrl: Array) -> float:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.ctrl[:] = ctrl
    start = time.perf_counter()
    for _ in range(STEPS):
        mujoco.mj_step(model, data)
    return (time.perf_counter() - start) / STEPS


def through_run(model: mujoco.MjModel, graph: Graph) -> float:
    start = time.perf_counter()
    run(model, graph, STEPS * model.opt.timestep, 0)
    return (time.perf_counter() - start) / STEPS


def main() -> int:
    print("robot      load_s  bare_steps_s  run_steps_s  overhead_us")
    failed = []
    for name, robot in ROBOTS.items():
        start = time.perf_counter()
        model = robot.model()
        load = time.perf_counter() - start
        graph, command = hold(model)
        ctrl = CtrlMap.of(parse_model(model).actuators).apply(command)
        b, r = [], []
        for _ in range(ROUNDS):
            b.append(bare(model, ctrl))
            r.append(through_run(model, graph))
        b_s, r_s = statistics.median(b), statistics.median(r)
        overhead = (r_s - b_s) * 1e6
        print(
            f"{name:<10} {load:6.3f}  {1 / b_s:12.0f}  {1 / r_s:11.0f}  {overhead:11.1f}"
        )
        if overhead > MAX_OVERHEAD_US:
            failed.append(
                f"{name} run adds {overhead:.1f} us per step > {MAX_OVERHEAD_US:g}"
            )
        if load > MAX_LOAD_S:
            failed.append(f"{name} loads in {load:.3f} s > {MAX_LOAD_S:g}")
    for line in failed:
        print("FAIL", line)
    print("ok" if not failed else f"{len(failed)} over budget")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
