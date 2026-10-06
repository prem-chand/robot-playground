import argparse
import hashlib
import math
import statistics
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

from robot_pg.contract import BlockSpec, Graph, SignalSpec
from robot_pg.schedule import Schedule, compile_graph

XML = """<mujoco><option timestep="0.002"/><worldbody><body>
<joint name="j" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 -.5" size=".03"/>
</body></worldbody><actuator><motor joint="j" ctrlrange="-50 50"/></actuator></mujoco>"""
DT, AMP, FREQ = 0.002, 0.8, 0.5
Q, QD = SignalSpec("qpos_joints", (1,), "rad"), SignalSpec("qvel_joints", (1,), "rad/s")
REF, CMD = SignalSpec("ref", (1,), "rad"), SignalSpec("command", (1,), "N m")
NOMINAL = SignalSpec("command_nominal", (1,), "N m")


class Sine:
    def __init__(self, latency_ticks: int) -> None:
        self.spec = BlockSpec(50, latency_ticks * DT, outputs=(REF,), budget_s=1e-3)
        self.n = 0

    def reset(self, seed: int) -> None:
        self.n = 0

    def step(self, inputs):
        self.n += 1
        return {
            "ref": np.array([AMP * math.sin(2 * math.pi * FREQ * (self.n - 1) / 50)])
        }


class Pd:
    spec = BlockSpec(500, inputs=(Q, QD, REF), outputs=(CMD,), budget_s=1e-4)

    def reset(self, seed: int) -> None:
        pass

    def step(self, inputs):
        q, qd, ref = (
            inputs["qpos_joints"][0],
            inputs["qvel_joints"][0],
            inputs["ref"][0],
        )
        return {"command": np.array([40.0 * (ref - q) - 4.0 * qd])}


class Residual:
    spec = BlockSpec(250, DT, inputs=(Q, NOMINAL), outputs=(CMD,), budget_s=1e-4)

    def reset(self, seed: int) -> None:
        pass

    def step(self, inputs):
        q, nominal = inputs["qpos_joints"][0], inputs["command_nominal"][0]
        return {"command": np.array([nominal + 0.1 * math.tanh(q)])}


class Pendulum:
    def __init__(self, sched: Schedule, ticks: int, record: bool = True) -> None:
        self.m = mujoco.MjModel.from_xml_string(XML)
        self.d = mujoco.MjData(self.m)
        self.sched = sched
        self.frames = np.zeros((ticks, sched.frame.size)) if record else None

    def sense(self, signals) -> None:
        signals["qpos_joints"][:] = self.d.qpos
        signals["qvel_joints"][:] = self.d.qvel

    def act(self, command) -> None:
        if self.frames is not None:
            self.frames[self.sched.k - 1] = self.sched.frame
        self.d.ctrl[:] = command
        mujoco.mj_step(self.m, self.d)


def schedule(latency_ticks: int) -> Schedule:
    blocks = {"outer": Sine(latency_ticks), "pd": Pd(), "residual": Residual()}
    graph = Graph(blocks, ("command",), {"pd.command": "command_nominal"})
    return compile_graph(graph, (Q, QD), DT, 1)


def run(latency_ticks: int, ticks: int) -> tuple[str, np.ndarray]:
    sched = schedule(latency_ticks)
    plant = Pendulum(sched, ticks)
    sched.reset(0)
    sched.advance(plant, ticks)
    q = plant.frames[:, sched.columns["qpos_joints"][0]][:, 0]
    return hashlib.sha256(plant.frames.tobytes()).hexdigest()[:16], q


def sweep(plot: Path, ticks: int = 2000) -> None:
    t = np.arange(ticks) * DT
    target = AMP * np.sin(2 * np.pi * FREQ * t)
    traces = {}
    for lat in (0, 4, 20):
        (h1, q), (h2, _) = run(lat, ticks), run(lat, ticks)
        err = np.abs(q[500:] - target[500:]).mean()
        print(
            f"outer latency={lat:2d} ticks  replay_equal={h1 == h2} hash={h1}"
            f"  mean|track err|={err:.4f}"
        )
        traces[lat] = q
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 3.5))
    ax = fig.add_subplot()
    ax.plot(t, target, "k--", lw=1, label="target")
    for lat, q in traces.items():
        ax.plot(t, q, lw=1, label=f"outer latency {lat} ticks")
    ax.set(xlabel="time (s)", ylabel="joint angle (rad)")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(plot, dpi=120, bbox_inches="tight")
    print(f"plot {plot}")


def bench(ticks: int, rounds: int = 5) -> None:
    bare, sched = [], []
    for _ in range(rounds):
        m = mujoco.MjModel.from_xml_string(XML)
        d = mujoco.MjData(m)
        t0 = time.perf_counter()
        for _ in range(ticks):
            mujoco.mj_step(m, d)
        bare.append((time.perf_counter() - t0) / ticks * 1e6)
        s = schedule(4)
        plant = Pendulum(s, ticks, record=False)
        s.reset(0)
        t0 = time.perf_counter()
        s.advance(plant, ticks)
        sched.append((time.perf_counter() - t0) / ticks * 1e6)
    b, w = statistics.median(bare), statistics.median(sched)
    print(
        f"median of {rounds} interleaved rounds of {ticks} ticks: bare mj_step "
        f"{b:.2f} us/tick, schedule {w:.2f} us/tick, added {w - b:.2f} us/tick"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot", type=Path, default=Path(tempfile.gettempdir()) / "latency_sweep.png"
    )
    parser.add_argument("--bench", type=int, metavar="N")
    args = parser.parse_args()
    if args.bench:
        bench(args.bench)
    else:
        sweep(args.plot)


if __name__ == "__main__":
    main()
