"""Panda DLS reach on position servos: torque-limit and latency limits.

uv run --extra dev --extra viz python experiments/panda_reach/run.py
"""

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from matplotlib.figure import Figure

from robot_pg import lab
from robot_pg.controllers import Reach
from robot_pg.scenarios import SCENARIOS, Targets

OUT = Path(__file__).parent
SEEDS = range(5)
WORKERS = 6
DT = 0.002
GRID: Targets = SCENARIOS["panda.reach_grid"].reference  # type: ignore[assignment]
HOLD = round(GRID.hold_s / DT)
SECONDS = len(GRID.points) * GRID.hold_s
TOL_MM = 10.0
GAIN, DAMPING = 2.0, 0.1
NOISE = {"qpos_joints": 1e-3, "ee_pos": 1e-3}
INK, MUTED, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"


class Policy:
    def __init__(self, targets: Targets) -> None:
        self.targets, self.reach = targets, Reach(7, hand=0.04, step_gain=GAIN, damping=DAMPING)

    def __call__(self, obs: lab.Obs) -> np.ndarray:
        k = np.array([round(obs["t"] / DT)])
        out = self.reach.step({**obs, "ref_ee_target": self.targets.values(k, DT)[0]})
        return np.r_[out["arm"], out["hand"]]


def model(torque: float = 1.0) -> mujoco.MjModel:
    m = lab.load("panda")
    m.actuator_forcerange[:7] *= torque
    return m


def draw(seed: int) -> tuple[Targets, np.ndarray]:
    """Target order and a joint offset of up to 0.02 rad. Seed 0 is nominal."""
    m = lab.load("panda")
    q0 = lab.home(m).qpos.copy()
    if seed == 0:
        return GRID, q0
    rng = np.random.default_rng(seed)
    q0[:7] += rng.uniform(-0.02, 0.02, 7)
    return GRID.permuted(tuple(int(i) for i in rng.permutation(len(GRID.points)))), q0


def reach_run(torque: float, latency: float, seed: int) -> tuple[lab.Run, Targets, mujoco.MjModel]:
    targets, q0 = draw(seed)
    m = model(torque)
    run = lab.rollout(m, Policy(targets), SECONDS, seed=seed, qpos0=q0,
                      latency_s=latency, obs_noise=NOISE)
    return run, targets, m


def hold_errors(run: lab.Run, targets: Targets, m: mujoco.MjModel) -> np.ndarray:
    """End-effector error in mm at the last tick of each hold. inf after a failure."""
    d = mujoco.MjData(m)
    out = np.full(len(targets.points), math.inf)
    for i, goal in enumerate(targets.points):
        k = (i + 1) * HOLD - 1
        if k < len(run.t):
            d.qpos[:] = run.qpos[k]
            mujoco.mj_kinematics(m, d)
            out[i] = 1000 * np.linalg.norm(d.site("ee").xpos - goal)
    return out


def worst(torque: float, latency: float, seed: int) -> float:
    """Worst end-of-hold error in mm, capped at 1000."""
    return float(min(hold_errors(*reach_run(torque, latency, seed)).max(), 1000.0))


def stop_holds(torque: float, latency: float, seed: int) -> float:
    """Holds whose last tick has an arm joint pressed on its stop with over 1 N m.
    There the stop, not the servo, carries part of the arm."""
    run, targets, m = reach_run(torque, latency, seed)
    d, count = mujoco.MjData(m), 0
    for i in range(len(targets.points)):
        k = min((i + 1) * HOLD, len(run.t)) - 1
        d.qpos[:], d.qvel[:], d.ctrl[:] = run.qpos[k], run.qvel[k], run.ctrl[k]
        mujoco.mj_forward(m, d)
        on_stop = (d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT) & (
            d.efc_id[: d.nefc] < 7) & (np.abs(d.efc_force[: d.nefc]) > 1.0)
        count += bool(on_stop.any())
    return float(count)


def servo_ok(torque: float, seed: int) -> bool:
    """Tracks within the tolerance and never rests an arm joint on its stop."""
    return torque_ok(torque, seed) and stop_holds(torque, 0.0, seed) == 0


def strict_boundary(seed: int) -> float:
    return boundary(servo_ok, 1.0, 0.5, seed) if servo_ok(1.0, seed) else math.nan


def tracks(torque: float, latency: float, seed: int) -> bool:
    return worst(torque, latency, seed) <= TOL_MM


def torque_ok(torque: float, seed: int) -> bool:
    return tracks(torque, 0.0, seed)


def latency_ok(latency: float, seed: int) -> bool:
    return tracks(1.0, latency, seed)


def latency_free_ok(latency: float, seed: int) -> bool:
    return tracks(1000.0, latency, seed)


def boundary(ok, good: float, bad: float, seed: int, steps: int = 8) -> float:
    """The last passing value of a bisection between a passing and a failing value."""
    for _ in range(steps):
        mid = (good + bad) / 2
        good, bad = (mid, bad) if ok(mid, seed) else (good, mid)
    return good


def gravity_bound(seed: int) -> float:
    """The torque fraction below which some hold needs more static torque than the
    servo can give: max over holds and joints of |gravity torque| / limit."""
    run, targets, m = reach_run(1.0, 0.0, seed)
    d = mujoco.MjData(m)
    worst_ratio = 0.0
    for i in range(len(targets.points)):
        d.qpos[:], d.qvel[:] = run.qpos[(i + 1) * HOLD - 1], 0.0
        mujoco.mj_forward(m, d)
        ratio = np.abs(d.qfrc_bias[:7]) / m.actuator_forcerange[:7, 1]
        worst_ratio = max(worst_ratio, float(ratio.max()))
    return worst_ratio


def linear_decay(latencies: np.ndarray) -> np.ndarray:
    """End error / start error after one 2 s hold, for a 1 cm step at `home`, in
    the delayed linear model from mjd_transitionFD. Force limits do not exist here."""
    m = lab.load("panda")
    d = lab.home(m)
    a, b = lab.linearize(m, d)
    jac = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jac, None, m.site("ee").id)
    jac = jac[:, :7]
    proj = jac.T @ np.linalg.solve(jac @ jac.T + DAMPING**2 * np.eye(3), jac)
    f = np.zeros((m.nu, 2 * m.nv))
    f[:7, :7] = np.eye(7) - GAIN * proj
    out = []
    for lat in latencies:
        x = np.zeros(2 * m.nv)
        x[:7] = np.linalg.pinv(jac) @ np.array([0.01, 0.0, 0.0])
        queue = [np.zeros(m.nu)] * round(lat / DT)
        start = np.linalg.norm(jac @ x[:7])
        peak = start
        for _ in range(HOLD):
            queue.append(f @ x)
            x = a @ x + b @ queue.pop(0)
            peak = max(peak, float(np.linalg.norm(jac @ x[:7])))
        out.append((np.linalg.norm(jac @ x[:7]) / start, peak / start))
    return np.array(out)


def main() -> None:
    torques = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1])
    latencies = np.round(np.arange(0.0, 0.41, 0.025), 3)
    map_torques = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3])
    map_latencies = np.round(np.arange(0.0, 0.31, 0.025), 3)

    by_torque = lab.sweep(worst, SEEDS, WORKERS, torque=torques, latency=[0.0])[:, 0]
    on_stop = lab.sweep(stop_holds, SEEDS, WORKERS, torque=torques, latency=[0.0])[:, 0]
    by_latency = lab.sweep(worst, SEEDS, WORKERS, torque=[1.0], latency=latencies)[0]
    by_latency_free = lab.sweep(worst, SEEDS, WORKERS, torque=[1000.0], latency=latencies)[0]
    grid = lab.sweep(worst, SEEDS, WORKERS, torque=map_torques, latency=map_latencies)
    with ProcessPoolExecutor(WORKERS) as pool:
        n = len(SEEDS)
        f_star = list(pool.map(boundary, [torque_ok] * n, [1.0] * n, [0.1] * n, SEEDS))
        tau_star = list(pool.map(boundary, [latency_ok] * n, [0.0] * n, [0.4] * n, SEEDS))
        tau_free = list(pool.map(boundary, [latency_free_ok] * n, [0.0] * n, [0.4] * n, SEEDS))
        g_bound = list(pool.map(gravity_bound, SEEDS))
        f_strict = list(pool.map(strict_boundary, SEEDS))
    lin = linear_decay(latencies)

    result = {
        "tolerance (mm, worst end-of-hold error)": TOL_MM,
        "nominal worst end-of-hold error (mm), per seed": by_torque[0].round(2).tolist(),
        "torque limit: last passing fraction, per seed": np.round(f_star, 4).tolist(),
        "torque limit, no load on a joint stop (NaN: fails at nominal), per seed":
            [None if math.isnan(f) else round(f, 4) for f in f_strict],
        "holds with an arm joint on its stop [torque][seed]": {
            str(t): row.astype(int).tolist() for t, row in zip(torques, on_stop, strict=True)},
        "gravity bound: torque fraction, per seed": np.round(g_bound, 4).tolist(),
        "latency limit at nominal torque (s), per seed": np.round(tau_star, 4).tolist(),
        "latency limit with no force limit (s), per seed": np.round(tau_free, 4).tolist(),
        "linear model: end/start error after one hold, by latency": dict(
            zip(latencies.tolist(), lin[:, 0].round(6).tolist(), strict=True)
        ),
        "linear model: peak/start error, max over latencies": float(lin[:, 1].max()),
        "map: fraction of seeds that track [torque][latency]": {
            str(t): dict(zip(map_latencies.tolist(), row.tolist(), strict=True))
            for t, row in zip(map_torques, (grid <= TOL_MM).mean(axis=2), strict=True)
        },
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    plot(torques, by_torque, on_stop, latencies, by_latency, by_latency_free, lin,
         map_torques, map_latencies, grid, f_star, f_strict, g_bound, tau_star, tau_free)

    clip = round(12.0 / DT)
    for name, torque, latency in (
        ("nominal", 1.0, 0.0),
        ("torque_on_stop", 0.7, 0.0),
        ("torque_fail", float(np.median(f_star)) * 0.8, 0.0),
        ("latency_fail", 1.0, float(np.median(tau_star)) * 1.5),
    ):
        run, _, m = reach_run(torque, latency, 0)
        short = lab.Run(run.t[:clip], run.qpos[:clip], run.qvel[:clip], run.ctrl[:clip], None)
        lab.video(m, short, OUT / f"{name}.mp4")


def plot(torques, by_torque, on_stop, latencies, by_latency, by_latency_free, lin,
         map_torques, map_latencies, grid, f_star, f_strict, g_bound, tau_star, tau_free) -> None:
    fig = Figure(figsize=(12, 8.5), layout="constrained")
    (ax1, ax4), (ax2, ax3) = fig.subplots(2, 2)

    lab.band(ax1, torques, np.log10(by_torque), "worst end-of-hold error (5 seeds)", BLUE)
    ax1.axhline(math.log10(TOL_MM), color=INK, lw=0.8, label="1 cm tolerance")
    ax1.axvline(float(np.median(f_star)), color=BLUE, ls="-.", lw=1,
                label=f"sim limit {np.median(f_star):.3f}")
    ax1.axvline(float(np.median(g_bound)), color=ORANGE, lw=1.6,
                label=f"gravity bound {np.median(g_bound):.3f}")
    ax1.axvline(0.25, color=MUTED, ls="--", lw=1, label="user: 0.25")
    ax1.axvspan(0.30, 0.40, color=MUTED, alpha=0.12, lw=0, label="Claude: 0.30-0.40")
    ax1.set(xlabel="servo force limit / nominal", ylabel="log10 error (mm)",
            title="Torque limit (no latency)")
    ax1.invert_xaxis()
    ax1.legend(fontsize=7, loc="upper left")

    lab.band(ax4, torques, on_stop, "holds with an arm joint on its stop (5 seeds)", BLUE)
    strict = np.array([f for f in f_strict if not math.isnan(f)])
    if strict.size:
        ax4.axvline(float(np.median(strict)), color=BLUE, ls="-.", lw=1,
                    label=f"limit, no stop load {np.median(strict):.3f} ({strict.size} of {len(f_strict)} seeds)")
    ax4.axvline(float(np.median(g_bound)), color=ORANGE, lw=1.6,
                label=f"gravity bound {np.median(g_bound):.3f}")
    ax4.set(xlabel="servo force limit / nominal", ylabel="holds (of 27)",
            title="Arm resting on a joint stop (no latency)")
    ax4.invert_xaxis()
    ax4.legend(fontsize=7, loc="upper left")

    lab.band(ax2, latencies, np.log10(by_latency), "sim, nominal force limit", BLUE)
    lab.band(ax2, latencies, np.log10(by_latency_free), "sim, no force limit", AQUA)
    ax2.plot(latencies, np.log10(np.maximum(300.0 * lin[:, 0], 1e-3)), color=ORANGE, lw=1.6,
             label="linear model: 0.3 m move, end of hold")
    ax2.axhline(math.log10(TOL_MM), color=INK, lw=0.8, label="1 cm tolerance")
    ax2.axvline(0.06, color=MUTED, ls="--", lw=1, label="user: 60 ms")
    ax2.axvspan(0.04, 0.08, color=MUTED, alpha=0.12, lw=0, label="Claude: 40-80 ms")
    ax2.axvline(float(np.median(tau_star)), color=BLUE, ls="-.", lw=1,
                label=f"limit {1000 * np.median(tau_star):.0f} ms")
    ax2.axvline(float(np.median(tau_free)), color=AQUA, ls="-.", lw=1,
                label=f"limit, no force limit {1000 * np.median(tau_free):.0f} ms")
    ax2.set(xlabel="latency (s)", ylabel="log10 error (mm)", title="Latency (nominal torque)")
    ax2.legend(fontsize=7, loc="upper left")

    frac = (grid <= TOL_MM).mean(axis=2)
    im = ax3.imshow(frac, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1,
                    extent=(map_latencies[0] - 0.0125, map_latencies[-1] + 0.0125,
                            -0.5, len(map_torques) - 0.5))
    ax3.set_yticks(range(len(map_torques)), [f"{t:g}" for t in map_torques])
    ax3.set(xlabel="latency (s)", ylabel="servo force limit / nominal",
            title="Fraction of seeds within 1 cm")
    fig.colorbar(im, ax=ax3, shrink=0.8)
    for ax in (ax1, ax2, ax4):
        ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(OUT / "limits.png", dpi=130)
    print(f"plot {OUT / 'limits.png'}")


if __name__ == "__main__":
    main()
