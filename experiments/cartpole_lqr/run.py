"""Cart-pole LQR from mjd_transitionFD: push, tilt, and pole-mass limits.

uv run --extra dev --extra viz python experiments/cartpole_lqr/run.py
"""

import json
import math
from pathlib import Path

import control
import mujoco
import numpy as np
from matplotlib.figure import Figure
from scipy.linalg import solve_discrete_are

from robot_pg import lab

OUT = Path(__file__).parent
SEEDS = range(5)
WORKERS = 6
U_MAX = 20.0
M_CART, M_POLE, L_COM, I_POLE, G = 1.0, 0.3, 0.4, 0.016, 9.81
Q, R = np.diag([1.0, 10.0, 0.1, 0.1]), np.array([[0.01]])
PUSH_S = 0.1
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"


def gains(model: mujoco.MjModel) -> np.ndarray:
    a, b = lab.linearize(model)
    p = solve_discrete_are(a, b, Q, R)
    return np.linalg.solve(R + b.T @ p @ b, b.T @ p @ a)


MODEL = lab.load("cartpole")
K = gains(MODEL)


def policy(obs: lab.Obs) -> np.ndarray:
    return -K @ np.r_[obs["qpos"], obs["qvel"]]


def fail(obs: lab.Obs) -> str | None:
    if abs(obs["qpos"][1]) > math.pi / 2:
        return "pole fell"
    if abs(obs["qpos"][0]) > 3.0:
        return "cart left the rail"
    return None


def draw(seed: int) -> tuple[np.ndarray, float, float]:
    """Initial offset, push sign, and push start for one seed."""
    rng = np.random.default_rng(seed)
    q0 = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.02, 0.02)])
    return q0, float(rng.choice([-1.0, 1.0])), 1.0 + float(rng.uniform(0.0, 0.2))


NOISE = {"qpos": 1e-3, "qvel": 1e-2}


def push_run(impulse: float, seed: int, model: mujoco.MjModel = MODEL) -> lab.Run:
    q0, sign, at = draw(seed)
    force = sign * impulse / PUSH_S
    return lab.rollout(
        model, policy, 6.0, seed=seed, qpos0=q0, obs_noise=NOISE, fail=fail,
        pushes=[lab.Push("pole", at, PUSH_S, (force, 0.0, 0.0))],
    )


def recovered(run: lab.Run) -> bool:
    return not run.failed and abs(run.qpos[-1, 1]) < 0.05


def push_peak(impulse: float, seed: int) -> float:
    """Peak tilt in rad, or NaN when the LQR does not recover."""
    run = push_run(impulse, seed)
    return float(np.abs(run.qpos[:, 1]).max()) if recovered(run) else math.nan


def release_ok(tilt: float, seed: int) -> float:
    q0, sign, _ = draw(seed)
    q0[1] = sign * tilt
    run = lab.rollout(MODEL, policy, 6.0, seed=seed, qpos0=q0, obs_noise=NOISE, fail=fail)
    return float(recovered(run))


def scaled(scale: float) -> mujoco.MjModel:
    m = lab.load("cartpole")
    pole = m.body("pole").id
    m.body_mass[pole] *= scale
    m.body_inertia[pole] *= scale
    return m


def growth(scale: float, seed: int) -> float:
    """Fitted growth rate in 1/s of the tilt envelope, with no noise."""
    q0, _, _ = draw(seed)
    q0[1] += 0.05
    run = lab.rollout(scaled(scale), policy, 6.0, seed=seed, qpos0=q0, fail=fail)
    th = np.abs(run.qpos[:, 1])
    w = round(0.25 / MODEL.opt.timestep)
    env = np.array([th[i : i + w].max() for i in range(0, len(th) - w + 1, w)])
    t = (np.arange(len(env)) + 0.5) * w * MODEL.opt.timestep
    use = (t > 0.5) & (env < 0.3) & (env > 1e-12)
    if use.sum() < 3:
        use = (env < 0.3) & (env > 1e-12)
    return float(np.polyfit(t[use], np.log(env[use]), 1)[0])


def stays_up(scale: float, seed: int) -> float:
    q0, _, _ = draw(seed)
    q0[1] += 0.05
    run = lab.rollout(scaled(scale), policy, 15.0, seed=seed, qpos0=q0,
                      obs_noise=NOISE, fail=fail)
    return float(recovered(run))


def linear_rate(scale: float) -> float:
    a, b = lab.linearize(scaled(scale))
    rho = np.abs(np.linalg.eigvals(a - b @ K)).max()
    return math.log(rho) / MODEL.opt.timestep


def static_capture() -> float:
    """Largest tilt at rest where full cart force still gives theta'' <= 0."""

    def theta_dd(th: float) -> float:
        mass = np.array([[M_CART + M_POLE, M_POLE * L_COM * math.cos(th)],
                         [M_POLE * L_COM * math.cos(th), I_POLE + M_POLE * L_COM**2]])
        rhs = np.array([U_MAX, M_POLE * G * L_COM * math.sin(th)])
        return float(np.linalg.solve(mass, rhs)[1])

    lo, hi = 0.0, math.pi / 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if theta_dd(mid) <= 0 else (lo, mid)
    return lo


def boundary(ok, lo: float, hi: float, seed: int, steps: int = 18) -> float:
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid, seed) else (lo, mid)
    return lo


def push_ok(impulse: float, seed: int) -> bool:
    return not math.isnan(push_peak(impulse, seed))


def checks() -> dict:
    a, b = lab.linearize(MODEL)
    k_ctl, _, poles_ctl = control.dlqr(a, b, Q, R)
    poles = np.linalg.eigvals(a - b @ K)
    x0 = np.array([0.0, 0.01, 0.0, 0.0])
    run = lab.rollout(MODEL, policy, 3.0, qpos0=x0[:2])
    lin = np.empty((len(run.t), 4))
    x = x0
    for i in range(len(run.t)):
        lin[i] = x
        x = (a - b @ K) @ x
    sim = np.c_[run.qpos, run.qvel]
    return {
        "K": K.ravel().tolist(),
        "max |K - K_python_control|": float(np.abs(K - k_ctl).max()),
        "max |poles - poles_python_control|": float(
            np.abs(np.sort_complex(poles) - np.sort_complex(poles_ctl)).max()
        ),
        "continuous poles (1/s)": [
            f"{z.real:.3f}{z.imag:+.3f}j" for z in np.log(poles.astype(complex)) / MODEL.opt.timestep
        ],
        "max |tilt_sim - tilt_linear| / 0.01 rad, 0.01 rad release": float(
            np.abs(sim[:, 1] - lin[:, 1]).max() / 0.01
        ),
        "peak |u| for 0.01 rad release (N)": float(np.abs(run.ctrl).max()),
        "static capture bound (rad)": static_capture(),
    }


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    result = checks()
    print(json.dumps(result, indent=2))

    impulses = np.round(np.arange(0.0, 1.21, 0.04), 3)
    tilts = np.round(np.arange(0.0, 1.21, 0.04), 3)
    scales = np.round(np.geomspace(0.05, 20.0, 36), 4)
    peak = lab.sweep(push_peak, SEEDS, WORKERS, impulse=impulses)
    release = lab.sweep(release_ok, SEEDS, WORKERS, tilt=tilts)
    rate = lab.sweep(growth, SEEDS, WORKERS, scale=scales)
    up = lab.sweep(stays_up, SEEDS, WORKERS, scale=scales)
    with ProcessPoolExecutor(WORKERS) as pool:
        lin = np.array(list(pool.map(linear_rate, scales)))
        j_star = np.array(list(pool.map(boundary, [push_ok] * 5, [0.0] * 5, [3.0] * 5, SEEDS)))
        t_star = np.array(list(pool.map(boundary, [release_ok] * 5, [0.0] * 5, [1.5] * 5, SEEDS)))

    peak_at_limit = [push_peak(j - 1e-3, s) for j, s in zip(j_star, SEEDS, strict=True)]
    unstable = scales[lin > 0]
    result |= {
        "push limit J* (N s), per seed": j_star.round(4).tolist(),
        "peak tilt just below J* (rad), per seed": np.round(peak_at_limit, 4).tolist(),
        "release tilt limit (rad), per seed": t_star.round(4).tolist(),
        "linear: unstable pole-mass scales": unstable.tolist(),
        "linear: first unstable scale above 1": float(unstable[unstable > 1].min()) if (unstable > 1).any() else None,
        "linear: last unstable scale below 1": float(unstable[unstable < 1].max()) if (unstable < 1).any() else None,
        "sim: scales where every seed stays up": scales[up.min(axis=1) == 1].tolist(),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    plot(impulses, peak, tilts, release, scales, rate, lin, j_star, t_star)

    lo = float(j_star.min())
    lab.video(MODEL, push_run(lo * 0.95, 0), OUT / "push_recover.mp4")
    lab.video(MODEL, push_run(lo * 1.10, 0), OUT / "push_fail.mp4")
    big = float(unstable[unstable > 1].min()) if (unstable > 1).any() else 20.0
    q0, _, _ = draw(0)
    q0[1] += 0.05
    lab.video(scaled(big * 1.2), lab.rollout(scaled(big * 1.2), policy, 6.0, qpos0=q0, fail=fail),
              OUT / "heavy_pole.mp4")


def plot(impulses, peak, tilts, release, scales, rate, lin, j_star, t_star) -> None:
    fig = Figure(figsize=(13, 3.8), layout="constrained")
    ax1, ax2, ax3 = fig.subplots(1, 3)

    ok = ~np.isnan(peak).any(axis=1)
    lab.band(ax1, impulses[ok], peak[ok], "peak tilt, recovered (5 seeds)", BLUE)
    ax1.axvline(float(np.median(j_star)), color=BLUE, lw=1, ls="-.", label=f"push limit J* {np.median(j_star):.3f} N s")
    ax1.axhline(0.6, color=MUTED, ls="--", lw=1, label="user: fails at 0.6 rad")
    ax1.axhline(0.5, color=MUTED, ls=":", lw=1, label="Claude: 0.5 rad")
    ax1.set(xlabel="push impulse on the pole (N s, over 0.1 s)", ylabel="peak tilt (rad)",
            title="Push: peak tilt until the LQR fails")
    ax1.legend(fontsize=7, loc="upper left")

    lab.band(ax2, tilts, release, "fraction recovered (5 seeds)", BLUE)
    ax2.axvline(float(np.median(t_star)), color=BLUE, lw=1, ls="-.", label=f"sim limit {np.median(t_star):.3f} rad")
    ax2.axvline(static_capture(), color=ORANGE, lw=1.6, label=f"static bound {static_capture():.3f} rad")
    ax2.axvline(0.6, color=MUTED, ls="--", lw=1, label="user: 0.6 rad")
    ax2.set(xlabel="initial tilt, released at rest (rad)", ylabel="recovered",
            title="Release from a tilt")
    ax2.legend(fontsize=7, loc="lower left")

    ax3.plot(scales, lin, color=ORANGE, lw=1.6, label="linear: log|eig(A-BK)| / dt")
    lab.band(ax3, scales, rate, "sim: fitted tilt growth (5 seeds)", BLUE)
    ax3.axhline(0, color=INK, lw=0.6)
    for lo, hi, style, who in ((0.6, 3.0, "--", "user"), (0.5, 2.5, ":", "Claude")):
        ax3.axvline(lo, color=MUTED, ls=style, lw=1)
        ax3.axvline(hi, color=MUTED, ls=style, lw=1, label=f"{who}: {lo - 1:+.0%} to {hi - 1:+.0%}")
    ax3.set_xscale("log")
    ax3.set_ylim(min(-12, float(np.nanmin(lin)) - 1), max(4, float(np.nanmax(lin)) + 1))
    ax3.set(xlabel="pole mass / nominal (gains fixed)", ylabel="growth rate (1/s), < 0 is stable",
            title="Pole-mass error")
    ax3.legend(fontsize=7, loc="upper left")
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(OUT / "limits.png", dpi=130)
    print(f"plot {OUT / 'limits.png'}")


if __name__ == "__main__":
    main()
