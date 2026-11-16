"""The user's cart-pole PPO policy against an LQR on robot-playground-v0's plant.

uv run --extra dev --extra viz python experiments/cartpole_ppo/run.py

Both controllers run at 50 Hz and hold the action between ticks. Each sweep
runs over the CONTROLLERS table.
"""

import json
import math
from collections.abc import Callable
from concurrent.futures import Executor
from pathlib import Path

import mujoco
import numpy as np
from matplotlib.figure import Figure
from scipy.linalg import solve_continuous_are
from scipy.optimize import root_scalar

from robot_pg import lab

OUT = Path(__file__).parent
SEEDS = range(5)
WORKERS = 6
U_MAX = 10.0
CONTROL_DT = 0.02
Q, R = np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[1.0]])
PUSH_S = 0.1
NOISE = {"qpos": 1e-3, "qvel": 1e-2}
PLANT = "cartpole_playground"
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"

NET = np.load(OUT / "policy.npz")
W0, B0, W1, B1, W2, B2 = (NET[k] for k in ("W0", "b0", "W1", "b1", "W2", "b2"))
MODEL = lab.load(PLANT)


def mlp(obs: np.ndarray) -> np.ndarray:
    """float32 throughout, as in the torch policy."""
    h = np.tanh(obs @ W0.T + B0)
    h = np.tanh(h @ W1.T + B1)
    return h @ W2.T + B2


def continuous(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Continuous A and B from the velocity rows of the one-step map.

    MuJoCo's Euler step is v' = v + dt (M + dt D)^-1 f(q, v, u), with the joint
    damping D implicit. So (I + dt M^-1 D) (A_d - I) / dt on those rows is the
    exact acceleration Jacobian M^-1 df. The position rows are the identity
    block, as in robot-playground-v0.
    """
    a, b = lab.linearize(model)
    dt, nv = model.opt.timestep, model.nv
    mass = np.zeros((nv, nv))
    mujoco.mj_fullM(model, lab.home(model), mass)
    implicit = np.eye(nv) + dt * np.linalg.solve(mass, np.diag(model.dof_damping))
    ac = np.zeros_like(a)
    ac[:nv, nv:] = np.eye(nv)
    ac[nv:] = implicit @ (a[nv:] - np.eye(2 * nv)[nv:]) / dt
    bc = np.zeros_like(b)
    bc[nv:] = implicit @ b[nv:] / dt
    return ac, bc


def b_exact(model: mujoco.MjModel) -> np.ndarray:
    """M^-1 times the cart force direction, the true continuous B rows."""
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, lab.home(model), mass)
    return np.linalg.solve(mass, np.array([[1.0], [0.0]]))


def care_gain(model: mujoco.MjModel) -> np.ndarray:
    a, b = continuous(model)
    p = solve_continuous_are(a, b, Q, R)
    return np.linalg.solve(R, b.T @ p)


K = care_gain(MODEL)


def ppo(obs: lab.Obs) -> np.ndarray:
    x = np.r_[obs["qpos"], obs["qvel"]].astype(np.float32)
    return mlp(x).astype(np.float64)


def lqr(obs: lab.Obs) -> np.ndarray:
    return -K @ np.r_[obs["qpos"], obs["qvel"]]


CONTROLLERS: dict[str, lab.Policy] = {"ppo": ppo, "lqr": lqr}
NAMES = list(CONTROLLERS)
COLOR = {"ppo": ORANGE, "lqr": BLUE}
PREDICTED = {"ppo": {"J*": 0.07, "tilt": 0.30, "mass": (0.3, 4.0)},
             "lqr": {"J*": 0.12, "tilt": 0.55, "mass": (0.05, 10.0)}}


class PeakOutput:
    """Wraps a policy and keeps its largest raw |output|, before any clip."""

    def __init__(self, policy: lab.Policy) -> None:
        self.policy, self.peak = policy, 0.0

    def __call__(self, obs: lab.Obs) -> np.ndarray:
        u = self.policy(obs)
        self.peak = max(self.peak, float(np.abs(u).max()))
        return u


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


def roll(policy: lab.Policy, seconds: float, seed: int, q0: np.ndarray,
         model: mujoco.MjModel = MODEL, pushes: tuple[lab.Push, ...] = ()) -> lab.Run:
    return lab.rollout(model, policy, seconds, seed=seed, qpos0=q0, control_dt=CONTROL_DT,
                       obs_noise=NOISE, fail=fail, pushes=pushes)


def recovered(run: lab.Run) -> bool:
    return bool(not run.failed and abs(run.qpos[-1, 1]) < 0.05)


def push_run(ctl: str, impulse: float, seed: int) -> lab.Run:
    q0, sign, at = draw(seed)
    force = sign * impulse / PUSH_S
    return roll(CONTROLLERS[ctl], 6.0, seed, q0,
                pushes=(lab.Push("pole", at, PUSH_S, (force, 0.0, 0.0)),))


def push_peak(ctl: str, impulse: float, seed: int) -> float:
    """Peak tilt in rad, or NaN when the controller does not recover."""
    run = push_run(ctl, impulse, seed)
    return float(np.abs(run.qpos[:, 1]).max()) if recovered(run) else math.nan


def push_ok(ctl: str, impulse: float, seed: int) -> bool:
    return not math.isnan(push_peak(ctl, impulse, seed))


def release(ctl: str, tilt: float, seed: int, side: float) -> float:
    q0, _, _ = draw(seed)
    q0[1] = side * tilt
    return float(recovered(roll(CONTROLLERS[ctl], 6.0, seed, q0)))


def release_ok(ctl: str, tilt: float, seed: int) -> float:
    """Tilt toward the seed's push sign."""
    return release(ctl, tilt, seed, draw(seed)[1])


def release_plus(ctl: str, tilt: float, seed: int) -> float:
    return release(ctl, tilt, seed, 1.0)


def release_minus(ctl: str, tilt: float, seed: int) -> float:
    return release(ctl, tilt, seed, -1.0)


def scaled(scale: float) -> mujoco.MjModel:
    m = lab.load(PLANT)
    pole = m.body("pole").id
    m.body_mass[pole] *= scale
    m.body_inertia[pole] *= scale
    return m


def stays_up(ctl: str, scale: float, seed: int) -> float:
    q0, _, _ = draw(seed)
    q0[1] += 0.05
    return float(recovered(roll(CONTROLLERS[ctl], 15.0, seed, q0, model=scaled(scale))))


def transfer(plant: str, seed: int, side: float | None = None) -> dict:
    """The PPO policy from a 0.05 rad release on `plant`, with its state and
    force where |tilt| first passes 0.2 rad, the end of its training range.
    `side` None tilts toward the seed's push sign."""
    q0, sign, _ = draw(seed)
    q0[1] = (sign if side is None else side) * 0.05
    policy = PeakOutput(ppo)
    run = roll(policy, 15.0, seed, q0, model=lab.load(plant))
    out = np.flatnonzero(np.abs(run.qpos[:, 1]) > 0.2)
    edge = None
    if out.size:
        i = int(out[0])
        edge = {"t (s)": float(run.t[i]), "tilt (rad)": round(float(run.qpos[i, 1]), 4),
                "tiltdot (rad/s)": round(float(run.qvel[i, 1]), 4), "u (N)": round(float(run.ctrl[i, 0]), 4)}
    return {"plant": plant, "seed": seed, "tilt0 (rad)": float(q0[1]), "recovered": recovered(run), "failed_at": run.failed_at,
            "peak |tilt| (rad)": round(float(np.abs(run.qpos[:, 1]).max()), 4),
            "peak raw |u| (N)": round(policy.peak, 4), "at first |tilt| > 0.2 rad": edge}


Ok = Callable[[str, float, int], float | bool]


def boundary(ok: Ok, ctl: str, lo: float, hi: float, seed: int, steps: int = 14) -> float:
    if not ok(ctl, lo, seed) or ok(ctl, hi, seed):
        raise ValueError(f"{ctl} seed {seed}: bracket [{lo}, {hi}] does not hold one boundary")
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(ctl, mid, seed) else (lo, mid)
    return lo


def bracket(grid: np.ndarray, passed: np.ndarray) -> tuple[float, float] | None:
    """Grid points around the first failure, or None when the first point fails.

    The bisection runs only inside this bracket, so a policy that passes, fails
    and passes again reports its first failure.
    """
    i = int(np.argmin(passed))
    if passed[i]:
        raise ValueError("no grid point fails, so widen the grid")
    return None if i == 0 else (float(grid[i - 1]), float(grid[i]))


def passes_above_first_fail(passed: np.ndarray) -> int:
    return int(passed[int(np.argmin(passed)):].sum())


def limits_by_bisection(pool: Executor, ok: Ok, grid: np.ndarray, passed: np.ndarray) -> np.ndarray:
    """Limit per (controller, seed) from a (controller, grid, seed) pass table.
    A seed that fails at the first grid point gets 0."""
    out = np.zeros((len(NAMES), len(SEEDS)))
    jobs = [(i, s, b) for i, c in enumerate(NAMES) for s in SEEDS
            if (b := bracket(grid, passed[i, :, s])) is not None]
    found = pool.map(boundary, [ok] * len(jobs), [NAMES[i] for i, _, _ in jobs],
                     [b[0] for *_, b in jobs], [b[1] for *_, b in jobs], [s for _, s, _ in jobs])
    for (i, s, _), limit in zip(jobs, found, strict=True):
        out[i, s] = limit
    return out


def baseline(ctl: str, seed: int) -> dict:
    """15 s with no push and no added tilt."""
    q0, _, _ = draw(seed)
    run = roll(CONTROLLERS[ctl], 15.0, seed, q0)
    return {"failed_at (s)": run.failed_at, "x at the end (m)": round(float(run.qpos[-1, 0]), 4),
            "min x (m)": round(float(run.qpos[:, 0].min()), 4)}


def mlp_jacobian(obs: np.ndarray) -> np.ndarray:
    w0, b0, w1, b1, w2 = (w.astype(np.float64) for w in (W0, B0, W1, B1, W2))
    h1 = np.tanh(w0 @ obs + b0)
    h2 = np.tanh(w1 @ h1 + b1)
    return w2 @ ((1 - h2**2)[:, None] * w1) @ ((1 - h1**2)[:, None] * w0)


def ppo_local_gains() -> dict:
    """PPO slope du/dstate at rest and upright, along the cart position, and its
    rest equilibria (u = 0). The eigenvalue fields are for the linear closed loop
    with that slope. Away from an equilibrium they are only local."""
    ac, bc = continuous(MODEL)

    def u_rest(x: float) -> float:
        return float(mlp(np.array([x, 0.0, 0.0, 0.0], dtype=np.float32))[0])

    def max_re(x: float) -> float:
        return round(float(np.linalg.eigvals(ac + bc @ mlp_jacobian(np.array([x, 0.0, 0.0, 0.0]))).real.max()), 4)

    xs = np.round(np.arange(-0.6, 0.61, 0.1), 2)
    fine = np.linspace(-1.5, 1.5, 301)
    signs = np.sign([u_rest(x) for x in fine])
    roots = [float(root_scalar(u_rest, bracket=(fine[i], fine[i + 1])).root) for i in np.flatnonzero(signs[:-1] != signs[1:])]
    return {
        "along x": [{"x (m)": float(x), "u at rest (N)": round(u_rest(x), 4),
                     "du/d(x, tilt, xdot, tiltdot)": mlp_jacobian(np.array([x, 0.0, 0.0, 0.0])).ravel().round(3).tolist(),
                     "max Re eig (1/s)": max_re(x)} for x in xs],
        "rest equilibria, |x| <= 1.5 m": [{"x (m)": round(r, 4), "max Re eig (1/s)": max_re(r)} for r in roots],
    }


def static_capture(model: mujoco.MjModel = MODEL) -> float:
    """Largest tilt at rest where full cart force still gives theta'' <= 0."""
    cart, pole = model.body("cart").id, model.body("pole").id
    m_cart, m_pole = model.body_mass[cart], model.body_mass[pole]
    l_com = model.body_ipos[pole, 2]
    i_pole = model.body_inertia[pole, 1] + model.dof_armature[model.jnt_dofadr[model.joint("pole").id]]
    g = -model.opt.gravity[2]

    def theta_dd(th: float) -> float:
        mass = np.array([[m_cart + m_pole, m_pole * l_com * math.cos(th)],
                         [m_pole * l_com * math.cos(th), i_pole + m_pole * l_com**2]])
        rhs = np.array([U_MAX, m_pole * g * l_com * math.sin(th)])
        return float(np.linalg.solve(mass, rhs)[1])

    lo, hi = 0.0, math.pi / 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if theta_dd(mid) <= 0 else (lo, mid)
    return lo


def replay() -> dict:
    """robot-playground-v0 rows are post-tick states, so row i is physics step 10 (i + 1)."""
    every = round(CONTROL_DT / MODEL.opt.timestep)
    policy = PeakOutput(ppo)
    run = lab.rollout(MODEL, policy, 10.0 + CONTROL_DT, qpos0=NET["ref_qpos0"],
                      control_dt=CONTROL_DT)
    rows = every * (np.arange(len(NET["ref_t"])) + 1)
    assert np.allclose(run.t[rows], NET["ref_t"]), "replay rows line up in time"
    dq = np.abs(run.qpos[rows] - NET["ref_qpos"]).max(axis=0)
    dv = np.abs(run.qvel[rows] - NET["ref_qvel"]).max(axis=0)
    du = np.abs(run.ctrl[rows - every] - NET["ref_ctrl"]).max()
    return {
        "max |dx| (m)": float(dq[0]), "max |dtilt| (rad)": float(dq[1]),
        "max |dxdot| (m/s)": float(dv[0]), "max |dtiltdot| (rad/s)": float(dv[1]),
        "max |du| (N)": float(du), "failed": run.failed,
        "peak raw |u| on the 5 deg start (N)": policy.peak,
    }


def checks() -> dict:
    k_play = NET["K_playground"]
    undamped = lab.load(PLANT)
    undamped.dof_damping[:] = 0.0
    a, b = lab.linearize(MODEL)
    every = round(CONTROL_DT / MODEL.opt.timestep)
    phi = np.linalg.matrix_power(a, every)
    gamma = sum(np.linalg.matrix_power(a, j) @ b for j in range(every))
    ac, bc = continuous(MODEL)
    numpy_out = mlp(NET["check_obs"])
    return {
        "max |numpy - torch|, 10000 obs, run.py": float(np.abs(numpy_out - NET["check_torch"]).max()),
        "max |numpy - torch|, 10000 obs, export": float(NET["max_abs_numpy_minus_torch"]),
        "replay of robot-playground-v0's 10 s PPO run": replay(),
        "K (lab.linearize, with damping)": K.ravel().tolist(),
        "K_playground": k_play.ravel().tolist(),
        "max |K - K_playground|": float(np.abs(K - k_play).max()),
        "max |K with damping zeroed - K_playground|": float(np.abs(care_gain(undamped) - k_play).max()),
        "max |B_c - M^-1 [1, 0]|, with damping": float(np.abs(bc[2:] - b_exact(MODEL)).max()),
        "max |B_c - M^-1 [1, 0]|, with damping, no implicit correction": float(
            np.abs(b[2:] / MODEL.opt.timestep - b_exact(MODEL)).max()
        ),
        "continuous closed-loop poles (1/s)": [
            f"{z.real:.3f}{z.imag:+.3f}j" for z in np.sort_complex(np.linalg.eigvals(ac - bc @ K))
        ],
        "50 Hz hold: closed-loop spectral radius": float(np.abs(np.linalg.eigvals(phi - gamma @ K)).max()),
        "static capture bound (rad)": static_capture(),
        "closed form atan(U / ((M + m) g)) (rad)": math.atan(
            U_MAX / (MODEL.body_mass.sum() * -MODEL.opt.gravity[2])
        ),
    }


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    result = checks()
    print(json.dumps(result, indent=2))

    impulses = np.round(np.r_[np.arange(0.0, 0.04, 0.001), np.arange(0.04, 0.601, 0.015)], 4)
    tilts = np.round(np.r_[np.arange(0.0, 0.14, 0.002), np.arange(0.14, 0.801, 0.02)], 4)
    scales = np.round(np.geomspace(0.05, 20.0, 36), 4)
    peak = lab.sweep(push_peak, SEEDS, WORKERS, ctl=NAMES, impulse=impulses.tolist())
    release = lab.sweep(release_ok, SEEDS, WORKERS, ctl=NAMES, tilt=tilts.tolist())
    sides = {side: lab.sweep(fn, SEEDS, WORKERS, ctl=NAMES, tilt=tilts.tolist())
             for side, fn in (("+", release_plus), ("-", release_minus))}
    up = lab.sweep(stays_up, SEEDS, WORKERS, ctl=NAMES, scale=scales.tolist())

    with ProcessPoolExecutor(WORKERS) as pool:
        j_star = limits_by_bisection(pool, push_ok, impulses, ~np.isnan(peak))
        t_star = limits_by_bisection(pool, release_ok, tilts, release == 1)
        t_side = {side: limits_by_bisection(pool, fn, tilts, sides[side] == 1)
                  for side, fn in (("+", release_plus), ("-", release_minus))}
        n = len(SEEDS)
        probe = list(pool.map(transfer, ["cartpole"] * n + [PLANT] * n + ["cartpole"] * n,
                              [*SEEDS] * 3, [None] * (2 * n) + [1.0] * n))
        quiet = {c: list(pool.map(baseline, [c] * len(SEEDS), SEEDS)) for c in NAMES}

    limits = {}
    for i, c in enumerate(NAMES):
        js, ts = j_star[i], t_star[i]
        limits[c] = {
            "no disturbance, 15 s, per seed": quiet[c],
            "push limit J* (N s), per seed": js.round(5).tolist(),
            "push limit J* median (N s)": float(np.median(js)),
            "peak tilt just below J* (rad), per seed": [
                round(push_peak(c, j - 1e-4, s), 4) if j > 0 else None for j, s in zip(js, SEEDS, strict=True)
            ],
            "push grid points that pass above the first failure, per seed": [
                passes_above_first_fail(~np.isnan(peak[i, :, s])) for s in SEEDS
            ],
            "release tilt limit (rad), per seed": ts.round(5).tolist(),
            "release tilt limit median (rad)": float(np.median(ts)),
            "release limit / static bound, median": float(np.median(ts) / static_capture()),
            "release tilt limit, +tilt side (rad), per seed": t_side["+"][i].round(5).tolist(),
            "release tilt limit, -tilt side (rad), per seed": t_side["-"][i].round(5).tolist(),
            "tilt grid points that pass above the first failure, per seed": [
                passes_above_first_fail(release[i, :, s] == 1) for s in SEEDS
            ],
            "sim: scales where every seed stays up": scales[up[i].min(axis=1) == 1].tolist(),
            "sim: fraction of seeds up, per scale": up[i].mean(axis=1).round(2).tolist(),
        }
    result |= {
        "draw per seed: (x0, tilt0, push sign)": [
            [round(float(q[0]), 4), round(float(q[1]), 4), sign] for q, sign, _ in map(draw, SEEDS)
        ],
        "PPO local gains": ppo_local_gains(),
        "grid: scales": scales.tolist(),
        "limits": limits,
        "transfer probe, 0.05 rad release (robot-pg cartpole, the vendored plant as control, then robot-pg cartpole at +tilt)": probe,
    }

    videos = {}
    j0 = {c: limits[c]["push limit J* (N s), per seed"][0] for c in NAMES}
    for name, c, f in (("ppo_push_recover", "ppo", 0.95), ("ppo_push_fail", "ppo", 1.10),
                       ("lqr_push_recover", "lqr", 0.95)):
        run = push_run(c, j0[c] * f, 0)
        lab.video(MODEL, run, OUT / f"{name}.mp4")
        videos[f"{name}.mp4"] = {"impulse (N s)": round(j0[c] * f, 5), "recovered": recovered(run),
                                 "failed_at (s)": run.failed_at}
    result["videos (seed 0)"] = videos
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    plot(impulses, peak, tilts, release, scales, up, limits)


def plot(impulses, peak, tilts, release, scales, up, limits) -> None:
    fig = Figure(figsize=(13, 5.2), layout="constrained")
    ax1, ax2, ax3 = fig.subplots(1, 3)
    for i, c in enumerate(NAMES):
        color, lim, guess = COLOR[c], limits[c], PREDICTED[c]
        j_med, t_med = lim["push limit J* median (N s)"], lim["release tilt limit median (rad)"]
        lab.band(ax1, impulses, (~np.isnan(peak[i])).astype(float), f"{c.upper()}: fraction recovered", color)
        ax1.axvline(j_med, color=color, lw=1, ls="-.", label=f"{c.upper()} J* {j_med:.3f} N s")
        ax1.axvline(guess["J*"], color=color, lw=1, ls=":", label=f"Claude: {guess['J*']} N s")
        lab.band(ax2, tilts, release[i], f"{c.upper()}: fraction recovered", color)
        ax2.axvline(t_med, color=color, lw=1, ls="-.", label=f"{c.upper()} limit {t_med:.3f} rad")
        ax2.axvline(guess["tilt"], color=color, lw=1, ls=":", label=f"Claude: {guess['tilt']} rad")
        plus = float(np.median(lim["release tilt limit, +tilt side (rad), per seed"]))
        ax2.axvline(plus, color=color, lw=1, ls="--", label=f"{c.upper()} +tilt side only {plus:.3f} rad")
        lab.band(ax3, scales, up[i], f"{c.upper()}: fraction up for 15 s", color)
        for end in guess["mass"]:
            ax3.axvline(end, color=color, lw=1, ls=":")
    ax2.axvline(static_capture(), color=INK, lw=1.6, label=f"static bound {static_capture():.3f} rad")
    ax3.plot([], [], color=MUTED, lw=1, ls=":", label="Claude: predicted range")
    ax1.set(xlabel="push impulse on the pole (N s, over 0.1 s)", ylabel="recovered",
            title="Push on the pole")
    ax2.set(xlabel="initial tilt, released at rest (rad)", ylabel="recovered",
            title="Release from a tilt")
    for ax in (ax1, ax2):
        ax.set_xscale("symlog", linthresh=0.01, linscale=0.5)
        ax.set_xlim(left=0.0)
    ax3.set_xscale("log")
    ax3.set(xlabel="pole mass / nominal (controller fixed)", ylabel="fraction of seeds up",
            title="Pole-mass error")
    for ax in (ax1, ax2, ax3):
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(OUT / "limits.png", dpi=130)
    print(f"plot {OUT / 'limits.png'}")


if __name__ == "__main__":
    main()
