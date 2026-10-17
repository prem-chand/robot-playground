"""Load a model, roll out a policy, sweep it, and save a plot and a video.

A policy maps an observation dict to an action array in command units. The
observation holds the plant signals plus the full `qpos`, `qvel`, and `t`.
"""

import itertools
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robot_pg.contract import Array
from robot_pg.plant import HOME_KEY, MujocoPlant, RunUnstable, parse_model
from robot_pg.robots import ASSETS, ROBOTS

Obs = Mapping[str, Any]
Policy = Callable[[Obs], Array]


@dataclass(frozen=True)
class Push:
    """A world-frame force at the center of mass of `body`."""

    body: str
    at_s: float
    duration_s: float
    force: tuple[float, float, float]


@dataclass(frozen=True)
class Run:
    """Row k is the state sensed at tick k and the ctrl applied over tick k."""

    t: Array
    qpos: Array
    qvel: Array
    ctrl: Array
    failed_at: float | None
    reason: str = ""

    @property
    def failed(self) -> bool:
        return self.failed_at is not None


def load(name: str, edit: Callable[[mujoco.MjSpec], None] | None = None) -> mujoco.MjModel:
    """A registry name, a folder under `robots/` holding `<name>.xml`, or a path."""
    robot = ROBOTS.get(name)
    if robot is not None:
        path = ASSETS / robot.mjcf
    elif name.endswith(".xml"):
        path = Path(name)
    else:
        path = ASSETS / name / f"{name}.xml"
    spec = mujoco.MjSpec.from_file(str(path))
    if robot is not None and (site := robot.ee_site) is not None:
        spec.body(site.body).add_site(name=site.name, pos=list(site.pos))
    if edit is not None:
        edit(spec)
    return spec.compile()


def home(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return data


def linearize(model: mujoco.MjModel, eps: float = 1e-6) -> tuple[Array, Array]:
    """Discrete A and B of one physics step at `home`, by centered differences."""
    data = home(model)
    nx = 2 * model.nv + model.na
    a, b = np.zeros((nx, nx)), np.zeros((nx, model.nu))
    mujoco.mjd_transitionFD(model, data, eps, True, a, b, None, None)
    return a, b


def rollout(
    model: mujoco.MjModel,
    policy: Policy,
    seconds: float,
    *,
    seed: int = 0,
    qpos0: Array | None = None,
    latency_s: float = 0.0,
    control_dt: float | None = None,
    pushes: Sequence[Push] = (),
    obs_noise: Mapping[str, float] | None = None,
    fail: Callable[[Obs], str | None] | None = None,
) -> Run:
    """Run from `home` (plus `qpos0` if given) and stop at the first failure.

    The policy runs every `control_dt` and its action reaches the plant
    `latency_s` later. Before the first action arrives, the plant holds the
    `home` ctrl. `obs_noise` adds Gaussian noise of that std to each named key.
    """
    dt = model.opt.timestep
    n = round(seconds / dt)
    every = 1 if control_dt is None else max(1, round(control_dt / dt))
    delay = round(latency_s / dt)
    rng = np.random.default_rng(seed)
    spec = parse_model(model)
    plant = MujocoPlant(model, spec)
    d = plant.data
    if qpos0 is not None:
        d.qpos[:] = qpos0
    signals = {s.name: np.zeros(s.shape) for s in spec.signals}
    held = d.ctrl / np.array([a.scale for a in spec.actuators])
    pending: deque[tuple[int, Array]] = deque()
    bodies = [(model.body(p.body).id, p) for p in pushes]
    qpos, qvel = np.zeros((n, model.nq)), np.zeros((n, model.nv))
    ctrl = np.zeros((n, model.nu))
    failed_at, reason, k = None, "", 0
    for k in range(n):
        plant.sense(signals)
        obs = {**{s: v.copy() for s, v in signals.items()}, "qpos": d.qpos.copy(),
               "qvel": d.qvel.copy(), "t": k * dt}
        qpos[k], qvel[k] = d.qpos, d.qvel
        if fail is not None and (why := fail(obs)):
            failed_at, reason = k * dt, why
            break
        if k % every == 0:
            for key, std in (obs_noise or {}).items():
                obs[key] = obs[key] + rng.normal(0.0, std, np.shape(obs[key]))
            pending.append((k + delay, np.asarray(policy(obs), dtype=np.float64)))
        while pending and pending[0][0] <= k:
            held = pending.popleft()[1]
        d.xfrc_applied[:] = 0.0
        for body, p in bodies:
            if p.at_s <= k * dt < p.at_s + p.duration_s:
                d.xfrc_applied[body, :3] = p.force
        try:
            plant.act(held)
        except RunUnstable as bad:
            failed_at, reason = k * dt, str(bad)
            break
        ctrl[k] = d.ctrl
    else:
        k = n
    return Run(np.arange(k) * dt, qpos[:k], qvel[:k], ctrl[:k], failed_at, reason)


def sweep(
    fn: Callable[..., float],
    seeds: Sequence[int],
    workers: int = 1,
    **axes: Sequence[Any],
) -> Array:
    """`fn(**point, seed=s)` over the grid of `axes` and `seeds`, shaped
    (*axis lengths, len(seeds)). `fn` must be a module-level function when
    `workers > 1`."""
    names = list(axes)
    points = [
        dict(zip(names, values, strict=True), seed=s)
        for values in itertools.product(*axes.values())
        for s in seeds
    ]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            out = list(pool.map(_call, itertools.repeat(fn), points, chunksize=4))
    else:
        out = [fn(**p) for p in points]
    return np.array(out, dtype=np.float64).reshape(
        *(len(v) for v in axes.values()), len(seeds)
    )


def _call(fn: Callable[..., float], point: dict[str, Any]) -> float:
    return fn(**point)


def video(model: mujoco.MjModel, run: Run, path: str | Path, fps: int = 30) -> Path:
    """Uses the model's camera named `lab` when it has one, else a fitted view."""
    import imageio.v2 as imageio

    from robot_pg.rendering import frames

    qpos = run.qpos[:: max(1, round(1 / (fps * model.opt.timestep)))]
    cams = [model.camera(i).name for i in range(model.ncam)]
    pictures: list[Any]
    if "lab" not in cams:
        pictures = list(frames(model, qpos))
    else:
        cam = cams.index("lab")
        data = mujoco.MjData(model)
        g = model.vis.global_
        with mujoco.Renderer(model, g.offheight, g.offwidth) as renderer:
            pictures = []
            for row in qpos:
                data.qpos[:] = row
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                pictures.append(renderer.render().copy())
    dest = Path(path)
    imageio.mimsave(dest, pictures, fps=fps)
    return dest


def band(ax: Any, x: Array, y: Array, label: str, color: str | None = None) -> None:
    """Mean line, a +-1 std band, and one dot per seed. `y` is (len(x), seeds)."""
    mean, std = y.mean(axis=1), y.std(axis=1)
    (line,) = ax.plot(x, mean, color=color, lw=1.6, label=label)
    c = line.get_color()
    ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.2, lw=0)
    ax.plot(np.repeat(x, y.shape[1]), y.ravel(), ".", color=c, ms=2, alpha=0.5)
