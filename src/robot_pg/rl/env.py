"""A gymnasium env on robot-playground-v0's cart-pole, with its control tick.

The tick holds ctrl = clip(action / gear, ctrlrange) for control_dt, then
runs mj_forward. The observation is [qpos, qvel] after that forward. The
reward sees the clipped command, in joint units.

A `TrainConfig` names one training distribution: the reward, the
termination, and the `Randomization` of the start state, the pole pushes
and the observation noise. CONFIGS holds v0 (robot-playground-v0's
setup) and v1. The reward and the termination see the true state. Only
the returned observation gets the noise.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

from robot_pg import lab

PLANT = "cartpole_playground"
CONTROL_DT = 0.02
INITIAL_QPOS = (0.0, 0.08726646259971647)
Q = np.diag([1.0, 1.0, 10.0, 1.0])
R = np.array([[1.0]])
ALIVE_BONUS = 5.0
POLE_LIMIT = 0.2
CART_LIMIT = 1.5
FALL_POLE_LIMIT = math.pi / 2
FALL_CART_LIMIT = 3.0
REWARD_FLOOR = 0.5
PUSH_S = 0.1
PUSH_TICKS = round(PUSH_S / CONTROL_DT)

Reward = Callable[[np.ndarray, np.ndarray], float]
Termination = Callable[[np.ndarray], bool]


def lqr_reward(obs: np.ndarray, command: np.ndarray) -> float:
    """LQR cost plus a constant alive bonus."""
    u = np.asarray(command, dtype=np.float64).ravel()
    return float(ALIVE_BONUS - (obs @ Q @ obs + u @ R @ u))


def floored_lqr_reward(obs: np.ndarray, command: np.ndarray) -> float:
    """lqr_reward, but never below REWARD_FLOOR > 0, so a step alive always
    pays more than the end of the episode."""
    return max(lqr_reward(obs, command), REWARD_FLOOR)


def balance_termination(obs: np.ndarray) -> bool:
    return bool(abs(obs[1]) > POLE_LIMIT or abs(obs[0]) > CART_LIMIT)


def fall_termination(obs: np.ndarray) -> bool:
    """The fail rule of the cart-pole experiments."""
    return bool(abs(obs[1]) > FALL_POLE_LIMIT or abs(obs[0]) > FALL_CART_LIMIT)


@dataclass(frozen=True)
class Randomization:
    """Each start component is uniform in center +- spread, in [x, tilt, xdot, tiltdot].

    Push starts form a Poisson process with a mean gap of `push_interval` s.
    Each push is a world-x force on the pole's center of mass for PUSH_S,
    with an impulse uniform in +-[0, push_max] N s. A new push starts only
    after the last one ends.
    """

    start_center: tuple[float, float, float, float] = (*INITIAL_QPOS, 0.0, 0.0)
    start_spread: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    push_max: float = 0.0
    push_interval: float = math.inf
    obs_noise: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class TrainConfig:
    name: str
    reward: Reward
    termination: Termination
    randomization: Randomization

    def env(self) -> "CartPoleEnv":
        return CartPoleEnv(self.reward, self.termination, randomization=self.randomization)

    def describe(self) -> dict:
        """Plain values only, so that torch.load(weights_only=True) reads them."""
        r = self.randomization
        return {
            "name": self.name, "reward": self.reward.__name__, "termination": self.termination.__name__,
            "start_center": list(r.start_center), "start_spread": list(r.start_spread),
            "push_max (N s)": r.push_max, "push_interval (s)": r.push_interval, "push_s": PUSH_S,
            "obs_noise (qpos, qvel)": list(r.obs_noise),
        }


CONFIGS = {
    "v0": TrainConfig("v0", lqr_reward, balance_termination, Randomization()),
    "v1": TrainConfig(
        "v1", floored_lqr_reward, fall_termination,
        Randomization(start_center=(0.0, 0.0, 0.0, 0.0), start_spread=(0.5, 0.5, 0.2, 0.2),
                      push_max=0.25, push_interval=3.0, obs_noise=(1e-3, 1e-2)),
    ),
}


class CartPoleEnv(gymnasium.Env):
    def __init__(
        self,
        reward: Reward,
        termination: Termination,
        *,
        max_duration: float = 10.0,
        randomization: Randomization = Randomization(),
    ) -> None:
        super().__init__()
        if not np.isfinite(max_duration) or max_duration <= 0.0:
            raise ValueError(f"max_duration must be finite and positive, got {max_duration}")
        self.model = lab.load(PLANT)
        self.data = mujoco.MjData(self.model)
        self.reward_fn, self.termination_fn = reward, termination
        self.max_duration = float(max_duration)
        self.physics_steps = round(CONTROL_DT / self.model.opt.timestep)
        self.gear = self.model.actuator_gear[:, 0].copy()
        self.ctrl_limits = self.model.actuator_ctrlrange.copy()
        limits = self.ctrl_limits * self.gear[:, None]
        self.observation_space = spaces.Box(-np.inf, np.inf, (2 * self.model.nv,), np.float64)
        self.action_space = spaces.Box(limits[:, 0], limits[:, 1], dtype=np.float64)
        self.randomization = randomization
        self.pole = self.model.body("pole").id
        n_ticks = math.ceil(self.max_duration / CONTROL_DT)
        self._sigma = np.repeat(randomization.obs_noise, self.model.nv)
        self._push_force = np.zeros(n_ticks + PUSH_TICKS)
        self.pushes: list[tuple[float, float]] = []
        self._ticks = 0

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _noisy(self, state: np.ndarray) -> np.ndarray:
        return state + self.np_random.normal(0.0, self._sigma)

    def _draw_pushes(self) -> None:
        """Fill `pushes` with (start s, impulse N s) and the per-tick force table."""
        r = self.randomization
        self._push_force[:] = 0.0
        self.pushes = []
        t = float(self.np_random.exponential(r.push_interval))
        while t < self.max_duration:
            tick = round(t / CONTROL_DT)
            impulse = r.push_max * float(self.np_random.uniform()) * float(self.np_random.choice([-1.0, 1.0]))
            self.pushes.append((tick * CONTROL_DT, impulse))
            self._push_force[tick : tick + PUSH_TICKS] = impulse / PUSH_S
            t = (tick + PUSH_TICKS) * CONTROL_DT + float(self.np_random.exponential(r.push_interval))

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        r = self.randomization
        start = np.asarray(r.start_center) + np.asarray(r.start_spread) * self.np_random.uniform(-1.0, 1.0, 4)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = start[:2]
        self.data.qvel[:] = start[2:]
        mujoco.mj_forward(self.model, self.data)
        self._ticks = 0
        self._draw_pushes()
        return self._noisy(self._obs()), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        command = np.asarray(action, dtype=np.float64)
        if command.shape != self.gear.shape:
            raise ValueError(f"action has shape {command.shape}, expected {self.gear.shape}")
        if not np.all(np.isfinite(command)):
            raise ValueError("action contains a nonfinite value")
        raw = command / self.gear
        ctrl = np.clip(raw, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
        self.data.ctrl[:] = ctrl
        self.data.xfrc_applied[self.pole, 0] = self._push_force[self._ticks]
        for _ in range(self.physics_steps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._ticks += 1
        obs = self._obs()
        reward = float(self.reward_fn(obs, ctrl * self.gear))
        terminated = bool(self.termination_fn(obs))
        truncated = not terminated and self._ticks * CONTROL_DT >= self.max_duration
        info = {"clipped": not np.array_equal(ctrl, raw), "state": obs}
        return self._noisy(obs), reward, terminated, truncated, info
