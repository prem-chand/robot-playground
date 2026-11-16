"""A gymnasium env on robot-playground-v0's cart-pole, with its control tick.

The tick holds ctrl = clip(action / gear, ctrlrange) for control_dt, then
runs mj_forward. The observation is [qpos, qvel] after that forward. The
reward sees the clipped command, in joint units.
"""

from collections.abc import Callable

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

Reward = Callable[[np.ndarray, np.ndarray], float]
Termination = Callable[[np.ndarray], bool]


def lqr_reward(obs: np.ndarray, command: np.ndarray) -> float:
    """LQR cost plus a constant alive bonus."""
    u = np.asarray(command, dtype=np.float64).ravel()
    return float(ALIVE_BONUS - (obs @ Q @ obs + u @ R @ u))


def balance_termination(obs: np.ndarray) -> bool:
    return bool(abs(obs[1]) > POLE_LIMIT or abs(obs[0]) > CART_LIMIT)


class CartPoleEnv(gymnasium.Env):
    def __init__(self, reward: Reward, termination: Termination, *, max_duration: float = 10.0) -> None:
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
        self._ticks = 0

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel])

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = INITIAL_QPOS
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._ticks = 0
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        command = np.asarray(action, dtype=np.float64)
        if command.shape != self.gear.shape:
            raise ValueError(f"action has shape {command.shape}, expected {self.gear.shape}")
        if not np.all(np.isfinite(command)):
            raise ValueError("action contains a nonfinite value")
        raw = command / self.gear
        ctrl = np.clip(raw, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
        self.data.ctrl[:] = ctrl
        for _ in range(self.physics_steps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._ticks += 1
        obs = self._obs()
        reward = float(self.reward_fn(obs, ctrl * self.gear))
        terminated = bool(self.termination_fn(obs))
        truncated = not terminated and self._ticks * CONTROL_DT >= self.max_duration
        return obs, reward, terminated, truncated, {"clipped": not np.array_equal(ctrl, raw)}
