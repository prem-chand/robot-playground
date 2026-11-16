"""Replay a robot-playground-v0 env trace (MuJoCo 3.8.0) through the port.

scripts/export_env_reference.py wrote tests/data/cartpole_env_reference.npz.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from robot_pg.rl.env import CartPoleEnv, balance_termination, lqr_reward

TRACE = Path(__file__).parent / "data" / "cartpole_env_reference.npz"


def replay(trace) -> dict[str, np.ndarray]:
    """Feed the trace actions to a fresh env, and reset where the trace reset."""
    env = CartPoleEnv(lqr_reward, balance_termination)
    obs, _ = env.reset(seed=0)
    out: dict[str, list] = {k: [] for k in ("obs", "next_obs", "reward", "terminated", "truncated", "clipped")}
    for action in trace["action"]:
        next_obs, reward, terminated, truncated, info = env.step(action)
        for key, value in zip(out, (obs, next_obs, reward, terminated, truncated, info["clipped"]), strict=True):
            out[key].append(value)
        obs = env.reset()[0] if terminated or truncated else next_obs
    return {k: np.asarray(v) for k, v in out.items()}


def test_replays_the_robot_playground_v0_env_trace():
    trace = np.load(TRACE)
    got = replay(trace)
    assert (trace["terminated"].sum(), trace["clipped"].sum()) == (56, 281)
    assert np.array_equal(got["terminated"], trace["terminated"])
    assert np.array_equal(got["truncated"], trace["truncated"])
    assert np.array_equal(got["clipped"], trace["clipped"])
    assert np.abs(got["obs"] - trace["obs"]).max() < 1e-10
    assert np.abs(got["next_obs"] - trace["next_obs"]).max() < 1e-10
    assert np.abs(got["reward"] - trace["reward"]).max() < 1e-9
