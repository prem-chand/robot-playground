import math

import numpy as np
import pytest

from robot_pg import lab


def idle(obs: lab.Obs) -> np.ndarray:
    return np.zeros(1)


@pytest.mark.parametrize("at", [0.0, 7 * 0.02])
def test_push_gives_its_impulse_as_horizontal_momentum(at):
    run = lab.rollout(
        lab.load("cartpole"),
        idle,
        at + 0.2,
        pushes=[lab.Push("pole", at, 0.1, (5.0, 0.0, 0.0))],
    )
    (x_dot, th_dot), th = run.qvel[-1], run.qpos[-1, 1]
    momentum = 1.3 * x_dot + 0.3 * 0.4 * math.cos(th) * th_dot
    # Semi-implicit Euler drifts the momentum by O(dt): 1.8e-3 at 2 ms, half at 1 ms.
    assert momentum == pytest.approx(0.5, rel=5e-3)


def test_latency_delays_the_action_by_whole_ticks():
    run = lab.rollout(
        lab.load("cartpole"), lambda obs: np.array([obs["t"] + 1.0]), 0.1, latency_s=0.01
    )
    assert run.ctrl[:6, 0].tolist() == [1.0] * 6
    assert run.ctrl[6:8, 0] == pytest.approx([1.002, 1.004])


def test_rollout_stops_at_the_first_failure():
    run = lab.rollout(
        lab.load("cartpole"),
        idle,
        5.0,
        qpos0=np.array([0.0, 0.1]),
        fail=lambda obs: "fell" if abs(obs["qpos"][1]) > 1.0 else None,
    )
    assert run.reason == "fell"
    assert run.failed_at == pytest.approx(len(run.t) * 0.002)
    assert abs(run.qpos[-1, 1]) <= 1.0
