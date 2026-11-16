import numpy as np
import pytest

pytest.importorskip("gymnasium")

from robot_pg.rl.env import CartPoleEnv, balance_termination, lqr_reward


def make() -> CartPoleEnv:
    return CartPoleEnv(lqr_reward, balance_termination)


def test_spaces_are_the_force_limits():
    env = make()
    assert env.observation_space.shape == (4,)
    assert env.observation_space.dtype == np.float64
    assert env.action_space.contains(np.array([-10.0]))
    assert env.action_space.contains(np.array([10.0]))
    assert not env.action_space.contains(np.array([10.001]))
    assert not env.action_space.contains(np.array([-10.001]))


def test_reset_is_five_degrees_at_rest():
    env = make()
    env.reset()
    env.step(np.array([3.0]))
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float64
    assert obs.tolist() == [0.0, 0.08726646259971647, 0.0, 0.0]
    assert env.data.time == 0.0


def test_one_tick_is_ten_physics_steps():
    env = make()
    env.reset()
    env.step(np.array([1.5]))
    assert env.data.time == pytest.approx(0.02)


def test_out_of_range_action_clips_and_the_reward_sees_the_clip():
    env = make()
    env.reset()
    obs, reward, _, _, info = env.step(np.array([99.0]))
    assert info["clipped"] is True
    assert reward == pytest.approx(5.0 - obs @ np.diag([1.0, 1.0, 10.0, 1.0]) @ obs - 100.0)
    _, _, _, _, info = env.step(np.array([10.0]))
    assert info["clipped"] is False


def test_reward_is_alive_bonus_minus_lqr_cost():
    obs = np.array([0.1, 0.05, 0.2, -0.3])
    assert lqr_reward(obs, np.array([1.5])) == pytest.approx(5.0 - (0.01 + 0.0025 + 0.4 + 0.09) - 2.25)


def test_termination_thresholds():
    assert balance_termination(np.array([0.0, 0.21, 0.0, 0.0]))
    assert balance_termination(np.array([0.0, -0.21, 0.0, 0.0]))
    assert balance_termination(np.array([1.51, 0.0, 0.0, 0.0]))
    assert not balance_termination(np.array([0.0, 0.19, 0.0, 0.0]))
    assert not balance_termination(np.array([1.49, 0.0, 0.0, 0.0]))


def test_default_duration_truncates_at_500_and_resets_clock():
    env = CartPoleEnv(lqr_reward, lambda obs: False)
    env.reset()
    for tick in range(1, 501):
        _, _, terminated, truncated, _ = env.step(np.zeros(1))
        assert not terminated
        assert truncated == (tick == 500)
    env.reset()
    assert env.data.time == 0.0
    _, _, _, truncated, _ = env.step(np.zeros(1))
    assert not truncated


def test_balance_episode_truncates_at_500_without_termination():
    """A hand-tuned state feedback holds the pole, so only the clock ends it."""
    env = make()
    obs, _ = env.reset()
    gain = np.array([1.0, 36.98, 3.92, 7.82])
    ends = []
    for tick in range(1, 501):
        obs, _, terminated, truncated, _ = env.step(np.array([gain @ obs]))
        if terminated or truncated:
            ends.append((tick, terminated, truncated))
    assert ends == [(500, False, True)]


def test_terminated_step_is_never_truncated():
    env = CartPoleEnv(lqr_reward, lambda obs: True, max_duration=0.02)
    env.reset()
    _, _, terminated, truncated, _ = env.step(np.zeros(1))
    assert (terminated, truncated) == (True, False)


def test_rejects_a_bad_action_or_duration():
    env = make()
    env.reset()
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(2))
    with pytest.raises(ValueError, match="nonfinite"):
        env.step(np.array([np.nan]))
    for value in (float("nan"), 0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="max_duration"):
            CartPoleEnv(lqr_reward, balance_termination, max_duration=value)


def test_gymnasium_env_checker_passes():
    import warnings

    expected = (
        ".*symmetric and normalized space",
        ".*observation space minimum value is -infinity",
        ".*observation space maximum value is infinity",
        ".*Not able to test alternative render modes",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for message in expected:
            warnings.filterwarnings("ignore", message, UserWarning)
        pytest.importorskip("gymnasium.utils.env_checker").check_env(make())
