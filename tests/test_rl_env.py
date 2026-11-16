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


def test_v0_reward_goes_negative_and_the_v1_floor_keeps_it_positive():
    from robot_pg.rl.env import floored_lqr_reward

    tilted = np.array([0.0, 0.5, 0.0, 0.0])
    assert lqr_reward(tilted, np.array([10.0])) == pytest.approx(5.0 - 0.25 - 100.0)
    assert floored_lqr_reward(tilted, np.array([10.0])) == 0.5
    assert floored_lqr_reward(np.zeros(4), np.array([1.0])) == pytest.approx(4.0)
    rng = np.random.default_rng(0)
    states = rng.uniform(-1.0, 1.0, (5000, 4)) * np.array([3.0, np.pi / 2, 20.0, 50.0])
    forces = rng.uniform(-10.0, 10.0, (5000, 1))
    assert min(floored_lqr_reward(s, u) for s, u in zip(states, forces, strict=True)) == 0.5


def test_fall_termination_is_the_experiment_fail_rule():
    from robot_pg.rl.env import fall_termination

    assert not fall_termination(np.array([2.99, 1.57, 0.0, 0.0]))
    assert fall_termination(np.array([0.0, -1.571, 0.0, 0.0]))
    assert fall_termination(np.array([-3.01, 0.0, 0.0, 0.0]))


def test_push_matches_the_experiment_push():
    """The env push and lab.Push give the same states under the same actions."""
    from robot_pg import lab
    from robot_pg.rl.env import Randomization

    env = CartPoleEnv(lqr_reward, lambda obs: False, randomization=Randomization(push_max=0.25, push_interval=0.4))
    env.reset(seed=3)
    pushes = [p for p in env.pushes if p[0] < 1.5]
    assert len(pushes) >= 2
    actions = np.sin(np.arange(100) / 7.0)
    states = np.array([env.step(np.array([a]))[4]["state"] for a in actions])

    def policy(obs):
        return np.array([np.r_[actions, 0.0][round(obs["t"] / 0.02)]])

    run = lab.rollout(env.model, policy, 2.0 + 0.02, qpos0=np.array([0.0, 0.08726646259971647]), control_dt=0.02,
                      pushes=[lab.Push("pole", at, 0.1, (j / 0.1, 0.0, 0.0)) for at, j in pushes])
    rows = 10 * (np.arange(100) + 1)
    assert np.abs(run.qpos[rows] - states[:, :2]).max() < 1e-9
    assert np.abs(run.qvel[rows] - states[:, 2:]).max() < 1e-9
    assert np.abs(states[-1] - CartPoleEnv(lqr_reward, lambda obs: False).reset()[0]).max() > 0.01


def test_reset_clears_the_push_and_draws_a_new_schedule():
    from robot_pg.rl.env import Randomization

    env = CartPoleEnv(lqr_reward, lambda obs: False, randomization=Randomization(push_max=0.25, push_interval=0.05))
    env.reset(seed=1)
    first = list(env.pushes)
    env.step(np.zeros(1))
    env.step(np.zeros(1))
    assert env.data.xfrc_applied[env.pole, 0] != 0.0
    env.reset()
    assert env.data.xfrc_applied.tolist() == np.zeros_like(env.data.xfrc_applied).tolist()
    assert env.pushes != first
    assert all(abs(j) <= 0.25 for _, j in env.pushes)
    assert all(b[0] - a[0] >= 0.1 - 1e-9 for a, b in zip(env.pushes, env.pushes[1:], strict=False))


def test_v1_noise_reaches_only_the_observation():
    from robot_pg.rl.env import CONFIGS, floored_lqr_reward

    env = CONFIGS["v1"].env()
    env.reset(seed=0)
    diffs, rewards = [], []
    for _ in range(200):
        obs, reward, terminated, _, info = env.step(np.array([0.3]))
        diffs.append(obs - info["state"])
        rewards.append(reward - floored_lqr_reward(info["state"], np.array([0.3])))
        if terminated:
            env.reset()
    std = np.std(diffs, axis=0)
    assert std == pytest.approx([1e-3, 1e-3, 1e-2, 1e-2], rel=0.15)
    assert max(map(abs, rewards)) == 0.0


def test_v1_starts_cover_both_tilt_signs_within_the_spread():
    from robot_pg.rl.env import CONFIGS

    env = CONFIGS["v1"].env()
    starts = np.array([env.reset(seed=s)[0] for s in range(300)])
    assert starts[:, 1].min() < -0.4 and starts[:, 1].max() > 0.4
    assert np.abs(starts).max(axis=0) == pytest.approx([0.5, 0.5, 0.2, 0.2], abs=0.05)
    assert np.array_equal(env.reset(seed=7)[0], env.reset(seed=7)[0])
