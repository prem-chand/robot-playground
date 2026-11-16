"""Training-side checks for the v1 PPO config. No experiment sweep runs here.

uv run --extra rl python experiments/cartpole_ppo/train_check.py [CHECKPOINT.pt]

1. The v0 reward against termination: the discounted v0-reward return of
   the LQR recovery from a 0.5 rad release, with no termination. A
   terminated episode returns 0 after its last step.
2. The yardstick: the mean episode length of the LQR in the v1 env, on
   episodes seeded 2_000_000 + i. Training (env seeds from 1_000_000) and
   the experiment (seeds 0 to 4) do not use these seeds.
3. With a checkpoint: the mean episode length of its mean action on the same
   episodes. v1 trains well if this is at least 0.9 of the LQR value.

The LQR gain is the experiment's K, read from result.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

from robot_pg.rl.env import CONFIGS, CartPoleEnv, lqr_reward
from robot_pg.rl.ppo import GAMMA

OUT = Path(__file__).parent
K = np.array(json.loads((OUT / "result.json").read_text())["K (lab.linearize, with damping)"])
EPISODES = 200
YARDSTICK_SEED = 2_000_000
PASS_FRACTION = 0.9


def lqr(obs: np.ndarray) -> np.ndarray:
    return np.array([-K @ obs])


def v0_reward_of_lqr_recovery(tilt: float) -> dict:
    env = CartPoleEnv(lqr_reward, lambda obs: False)
    env.reset()
    env.data.qpos[:] = (0.0, tilt)
    obs = np.array([0.0, tilt, 0.0, 0.0])
    rewards = []
    for _ in range(500):
        obs, reward, _, _, _ = env.step(lqr(obs))
        rewards.append(reward)
    r = np.array(rewards)
    return {"release tilt (rad)": tilt, "discounted return, gamma 0.99": float(r @ GAMMA ** np.arange(r.size)),
            "min step reward": float(r.min()), "steps with reward < 0": int((r < 0).sum()),
            "final |tilt| (rad)": float(np.abs(obs[1])), "return of terminating at once": 0.0}


def episode_lengths(act) -> np.ndarray:
    env = CONFIGS["v1"].env()
    lengths = []
    for i in range(EPISODES):
        obs, _ = env.reset(seed=YARDSTICK_SEED + i)
        for tick in range(1, 501):
            obs, _, terminated, truncated, _ = env.step(act(obs))
            if terminated or truncated:
                break
        lengths.append(tick)
    return np.array(lengths)


def summary(lengths: np.ndarray) -> dict:
    return {"mean episode length (ticks)": float(lengths.mean()), "episodes that reach 500": int((lengths == 500).sum()),
            "episodes": int(lengths.size)}


def main() -> None:
    report = {"v0 reward, LQR recovery": [v0_reward_of_lqr_recovery(t) for t in (0.1, 0.3, 0.5)]}
    lqr_lengths = episode_lengths(lqr)
    report["LQR in the v1 env"] = summary(lqr_lengths)
    if len(sys.argv) > 1:
        import torch

        from robot_pg.rl.train import load_policy

        net, metadata = load_policy(Path(sys.argv[1]))

        def ppo(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                mean, _ = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            return mean.squeeze(0).numpy().astype(np.float64)

        ppo_lengths = episode_lengths(ppo)
        ratio = float(ppo_lengths.mean() / lqr_lengths.mean())
        report["checkpoint"] = {"path": sys.argv[1], "config": metadata["config"]["name"], **summary(ppo_lengths),
                                "mean length / LQR mean length": ratio, "pass (>= 0.9)": ratio >= PASS_FRACTION}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
