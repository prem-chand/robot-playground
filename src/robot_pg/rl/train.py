"""The cart-pole PPO train loop and its checkpoint, as in robot-playground-v0."""

import argparse
import sys
from pathlib import Path

import gymnasium
import numpy as np
import torch

from robot_pg.rl.env import CONTROL_DT, CartPoleEnv, balance_termination, lqr_reward
from robot_pg.rl.ppo import (
    LR,
    ROLLOUT_STEPS,
    TOTAL_STEPS,
    PolicyNet,
    ValueNet,
    gae,
    ppo_update,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robot_pg.rl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    train_cart_pole(args.output, seed=args.seed, total_steps=args.steps)
    return 0


def checkpoint_metadata(seed: int, total_steps: int) -> dict:
    """Plain values only, so that torch.load(weights_only=True) reads them."""
    return {
        "robot": "cartpole_playground",
        "joint_names": ["cart", "pole"],
        "actuator_names": ["force"],
        "control_dt": CONTROL_DT,
        "observation_size": 4,
        "action_size": 1,
        "hidden_sizes": [64, 64],
        "activation": "tanh",
        "seed": seed,
        "total_steps": total_steps,
        "gymnasium_version": str(gymnasium.__version__),
        "torch_version": str(torch.__version__),
    }


def load_policy(path: Path) -> tuple[PolicyNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    net = PolicyNet(metadata["observation_size"], metadata["action_size"], tuple(metadata["hidden_sizes"]))
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    return net, metadata


def train_cart_pole(output: Path, seed: int, total_steps: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = CartPoleEnv(lqr_reward, balance_termination)
    policy = PolicyNet(4, 1)
    value_net = ValueNet(4)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=LR)
    obs, _ = env.reset(seed=seed)
    steps_done = 0
    ep_len = 0
    finished_lens: list[int] = []
    while steps_done < total_steps:
        batch_obs, batch_act, batch_logp = [], [], []
        batch_rew, batch_done, batch_trunc = [], [], []
        batch_val, batch_next_v = [], []
        finished_lens.clear()
        for _ in range(ROLLOUT_STEPS):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                mean, log_std = policy(obs_t.unsqueeze(0))
                dist = torch.distributions.Normal(mean, log_std.exp())
                raw = dist.sample()
                logp = dist.log_prob(raw).sum(-1)
                value = value_net(obs_t.unsqueeze(0))
            action = raw.squeeze(0).numpy().astype(np.float64)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            with torch.no_grad():
                nv = value_net(torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)).squeeze()
            batch_obs.append(obs)
            batch_act.append(raw.squeeze(0))
            batch_logp.append(logp.squeeze(0))
            batch_rew.append(reward)
            batch_done.append(terminated)
            batch_trunc.append(truncated)
            batch_val.append(value.squeeze())
            batch_next_v.append(nv)
            steps_done += 1
            ep_len += 1
            if terminated or truncated:
                finished_lens.append(ep_len)
                ep_len = 0
                next_obs, _ = env.reset()
            obs = next_obs
            if steps_done >= total_steps:
                break
        with torch.no_grad():
            adv, ret = gae(
                torch.tensor(batch_rew, dtype=torch.float32),
                torch.stack(batch_val),
                torch.tensor(batch_done, dtype=torch.bool),
                torch.tensor(batch_trunc, dtype=torch.bool),
                torch.stack(batch_next_v),
            )
        ppo_update(
            policy, value_net, optimizer,
            torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
            torch.stack(batch_act).detach(),
            torch.stack(batch_logp).detach(),
            adv.detach(), ret.detach(),
        )
        mean_len = float(np.mean(finished_lens)) if finished_lens else float(ep_len)
        print(
            f"steps {steps_done}/{total_steps} mean episode length "
            f"{mean_len:.1f} log_std {policy.log_std.item():.3f}",
            file=sys.stderr,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "metadata": checkpoint_metadata(seed, total_steps)}, output)
