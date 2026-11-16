"""The cart-pole PPO train loop and its checkpoint, as in robot-playground-v0."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import gymnasium
import numpy as np
import torch

from robot_pg.rl.env import CONFIGS, CONTROL_DT, TrainConfig
from robot_pg.rl.ppo import LR, ROLLOUT_STEPS, TOTAL_STEPS, PolicyNet, ValueNet, gae, ppo_update

# gymnasium seeds the env with the same PCG64 stream as np.random.default_rng(seed).
# The cart-pole experiments draw seeds 0 to 4 that way, so the env stream starts here.
ENV_SEED_OFFSET = 1_000_000
CURVE_FIELDS = ("steps", "episodes", "mean episode length", "mean episode return", "log_std")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robot_pg.rl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--config", choices=list(CONFIGS), default="v0")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    train_cart_pole(args.output, seed=args.seed, total_steps=args.steps, config=CONFIGS[args.config])
    return 0


def _git(*args: str) -> str | None:
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=False, cwd=Path(__file__).parent)
    return done.stdout.strip() if done.returncode == 0 else None


def checkpoint_metadata(seed: int, total_steps: int, config: TrainConfig = CONFIGS["v0"]) -> dict:
    """Plain values only, so that torch.load(weights_only=True) reads them."""
    status = _git("status", "--porcelain", "--untracked-files=no")
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
        "config": config.describe(),
        "env_seed": ENV_SEED_OFFSET + seed,
        "trainer_commit": _git("rev-parse", "HEAD"),
        "trainer_tree_clean": None if status is None else status == "",
    }


def load_policy(path: Path) -> tuple[PolicyNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    net = PolicyNet(metadata["observation_size"], metadata["action_size"], tuple(metadata["hidden_sizes"]))
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    return net, metadata


def train_cart_pole(output: Path, seed: int, total_steps: int, config: TrainConfig = CONFIGS["v0"]) -> None:
    """Write the checkpoint to `output` and the learning curve, one row per update, next to it as CSV."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = config.env()
    policy = PolicyNet(4, 1)
    value_net = ValueNet(4)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=LR)
    obs, _ = env.reset(seed=ENV_SEED_OFFSET + seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    curve_file = output.with_suffix(".csv").open("w", newline="")
    curve = csv.writer(curve_file)
    curve.writerow(CURVE_FIELDS)
    steps_done = 0
    ep_len, ep_ret = 0, 0.0
    finished_lens: list[int] = []
    finished_rets: list[float] = []
    while steps_done < total_steps:
        batch_obs, batch_act, batch_logp = [], [], []
        batch_rew, batch_done, batch_trunc = [], [], []
        batch_val, batch_next_v = [], []
        finished_lens.clear()
        finished_rets.clear()
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
            ep_ret += reward
            if terminated or truncated:
                finished_lens.append(ep_len)
                finished_rets.append(ep_ret)
                ep_len, ep_ret = 0, 0.0
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
        mean_ret = float(np.mean(finished_rets)) if finished_rets else float("nan")
        curve.writerow((steps_done, len(finished_lens), f"{mean_len:.2f}", f"{mean_ret:.2f}",
                        f"{policy.log_std.item():.4f}"))
        curve_file.flush()
        print(
            f"steps {steps_done}/{total_steps} mean episode length "
            f"{mean_len:.1f} log_std {policy.log_std.item():.3f}",
            file=sys.stderr,
        )
    curve_file.close()
    metadata = checkpoint_metadata(seed, total_steps, config)
    torch.save({"state_dict": policy.state_dict(), "metadata": metadata}, output)
