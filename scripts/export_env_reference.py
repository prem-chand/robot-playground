# pyright: reportMissingImports=false
"""Record a cart-pole env trace in robot-playground-v0 for the robot-pg port check.

cd <robot-playground-v0 checkout> && uv run --extra rl python -I <this file> OUT.npz

The actions come from the trained policy plus a fixed offset schedule that
pushes some actions past the 10 N limit. The replay feeds the same actions,
so it does not depend on the policy. The script runs only in the
robot-playground-v0 checkout, so its robot_playground imports do not
resolve here.
"""

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from robot_playground.controllers.policy import PolicyNet
from robot_playground.envs import (
    cart_pole_balance_termination,
    cart_pole_lqr_reward,
    make_env,
)

STEPS = 1200
CKPT = Path("src/robot_playground/policies/cart_pole_ppo.pt")

out = Path(sys.argv[1])
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
net = PolicyNet(4, 1)
net.load_state_dict(ckpt["state_dict"])
net.eval()

env = make_env("cart_pole", cart_pole_lqr_reward, cart_pole_balance_termination)
obs, _ = env.reset(seed=0)
rows = {k: [] for k in ("obs", "action", "next_obs", "reward", "terminated", "truncated", "clipped")}
for k in range(STEPS):
    with torch.no_grad():
        mean, _ = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    offset = 15.0 * np.sin(k / 9.0) if (k // 150) % 2 else 0.0
    action = mean.squeeze(0).numpy().astype(np.float64) + offset
    next_obs, reward, terminated, truncated, info = env.step(action)
    for key, val in (("obs", obs), ("action", action), ("next_obs", next_obs), ("reward", reward),
                     ("terminated", terminated), ("truncated", truncated), ("clipped", info["clipped"])):
        rows[key].append(val)
    obs = env.reset()[0] if terminated or truncated else next_obs

commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
np.savez(out, **{k: np.asarray(v) for k, v in rows.items()}, source_commit=commit,
         mujoco_version=mujoco.__version__)
print(f"{out}: {STEPS} steps, {int(np.sum(rows['terminated']))} terminations, "
      f"{int(np.sum(rows['truncated']))} truncations, {int(np.sum(rows['clipped']))} clipped, "
      f"commit {commit[:7]}, mujoco {mujoco.__version__}")
