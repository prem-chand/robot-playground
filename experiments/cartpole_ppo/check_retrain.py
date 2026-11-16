"""Compare a retrained cart-pole PPO checkpoint with the robot-playground-v0 weights.

uv run --extra rl python experiments/cartpole_ppo/check_retrain.py RETRAINED.pt [V0.pt]

The v0 mean-head weights come from policy.npz next to this script. A
v0 checkpoint, if given, adds log_std. The gate is robot-playground-v0's
test_policy_vs_lqr: 10 s from the 5 degree start, |tilt| <= 0.2 rad,
|x| <= 1.5 m, and a pole RMS at most twice the RMS of v0's LQR.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from robot_pg.rl.env import CartPoleEnv, lqr_reward
from robot_pg.rl.train import load_policy

V0_NPZ = Path(__file__).parent / "policy.npz"
KEYS = {f"{kind}{i}": f"mean_head.{2 * i}.{name}" for i in range(3) for kind, name in (("W", "weight"), ("b", "bias"))}


def episode(act) -> dict:
    """500 control ticks with no termination. Rows are post-tick states, as in v0's run()."""
    env = CartPoleEnv(lqr_reward, lambda obs: False)
    obs, _ = env.reset()
    rows, clipped = [], 0
    for _ in range(500):
        obs, _, _, _, info = env.step(act(obs))
        rows.append(obs)
        clipped += info["clipped"]
    states = np.array(rows)
    return {"max |tilt| (rad)": float(np.abs(states[:, 1]).max()), "max |x| (m)": float(np.abs(states[:, 0]).max()),
            "pole RMS (rad)": float(np.sqrt(np.mean(states[:, 1] ** 2))), "clip fraction": clipped / 500,
            "finite": bool(np.isfinite(states).all())}


def main() -> None:
    net, metadata = load_policy(Path(sys.argv[1]))
    sd = {k: v.numpy() for k, v in net.state_dict().items()}
    v0 = np.load(V0_NPZ)
    diffs = {k: float(np.abs(sd[name] - v0[k]).max()) for k, name in KEYS.items()}
    report: dict = {"metadata": metadata, "max |w - w_v0| per tensor": diffs,
                    "bitwise equal to v0 mean head": all(np.array_equal(sd[n], v0[k]) for k, n in KEYS.items())}
    if len(sys.argv) > 2:
        v0_sd = torch.load(sys.argv[2], map_location="cpu", weights_only=True)["state_dict"]
        report["bitwise equal to the v0 checkpoint"] = all(torch.equal(v, v0_sd[k]) for k, v in net.state_dict().items())
        report["log_std, v0 log_std"] = [float(net.log_std), float(v0_sd["log_std"])]

    def ppo(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy().astype(np.float64)

    k_v0 = v0["K_playground"]
    runs = {"ppo": episode(ppo), "lqr (v0 gain)": episode(lambda obs: -k_v0 @ obs)}
    ppo_run, lqr_run = runs["ppo"], runs["lqr (v0 gain)"]
    report["episodes"] = runs
    report["v0 gate passes"] = bool(
        ppo_run["finite"] and ppo_run["max |tilt| (rad)"] <= 0.2 and ppo_run["max |x| (m)"] <= 1.5
        and ppo_run["pole RMS (rad)"] <= 2 * lqr_run["pole RMS (rad)"] and ppo_run["clip fraction"] < 1.0
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
