"""Export the author's cart-pole PPO checkpoint to policy.npz for the NumPy port.

Run it inside the robot-playground-v0 checkout, with its rl extra:

cd <robot-playground-v0> && uv run --extra rl python <this repo>/experiments/cartpole_ppo/export_policy.py

The weights keep their stored float32 values. The reference trajectory comes
from robot-playground-v0's own run() from the declared 5 degree start.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import torch
from robot_playground.contracts import ControllerContext
from robot_playground.controllers.lqr import LQRController
from robot_playground.controllers.policy import PolicyController, PolicyNet
from robot_playground.registry import ROBOTS
from robot_playground.simulation import RunConfig, run

PT = Path("src/robot_playground/policies/cart_pole_ppo.pt")
OUT = Path(__file__).parent / "policy.npz"
CHECK_N = 10_000
CHECK_BOX = np.array([3.0, np.pi / 2, 5.0, 10.0], dtype=np.float32)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def numpy_mean(w: dict[str, np.ndarray], obs: np.ndarray) -> np.ndarray:
    h = np.tanh(obs @ w["W0"].T + w["b0"])
    h = np.tanh(h @ w["W1"].T + w["b1"])
    return h @ w["W2"].T + w["b2"]


def playground_gain() -> np.ndarray:
    spec = ROBOTS["cart_pole"]
    model = mujoco.MjModel.from_xml_path(str(spec.xml_path))
    lqr = LQRController()
    lqr.reset(ControllerContext(
        joint_names=spec.joint_names, actuator_names=spec.actuator_names,
        control_dt=spec.control_dt, ctrl_limits=np.array([[-10.0, 10.0]]),
        model=model, robot=spec.name,
    ))
    assert lqr._gain is not None
    return lqr._gain


def main() -> None:
    ckpt = torch.load(PT, map_location="cpu", weights_only=True)
    sd, meta = ckpt["state_dict"], ckpt["metadata"]
    w = {
        f"{kind}{i}": sd[f"mean_head.{2 * i}.{name}"].numpy()
        for i in range(3)
        for kind, name in (("W", "weight"), ("b", "bias"))
    }
    net = PolicyNet(4, 1, tuple(meta["hidden_sizes"]))
    net.load_state_dict(sd, strict=True)
    net.eval()

    rng = np.random.default_rng(0)
    check_obs = (rng.uniform(-1.0, 1.0, (CHECK_N, 4)) * CHECK_BOX).astype(np.float32)
    with torch.no_grad():
        check_torch = net(torch.from_numpy(check_obs))[0].numpy()
    max_diff = float(np.abs(numpy_mean(w, check_obs) - check_torch).max())

    ref = run(RunConfig(robot="cart_pole", controller=PolicyController(PT), duration=10.0))

    np.savez(
        OUT,
        **w,
        control_dt=np.float64(meta["control_dt"]),
        source_path=np.str_(str(PT)),
        source_commit=np.str_(git("rev-parse", "HEAD")),
        source_tree=np.str_(git("rev-parse", "HEAD^{tree}")),
        pt_commit=np.str_(git("log", "-1", "--format=%H", "--", str(PT))),
        pt_sha256=np.str_(hashlib.sha256(PT.read_bytes()).hexdigest()),
        torch_version=np.str_(torch.__version__),
        mujoco_version=np.str_(mujoco.__version__),
        metadata=np.str_(json.dumps(meta, sort_keys=True)),
        K_playground=playground_gain(),
        ref_t=ref.time,
        ref_qpos=ref.qpos,
        ref_qvel=ref.qvel,
        ref_ctrl=ref.ctrl,
        ref_qpos0=np.asarray(ROBOTS["cart_pole"].initial_qpos),
        check_obs=check_obs,
        check_torch=check_torch,
        max_abs_numpy_minus_torch=np.float64(max_diff),
    )
    print(f"wrote {OUT}, max |numpy - torch| {max_diff:.3e}")


if __name__ == "__main__":
    main()
