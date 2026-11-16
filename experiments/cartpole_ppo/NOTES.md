# Cart-pole PPO against LQR: push, tilt and mass limits

Run: pending.

## Setup

- Model: `robots/cartpole_playground/cartpole_playground.xml`, a byte-for-byte copy of robot-playground's `cart_pole.xml`. Cart 1.0 kg with slide damping 0.5. Pole 0.1 kg, a 0.6 m capsule with its center of mass 0.3 m above the pivot, hinge damping 0.01, armature 0.001. Cart force limit ±10 N. dt 2 ms. This is not robot-pg's own `cartpole` plant.
- PPO: the user's own policy. The user trained it in robot-playground with seed 0 for 200k steps. The checkpoint is `cart_pole_ppo.pt` at commit c67f5d8 (it entered the repo in 2650c98, sha256 `2dc28c24…f4b0`). The net is a 4-64-64-1 tanh MLP. The controller uses the mean action. `export_policy.py` writes the float32 weights to `policy.npz`, and `run.py` runs them in NumPy.
- PPO training setup (from robot-playground): reward 5 minus the LQR cost with Q = diag(1, 1, 10, 1) and R = 1. An episode ends at |tilt| > 0.2 rad, at |x| > 1.5 m, or at 10 s. Every episode starts from the same 5° tilt. The final action std is 0.29 N (log_std -1.242).
- LQR: a continuous CARE on the `lab.linearize` model of the same plant, with Q = diag(1, 1, 10, 1) and R = 1. These are the weights of robot-playground's LQR.
- Both controllers run at 50 Hz (control_dt 0.02 s) and hold the action between ticks, as in robot-playground.
- Failure, "recovered", seeds and noise are the same as in `cartpole_lqr`. Failure: the pole passes 90°, or the cart passes ±3 m. "Recovered" also needs |tilt| < 0.05 rad at the end. Each of 5 seeds draws a cart offset of ±0.05 m, a tilt offset of ±0.02 rad, the push sign, a push start from 1.0 to 1.2 s, and sensor noise (σ 1e-3 on qpos, 1e-2 on qvel).

## Question

1. How large a push on the pole, and how large a release tilt, can the user's PPO policy recover from? How does that compare with an LQR on the same plant?
2. With the policy fixed, what range of pole-mass error keeps the pole up for 15 s?
3. Does the policy stay up on robot-pg's own `cartpole` plant, which it never saw?

## Prediction (recorded 2026-11-16, before any sweep)

No user prediction was recorded. The prediction below is Claude's.

Before this prediction, Claude saw these facts and no sweep result:

- The checkpoint metadata and the PPO training setup above.
- The LQR gains. robot-playground's gain has a tilt gain of -36.98 N/rad. The `lab.linearize` gain has -38.54 N/rad.
- One LQR run from the 5° start on the vendored plant. It stayed up and its peak force was 3.2 N.
- robot-playground's own 10 s PPO run from the 5° start (the reference trajectory in `policy.npz`). It stayed up, its peak force was 2.5 N, and the cart ended at 0.21 m.
- The static capture bound by hand: atan(10 / (1.1 × 9.81)) = 0.747 rad.

Claude's prediction:

| Quantity | PPO | LQR |
|---|---|---|
| Release-from-rest tilt limit | 0.30 rad | 0.55 rad (about 0.75 of the static bound) |
| Push limit J* (0.1 s push on the pole) | 0.07 N s | 0.12 N s |
| Pole-mass range where all seeds stay up | 0.3× to 4× | 0.05× to 10× |

- The reason: the PPO policy never saw a tilt above 0.2 rad or a pole mass other than 0.1 kg. Outside that range the net extrapolates, and it has no reason to do well. The LQR knows the linear model and saturates only at the force limit.
- Transfer: the PPO policy stays up on robot-pg's `cartpole` plant from a 0.05 rad release for 5 of 5 seeds. Its raw output stays below 10 N there.
- Replay: robot-pg's NumPy port on MuJoCo 3.10.0 tracks robot-playground's torch run on MuJoCo 3.8.0 to better than 1e-6 in each state.
