"""The cart-pole PPO trainer. It needs the rl extra (torch and gymnasium).

uv run --extra rl python -m robot_pg.rl --output cart_pole_ppo.pt --seed 0

Ported from robot-playground-v0 at c67f5d8 (rl/ppo.py, rl/train.py,
envs/env.py, envs/cart_pole.py, and PolicyNet in controllers/policy.py).
The port keeps the PPO constants, the GAE and update code, and the env
tick of that source. `import robot_pg` does not import this package.
"""
