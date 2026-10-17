# Lab reset: audit of Phase 1

Date: 2026-10-17. Branch: p1-05 (commit 1225026). Nothing was deleted.

## Verdict per module

"Experiment" means the module makes a sweep faster to write. "Rigor" means it serves only replay, perf, or contract checks.

| Module | Lines | Serves | Proposal |
|---|---|---|---|
| `robots/` assets (panda, diffdrive MJCF) | - | Experiment | Keep. |
| `robots/__init__.py` `RobotSpec`, `_compiled`, `Site` | ~60 of 378 | Experiment | Keep. Load by name, add the `ee` site with `MjSpec`. |
| `robots/__init__.py` `Episode`, `Sample`, `_draw`, `FailRule`, trunk contact | ~300 of 378 | Rigor | Freeze. Each experiment writes its own 3-line fail check. |
| `plant.py` `parse_model`, `MujocoPlant.sense`/`act` | 325 | Experiment, partly | Simplify. Reuse the signal dict (`ee_pos`, `ee_jacobian`, base state) and the `RunUnstable` check. The obs lacks unactuated joints, so a cart-pole pole angle is missing. The lab adds full `qpos`/`qvel`. |
| `plant.py` `check_command`, `UNITS` | ~30 | Rigor | Freeze. |
| `controllers/` `JointPd`, `Reach`, `PathPursuit` | ~150 of 244 | Experiment | Keep. `step(inputs) -> outputs` is already obs to action. Call it in a plain loop. |
| `controllers/` `Driver`, `drive`, `Setpoint` | ~60 | Rigor | Freeze. |
| `scenarios/` `Targets`, `Path`, `Velocity`, `Push` | ~200 of 356 | Experiment | Keep the reference shapes. `Push` is never applied: no code writes `xfrc_applied`. |
| `scenarios/` registry, `to_dict`/`from_dict`, `SeedSpace` | ~150 | Rigor | Freeze. |
| `schedule.py` | 224 | Rigor, with one useful feature | Freeze. Its one lab feature is latency. A 5-line delay ring in the lab replaces it. |
| `contract.py` | 75 | Rigor | Freeze. `SignalSpec` stays because `plant.py` uses it. |
| `runtime.py` `Session`, `run` | 103 | Rigor | Freeze. |
| `trajectory.py` `Recorder`, NPZ | 170 | Rigor | Freeze. The lab returns plain arrays. |
| `provenance.py` content hash | 62 | Rigor | Freeze. |
| `rendering.py` `frames`, `write_png` | 83 | Experiment | Keep. `frames` feeds the video. |
| `cli.py` `inspect`, `render` | 180 | Experiment | Keep. |
| `scripts/bench_robots.py`, `examples/latency_sweep.py` | 258 | Rigor | Freeze. |

About 65% of `src/` is rigor. The parts that speed up an experiment are the assets, model loading, the signal dict, three controllers, the reference shapes, and rendering.

"Freeze" means: no new features, no new callers from the lab, tests stay green. A later PR can delete what the lab never imports.

## Proposed minimum lab

One file, `src/robot_pg/lab.py`, about 120 lines:

```python
def load(name_or_path: str, edit: Callable[[MjSpec], None] | None = None) -> MjModel
    # MJCF + `home` keyframe. Reuses RobotSpec for registry names.
def rollout(model, policy, seconds, *, latency_s=0.0, push=None, seed=0,
            obs_noise=0.0, fail=None) -> Run
    # policy(obs: dict) -> action. obs = sense() signals + qpos + qvel + t.
    # A delay ring gives latency. `push` writes xfrc_applied on the base body.
    # Run holds t, qpos, qvel, ctrl, failed_at, as plain numpy arrays.
def sweep(run_one, grid: dict[str, list], seeds: range) -> xarray-like ndarray
def video(model, qpos, path, fps=30)  # rendering.frames + imageio
def limit_plot(grid, success, path)    # mean and spread over seeds
```

An experiment is `experiments/<name>/run.py` plus `NOTES.md`, a plot, and an MP4.

## Blockers found

1. **No GPU.** The host is an Intel i7-9750H (6 cores, 16 GB, x86_64 macOS). JAX stops at 0.4.38 here. `playground` resolves only to 0.0.5. A legged-robot PPO run takes minutes on a GPU and probably 10 or more hours on this CPU, if it runs at all.
2. **`python-control` is not in the project venv.** The LQR check needs it as a dev extra.
3. **Panda actuators are position servos.** "Torque limit" means scaling `forcerange`. That fits the existing DLS `Reach`, so DLS is the first choice over OSC.
4. **Pushes need lab code.** `RobotSpec.episode` refuses a push on a fixed base, and a cart-pole has no free joint. The lab applies its own pushes.
