# robot-pg

robot-pg is a platform for robot experiments in simulation. Each controller is a block with a declared rate, latency, input signals, output signals, and compute budget. Each run gets a content hash, so you can replay it bitwise.

The concept is in [ROADMAP.md](ROADMAP.md). The original TextEdit file is in [docs/ROADMAP.rtf](docs/ROADMAP.rtf).

## Set up

Install [uv](https://docs.astral.sh/uv/). Then run these commands:

```sh
uv sync --extra dev --extra viz
uv run robot-pg doctor
```

`robot-pg doctor` prints the Python, MuJoCo, and NumPy versions and the CPU. The last line is `ok`.

## Test

```sh
uv run --extra dev pytest -q
uv run --extra dev ruff check
uv run --extra dev pyright
```

## Plan evidence

[docs/plan-evidence/](docs/plan-evidence/) holds the survey reports and the prototype scripts and models that the Phase 1 plan cites.
