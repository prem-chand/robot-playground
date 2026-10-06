import argparse
import platform
import subprocess
import sys


def _cpu() -> str:
    if sys.platform == "darwin":
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    return platform.processor() or platform.machine()


def doctor() -> int:
    import mujoco
    import numpy

    print(f"python {platform.python_version()}")
    print(f"mujoco {mujoco.__version__}")
    print(f"numpy {numpy.__version__}")
    print(f"cpu {_cpu()}")
    print("ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robot-pg")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="print the Python, MuJoCo, NumPy, and CPU in use, then ok"
    ).set_defaults(run=doctor)
    return parser.parse_args(argv).run()
