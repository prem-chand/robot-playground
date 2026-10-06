import subprocess
import sys


def test_import_does_not_load_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import robot_pg, sys; assert "torch" not in sys.modules',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
