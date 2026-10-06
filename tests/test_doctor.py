import platform

from robot_pg.cli import main


def test_doctor_prints_pinned_versions_then_ok(capsys):
    assert main(["doctor"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"python {platform.python_version()}"
    assert lines[1] == "mujoco 3.10.0"
    assert lines[2].startswith("numpy ")
    assert lines[3].startswith("cpu ") and len(lines[3]) > len("cpu ")
    assert lines[-1] == "ok"
