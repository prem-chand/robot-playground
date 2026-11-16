import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
SHA = "4d038b3feae26ec82b46a4d586379114012a8ac7"
FOLDERS = ("unitree_g1", "franka_emika_panda")
DEST = Path(__file__).resolve().parent.parent / "src" / "robot_pg" / "robots"


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        git(repo, "init", "-q")
        git(repo, "remote", "add", "origin", URL)
        git(repo, "sparse-checkout", "set", "--no-cone", *(f"/{f}/" for f in FOLDERS))
        git(repo, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", SHA)
        git(repo, "checkout", "-q", "FETCH_HEAD")
        for folder in FOLDERS:
            shutil.rmtree(DEST / folder, ignore_errors=True)
            shutil.copytree(repo / folder, DEST / folder)
            (DEST / folder / "VENDORED.txt").write_text(f"{URL}\n{SHA}\n{folder}/\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
