"""Create a NEW DATASET VERSION with DVC + Git (the script from the lab slides, made safer).

    python src/data/update_dataDVC.py            # add + commit + tag (local only)
    python src/data/update_dataDVC.py --push     # ... and also git push / dvc push
    python src/data/update_dataDVC.py --dry-run  # only print the commands

The version number comes from `data_version` in params.yaml (increase it before each new version).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_params  # noqa: E402

DATA_FOLDER = "data/raw"          # the folder tracked by DVC


def run(cmd: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT)


def tag_exists(tag: str) -> bool:
    out = subprocess.run(["git", "tag", "--list", tag], capture_output=True, text=True, cwd=ROOT)
    return tag in out.stdout.split()


def nothing_staged() -> bool:
    return subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0


def update_dataDVC(data_ver, push: bool = False, dry_run: bool = False) -> None:
    tag = "v." + str(data_ver)
    if not dry_run and tag_exists(tag):
        sys.exit(f"The tag {tag} already exists. Increase `data_version` in params.yaml first.")
    run(["dvc", "add", DATA_FOLDER], dry_run)
    to_add = [f"{DATA_FOLDER}.dvc", "data/.gitignore", "params.yaml", "dvc.lock"]
    run(["git", "add"] + [p for p in to_add if dry_run or (ROOT / p).exists()], dry_run)
    if dry_run or not nothing_staged():
        run(["git", "commit", "-m", f"create new dataset version {tag}"], dry_run)
    else:
        print("(nothing new to commit - the tag will point to the current commit)")
    run(["git", "tag", "-a", tag, "-m", "Dataset version"], dry_run)
    if push:
        run(["git", "push"], dry_run)
        run(["git", "push", "--tags"], dry_run)
        run(["dvc", "push"], dry_run)


if __name__ == "__main__":
    params = load_params()
    update_dataDVC(params["data_version"], push="--push" in sys.argv, dry_run="--dry-run" in sys.argv)
