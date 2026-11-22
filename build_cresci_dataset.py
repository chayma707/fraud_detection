"""Merge the Cresci-2017 (or any similar "one folder per account type") download into ONE labelled CSV.

Usage
-----
    python build_cresci_dataset.py PATH_TO_DOWNLOADED_FOLDER

What it does
------------
* looks for every file named ``users.csv`` (Cresci-2017 layout: one folder per account type such as
  ``genuine_accounts``, ``fake_followers``, ``social_spambots_1`` ...). If there is no ``users.csv``
  it uses every ``.csv`` file it finds.
* label ``is_fake`` = 0 when the folder / file name contains "genuine", "human" or "real", otherwise 1
* the account type is stored in the column ``account_group`` (the app drops it by default because
  it would reveal the label)
* writes ``data/raw/cresci_2017_users.csv``  (the app picks it up automatically)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data_loader import read_csv_flexible

GENUINE_WORDS = ("genuine", "human", "real")


def find_csv_files(root: Path) -> list[Path]:
    users = sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower() == "users.csv")
    if users:
        return users
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1]).expanduser()
    if not root.exists():
        sys.exit(f"Folder not found: {root}")

    frames = []
    for path in find_csv_files(root):
        group_name = path.parent.name if path.name.lower() == "users.csv" else path.stem
        label_text = f"{path.parent.name} {path.stem}".lower()
        is_genuine = any(w in label_text for w in GENUINE_WORDS)
        try:
            df = read_csv_flexible(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skipped {path} ({exc})")
            continue
        df["account_group"] = group_name
        df["is_fake"] = 0 if is_genuine else 1
        frames.append(df)
        print(f"  + {path.relative_to(root)}: {len(df):,} rows -> is_fake={0 if is_genuine else 1}")

    if not frames:
        sys.exit("No CSV files found.")
    out = pd.concat(frames, ignore_index=True, sort=False)
    dest = Path(__file__).resolve().parent / "data" / "raw" / "cresci_2017_users.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\nSaved {len(out):,} rows x {out.shape[1]} columns to {dest}")
    print(out["is_fake"].value_counts().rename({0: "genuine (0)", 1: "fake / bot (1)"}).to_string())


if __name__ == "__main__":
    main()
