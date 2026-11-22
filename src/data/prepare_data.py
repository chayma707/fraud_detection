"""Stage 2 - clean + transform the raw data and save the processed data.

    python src/data/prepare_data.py

Reads params.yaml  ->  writes data/processed/{train,test,cleaned}.csv and data/processed/info.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import data_loader as dl  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src.config import load_params, resolve  # noqa: E402

A_KEYS = ["standardize", "fix_negatives", "drop_dups", "max_cat", "missing_flags", "ratios", "text", "dates"]


def main() -> None:
    params = load_params()
    d, prep = params["data"], params["preprocessing"]
    raw = dl.read_csv_flexible(resolve(d["raw_path"]))
    target = d["target"]

    positive = str(d["positive_label"])
    if positive == "auto":
        positive = pl.guess_positive(pl.target_value_counts(raw[target]))
    drop_cols = d["drop_columns"]
    if drop_cols == "auto":
        drop_cols = dl.suggest_drop_columns(raw, target)

    cfg_a = {k: prep[k] for k in A_KEYS}
    A = pl.prepare_features(raw, target, positive, list(drop_cols), cfg_a)
    B = pl.preprocess(A.X, A.y, dict(prep))

    out_dir = resolve(d["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    B.X_train.assign(target=B.y_train).to_csv(out_dir / "train.csv", index=False)
    B.X_test.assign(target=B.y_test).to_csv(out_dir / "test.csv", index=False)
    pd.concat([B.train_clean.assign(target=B.y_train_clean), B.test_clean.assign(target=B.y_test)]) \
        .sort_index().to_csv(out_dir / "cleaned.csv")
    info = {
        "data_version": params["data_version"], "raw_rows": int(len(raw)),
        "train_rows": int(len(B.X_train)), "test_rows": int(len(B.X_test)),
        "n_features": int(B.X_train.shape[1]), "dropped_columns": list(drop_cols),
        "duplicates_removed": A.log.get("duplicates_removed", 0),
        "outlier_rows_removed": B.outlier_info["removed"], "steps": B.steps,
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
