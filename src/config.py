"""Read params.yaml (with safe defaults) - shared by every script."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "data_version": 1,
    "data": {
        "raw_path": "data/raw/cresci_2017_users.csv", "processed_dir": "data/processed",
        "target": "is_fake", "positive_label": "1", "drop_columns": "auto",
        "min_rows": 500, "min_minority_pct": 5,
    },
    "preprocessing": {
        "standardize": True, "fix_negatives": True, "drop_dups": True, "max_cat": 30,
        "missing_flags": True, "ratios": True, "text": True, "dates": True,
        "test_size": 0.2, "seed": 42, "num_impute": "median", "cat_impute": "mode", "knn_k": 5,
        "out_method": "zscore", "out_action": "remove", "z_thr": 3.0, "iqr_k": 1.5, "max_remove_pct": 25,
        "log_skewed": True, "scaling": "standard", "encoding": "one-hot",
        "discretize_cols": [], "n_bins": 5, "agg": None,
    },
    "selection": {"use_selection": True, "method": "Embedded: LASSO", "k": 15},
    "model": {"name": "Random Forest", "seed": 42},
}


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_params(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return _merge(DEFAULTS, loaded)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
