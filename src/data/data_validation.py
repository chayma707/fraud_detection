"""Stage 1 - validate the raw data BEFORE anything else runs.

    python src/data/data_validation.py

Writes reports/validation.json and exits with an error (code 1) if a check fails,
so the DVC pipeline stops instead of training on bad data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_params, resolve  # noqa: E402
from src.data_loader import read_csv_flexible  # noqa: E402
from src import pipeline as pl  # noqa: E402


def validate(params: dict) -> dict:
    d = params["data"]
    raw_path = resolve(d["raw_path"])
    report: dict = {"data_version": params["data_version"], "raw_path": str(d["raw_path"]), "checks": {}}
    checks = report["checks"]

    checks["file_exists"] = raw_path.exists()
    if not checks["file_exists"]:
        report["checks"] = {k: bool(v) for k, v in checks.items()}
        report["passed"] = False
        return report

    df = read_csv_flexible(raw_path)
    report.update(rows=int(len(df)), columns=int(df.shape[1]),
                  duplicate_rows=int(df.duplicated().sum()),
                  missing_cells=int(df.isna().sum().sum()))
    checks["enough_rows"] = len(df) >= int(d["min_rows"])
    checks["target_column_exists"] = d["target"] in df.columns
    if checks["target_column_exists"]:
        counts = pl.target_value_counts(df[d["target"]])
        report["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        checks["two_or_more_classes"] = len(counts) >= 2
        if len(counts) >= 2:
            minority = 100 * counts.min() / counts.sum()
            report["minority_class_pct"] = round(float(minority), 2)
            checks["minority_class_big_enough"] = minority >= float(d["min_minority_pct"])
        positive = str(d["positive_label"])
        checks["positive_label_present"] = positive == "auto" or positive in counts.index
    report["checks"] = {k: bool(v) for k, v in checks.items()}
    report["passed"] = all(report["checks"].values())
    return report


def main() -> None:
    params = load_params()
    report = validate(params)
    out = ROOT / "reports" / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        failed = [k for k, v in report["checks"].items() if not v]
        sys.exit(f"DATA VALIDATION FAILED: {failed}")
    print("Data validation passed.")


if __name__ == "__main__":
    main()
