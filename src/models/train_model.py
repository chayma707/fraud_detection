"""Stage 3 - select features, train the model, save the model and its metrics.

    python src/models/train_model.py

Reads data/processed/*.csv + params.yaml -> writes models/model.joblib and reports/metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src import model as md  # noqa: E402
from src import selection as sel  # noqa: E402
from src.config import load_params, resolve  # noqa: E402


def main() -> None:
    params = load_params()
    proc = resolve(params["data"]["processed_dir"])
    train = pd.read_csv(proc / "train.csv")
    test = pd.read_csv(proc / "test.csv")
    X_tr, y_tr = train.drop(columns="target"), train["target"]
    X_te, y_te = test.drop(columns="target"), test["target"]

    s = params["selection"]
    seed = int(params["model"]["seed"])
    features = list(X_tr.columns)
    if s["use_selection"]:
        scores = sel.compute_feature_scores(X_tr, y_tr, int(s["k"]), seed)
        features = sel.top_k(scores, s["method"], int(s["k"]))

    res = md.evaluate(params["model"]["name"], X_tr[features], y_tr, X_te[features], y_te, seed)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": res["model"], "features": features}, ROOT / "models" / "model.joblib")
    metrics = {**{k: round(v, 4) for k, v in res["metrics"].items()},
               "n_features": len(features), "data_version": params["data_version"]}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
