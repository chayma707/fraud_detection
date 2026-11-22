"""End-to-end preprocessing pipeline.

Stage A (row-wise, nothing fitted):  target -> inconsistencies -> impossible values -> duplicates -> feature engineering
Stage B (fitted on TRAIN only):      split -> impute -> outliers -> log -> discretize -> aggregate -> encode -> scale
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import cleaning as cl
from . import features as ft
from . import transform as tf

TARGET = "target_label"
POSITIVE_HINTS = ("fake", "bot", "fraud", "spam", "scam", "malicious", "suspicious", "1", "true", "yes")


# --------------------------------------------------------------------------
# Target helpers
# --------------------------------------------------------------------------
def _norm_value(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else str(v)
    return str(v).strip().lower()


def target_value_counts(series: pd.Series) -> pd.Series:
    return series.map(_norm_value).value_counts()


def guess_positive(counts: pd.Series) -> str:
    values = list(counts.index)
    for hint in POSITIVE_HINTS:
        for v in values:
            if hint == v or (len(hint) > 1 and hint in v):
                return v
    return values[-1]  # least frequent class


def binarize_target(series: pd.Series, positive: str) -> pd.Series:
    normed = series.map(_norm_value)
    y = (normed == str(positive)).astype(float)
    return y.where(series.notna())


# --------------------------------------------------------------------------
# Stage A
# --------------------------------------------------------------------------
@dataclass
class StageA:
    X: pd.DataFrame
    y: pd.Series
    struct: pd.DataFrame            # features after structural cleaning (before FE) + TARGET
    log: dict = field(default_factory=dict)
    fe_log: dict = field(default_factory=dict)


def prepare_features(raw: pd.DataFrame, target: str, positive: str, drop_cols: list, cfg: dict) -> StageA:
    df = raw.copy()
    y = binarize_target(df[target], positive)
    drops = [c for c in drop_cols if c in df.columns and c != target]
    X = df.drop(columns=[target] + drops)
    keep = y.notna()
    X, y = X.loc[keep].copy(), y.loc[keep]
    log: dict = {"rows_raw": int(len(raw)), "rows_without_target": int((~keep).sum()),
                 "columns_dropped": drops}

    if cfg.get("standardize", True):
        X, tlog = cl.standardize_text(X)
        log.update(tlog)
    if cfg.get("fix_negatives", True):
        X, neg = cl.fix_impossible_values(X)
        log["negatives_fixed"] = neg

    work = X.copy()
    work[TARGET] = y.to_numpy()
    n_dups = 0
    if cfg.get("drop_dups", True):
        work, n_dups = cl.remove_duplicates(work)
    log["duplicates_removed"] = int(n_dups)
    X = work.drop(columns=TARGET)
    y = work[TARGET].astype(int)

    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(float)

    struct = X.copy()
    struct[TARGET] = y
    X_fe, fe_log = ft.engineer_features(
        X, max_cat=cfg.get("max_cat", 30), ratios=cfg.get("ratios", True), text=cfg.get("text", True),
        dates=cfg.get("dates", True), missing_flags=cfg.get("missing_flags", True),
    )
    log["rows_after_structural"] = int(len(X_fe))
    return StageA(X=X_fe, y=y, struct=struct, log=log, fe_log=fe_log)


# --------------------------------------------------------------------------
# Stage B
# --------------------------------------------------------------------------
@dataclass
class StageB:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_clean: pd.DataFrame       # imputed + outliers handled, ORIGINAL scale, readable
    test_clean: pd.DataFrame
    y_train_clean: pd.Series
    outlier_info: dict
    outlier_counts: pd.Series
    steps: list
    num_cols: list
    cat_cols: list
    log_cols: list
    bin_cols: list
    agg_col: str | None
    encoded_cols: list
    scaled_cols: list
    dropped_constant: list
    split_sizes: dict


def preprocess(X: pd.DataFrame, y: pd.Series, cfg: dict) -> StageB:
    steps: list[str] = []
    if y.nunique() < 2:
        raise ValueError("After cleaning only one class is left in the target. Check the target / positive class.")
    if y.value_counts().min() < 2:
        raise ValueError("One class has fewer than 2 rows, a stratified split is impossible.")

    # ---- split first (everything below is fitted on TRAIN only)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["seed"]
    )
    steps.append(f"Stratified split {int((1 - cfg['test_size']) * 100)}% / {int(cfg['test_size'] * 100)}% (seed {cfg['seed']})")
    split_sizes = {"train": len(X_tr), "test": len(X_te)}

    kinds = ft.classify_columns(X_tr, cfg["max_cat"])
    num_cols = [c for c in kinds["numeric"] if X_tr[c].notna().any()]
    cat_cols = [c for c in kinds["categorical"] if X_tr[c].notna().any()]
    keep = num_cols + cat_cols
    X_tr, X_te = X_tr[keep].copy(), X_te[keep].copy()
    if not keep:
        raise ValueError("No usable feature columns are left. Un-drop some columns in the sidebar.")

    # ---- missing values
    X_tr, X_te = cl.impute_train_test(
        X_tr, X_te, num_cols, cat_cols, cfg["num_impute"], cfg["cat_impute"], cfg["knn_k"]
    )
    steps.append(f"Missing values: numeric = {cfg['num_impute']}, categorical = {cfg['cat_impute']}")

    # ---- outliers
    outlier_info = {"method": cfg["out_method"], "action": cfg["out_action"], "removed": 0,
                    "skipped": False, "reason": ""}
    outlier_counts = pd.Series(dtype=int)
    if cfg["out_method"] != "none":
        handler = cl.OutlierHandler(cfg["out_method"], cfg["z_thr"], cfg["iqr_k"]).fit(X_tr, num_cols)
        outlier_counts = handler.counts(X_tr)
        if cfg["out_action"] == "remove":
            mask = handler.outlier_mask(X_tr)
            frac = float(mask.mean())
            remaining_classes = y_tr.loc[X_tr.index[~mask]].nunique()
            if frac > cfg["max_remove_pct"] / 100.0:
                outlier_info.update(skipped=True, reason=f"{frac:.1%} of train rows would be removed "
                                                         f"(limit {cfg['max_remove_pct']}%)")
            elif remaining_classes < 2:
                outlier_info.update(skipped=True, reason="removal would delete a whole class")
            else:
                X_tr = X_tr.loc[~mask]
                y_tr = y_tr.loc[X_tr.index]
                outlier_info["removed"] = int(mask.sum())
        else:
            X_tr, X_te = handler.cap(X_tr), handler.cap(X_te)
        steps.append(f"Outliers: {cfg['out_method']} / {cfg['out_action']}"
                     + (" (skipped)" if outlier_info["skipped"] else ""))
    else:
        steps.append("Outliers: not handled")

    train_clean, test_clean = X_tr.copy(), X_te.copy()
    y_train_clean = y_tr.copy()

    # ---- log transform of skewed columns
    log_cols: list[str] = []
    if cfg["log_skewed"]:
        X_tr, X_te, log_cols = tf.log_transform(X_tr, X_te, num_cols)
        steps.append(f"log1p on {len(log_cols)} skewed columns")

    # ---- discretization
    bin_cols: list[str] = []
    disc = [c for c in cfg.get("discretize_cols", []) if c in X_tr.columns]
    if disc:
        X_tr, X_te, bin_cols = tf.discretize(X_tr, X_te, disc, cfg["n_bins"])
        steps.append(f"Discretized {len(disc)} columns into {cfg['n_bins']} quantile bins")

    # ---- aggregation feature
    agg_col = None
    ag = cfg.get("agg")
    if ag and ag["cat"] in X_tr.columns and ag["num"] in X_tr.columns:
        X_tr, X_te, agg_col = tf.group_aggregate(X_tr, X_te, ag["cat"], ag["num"], ag["func"])
        steps.append(f"Aggregation feature: {agg_col}")

    # ---- encoding
    cats_now = [c for c in cat_cols if c in X_tr.columns]
    X_tr, X_te, encoded_cols, onehot_cols = tf.encode_categoricals(X_tr, X_te, cats_now, cfg["encoding"])
    if cats_now:
        steps.append(f"Encoded {len(cats_now)} categorical columns ({cfg['encoding']})")

    # ---- drop constant columns (no information)
    std = X_tr.std(ddof=0)
    constant = [c for c in X_tr.columns if not np.isfinite(std[c]) or std[c] == 0]
    if constant:
        X_tr, X_te = X_tr.drop(columns=constant), X_te.drop(columns=constant)
        steps.append(f"Dropped {len(constant)} constant columns")

    # ---- scaling (not for one-hot / binary columns)
    binary = [c for c in X_tr.columns if X_tr[c].nunique() <= 2]
    to_scale = [c for c in X_tr.columns if c not in binary and c not in onehot_cols]
    X_tr, X_te = tf.scale_features(X_tr, X_te, to_scale, cfg["scaling"])
    steps.append(f"Scaling: {cfg['scaling']} on {len(to_scale)} columns")

    X_tr = X_tr.astype(float)
    X_te = X_te.astype(float)
    return StageB(
        X_train=X_tr, X_test=X_te, y_train=y_tr.astype(int), y_test=y_te.astype(int),
        train_clean=train_clean, test_clean=test_clean, y_train_clean=y_train_clean.astype(int),
        outlier_info=outlier_info, outlier_counts=outlier_counts, steps=steps,
        num_cols=num_cols, cat_cols=cat_cols, log_cols=log_cols, bin_cols=bin_cols, agg_col=agg_col,
        encoded_cols=encoded_cols, scaled_cols=to_scale, dropped_constant=constant, split_sizes=split_sizes,
    )
