"""Transformations fitted on TRAIN and applied to TRAIN + TEST (no data leakage)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler


def log_transform(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], skew_thr: float = 2.0):
    """log1p on non-negative, highly skewed columns (chosen on train)."""
    train, test = train.copy(), test.copy()
    chosen = []
    for c in cols:
        s = train[c]
        if s.nunique() <= 2 or s.min() < 0:
            continue
        if abs(float(s.skew())) > skew_thr:
            chosen.append(c)
    for c in chosen:
        train[c] = np.log1p(train[c])
        test[c] = np.log1p(test[c].clip(lower=0))
    return train, test, chosen


def discretize(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], n_bins: int = 5):
    """Quantile binning -> new ordinal columns `<col>_bin`."""
    train, test = train.copy(), test.copy()
    new_cols = []
    for c in cols:
        if c not in train.columns:
            continue
        qs = np.quantile(train[c].dropna(), np.linspace(0, 1, int(n_bins) + 1)[1:-1])
        edges = np.unique(qs)
        name = f"{c}_bin"
        train[name] = np.digitize(train[c].to_numpy(dtype=float), edges).astype(float)
        test[name] = np.digitize(test[c].to_numpy(dtype=float), edges).astype(float)
        new_cols.append(name)
    return train, test, new_cols


def group_aggregate(train: pd.DataFrame, test: pd.DataFrame, cat_col: str, num_col: str, func: str = "mean"):
    """Aggregate a numeric column by a categorical column (statistic learned on train)."""
    train, test = train.copy(), test.copy()
    name = f"{num_col}_{func}_by_{cat_col}"
    mapping = train.groupby(cat_col)[num_col].agg(func)
    fallback = getattr(train[num_col], func)()
    train[name] = train[cat_col].map(mapping).fillna(fallback).astype(float)
    test[name] = test[cat_col].map(mapping).fillna(fallback).astype(float)
    return train, test, name


def aggregate_table(df: pd.DataFrame, cat_col: str, num_cols: list[str], func: str = "mean") -> pd.DataFrame:
    g = df.groupby(cat_col)[num_cols].agg(func)
    g.insert(0, "count", df.groupby(cat_col).size())
    return g.sort_values("count", ascending=False).head(25).round(3).reset_index()


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list[str], method: str = "one-hot"):
    """Returns train, test, encoded_columns, onehot_columns."""
    if not cat_cols:
        return train, test, [], []
    train, test = train.copy(), test.copy()
    if method == "one-hot":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        a = enc.fit_transform(train[cat_cols].astype(str))
        b = enc.transform(test[cat_cols].astype(str))
        names = [str(n).replace(" ", "_") for n in enc.get_feature_names_out(cat_cols)]
        tr = pd.DataFrame(a, index=train.index, columns=names)
        te = pd.DataFrame(b, index=test.index, columns=names)
        train = pd.concat([train.drop(columns=cat_cols), tr], axis=1)
        test = pd.concat([test.drop(columns=cat_cols), te], axis=1)
        return train, test, names, names
    if method == "ordinal":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train[cat_cols] = enc.fit_transform(train[cat_cols].astype(str))
        test[cat_cols] = enc.transform(test[cat_cols].astype(str))
        return train, test, list(cat_cols), []
    if method == "frequency":
        for c in cat_cols:
            freq = train[c].astype(str).value_counts(normalize=True)
            train[c] = train[c].astype(str).map(freq).fillna(0.0).astype(float)
            test[c] = test[c].astype(str).map(freq).fillna(0.0).astype(float)
        return train, test, list(cat_cols), []
    raise ValueError(f"Unknown encoding method: {method}")


def scale_features(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], method: str = "standard"):
    if method == "none" or not cols:
        return train, test
    scaler = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[method]()
    train, test = train.copy(), test.copy()
    train[cols] = scaler.fit_transform(train[cols])
    test[cols] = scaler.transform(test[cols])
    return train, test
