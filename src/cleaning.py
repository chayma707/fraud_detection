"""Data cleaning: inconsistencies, duplicates, impossible values, outliers, missing values."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na", "?", "-", "unknown",
                  "undefined", "missing", "nil", "<na>"}
TRUE_TOKENS = {"yes", "y", "true", "t", "1", "1.0"}
FALSE_TOKENS = {"no", "n", "false", "f", "0", "0.0"}


def is_text_like(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


# --------------------------------------------------------------------------
# Inconsistencies
# --------------------------------------------------------------------------
def standardize_text(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Trim / lower-case text, turn disguised-missing tokens into NaN,
    unify yes/no/true/false/1/0 columns into 0/1 and numbers stored as text into numbers."""
    out = df.copy()
    log = {"tokens_to_nan": 0, "boolean_columns": [], "numeric_converted": [], "text_columns": 0}
    for col in list(out.columns):
        s = out[col]
        if not is_text_like(s):
            continue
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
        log["text_columns"] += 1
        notna = s.notna()
        cleaned = (
            s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
        )
        cleaned = cleaned.where(notna)
        is_token = cleaned.isin(MISSING_TOKENS)
        log["tokens_to_nan"] += int(is_token.sum())
        cleaned = cleaned.mask(is_token)

        values = set(cleaned.dropna().unique())
        if values and values.issubset(TRUE_TOKENS | FALSE_TOKENS):
            out[col] = cleaned.map(
                lambda v: 1.0 if v in TRUE_TOKENS else (0.0 if v in FALSE_TOKENS else np.nan),
                na_action="ignore",
            ).astype(float)
            log["boolean_columns"].append(col)
            continue

        numeric = pd.to_numeric(cleaned, errors="coerce")
        n_valid = int(cleaned.notna().sum())
        if n_valid > 0 and int(numeric.notna().sum()) >= 0.95 * n_valid:
            out[col] = numeric.astype(float)
            log["numeric_converted"].append(col)
            continue

        out[col] = cleaned
    return out, log


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    out = df.drop_duplicates()
    return out, before - len(out)


# --------------------------------------------------------------------------
# Impossible values (e.g. negative counts)
# --------------------------------------------------------------------------
def fix_impossible_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    fixed: dict[str, int] = {}
    for col in out.columns:
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        nn = s.dropna()
        if len(nn) == 0 or not (nn < 0).any():
            continue
        if (nn >= 0).mean() >= 0.90:  # a "count-like" column: negatives are impossible
            n_bad = int((s < 0).sum())
            out[col] = s.mask(s < 0)
            fixed[col] = n_bad
    return out, fixed


# --------------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------------
def zscore_bounds(s: pd.Series, thr: float = 3.0):
    s = s.dropna()
    sd = s.std(ddof=0)
    if len(s) < 5 or sd == 0:
        return None
    mu = s.mean()
    return float(mu - thr * sd), float(mu + thr * sd)


def iqr_bounds(s: pd.Series, k: float = 1.5):
    s = s.dropna()
    if len(s) < 5:
        return None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    return float(q1 - k * iqr), float(q3 + k * iqr)


def outlier_report(df: pd.DataFrame, cols: list[str], z_thr: float = 3.0, iqr_k: float = 1.5) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c]
        zb, ib = zscore_bounds(s, z_thr), iqr_bounds(s, iqr_k)
        n_z = int(((s < zb[0]) | (s > zb[1])).sum()) if zb else 0
        n_i = int(((s < ib[0]) | (s > ib[1])).sum()) if ib else 0
        n = max(int(s.notna().sum()), 1)
        rows.append({
            "column": c,
            "outliers (Z-score)": n_z,
            "% (Z-score)": round(100 * n_z / n, 2),
            "outliers (IQR)": n_i,
            "% (IQR)": round(100 * n_i / n, 2),
            "skewness": round(float(s.skew()), 2) if s.notna().sum() > 2 else np.nan,
        })
    return pd.DataFrame(rows)


class OutlierHandler:
    """Learn bounds on TRAIN only, then flag rows or cap values."""

    def __init__(self, method: str = "zscore", z_thr: float = 3.0, iqr_k: float = 1.5):
        self.method = method
        self.z_thr = z_thr
        self.iqr_k = iqr_k
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame, cols: list[str]):
        self.bounds_ = {}
        for c in cols:
            s = df[c]
            if s.dropna().nunique() <= 2:  # skip binary flags
                continue
            b = zscore_bounds(s, self.z_thr) if self.method == "zscore" else iqr_bounds(s, self.iqr_k)
            if b is not None:
                self.bounds_[c] = b
        return self

    def outlier_mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        for c, (lo, hi) in self.bounds_.items():
            mask |= (df[c] < lo) | (df[c] > hi)
        return mask

    def counts(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series({c: int(((df[c] < lo) | (df[c] > hi)).sum()) for c, (lo, hi) in self.bounds_.items()})

    def cap(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, (lo, hi) in self.bounds_.items():
            out[c] = out[c].clip(lower=lo, upper=hi)
        return out


# --------------------------------------------------------------------------
# Missing values
# --------------------------------------------------------------------------
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = max(len(df), 1)
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing %": (df.isna().sum() / n * 100).round(2),
        "unique": df.nunique(dropna=True),
    })
    out.index.name = "column"
    return out.reset_index()


def impute_train_test(train: pd.DataFrame, test: pd.DataFrame, num_cols: list[str], cat_cols: list[str],
                      num_method: str = "median", cat_method: str = "mode", knn_k: int = 5):
    """Fit imputers on TRAIN only and apply to train and test."""
    train, test = train.copy(), test.copy()
    if num_cols:
        if num_method in ("mean", "median", "mode"):
            strategy = {"mean": "mean", "median": "median", "mode": "most_frequent"}[num_method]
            imp = SimpleImputer(strategy=strategy)
            train[num_cols] = imp.fit_transform(train[num_cols])
            test[num_cols] = imp.transform(test[num_cols])
        elif num_method == "knn":
            scaler = StandardScaler().fit(train[num_cols])
            knn = KNNImputer(n_neighbors=int(knn_k))
            tr = knn.fit_transform(scaler.transform(train[num_cols]))
            te = knn.transform(scaler.transform(test[num_cols]))
            train[num_cols] = scaler.inverse_transform(tr)
            test[num_cols] = scaler.inverse_transform(te)
        elif num_method == "ffill":
            med = train[num_cols].median()
            train[num_cols] = train[num_cols].ffill().bfill().fillna(med)
            test[num_cols] = test[num_cols].ffill().bfill().fillna(med)
        else:
            raise ValueError(f"Unknown numeric imputation method: {num_method}")
    for c in cat_cols:
        if cat_method == "mode":
            m = train[c].mode(dropna=True)
            fill = m.iloc[0] if len(m) else "missing"
        else:
            fill = "missing"
        train[c] = train[c].fillna(fill)
        test[c] = test[c].fillna(fill)
    return train, test
