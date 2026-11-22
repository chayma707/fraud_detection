"""Column typing + feature engineering (row-wise, nothing is fitted here)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .cleaning import is_text_like

DATE_NAME_HINTS = ("date", "created", "_at", "time", "updated", "joined", "signup")
ENGINEERED_SUFFIXES = ("_was_missing", "_len", "_digit_ratio", "_age_days")


def _to_datetime(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce", utc=True)


def looks_like_date(name: str, s: pd.Series) -> bool:
    if not any(h in str(name).lower() for h in DATE_NAME_HINTS):
        return False
    sample = s.dropna().head(300)
    if len(sample) == 0:
        return False
    parsed = _to_datetime(sample.astype(str))
    return float(parsed.notna().mean()) >= 0.8


def classify_columns(df: pd.DataFrame, max_cat: int = 30) -> dict[str, list[str]]:
    """numeric / categorical (low-cardinality text) / text (high-cardinality) / date."""
    out = {"numeric": [], "categorical": [], "text": [], "date": []}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            out["date"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            out["numeric"].append(col)
        elif is_text_like(s):
            if looks_like_date(col, s):
                out["date"].append(col)
            elif s.nunique(dropna=True) <= max_cat:
                out["categorical"].append(col)
            else:
                out["text"].append(col)
        else:
            out["text"].append(col)
    return out


def _find(cols: list[str], keywords: list[str]) -> str | None:
    for c in cols:
        low = str(c).lower()
        if low.endswith(ENGINEERED_SUFFIXES) or "ratio" in low or "per_" in low:
            continue
        if any(k in low for k in keywords):
            return c
    return None


def engineer_features(X: pd.DataFrame, max_cat: int = 30, ratios: bool = True, text: bool = True,
                      dates: bool = True, missing_flags: bool = True) -> tuple[pd.DataFrame, dict]:
    """Create new informative columns (works on any social-media-like table)."""
    out = X.copy()
    log: dict[str, list[str]] = {"missing_flags": [], "text_features": [], "date_features": [],
                                 "ratio_features": [], "dropped_raw": []}
    kinds = classify_columns(out, max_cat)

    # 1) missing-value indicators (missingness itself is a fraud signal)
    if missing_flags:
        for c in list(out.columns):
            rate = float(out[c].isna().mean())
            if 0.01 <= rate <= 0.95:
                name = f"{c}_was_missing"
                out[name] = out[c].isna().astype(float)
                log["missing_flags"].append(name)

    # 2) dates -> age in days (relative to the newest date in the column, reproducible)
    for c in kinds["date"]:
        if dates:
            dt = _to_datetime(out[c])
            ref = dt.max()
            name = f"{c}_age_days"
            out[name] = (ref - dt).dt.total_seconds() / 86400.0
            log["date_features"].append(name)
        out = out.drop(columns=c)
        log["dropped_raw"].append(c)

    # 3) free text / high-cardinality columns -> length + digit ratio
    for c in kinds["text"]:
        if text:
            s = out[c].fillna("").astype(str)
            out[f"{c}_len"] = s.str.len().astype(float)
            out[f"{c}_digit_ratio"] = s.map(
                lambda v: (sum(ch.isdigit() for ch in v) / len(v)) if len(v) else 0.0
            ).astype(float)
            log["text_features"] += [f"{c}_len", f"{c}_digit_ratio"]
        out = out.drop(columns=c)
        log["dropped_raw"].append(c)

    # 4) behavioural ratios typical for bot detection
    if ratios:
        num = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        fol = _find(num, ["follower"])
        fng = _find(num, ["following", "friends", "followee"])
        pst = _find(num, ["post", "statuses", "tweet"])
        lik = _find(num, ["like", "favourite", "favorite"])
        age = _find(num, ["account_age", "age_days", "_age"])
        if fol and fng:
            out["follower_following_ratio"] = out[fol] / (out[fng] + 1)
            out["follower_share"] = out[fol] / (out[fol] + out[fng] + 1)
            log["ratio_features"] += ["follower_following_ratio", "follower_share"]
        if pst and age:
            out["posts_per_day"] = out[pst] / (out[age] + 1)
            log["ratio_features"].append("posts_per_day")
        if lik and pst:
            out["likes_per_post"] = out[lik] / (out[pst] + 1)
            log["ratio_features"].append("likes_per_post")

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, log
