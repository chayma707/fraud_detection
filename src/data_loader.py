"""Data loading utilities + a built-in NOISY synthetic social-media dataset.

The synthetic dataset is only a fallback so the app always runs.
For the real project use a real dataset (e.g. Cresci-2017, see README.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET_HINTS = [
    "is_fake", "fake", "is_bot", "bot", "is_fraud", "fraud",
    "label", "target", "class", "y",
]

# Columns that are identifiers or that would leak the label
ID_LIKE_NAMES = {
    "id", "user_id", "userid", "uid", "split",
    "profile_image_url", "profile_banner_url", "profile_background_image_url",
    "profile_image_url_https", "profile_background_image_url_https",
    "profile_text_color", "profile_link_color", "profile_sidebar_fill_color",
    "profile_sidebar_border_color", "profile_background_color",
    "test_set_1", "test_set_2", "dataset", "account_group", "source",
    "crawled_at", "updated",
}


def read_csv_flexible(source) -> pd.DataFrame:
    """Read a CSV from a path or file-like object, trying several encodings/separators."""
    last_error: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding=enc, low_memory=False)
            if df.shape[1] == 1:  # maybe a different separator (; or tab)
                try:
                    if hasattr(source, "seek"):
                        source.seek(0)
                    df2 = pd.read_csv(source, encoding=enc, sep=None, engine="python")
                    if df2.shape[1] > 1:
                        df = df2
                except Exception:
                    pass
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not read the CSV file: {last_error}")


def list_raw_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.csv") if p.is_file())


def guess_target(df: pd.DataFrame) -> str:
    lowered = {c: str(c).lower() for c in df.columns}
    for hint in TARGET_HINTS:
        for col, low in lowered.items():
            if low == hint:
                return col
    for hint in TARGET_HINTS:
        for col, low in lowered.items():
            if hint in low and len(hint) > 2:
                return col
    return df.columns[-1]


def suggest_drop_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Identifier-like / leakage-prone columns that should normally be excluded."""
    drops = []
    for col in df.columns:
        if col == target:
            continue
        low = str(col).lower()
        if low in ID_LIKE_NAMES or low.endswith("_id") or low.endswith("_url"):
            drops.append(col)
    return drops


def generate_noisy_dataset(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Synthetic social-media accounts with realistic *noise*:
    missing values, disguised-missing tokens, duplicates, outliers, impossible
    (negative) values and inconsistent categories/booleans."""
    rng = np.random.default_rng(seed)
    n = int(n_rows)
    is_fake = (rng.random(n) < 0.30).astype(int)
    fake = is_fake == 1

    age = np.where(fake, rng.gamma(2.0, 90, n), rng.gamma(2.5, 220, n)).round() + 1
    followers = np.where(fake, rng.lognormal(3.9, 1.5, n), rng.lognormal(4.6, 1.5, n)).round()
    following = np.where(fake, rng.lognormal(5.6, 0.9, n), rng.lognormal(5.0, 1.0, n)).round()
    posts_per_day = np.where(fake, rng.gamma(2.0, 3.5, n), rng.gamma(2.0, 1.6, n))
    posts = (posts_per_day * np.minimum(age, 1500)).round()
    likes = np.where(fake, rng.lognormal(4.6, 1.4, n), rng.lognormal(5.3, 1.5, n)).round()
    bio_len = np.where(fake, rng.poisson(28, n), rng.poisson(48, n)).astype(float)
    has_pic = np.where(fake, rng.random(n) < 0.60, rng.random(n) < 0.88)
    verified = np.where(fake, False, rng.random(n) < 0.06)

    account_type = np.where(
        fake,
        rng.choice(["personal", "business", "creator"], n, p=[0.45, 0.40, 0.15]),
        rng.choice(["personal", "business", "creator"], n, p=[0.70, 0.15, 0.15]),
    )
    countries = np.array(["TN", "FR", "US", "DE", "EG", "MA", "DZ", "IN", "BR", "GB"])
    country = rng.choice(countries, n)
    device = np.where(
        fake,
        rng.choice(["android", "ios", "web", "api"], n, p=[0.30, 0.20, 0.30, 0.20]),
        rng.choice(["android", "ios", "web", "api"], n, p=[0.45, 0.30, 0.20, 0.05]),
    )
    email_domain = np.where(
        fake,
        rng.choice(["gmail.com", "yahoo.com", "tempmail.xyz", "mailinator.com"], n, p=[0.45, 0.15, 0.25, 0.15]),
        rng.choice(["gmail.com", "yahoo.com", "outlook.com", "tempmail.xyz"], n, p=[0.55, 0.20, 0.20, 0.05]),
    )
    signup = pd.Timestamp("2024-12-31") - pd.to_timedelta(age, unit="D")

    def make_username(is_f: bool) -> str:
        base = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), rng.integers(4, 9)))
        if is_f and rng.random() < 0.6:
            n_digits = rng.integers(2, 8)      # bots often append many digits
        else:
            n_digits = rng.integers(0, 4)
        digits = "".join(rng.choice(list("0123456789"), n_digits))
        return base + digits

    username = [make_username(bool(f)) for f in fake]

    df = pd.DataFrame({
        "user_id": np.arange(100000, 100000 + n),
        "username": username,
        "signup_date": signup.strftime("%Y-%m-%d"),
        "account_age_days": age,
        "followers_count": followers,
        "following_count": following,
        "posts_count": posts,
        "likes_given": likes,
        "bio_length": bio_len,
        "profile_has_picture": has_pic,
        "verified": verified,
        "account_type": account_type,
        "country": country,
        "device": device,
        "email_domain": email_domain,
        "is_fake": is_fake,
    })

    # ---------------- inject NOISE ----------------
    flip = rng.random(n) < 0.03            # 3% label noise (hard-to-label accounts)
    df.loc[flip, "is_fake"] = 1 - df.loc[flip, "is_fake"]
    # inconsistent booleans
    pic_variants = {True: ["Yes", "yes", "Y", "1", "TRUE", "true"], False: ["No", "no", "N", "0", "FALSE", "false"]}
    df["profile_has_picture"] = [rng.choice(pic_variants[bool(v)]) for v in df["profile_has_picture"]]
    df["verified"] = df["verified"].map({True: "True", False: "False"})
    # inconsistent text (case / whitespace)
    def messy(v: str) -> str:
        r = rng.random()
        if r < 0.10:
            return v.upper()
        if r < 0.20:
            return f"  {v} "
        if r < 0.25:
            return v.capitalize()
        return v
    for col in ["account_type", "country", "device"]:
        df[col] = [messy(str(v)) for v in df[col]]
    # outliers (follower explosion) + impossible negative values
    out_idx = rng.choice(n, size=max(1, int(0.012 * n)), replace=False)
    df.loc[out_idx, "followers_count"] = df.loc[out_idx, "followers_count"] * 1000
    neg_idx = rng.choice(n, size=max(1, int(0.006 * n)), replace=False)
    df.loc[neg_idx, "posts_count"] = -df.loc[neg_idx, "posts_count"].abs() - 1
    # missing values
    for col, rate in [("followers_count", 0.06), ("likes_given", 0.08), ("bio_length", 0.07),
                      ("posts_count", 0.05), ("account_age_days", 0.04), ("following_count", 0.04),
                      ("country", 0.05), ("device", 0.06), ("email_domain", 0.05),
                      ("account_type", 0.04), ("signup_date", 0.03)]:
        mask = rng.random(n) < rate
        df[col] = df[col].astype(object).where(~mask, np.nan) if df[col].dtype == object else df[col].where(~mask, np.nan)
    # disguised missing tokens
    for col in ["country", "device", "email_domain"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = rng.choice(["N/A", "unknown", "?", "none"], mask.sum())
    # duplicated rows
    dup = df.sample(frac=0.03, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df
