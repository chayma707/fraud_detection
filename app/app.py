"""Social-media fraud detection - Data Mining project (Streamlit dashboard).

Run with:   streamlit run app/app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project folder (independent of where the app is launched from)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src import cleaning as cl
from src import data_loader as dl
from src import features as ft
from src import model as md
from src import pipeline as pl
from src import reduction as rd
from src import selection as sel
from src import transform as tf
from src import versioning as vz

TARGET = pl.TARGET
CLASS_NAMES = {0: "Genuine (0)", 1: "Fraud (1)"}
COLORS = {0: "#2b8cbe", 1: "#e34a33"}

st.set_page_config(page_title="Social Media Fraud Detection", page_icon="🕵️", layout="wide")


# ==========================================================================
# Helpers
# ==========================================================================
def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make mixed-type columns displayable."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)):
            out[c] = s.map(lambda v: v if pd.isna(v) else str(v))
    return out


def show_df(df: pd.DataFrame, rows: int = 200):
    st.dataframe(arrow_safe(df.head(rows)))


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def takeaway(text: str):
    st.markdown(f"> 💡 **Takeaway:** {text}")


def numeric_cols(df: pd.DataFrame, exclude=()) -> list[str]:
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]


def fmt(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"


# ==========================================================================
# Cached loaders / stages
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_demo(n_rows: int, seed: int) -> pd.DataFrame:
    return dl.generate_noisy_dataset(n_rows, seed)


@st.cache_data(show_spinner="Reading the file...")
def load_file(path: str, mtime: float) -> pd.DataFrame:
    return dl.read_csv_flexible(path)


@st.cache_data(show_spinner="Reading the file...")
def load_bytes(data: bytes) -> pd.DataFrame:
    return dl.read_csv_flexible(io.BytesIO(data))


@st.cache_data(show_spinner="Structural cleaning + feature engineering...")
def run_stage_a(raw, target, positive, drop_cols, cfg):
    return pl.prepare_features(raw, target, positive, list(drop_cols), cfg)


@st.cache_data(show_spinner="Splitting, imputing, handling outliers, transforming...")
def run_stage_b(X, y, cfg):
    return pl.preprocess(X, y, cfg)


@st.cache_data(show_spinner="Computing filter / wrapper / embedded scores...")
def run_selection(X, y, k, seed):
    return sel.compute_feature_scores(X, y, k, seed)


@st.cache_data(show_spinner="Running t-SNE (can take a moment)...")
def run_tsne_cached(X, y, max_points, perplexity, seed):
    return rd.run_tsne(X, y, max_points, perplexity, seed)


@st.cache_data(show_spinner="Training the autoencoder...")
def run_ae_cached(X_train, X_test, bottleneck, max_iter, seed):
    return rd.run_autoencoder(X_train, X_test, bottleneck, max_iter, seed)


# ---- Git + DVC helpers (the "rev" argument changes after every saved version -> cache refresh)
if "versions_rev" not in st.session_state:
    st.session_state["versions_rev"] = 0
REV = st.session_state["versions_rev"]


@st.cache_data(show_spinner=False)
def cached_status(rev: int):
    return vz.project_status()


@st.cache_data(show_spinner=False)
def cached_versions(rev: int):
    return vz.list_versions()


@st.cache_data(show_spinner=False)
def cached_version_info(tag: str, rev: int):
    return vz.read_version_info(tag)


@st.cache_data(show_spinner="Loading this version from DVC...")
def cached_load_version(tag: str):
    return vz.load_version(tag)   # a tag never changes, so this can be cached forever


# ==========================================================================
# SIDEBAR - data
# ==========================================================================
st.sidebar.title("⚙️ Project settings")
st.sidebar.caption("Every option here re-runs the whole pipeline live.")

raw_dir = ROOT / "data" / "raw"
raw_files = dl.list_raw_files(raw_dir)
SRC_OPTIONS = ["Upload a CSV", "File from data/raw/", "Built-in noisy demo dataset",
               "Tracked dataset version (DVC tag)"]

with st.sidebar.expander("1️⃣ Data source & target", expanded=True):
    source = st.radio("Data source", SRC_OPTIONS, index=1 if raw_files else 2)
    if source == SRC_OPTIONS[0]:
        up = st.file_uploader("CSV file", type=["csv"])
        if up is None:
            st.info("Upload a CSV file to start (or choose another data source).")
            st.stop()
        df_raw = load_bytes(up.getvalue())
        dataset_name = up.name
    elif source == SRC_OPTIONS[1]:
        if not raw_files:
            st.warning("No CSV found in data/raw/. Put your dataset there or pick another source.")
            st.stop()
        choice = st.selectbox("File in data/raw/", [f.name for f in raw_files])
        path = raw_dir / choice
        df_raw = load_file(str(path), path.stat().st_mtime)
        dataset_name = choice
    elif source == SRC_OPTIONS[2]:
        n_demo = st.slider("Number of rows", 1000, 10000, 3000, 500)
        df_raw = load_demo(int(n_demo), 42)
        dataset_name = "demo_noisy_social_media_accounts"
    else:
        _status = cached_status(REV)
        _versions = cached_versions(REV) if _status["ready"] else []
        if not _versions:
            st.warning("No dataset version found. Initialise Git + DVC and create the tag v1.0 first "
                       "(see the tab 'Data versions').")
            st.stop()
        _labels = {v["tag"]: f"{v['tag']} - {v['message']}" for v in _versions}
        _tag = st.selectbox("Dataset version", list(_labels), format_func=_labels.get)
        df_raw = cached_load_version(_tag)
        dataset_name = f"DVC version {_tag}"

    if df_raw.shape[0] < 50 or df_raw.shape[1] < 3:
        st.error("The dataset is too small (need at least 50 rows and 3 columns).")
        st.stop()

    all_cols = list(df_raw.columns)
    guess = dl.guess_target(df_raw)
    target = st.selectbox("Target column (fraud label)", all_cols, index=all_cols.index(guess))
    vcounts = pl.target_value_counts(df_raw[target])
    if len(vcounts) < 2:
        st.error("The target column has a single value, choose another one.")
        st.stop()
    if len(vcounts) > 30:
        st.error("The target column has too many distinct values to be a class label.")
        st.stop()
    pos_options = list(vcounts.index)
    pos_default = pl.guess_positive(vcounts)
    positive = st.selectbox(
        "Value meaning FRAUD / fake / bot (class 1)", pos_options, index=pos_options.index(pos_default),
        format_func=lambda v: f"{v}   ({int(vcounts[v]):,} rows)",
    )
    drop_default = dl.suggest_drop_columns(df_raw, target)
    drop_cols = st.multiselect(
        "Columns to drop (IDs / leakage)", [c for c in all_cols if c != target], default=drop_default,
        help="Identifiers and columns that reveal the label must not be used as features.",
    )

with st.sidebar.expander("2️⃣ Cleaning & feature engineering"):
    c_std = st.checkbox("Fix inconsistencies (case, spaces, yes/no, 'N/A' tokens)", True)
    c_neg = st.checkbox("Impossible negative values -> missing", True)
    c_dup = st.checkbox("Remove duplicate rows", True)
    max_cat = st.slider("Max distinct values for a 'categorical' column", 5, 100, 30)
    f_miss = st.checkbox("Add missing-value indicator columns", True)
    f_ratio = st.checkbox("Add behavioural ratios (followers/following, posts/day...)", True)
    f_text = st.checkbox("Text features (length, digit ratio) for names / bios", True)
    f_date = st.checkbox("Date columns -> age in days", True)

cfg_a = {"standardize": c_std, "fix_negatives": c_neg, "drop_dups": c_dup, "max_cat": int(max_cat),
         "missing_flags": f_miss, "ratios": f_ratio, "text": f_text, "dates": f_date}

try:
    A = run_stage_a(df_raw, target, positive, tuple(drop_cols), cfg_a)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not prepare the data: {exc}")
    st.stop()

kinds_fe = ft.classify_columns(A.X, int(max_cat))
disc_candidates = [c for c in kinds_fe["numeric"] if A.X[c].nunique() > 5]

with st.sidebar.expander("3️⃣ Split, missing values, outliers"):
    test_size = st.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
    seed = int(st.number_input("Random seed", 0, 9999, 42))
    num_impute = st.selectbox("Numeric imputation", ["median", "mean", "mode", "knn", "ffill"], index=0,
                              help="mode = most frequent value, ffill = forward fill")
    knn_k = st.slider("k for KNN imputation", 2, 15, 5) if num_impute == "knn" else 5
    cat_impute = st.selectbox("Categorical imputation", ["mode", "constant 'missing'"], index=0)
    out_method_label = st.selectbox("Outlier detection", ["Z-score", "IQR", "None"], index=0)
    out_action_label = st.selectbox("Outlier action", ["Remove rows (train only)", "Cap (winsorize)"], index=0)
    z_thr = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.5)
    iqr_k = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.25)
    max_remove_pct = st.slider("Safety limit: max % of train rows removable", 1, 60, 25,
                               help="If outlier removal would delete more rows than this, it is skipped.")

with st.sidebar.expander("4️⃣ Transformation"):
    log_skewed = st.checkbox("log1p transform of very skewed columns", True)
    scaling = st.selectbox("Scaling / normalisation", ["standard", "minmax", "robust", "none"], index=0)
    encoding = st.selectbox("Categorical encoding", ["one-hot", "ordinal", "frequency"], index=0)
    disc_cols = st.multiselect("Discretize columns (quantile bins)", disc_candidates, default=[])
    n_bins = st.slider("Number of bins", 3, 10, 5)
    cats_agg = [c for c in kinds_fe["categorical"]]
    nums_agg = [c for c in kinds_fe["numeric"] if A.X[c].nunique() > 5]
    agg_on = st.checkbox("Add aggregation feature (group statistic)", False,
                         disabled=not (cats_agg and nums_agg))
    agg_cfg = None
    if agg_on and cats_agg and nums_agg:
        a1 = st.selectbox("Group by (categorical)", cats_agg)
        a2 = st.selectbox("Aggregate (numeric)", nums_agg)
        a3 = st.selectbox("Statistic", ["mean", "median", "std", "max"])
        agg_cfg = {"cat": a1, "num": a2, "func": a3}

cfg_b = {
    "test_size": float(test_size), "seed": seed, "max_cat": int(max_cat),
    "num_impute": num_impute, "cat_impute": "mode" if cat_impute == "mode" else "constant",
    "knn_k": int(knn_k),
    "out_method": {"Z-score": "zscore", "IQR": "iqr", "None": "none"}[out_method_label],
    "out_action": "remove" if out_action_label.startswith("Remove") else "cap",
    "z_thr": float(z_thr), "iqr_k": float(iqr_k), "max_remove_pct": float(max_remove_pct),
    "log_skewed": log_skewed, "scaling": scaling, "encoding": encoding,
    "discretize_cols": list(disc_cols), "n_bins": int(n_bins), "agg": agg_cfg,
}

try:
    B = run_stage_b(A.X, A.y, cfg_b)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not run the preprocessing pipeline: {exc}")
    st.stop()

n_feat = B.X_train.shape[1]
with st.sidebar.expander("5️⃣ Feature selection & reduction"):
    k_sel = st.slider("Number of features to keep (k)", 1, max(1, n_feat), min(15, n_feat))
    sel_method = st.selectbox("Method used for the selected feature set", sel.METHODS, index=3)
    pca_n = st.slider("PCA components kept", 1, max(1, min(n_feat, 30)), min(5, max(1, n_feat)))

# ==========================================================================
# Derived objects
# ==========================================================================
scores = run_selection(B.X_train, B.y_train, int(k_sel), seed)
selected = sel.top_k(scores, sel_method, int(k_sel))
pca_full, pca_tr, pca_te = rd.run_pca(B.X_train, B.X_test, int(pca_n), seed)
clean_all = pd.concat([B.train_clean.assign(**{TARGET: B.y_train_clean}),
                       B.test_clean.assign(**{TARGET: B.y_test})]).sort_index()
final_all = pd.concat([B.X_train.assign(**{TARGET: B.y_train}),
                       B.X_test.assign(**{TARGET: B.y_test})]).sort_index()
ae_result = None

# ==========================================================================
# MAIN
# ==========================================================================
st.title("🕵️ Social Media Fraud Detection — Data Mining Pipeline")
st.caption(f"Dataset: **{dataset_name}** · target: **{target}** (fraud = `{positive}`) · "
           f"{len(df_raw):,} raw rows · {df_raw.shape[1]} raw columns")

tabs = st.tabs([
    "📋 Overview", "📂 Raw data", "🔎 EDA", "🧹 Cleaning", "🔧 Transformation & features",
    "✂️ Train/Test split", "🎯 Feature selection", "📉 Dimensionality reduction",
    "🤖 Model", "↔️ Before vs After", "🧬 MLOps / Versioning", "🏷️ Data versions",
])

# --------------------------------------------------------------------------
# 0. Overview
# --------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Project checklist (live: reflects the current sidebar settings)")
    n_missing_raw = int(df_raw.isna().sum().sum()) + int(A.log.get("tokens_to_nan", 0))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw rows", f"{len(df_raw):,}")
    c2.metric("Rows after cleaning", f"{len(clean_all):,}")
    c3.metric("Final features", f"{n_feat}")
    c4.metric("Fraud share", f"{100 * float(A.y.mean()):.1f}%")

    fe_total = sum(len(v) for k, v in A.fe_log.items() if k != "dropped_raw")
    rows = [
        ("Exploratory data analysis", "✅", "Tab EDA: distribution, heatmap, bar comparison, trend"),
        ("Missing values", "✅", f"{n_missing_raw:,} missing cells -> {num_impute} / {cat_impute}"),
        ("Outliers (Z-score / IQR)", "✅" if cfg_b["out_method"] != "none" else "⚪",
         f"{out_method_label}, {out_action_label}; removed {B.outlier_info['removed']} train rows"
         + (f" (skipped: {B.outlier_info['reason']})" if B.outlier_info["skipped"] else "")),
        ("Duplicates", "✅" if c_dup else "⚪", f"{A.log.get('duplicates_removed', 0)} duplicate rows removed"),
        ("Inconsistencies", "✅" if c_std else "⚪",
         f"{A.log.get('tokens_to_nan', 0)} disguised-missing tokens, "
         f"{len(A.log.get('boolean_columns', []))} yes/no columns unified"),
        ("Normalisation / standardisation", "✅" if scaling != "none" else "⚪", f"{scaling} on {len(B.scaled_cols)} columns"),
        ("Categorical encoding", "✅", f"{encoding} ({len(B.cat_cols)} categorical columns)"),
        ("Aggregation / discretisation", "✅" if (disc_cols or agg_cfg) else "⚪",
         f"{len(B.bin_cols)} binned columns, aggregation: {B.agg_col or 'off'}"),
        ("Feature engineering", "✅" if fe_total else "⚪", f"{fe_total} engineered columns"),
        ("Feature selection - filter", "✅", "correlation + mutual information"),
        ("Feature selection - wrapper", "✅", "RFE (logistic regression)"),
        ("Feature selection - embedded", "✅", "LASSO + Decision-Tree importance"),
        ("Train / test split", "✅", f"{B.split_sizes['train']:,} / {B.split_sizes['test']:,} (stratified)"),
        ("PCA (visualisation + features)", "✅", f"{pca_n} components kept"),
        ("t-SNE", "✅", "Tab Dimensionality reduction (tick the box)"),
        ("Autoencoder", "✅", "Tab Dimensionality reduction (tick the box)"),
        ("Model + evaluation", "✅", "Tab Model"),
    ]
    st.table(pd.DataFrame(rows, columns=["Step", "Status", "Details"]))
    st.markdown("**Pipeline order:** " + " → ".join(B.steps))

# --------------------------------------------------------------------------
# 1. Raw data
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Data before cleaning")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df_raw.shape[0]:,}")
    c2.metric("Columns", f"{df_raw.shape[1]}")
    c3.metric("Missing cells", f"{int(df_raw.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df_raw.duplicated().sum()):,}")
    st.markdown("**First rows**")
    show_df(df_raw, 100)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Data types & null counts**")
        st.dataframe(arrow_safe(cl.missing_summary(df_raw)))
    with cc2:
        st.markdown("**Target distribution (raw)**")
        st.dataframe(vcounts.rename("rows").reset_index().rename(columns={"index": target, target: target}))
    st.caption("The raw file is never modified: all cleaning happens in memory.")

# --------------------------------------------------------------------------
# 2. EDA
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Exploratory data analysis")
    st.caption("Computed after the structural cleaning (case/spaces, duplicates, impossible values) but "
               "BEFORE imputation and outlier handling, so the noise is still visible.")
    S = A.struct
    nums = numeric_cols(S, [TARGET])

    # --- target balance
    st.markdown("#### 1. Target balance")
    bal = S[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.bar([CLASS_NAMES[i] for i in bal.index], bal.values, color=[COLORS[i] for i in bal.index])
    ax.set_ylabel("accounts")
    show_fig(fig)
    share = 100 * bal.get(1, 0) / bal.sum()
    takeaway(f"{share:.1f}% of the accounts are fraudulent. "
             + ("The classes are imbalanced, so accuracy alone is misleading: use precision, recall and F1."
                if share < 40 or share > 60 else "The classes are fairly balanced."))

    if not nums:
        st.info("No numeric columns with enough distinct values for the charts below.")
    else:
        # --- distribution
        st.markdown("#### 2. Distribution")
        col = st.selectbox("Numeric column", nums, key="eda_dist")
        use_log = st.checkbox("Log scale (log1p)", value=bool(abs(S[col].skew()) > 2), key="eda_log")
        fig, ax = plt.subplots(figsize=(7, 3.4))
        vals = {}
        for cls in (0, 1):
            v = S.loc[S[TARGET] == cls, col].dropna()
            v = np.log1p(v.clip(lower=0)) if use_log else v
            vals[cls] = v
            if len(v):
                ax.hist(v, bins=40, alpha=0.55, density=True, color=COLORS[cls], label=CLASS_NAMES[cls])
        ax.set_xlabel(f"log1p({col})" if use_log else col)
        ax.legend()
        show_fig(fig)
        med0, med1 = S.loc[S[TARGET] == 0, col].median(), S.loc[S[TARGET] == 1, col].median()
        sk = S[col].skew()
        takeaway(f"Median of `{col}`: genuine = {fmt(med0)}, fraud = {fmt(med1)}. "
                 f"Skewness = {sk:.2f} ("
                 + ("strongly skewed: a log transform / outlier handling is useful" if abs(sk) > 2
                    else "reasonably symmetric") + ").")

        # --- heatmap
        st.markdown("#### 3. Correlation heatmap")
        corr_t = S[nums].corrwith(S[TARGET]).abs().sort_values(ascending=False)
        top_cols = list(corr_t.head(15).index)
        corr = S[top_cols + [TARGET]].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=len(top_cols) <= 10, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                    cbar_kws={"shrink": 0.7})
        show_fig(fig)
        best = corr_t.index[0]
        pairs = corr.loc[top_cols, top_cols].where(~np.eye(len(top_cols), dtype=bool)).abs().unstack().dropna()
        pair_txt = ""
        if len(pairs):
            (p1, p2), pv = pairs.idxmax(), pairs.max()
            pair_txt = f" The most correlated pair of features is `{p1}` / `{p2}` (|r| = {pv:.2f})" \
                       + (", which signals redundancy." if pv > 0.8 else ".")
        takeaway(f"`{best}` has the strongest linear link with the fraud label (|r| = {corr_t.iloc[0]:.2f})."
                 + pair_txt)

        # --- bar comparison
        st.markdown("#### 4. Bar comparison by class")
        bcol = st.selectbox("Numeric column", nums, index=nums.index(best), key="eda_bar")
        g = S.groupby(TARGET)[bcol].agg(["mean", "median"]).reindex([0, 1])
        fig, ax = plt.subplots(figsize=(6, 3.2))
        xpos = np.arange(2)
        ax.bar(xpos - 0.18, g["mean"], 0.36, label="mean", color="#9ecae1")
        ax.bar(xpos + 0.18, g["median"], 0.36, label="median", color="#3182bd")
        ax.set_xticks(xpos)
        ax.set_xticklabels([CLASS_NAMES[0], CLASS_NAMES[1]])
        ax.set_ylabel(bcol)
        ax.legend()
        show_fig(fig)
        m0, m1 = g.loc[0, "median"], g.loc[1, "median"]
        ratio = (m1 / m0) if m0 not in (0, np.nan) and np.isfinite(m0) and m0 != 0 else np.nan
        takeaway(f"For `{bcol}` the fraud median is {fmt(m1)} vs {fmt(m0)} for genuine accounts"
                 + (f" ({ratio:.1f}x)." if np.isfinite(ratio) else ".")
                 + " A large gap means the feature separates the classes well.")

        # --- trend
        st.markdown("#### 5. Trend")
        t1, t2 = st.columns(2)
        age_like = [c for c in nums if "age" in c.lower() or "day" in c.lower()]
        xcol = t1.selectbox("X axis (binned)", nums, index=nums.index(age_like[0]) if age_like else 0,
                            key="eda_tx")
        ycol = t2.selectbox("Y axis (mean per bin)", nums, index=nums.index(best), key="eda_ty")
        tmp = S[[xcol, ycol, TARGET]].dropna()
        try:
            tmp["bin"] = pd.qcut(tmp[xcol], q=10, duplicates="drop")
            gg = tmp.groupby(["bin", TARGET], observed=True)[ycol].mean().reset_index()
            cats = list(tmp["bin"].cat.categories)
            gg["idx"] = gg["bin"].cat.codes
            fig, ax = plt.subplots(figsize=(8, 3.4))
            for cls in (0, 1):
                d = gg[gg[TARGET] == cls]
                ax.plot(d["idx"], d[ycol], marker="o", color=COLORS[cls], label=CLASS_NAMES[cls])
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels([f"{iv.mid:,.0f}" if abs(iv.mid) >= 10 else f"{iv.mid:.2f}" for iv in cats],
                               rotation=45)
            ax.set_xlabel(f"{xcol} (bin midpoint)")
            ax.set_ylabel(f"mean {ycol}")
            ax.legend()
            show_fig(fig)
            msgs = []
            for cls in (0, 1):
                d = gg[gg[TARGET] == cls].sort_values("idx")
                if len(d) >= 4:
                    half = len(d) // 2
                    first, last = d[ycol].iloc[:half].mean(), d[ycol].iloc[-half:].mean()
                    msgs.append(f"{CLASS_NAMES[cls]}: {'increases' if last > first else 'decreases'} "
                                f"with `{xcol}` ({fmt(first)} → {fmt(last)})")
            takeaway("; ".join(msgs) + "." if msgs else "Not enough bins to describe a trend.")
        except ValueError:
            st.info("This X column has too few distinct values to build bins. Pick another one.")

    # --- categorical
    cat_eda = [c for c in S.columns if c != TARGET and cl.is_text_like(S[c]) and 1 < S[c].nunique() <= 30]
    if cat_eda:
        st.markdown("#### 6. Fraud rate by category")
        cc = st.selectbox("Categorical column", cat_eda, key="eda_cat")
        rate = (S.groupby(cc)[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False) * [100, 1])
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(rate.index.astype(str), rate["mean"], color="#e34a33")
        ax.set_ylabel("% fraud")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        show_fig(fig)
        takeaway(f"Highest fraud rate in `{cc}`: **{rate.index[0]}** ({rate['mean'].iloc[0]:.1f}%), "
                 f"lowest: **{rate.index[-1]}** ({rate['mean'].iloc[-1]:.1f}%).")

# --------------------------------------------------------------------------
# 3. Cleaning
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Data cleaning")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duplicates removed", f"{A.log.get('duplicates_removed', 0):,}")
    c2.metric("Disguised-missing tokens", f"{A.log.get('tokens_to_nan', 0):,}")
    c3.metric("Impossible values fixed", f"{sum(A.log.get('negatives_fixed', {}).values()):,}")
    c4.metric("Outlier rows removed (train)", f"{B.outlier_info['removed']:,}")
    if B.outlier_info["skipped"]:
        st.warning(f"Outlier removal skipped: {B.outlier_info['reason']}. "
                   "Raise the safety limit, switch to IQR/cap, or use a higher threshold.")

    st.markdown("#### Missing values")
    ms_before = cl.missing_summary(A.X)[["column", "missing", "missing %"]].rename(
        columns={"missing": "missing (before)", "missing %": "missing % (before)"})
    after_all = pd.concat([B.train_clean, B.test_clean])
    ms_after = pd.DataFrame({"column": after_all.columns, "missing (after)": after_all.isna().sum().values})
    st.dataframe(arrow_safe(ms_before.merge(ms_after, on="column", how="left")))
    worst = ms_before.sort_values("missing (before)", ascending=False).iloc[0]
    takeaway(f"`{worst['column']}` is the most incomplete column ({worst['missing % (before)']:.1f}% missing). "
             f"After {num_impute} imputation (fitted on train only) no missing value remains.")

    st.markdown("#### Outliers")
    ncols = [c for c in B.num_cols if A.X[c].nunique() > 2]
    if ncols:
        rep = cl.outlier_report(A.X, ncols, cfg_b["z_thr"], cfg_b["iqr_k"]).sort_values(
            "% (IQR)", ascending=False)
        st.dataframe(rep)
        takeaway("Count-like columns (followers, posts, likes) are heavy-tailed, so many values look extreme. "
                 "In fraud detection extreme values are often the signal itself: compare the model "
                 "with/without outlier removal in the Model tab.")
        bcol = st.selectbox("Box plot - column", ncols, key="clean_box")
        sym = st.checkbox("Symmetric-log y axis", value=True, key="clean_sym")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
        axes[0].boxplot(A.X[bcol].dropna(), showfliers=True)
        axes[0].set_title("Before")
        axes[1].boxplot(after_all[bcol].dropna(), showfliers=True)
        axes[1].set_title("After")
        if sym:
            axes[0].set_yscale("symlog")
        show_fig(fig)
    else:
        st.info("No numeric columns to analyse.")

    st.markdown("#### Inconsistencies (categories / text)")
    txt_cols = [c for c in df_raw.columns if c not in (target, *drop_cols) and cl.is_text_like(df_raw[c])
                and df_raw[c].nunique() <= 60]
    if txt_cols:
        tcol = st.selectbox("Column", txt_cols, key="clean_txt")
        l, r = st.columns(2)
        l.markdown("**Raw values**")
        l.dataframe(arrow_safe(df_raw[tcol].value_counts(dropna=False).head(20).rename("rows").reset_index()))
        r.markdown("**After cleaning**")
        if tcol in A.struct.columns:
            r.dataframe(arrow_safe(A.struct[tcol].value_counts(dropna=False).head(20).rename("rows").reset_index()))
        takeaway("Variants such as `TN`, `tn`, ` Tn ` or `Yes`/`Y`/`1` are merged into one value, and tokens like "
                 "`N/A` or `?` become real missing values handled by imputation.")
    else:
        st.info("No low-cardinality text columns found.")
    if A.log.get("boolean_columns"):
        st.caption("Yes/No-like columns converted to 0/1: " + ", ".join(A.log["boolean_columns"]))

# --------------------------------------------------------------------------
# 4. Transformation & feature engineering
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Transformation & feature engineering")
    st.markdown("**Pipeline steps applied (in order):**")
    for i, s in enumerate(B.steps, 1):
        st.markdown(f"{i}. {s}")

    st.markdown("#### Engineered features")
    eng_rows = []
    for kind, names in A.fe_log.items():
        for n in names:
            eng_rows.append({"type": kind.replace("_", " "), "column": n})
    if eng_rows:
        st.dataframe(pd.DataFrame(eng_rows))
        new_in_X = [r["column"] for r in eng_rows if r["column"] in A.X.columns][:12]
        if new_in_X:
            show_df(A.X[new_in_X], 10)
        takeaway("Ratios (followers/following, posts/day) and text statistics (name length, digits) capture "
                 "bot behaviour much better than raw counts; missing-value flags keep the information that a "
                 "field was empty.")
    else:
        st.info("No engineered features (options disabled or columns not found).")

    st.markdown("#### Aggregation (group statistics)")
    cats_here = [c for c in A.struct.columns if c != TARGET and cl.is_text_like(A.struct[c])
                 and 1 < A.struct[c].nunique() <= 30]
    nums_here = numeric_cols(A.struct, [TARGET])
    if cats_here and nums_here:
        g1, g2 = st.columns(2)
        gcat = g1.selectbox("Group by", cats_here, key="agg_cat")
        gnum = g2.multiselect("Numeric columns", nums_here, default=nums_here[:3], key="agg_num")
        gfun = st.selectbox("Statistic", ["mean", "median", "std", "max"], key="agg_fun")
        if gnum:
            st.dataframe(tf.aggregate_table(A.struct, gcat, gnum, gfun))
    else:
        st.info("Aggregation needs at least one categorical and one numeric column.")

    st.markdown("#### Discretisation / log / encoding / scaling")
    st.write(f"- log1p applied to: {', '.join(B.log_cols) if B.log_cols else 'none'}")
    st.write(f"- binned columns: {', '.join(B.bin_cols) if B.bin_cols else 'none (choose columns in the sidebar)'}")
    st.write(f"- aggregation feature: {B.agg_col or 'none (enable it in the sidebar)'}")
    st.write(f"- categorical columns encoded ({encoding}): {', '.join(B.cat_cols) if B.cat_cols else 'none'}")
    if B.dropped_constant:
        st.write(f"- constant columns dropped: {', '.join(B.dropped_constant)}")
    cmp_cols = [c for c in B.scaled_cols if c in B.train_clean.columns][:6]
    if cmp_cols:
        st.markdown("**Effect of scaling (train set)**")
        a = B.train_clean[cmp_cols].agg(["mean", "std"]).T.add_suffix(" (before)")
        b = B.X_train[cmp_cols].agg(["mean", "std"]).T.add_suffix(" (after)")
        st.dataframe(a.join(b).round(3))
        takeaway("After scaling, columns are on comparable scales (e.g. mean ≈ 0, std ≈ 1 for standard scaling), "
                 "which matters for distance/linear models and for PCA.")
    st.markdown("**Final model-ready matrix (train)**")
    show_df(B.X_train, 10)

# --------------------------------------------------------------------------
# 5. Train / test split
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Train / test split")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train rows", f"{len(B.X_train):,}")
    c2.metric("Test rows", f"{len(B.X_test):,}")
    c3.metric("Features", f"{n_feat}")
    dist = pd.DataFrame({
        "train": B.y_train.value_counts(normalize=True).reindex([0, 1]).fillna(0) * 100,
        "test": B.y_test.value_counts(normalize=True).reindex([0, 1]).fillna(0) * 100,
    }).round(2)
    dist.index = [CLASS_NAMES[i] for i in dist.index]
    fig, ax = plt.subplots(figsize=(6, 3))
    dist.plot(kind="bar", ax=ax, color=["#6baed6", "#fd8d3c"])
    ax.set_ylabel("% of rows")
    plt.setp(ax.get_xticklabels(), rotation=0)
    show_fig(fig)
    st.dataframe(dist)
    takeaway("The split is stratified, so the fraud share is almost identical in train and test. "
             "Imputers, outlier bounds, scalers, encoders, selectors and PCA are fitted on the TRAIN set only "
             "and then applied to the test set, which avoids data leakage.")

# --------------------------------------------------------------------------
# 6. Feature selection
# --------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Feature selection (fitted on train)")
    st.caption("Scores are scaled to [0, 1] per method. Wrapper score = 1 / RFE rank.")
    vt = sel.votes_table(scores, int(k_sel))
    st.markdown(f"**Top-{k_sel} features by method: `{sel_method}`**")
    top_scores = scores[sel_method].sort_values(ascending=False).head(int(k_sel))[::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.28 * len(top_scores))))
    ax.barh(top_scores.index, top_scores.values, color="#3182bd")
    ax.set_xlabel("normalised score")
    show_fig(fig)
    takeaway(f"`{top_scores.index[-1]}` is ranked first by {sel_method}. Features chosen by several methods "
             "(see the votes column) are the most reliable.")
    st.markdown("**All methods side by side (sorted by votes)**")
    st.dataframe(vt.round(3))
    top_n = vt.head(min(25, len(vt))).drop(columns="votes (top-k)")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(top_n))))
    sns.heatmap(top_n, cmap="YlGnBu", annot=len(top_n) <= 15, fmt=".2f", ax=ax, cbar_kws={"shrink": 0.7})
    show_fig(fig)
    unanimous = [i for i, v in vt["votes (top-k)"].items() if v == len(sel.METHODS)]
    takeaway(f"{len(unanimous)} feature(s) are selected by all 5 methods"
             + (": " + ", ".join(f"`{u}`" for u in unanimous[:8]) if unanimous else "") + ".")
    st.markdown(f"**Selected feature set ({sel_method}, k={k_sel}):** " + ", ".join(f"`{s}`" for s in selected))

# --------------------------------------------------------------------------
# 7. Dimensionality reduction
# --------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Dimensionality reduction / feature extraction")

    st.markdown("#### PCA (linear projection maximising variance)")
    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_show = min(len(evr), 30)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(range(1, n_show + 1), evr[:n_show], color="#9ecae1", label="per component")
    ax.plot(range(1, n_show + 1), cum[:n_show], color="#e34a33", marker="o", label="cumulative")
    ax.axhline(0.95, color="grey", ls="--", lw=1)
    ax.set_xlabel("component")
    ax.set_ylabel("explained variance ratio")
    ax.legend()
    show_fig(fig)
    n95 = rd.components_for_variance(pca_full, 0.95)
    takeaway(f"{n95} of {n_feat} components explain 95% of the variance; the first two explain "
             f"{100 * cum[min(1, len(cum) - 1)]:.1f}%.")

    sample_n = min(3000, len(pca_tr))
    idx = np.random.default_rng(seed).choice(len(B.X_train), size=sample_n, replace=False)
    z2 = pca_full.transform(B.X_train.to_numpy())[:, :2] if n_feat >= 2 else None
    if z2 is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cls in (0, 1):
            m = B.y_train.to_numpy()[idx] == cls
            ax.scatter(z2[idx][m, 0], z2[idx][m, 1], s=8, alpha=0.5, color=COLORS[cls], label=CLASS_NAMES[cls])
        ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}%)")
        ax.legend()
        show_fig(fig)
        takeaway("If fraud and genuine accounts form separate clouds in the PC1/PC2 plane, the classes are "
                 "linearly separable in a low-dimensional space; heavy overlap means more complex models are needed.")
        l1, l2 = st.columns(2)
        l1.markdown("**Top loadings PC1**")
        l1.dataframe(rd.pca_loadings(pca_full, list(B.X_train.columns), 0))
        l2.markdown("**Top loadings PC2**")
        l2.dataframe(rd.pca_loadings(pca_full, list(B.X_train.columns), 1))

    st.markdown("#### t-SNE (non-linear, visualisation only)")
    do_tsne = st.checkbox("Compute t-SNE (a few seconds)", value=False)
    if do_tsne:
        ts_pts = st.slider("Points used", 500, 4000, 2000, 500)
        ts_perp = st.slider("Perplexity", 5, 50, 30)
        emb = run_tsne_cached(B.X_train, B.y_train, int(ts_pts), float(ts_perp), seed)
        fig, ax = plt.subplots(figsize=(7, 5))
        for cls in (0, 1):
            d = emb[emb["target"] == cls]
            ax.scatter(d["tSNE-1"], d["tSNE-2"], s=8, alpha=0.6, color=COLORS[cls], label=CLASS_NAMES[cls])
        ax.legend()
        show_fig(fig)
        takeaway("t-SNE preserves local neighbourhoods: compact fraud islands mean fraud accounts resemble each "
                 "other. Distances between clusters are not meaningful, so use it only for visual exploration.")

    st.markdown("#### Autoencoder (neural-network feature extraction)")
    do_ae = st.checkbox("Train autoencoder", value=False)
    if do_ae:
        a1, a2 = st.columns(2)
        ae_dim = a1.slider("Bottleneck size (extracted features)", 2, max(2, min(16, n_feat)), min(4, max(2, n_feat)))
        ae_iter = a2.slider("Max training iterations", 50, 500, 200, 50)
        ae_result = run_ae_cached(B.X_train, B.X_test, int(ae_dim), int(ae_iter), seed)
        m1, m2 = st.columns(2)
        m1.metric("Reconstruction MSE (train)", f"{ae_result['mse_train']:.4f}")
        m2.metric("Reconstruction MSE (test)", f"{ae_result['mse_test']:.4f}")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(ae_result["loss_curve"], color="#3182bd")
        ax.set_xlabel("iteration")
        ax.set_ylabel("training loss")
        show_fig(fig)
        zt = ae_result["z_train"]
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = np.random.default_rng(seed).choice(len(zt), size=min(3000, len(zt)), replace=False)
        yy = B.y_train.to_numpy()[sub]
        for cls in (0, 1):
            m = yy == cls
            ax.scatter(zt.iloc[sub, 0][m], zt.iloc[sub, 1][m], s=8, alpha=0.5, color=COLORS[cls],
                       label=CLASS_NAMES[cls])
        ax.set_xlabel("AE1")
        ax.set_ylabel("AE2")
        ax.legend()
        show_fig(fig)
        takeaway(f"The network compresses {n_feat} features into {ae_result['bottleneck']} learned ones. "
                 "A train MSE close to the test MSE means the compression generalises; these features can be "
                 "used for modelling (Model tab).")

# --------------------------------------------------------------------------
# 8. Model
# --------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Model & evaluation")
    sets = {"All features": (B.X_train, B.X_test),
            f"Selected features ({sel_method}, k={len(selected)})": (B.X_train[selected], B.X_test[selected]),
            f"PCA ({pca_tr.shape[1]} components)": (pca_tr, pca_te)}
    if ae_result is not None:
        sets[f"Autoencoder ({ae_result['bottleneck']} features)"] = (ae_result["z_train"], ae_result["z_test"])
    m1, m2 = st.columns(2)
    mname = m1.selectbox("Model", md.MODEL_NAMES, index=0)
    fset = m2.selectbox("Feature set", list(sets.keys()), index=0)
    if ae_result is None:
        st.caption("Tick “Train autoencoder” in the Dimensionality-reduction tab to add it as a feature set.")

    if st.button("▶ Train & evaluate"):
        Xtr, Xte = sets[fset]
        with st.spinner("Training..."):
            res = md.evaluate(mname, Xtr, B.y_train, Xte, B.y_test, seed)
        cols = st.columns(5)
        for c, (k, v) in zip(cols, res["metrics"].items()):
            c.metric(k, f"{v:.3f}" if np.isfinite(v) else "n/a")
        l, r = st.columns(2)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        l.pyplot(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.plot(res["fpr"], res["tpr"], color="#e34a33", label=f"AUC = {res['metrics']['ROC-AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend()
        r.pyplot(fig)
        plt.close(fig)
        tn, fp, fn, tp = res["cm"].ravel()
        takeaway(f"The model catches {tp} of {tp + fn} fraudulent test accounts (recall "
                 f"{res['metrics']['Recall']:.2f}) and wrongly flags {fp} genuine ones (precision "
                 f"{res['metrics']['Precision']:.2f}). In fraud detection, missing a fraud (FN) is usually "
                 "costlier than a false alarm (FP).")
        if res["importance"] is not None:
            imp = res["importance"].sort_values(ascending=False).head(15)[::-1]
            fig, ax = plt.subplots(figsize=(7, max(2.5, 0.28 * len(imp))))
            ax.barh(imp.index, imp.values, color="#31a354")
            ax.set_xlabel("importance")
            show_fig(fig)

    st.markdown("#### Compare all feature sets")
    if st.button("⚖ Compare feature sets with the chosen model"):
        rows = []
        with st.spinner("Training on every feature set..."):
            for name, (Xtr, Xte) in sets.items():
                r_ = md.evaluate(mname, Xtr, B.y_train, Xte, B.y_test, seed)
                rows.append({"feature set": name, "n features": Xtr.shape[1], **r_["metrics"]})
        comp = pd.DataFrame(rows).set_index("feature set")
        st.dataframe(comp.round(3))
        fig, ax = plt.subplots(figsize=(7, 3.4))
        comp[["Precision", "Recall", "F1"]].plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        show_fig(fig)
        bestf = comp["F1"].idxmax()
        takeaway(f"Best F1 with **{bestf}** ({comp.loc[bestf, 'F1']:.3f}) using {int(comp.loc[bestf, 'n features'])} "
                 "features. Fewer features at similar performance means a simpler, faster and more robust model.")

# --------------------------------------------------------------------------
# 9. Before vs After
# --------------------------------------------------------------------------
with tabs[9]:
    st.subheader("Before vs After cleaning")
    raw_missing = int(df_raw.isna().sum().sum()) + int(A.log.get("tokens_to_nan", 0))
    left, right = st.columns(2)
    with left:
        st.markdown("### 🔴 Before (raw)")
        st.metric("Rows", f"{len(df_raw):,}")
        st.metric("Missing cells (incl. 'N/A', '?')", f"{raw_missing:,}")
        st.metric("Duplicate rows", f"{int(df_raw.duplicated().sum()):,}")
    with right:
        st.markdown("### 🟢 After (cleaned)")
        st.metric("Rows", f"{len(clean_all):,}")
        st.metric("Missing cells", f"{int(clean_all.drop(columns=TARGET).isna().sum().sum()):,}")
        st.metric("Duplicate rows", f"{int(clean_all.drop(columns=TARGET).duplicated().sum()):,}")

    common = [c for c in numeric_cols(A.struct, [TARGET]) if c in clean_all.columns]
    if common:
        ccol = st.selectbox("Compare a column", common, key="ba_col")
        log_ba = st.checkbox("log1p x axis", value=True, key="ba_log")
        bef = pd.to_numeric(A.struct[ccol], errors="coerce").dropna()
        aft = clean_all[ccol].dropna()
        if log_ba:
            bef, aft = np.log1p(bef.clip(lower=0)), np.log1p(aft.clip(lower=0))
        fl, fr = st.columns(2)
        for holder, data, title, color in ((fl, bef, "Before", "#e34a33"), (fr, aft, "After", "#31a354")):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.hist(data, bins=40, color=color, alpha=0.8)
            ax.set_title(f"{title}: {ccol}")
            holder.pyplot(fig)
            plt.close(fig)
        stats = pd.DataFrame({
            "before": pd.to_numeric(A.struct[ccol], errors="coerce").describe(),
            "after": clean_all[ccol].describe(),
        }).round(2)
        st.dataframe(stats)
        takeaway(f"For `{ccol}` the mean moves from {fmt(stats.loc['mean', 'before'])} to "
                 f"{fmt(stats.loc['mean', 'after'])} and the max from {fmt(stats.loc['max', 'before'])} to "
                 f"{fmt(stats.loc['max', 'after'])}: imputation fills the gaps and outlier handling tames extremes.")

    st.markdown("### 📥 Downloads")
    d1, d2 = st.columns(2)
    d1.download_button("Download cleaned CSV (original scale)", clean_all.to_csv().encode("utf-8"),
                       file_name="cleaned_dataset.csv", mime="text/csv")
    d2.download_button("Download model-ready CSV (encoded + scaled)", final_all.to_csv().encode("utf-8"),
                       file_name="model_ready_dataset.csv", mime="text/csv")
    st.caption("The raw file in data/raw/ is never overwritten: every transformation is done in memory.")


# --------------------------------------------------------------------------
# 10. MLOps / Versioning (use case 2)
# --------------------------------------------------------------------------
with tabs[10]:
    import json

    st.subheader("MLOps: project structure, params.yaml and DVC data versioning")
    st.caption("Everything in this tab is read from files of the project folder.")
    params_path, dvc_path = ROOT / "params.yaml", ROOT / "dvc.yaml"
    metrics_path, valid_path = ROOT / "reports" / "metrics.json", ROOT / "reports" / "validation.json"
    info_path = ROOT / "data" / "processed" / "info.json"

    left, right = st.columns(2)
    with left:
        st.markdown("#### params.yaml (all settings in one file)")
        if params_path.exists():
            st.code(params_path.read_text(encoding="utf-8"), language="yaml")
        else:
            st.info("params.yaml not found.")
    with right:
        st.markdown("#### dvc.yaml (pipeline stages)")
        if dvc_path.exists():
            st.code(dvc_path.read_text(encoding="utf-8"), language="yaml")
        else:
            st.info("dvc.yaml not found.")
        st.markdown("**Stages:** 1-data_validation → 2-prepare_data → 3-train")

    st.markdown("#### Results of the last `dvc repro`")
    c1, c2, c3 = st.columns(3)
    for col, path, title in ((c1, valid_path, "Data validation"), (c2, info_path, "Prepared data"),
                             (c3, metrics_path, "Model metrics")):
        with col:
            st.markdown(f"**{title}**")
            if path.exists():
                st.json(json.loads(path.read_text(encoding="utf-8")))
            else:
                st.caption("Not generated yet: run `dvc repro` in the terminal.")
    takeaway("Changing a value in params.yaml and running `dvc repro` re-runs only the stages that depend on "
             "it. `dvc params diff` and `dvc metrics diff` then show exactly what changed between two versions.")


# --------------------------------------------------------------------------
# 11. Data versions (use case 3): DVC + Git driven from the app
# --------------------------------------------------------------------------
with tabs[11]:
    st.subheader("Dataset versioning: the data goes to DVC, the metadata to Git, the version gets a tag")
    status = cached_status(REV)

    def chip(ok: bool, label: str) -> str:
        return ("✅ " if ok else "❌ ") + label

    sc = st.columns(5)
    sc[0].write(chip(status["git_installed"], "Git installed"))
    sc[1].write(chip(status["dvc_installed"], "DVC installed"))
    sc[2].write(chip(status["git_repo"], "git init done"))
    sc[3].write(chip(status["dvc_repo"], "dvc init done"))
    sc[4].write(chip(status["git_user"], "Git user.name / email"))
    st.caption(("DVC remote: " + ("configured" if status["dvc_remote"] else "none (optional, needed for dvc push)"))
               + " · Git remote: " + ("configured" if status["git_remote"] else "none (optional)"))
    if st.button("🔄 Refresh"):
        st.session_state["versions_rev"] += 1
        st.rerun()

    if not status["ready"]:
        st.warning("The project is not ready yet. Run these commands in the VS Code terminal (project folder), "
                   "then press Refresh:")
        st.code("git init\ndvc init\ngit config user.name \"Your Name\"\ngit config user.email you@mail.com",
                language="bash")
    else:
        versions = cached_versions(REV)

        # ---------------- create a version ----------------
        st.markdown("### 1. Save the current result as a new version")
        last = st.session_state.get("last_save")
        if last:
            if last["success"]:
                st.success(f"Version **{last['tag']}** recorded: {last['rows']:,} rows x {last['columns']} columns, "
                           f"DVC hash `{last['md5']}`.")
                if last.get("push_requested") and not last.get("push_success"):
                    st.warning(f"The version is saved locally, but pushing failed: {last['push_error']}")
                elif last.get("push_requested"):
                    st.info("The Git commit and tag were pushed to the configured remote.")
                if last.get("push_skipped"):
                    st.caption("Not pushed: " + "; ".join(last["push_skipped"]) + ".")
            else:
                st.error(last["error"])
            with st.expander("Commands executed by Python"):
                for stp in last["steps"]:
                    st.markdown(("✅ " if stp["ok"] else "❌ ") + f"`{stp['command']}`")
                    if stp["output"]:
                        st.caption(stp["output"])

        content = st.selectbox("What do you want to save?", [
            "Model-ready data (all features, encoded + scaled)",
            "Model-ready data (selected features only)",
            "Cleaned data (original scale, readable)",
        ])
        suggested = vz.suggest_next_tag([v["tag"] for v in versions])
        c1, c2 = st.columns(2)
        tag = c1.text_input("Version tag", value=suggested, key=f"tag_{REV}",
                            help="Format: v1.1, v1.2, v2.0 ...")
        msg = c2.text_input("Commit message", value="", key=f"msg_{REV}",
                            placeholder="e.g. median imputation + one-hot, 15 selected features")
        can_push = status["dvc_remote"] or status["git_remote"]
        do_push = st.checkbox("Also push DVC data and the Git commit/tag to configured remotes",
                              value=status["dvc_remote"] or status["git_remote"],
                              disabled=not can_push)
        if not status["git_remote"]:
            st.info("No Git remote is configured. A saved version and tag will stay local until you add a GitHub remote.")

        def build_version_df(choice: str) -> pd.DataFrame:
            if choice.startswith("Model-ready data (all"):
                d = final_all.copy()
            elif choice.startswith("Model-ready data (selected"):
                d = final_all[list(selected) + [TARGET]].copy()
            else:
                d = clean_all.copy()
            d["split"] = np.where(d.index.isin(B.X_test.index), "test", "train")
            out_name = target if target not in d.columns else f"{target}_label"
            return d.rename(columns={TARGET: out_name})

        preview_df = build_version_df(content)
        st.caption(f"This version will contain {len(preview_df):,} rows x {preview_df.shape[1]} columns "
                   "(including a `split` column = train/test).")
        params_record = {
            "dataset": dataset_name, "target": target, "positive_label": positive,
            "dropped_columns": list(drop_cols), "content": content,
            "cleaning_and_features": cfg_a, "pipeline": cfg_b,
            "feature_selection": {"method": sel_method, "k": int(k_sel)},
            "pca_components": int(pca_n), "pipeline_steps": list(B.steps),
        }
        if st.button("💾 Save as new version", type="primary"):
            with st.spinner("dvc add -> git commit -> git tag ..."):
                res = vz.version_data(preview_df, params_record, tag.strip(), msg, do_push)
            st.session_state["last_save"] = res
            if res["success"]:
                st.session_state["versions_rev"] += 1
                st.rerun()
            else:
                st.error(res["error"])
                with st.expander("Commands executed by Python"):
                    for stp in res["steps"]:
                        st.markdown(("✅ " if stp["ok"] else "❌ ") + f"`{stp['command']}`")
                        if stp["output"]:
                            st.caption(stp["output"])

        # ---------------- browse versions ----------------
        st.markdown("### 2. Browse and load any past version")
        if not versions:
            st.info("No version yet. Create v1.0 in the terminal (see the guide) or save a version above.")
        else:
            st.dataframe(pd.DataFrame(versions))
            labels = {v["tag"]: f"{v['tag']} - {v['message']}" for v in versions}
            pick = st.selectbox("Version to open", list(labels), format_func=labels.get, key="ver_pick")
            try:
                df_v = cached_load_version(pick)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                df_v = None
            info = cached_version_info(pick, REV)
            if df_v is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows", f"{len(df_v):,}")
                m2.metric("Columns", f"{df_v.shape[1]}")
                m3.metric("Created", (info or {}).get("created_at", "n/a"))
                st.markdown("**Parameters that produced this version**")
                if info:
                    st.json(info["params"], expanded=False)
                else:
                    st.caption("No recorded parameters (this version was created manually in the terminal).")
                show_df(df_v, 100)
                st.download_button(f"Download {pick} as CSV", df_v.to_csv(index=False).encode("utf-8"),
                                   file_name=f"dataset_{pick}.csv", mime="text/csv")

            if len(versions) >= 2:
                st.markdown("### 3. Compare two versions")
                cmp1, cmp2 = st.columns(2)
                tags_list = list(labels)
                va = cmp1.selectbox("Version A", tags_list, index=min(1, len(tags_list) - 1), key="cmp_a")
                vb = cmp2.selectbox("Version B", tags_list, index=0, key="cmp_b")
                try:
                    da, db = cached_load_version(va), cached_load_version(vb)
                    ca, cb = set(da.columns), set(db.columns)
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Rows A -> B", f"{len(da):,} -> {len(db):,}", delta=len(db) - len(da))
                    k2.metric("Columns A -> B", f"{da.shape[1]} -> {db.shape[1]}", delta=db.shape[1] - da.shape[1])
                    k3.metric("Columns only in B", len(cb - ca))
                    if ca - cb:
                        st.write("Only in A:", ", ".join(sorted(map(str, ca - cb))[:30]))
                    if cb - ca:
                        st.write("Only in B:", ", ".join(sorted(map(str, cb - ca))[:30]))
                    pd_ = vz.params_diff(cached_version_info(va, REV), cached_version_info(vb, REV))
                    st.markdown("**Parameters that differ**")
                    if len(pd_):
                        st.dataframe(pd_)
                    else:
                        st.caption("No difference in the recorded parameters (or none recorded).")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        takeaway("Each version = a DVC hash of the data (stored in the DVC cache/remote) + a small .dvc file, the "
                 "recorded parameters and a tag in Git. Any tag can be reloaded exactly as it was.")
