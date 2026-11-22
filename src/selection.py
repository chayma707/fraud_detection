"""Feature selection: filter, wrapper and embedded methods (all fitted on TRAIN only)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

METHODS = [
    "Filter: correlation",
    "Filter: mutual information",
    "Wrapper: RFE",
    "Embedded: LASSO",
    "Embedded: Decision Tree",
]


def _normalise(s: pd.Series) -> pd.Series:
    s = s.astype(float).fillna(0.0)
    m = s.max()
    return s / m if m > 0 else s


def compute_feature_scores(X: pd.DataFrame, y: pd.Series, k: int, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame (features x methods) with scores scaled to [0, 1]."""
    names = list(X.columns)
    n_feat = len(names)
    k = max(1, min(int(k), n_feat))
    out = pd.DataFrame(index=names)

    # --- Filter: correlation with the target
    out[METHODS[0]] = _normalise(X.corrwith(y.astype(float)).abs())

    # --- Filter: mutual information
    mi = mutual_info_classif(X.to_numpy(), y.to_numpy(), random_state=seed)
    out[METHODS[1]] = _normalise(pd.Series(mi, index=names))

    # --- Wrapper: Recursive Feature Elimination (logistic regression)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = LogisticRegression(max_iter=2000, class_weight="balanced")
        rfe = RFE(est, n_features_to_select=k, step=max(1, int(0.1 * n_feat)))
        rfe.fit(X.to_numpy(), y.to_numpy())
    out[METHODS[2]] = pd.Series(1.0 / rfe.ranking_, index=names)

    # --- Embedded: LASSO (L1 regularisation, alpha chosen by cross-validation)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = LassoCV(cv=5, random_state=seed, max_iter=10000)
        lasso.fit(X.to_numpy(), y.to_numpy().astype(float))
    out[METHODS[3]] = _normalise(pd.Series(np.abs(lasso.coef_), index=names))

    # --- Embedded: Decision-tree importance
    tree = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=seed)
    tree.fit(X.to_numpy(), y.to_numpy())
    out[METHODS[4]] = _normalise(pd.Series(tree.feature_importances_, index=names))
    return out


def top_k(scores: pd.DataFrame, method: str, k: int) -> list[str]:
    k = max(1, min(int(k), len(scores)))
    return list(scores[method].sort_values(ascending=False).head(k).index)


def votes_table(scores: pd.DataFrame, k: int) -> pd.DataFrame:
    """How many of the 5 methods put each feature in their top-k."""
    votes = pd.Series(0, index=scores.index, dtype=int)
    for m in METHODS:
        votes.loc[top_k(scores, m, k)] += 1
    out = scores.copy()
    out.insert(0, "votes (top-k)", votes)
    return out.sort_values(["votes (top-k)", METHODS[3]], ascending=False)
