"""Model training + evaluation."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.tree import DecisionTreeClassifier

MODEL_NAMES = ["Random Forest", "Logistic Regression", "Decision Tree", "Gradient Boosting"]


def make_model(name: str, seed: int = 42):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=seed)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def evaluate(name: str, X_train, y_train, X_test, y_test, seed: int = 42) -> dict:
    model = make_model(name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    try:
        auc = float(roc_auc_score(y_test, proba))
        fpr, tpr, _ = roc_curve(y_test, proba)
    except ValueError:
        auc, fpr, tpr = float("nan"), np.array([0, 1]), np.array([0, 1])
    metrics = {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1": float(f1_score(y_test, pred, zero_division=0)),
        "ROC-AUC": auc,
    }
    importance = None
    cols = list(X_train.columns) if hasattr(X_train, "columns") else None
    if cols is not None:
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=cols)
        elif hasattr(model, "coef_"):
            importance = pd.Series(np.abs(model.coef_[0]), index=cols)
    return {
        "model": model, "metrics": metrics, "cm": confusion_matrix(y_test, pred, labels=[0, 1]),
        "fpr": fpr, "tpr": tpr, "importance": importance,
    }
