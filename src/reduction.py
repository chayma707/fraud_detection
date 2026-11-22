"""Dimensionality reduction / feature extraction: PCA, t-SNE, Autoencoder."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------
def run_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, n_components: int, seed: int = 42):
    """Fit a full PCA on TRAIN, return it with the first `n_components` scores for train/test."""
    full = PCA(random_state=seed).fit(X_train.to_numpy())
    n = max(1, min(int(n_components), full.n_components_))
    cols = [f"PC{i + 1}" for i in range(n)]
    z_tr = pd.DataFrame(full.transform(X_train.to_numpy())[:, :n], index=X_train.index, columns=cols)
    z_te = pd.DataFrame(full.transform(X_test.to_numpy())[:, :n], index=X_test.index, columns=cols)
    return full, z_tr, z_te


def components_for_variance(pca: PCA, threshold: float = 0.95) -> int:
    cum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cum, threshold) + 1)


def pca_loadings(pca: PCA, feature_names: list[str], pc: int = 0, top: int = 10) -> pd.DataFrame:
    load = pd.Series(pca.components_[pc], index=feature_names)
    idx = load.abs().sort_values(ascending=False).head(top).index
    return pd.DataFrame({"feature": idx, f"loading PC{pc + 1}": load.loc[idx].round(3).values})


# --------------------------------------------------------------------------
# t-SNE (visualisation only)
# --------------------------------------------------------------------------
def run_tsne(X: pd.DataFrame, y: pd.Series, max_points: int = 2000, perplexity: float = 30.0, seed: int = 42):
    n = len(X)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
    else:
        idx = np.arange(n)
    Xs = X.iloc[idx]
    ys = y.iloc[idx]
    perp = float(min(perplexity, max(2.0, (len(Xs) - 1) / 3.0)))
    tsne = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed)
    emb = tsne.fit_transform(Xs.to_numpy())
    return pd.DataFrame({"tSNE-1": emb[:, 0], "tSNE-2": emb[:, 1], "target": ys.to_numpy()})


# --------------------------------------------------------------------------
# Autoencoder (neural network, pure scikit-learn: input -> h -> bottleneck -> h -> input)
# --------------------------------------------------------------------------
class SimpleAutoencoder:
    def __init__(self, bottleneck: int = 4, max_iter: int = 200, seed: int = 42):
        self.bottleneck = int(bottleneck)
        self.max_iter = int(max_iter)
        self.seed = int(seed)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        k = max(1, min(self.bottleneck, d))
        h = max(8, min(64, 2 * d))
        self.k_ = k
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(h, k, h), activation="tanh", solver="adam",
            max_iter=self.max_iter, random_state=self.seed, learning_rate_init=1e-3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(X, X)
        return self

    def encode(self, X) -> np.ndarray:
        a = np.asarray(X, dtype=float)
        for W, b in zip(self.model_.coefs_[:2], self.model_.intercepts_[:2]):
            a = np.tanh(a @ W + b)
        return a

    def reconstruction_mse(self, X) -> float:
        X = np.asarray(X, dtype=float)
        return float(np.mean((X - self.model_.predict(X)) ** 2))

    @property
    def loss_curve(self):
        return self.model_.loss_curve_


def run_autoencoder(X_train: pd.DataFrame, X_test: pd.DataFrame, bottleneck: int, max_iter: int, seed: int = 42):
    ae = SimpleAutoencoder(bottleneck, max_iter, seed).fit(X_train)
    cols = [f"AE{i + 1}" for i in range(ae.k_)]
    z_tr = pd.DataFrame(ae.encode(X_train), index=X_train.index, columns=cols)
    z_te = pd.DataFrame(ae.encode(X_test), index=X_test.index, columns=cols)
    return {
        "z_train": z_tr, "z_test": z_te, "loss_curve": list(ae.loss_curve),
        "mse_train": ae.reconstruction_mse(X_train), "mse_test": ae.reconstruction_mse(X_test),
        "bottleneck": ae.k_,
    }
