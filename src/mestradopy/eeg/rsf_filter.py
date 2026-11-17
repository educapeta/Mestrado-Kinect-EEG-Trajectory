# -*- coding: utf-8 -*-
"""Riemannian Geometry-Based Spatial Filtering (RSF).

Implementacao baseada na formulacao de Pan et al., "Enhancing Motor Imagery
EEG Classification with a Riemannian Geometry-Based Spatial Filtering (RSF)
Method", Neural Networks 188 (2025), doi:10.1016/j.neunet.2025.107511.

A ideia: estimar as covariancias medias de cada classe no manifold de
matrizes SPD e achar W que maximiza a distancia riemanniana entre
W^T C1 W e W^T C2 W. O filtro e ajustado SOMENTE nos dados de treino.

Uso:
    filt = RSFFilter(dim=3)
    filt.fit(X_train_trials, y_train)  # X: (n_trials, n_channels, n_times)
    Xtr = filt.transform(X_train_windows)
    Xte = filt.transform(X_test_windows)

A implementacao usa pyriemann para covariancias/medias riemannianas. O
objetivo e identico ao `optimizeRiemann` do repositorio oficial:
    - sum_i log(lambda_i)^2,
onde lambda_i sao os autovalores generalizados de (W^T C1 W, W^T C2 W).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.spatialfilters import CSP


def _objective(w_flat, c1, c2, n_channels):
    w = w_flat.reshape(n_channels, -1)
    a = w.T @ c1 @ w
    b = w.T @ c2 @ w
    vals = eigh(a, b, eigvals_only=True)
    vals = np.clip(vals, 1e-12, None)
    return -np.sum(np.log(vals) ** 2)


class RSFFilter(BaseEstimator, TransformerMixin):
    """Filtro espacial Riemanniano para EEG.

    Parameters
    ----------
    dim : int
        numero de componentes de saida (<= n_channels).
    method : {"default", "rsf-csp"}
        "default" inicializa W aleatoriamente; "rsf-csp" inicializa com CSP.
    solver : str
        solver do scipy.optimize.minimize.
    maxiter : int
        iteracoes maximas.
    random_state : int
        semente da inicializacao.
    """

    def __init__(self, dim: int = 3, method: str = "default",
                 solver: str = "trust-constr", maxiter: int = 1000,
                 random_state: int = 42):
        self.dim = int(dim)
        self.method = method
        self.solver = solver
        self.maxiter = int(maxiter)
        self.random_state = int(random_state)
        self.W_ = None
        self.class_covs_ = None
        self.distance_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_channels = X.shape[1]
        if not (1 <= self.dim <= n_channels):
            raise ValueError("dim deve estar entre 1 e n_channels")
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError("RSFFilter implementado para 2 classes")
        rng = np.random.default_rng(self.random_state)
        w0 = rng.standard_normal((n_channels, n_channels))
        w0, _ = np.linalg.qr(w0)
        w0 = w0[:, :self.dim]

        cov = covariances(X, estimator="lwf")
        c1 = mean_covariance(cov[y == labels[0]], metric="riemann")
        c2 = mean_covariance(cov[y == labels[1]], metric="riemann")
        self.class_covs_ = (c1, c2)

        if self.method == "rsf-csp":
            csp = CSP(nfilter=self.dim, metric="euclid")
            csp.fit(cov, y)
            w0 = csp.filters_.T[:, :self.dim]

        res = minimize(
            _objective, w0.ravel(), args=(c1, c2, n_channels),
            method=self.solver,
            options={"maxiter": self.maxiter, "gtol": 1e-8},
        )
        candidates = [w0, res.x.reshape(n_channels, -1)]
        best = min(candidates,
                   key=lambda w: _objective(w.ravel(), c1, c2, n_channels))
        self.W_ = best
        a = best.T @ c1 @ best
        b = best.T @ c2 @ best
        vals = np.clip(eigh(a, b, eigvals_only=True), 1e-12, None)
        self.distance_ = float(np.sqrt(np.sum(np.log(vals) ** 2)))
        return self

    def transform(self, X):
        if self.W_ is None:
            raise RuntimeError("RSFFilter nao ajustado")
        X = np.asarray(X)
        # (n_trials, canais, tempo) -> (n_trials, dim, tempo)
        return np.einsum("ij,kjl->kil", self.W_.T, X)
