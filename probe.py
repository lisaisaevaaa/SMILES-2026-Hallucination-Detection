from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class HallucinationProbe:
    def __init__(self) -> None:
        self._lr:    LogisticRegression | None = None
        self._sc:    StandardScaler = StandardScaler()
        self._sc_cl: StandardScaler = StandardScaler()
        self._pca:   PCA | None = None
        self._km:    KMeans | None = None
        self._hc:    int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        self._sc = StandardScaler().fit(X)
        self._lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        self._lr.fit(self._sc.transform(X), y)

        n_comp = min(50, X.shape[1] - 1, X.shape[0] - 1)
        self._sc_cl = StandardScaler().fit(X)
        self._pca   = PCA(n_components=n_comp, random_state=42).fit(
            self._sc_cl.transform(X)
        )
        Z = self._pca.transform(self._sc_cl.transform(X))
        self._km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(Z)

        c = self._km.labels_
        self._hc = max([0, 1], key=lambda ci: float(y[c == ci].mean()))
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        return self

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        return self._km.predict(self._pca.transform(self._sc_cl.transform(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self._lr.predict(self._sc.transform(X)).copy()
        pred[self._cluster(X) == self._hc] = 1
        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self._lr.predict_proba(self._sc.transform(X)).copy()
        proba[self._cluster(X) == self._hc] = [0.0, 1.0]
        return proba
