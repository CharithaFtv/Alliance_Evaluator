"""Elkan-Noto PU wrapper around a probabilistic classifier.

g(x) = P(s=1 | x) is learned on positive-vs-unlabeled labels; the label
frequency c = P(s=1 | y=1) is estimated on HELD-OUT positives, and
P(y=1 | x) ~= min(g(x) / c, 1). Ranking is identical for g and g / c.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def default_estimator() -> LogisticRegression:
    return LogisticRegression(C=0.1, max_iter=2000, solver="liblinear", random_state=0)


def estimate_label_frequency(estimator, X, s, n_splits=5, seed=0) -> float:
    """Mean out-of-fold g(x) over known positives."""
    g = np.zeros(len(s))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, s):
        g[te] = clone(estimator).fit(X[tr], s[tr]).predict_proba(X[te])[:, 1]
    return float(g[s == 1].mean())


class ElkanNotoPU(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, n_splits=5, seed=0):
        self.estimator = estimator
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, X, s):
        s = np.asarray(s)
        base = self.estimator or default_estimator()
        self.c_ = estimate_label_frequency(base, X, s, self.n_splits, self.seed)
        self.estimator_ = clone(base).fit(X, s)
        return self

    def score_labeled(self, X) -> np.ndarray:
        """g(x): P(labeled | x)."""
        return self.estimator_.predict_proba(X)[:, 1]

    def predict_proba_positive(self, X) -> np.ndarray:
        """Estimated P(y=1 | x), clipped to [0, 1]."""
        return np.clip(self.score_labeled(X) / self.c_, 0.0, 1.0)
