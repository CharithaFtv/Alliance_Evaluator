"""PU-aware ranking metrics. Unlabeled rows are NOT negatives, so absolute
precision is understated; read these as relative ranking quality.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TOP_FRACTIONS = (0.05, 0.10, 0.20)


def recall_at_top(scores, s, fraction: float) -> float:
    """Share of known positives captured in the top `fraction` of ranked rows."""
    k = max(1, int(round(fraction * len(s))))
    top = np.argsort(-scores, kind="stable")[:k]
    return float(s[top].sum() / s.sum())


def ranking_metrics(scores, s) -> dict:
    s = np.asarray(s)
    out = {
        "roc_auc_pos_vs_unlabeled": float(roc_auc_score(s, scores)),
        "average_precision": float(average_precision_score(s, scores)),
        "positive_base_rate": float(s.mean()),
    }
    for f in TOP_FRACTIONS:
        out[f"recall_top_{int(f * 100)}pct"] = recall_at_top(scores, s, f)
    return out
