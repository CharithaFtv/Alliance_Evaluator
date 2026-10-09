"""Train the PU baseline from saved preprocessing artifacts.

Run from company-fit-pu/:  python -m src.train
CV here is contiguous-block CV inside the training file; the separate
validation file is scored by src.validate. Test data is never touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import KFold

from . import config as C
from . import evaluate, pipeline, pu_model

MODEL_FILE = "pu_model.joblib"
METRICS_FILE = "train_cv_metrics.json"
OOF_FILE = "train_oof_scores.csv"


def cross_validate(X, s, n_splits: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    """Out-of-fold scores over CONTIGUOUS blocks (the file is clustered by source,
    so shuffled folds overstate performance); each fold re-estimates c."""
    oof = np.zeros(len(s))
    folds = []
    for tr, te in KFold(n_splits).split(X, s):
        model = pu_model.ElkanNotoPU(seed=seed).fit(X[tr], s[tr])
        oof[te] = model.score_labeled(X[te])
        folds.append({**evaluate.ranking_metrics(oof[te], s[te]), "c_estimate": model.c_})
    return oof, folds


def summarize(folds: list[dict]) -> dict:
    keys = [k for k in folds[0]]
    return {k: {"mean": float(np.mean([f[k] for f in folds])),
                "std": float(np.std([f[k] for f in folds]))} for k in keys}


def run(art_dir: Path, n_splits: int, seed: int) -> dict:
    art = pipeline.load_artifacts(art_dir)
    X = art["X"].tocsr()
    mapping = art["row_mapping"]
    s = mapping[C.LABEL_COLUMN].to_numpy()

    oof, folds = cross_validate(X, s, n_splits, seed)
    model = pu_model.ElkanNotoPU(seed=seed).fit(X, s)

    joblib.dump(model, art_dir / MODEL_FILE)
    out = mapping[[C.ID_COLUMN, C.LABEL_COLUMN]].copy()
    out["oof_score"] = oof
    out.to_csv(art_dir / OOF_FILE, index=False)

    report = {
        "n_splits": n_splits,
        "pooled_oof": evaluate.ranking_metrics(oof, s),
        "folds_mean_std": summarize(folds),
        "final_model_c_estimate": model.c_,
        "note": "CV inside training only; preprocessing was fit on all training rows. "
                "Unlabeled != negative, so absolute precision is understated.",
    }
    (art_dir / METRICS_FILE).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=C.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run(args.artifacts_dir, args.n_splits, args.seed)
    print(json.dumps(report["pooled_oof"], indent=2))
    print("c estimate (final):", round(report["final_model_c_estimate"], 3))


if __name__ == "__main__":
    main()
