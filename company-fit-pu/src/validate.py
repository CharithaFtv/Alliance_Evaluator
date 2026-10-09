"""Score the held-out validation file with the frozen preprocessor + model.

Run from company-fit-pu/:  python -m src.validate
Nothing is refit. Writes <tag>_metrics.json and <tag>_scores.csv (tag defaults to 'validation').
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from . import audit, evaluate, pipeline
from . import config as C
from .preprocess import load_raw

DEFAULT_VALIDATION = C.ROOT / "data" / "flagship_ets_validation.csv"


def run(input_path: Path, art_dir: Path, tag: str = "validation") -> dict:
    raw = load_raw(input_path)
    audit.validate_ids(raw)
    s = audit.parse_labels(raw).to_numpy()
    pre = pipeline.load_artifacts(art_dir)["preprocessor"]
    model = joblib.load(art_dir / "pu_model.joblib")

    X, ids = pipeline.transform(pre, raw)
    g = model.score_labeled(X)
    out = ids.to_frame(C.ID_COLUMN)
    out[C.LABEL_COLUMN] = s
    out["g_score"] = g
    out["p_positive"] = model.predict_proba_positive(X)
    out.to_csv(art_dir / f"{tag}_scores.csv", index=False)

    report = {"rows": len(s), "metrics": evaluate.ranking_metrics(g, s)}
    (art_dir / f"{tag}_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_VALIDATION)
    p.add_argument("--tag", default="validation")
    p.add_argument("--artifacts-dir", type=Path, default=C.DEFAULT_OUTPUT_DIR)
    a = p.parse_args()
    print(json.dumps(run(a.input, a.artifacts_dir, a.tag), indent=2))


if __name__ == "__main__":
    main()
