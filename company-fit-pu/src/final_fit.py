"""Production fit: refit preprocessing + model on training AND validation rows.

Run from company-fit-pu/:  python -m src.final_fit
Writes to artifacts_final/ (the train-only artifacts/ stay untouched so the
validation result remains reproducible). Score the test file afterwards with
python -m src.validate --input data/flagship_ets_test.csv --artifacts-dir artifacts_final --tag test
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config as C
from . import preprocess, train
from .preprocess import load_raw

TRAIN = C.ROOT / "data" / "flagship_ets_training.csv"
VALIDATION = C.ROOT / "data" / "flagship_ets_validation.csv"
COMBINED = C.ROOT / "work" / "train_plus_validation.csv"
OUT_DIR = C.ROOT / "artifacts_final"


def main() -> None:
    parts = [load_raw(TRAIN), load_raw(VALIDATION)]
    combined = pd.concat(parts, ignore_index=True)
    assert not combined[C.ID_COLUMN].duplicated().any(), "ID overlap between train and validation"
    COMBINED.parent.mkdir(exist_ok=True)
    combined.to_csv(COMBINED, index=False)
    pre = preprocess.run(COMBINED, OUT_DIR)
    print("rows", pre["rows"], "labels", pre["label_counts"], "matrix", pre["matrix"])
    rep = train.run(OUT_DIR, n_splits=10, seed=0)
    print(json.dumps(rep["pooled_oof"], indent=2), "\nc:", round(rep["final_model_c_estimate"], 3))


if __name__ == "__main__":
    main()
