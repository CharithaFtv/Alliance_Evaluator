"""Fit preprocessing on the TRAINING file only and save artifacts.

Future data: use pipeline.transform(fitted_preprocessor, raw_df); never refit.
Run from company-fit-pu/:  python -m src.preprocess --input <csv> --output-dir artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import audit, cleaning, pipeline
from . import config as C


def load_raw(path: Path) -> pd.DataFrame:
    """All columns as strings, no NA coercion: preserves IDs and raw tokens."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def verify(X, ids, labels, feature_names, preprocessor, raw, out_dir: Path) -> dict:
    """Post-run checks, including a reload round trip."""
    reloaded = pipeline.load_artifacts(out_dir)
    X2, ids2 = pipeline.transform(reloaded["preprocessor"], raw)
    checks = {
        "rows_match": X.shape[0] == len(ids) == len(labels) == len(raw),
        "ids_in_input_order": bool((ids.values == raw[C.ID_COLUMN].values).all()),
        "feature_names_match_columns": X.shape[1] == len(feature_names) == len(reloaded["feature_names"]),
        "is_sparse": sp.issparse(X),
        "all_finite": bool(np.isfinite(X.data).all()),
        "saved_matrix_identical": (reloaded["X"] != X).nnz == 0,
        "saved_row_mapping_aligned": bool(
            (reloaded["row_mapping"][C.ID_COLUMN].values == ids.values).all()
            and (reloaded["row_mapping"][C.LABEL_COLUMN].values == labels.values).all()
        ),
        "reloaded_transform_identical": (X2 != X).nnz == 0 and bool((ids2.values == ids.values).all()),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"Verification failed: {failed}")
    return checks


def run(input_path: Path, out_dir: Path) -> dict:
    raw = load_raw(input_path)
    audit.validate_ids(raw)
    labels = audit.parse_labels(raw)
    clean = cleaning.clean_frame(raw)

    preprocessor, config = pipeline.fit_preprocessor(clean.frame)
    X = sp.csr_matrix(preprocessor.transform(clean.frame))
    feature_names = list(preprocessor.get_feature_names_out())
    ids = clean.frame[C.ID_COLUMN]

    report = audit.build_audit(raw, labels, clean)
    report["dropped_empty_numeric"] = config["dropped_empty_numeric"]
    report["matrix"] = {"shape": list(X.shape), "nnz": int(X.nnz)}
    pipeline.save_artifacts(out_dir, preprocessor, config, X, ids, labels, feature_names, report)
    report["verification"] = verify(X, ids, labels, feature_names, preprocessor, raw, out_dir)
    (out_dir / pipeline.AUDIT_FILE).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=C.DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=C.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run(args.input, args.output_dir)
    print(f"rows={report['rows']} labels={report['label_counts']} matrix={report['matrix']}")
    print("verification:", report["verification"])
    for flag in report["unit_flags"]:
        print("FLAG:", flag)


if __name__ == "__main__":
    main()
