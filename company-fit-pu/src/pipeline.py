"""Build, fit, persist and reload the sklearn preprocessing pipeline."""
from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import cleaning
from . import config as C

PREPROCESSOR_FILE = "preprocessor.joblib"
CONFIG_FILE = "feature_config.json"
MATRIX_FILE = "X_train.npz"
ROW_MAP_FILE = "row_mapping.csv"
FEATURE_FILE = "feature_names.txt"
AUDIT_FILE = "audit_report.json"
VERSIONS_FILE = "package_versions.json"


def split_nonempty(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    """Fit-time only: drop numeric columns with no observed values."""
    kept = [c for c in columns if frame[c].notna().any()]
    return kept, [c for c in columns if c not in kept]


def _numeric_pipeline(log: bool) -> Pipeline:
    steps = []
    if log:
        steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    return Pipeline(steps)


def build_preprocessor(log_cols: list[str], linear_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if log_cols:
        transformers.append(("num_log", _numeric_pipeline(True), log_cols))
    if linear_cols:
        transformers.append(("num", _numeric_pipeline(False), linear_cols))
    transformers += [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), C.CATEGORICAL),
        ("text", TfidfVectorizer(**C.TFIDF_PARAMS), C.TEXT_COLUMN),  # str => 1-D input
    ]
    return ColumnTransformer(transformers, sparse_threshold=1.0, remainder="drop")


def fit_preprocessor(frame: pd.DataFrame) -> tuple[ColumnTransformer, dict]:
    log_cols, dropped_log = split_nonempty(frame, C.LOG_NUMERIC)
    lin_cols, dropped_lin = split_nonempty(frame, C.LINEAR_NUMERIC)
    preprocessor = build_preprocessor(log_cols, lin_cols).fit(frame)
    config = {
        "log_numeric": log_cols,
        "linear_numeric": lin_cols,
        "dropped_empty_numeric": dropped_log + dropped_lin,
        "categorical": C.CATEGORICAL,
        "text_source_columns": C.TEXT_SOURCES,
        "text_column": C.TEXT_COLUMN,
        "tfidf_params": {k: list(v) if isinstance(v, tuple) else v for k, v in C.TFIDF_PARAMS.items()},
        "reference_year": C.REFERENCE_YEAR,
        "zero_as_missing": C.ZERO_AS_MISSING,
        "excluded": C.AUDIT_ONLY,
    }
    return preprocessor, config


def transform(preprocessor: ColumnTransformer, raw: pd.DataFrame) -> tuple[sp.csr_matrix, pd.Series]:
    """Clean a raw frame and apply an ALREADY FITTED preprocessor (never refits)."""
    frame = cleaning.clean_frame(raw).frame
    X = sp.csr_matrix(preprocessor.transform(frame))
    return X, frame[C.ID_COLUMN]


def package_versions() -> dict:
    names = ["pandas", "numpy", "scipy", "scikit-learn", "joblib"]
    return {"python": platform.python_version(), **{n: metadata.version(n) for n in names}}


def save_artifacts(out_dir: Path, preprocessor, config, X, ids, labels, feature_names, audit) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_dir / PREPROCESSOR_FILE)
    sp.save_npz(out_dir / MATRIX_FILE, X)
    pd.DataFrame({C.ID_COLUMN: ids.values, C.LABEL_COLUMN: labels.values}).to_csv(
        out_dir / ROW_MAP_FILE, index_label="row_index"
    )
    (out_dir / FEATURE_FILE).write_text("\n".join(feature_names) + "\n")
    (out_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2))
    (out_dir / AUDIT_FILE).write_text(json.dumps(audit, indent=2))
    (out_dir / VERSIONS_FILE).write_text(json.dumps(package_versions(), indent=2))


def load_artifacts(out_dir: Path) -> dict:
    return {
        "preprocessor": joblib.load(out_dir / PREPROCESSOR_FILE),
        "X": sp.load_npz(out_dir / MATRIX_FILE),
        "row_mapping": pd.read_csv(out_dir / ROW_MAP_FILE, dtype={C.ID_COLUMN: str}),
        "feature_names": (out_dir / FEATURE_FILE).read_text().splitlines(),
        "config": json.loads((out_dir / CONFIG_FILE).read_text()),
    }
