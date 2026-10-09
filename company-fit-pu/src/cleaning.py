"""Deterministic, stateless cleaning. Safe to apply to any future dataset.

Nothing here learns from data; fitted statistics live only in the sklearn
preprocessor. Raw input frames are never mutated.
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in C.MISSING_TOKENS


# ---------- numeric ----------
def parse_number(value) -> tuple[float, str | None]:
    """Return (value, issue). issue in {None, 'unparsable', 'nonfinite', 'negative'}."""
    if is_missing(value):
        return np.nan, None
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", "").lstrip("$")
        number = float(value)
    except (TypeError, ValueError):
        return np.nan, "unparsable"
    if not math.isfinite(number):
        return np.nan, "nonfinite"
    if number < 0:
        return np.nan, "negative"
    return number, None


def clean_numeric(series: pd.Series, zero_as_missing: bool = False) -> tuple[pd.Series, dict]:
    parsed = [parse_number(v) for v in series]
    values = pd.Series([p[0] for p in parsed], index=series.index, dtype="float64")
    issues: dict = {"unparsable": 0, "nonfinite": 0, "negative": 0, "zero_set_missing": 0}
    examples: list[str] = []
    for raw, (_, issue) in zip(series, parsed):
        if issue:
            issues[issue] += 1
            if issue == "unparsable" and len(examples) < 5:
                examples.append(str(raw))
    if zero_as_missing:
        zeros = values == 0
        issues["zero_set_missing"] = int(zeros.sum())
        values[zeros] = np.nan
    issues["unparsable_examples"] = examples
    return values, issues


# ---------- text ----------
def parse_text_value(value) -> tuple[str, bool]:
    """Flatten native arrays or serialized JSON/list strings. Returns (text, ok)."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return " ".join(str(v).strip() for v in value if not is_missing(v)), True
    if is_missing(value):
        return "", True
    text = str(value).strip()
    if text[:1] in "[{":
        for loader in (json.loads, ast.literal_eval):
            try:
                loaded = loader(text)
            except (ValueError, SyntaxError):
                continue
            if isinstance(loaded, dict):
                loaded = list(loaded.values())
            if isinstance(loaded, (list, tuple)):
                return parse_text_value(list(loaded))[0], True
        return text, False  # looks serialized but could not be parsed
    return text, True


def build_business_text(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    parts, failures = [], {}
    for col in C.TEXT_SOURCES:
        parsed = [parse_text_value(v) for v in df[col]]
        parts.append(pd.Series([p[0] for p in parsed], index=df.index))
        failures[col] = int(sum(not p[1] for p in parsed))
    text = pd.concat(parts, axis=1).apply(lambda r: " ".join(t for t in r if t), axis=1)
    return text.astype(str), failures


# ---------- categorical ----------
def clean_categorical(series: pd.Series) -> pd.Series:
    def norm(v):
        return C.MISSING_CATEGORY if is_missing(v) else " ".join(str(v).split()).lower()

    return series.map(norm)


# ---------- frame ----------
@dataclass
class CleanResult:
    frame: pd.DataFrame  # ID + model inputs (numeric, categorical, BUSINESS_TEXT)
    numeric_issues: dict = field(default_factory=dict)
    text_parse_failures: dict = field(default_factory=dict)


def clean_frame(df: pd.DataFrame) -> CleanResult:
    """Raw frame -> model-input frame, row order preserved."""
    out = pd.DataFrame({C.ID_COLUMN: df[C.ID_COLUMN].astype(str)}, index=df.index)
    issues = {}
    for col in C.NUMERIC:
        out[col], issues[col] = clean_numeric(df[col], col in C.ZERO_AS_MISSING)
        if col in C.MONEY_CUTOFF_MILLIONS:
            as_units = out[col] >= C.MONEY_CUTOFF_MILLIONS[col]
            issues[col]["already_millions"] = int((out[col] < C.MONEY_CUTOFF_MILLIONS[col]).sum())
            out.loc[as_units, col] /= 1e6
    for col in C.CATEGORICAL:
        out[col] = clean_categorical(df[col])
    out[C.TEXT_COLUMN], failures = build_business_text(df)
    return CleanResult(out, issues, failures)
