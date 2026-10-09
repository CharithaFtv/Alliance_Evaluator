"""Input validation and the compact audit report (no raw company content)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cleaning
from . import config as C


def validate_ids(df: pd.DataFrame) -> None:
    ids = df[C.ID_COLUMN]
    missing = ids.map(cleaning.is_missing)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} rows have missing {C.ID_COLUMN}")
    if ids.duplicated().any():
        raise ValueError(f"{int(ids.duplicated().sum())} duplicated {C.ID_COLUMN} values")


def parse_labels(df: pd.DataFrame) -> pd.Series:
    raw = df[C.LABEL_COLUMN].astype(str).str.strip()
    bad = ~raw.isin(["0", "1"])
    if bad.any():
        raise ValueError(f"Invalid {C.LABEL_COLUMN} values: {sorted(raw[bad].unique())[:10]}")
    return raw.astype(int)


def _missing_counts(df: pd.DataFrame) -> dict:
    return {c: int(df[c].map(cleaning.is_missing).sum()) for c in df.columns}


def _unit_flags(raw: pd.DataFrame) -> list[str]:
    """Heuristic flags only. Values are never rescaled or assigned currencies."""
    flags = []
    for col in ["REVENUE", "TOTAL_CAPITAL_RAISED", "MOST_RECENT_FUNDING_AMT"]:
        vals = pd.Series([cleaning.parse_number(v)[0] for v in raw[col]]).dropna()
        vals = vals[vals > 0]
        if vals.empty:
            continue
        small = float((vals < 1_000).mean())
        flags.append(
            f"{col}: unit/currency not documented; median={vals.median():,.0f}, "
            f"max={vals.max():.3g}, {small:.0%} of positive values < 1,000 "
            "(possible mixed units, e.g. millions vs. units). Needs clarification."
        )
    return flags


def build_audit(raw: pd.DataFrame, labels: pd.Series, clean: cleaning.CleanResult) -> dict:
    age = pd.Series([cleaning.parse_number(v)[0] for v in raw["COMPANY_AGE"]])
    year = pd.Series([cleaning.parse_number(v)[0] for v in raw["YEAR_FOUNDED"]])
    both = age.notna() & year.notna()
    age_mismatch = int(((age + year) != C.REFERENCE_YEAR)[both].sum())
    per_subsector = raw.groupby("SUBSECTOR")[["TAM_B", "SAM_B"]].nunique().max()
    return {
        "rows": int(len(raw)),
        "unique_ids": int(raw[C.ID_COLUMN].nunique()),
        "label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "positive_rate": round(float(labels.mean()), 4),
        "label_note": "0 = unlabeled, NOT confirmed negative",
        "missing_by_column": _missing_counts(raw),
        "numeric_issues": clean.numeric_issues,
        "text_parse_failures": clean.text_parse_failures,
        "empty_business_text_rows": int((clean.frame[C.TEXT_COLUMN].str.len() == 0).sum()),
        "company_age": {
            "rows_checked": int(both.sum()),
            "age_plus_year_founded_ne_%d" % C.REFERENCE_YEAR: age_mismatch,
            "age_above_%d_flagged_not_altered" % C.SUSPECT_AGE_MAX: int((age > C.SUSPECT_AGE_MAX).sum()),
            "note": f"COMPANY_AGE assumed derived relative to {C.REFERENCE_YEAR}",
        },
        "unit_flags": _unit_flags(raw),
        "tam_sam_note": (
            "TAM_B/SAM_B max distinct values within one SUBSECTOR: "
            f"{int(per_subsector.max())} -> they are subsector-level constants, "
            "redundant with SUBSECTOR; units (billions, currency) unconfirmed."
            if int(per_subsector.max()) == 1 else "TAM_B/SAM_B vary within subsector."
        ),
        "zero_employees_policy": "NUM_OF_EMPLOYEES == 0 treated as missing (baseline)",
        "funding_policy": "MOST_RECENT_FUNDING_AMT excluded (low coverage); missing never zero-filled",
    }
