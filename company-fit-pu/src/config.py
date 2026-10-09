"""Single source of truth for column roles and preprocessing settings."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "flagship_ets_training.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts"

ID_COLUMN = "ID"
LABEL_COLUMN = "IS_LABELED_POSITIVE"  # 1 = known positive, 0 = UNLABELED (not negative)

# Numerics. Log-scaled columns get log1p; the rest are only imputed + scaled.
LOG_NUMERIC = ["NUM_OF_EMPLOYEES", "REVENUE", "TOTAL_CAPITAL_RAISED"]
LINEAR_NUMERIC = ["COMPANY_AGE", "TAM_B", "SAM_B"]
NUMERIC = LOG_NUMERIC + LINEAR_NUMERIC
ZERO_AS_MISSING = ["NUM_OF_EMPLOYEES"]  # baseline choice: 0 employees = unknown

# Money columns are modelled in $M. Sources mix $M and raw $ (a sparse gap separates
# them), so values below the cutoff are taken as already in $M, the rest as $.
MONEY_CUTOFF_MILLIONS = {"TOTAL_CAPITAL_RAISED": 50_000, "REVENUE": 1_000}

# OWNERSHIP and REVENUE_RANGE are excluded: OWNERSHIP separates positives from
# unlabeled in training but not validation (sourcing artefact); REVENUE_RANGE
# duplicates REVENUE.
CATEGORICAL = ["COUNTRY", "SUBSECTOR"]

TEXT_SOURCES = ["DESCRIPTION", "TAGS", "GR_PRODUCTS_AND_SERVICES"]
TEXT_COLUMN = "BUSINESS_TEXT"

# Excluded from model inputs: YEAR_FOUNDED (redundant with COMPANY_AGE),
# MOST_RECENT_FUNDING_AMT (low coverage), names/URLs (linking only).
AUDIT_ONLY = ["YEAR_FOUNDED", "MOST_RECENT_FUNDING_AMT", "OWNERSHIP", "REVENUE_RANGE"]

REFERENCE_YEAR = 2026  # COMPANY_AGE was derived relative to this year
SUSPECT_AGE_MAX = 150  # ages above this are flagged, not altered

# Raw tokens treated as missing (compared lower-cased, stripped). "\N" is the
# database NULL marker observed in the training export (numeric columns carry
# it escaped as "\\N").
MISSING_TOKENS = {"", "\\n", "\\\\n", "nan", "none", "null"}
MISSING_CATEGORY = "__missing__"

TFIDF_PARAMS = dict(
    lowercase=True,
    strip_accents="unicode",
    min_df=2,
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
