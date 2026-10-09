# company-fit-pu

Positive-Unlabeled (PU) company-fit ranking. Pilot: Flagship fund, ETS sector.
**Stage: baseline trained (Elkan-Noto logistic regression); validated on the held-out recent-period file.**

`IS_LABELED_POSITIVE`: 1 = known positive, 0 = **unlabeled** (not a confirmed negative).

## Layout

| Path | Purpose |
|------|---------|
| `src/config.py` | Column roles, TF-IDF settings, missing tokens, reference year |
| `src/cleaning.py` | Stateless deterministic cleaning (numeric, text, categorical) |
| `src/audit.py` | ID/label validation and audit report |
| `src/pipeline.py` | Build/fit/save/load sklearn `ColumnTransformer`; `transform()` for new data |
| `src/preprocess.py` | CLI: fit on training file, save artifacts, verify |
| `data/`, `artifacts/`, `work/` | Inputs and generated outputs; gitignored |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # from the repo root
```

## Run (from `company-fit-pu/`)

```bash
python -m src.preprocess --input data/flagship_ets_training.csv --output-dir artifacts
```

Writes to `artifacts/`: `preprocessor.joblib`, `feature_config.json`, `X_train.npz`,
`row_mapping.csv` (ID, label), `feature_names.txt`, `audit_report.json`,
`package_versions.json`. The run fails if row alignment, finiteness, sparsity, or
artifact reload checks fail.

## Scoring future data (never refit)

```python
from pathlib import Path
from src import pipeline
from src.preprocess import load_raw
pre = pipeline.load_artifacts(Path("artifacts"))["preprocessor"]
X, ids = pipeline.transform(pre, load_raw(Path("data/new_companies.csv")))
```

## Baseline decisions

- Missing tokens: `\N`, `\\N`, empty, `nan`, `none`, `null`. Missing is never zero-filled.
- `NUM_OF_EMPLOYEES == 0` is treated as missing. Negative/non-finite numbers become missing and are counted.
- log1p on employees, revenue, capital raised; median impute + standard scale. No missingness indicators.
- `COMPANY_AGE` is used as given (derived relative to 2026; consistency is audited).
- Money columns (`REVENUE`, `TOTAL_CAPITAL_RAISED`) are normalised to $M; sources mix $M and raw $, split by a per-column cutoff (`MONEY_CUTOFF_MILLIONS`).
- Excluded: `YEAR_FOUNDED`, `MOST_RECENT_FUNDING_AMT`, `OWNERSHIP` (sourcing artefact: separates positives in training, not validation), `REVENUE_RANGE` (duplicates `REVENUE`), names, URLs.

## Train / validate

```bash
python -m src.preprocess && python -m src.train && python -m src.validate
```

`train` uses contiguous-block CV (file is clustered by source). `validate` scores the frozen model on `data/flagship_ets_validation.csv`; never tune on it.
Last result: block-CV AUC 0.80, validation AUC 0.663 (AP 0.32, recall@20% 36%).
