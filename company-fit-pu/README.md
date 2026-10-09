# company-fit-pu

Positive-Unlabeled (PU) learning for company fit scoring.

## Layout

| Path | Purpose |
|------|---------|
| `data/` | Raw inputs (e.g. `flagship_ets_train.csv`); gitignored |
| `src/preprocess.py` | Load, clean, and split data; each step is an independent function |
| `artifacts/` | Generated outputs (processed data, models); gitignored |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m src.preprocess --input data/flagship_ets_train.csv --output artifacts/train_processed.csv
```
