"""Preprocessing pipeline: load -> clean -> save.

Each step is a standalone function so steps can be reused, tested, or swapped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "flagship_ets_train.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "train_processed.csv"


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Compose cleaning steps. Add new steps here."""
    for step in (normalize_columns, drop_duplicates):
        df = step(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = preprocess(load_data(args.input))
    save_data(df, args.output)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
