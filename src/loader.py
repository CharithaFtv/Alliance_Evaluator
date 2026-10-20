"""Step 1: load flagship_ets_only.csv, repair broken rows, build labels and money columns."""
import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "historical_data" / "flagship_ets_only.csv"

# Dropped by explicit user instruction. Nothing else is dropped.
DROP_COLS = ["TAM_B", "SAM_B", "SUBTYPES", "FTV_STRATEGY", "CONSOLIDATED_SECTOR",
             "VALUATION", "PB_EBITDA", "LABEL"]

_RECORD_START = re.compile(r"^\d+,\d{5,},", re.M)


def _read_records(path):
    """Split on record starts (ROW_ID,ID,) so one bad quote cannot swallow neighbours."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    first_nl = text.index("\n")
    header = next(csv.reader(io.StringIO(text[:first_nl])))
    body = text[first_nl + 1:]
    starts = [m.start() for m in _RECORD_START.finditer(body)] + [len(body)]
    csv.field_size_limit(10**9)
    good, bad = [], []
    for a, b in zip(starts, starts[1:]):
        chunk = body[a:b].rstrip("\r\n")
        row = next(csv.reader(io.StringIO(chunk)), [])
        if len(row) != len(header):  # retry with a closing quote for an unterminated field
            row = next(csv.reader(io.StringIO(chunk + '"')), [])
        (good if len(row) == len(header) else bad).append(row if len(row) == len(header) else chunk[:80])
    return header, good, bad


def load_raw(path=DATA):
    header, rows, bad = _read_records(path)
    df = pd.DataFrame(rows, columns=header).replace("", np.nan).infer_objects()
    return df, bad


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def build(path=DATA):
    df, bad = load_raw(path)
    # Quarantined: EMPLOYEE_COUNT_BY_MONTH holds corrupt binary data, so later columns
    # (incl. label flags) cannot be recovered. Ask for a re-export of these IDs.
    report = {"quarantined_ids": [b.split(",")[1] for b in bad], "rows_raw": len(df)}

    df = df.drop_duplicates(subset="ID", keep="first")
    report["rows_after_id_dedupe"] = len(df)

    df = df[df["TYPE"] != "Capital Provider"].copy()
    report["rows_after_type_filter"] = len(df)

    # Labels: overlap rows are positive AND negative (one flag each); neither = unknown.
    df["y_pos"] = (_num(df["IS_LABELED_POSITIVE"]) == 1).astype(int)
    df["y_neg"] = (_num(df["IS_LABELED_NEGATIVE"]) == 1).astype(int)
    df["group"] = np.select(
        [(df.y_pos == 1) & (df.y_neg == 1), df.y_pos == 1, df.y_neg == 1],
        ["overlap", "positive_only", "negative_only"], "unknown")

    df = add_money_cols(df)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df.reset_index(drop=True), report


def add_money_cols(df):
    """Money in $M: *_UPDATED first, else AI_EXTRACTED_* (already $M); raw columns ignored."""
    df = df.copy()
    for new, upd, ai in [("revenue_m", "REVENUE_UPDATED", "AI_EXTRACTED_REVENUE"),
                         ("capital_raised_m", "TOTAL_CAPITAL_RAISED_UPDATED",
                          "AI_EXTRACTED_TOTAL_CAPITAL_RAISED")]:
        u = _num(df[upd]) if upd in df else pd.Series(np.nan, index=df.index)
        a = _num(df[ai]) if ai in df else pd.Series(np.nan, index=df.index)
        df[new] = u.fillna(a)
        df[new + "_src"] = np.select([u.notna(), a.notna()], ["updated", "ai_extracted"], "missing")
    return df


if __name__ == "__main__":
    d, rep = build()
    print(rep)
    print(d["group"].value_counts().to_string())
    print(d["TYPE"].value_counts().to_string())
    print(d["revenue_m_src"].value_counts().to_string())
    print(d["capital_raised_m_src"].value_counts().to_string())
    print(d.shape)
