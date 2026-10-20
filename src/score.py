"""Step 6b: score NEW companies. Input: CSV with the same raw column names as the training file
(only external fields are read; internal/label columns are ignored if present).

    python src/score.py new_companies.csv out.csv
Output per company: fit score (calibrated like prob), percentile vs sourced universe, dislike score,
likely reject reasons, top drivers up/down, data completeness, and flags.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit

from features import CAT_COLS, assert_no_forbidden, engineer
from loader import add_money_cols

import os

# production refit (train + test) is used when present; the pre-refit train-only model is the fallback
ART = "artifacts/production" if os.path.exists("artifacts/production/models.joblib") else "artifacts"
KEY_FIELDS = ["DESCRIPTION", "YEAR_FOUNDED", "NUM_OF_EMPLOYEES", "revenue_m", "capital_raised_m",
              "COUNTRY", "SUBSECTOR", "INVESTORS", "MOST_RECENT_FUNDING_STAGE", "EMPLOYEE_COUNT_BY_MONTH"]
REQUIRED_RAW = ["DESCRIPTION", "TAGS", "YEAR_FOUNDED", "OWNERSHIP", "NUM_OF_EMPLOYEES", "STATE", "COUNTRY",
                "REVENUE_UPDATED", "TOTAL_CAPITAL_RAISED_UPDATED", "MOST_RECENT_FUNDING_AMT", "TYPE", "SUBSECTOR",
                "GR_PRODUCTS_AND_SERVICES", "VERTICALS", "MAIN_INDUSTRY", "GR_END_CUSTOMER", "EMPLOYEE_COUNT_BY_MONTH",
                "IS_BOOTSTRAPPED", "EMPLOYEE_PCT_BY_COUNTRY", "MOST_RECENT_FUNDING_STAGE", "PQ_SUB_INDUSTRY",
                "GR_SIX_MONTH_EMPLOYEE_GROWTH_PCT", "GR_ONE_YEAR_EMPLOYEE_GROWTH_PCT",
                "PDL_SIX_MONTH_EMPLOYEE_CHURN_PCT", "PDL_ONE_YEAR_EMPLOYEE_CHURN_PCT", "INVESTORS", "REVENUE_RANGE"]


FRIENDLY = {
    "n_investors": "number of investors", "hc_slope": "headcount trend", "hc_hist_m": "length of headcount history",
    "hc_g12": "12-month headcount growth", "hc_g36": "3-year headcount growth", "hc_latest": "current headcount",
    "hc_peak_ratio": "headcount vs its peak", "revenue_m": "revenue", "capital_raised_m": "capital raised",
    "last_funding_m": "last funding round size", "age": "company age", "desc_len": "description length",
    "ec_top_pct": "largest country share of staff", "ec_n": "number of staffed countries",
    "ec_us_pct": "US share of staff", "revenue_per_emp_k": "revenue per employee", "n_tags": "number of tags",
    "NUM_OF_EMPLOYEES": "employee count", "GR_SIX_MONTH_EMPLOYEE_GROWTH_PCT": "6-month employee growth",
    "GR_ONE_YEAR_EMPLOYEE_GROWTH_PCT": "1-year employee growth",
    "PDL_SIX_MONTH_EMPLOYEE_CHURN_PCT": "6-month employee churn",
    "PDL_ONE_YEAR_EMPLOYEE_CHURN_PCT": "1-year employee churn",
}


def readable(name):
    kind, _, rest = name.partition("__")
    rest = FRIENDLY.get(rest, rest).replace("_infrequent_sklearn", " (rare value)").replace("RANGE_", "")
    if kind == "cat":
        for col in sorted(CAT_COLS, key=len, reverse=True):  # column names contain underscores
            if rest.startswith(col + "_"):
                return f"{col.lower().replace('_', ' ')} = {rest[len(col) + 1:]}"
        return rest
    if kind.startswith("mh_"):
        return f"{kind[3:].lower().replace('_', ' ')}: {rest}"
    if kind == "miss":
        return f"{rest} is missing"
    if kind == "text":
        return "description wording"
    return rest.replace("_", " ")


def top_drivers(expl, X, k=4):
    """Additive LR contributions (coef x standardised value) aggregated per readable feature."""
    Z = expl.named_steps["feat"].transform(X)
    names = expl.named_steps["feat"].get_feature_names_out()
    C = Z * expl.named_steps["est"].coef_[0]
    pos, neg = [], []
    for row in C:
        o = np.argsort(row)
        neg.append("; ".join(readable(names[i]) for i in o[:k] if row[i] < -0.05))
        pos.append("; ".join(readable(names[i]) for i in o[::-1][:k] if row[i] > 0.05))
    return pos, neg


def score(raw):
    raw = raw.copy()
    for c in REQUIRED_RAW:
        if c not in raw:
            raw[c] = np.nan
    raw = add_money_cols(raw)
    X = engineer(raw)
    assert_no_forbidden(X.columns)
    art = joblib.load(f"{ART}/models.joblib")
    expl = joblib.load(f"{ART}/explainer.joblib")
    ref = joblib.load(f"{ART}/reference_like_raw.joblib")
    like_raw = art["like"].predict_proba(X)[:, 1]
    p = art["cal"].predict_proba(logit(np.clip(like_raw, 1e-4, 1 - 1e-4)).reshape(-1, 1))[:, 1]
    out = pd.DataFrame(index=raw.index)
    if "NAME" in raw:
        out["NAME"] = raw["NAME"]
    out["fit_score"] = p.round(3)
    out["percentile_vs_universe"] = (np.searchsorted(ref, like_raw) / len(ref) * 100).round(0)
    out["dislike_score"] = art["dislike"].predict_proba(X)[:, 1].round(3)
    why = pd.DataFrame({f: m.predict_proba(X)[:, 1] for f, m in art["reasons"].items()}, index=raw.index)
    out["likely_reject_reason"] = why.idxmax(axis=1)
    out["reject_reason_prob"] = why.max(axis=1).round(2)
    out["drivers_up"], out["drivers_down"] = top_drivers(expl, X)
    present = pd.DataFrame({c: raw[c].notna() if c in raw else False for c in KEY_FIELDS})
    out["data_completeness"] = present.mean(axis=1).round(2)
    out["flag"] = np.where(out.data_completeness < 0.7, "low data: treat score with caution", "")
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    score(pd.read_csv(src, low_memory=False)).to_csv(dst, index=False)
    print("wrote", dst)
