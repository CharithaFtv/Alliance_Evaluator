"""Step 3: feature pipeline. Row-wise parsing is stateless; everything with learned state
(clipping, imputing, scaling, vocabularies, TF-IDF/SVD) lives in sklearn steps fit on TRAIN only."""
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

REF_YEAR = 2026

# Columns that must never reach X (label derivation, identifiers, user-ignored raw money,
# AI_EXTRACTED_*: fill rate is label-skewed, see report). They stay in the dataframe.
FORBIDDEN = {
    "ID", "ROW_ID", "NAME", "WEBSITE", "LINKEDIN_URL", "WEBSITE_URL",
    "ACTIVITY_COUNT", "DAYS_SINCE_LAST_CONTACT", "PARTNER_REVIEW", "NEGATIVE_FLAGS",
    "IS_LABELED_POSITIVE", "IS_LABELED_NEGATIVE", "y_pos", "y_neg", "group", "block",
    "role", "cv_fold", "REVENUE", "TOTAL_CAPITAL_RAISED", "revenue_m_src",
    "capital_raised_m_src",
} | {f"AI_EXTRACTED_{s}" for s in [
    "REVENUE", "REVENUE_RANGE", "TOTAL_CAPITAL_RAISED", "PQ_SUB_INDUSTRY", "FUNDING_AMT",
    "FUNDING_STAGE", "PB_EBITDA", "VALUATION"]}

OWNERSHIP_MAP = {"Private Sub": "Private Subsidiary", "Public Sub": "Public Subsidiary"}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _json(x):
    try:
        return json.loads(x) if isinstance(x, str) else []
    except ValueError:
        return []


def _headcount(x):
    pts = [(p["MONTH"], p["COUNT"]) for p in _json(x) if isinstance(p, dict) and "COUNT" in p]
    if len(pts) < 3:
        return dict(hc_latest=np.nan, hc_g12=np.nan, hc_g36=np.nan, hc_slope=np.nan,
                    hc_hist_m=len(pts), hc_peak_ratio=np.nan)
    c = np.array([v for _, v in sorted(pts)], dtype=float)
    last = c[-1]
    g = lambda k: (last - c[-1 - k]) / max(c[-1 - k], 1) if len(c) > k else np.nan
    t = np.arange(min(len(c), 36))
    slope = np.polyfit(t, np.log1p(c[-len(t):]), 1)[0]
    return dict(hc_latest=last, hc_g12=g(12), hc_g36=g(36), hc_slope=slope,
                hc_hist_m=len(c), hc_peak_ratio=last / max(c.max(), 1))


def _countries(x):
    d = [(p.get("COUNTRY"), float(p["PCT"])) for p in _json(x)
         if isinstance(p, dict) and p.get("PCT") is not None]
    if not d:
        return dict(ec_top_pct=np.nan, ec_n=np.nan, ec_us_pct=np.nan)
    pct = np.array([v for _, v in d])
    return dict(ec_top_pct=pct.max(), ec_n=int((pct >= 5).sum()),
                ec_us_pct=sum(v for k, v in d if k in ("united states", "usa")))


def _tokens_json_names(x):
    return [str(p["NAME"]).strip().lower() for p in _json(x) if isinstance(p, dict) and "NAME" in p]


def _tokens_json_list(x):
    return [str(t).strip().lower() for t in _json(x) if isinstance(t, str)]


def _split(sep):
    return lambda x: [t.strip().lower() for t in x.split(sep)] if isinstance(x, str) else []


MULTI = {  # column -> (token parser, min_df)
    "TAGS": (_tokens_json_list, 25), "GR_PRODUCTS_AND_SERVICES": (_tokens_json_list, 20),
    "GR_END_CUSTOMER": (_split(","), 20), "VERTICALS": (_tokens_json_names, 20),
    "PQ_SUB_INDUSTRY": (_split(";"), 20), "INVESTORS": (_tokens_json_names, 8),
}
NUM_COLS = ["age", "NUM_OF_EMPLOYEES", "revenue_m", "capital_raised_m", "last_funding_m",
            "GR_SIX_MONTH_EMPLOYEE_GROWTH_PCT", "GR_ONE_YEAR_EMPLOYEE_GROWTH_PCT",
            "PDL_SIX_MONTH_EMPLOYEE_CHURN_PCT", "PDL_ONE_YEAR_EMPLOYEE_CHURN_PCT",
            "hc_latest", "hc_g12", "hc_g36", "hc_slope", "hc_hist_m", "hc_peak_ratio",
            "ec_top_pct", "ec_n", "ec_us_pct", "n_investors", "n_tags", "desc_len",
            "revenue_per_emp_k"]
LOG_COLS = ["NUM_OF_EMPLOYEES", "revenue_m", "capital_raised_m", "last_funding_m", "hc_latest",
            "revenue_per_emp_k"]
MISSING_FLAGS = ["capital_raised_m", "last_funding_m", "hc_latest", "ec_top_pct",
                 "PDL_ONE_YEAR_EMPLOYEE_CHURN_PCT", "GR_ONE_YEAR_EMPLOYEE_GROWTH_PCT"]
CAT_COLS = ["COUNTRY", "STATE", "OWNERSHIP", "TYPE", "IS_BOOTSTRAPPED", "SUBSECTOR",
            "MAIN_INDUSTRY", "MOST_RECENT_FUNDING_STAGE", "REVENUE_RANGE"]
TEXT_COL = "text"
# Parsed into X (kept for reporting/baselines) but never given to the model, by user decision:
# subsector information should come from description/tags/products text instead.
NOT_FEATURES = {"SUBSECTOR"}


def engineer(df):
    """Stateless row-wise parsing -> modelling frame. No statistics learned here."""
    o = pd.DataFrame(index=df.index)
    yr = _num(df["YEAR_FOUNDED"]).where(lambda s: (s >= 1850) & (s <= REF_YEAR))
    o["age"] = REF_YEAR - yr
    for c in ["NUM_OF_EMPLOYEES", "revenue_m", "capital_raised_m",
              "GR_SIX_MONTH_EMPLOYEE_GROWTH_PCT", "GR_ONE_YEAR_EMPLOYEE_GROWTH_PCT",
              "PDL_SIX_MONTH_EMPLOYEE_CHURN_PCT", "PDL_ONE_YEAR_EMPLOYEE_CHURN_PCT"]:
        o[c] = _num(df[c])
    o["last_funding_m"] = _num(df["MOST_RECENT_FUNDING_AMT"]) / 1e6  # raw $ in source
    o = o.join(pd.DataFrame(df["EMPLOYEE_COUNT_BY_MONTH"].map(_headcount).tolist(), index=df.index))
    o = o.join(pd.DataFrame(df["EMPLOYEE_PCT_BY_COUNTRY"].map(_countries).tolist(), index=df.index))
    o["n_investors"] = df["INVESTORS"].map(lambda x: len(_tokens_json_names(x)))
    o["n_tags"] = df["TAGS"].map(lambda x: len(_tokens_json_list(x)))
    o["desc_len"] = df["DESCRIPTION"].fillna("").str.len()
    o["revenue_per_emp_k"] = o["revenue_m"] * 1000 / o["NUM_OF_EMPLOYEES"].where(o["NUM_OF_EMPLOYEES"] > 0)
    for c in CAT_COLS:
        s = df[c].astype("string").str.strip()
        if c == "OWNERSHIP":
            s = s.replace(OWNERSHIP_MAP)
        if c == "REVENUE_RANGE":
            s = s.str.replace("–", "-", regex=False).str.replace(r"\$0- ", "$0-", regex=True)
        o[c] = s.fillna("missing")
    o[TEXT_COL] = (df["DESCRIPTION"].fillna("") + " " + df["MAIN_INDUSTRY"].fillna("")).str.lower()
    for c, (parser, _) in MULTI.items():
        o[c] = df[c].map(parser)
    return o


class QuantileClip(BaseEstimator, TransformerMixin):
    """Learns [q_lo, q_hi] per column on train; clips everything (incl. test/unknown) to it."""
    def __init__(self, lo=0.01, hi=0.99):
        self.lo, self.hi = lo, hi

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lo, axis=0)
        self.hi_ = np.nanquantile(X, self.hi, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, names=None):
        return np.asarray(names)


def _log(X):
    return np.log1p(np.clip(X, 0, None))


def _is_missing(X):
    return np.isnan(np.asarray(X, float)).astype(float)


def _identity(t):
    return t


def build_features(n_svd=60, seed=42):
    num_steps = lambda log: [("clip", QuantileClip())] + (
        [("log", FunctionTransformer(_log, feature_names_out="one-to-one"))] if log else []) + [
        ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    plain = [c for c in NUM_COLS if c not in LOG_COLS]
    parts = [
        ("num", Pipeline(num_steps(False)), plain),
        ("num_log", Pipeline(num_steps(True)), LOG_COLS),
        ("miss", Pipeline([("ind", FunctionTransformer(_is_missing,
                                                       feature_names_out="one-to-one"))]), MISSING_FLAGS),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                              sparse_output=False), [c for c in CAT_COLS if c not in NOT_FEATURES]),
        ("text", Pipeline([("tfidf", TfidfVectorizer(min_df=5, max_df=0.5, ngram_range=(1, 2),
                                                     stop_words="english", sublinear_tf=True)),
                           ("svd", TruncatedSVD(n_svd, random_state=seed))]), TEXT_COL),
    ]
    for c, (_, mdf) in MULTI.items():
        parts.append((f"mh_{c}", CountVectorizer(analyzer=_identity, binary=True, min_df=mdf), c))
    return ColumnTransformer(parts, sparse_threshold=0.0)


def assert_no_forbidden(cols):
    bad = FORBIDDEN & set(cols)
    assert not bad, f"forbidden columns in X: {bad}"


if __name__ == "__main__":
    from loader import build
    data, _ = build()
    sp = pd.read_csv("artifacts/splits.csv", dtype={"ID": str})
    data = data.merge(sp[["ID", "role", "cv_fold"]], on="ID", how="left")
    assert data["role"].notna().all()
    X = engineer(data)
    assert_no_forbidden(X.columns)
    tr = X[data.role == "train"]
    ct = build_features().fit(tr)                      # TRAIN ONLY
    out = {r: ct.transform(X[data.role == r]) for r in ["train", "test", "unknown"]}
    names = ct.get_feature_names_out()
    for r, m in out.items():
        print(r, m.shape, "NaN:", int(np.isnan(m).sum()))
    from collections import Counter
    print(Counter(n.split("__")[0] for n in names))
    print("zero-variance on train:", int((out["train"].std(0) == 0).sum()))
    pd.Series(names).to_csv("artifacts/feature_names.csv", index=False, header=["feature"])
