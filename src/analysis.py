"""Step 5a: pattern analysis on TRAIN ONLY (grouped CV). Test rows are never loaded into any fit."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import models as M
from features import CAT_COLS, MULTI

pd.set_option("display.width", 200)
GROUPS = {
    "main_industry": ["MAIN_INDUSTRY"],
    "country_state": ["COUNTRY", "STATE", "ec_top_pct", "ec_n", "ec_us_pct"],
    "ownership": ["OWNERSHIP"], "is_bootstrapped": ["IS_BOOTSTRAPPED"], "type": ["TYPE"],
    "revenue_range": ["REVENUE_RANGE"], "revenue_size": ["revenue_m", "revenue_per_emp_k"],
    "employees": ["NUM_OF_EMPLOYEES"], "age": ["age"],
    "capital_funding": ["capital_raised_m", "last_funding_m", "MOST_RECENT_FUNDING_STAGE"],
    "headcount_trend": ["hc_latest", "hc_g12", "hc_g36", "hc_slope", "hc_hist_m", "hc_peak_ratio"],
    "growth_churn_pct": ["GR_SIX_MONTH_EMPLOYEE_GROWTH_PCT", "GR_ONE_YEAR_EMPLOYEE_GROWTH_PCT",
                         "PDL_SIX_MONTH_EMPLOYEE_CHURN_PCT", "PDL_ONE_YEAR_EMPLOYEE_CHURN_PCT"],
    "investors": ["INVESTORS", "n_investors"], "text_description": ["text", "desc_len"],
    "tags": ["TAGS", "n_tags"], "products_services": ["GR_PRODUCTS_AND_SERVICES"],
    "end_customer": ["GR_END_CUSTOMER"], "verticals": ["VERTICALS"], "pq_sub_industry": ["PQ_SUB_INDUSTRY"],
}


def neutralise(X, cols):
    X = X.copy()
    for c in cols:
        if c in MULTI or c == "text":  # shuffle rows: keeps vocabulary, destroys the signal
            p = np.random.default_rng(1).permutation(len(X))
            X[c] = [X[c].iloc[i] for i in p]
        elif c in CAT_COLS:
            X[c] = "missing"
        else:
            X[c] = np.nan
    return X


def main():
    data, X = M.load()
    tr = (data.role == "train").values
    Xt, dt = X[tr].reset_index(drop=True), data[tr].reset_index(drop=True)
    y, folds = dt.y_pos.values, dt.cv_fold
    mk = lambda: M.pipe(M.gb(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0))
    rng = np.random.default_rng(0)

    # ---- 1. grouped permutation importance (AUC drop on held-out folds, 3 repeats)
    imp = {g: [] for g in GROUPS}
    oof = np.zeros(len(y))
    for k in sorted(folds.unique()):
        va = (folds == k).values
        m = mk().fit(Xt[~va], y[~va])
        Xv = Xt[va].reset_index(drop=True)
        base = roc_auc_score(y[va], m.predict_proba(Xv)[:, 1])
        oof[va] = m.predict_proba(Xv)[:, 1]
        for g, cols in GROUPS.items():
            drops = []
            for _ in range(3):
                Xp = Xv.copy()
                p = rng.permutation(len(Xp))
                for c in cols:
                    Xp[c] = Xv[c].iloc[p].values if c not in MULTI else [Xv[c].iloc[i] for i in p]
                drops.append(base - roc_auc_score(y[va], m.predict_proba(Xp)[:, 1]))
            imp[g].append(np.mean(drops))
    pi = pd.DataFrame({"auc_drop": {g: np.mean(v) for g, v in imp.items()},
                       "sd_across_folds": {g: np.std(v) for g, v in imp.items()}}).sort_values("auc_drop", ascending=False)
    print("\n== Grouped permutation importance (Like head, train CV) ==\n", pi.round(4).to_string())
    pi.to_csv("artifacts/perm_importance.csv")

    # ---- 2. ablations (neutralise a group, refit, grouped-CV AUC)
    base_auc = roc_auc_score(y, oof)
    nf = dt.NEGATIVE_FLAGS.map(lambda s: json.loads(s) if isinstance(s, str) else {})
    svc = nf.map(lambda d: d.get("services_model", 0)).values
    hard = (y == 1) | ((dt.group.values == "negative_only") & (svc == 0))
    rows = [("full model", base_auc, roc_auc_score(y[hard], oof[hard]))]
    for name, cols in {"no OWNERSHIP": ["OWNERSHIP"], "no REVENUE_RANGE": ["REVENUE_RANGE"],
                       "no capital/funding (+missing flags)": GROUPS["capital_funding"],
                       "no geography": GROUPS["country_state"],
                       "no description text": GROUPS["text_description"],
                       "no headcount trend": GROUPS["headcount_trend"],
                       "no investors": GROUPS["investors"],
                       "no multi-valued tags/products/etc": list(MULTI),
                       "no main_industry": ["MAIN_INDUSTRY"]}.items():
        Xa = neutralise(Xt, cols)
        o = M.grouped_oof(mk, Xa, y, folds)
        rows.append((name, roc_auc_score(y, o), roc_auc_score(y[hard], o[hard])))
    ab = pd.DataFrame(rows, columns=["variant", "cv_auc", "cv_auc_hard_negatives"]).set_index("variant")
    print("\n== Ablations ==\n", ab.round(3).to_string())
    ab.to_csv("artifacts/ablations.csv")

    # ---- 3. interpretable LR coefficients (fit on full train; standardised features)
    lrp = M.pipe(M.lr(0.1)).fit(Xt, y)
    names = lrp.named_steps["feat"].get_feature_names_out()
    coef = pd.Series(lrp.named_steps["est"].coef_[0], index=names).sort_values()
    print("\n== LR: strongest NEGATIVE (less liked) ==\n", coef.head(18).round(3).to_string())
    print("\n== LR: strongest POSITIVE (more liked) ==\n", coef.tail(18)[::-1].round(3).to_string())
    coef.to_csv("artifacts/lr_coefficients.csv")

    # ---- 4. per-subsector positive rate vs OOF predicted
    t = pd.DataFrame({"sub": Xt.SUBSECTOR, "y": y, "pred": oof, "neg": dt.y_neg.values})
    s = t.groupby("sub").agg(n=("y", "size"), pos_rate=("y", "mean"), neg_rate=("neg", "mean"), mean_pred=("pred", "mean"))
    print("\n== Subsector (train, n>=25) ==\n", s[s.n >= 25].sort_values("pos_rate", ascending=False).round(2).to_string())

    # ---- 5. descriptive univariate patterns: positives vs negatives (train)
    d = Xt.assign(y=y, grp=dt.group.values)
    print("\n== Medians by group ==\n", d.groupby("grp")[["age", "NUM_OF_EMPLOYEES", "revenue_m", "capital_raised_m",
                                                      "hc_g12", "hc_slope", "ec_us_pct", "n_investors"]].median().round(2).to_string())
    print("\nOWNERSHIP share by group\n", pd.crosstab(d.grp, d.OWNERSHIP, normalize="index").round(2).loc[:, lambda x: x.max() > 0.05].to_string())
    print("\nREVENUE_RANGE share by group\n", pd.crosstab(d.grp, d.REVENUE_RANGE, normalize="index").round(2).to_string())
    print("\ncapital missing by group\n", d.groupby("grp").capital_raised_m.apply(lambda s: s.isna().mean()).round(2).to_dict())


if __name__ == "__main__":
    main()
