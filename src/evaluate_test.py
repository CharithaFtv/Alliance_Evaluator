"""Step 5b: ONE-TIME evaluation on the sealed test set. Models were selected and calibrated on train CV only."""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import average_precision_score, roc_auc_score

import models as M

OUT = "artifacts/test_results.csv"
if os.path.exists(OUT) and "--force" not in sys.argv:
    sys.exit(f"{OUT} exists: the sealed test set was already scored. Re-running would be test peeking.")


def p_at(y, s, f):
    k = max(1, int(len(y) * f))
    return y[np.argsort(-s)[:k]].mean()


def stats(y, s):
    return dict(auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
                p_at_10=p_at(y, s, 0.10), p_at_20=p_at(y, s, 0.20))


def cluster_ci(y, s, blocks, n=2000, seed=0):
    """Bootstrap resampling whole ROW_ID blocks (rows within a block are correlated)."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    idx = {b: np.where(blocks == b)[0] for b in ub}
    out = []
    for _ in range(n):
        take = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        if 0 < y[take].sum() < len(take):
            out.append(stats(y[take], s[take]))
    d = pd.DataFrame(out)
    return d.quantile(0.025), d.quantile(0.975)


def main():
    data, X = M.load()
    sp = pd.read_csv("artifacts/splits.csv", dtype={"ID": str})[["ID", "block"]]
    data = data.merge(sp, on="ID", how="left")
    tr, te = (data.role == "train").values, (data.role == "test").values
    Xt, dt = X[tr].reset_index(drop=True), data[tr].reset_index(drop=True)
    Xe, de = X[te].reset_index(drop=True), data[te].reset_index(drop=True)
    art = joblib.load("artifacts/models.joblib")
    y, blocks, grp = de.y_pos.values, de.block.values, de.group.values
    nf = de.NEGATIVE_FLAGS.map(lambda s: json.loads(s) if isinstance(s, str) else {})
    svc = nf.map(lambda d: d.get("services_model", 0)).values

    scores = {"selected_gb": art["like"].predict_proba(Xe)[:, 1]}
    # pre-specified secondary models, fit on train only (no selection on test)
    scores["logreg_C0.1"] = M.pipe(M.lr(0.1)).fit(Xt, dt.y_pos.values).predict_proba(Xe)[:, 1]
    scores["similarity_knn"] = M.SimilarityModel().fit(Xt, dt.y_pos.values).predict_proba(Xe)[:, 1]
    rate = pd.Series(dt.y_pos.values).groupby(Xt.SUBSECTOR.values).mean()
    scores["baseline_subsector_rate"] = Xe.SUBSECTOR.map(rate).fillna(dt.y_pos.mean()).values

    rows = []
    subsets = {"all labelled": np.ones(len(y), bool),
               "hard (pos vs neg w/o services_model)": (y == 1) | ((grp == "negative_only") & (svc == 0)),
               "pos vs services_model negatives": (y == 1) | ((grp == "negative_only") & (svc == 1)),
               "positive_only vs overlap": np.isin(grp, ["positive_only", "overlap"])}
    for name, s in scores.items():
        for sub, m in subsets.items():
            yy = y[m] if sub != "positive_only vs overlap" else (grp[m] == "positive_only").astype(int)
            lo, hi = cluster_ci(yy, s[m], blocks[m], n=1000)
            st = stats(yy, s[m])
            rows.append(dict(model=name, subset=sub, n=int(m.sum()), n_pos=int(yy.sum()),
                             auc=st["auc"], auc_lo=lo["auc"], auc_hi=hi["auc"], pr_auc=st["pr_auc"],
                             p_at_10=st["p_at_10"], p_at_20=st["p_at_20"], base_rate=yy.mean()))
    # Dislike head and reasons
    yd = de.y_neg.values
    sd = art["dislike"].predict_proba(Xe)[:, 1]
    lo, hi = cluster_ci(yd, sd, blocks, n=1000)
    rows.append(dict(model="dislike_head", subset="all labelled", n=len(yd), n_pos=int(yd.sum()),
                     auc=roc_auc_score(yd, sd), auc_lo=lo["auc"], auc_hi=hi["auc"],
                     pr_auc=average_precision_score(yd, sd), base_rate=yd.mean()))
    negm = yd == 1
    for f, m in art["reasons"].items():
        yr = nf[negm].map(lambda d: d.get(f, np.nan)).dropna()
        if yr.sum() >= 5 and (1 - yr).sum() >= 5:
            Xr = Xe[negm].loc[yr.index.map(lambda i: list(nf[negm].index).index(i))] if False else Xe[negm].loc[yr.index]
            rows.append(dict(model=f"reason_{f}", subset="test negatives", n=len(yr), n_pos=int(yr.sum()),
                             auc=roc_auc_score(yr, m.predict_proba(Xr)[:, 1]), base_rate=yr.mean()))
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    # calibration of the selected model on test (calibrated prob vs outcome), by decile bins of 5
    p = art["cal"].predict_proba(logit(np.clip(scores["selected_gb"], 1e-4, 1 - 1e-4)).reshape(-1, 1))[:, 1]
    b = pd.qcut(p, 5, duplicates="drop")
    print("\ncalibration (5 bins):\n", pd.DataFrame({"pred": p, "obs": y}).groupby(b, observed=True).mean().round(3).to_string())
    print("\nmean calibrated prob by group:", pd.Series(p).groupby(grp).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
