"""Step 8: production refit on ALL labelled rows (train + test). Same model family and settings as selected
on train CV (gradient boosting, depth 3, lr 0.05, 150 iters, TF-IDF text, no SUBSECTOR). The test set is no
longer a holdout after this. Calibration is re-fit on grouped out-of-fold scores over all labelled rows."""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import models as M

OUT = "artifacts/production"
SEED = 42


def gbm():
    return M.pipe(M.gb(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0))


def main():
    os.makedirs(OUT, exist_ok=True)
    data, X = M.load()
    sp = pd.read_csv("artifacts/splits.csv", dtype={"ID": str})[["ID", "block"]]
    data = data.merge(sp, on="ID", how="left")
    lab = (data.group != "unknown").values
    Xl, dl = X[lab].reset_index(drop=True), data[lab].reset_index(drop=True)
    y_like, y_dis = dl.y_pos.values, dl.y_neg.values
    print("labelled rows:", len(dl), "| positives:", y_like.sum(), "| negatives:", y_dis.sum())

    # calibration: grouped OOF over all labelled rows
    folds = np.zeros(len(dl), int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xl, dl.group, dl.block)):
        folds[va] = k
    folds = pd.Series(folds)
    oof = M.grouped_oof(gbm, Xl, y_like, folds)
    print("OOF AUC on all labelled (sanity, not a test):", round(roc_auc_score(y_like, oof), 3))
    cal = LogisticRegression().fit(logit(np.clip(oof, 1e-4, 1 - 1e-4)).reshape(-1, 1), y_like)

    like = gbm().fit(Xl, y_like)
    dis = gbm().fit(Xl, y_dis)
    expl = M.pipe(M.lr(0.1)).fit(Xl, y_like)

    neg = y_dis == 1
    flags = dl.loc[neg, "NEGATIVE_FLAGS"].dropna().map(json.loads).apply(pd.Series)
    reasons = {}
    for f in flags.columns:
        if flags[f].sum() >= 30:
            reasons[f] = M.pipe(M.lr(0.03)).fit(Xl.loc[flags.index], flags[f].values.astype(int))

    # score unknowns with the production model
    unk = (data.role == "unknown").values
    Xu = X[unk]
    raw = like.predict_proba(Xu)[:, 1]
    out = pd.DataFrame({"ID": data.loc[unk, "ID"].values, "NAME": data.loc[unk, "NAME"].values,
                        "like_prob": cal.predict_proba(logit(np.clip(raw, 1e-4, 1 - 1e-4)).reshape(-1, 1))[:, 1],
                        "like_raw": raw, "dislike_raw": dis.predict_proba(Xu)[:, 1]})
    for f, m in reasons.items():
        out[f"why_{f}"] = m.predict_proba(Xu)[:, 1]
    out["scored_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.sort_values("like_prob", ascending=False).to_csv(f"{OUT}/unknown_scores_production.csv", index=False)

    joblib.dump(dict(like=like, cal=cal, dislike=dis, reasons=reasons, best="gb_d3_lr0.05 (production refit)"),
                f"{OUT}/models.joblib")
    joblib.dump(expl, f"{OUT}/explainer.joblib")
    joblib.dump(np.sort(raw), f"{OUT}/reference_like_raw.joblib")

    # how much did scores move vs the pre-refit (train-only) model?
    old = pd.read_csv("artifacts/unknown_scores_frozen.csv", dtype={"ID": str}).set_index("ID").loc[out.ID]
    print("Spearman old vs production like score on unknowns:", round(spearmanr(old.like_prob.values, out.like_prob.values)[0], 3))
    top_old = set(old.sort_values("like_prob", ascending=False).head(150).index)
    top_new = set(out.sort_values("like_prob", ascending=False).head(150).ID)
    print("top-150 overlap old vs production:", len(top_old & top_new))
    print("share >=0.5: old", round((old.like_prob >= .5).mean(), 3), "new", round((out.like_prob >= .5).mean(), 3))


if __name__ == "__main__":
    main()
