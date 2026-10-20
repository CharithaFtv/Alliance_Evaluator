"""Step 4: Like / Dislike heads, reason model, similarity score. Everything is fit and tuned on
TRAIN rows only via the saved grouped folds. The sealed test set is NOT touched here."""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from features import FORBIDDEN, assert_no_forbidden, build_features, engineer
from loader import build

SEED = 42
ART = "artifacts"


def load():
    data, _ = build()
    sp = pd.read_csv(f"{ART}/splits.csv", dtype={"ID": str})
    data = data.merge(sp[["ID", "role", "cv_fold"]], on="ID", how="left")
    X = engineer(data)
    assert_no_forbidden(X.columns)
    return data, X


def pipe(est):
    return Pipeline([("feat", build_features()), ("est", est)])


def lr(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)


def gb(**kw):
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, **kw)


def p_at_k(y, s, frac=0.10):
    k = max(1, int(len(y) * frac))
    return y[np.argsort(-s)[:k]].mean()


def grouped_oof(make, X, y, folds, score=None):
    """Out-of-fold scores; the full pipeline (incl. vocabularies/TF-IDF/SVD) refits per fold."""
    oof = np.full(len(y), np.nan)
    for k in sorted(folds.unique()):
        va = (folds == k).values
        m = make().fit(X[~va], y[~va])
        oof[va] = (score or (lambda m, Z: m.predict_proba(Z)[:, 1]))(m, X[va])
    return oof


def metrics(y, s, folds):
    per = [roc_auc_score(y[folds == k], s[folds == k]) for k in sorted(folds.unique())]
    return dict(auc=roc_auc_score(y, s), auc_fold_sd=np.std(per),
                pr_auc=average_precision_score(y, s), base_rate=y.mean(), p_at_10=p_at_k(y, s))


class SimilarityModel:
    """Score = mean top-k cosine sim to train positives minus to train negatives."""
    def __init__(self, k=10):
        self.k = k
        self.feat = build_features()

    def fit(self, X, y):
        Z = normalize(self.feat.fit_transform(X))
        self.P, self.N = Z[y == 1], Z[y == 0]
        return self

    def _top(self, S):
        k = min(self.k, S.shape[1])
        return np.sort(S, axis=1)[:, -k:].mean(1)

    def predict_proba(self, X):
        Z = normalize(self.feat.transform(X))
        s = self._top(Z @ self.P.T) - self._top(Z @ self.N.T)
        return np.column_stack([-s, s])


def main():
    data, X = load()
    tr = (data.role == "train").values
    Xt, dt = X[tr].reset_index(drop=True), data[tr].reset_index(drop=True)
    folds = dt["cv_fold"]
    y_like, y_dis = dt["y_pos"].values, dt["y_neg"].values
    res = {}

    # ---- Like head: baselines + candidates
    cands = {f"lr_C{C}": (lambda C=C: pipe(lr(C))) for C in [0.003, 0.01, 0.03, 0.1, 0.3]}
    cands.update({f"gb_d{d}_lr{r}": (lambda d=d, r=r: pipe(gb(max_depth=d, learning_rate=r,
                                                             max_iter=150, l2_regularization=1.0)))
                  for d, r in [(3, 0.05), (None, 0.03)]})
    cands["similarity_knn"] = lambda: SimilarityModel()
    oofs = {}
    for name, mk in cands.items():
        oofs[name] = grouped_oof(mk, Xt, y_like, folds)
        res[name] = metrics(y_like, oofs[name], folds)
        print(name, {k: round(v, 3) for k, v in res[name].items()})

    # ---- Baseline: SUBSECTOR-only rule (train-fold positive rate) and a leakage canary
    sub_oof = np.zeros(len(y_like))
    for k in folds.unique():
        va = (folds == k).values
        rate = pd.Series(y_like[~va]).groupby(Xt.loc[~va, "SUBSECTOR"].values).mean()
        sub_oof[va] = Xt.loc[va, "SUBSECTOR"].map(rate).fillna(y_like[~va].mean()).values
    res["baseline_subsector_rate"] = metrics(y_like, sub_oof, folds)
    print("baseline_subsector_rate", {k: round(v, 3) for k, v in res["baseline_subsector_rate"].items()})

    can = dt[["ACTIVITY_COUNT", "DAYS_SINCE_LAST_CONTACT"]].apply(pd.to_numeric, errors="coerce")
    can = can.fillna(can.median())
    co = np.zeros(len(y_like))
    for k in folds.unique():
        va = (folds == k).values
        co[va] = LogisticRegression(max_iter=1000).fit(can[~va], y_like[~va]).predict_proba(can[va])[:, 1]
    print("CANARY (internal cols only, must be high => correctly excluded): AUC",
          round(roc_auc_score(y_like, co), 3))

    # ---- choose best by CV AUC among non-similarity models; refit on all train
    best = max((n for n in cands if n != "similarity_knn"), key=lambda n: res[n]["auc"])
    print("best like model:", best)
    like = cands[best]().fit(Xt, y_like)
    # Platt calibration on OOF scores (train only)
    from scipy.special import logit
    o = np.clip(oofs[best], 1e-4, 1 - 1e-4)
    cal = LogisticRegression().fit(logit(o).reshape(-1, 1), y_like)

    # ---- Dislike head (same family as best like model)
    dis_oof = grouped_oof(cands[best], Xt, y_dis, folds)
    res["dislike_head"] = metrics(y_dis, dis_oof, folds)
    print("dislike_head", {k: round(v, 3) for k, v in res["dislike_head"].items()})
    dis = cands[best]().fit(Xt, y_dis)

    # ---- Reason model: NEGATIVE_FLAGS among train rows with y_neg==1 (flags used as targets only)
    neg = dt["y_neg"].values == 1
    flags = dt.loc[neg, "NEGATIVE_FLAGS"].dropna().map(json.loads).apply(pd.Series)
    reasons = {}
    for f in flags.columns:
        if flags[f].sum() < 30:
            continue
        idx = flags.index
        Xr, yr, fr = Xt.loc[idx].reset_index(drop=True), flags[f].values.astype(int), folds.loc[idx].reset_index(drop=True)
        oo = grouped_oof(lambda: pipe(lr(0.03)), Xr, yr, fr)
        res[f"reason_{f}"] = metrics(yr, oo, fr)
        print(f"reason {f}: n+={yr.sum()} auc={res[f'reason_{f}']['auc']:.3f}")
        reasons[f] = pipe(lr(0.03)).fit(Xr, yr)

    # ---- score unknowns only (frozen, timestamped). Test rows are NOT scored.
    unk = (data.role == "unknown").values
    Xu = X[unk]
    raw = like.predict_proba(Xu)[:, 1]
    out = pd.DataFrame({"ID": data.loc[unk, "ID"].values, "NAME": data.loc[unk, "NAME"].values,
                        "like_prob": cal.predict_proba(logit(np.clip(raw, 1e-4, 1 - 1e-4)).reshape(-1, 1))[:, 1],
                        "like_raw": raw, "dislike_raw": dis.predict_proba(Xu)[:, 1]})
    for f, m in reasons.items():
        out[f"why_{f}"] = m.predict_proba(Xu)[:, 1]
    out["scored_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.sort_values("like_prob", ascending=False).to_csv(f"{ART}/unknown_scores_frozen.csv", index=False)

    joblib.dump(dict(like=like, cal=cal, dislike=dis, reasons=reasons, best=best), f"{ART}/models.joblib")
    pd.DataFrame(res).T.to_csv(f"{ART}/cv_results.csv")
    print(out[["NAME", "like_prob", "dislike_raw"]].head(10).to_string())
    print(out.like_prob.describe().round(3).to_dict())


if __name__ == "__main__":
    main()
