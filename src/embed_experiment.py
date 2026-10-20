"""Step 7: do sentence embeddings beat/add to TF-IDF? Train-only grouped CV; test set untouched.
Embeddings come from a frozen pre-trained model (no fitting on our labels), so computing them for all rows
leaks nothing; PCA on top is fit inside each CV fold."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import models as M
from features import build_features, _tokens_json_list

ART = "artifacts"
MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "mpnet": "sentence-transformers/all-mpnet-base-v2"}


def embed_all(data, key):
    path = f"{ART}/emb_{key}.npy"
    if os.path.exists(path):
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    txt = (data.DESCRIPTION.fillna("") + ". Tags: " + data.TAGS.map(lambda x: ", ".join(_tokens_json_list(x)[:15]))).tolist()
    E = SentenceTransformer(MODELS[key], device="cpu").encode(txt, batch_size=64, normalize_embeddings=True,
                                                                show_progress_bar=False)
    np.save(path, E)
    return E


def make(mode, d, n_pca, est):
    ct = build_features()
    parts = list(ct.transformers)
    if mode in ("emb", "emb_only_text"):
        parts = [p for p in parts if p[0] != "text"]
    if mode in ("emb", "both"):
        parts.append(("emb", Pipeline([("pca", PCA(n_pca, random_state=42)), ("sc", StandardScaler())]),
                      [f"emb_{i}" for i in range(d)]))
    ct = ColumnTransformer(parts, sparse_threshold=0.0)
    return Pipeline([("feat", ct), ("est", est)])


def main():
    data, X = M.load()
    tr = (data.role == "train").values
    dt = data[tr].reset_index(drop=True)
    y, folds = dt.y_pos.values, dt.cv_fold
    nf = dt.NEGATIVE_FLAGS.map(lambda s: json.loads(s) if isinstance(s, str) else {})
    svc = nf.map(lambda d: d.get("services_model", 0)).values
    hard = (y == 1) | ((dt.group.values == "negative_only") & (svc == 0))
    gb = lambda: M.gb(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0)
    rows = []
    for key in MODELS:
        E = embed_all(data, key)
        Xe = X.copy()
        emb = pd.DataFrame(E, columns=[f"emb_{i}" for i in range(E.shape[1])], index=X.index)
        Xe = pd.concat([X, emb], axis=1)
        Xt = Xe[tr].reset_index(drop=True)
        for mode, n in [("tfidf", 0), ("emb", 20), ("emb", 40), ("both", 40)]:
            if key != "minilm" and mode == "tfidf":
                continue  # baseline is model-independent, run once
            for estname, est in [("gb", gb), ("lr", lambda: M.lr(0.1))]:
                o = M.grouped_oof(lambda: make(mode, E.shape[1], max(n, 1), est()), Xt, y, folds)
                per = [roc_auc_score(y[folds == k], o[folds == k]) for k in sorted(folds.unique())]
                rows.append(dict(embedder=key if mode != "tfidf" else "-", text_features=f"{mode}{n or ''}",
                                 est=estname, auc=roc_auc_score(y, o), fold_sd=np.std(per),
                                 auc_hard=roc_auc_score(y[hard], o[hard]), pr_auc=average_precision_score(y, o),
                                 p_at_10=M.p_at_k(y, o)))
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(f"{ART}/embedding_experiment.csv", index=False)


if __name__ == "__main__":
    main()
