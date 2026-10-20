"""Step 2: sealed train/test split on labelled rows. Unknowns stay outside both (scored only)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from loader import build

SEED = 42
BLOCK = 250          # ROW_ID block width: rows from one sourcing batch stay together
TEST_FOLDS = 5       # 1 of 5 folds -> 20% test
CV_FOLDS = 5         # tuning folds inside train only
OUT = "artifacts/splits.csv"


def assign_groups(df):
    g = (pd.to_numeric(df["ROW_ID"]) // BLOCK).astype(int)
    # duplicate LinkedIn URLs share the first occurrence's group
    li = df["LINKEDIN_URL"]
    first = g.groupby(li).transform("first")
    return g.where(li.isna(), first)


def make_split(df):
    df = df.copy()
    df["block"] = assign_groups(df)
    lab = df[df["group"] != "unknown"]
    sgkf = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, te_idx = next(sgkf.split(lab, lab["group"], lab["block"]))
    train, test = lab.iloc[tr_idx], lab.iloc[te_idx]

    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=df.index)
    for k, (_, va) in enumerate(cv.split(train, train["group"], train["block"])):
        fold.loc[train.index[va]] = k

    df["role"] = "unknown"
    df.loc[train.index, "role"] = "train"
    df.loc[test.index, "role"] = "test"
    df["cv_fold"] = fold
    return df


def check(df):
    tr, te = df[df.role == "train"], df[df.role == "test"]
    assert not set(tr.ID) & set(te.ID)
    assert not set(tr.block) & set(te.block), "block leaks across train/test"
    assert not set(tr.LINKEDIN_URL.dropna()) & set(te.LINKEDIN_URL.dropna())
    assert (df[df.role == "unknown"].group == "unknown").all()
    assert (df[df.group != "unknown"].role != "unknown").all()
    for k in range(CV_FOLDS):
        va = tr[tr.cv_fold == k]
        assert not set(va.block) & set(tr[tr.cv_fold != k].block)


if __name__ == "__main__":
    data, _ = build()
    s = make_split(data)
    check(s)
    s[["ID", "ROW_ID", "block", "group", "role", "cv_fold"]].to_csv(OUT, index=False)
    print(pd.crosstab(s.group, s.role, margins=True).to_string())
    print((pd.crosstab(s.group, s.role, normalize="columns") * 100).round(1).to_string())
    print("blocks train/test:", s[s.role == "train"].block.nunique(), s[s.role == "test"].block.nunique())
    print(pd.crosstab(s[s.role == "train"].group, s[s.role == "train"].cv_fold).to_string())
