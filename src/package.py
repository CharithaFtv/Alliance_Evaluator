"""Step 6a: build scoring artifacts from TRAIN rows only (explainer, reference distribution),
and the unknown-company review sheet. Does not touch the test set or refit the scorer."""
import joblib
import numpy as np
import pandas as pd

import models as M
from score import readable, top_drivers

ART = "artifacts"


def main():
    data, X = M.load()
    tr = (data.role == "train").values
    Xt, dt = X[tr].reset_index(drop=True), data[tr].reset_index(drop=True)
    # LR is used only to EXPLAIN (additive per-feature contributions); GB gives the score.
    expl = M.pipe(M.lr(0.1)).fit(Xt, dt.y_pos.values)
    joblib.dump(expl, f"{ART}/explainer.joblib")

    unk = pd.read_csv(f"{ART}/unknown_scores_frozen.csv", dtype={"ID": str})
    ref = np.sort(unk.like_raw.values)
    joblib.dump(ref, f"{ART}/reference_like_raw.joblib")  # percentile reference = sourced universe

    u = data[data.role == "unknown"].merge(unk[["ID", "like_prob", "dislike_raw"] + [c for c in unk if c.startswith("why_")]], on="ID")
    Xu = M.engineer(u)
    pos, neg = top_drivers(expl, Xu, k=4)
    u["why_liked"], u["why_not"] = pos, neg
    u["likely_reject_reason"] = unk.set_index("ID").loc[u.ID, [c for c in unk if c.startswith("why_")]].idxmax(axis=1).str.replace("why_", "").values
    cols = ["NAME", "WEBSITE", "COUNTRY", "SUBSECTOR", "MAIN_INDUSTRY", "YEAR_FOUNDED", "NUM_OF_EMPLOYEES",
            "revenue_m", "capital_raised_m", "MOST_RECENT_FUNDING_STAGE", "like_prob", "dislike_raw",
            "why_liked", "why_not", "likely_reject_reason", "DESCRIPTION"]
    out = u.sort_values("like_prob", ascending=False)[cols]
    out.head(150).to_csv(f"{ART}/review_top150_unknowns.csv", index=False)
    out.tail(50).to_csv(f"{ART}/review_bottom50_unknowns.csv", index=False)
    print(out.head(15)[["NAME", "SUBSECTOR", "like_prob", "dislike_raw", "why_liked", "why_not"]].to_string())
    print("\ntop-150 subsector mix:\n", out.head(150).SUBSECTOR.value_counts().head(8).to_string())
    print("\nunknown universe: share with like_prob>=0.5:", round((unk.like_prob >= 0.5).mean(), 3),
          "| n>=0.6:", int((unk.like_prob >= 0.6).sum()))


if __name__ == "__main__":
    main()
