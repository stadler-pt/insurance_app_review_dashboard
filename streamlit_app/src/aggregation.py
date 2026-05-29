import pandas as pd


def review_summary(df):
    if df is None or df.empty:
        return {
            "total_reviews": 0,
            "avg_rating": None,
            "google_reviews": 0,
            "apple_reviews": 0,
        }

    out = df.copy()

    if "sourcestore" in out.columns:
        out["sourcestore"] = (
            out["sourcestore"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        out["sourcestore"] = ""

    if "rating" in out.columns:
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    else:
        out["rating"] = pd.Series(dtype="float64")

    return {
        "total_reviews": int(len(out)),
        "avg_rating": float(out["rating"].mean()) if out["rating"].notna().any() else None,
        "google_reviews": int((out["sourcestore"] == "googleplay").sum()),
        "apple_reviews": int((out["sourcestore"] == "appleappstore").sum()),
    }


def label_summary(scored_df, top4_labels):
    rows = []

    for label in top4_labels:
        pred_col = f"pred_{label}"
        prob_col = f"prob_{label}"

        if pred_col in scored_df.columns:
            count_value = int(pd.to_numeric(scored_df[pred_col], errors="coerce").fillna(0).sum())
        else:
            count_value = 0

        if prob_col in scored_df.columns:
            avg_prob = float(pd.to_numeric(scored_df[prob_col], errors="coerce").mean())
        else:
            avg_prob = None

        rows.append(
            {
                "label_key": label,
                "anzahl": count_value,
                "durchschnittliche_wahrscheinlichkeit": avg_prob,
            }
        )

    return pd.DataFrame(rows)