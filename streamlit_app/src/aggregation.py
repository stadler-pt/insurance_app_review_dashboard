import pandas as pd


def review_summary(df):
    # Return default summary values if no review data is available.
    if df is None or df.empty:
        return {
            "total_reviews": 0,
            "avg_rating": None,
            "google_reviews": 0,
            "apple_reviews": 0,
        }

    # Work on a copy to avoid modifying the original DataFrame.
    out = df.copy()

    # Standardize store names for consistent counting.
    if "sourcestore" in out.columns:
        out["sourcestore"] = (
            out["sourcestore"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        out["sourcestore"] = ""

    # Convert ratings to numeric values so the mean can be calculated safely.
    if "rating" in out.columns:
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    else:
        out["rating"] = pd.Series(dtype="float64")

    # Build a compact summary of review count, average rating, and store split.
    return {
        "total_reviews": int(len(out)),
        "avg_rating": float(out["rating"].mean()) if out["rating"].notna().any() else None,
        "google_reviews": int((out["sourcestore"] == "googleplay").sum()),
        "apple_reviews": int((out["sourcestore"] == "appleappstore").sum()),
    }


def label_summary(scored_df, top4_labels):
    rows = []

    # Collect one summary row per label.
    for label in top4_labels:
        pred_col = f"pred_{label}"
        prob_col = f"prob_{label}"

         # Count how many reviews were assigned to the current label.
        if pred_col in scored_df.columns:
            count_value = int(pd.to_numeric(scored_df[pred_col], errors="coerce").fillna(0).sum())
        else:
            count_value = 0

        # Compute the mean model probability for the current label.
        if prob_col in scored_df.columns:
            avg_prob = float(pd.to_numeric(scored_df[prob_col], errors="coerce").mean())
        else:
            avg_prob = None

        # Store the aggregated values in a row dictionary.
        rows.append(
            {
                "label_key": label,
                "anzahl": count_value,
                "durchschnittliche_wahrscheinlichkeit": avg_prob,
            }
        )

    return pd.DataFrame(rows)