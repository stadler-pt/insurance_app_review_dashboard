import pandas as pd


def normalize_text(x):
    """
    Normalize a text value for downstream processing.

    - Convert missing values to an empty string
    - Cast everything else to string
    - Remove leading and trailing whitespace
    """
    if pd.isna(x):
        return ""
    return str(x).strip()


def prepare_reviews(df):
    """
    Clean and standardize a raw reviews DataFrame.

    The function:
    - normalizes column names
    - renames selected source-specific columns to a common schema
    - cleans text fields
    - standardizes categorical fields
    - parses dates and ratings
    - removes rows without a valid review date
    - creates a combined full_text field
    """
    out = df.copy()

    # Normalize all column names first so downstream access becomes consistent.
    out.columns = [str(c).strip().lower() for c in out.columns]

    # Map common source column names to the internal schema used by the app.
    rename_map = {
        "source_store": "sourcestore",
        "review_text": "reviewtext",
        "review_title": "reviewtitle",
        "review_date": "reviewdate",
        "app_name": "appname",
    }
    out = out.rename(columns=rename_map)

    # Clean text columns so they are always safe to concatenate and display.
    out["reviewtext"] = out.get("reviewtext", "").apply(normalize_text)
    out["reviewtitle"] = out.get("reviewtitle", "").apply(normalize_text)
    out["appname"] = out.get("appname", "").apply(normalize_text)

    # Standardize categorical fields to lower case for easier filtering/grouping.
    out["sourcestore"] = (
        out.get("sourcestore", "")
        .apply(normalize_text)
        .str.lower()
    )
    out["country"] = (
        out.get("country", "")
        .apply(normalize_text)
        .str.lower()
    )
    out["language"] = (
        out.get("language", "")
        .apply(normalize_text)
        .str.lower()
    )

    # Parse review dates into datetime objects.
    # Invalid values become NaT and are removed below.
    out["reviewdate"] = pd.to_datetime(
        out.get("reviewdate", pd.NaT),
        errors="coerce",
    )

    # Convert ratings to numeric values and keep only the valid 1–5 range.
    out["rating"] = pd.to_numeric(out.get("rating", pd.NA), errors="coerce")
    out.loc[~out["rating"].between(1, 5), "rating"] = pd.NA
    
    # Drop rows without a valid date because they are hard to place in timelines.
    out = out.dropna(subset=["reviewdate"]).reset_index(drop=True)

    # Create a combined text field for clustering, inference, and search.
    out["full_text"] = (
        out["reviewtitle"].fillna("") + " " + out["reviewtext"].fillna("")
    ).str.strip()

    return out