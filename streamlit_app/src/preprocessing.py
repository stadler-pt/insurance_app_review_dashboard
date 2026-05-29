import pandas as pd


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def prepare_reviews(df):
    out = df.copy()

    out.columns = [str(c).strip().lower() for c in out.columns]

    rename_map = {
        "source_store": "sourcestore",
        "review_text": "reviewtext",
        "review_title": "reviewtitle",
        "review_date": "reviewdate",
        "app_name": "appname",
    }
    out = out.rename(columns=rename_map)

    out["reviewtext"] = out.get("reviewtext", "").apply(normalize_text)
    out["reviewtitle"] = out.get("reviewtitle", "").apply(normalize_text)
    out["appname"] = out.get("appname", "").apply(normalize_text)

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

    out["reviewdate"] = pd.to_datetime(
        out.get("reviewdate", pd.NaT),
        errors="coerce",
    )

    out["rating"] = pd.to_numeric(out.get("rating", pd.NA), errors="coerce")
    out.loc[~out["rating"].between(1, 5), "rating"] = pd.NA

    out = out.dropna(subset=["reviewdate"]).reset_index(drop=True)

    out["full_text"] = (
        out["reviewtitle"].fillna("") + " " + out["reviewtext"].fillna("")
    ).str.strip()

    return out