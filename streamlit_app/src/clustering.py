import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Stopword list used to remove frequent but less informative German terms.
STOPWORDS = {
    "der", "die", "das", "ein", "eine", "und", "oder", "aber", "ist", "sind",
    "war", "waren", "ich", "du", "er", "sie", "es", "wir", "ihr", "nicht",
    "kein", "keine", "mit", "für", "von", "auf", "im", "in", "am", "an",
    "zu", "den", "dem", "des", "dass", "doch", "auch", "nur", "noch",
    "app", "apps", "diese", "dieser", "meine", "mein", "habe", "hat", "haben",
    "halben", "muss", "schon", "einfach", "immer", "sich", "wenn", "da", "wieder", "ständig", "kann",
    "bei", "gibt"
}


def build_word_clusters(df, text_col="reviewtext", n_clusters=6, max_features=1000):
    """Build TF-IDF-based text clusters and attach cluster labels and keywords to the input data."""
    if df is None or df.empty or text_col not in df.columns:
        return pd.DataFrame()

    # Work on a cleaned copy and remove empty texts before vectorization.
    work = df.copy()
    work[text_col] = work[text_col].fillna("").astype(str).str.strip()
    work = work[work[text_col] != ""].copy()

    if work.empty:
        return pd.DataFrame()

    # Convert review texts into TF-IDF features based on bi-grams and tri-grams.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(STOPWORDS),
        max_features=1500,
        ngram_range=(2, 3),
        min_df=2,
        max_df=0.6
    )
    X = vectorizer.fit_transform(work[text_col])

    # Limit the number of clusters to a valid range for the current dataset.
    n_clusters = max(2, min(n_clusters, len(work)))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    work["cluster_id"] = model.fit_predict(X)

    # Extract the most important terms per cluster from the centroid weights.
    terms = vectorizer.get_feature_names_out()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    cluster_keywords = {}
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :8]]
        cluster_keywords[i] = ", ".join(top_terms)

    # Attach readable keyword summaries to each clustered review.
    work["cluster_keywords"] = work["cluster_id"].map(cluster_keywords)
    return work