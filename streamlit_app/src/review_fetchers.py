import re
import time
import requests
import pandas as pd
from google_play_scraper import Sort, reviews
import requests_cache

requests_cache.install_cache("app_reviews_cache", expire_after=3600)

DEFAULT_COUNTRIES = ["de", "fr", "pl", "gb", "us"]
DEFAULT_LANGUAGES = ["de", "en", "fr", "pl"]

# ----------------------------
# Utils
# ----------------------------

def _extract_apple_app_id(app_id_or_url: str) -> str:
    match = re.search(r"id(\d+)", str(app_id_or_url))
    return match.group(1) if match else str(app_id_or_url).strip()

# ----------------------------
# APPLE RSS
# ----------------------------

def fetch_apple_reviews_rss(
    app_id_or_url,
    country,
    app_name=None,
    max_pages=30,
    max_reviews=1000,
    debug=False,
):
    app_id = _extract_apple_app_id(app_id_or_url)
    country = str(country).lower().strip()
    rows = []

    if debug:
        print("\n[APPLE RSS API]")
        print(f"App ID: {app_id} | Country: {country}")

    effective_max_pages = min(max_pages, 10)

    def _get_label(field, default=""):
        if isinstance(field, dict):
            return str(field.get("label", default))
        return default

    for page in range(1, effective_max_pages + 1):
        if len(rows) >= max_reviews:
            break

        url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/json"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                if debug:
                    print(f"[STEP] Page {page} returned status code: {response.status_code}")
                break

            data = response.json()
            entries = data.get('feed', {}).get('entry', [])

            if isinstance(entries, dict):
                entries = [entries]
            elif not isinstance(entries, list):
                continue

            if not entries:
                break

            for entry in entries:
                if len(rows) >= max_reviews:
                    break

                if not isinstance(entry, dict) or 'im:rating' not in entry:
                    continue

                try:
                    review_id = _get_label(entry.get('id'))
                    rating = _get_label(entry.get('im:rating'))
                    text = _get_label(entry.get('content'))
                    title = _get_label(entry.get('title'))
                    version = _get_label(entry.get('im:version'))
                    date_raw = _get_label(entry.get('updated'))
                    
                    author_obj = entry.get('author', {})
                    author_name = ""
                    if isinstance(author_obj, dict):
                        author_name = _get_label(author_obj.get('name'))

                    rows.append({
                        "review_id": review_id,
                        "source_store": "appleappstore",
                        "app_name": app_name,
                        "app_identifier": app_id,
                        "review_date": date_raw,
                        "rating": int(rating) if rating.isdigit() else None,
                        "review_title": title.strip(),
                        "review_text": text.strip(),
                        "review_version": version,
                        "author": author_name,
                        "country": country,
                        "language": country,
                    })

                except Exception as parse_e:
                    if debug:
                        print(f"[PARSE ERROR INSIDE LOOP] {parse_e}")

        except Exception as e:
            if debug:
                print(f"[NETWORK/JSON ERROR] {e}")
            break

    return pd.DataFrame(rows[:max_reviews])

# ----------------------------
# GOOGLE PLAY
# ----------------------------

def fetch_google_play_reviews(app_name, package_name, countries, languages, count_per_request=200):
    rows = []
    
    country_to_lang_map = {
        "de": "de",
        "fr": "fr",
        "pl": "pl",
        "gb": "en",
        "us": "en"
    }

    pairs_to_fetch = []
    cleaned_languages = [str(l).lower().strip() for l in languages]

    for c in countries:
        c_clean = str(c).lower().strip()
        target_lang = country_to_lang_map.get(c_clean, "en")

        if target_lang in cleaned_languages:
            pairs_to_fetch.append((target_lang, c_clean))

    if not pairs_to_fetch and countries and languages:
        fallback_lang = cleaned_languages[0]
        for c in countries:
            pairs_to_fetch.append((fallback_lang, str(c).lower().strip()))

    for lang, country in pairs_to_fetch:
        token = None
        while True:
            result, token = reviews(
                package_name,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=count_per_request,
                continuation_token=token,
            )

            if not result:
                break

            for r in result:
                rows.append({
                    "review_id": r.get("reviewId"),
                    "source_store": "googleplay",
                    "app_name": app_name,
                    "app_identifier": package_name,
                    "review_date": r.get("at"),
                    "rating": r.get("score"),
                    "review_title": "",
                    "review_text": r.get("content", ""),
                    "review_version": r.get("reviewCreatedVersion"),
                    "author": r.get("userName"),
                    "country": country,
                    "language": lang,
                })

            if token is None:
                break
            time.sleep(0.5)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True)
    df["review_date"] = df["review_date"].dt.tz_convert(None)

    return df.drop_duplicates(subset=["source_store","app_identifier","review_id"]).reset_index(drop=True)

# ----------------------------
# MERGE
# ----------------------------

def fetch_live_reviews(
    app_name,
    googleplay_app_id=None,
    apple_app_id_or_url=None,
    countries=None,
    languages=None,
    apple_max_pages=10,
):
    countries = countries or DEFAULT_COUNTRIES
    languages = languages or DEFAULT_LANGUAGES

    frames = []

    if googleplay_app_id:
        gpdf = fetch_google_play_reviews(
            app_name=app_name, 
            package_name=googleplay_app_id,
            countries=countries,
            languages=languages
        )
        if not gpdf.empty:
            frames.append(gpdf)

    if apple_app_id_or_url:
        apple_country_frames = []
        for country in countries:
            appledf = fetch_apple_reviews_rss(
                app_id_or_url=apple_app_id_or_url,
                country=country,
                app_name=app_name,
                max_pages=apple_max_pages,
            )
            if not appledf.empty:
                apple_country_frames.append(appledf)

        if apple_country_frames:
            frames.append(pd.concat(apple_country_frames, ignore_index=True))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True)
    df["review_date"] = df["review_date"].dt.tz_convert(None)

    return df.reset_index(drop=True)

def filter_date_range(df, start_date, end_date):
    if df.empty:
        return df.copy()

    out = df.copy()
    out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce", utc=True).dt.tz_convert(None)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    return out[(out["review_date"] >= start_ts) & (out["review_date"] < end_ts)].copy()