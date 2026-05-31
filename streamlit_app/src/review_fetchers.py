import re
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from google_play_scraper import Sort, reviews

DEFAULT_COUNTRIES = ["de", "fr", "pl", "gb", "us"]
DEFAULT_LANGUAGES = ["de", "en", "fr", "pl"]

# HTTP headers used for Apple requests.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# Status codes that are likely temporary and worth retrying.
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


# ----------------------------
# Utils
# ----------------------------

def _extract_apple_app_id(app_id_or_url: str) -> str:
    """Extract the numeric Apple App Store ID from either a raw ID or a full URL."""
    match = re.search(r"id(\d+)", str(app_id_or_url))
    return match.group(1) if match else str(app_id_or_url).strip()


def _safe_strip(value):
    """Return a stripped string representation or an empty string."""
    if value is None:
        return ""
    return str(value).strip()


def _request_with_retry(session, url, timeout=20, max_retries=3, debug=False):
    """Perform an HTTP GET request with basic retry and backoff handling."""
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            if debug:
                print(f"[HTTP] GET attempt {attempt}/{max_retries}: {url}")

            response = session.get(url, headers=REQUEST_HEADERS, timeout=timeout, allow_redirects=True)

            if debug:
                print(f"[HTTP] Status: {response.status_code} | Final URL: {response.url}")

            if response.status_code in RETRY_STATUS_CODES:
                wait_s = attempt * 1.5
                if debug:
                    print(f"[HTTP] Retryable status code received. Sleeping for {wait_s:.1f}s")
                time.sleep(wait_s)
                continue

            return response

        except Exception as e:
            last_exception = e
            wait_s = attempt * 1.5
            if debug:
                print(f"[HTTP] Request error: {e}")
                print(f"[HTTP] Sleeping for {wait_s:.1f}s before retry")
            time.sleep(wait_s)

    if last_exception:
        raise last_exception

    return None


def _build_apple_rss_json_urls(app_id, country, page):
    """Build several JSON RSS URL variants for the Apple review feed."""
    country = _safe_strip(country).lower()
    return [
        f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/json",
        f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostRecent/page={page}/json",
        f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/json",
        f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/page={page}/json",
        f"https://itunes.apple.com/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/json",
    ]


def _build_apple_rss_xml_urls(app_id, country, page):
    """Build several XML RSS URL variants for the Apple review feed."""
    country = _safe_strip(country).lower()
    return [
        f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/xml",
        f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostRecent/page={page}/xml",
        f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/xml",
        f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/page={page}/xml",
        f"https://itunes.apple.com/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/xml",
    ]


def _parse_apple_json_entries(data, app_id, country, app_name=None, debug=False):
    """Parse Apple RSS JSON entries into normalized review rows."""
    rows = []

    def _get_label(field, default=""):
        if isinstance(field, dict):
            return str(field.get("label", default))
        return default

    feed = data.get("feed", {}) if isinstance(data, dict) else {}
    entries = feed.get("entry", [])

    # Apple may return a single entry as a dict instead of a list.
    if isinstance(entries, dict):
        entries = [entries]
    elif not isinstance(entries, list):
        entries = []

    if debug:
        print(f"[APPLE JSON] Raw entry count: {len(entries)}")

    for entry in entries:
        # Skip feed-level metadata entries that are not actual reviews.
        if not isinstance(entry, dict) or "im:rating" not in entry:
            continue

        try:
            review_id = _get_label(entry.get("id"))
            rating = _get_label(entry.get("im:rating"))
            text = _get_label(entry.get("content"))
            title = _get_label(entry.get("title"))
            version = _get_label(entry.get("im:version"))
            date_raw = _get_label(entry.get("updated"))

            author_obj = entry.get("author", {})
            author_name = ""
            if isinstance(author_obj, dict):
                author_name = _get_label(author_obj.get("name"))

            rows.append({
                "review_id": review_id,
                "source_store": "appleappstore",
                "app_name": app_name,
                "app_identifier": app_id,
                "review_date": date_raw,
                "rating": int(rating) if str(rating).isdigit() else None,
                "review_title": _safe_strip(title),
                "review_text": _safe_strip(text),
                "review_version": _safe_strip(version),
                "author": _safe_strip(author_name),
                "country": country,
                "language": "",
            })

        except Exception as parse_e:
            if debug:
                print(f"[APPLE JSON] Parse error inside entry loop: {parse_e}")

    if debug:
        print(f"[APPLE JSON] Parsed review rows: {len(rows)}")

    return rows


def _parse_apple_xml_entries(xml_text, app_id, country, app_name=None, debug=False):
    """Parse Apple RSS XML entries into normalized review rows."""
    rows = []

    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        if debug:
            print(f"[APPLE XML] XML parse failed: {e}")
        return rows

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "im": "http://itunes.apple.com/rss",
    }

    entries = root.findall("atom:entry", ns)

    if debug:
        print(f"[APPLE XML] Raw entry count: {len(entries)}")

    for entry in entries:
        try:
            rating_el = entry.find("im:rating", ns)
            if rating_el is None:
                continue

            review_id = _safe_strip(entry.findtext("atom:id", default="", namespaces=ns))
            title = _safe_strip(entry.findtext("atom:title", default="", namespaces=ns))
            text = _safe_strip(entry.findtext("atom:content", default="", namespaces=ns))
            version = _safe_strip(entry.findtext("im:version", default="", namespaces=ns))
            date_raw = _safe_strip(entry.findtext("atom:updated", default="", namespaces=ns))
            rating = _safe_strip(rating_el.text)

            author_name = ""
            author_el = entry.find("atom:author", ns)
            if author_el is not None:
                author_name = _safe_strip(author_el.findtext("atom:name", default="", namespaces=ns))

            rows.append({
                "review_id": review_id,
                "source_store": "appleappstore",
                "app_name": app_name,
                "app_identifier": app_id,
                "review_date": date_raw,
                "rating": int(rating) if str(rating).isdigit() else None,
                "review_title": title,
                "review_text": text,
                "review_version": version,
                "author": author_name,
                "country": country,
                "language": "",
            })

        except Exception as parse_e:
            if debug:
                print(f"[APPLE XML] Parse error inside entry loop: {parse_e}")

    if debug:
        print(f"[APPLE XML] Parsed review rows: {len(rows)}")

    return rows


def _fetch_apple_page_json(session, app_id, country, page, app_name=None, debug=False):
    """Fetch one Apple RSS page using JSON URL variants."""
    urls = _build_apple_rss_json_urls(app_id, country, page)

    for idx, url in enumerate(urls, start=1):
        try:
            if debug:
                print(f"[APPLE JSON] Trying URL variant {idx}: {url}")

            response = _request_with_retry(session, url, debug=debug)

            if response is None:
                continue

            if response.status_code != 200:
                if debug:
                    print(f"[APPLE JSON] Non-200 response: {response.status_code}")
                continue

            content_type = response.headers.get("Content-Type", "")
            if debug:
                print(f"[APPLE JSON] Content-Type: {content_type}")
                print(f"[APPLE JSON] Response preview: {response.text[:250]}")

            data = response.json()
            rows = _parse_apple_json_entries(data, app_id, country, app_name=app_name, debug=debug)

            if rows:
                return rows

        except Exception as e:
            if debug:
                print(f"[APPLE JSON] Variant {idx} failed: {e}")

    return []


def _fetch_apple_page_xml(session, app_id, country, page, app_name=None, debug=False):
    """Fetch one Apple RSS page using XML URL variants."""
    urls = _build_apple_rss_xml_urls(app_id, country, page)

    for idx, url in enumerate(urls, start=1):
        try:
            if debug:
                print(f"[APPLE XML] Trying URL variant {idx}: {url}")

            response = _request_with_retry(session, url, debug=debug)

            if response is None:
                continue

            if response.status_code != 200:
                if debug:
                    print(f"[APPLE XML] Non-200 response: {response.status_code}")
                continue

            if debug:
                print(f"[APPLE XML] Response preview: {response.text[:250]}")

            rows = _parse_apple_xml_entries(response.text, app_id, country, app_name=app_name, debug=debug)

            if rows:
                return rows

        except Exception as e:
            if debug:
                print(f"[APPLE XML] Variant {idx} failed: {e}")

    return []


# ----------------------------
# APPLE RSS
# ----------------------------

def fetch_apple_reviews_rss(
    app_id_or_url,
    country,
    app_name=None,
    max_pages=30,
    max_reviews=1000,
    debug=True,
):
    """Fetch Apple App Store reviews from the public RSS feed for a single country."""
    app_id = _extract_apple_app_id(app_id_or_url)
    country = str(country).lower().strip()
    rows = []

    print("\n[APPLE RSS API]")
    print(f"App ID: {app_id} | Country: {country} | App Name: {app_name}")

    effective_max_pages = min(max_pages, 10)
    print(f"[APPLE RSS API] Effective max pages: {effective_max_pages}")

    with requests.Session() as session:
        for page in range(1, effective_max_pages + 1):
            if len(rows) >= max_reviews:
                print("[APPLE RSS API] Max review limit reached")
                break

            print(f"\n[APPLE RSS API] Fetching page {page}")

            page_rows = _fetch_apple_page_json(
                session=session,
                app_id=app_id,
                country=country,
                page=page,
                app_name=app_name,
                debug=debug,
            )

            if not page_rows:
                print(f"[APPLE RSS API] JSON fetch returned no rows for page {page}. Trying XML fallback.")
                page_rows = _fetch_apple_page_xml(
                    session=session,
                    app_id=app_id,
                    country=country,
                    page=page,
                    app_name=app_name,
                    debug=debug,
                )

            if not page_rows:
                print(f"[APPLE RSS API] No rows found for page {page}. Stopping pagination.")
                break

            print(f"[APPLE RSS API] Page {page} yielded {len(page_rows)} rows")
            rows.extend(page_rows)

            # Small delay to reduce the chance of throttling.
            time.sleep(0.5)

    df = pd.DataFrame(rows[:max_reviews])

    if df.empty:
        print("[APPLE RSS API] Final result is empty")
        return df

    # Remove duplicate reviews that may appear across page or country requests.
    df = df.drop_duplicates(subset=["source_store", "app_identifier", "review_id"]).reset_index(drop=True)

    print(f"[APPLE RSS API] Final unique review count: {len(df)}")
    return df


# ----------------------------
# GOOGLE PLAY
# ----------------------------

def fetch_google_play_reviews(app_name, package_name, countries, languages, count_per_request=200):
    """Fetch Google Play reviews for the selected country-language combinations."""
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

    # Fallback to the first selected language if no pair matched.
    if not pairs_to_fetch and countries and languages:
        fallback_lang = cleaned_languages[0]
        for c in countries:
            pairs_to_fetch.append((fallback_lang, str(c).lower().strip()))

    print("\n[GOOGLE PLAY]")
    print(f"Package: {package_name} | App Name: {app_name}")
    print(f"Pairs to fetch: {pairs_to_fetch}")

    for lang, country in pairs_to_fetch:
        token = None
        page_counter = 0

        while True:
            page_counter += 1
            print(f"[GOOGLE PLAY] Fetching page {page_counter} for country={country}, lang={lang}")

            result, token = reviews(
                package_name,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=count_per_request,
                continuation_token=token,
            )

            print(f"[GOOGLE PLAY] Retrieved {len(result)} reviews")

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
                print("[GOOGLE PLAY] No continuation token left")
                break

            time.sleep(0.5)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[GOOGLE PLAY] Final result is empty")
        return df

    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True)
    df["review_date"] = df["review_date"].dt.tz_convert(None)

    df = df.drop_duplicates(subset=["source_store", "app_identifier", "review_id"]).reset_index(drop=True)
    print(f"[GOOGLE PLAY] Final unique review count: {len(df)}")
    return df


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
    """Fetch reviews from the selected stores and merge them into a single DataFrame."""
    countries = countries or DEFAULT_COUNTRIES
    languages = languages or DEFAULT_LANGUAGES

    print("\n[FETCH LIVE REVIEWS]")
    print(f"App name: {app_name}")
    print(f"Google Play ID: {googleplay_app_id}")
    print(f"Apple App ID/URL: {apple_app_id_or_url}")
    print(f"Countries: {countries}")
    print(f"Languages: {languages}")
    print(f"Apple max pages: {apple_max_pages}")

    frames = []

    if googleplay_app_id:
        gpdf = fetch_google_play_reviews(
            app_name=app_name,
            package_name=googleplay_app_id,
            countries=countries,
            languages=languages
        )
        if not gpdf.empty:
            print(f"[FETCH LIVE REVIEWS] Google Play rows: {len(gpdf)}")
            frames.append(gpdf)
        else:
            print("[FETCH LIVE REVIEWS] No Google Play reviews fetched")

    if apple_app_id_or_url:
        apple_country_frames = []
        for country in countries:
            print(f"\n[FETCH LIVE REVIEWS] Fetching Apple reviews for country={country}")
            appledf = fetch_apple_reviews_rss(
                app_id_or_url=apple_app_id_or_url,
                country=country,
                app_name=app_name,
                max_pages=apple_max_pages,
                debug=True,
            )
            if not appledf.empty:
                print(f"[FETCH LIVE REVIEWS] Apple rows for {country}: {len(appledf)}")
                apple_country_frames.append(appledf)
            else:
                print(f"[FETCH LIVE REVIEWS] No Apple reviews fetched for {country}")

        if apple_country_frames:
            apple_merged = pd.concat(apple_country_frames, ignore_index=True)
            apple_merged = apple_merged.drop_duplicates(
                subset=["source_store", "app_identifier", "review_id"]
            ).reset_index(drop=True)
            print(f"[FETCH LIVE REVIEWS] Apple unique merged rows: {len(apple_merged)}")
            frames.append(apple_merged)

    if not frames:
        print("[FETCH LIVE REVIEWS] No frames collected from any store")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True)
    df["review_date"] = df["review_date"].dt.tz_convert(None)

    df = df.drop_duplicates(subset=["source_store", "app_identifier", "review_id"]).reset_index(drop=True)

    print(f"[FETCH LIVE REVIEWS] Final merged unique rows: {len(df)}")
    return df


def filter_date_range(df, start_date, end_date):
    """Filter reviews to the selected inclusive date range."""
    if df.empty:
        print("[FILTER DATE RANGE] Input DataFrame is empty")
        return df.copy()

    out = df.copy()
    out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce", utc=True).dt.tz_convert(None)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    filtered = out[(out["review_date"] >= start_ts) & (out["review_date"] < end_ts)].copy()

    print("\n[FILTER DATE RANGE]")
    print(f"Input rows: {len(out)}")
    print(f"Start: {start_ts} | End (exclusive): {end_ts}")
    print(f"Output rows: {len(filtered)}")

    return filtered