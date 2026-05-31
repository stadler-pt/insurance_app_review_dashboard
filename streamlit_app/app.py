import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
SRC_DIR = APP_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from aggregation import label_summary, review_summary
from charts import plot_label_counts, plot_rating_distribution, plot_review_volume
from config_loader import get_app_config, get_label_metadata, load_json
from clustering import build_word_clusters

import pandas as pd
import streamlit as st

# Streamlit page-level configuration.
st.set_page_config(
    page_title="Review-Dashboard",
    page_icon="📊",
    layout="wide",
)

# Default filter values for country and language selection.
DEFAULT_COUNTRIES = ["de", "fr", "pl", "gb", "us"]
DEFAULT_LANGUAGES = ["de", "en", "fr", "pl"]

# Global configuration objects loaded once during script execution.
# The application configuration contains deployment- and project-specific
# parameters, while label metadata can be used to describe available labels.
cfg = get_app_config()
label_meta = get_label_metadata()

# Paths used for persistence and deployment diagnostics.
# /data is the typical writable location in Hugging Face Spaces when
# persistent storage is available.
BUCKET_ROOT = Path("/data")
HF_CACHE_ROOT = Path("/data/.huggingface")

def bucket_write_test():
    """
    Performs a simple write test in the deployment storage location.

    The function is intended as a lightweight health check for writable
    persistent storage. It attempts to create a small directory and file
    under /data and returns a status flag together with either the created
    file path or the raised exception message.
    """
    try:
        BUCKET_ROOT.mkdir(parents=True, exist_ok=True)
        test_dir = BUCKET_ROOT / "healthcheck"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / "write_test.txt"
        test_file.write_text("bucket write works", encoding="utf-8")
        return True, str(test_file)
    except Exception as e:
        return False, str(e)

@st.cache_data(show_spinner=False)
def get_clustered_df(scored_df):
    """
    Caches the clustering result derived from the scored review DataFrame.

    Clustering is computationally more expensive than simple filtering or
    aggregation. Caching prevents unnecessary recomputation during reruns
    as long as the underlying scored review data remains unchanged.
    """
    return build_word_clusters(scored_df, text_col="reviewtext", n_clusters=6)

# CSS-Styles for the Dashboard
st.markdown("""
<style>
:root {
    --color-primary: #00A9D9;
    --color-primary-dark: #007FA3;
    --color-primary-soft: #EAF7FB;
    --color-bg: #F4F9FB;
    --color-surface: #FFFFFF;
    --color-border: #D8E7EE;
    --color-text: #1F2D3D;
    --color-muted: #607080;
    --color-success-bg: #E8F6EE;
    --color-success-text: #2E7D57;
}
.stApp {
    background: var(--color-bg);
    color: var(--color-text);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: var(--color-text);
}
p, label, .stCaption {
    color: var(--color-muted);
}
/* Inputs / Selectbox / Multiselect */
.stTextInput input,
.stDateInput input,
.stNumberInput input,
div[data-baseweb="select"] > div {
    background: var(--color-surface) !important;
    color: var(--color-text) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: 10px !important;
}
/* Multiselect tags */
[data-baseweb="tag"] {
    background-color: var(--color-primary-soft) !important;
    color: var(--color-primary-dark) !important;
    border: 1px solid #BFE4EF !important;
}
/* Buttons */
.stButton > button {
    background: var(--color-primary);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}
.stButton > button:hover {
    background: var(--color-primary-dark);
}
/* Tabs */
div[data-baseweb="tab-list"] {
    gap: 0.35rem;
}
button[data-baseweb="tab"] {
    background: var(--color-surface);
    color: var(--color-muted);
    border: 1px solid var(--color-border);
    border-bottom: none;
    border-radius: 10px 10px 0 0;
    padding: 0.55rem 0.9rem;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--color-primary-soft);
    color: var(--color-primary-dark);
    border: 1px solid var(--color-border);
    border-bottom: 3px solid var(--color-primary);
    font-weight: 600;
}
/* Metrics / cards */
div[data-testid="stMetric"] {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 14px;
    padding: 1rem;
}
div[data-testid="stMetricLabel"] {
    color: var(--color-muted);
}
div[data-testid="stMetricValue"] {
    color: var(--color-text);
}
/* Dataframes */
div[data-testid="stDataFrame"] {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 14px;
    overflow: hidden;
}
/* Info / success messages */
div[data-testid="stInfo"] {
    background: var(--color-primary-soft);
    border: 1px solid #BFE4EF;
    color: var(--color-text);
    border-radius: 12px;
}
div[data-testid="stAlert"] {
    border-radius: 12px;
}
            
/* DataFrame overall container */
div[data-testid="stDataFrame"] {
    background: #FFFFFF !important;
    border: 1px solid #D8E7EE !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
/* Glide data grid root */
div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
    border: none !important;
}
/* Try to align internal data editor colors */
div.stDataFrameGlideDataEditor {
    --gdg-bg-cell: #FFFFFF !important;
    --gdg-bg-cell-medium: #F8FBFD !important;
    --gdg-bg-header: #EAF7FB !important;
    --gdg-bg-header-has-focus: #D9F0F7 !important;
    --gdg-border-color: #D8E7EE !important;
    --gdg-horizontal-border-color: #E6EEF2 !important;
    --gdg-vertical-border-color: #E6EEF2 !important;
    --gdg-text-dark: #1F2D3D !important;
    --gdg-text-medium: #607080 !important;
    --gdg-accent-color: #00A9D9 !important;
    --gdg-header-font-style: 600 14px sans-serif !important;
}
/* Optional: toolbar above dataframe */
div[data-testid="stDataFrameToolbar"] {
    background: #FFFFFF !important;
    border-bottom: 1px solid #E6EEF2 !important;
}
/* Standard buttons */
.stButton > button {
    background: #00A9D9 !important;
    color: #FFFFFF !important;
    border: 1px solid #00A9D9 !important;
}
.stButton > button p {
    color: #FFFFFF !important;
}
.stButton > button:hover {
    background: #0094BF !important;
    color: #FFFFFF !important;
    border: 1px solid #0094BF !important;
}
.stButton > button:hover p {
    color: #FFFFFF !important;
}
.stButton > button:focus,
.stButton > button:focus-visible,
.stButton > button:active {
    color: #FFFFFF !important;
    border-color: #0094BF !important;
    box-shadow: 0 0 0 0.2rem rgba(0, 169, 217, 0.20) !important;
}
.stButton > button:focus p,
.stButton > button:focus-visible p,
.stButton > button:active p {
    color: #FFFFFF !important;
}                         
</style>
""", unsafe_allow_html=True)

# Mapping from internal column names to user-facing German labels.
DISPLAY_COLUMNS = {
    "reviewdate": "Datum",
    "sourcestore": "Store",
    "country": "Land",
    "language": "Sprache",
    "appname": "App",
    "rating": "Bewertung",
    "reviewtitle": "Reviewtitel",
    "reviewtext": "Reviewtext",
    "label_key": "Erkannte Label",
    "anzahl": "Absolute Anzahl Reviews",
    "durchschnittliche_wahrscheinlichkeit": "Modellwahrscheinlichkeit (Durchschnitt)",
    "predicted_dashboard_labels": "Vorhergesagte Label",
}

# Mapping from technical label identifiers to readable German topic names.
LABEL_NAME_MAP = {
    "auth_registration": "Login & Registrierung",
    "tech_stability_crash": "Performance, Stabilität & Abstürze",
    "general_feedback": "Allgemeines Feedback",
    "document_management": "Dokumentenmanagement",
    "smarthealth_epa_features": "Smart Health / ePA",
    "usability_ui": "Usability",
    "updates_versions": "Updates",
    "customer_service": "Support & Kundenservice",
}

# Prefix used for dynamically generated probability column labels in the detailed review table.
PROB_PREFIX = "Modellwahrscheinlichkeit: "

# Active labels that are currently included in the prediction workflow.
# These labels represent the subset of topics for which the model output is displayed and aggregated in the dashboard.
labels = [
    'auth_registration',
    'tech_stability_crash',
    'general_feedback',
    'document_management',
    'smarthealth_epa_features'
]

# Main page title and introductory caption.
st.title("Automatisierte Analyse von App-Reviews zur Unterstützung des Produktmanagements")
st.caption(
    "Live-Extraktion aus Google Play und Apple App Store mit modellgestützter Analyse vordefinierter Label."
)

# The storage write test is executed once during app initialization.
# The result can be used for diagnostics during deployment.
bucket_ok, bucket_info = bucket_write_test()

# Variables for selected app identifiers are initialized up front and later filled depending on the active store selection.
selected_google_id = None
selected_apple_id = None

# Configuration defaults for the app selection form are read from JSON.
app_config_path = APP_ROOT / "config" / "app_config.json"
app_config_raw = load_json(app_config_path)

def dismiss_welcome_gate():
    """
    Mark the introductory blocking screen as acknowledged
    for the current session.
    """
    st.session_state.welcome_acknowledged = True


if "welcome_acknowledged" not in st.session_state:
    st.session_state.welcome_acknowledged = False


if not st.session_state.welcome_acknowledged:
    with st.container():
        st.subheader("Hinweise zum Dashboard")

        st.warning(
            "**Wichtiger Hinweis: Forschungsprototyp** "
            "Dieses Dashboard ist ein experimenteller Prototyp, der im Rahmen einer Machbarkeitsstudie entwickelt wurde. "
            "Alle dargestellten Auswertungen basieren auf maschinellen Schätzungen (Machine Learning) und sind grundsätzlich fehlerbehaftet. ",
            icon="⚠️"            
        )
        
        st.markdown("### Zweck des Dashboards")
        st.markdown(
            "Dieses Dashboard visualisiert App-Reviews und ordnet diesen mithilfe eines Machine-Learning-Modells automatisch Themenkategorien zu. "
            "Es dient der explorativen Datenanalyse und dem quantitativen Monitoring von Nutzerfeedback zur Unterstützung des Produktmanagements."
        )

        st.divider()

        st.markdown("### Kategorien und Modellzuverlässigkeit")
        st.markdown("Das Modell ordnet App-Reviews automatisch fünf Hauptthemen zu, deren Vorhersagegenauigkeit stark variiert:")

        st.markdown("**🟢 1. Gute Zuverlässigkeit**")
        st.markdown(
            """
            * Login & Registrierung
            * Performance, Stabilität & Abstürze
            """
        )
        st.info(
            "**Hinweis:** Das Modell funktioniert hier am besten. Dennoch gilt: Auch diese Metriken sind Wahrscheinlichkeiten, keine absoluten Wahrheiten. "
            "Ausschläge sind verlässliche Indikatoren, können aber vereinzelte Fehlklassifikationen enthalten.",
            icon="✅"
        )

        st.markdown("**🟡 2. Geringere Zuverlässigkeit**")
        st.markdown(
            """
            * Allgemeines Feedback
            * Dokumentenmanagement
            * Smart Health / ePA
            """
        )
        st.warning(
            "**Hinweis:** Aufgrund geringer Trainingsdaten ist das Modell für diese Themengebiete fehleranfälliger "
            "und erzeugt mehr Fehlklassifikationen (insbesondere False Positives). Aufgrund der praktischen Relevanz "
            "wurden sie jedoch beibehalten. Die Auswertungen zu diesen Themen dienen daher primär der Trenderkennung; "
            "auffällige Ausschläge erfordern eine stichprobenartige, manuelle Prüfung der zugrunde liegenden Texte.",
            icon="⚠️"
        )

        st.markdown("**⚪ 3. Ausgeschlossene Themen**")
        st.markdown(
            "Aufgrund unzureichender Modellleistung werden Themen wie **Usability**, **Updates** und **Kundenservice** "
            "aktuell nicht automatisch klassifiziert. Auch eine Aufgliederung der Themen in die einzelnen App-Funktionen "
            "ist aus diesem Grund derzeit nicht möglich."
        )

        st.divider()

        st.markdown("### Systemgrenzen und Interpretation")
        st.markdown(
            """
            * **Assistenzfunktion:** Die Metriken sind maschinelle Schätzungen. Sie unterstützen die Analyse, ersetzen aber keine qualitative Einzelfallprüfung.
            * **Instabile Datenquelle:** Instabile Datenquelle: Das Laden von Bewertungen aus dem Apple App Store ist technisch bedingt nicht immer stabil. Unangekündigte Änderungen seitens Apple können hin und wieder zu unvollständigen Ergebnissen führen.
            * **Sprachliche Limitierungen:** Sehr kurze Texte, Sarkasmus oder implizite Kritik können zu fehlerhaften Zuordnungen führen.
            * **Daten-Limitierung:** Das zugrunde liegende Modell wurde mit einer begrenzten Datenmenge trainiert. Falsche Schlussfolgerungen sind möglich, wenn die Ergebnisse unreflektiert übernommen werden.
            """
        )

        if st.button("Verstanden", type="primary", key="welcome_ack_button"):
            dismiss_welcome_gate()
            st.rerun()

    st.stop()

# Default form values are read from the configuration file.
app_name_default = app_config_raw.get("app_name", "HEK Service-App")
googleplay_default = app_config_raw.get("googleplay_app_id", "de.hek.serviceapp")
apple_default = app_config_raw.get("apple_app_id", "1287511413")

# The top section of the interface contains all controls needed to define the review extraction query.
selection_card = st.container()
with selection_card:
    st.subheader("Review-Auswahl")

    c1, c2, c3 = st.columns(3)

    with c1:
        app_name = st.text_input("App-Name", value=app_name_default)
        googleplay_app_id = st.text_input("Google-Play-App-ID", value=googleplay_default)

    with c2:
        apple_app_id = st.text_input("Apple-App-ID oder URL", value=apple_default)
        stores = st.multiselect(
            "Stores",
            options=["googleplay", "appleappstore"],
            default=["googleplay", "appleappstore"],
            format_func=lambda x: "Google Play" if x == "googleplay" else "Apple App Store",
        )

    # Column 3 contains the date range selector with a default lookback window of 180 days.
    with c3:
        today = pd.Timestamp.today().date()
        default_start = (pd.Timestamp.today() - pd.Timedelta(days=180)).date()
        date_range = st.date_input(
            "Zeitraum",
            value=(default_start, today),
            max_value=today,
        )

    c4, c5 = st.columns(2)

    with c4:
        countries = st.multiselect(
            "Länder",
            options=DEFAULT_COUNTRIES,
            default=DEFAULT_COUNTRIES,
        )

    with c5:
        languages = st.multiselect(
            "Sprachen",
            options=DEFAULT_LANGUAGES,
            default=DEFAULT_LANGUAGES,
        )

    # Validation and unpacking of the date range input.
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, today

    # Main action button that triggers the full pipeline:
    # extraction, filtering, preprocessing, and model inference.
    fetch_clicked = st.button(
        "Reviews live laden",
        type="primary",
        use_container_width="stretch",
    )

# Session state container for the processed review data.
# This allows the resulting DataFrame to persist across reruns after the user has triggered the analysis once.
if "scored_df" not in st.session_state:
    st.session_state["scored_df"] = None

# Placeholders for a progress bar and a status container.
progress_bar = st.empty()
status_placeholder = st.empty()

if fetch_clicked:
    # Heavy modules are imported only when needed. This reduces startup cost
    # during initial page rendering and keeps the app responsive until the user explicitly starts the analysis pipeline.
    from inference import predict_dashboard_labels
    from preprocessing import prepare_reviews
    from review_fetchers import fetch_live_reviews, filter_date_range
    
    selected_google_id = googleplay_app_id.strip() if "googleplay" in stores else None
    selected_apple_id = apple_app_id.strip() if "appleappstore" in stores else None

    if not selected_google_id and not selected_apple_id:
        st.error("Bitte mindestens einen Store mit gültiger App-ID auswählen.")
    elif not countries or not languages:
        st.error("Bitte mindestens ein Land und eine Sprache auswählen.")
    else:
        progress_bar = st.progress(0, text="Initialisiere...")
        status_placeholder = st.empty()

        with status_placeholder.container():
            with st.status("Starte Review-Verarbeitung...", expanded=True) as status:
                # Step 1: Live extraction from the selected app stores.
                st.write("1/4 Extrahiere Reviews aus den Stores...")

                live_df = fetch_live_reviews(
                    app_name=app_name.strip() or app_name_default,
                    googleplay_app_id=selected_google_id,
                    apple_app_id_or_url=selected_apple_id,
                    countries=countries,
                    languages=languages,
                    apple_max_pages=20,
                )

                raw_count = len(live_df) if live_df is not None else 0
                progress_bar.progress(
                    35,
                    text=f"Extraktion abgeschlossen: {raw_count} Reviews gefunden"
                )
                st.write(f"✓ Extraktion abgeschlossen: {raw_count} Reviews gefunden")

                # Step 2: Filter reviews by the selected date range.
                st.write("2/4 Filtere Reviews nach Datumsbereich...")
                live_df = filter_date_range(live_df, start_date, end_date)

                filtered_count = len(live_df) if live_df is not None else 0
                progress_bar.progress(
                    55,
                    text=f"Datumsfilter abgeschlossen: {filtered_count} Reviews im gewählten Zeitraum"
                )
                st.write(
                    f"✓ Datumsfilter abgeschlossen: {filtered_count} Reviews im gewählten Zeitraum"
                )

                # Step 3: Preprocessing the raw reviews for inference.
                st.write("3/4 Bereite Reviews für die Analyse vor...")
                live_df = prepare_reviews(live_df)

                prepared_count = len(live_df) if live_df is not None else 0
                progress_bar.progress(
                    75,
                    text=f"Vorverarbeitung abgeschlossen: {prepared_count} Reviews bereit für Analyse"
                )
                st.write(
                    f"✓ Vorverarbeitung abgeschlossen: {prepared_count} Reviews bereit für Analyse"
                )

                # Step 4: Model inference assigns the dashboard labels and stores associated probabilities.
                st.write("4/4 Analysiere Reviews mit dem Modell... (dies kann einige Minuten dauern)")

                if live_df.empty:
                    st.session_state["scored_df"] = None
                    progress_bar.progress(
                        100,
                        text="Keine Reviews zur Analyse vorhanden"
                    )
                    st.write("⚠ Keine Reviews zur Analyse vorhanden")
                    status.update(
                        label="Abgeschlossen: Keine Reviews im gewählten Zeitraum gefunden",
                        state="complete",
                        expanded=False,
                    )
                else:
                    # The full scored review DataFrame is stored in session state so that all tabs can reuse the result.
                    st.session_state["scored_df"] = predict_dashboard_labels(live_df)

                    scored_count = (
                        len(st.session_state["scored_df"])
                        if st.session_state["scored_df"] is not None
                        else 0
                    )
                    progress_bar.progress(
                        100,
                        text=f"Analyse abgeschlossen: {scored_count} Reviews analysiert"
                    )
                    st.write(f"✓ Analyse abgeschlossen: {scored_count} Reviews analysiert")

                    status.update(
                        label=f"Abgeschlossen: {scored_count} Reviews extrahiert und analysiert",
                        state="complete",
                        expanded=False,
                    )

# The analyzed review dataset is retrieved from session state.
scored_df = st.session_state.get("scored_df")

# If no data is available yet, the app stops here after presenting a helpful hint.
if scored_df is None or scored_df.empty:
    st.info(
        "Noch keine Reviews geladen. Bitte oben die Suchkriterien festlegen und anschließend auf „Reviews live laden“ klicken."
    )
    st.stop()

# Summary statistics are computed once and then reused in the overview and label analysis sections.
summary = review_summary(scored_df)
label_df = label_summary(scored_df, labels)

st.success(f"{len(scored_df)} Reviews wurden live geladen und analysiert.")

# Collapsible explanation of the available labels.
with st.expander("📖 Legende: Was bedeuten die verschiedenen Label?", expanded=False):
    st.markdown("""
    Das System teilt die Bewertungen automatisch in fünf Themenbereiche ein:
    
    * **Login & Registrierung:** Probleme oder Fragen rund um das Anmelden, Passwörter, PINs oder die Ersteinrichtung der App.
    * **Performance, Stabilität & Abstürze:** Wenn die App langsam lädt, einfriert, unerwartet schließt (Absturz) oder Internet-Verbindungsfehler anzeigt.
    * **Allgemeines Feedback:** Pauschales Lob ("Tolle App!") oder allgemeiner Frust, ohne dass ein konkretes technisches Problem genannt wird.
    * **Dokumentenmanagement:** Alles rund um das Herunterladen und Einreichen von Dokumenten oder deren Handhabung.
    * **Smart Health / ePA:** Bewertungen, die sich speziell auf die elektronische Patientenakte (ePA) oder andere Smart Health-Funktionen beziehen.
    """)

# The application is structured into four thematic tabs:
# overview, label analysis, detailed review inspection, and word clustering.
tab1, tab2, tab3, tab4 = st.tabs(
    ["Übersicht", "Label-Analyse", "Einzelreviews", "Wortcluster"],
    key="main_tabs"
)

clustered_df = get_clustered_df(scored_df)

with tab1:
    # Tab 1 presents the high-level descriptive overview of the review corpus.
    st.subheader("Überblick")

    st.caption(
    "Dieser Tab zeigt eine zusammenfassende Übersicht der extrahierten Reviews. "
    "Dargestellt werden Anzahl, Herkunft, zeitliche Verteilung und Bewertungsverteilung der aktuell geladenen Reviews."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Anzahl Reviews", int(summary.get("total_reviews", len(scored_df))))

    avg_rating = summary.get("avg_rating")
    m2.metric(
        "Durchschnittsbewertung",
        round(float(avg_rating), 2) if avg_rating is not None else "n/a",
    )

    m3.metric("Google Play", int(summary.get("google_reviews", 0)))
    m4.metric("Apple App Store", int(summary.get("apple_reviews", 0)))

    top_left, top_right = st.columns(2)

    volume_chart = plot_review_volume(scored_df)
    rating_chart = plot_rating_distribution(scored_df)

    if volume_chart:
        top_left.plotly_chart(volume_chart, use_container_width="stretch")

    if rating_chart:
        top_right.plotly_chart(rating_chart, use_container_width="stretch")

with tab2:
    # Tab 2 focuses on aggregated topic distributions across the predicted label space.
    st.subheader("Label-Analyse")

    st.caption(
        "Dieser Tab zeigt, wie häufig die einzelnen Themenlabels in den analysierten Reviews vorkommen. "
        "Die durchschnittliche Wahrscheinlichkeit beschreibt, mit welcher mittleren Modellsicherheit ein Label vorhergesagt wurde. "
        "Die Modellwahrscheinlichkeit beschreibt, wie stark das Modell das jeweilige Label für eine Review unterstützt."
    )

    label_chart = plot_label_counts(label_df)

    if label_chart:
        st.plotly_chart(label_chart, use_container_width="stretch")

    label_df_display = label_df.copy()

    label_df_display["label_key"] = label_df_display["label_key"].map(
        lambda x: LABEL_NAME_MAP.get(x, x)
    )

    label_df_display = label_df_display.rename(columns=DISPLAY_COLUMNS)

    st.data_editor(
        label_df_display,
        width='stretch',
        hide_index=True,
        disabled=True,
        key="tab2"
    )

with tab3:
    # Tab 3 exposes the individual review-level records and thereby enables manual validation and deeper inspection of model output.
    st.subheader("Einzelreviews")

    st.caption(
        "Dieser Tab zeigt die einzelnen extrahierten Reviews einschließlich ihrer Metadaten und Modellvorhersagen. "
        "Er dient dazu, aggregierte Ergebnisse aus den anderen Tabs auf Review-Ebene nachzuvollziehen und ermöglicht bei Bedarf eine detaillierte manuelle Analyse. "
        "Die Modellwahrscheinlichkeit beschreibt, wie stark das Modell das jeweilige Label für eine Review unterstützt."
    )

    f1, f2, f3 = st.columns([1, 1, 2])

    with f1:
        selected_label = st.selectbox(
            "Label-Filter",
            ["Alle"] + labels,
            format_func=lambda x: "Alle" if x == "Alle" else LABEL_NAME_MAP.get(x, x)
        )

    with f2:
        rating_options = (
            sorted([int(x) for x in scored_df["rating"].dropna().unique().tolist()])
            if "rating" in scored_df.columns
            else []
        )
        selected_ratings = st.multiselect(
            "Bewertung",
            options=rating_options,
            default=rating_options,
        )

    with f3:
        search_term = st.text_input("Suchbegriff im Reviewtext")

    view_df = scored_df.copy()

    if selected_label != "Alle" and f"pred_{selected_label}" in view_df.columns:
        view_df = view_df[view_df[f"pred_{selected_label}"] == 1].copy()

    if selected_ratings and "rating" in view_df.columns:
        view_df = view_df[view_df["rating"].isin(selected_ratings)].copy()

    if search_term.strip() and "full_text" in view_df.columns:
        view_df = view_df[
            view_df["full_text"].str.contains(search_term.strip(), case=False, na=False)
        ].copy()

    show_cols = [
        c for c in [
            "reviewdate",
            "sourcestore",
            "country",
            "language",
            "appname",
            "rating",
            "reviewtext",
            "predicted_dashboard_labels",
        ]
        if c in view_df.columns
    ]

    prob_cols = [f"prob_{label}" for label in labels if f"prob_{label}" in view_df.columns]

    if "reviewdate" in view_df.columns:
        view_df = view_df.sort_values("reviewdate", ascending=False)

    table_df = view_df[show_cols + prob_cols].copy()

    if "predicted_dashboard_labels" in view_df.columns:
        def translate_labels(val):
            if pd.isna(val) or not val:
                return ""
            # Falls das Modell die Label bereits als Liste übergibt
            if isinstance(val, list):
                return ", ".join([LABEL_NAME_MAP.get(l, l) for l in val])
            # Falls es ein Semikolon-separierter String ist (wie in den Daten zu sehen)
            if isinstance(val, str):
                parts = [p.strip() for p in val.split(";") if p.strip()]
                return "; ".join([LABEL_NAME_MAP.get(p, p) for p in parts])
            return str(val)

        view_df["predicted_dashboard_labels"] = view_df["predicted_dashboard_labels"].apply(translate_labels)

    table_df = view_df[show_cols + prob_cols].copy()

    display_map = DISPLAY_COLUMNS.copy()

    for label in labels:
        prob_col = f"prob_{label}"
        if prob_col in table_df.columns:
            pretty_label = LABEL_NAME_MAP.get(label, label)
            display_map[prob_col] = f"{PROB_PREFIX}{pretty_label}"

    table_df = table_df.rename(columns=display_map)

    st.data_editor(
        table_df,
        width='stretch',
        hide_index=True,
        disabled=True,
        key="tab3",
        column_config={
            "Datum": st.column_config.DatetimeColumn("Datum", format="DD-MM-YYYY"),
            "Bewertung": st.column_config.NumberColumn("Bewertung", format="%d"),
            "Reviewtitel": st.column_config.TextColumn("Reviewtitel", width="medium"),
            "Reviewtext": st.column_config.TextColumn("Reviewtext", width="large"),
        },
    )

with tab4:
    # Tab 4 performs exploratory clustering on the review texts in order to reveal recurring linguistic patterns beyond the predefined labels.
    st.subheader("Wortcluster")

    st.caption(
        "Dieser Tab gruppiert Reviews nach inhaltlicher Ähnlichkeit. "
        "Die Cluster dienen der explorativen Themenanalyse und können helfen, wiederkehrende Muster sowie mögliche neue oder unscharfe Labels zu erkennen. "
        "Die Top-Phrasen geben einen ersten Hinweis auf die zentralen Begriffe innerhalb der Cluster. "
        "Es werden grundsätzlich sechs verschiedene Cluster gebildet."
    )

    if clustered_df.empty:
        st.info("Für die aktuelle Auswahl konnten keine Cluster gebildet werden.")
    else:
        cluster_options = sorted(clustered_df["cluster_id"].unique().tolist())
        selected_cluster = st.selectbox(
            "Cluster auswählen",
            options=cluster_options,
            format_func=lambda x: f"Cluster {x + 1}",
            key="cluster_selectbox"
        )

        cluster_view = clustered_df[clustered_df["cluster_id"] == selected_cluster].copy()

        keywords = cluster_view["cluster_keywords"].iloc[0]
        
        st.markdown(f"**Top-Phrasen:** {keywords}")
        st.markdown(f"**Anzahl Reviews:** {len(cluster_view)}")

        cluster_view["cluster_label"] = cluster_view["cluster_keywords"].str.split(",").str[:3].str.join(", ")

        cols_to_show = [c for c in ["reviewdate", "appname", "rating", "reviewtext"] if c in cluster_view.columns]

        cluster_table = cluster_view[cols_to_show].copy()
        cluster_table = cluster_table.rename(columns=DISPLAY_COLUMNS)

        st.data_editor(
            cluster_table,
            width='stretch',
            hide_index=True,
            disabled=True,
            key="tab4"
        )