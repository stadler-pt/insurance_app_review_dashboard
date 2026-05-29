import sys
from pathlib import Path

from aggregation import label_summary, review_summary
from charts import plot_label_counts, plot_rating_distribution, plot_review_volume
from config_loader import get_app_config, get_label_metadata, load_json
from inference import load_model_bundle, predict_dashboard_labels
from preprocessing import prepare_reviews
from clustering import build_word_clusters
from review_fetchers import (
    DEFAULT_COUNTRIES,
    DEFAULT_LANGUAGES,
    fetch_live_reviews,
    filter_date_range,
)

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Review-Dashboard",
    page_icon="📊",
    layout="wide",
)

cfg = get_app_config()
label_meta = get_label_metadata()

APP_ROOT = Path(__file__).resolve().parent
SRC_DIR = APP_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

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

PROB_PREFIX = "Modellwahrscheinlichkeit: "

def close_welcome_dialog():
    st.session_state.show_welcome_dialog = False

labels = [
    'auth_registration',
    'tech_stability_crash',
    'general_feedback',
    'document_management',
    'smarthealth_epa_features'
]

st.title("Automatisierte Analyse von App-Reviews zur Unterstützung des Produktmanagements")
st.caption(
    "Live-Extraktion aus Google Play und Apple App Store mit modellgestützter Analyse vordefinierter Label."
)

selected_google_id = None
selected_apple_id = None

app_config_path = APP_ROOT / "config" / "app_config.json"
app_config_raw = load_json(app_config_path)

if "show_welcome_dialog" not in st.session_state:
    st.session_state.show_welcome_dialog = True

@st.dialog("Hinweise zum Dashboard", width="large")
def show_welcome_dialog():
    st.markdown("#### Zweck des Dashboards")
    st.markdown(
        "Dieses Dashboard visualisiert App-Reviews und ordnet diesen mithilfe eines Machine-Learning-Modells "
        "automatisch Themenkategorien zu. Es dient der explorativen Datenanalyse und dem quantitativen Monitoring "
        "von Nutzerfeedback zur Unterstützung des Produktmanagements."
    )
    
    st.divider()
    
    st.markdown("#### Kategorien und Modellzuverlässigkeit")
    st.markdown("Das Dashboard zeigt fünf Hauptthemen, die sich in ihrer Vorhersagegenauigkeit unterscheiden:")
    
    st.markdown("**🟢 1. Gute Zuverlässigkeit**")
    st.markdown(
        """
        * Login & Registrierung
        * Performance, Stabilität & Abstürze
        """
    )
    st.info("**Hinweis:** Das Modell besitzt hier eine hohe Präzision. Ausschläge in diesen Metriken sind verlässlich.", icon="✅")
    
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
    
    st.markdown("#### Systemgrenzen und Interpretation")
    st.markdown(
        """
        * **Assistenzfunktion:** Die Metriken sind maschinelle Schätzungen. Sie unterstützen die Analyse, ersetzen aber keine qualitative Einzelfallprüfung.
        * **Technische Datenextraktion (App Store):** Das automatische Laden von Bewertungen aus dem Apple App Store läuft nicht immer ganz stabil. Da Apple die technischen Regeln dafür jederzeit ohne Ankündigung ändern kann, kann es hin und wieder zu kurzen Ausfällen oder unvollständigen Ergebnissen kommen.
        * **Sprachliche Limitierungen:** Sehr kurze Texte, Sarkasmus oder implizite Kritik können zu fehlerhaften Zuordnungen führen.
        * **Limitierungen durch Trainingssdaten:** Es handelt sich bei diesem Dashboard um einen Prototypen, der mit einer begrenzten Anzahl von Daten trainiert wurde. Die Ergebnisse sollten vor diesem Hintergrund interpretiert werden. Insbesondere bei den weniger zuverlässigen Kategorien können Fehlklassifikationen auftreten, die zu falschen Schlussfolgerungen führen können, wenn sie nicht manuell überprüft werden.
        """
    )

    if st.button("Verstanden", type="primary"):
        st.session_state.show_welcome_dialog = False
        st.rerun()

if st.session_state.show_welcome_dialog:
    show_welcome_dialog()

app_name_default = app_config_raw.get("app_name", "HEK Service-App")
googleplay_default = app_config_raw.get("googleplay_app_id", "de.hek.serviceapp")
apple_default = app_config_raw.get("apple_app_id", "1287511413")

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

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, today

    fetch_clicked = st.button(
        "Reviews live laden",
        type="primary",
        use_container_width="stretch",
    )

if "scored_df" not in st.session_state:
    st.session_state["scored_df"] = None

progress_bar = st.empty()
status_placeholder = st.empty()

if fetch_clicked:
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

                st.write("4/4 Analysiere Reviews mit dem Modell...")

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
                    # --- CHANGED: Call the updated inference function ---
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

scored_df = st.session_state.get("scored_df")

if scored_df is None or scored_df.empty:
    st.info(
        "Noch keine Reviews geladen. Bitte oben die Suchkriterien festlegen und anschließend auf „Reviews live laden“ klicken."
    )
    st.stop()

summary = review_summary(scored_df)
label_df = label_summary(scored_df, labels)

st.success(f"{len(scored_df)} Reviews wurden live geladen und analysiert.")


with st.expander("📖 Legende: Was bedeuten die erkannten Label?", expanded=False):
    st.markdown("""
    Das System teilt die Bewertungen automatisch in fünf Themenbereiche ein:
    
    * **Login & Registrierung:** Probleme oder Fragen rund um das Anmelden, Passwörter, PINs oder die Ersteinrichtung der App.
    * **Performance, Stabilität & Abstürze:** Wenn die App langsam lädt, einfriert, unerwartet schließt (Absturz) oder Internet-Verbindungsfehler anzeigt.
    * **Allgemeines Feedback:** Pauschales Lob ("Tolle App!") oder allgemeiner Frust, ohne dass ein konkretes technisches Problem genannt wird.
    * **Dokumentenmanagement:** Alles rund um das Herunterladen und Einreichen von Dokumenten oder deren Handhabung.
    * **Smart Health / ePA:** Bewertungen, die sich speziell auf die elektronische Patientenakte (ePA) oder andere Smart Health-Funktionen beziehen.
    """)


tab1, tab2, tab3, tab4 = st.tabs(
    ["Übersicht", "Label-Analyse", "Einzelreviews", "Wortcluster"],
    key="main_tabs"
)

clustered_df = build_word_clusters(scored_df, text_col="reviewtext", n_clusters=6)

with tab1:
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
    st.subheader("Einzelreviews")

    st.caption(
        "Dieser Tab zeigt die einzelnen extrahierten Reviews einschließlich ihrer Metadaten und Modellvorhersagen. "
        "Er dient dazu, aggregierte Ergebnisse aus den anderen Tabs auf Review-Ebene nachzuvollziehen und ermöglicht bei Bedarf eine detaillierte manuelle Analyse. "
        "Die Modellwahrscheinlichkeit beschreibt, wie stark das Modell das jeweilige Label für eine Review unterstützt."
    )

    f1, f2, f3 = st.columns([1, 1, 2])

    with f1:
        selected_label = st.selectbox("Label-Filter", ["Alle"] + labels)

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