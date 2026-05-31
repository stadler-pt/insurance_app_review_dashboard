import pandas as pd
import plotly.graph_objects as go
from config_loader import get_label_metadata

# Color palette used across all charts for a consistent visual style.
COLOR_PRIMARY = "#00A9D9"
COLOR_PRIMARY_DARK = "#007FA3"
COLOR_PRIMARY_LIGHT = "#7CCFE6"

COLOR_TEXT = "#1F2D3D"
COLOR_TEXT_MUTED = "#607080"

COLOR_GRID = "#E6EEF2"
COLOR_BORDER = "#D8E7EE"
COLOR_BACKGROUND = "#FFFFFF"

# Mapping from internal label keys to display names used in the charts.
LABEL_DISPLAY_MAP = {
    "auth_registration": "Login & Registrierung",
    "tech_stability_crash": "Performance, Stabilität & Abstürze",
    "general_feedback": "Allgemeines Feedback",
    "document_management": "Dokumentenmanagement",
    "smarthealth_epa_features": "Smart Health / ePA",
    "usability_ui": "Usability",
    "updates_versions": "Updates",
    "customer_service": "Support & Kundenservice",
}


def apply_chart_style(fig):
    """Apply a shared layout and axis style to all Plotly figures."""
    if fig is None:
        return None

    fig.update_layout(
        paper_bgcolor=COLOR_BACKGROUND,
        plot_bgcolor=COLOR_BACKGROUND,
        font=dict(color=COLOR_TEXT),
        margin=dict(l=20, r=20, t=60, b=20),
        title_font=dict(color=COLOR_TEXT, size=18),
        hoverlabel=dict(
            bgcolor=COLOR_BACKGROUND,
            bordercolor=COLOR_BORDER,
            font=dict(color=COLOR_TEXT),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        showline=True,
        linecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_TEXT_MUTED),
        title_font=dict(color=COLOR_TEXT),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        showline=True,
        linecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_TEXT_MUTED),
        title_font=dict(color=COLOR_TEXT),
    )

    return fig


def plot_review_volume(df):
    """Create a cumulative review volume chart based on review dates."""
    if df is None or df.empty or "reviewdate" not in df.columns:
        return None

    # Convert dates and remove invalid values before aggregation.
    chart_df = df.copy()
    chart_df["reviewdate"] = pd.to_datetime(chart_df["reviewdate"], errors="coerce")
    chart_df = chart_df.dropna(subset=["reviewdate"])

    if chart_df.empty:
        return None

    # Aggregate reviews by day and compute the cumulative total.
    daily_counts = (
        chart_df.assign(datum=chart_df["reviewdate"].dt.floor("D"))
        .groupby("datum", as_index=False)
        .agg(reviews_pro_tag=("datum", "size"))
        .sort_values("datum")
    )

    daily_counts["kumulierte_reviews"] = daily_counts["reviews_pro_tag"].cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_counts["datum"].tolist(),
            y=daily_counts["kumulierte_reviews"].tolist(),
            mode="lines+markers",
            name="Kumulierte Reviews",
            line=dict(color=COLOR_PRIMARY, width=3),
            marker=dict(color=COLOR_PRIMARY_DARK, size=7),
        )
    )

    fig.update_layout(
        title="Kumulierte Anzahl der extrahierten Reviews",
        xaxis_title="Datum",
        yaxis_title="Kumulierte Reviews",
        showlegend=False,
    )

    fig = apply_chart_style(fig)
    return fig


def plot_rating_distribution(df):
    """Create a bar chart showing the distribution of review ratings."""
    if df is None or df.empty or "rating" not in df.columns:
        return None

    # Convert ratings to numeric values and keep only valid star ratings.
    chart_df = df.copy()
    chart_df["rating"] = pd.to_numeric(chart_df["rating"], errors="coerce")
    chart_df = chart_df.dropna(subset=["rating"])
    chart_df = chart_df[chart_df["rating"].between(1, 5)]

    if chart_df.empty:
        return None

    # Count rating frequencies for the full 1-to-5 range.
    rating_counts = (
        chart_df["rating"]
        .astype(int)
        .value_counts()
        .reindex([1, 2, 3, 4, 5], fill_value=0)
        .sort_index()
    )

    x_vals = [1, 2, 3, 4, 5]
    y_vals = [int(rating_counts.loc[x]) for x in x_vals]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals,
            text=y_vals,
            textposition="outside",
            name="Bewertungen",
            marker=dict(
                color=COLOR_PRIMARY,
                line=dict(color=COLOR_PRIMARY_DARK, width=1),
            ),
        )
    )

    fig.update_layout(
        title="Bewertungsverteilung der extrahierten Reviews",
        xaxis_title="Bewertung",
        yaxis_title="Anzahl",
        xaxis=dict(
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5],
            type="category",
        ),
        yaxis=dict(rangemode="tozero"),
        showlegend=False,
    )

    fig = apply_chart_style(fig)
    return fig


def plot_label_counts(label_df):
    """Create a bar chart showing the number of reviews per predicted label."""
    if label_df is None or label_df.empty:
        return None

    plot_df = label_df.copy()

    # Ensure the required aggregation columns are present.
    if "label_key" not in plot_df.columns or "anzahl" not in plot_df.columns:
        return None

    # Replace internal label keys with human-readable display names.
    x_labels = [
        LABEL_DISPLAY_MAP.get(str(x), str(x))
        for x in plot_df["label_key"].tolist()
    ]
    y_counts = [
        int(x) if x is not None else 0
        for x in plot_df["anzahl"].tolist()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_counts,
            text=y_counts,
            textposition="outside",
            marker=dict(
                color=COLOR_PRIMARY,
                line=dict(color=COLOR_PRIMARY_DARK, width=1),
            ),
        )
    )

    fig.update_layout(
        title="Label-Häufigkeiten",
        xaxis_title="Label",
        yaxis_title="Anzahl",
        yaxis=dict(rangemode="tozero"),
        showlegend=False,
    )

    fig = apply_chart_style(fig)
    return fig