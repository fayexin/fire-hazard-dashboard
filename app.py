from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Wildfire Activity and Probability Dashboard",
    layout="wide",
)


RECENT_DATA_PATHS = [
    Path("data/active_fire/firms_viirs_snpp_nrt_recent.parquet"),
    Path("data/fires/viirs_west_recent.parquet"),
]


@st.cache_data
def load_recent_snapshot(path_str: str, file_mtime: float) -> pd.DataFrame:
    """Load the most recent active-fire snapshot if it exists."""
    data = pd.read_parquet(path_str)
    data["acq_date"] = pd.to_datetime(data["acq_date"], errors="coerce")
    data["frp"] = pd.to_numeric(data.get("frp"), errors="coerce")
    return data.dropna(subset=["acq_date"])


def find_recent_data_path() -> Path | None:
    for path in RECENT_DATA_PATHS:
        if path.exists():
            return path
    return None


def format_snapshot_age(latest_detection: pd.Timestamp) -> str:
    days_since = (date.today() - latest_detection.date()).days
    if days_since <= 1:
        return "Updated within the last day"
    if days_since <= 7:
        return f"Most recent detection {days_since} days ago"
    return f"Most recent detection {days_since} days ago; snapshot may be stale"


def page_card(title: str, status: str, description: str) -> None:
    st.markdown(
        f"""
        <div style="border: 1px solid rgba(250, 250, 250, 0.18); border-radius: 0.75rem; padding: 1rem; min-height: 11rem;">
            <h3 style="margin-top: 0;">{title}</h3>
            <p style="font-size: 0.9rem; opacity: 0.75;"><b>Status:</b> {status}</p>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("Wildfire Activity and Probability Dashboard — U.S. West")

st.write(
    "This project analyzes wildfire activity across the western United States using "
    "satellite active-fire detections, historical fire records, environmental data, "
    "and an experimental monthly fire probability model."
)

st.warning(
    "Research and visualization use only. This dashboard is not an official fire "
    "forecast, evacuation tool, or emergency warning system."
)

recent_path = find_recent_data_path()

if recent_path is None:
    st.info(
        "No recent active-fire snapshot was found. Run `python fetch_firms.py --recent` "
        "from the repository root to create the current FIRMS snapshot."
    )
else:
    recent = load_recent_snapshot(str(recent_path), recent_path.stat().st_mtime)

    latest = recent["acq_date"].max()
    earliest = recent["acq_date"].min()
    max_frp = recent["frp"].max(skipna=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recent detections", f"{len(recent):,}")
    col2.metric("Date range", f"{earliest.date()} → {latest.date()}")
    col3.metric("Maximum FRP", f"{max_frp:.0f} MW" if pd.notna(max_frp) else "—")
    col4.metric("Snapshot status", format_snapshot_age(latest))

    st.caption(f"Loaded active-fire snapshot: `{recent_path}`")

st.divider()

st.header("What this dashboard shows")

show_col, not_col = st.columns(2)

with show_col:
    st.subheader("Included")
    st.write(
        "Satellite-detected active fire pixels, recent fire activity maps, historical "
        "fire-event views, trend summaries, and an experimental county-month fire "
        "occurrence probability model."
    )

with not_col:
    st.subheader("Not included")
    st.write(
        "Official fire perimeters, evacuation guidance, structure-level risk, or "
        "public-safety forecasts. FIRMS detections are thermal anomaly pixels, not "
        "complete burned-area maps."
    )

st.divider()

st.header("Dashboard sections")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    page_card(
        "Live Fire Activity",
        "Available",
        "Map recent VIIRS active-fire detections. Filter by fire radiative power, "
        "confidence, recency, and display style.",
    )

with row1_col2:
    page_card(
        "Fire Probability Model",
        "Next build",
        "Estimate monthly county-level fire occurrence probability from historical "
        "fire records, weather, drought, terrain, fuels, and seasonality.",
    )

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    page_card(
        "Historical Event Explorer",
        "Planned",
        "Explore major wildfire events using event windows, detection maps, daily "
        "activity curves, and fire radiative power summaries.",
    )

with row2_col2:
    page_card(
        "Fire Trends",
        "Planned",
        "Analyze annual activity, total fire radiative power, seasonal timing, and "
        "state-level differences while keeping sensor sources separate.",
    )

row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    page_card(
        "Fire Hazard Context",
        "Planned",
        "Show environmental layers related to fire behavior, including drought, "
        "weather, fuels, vegetation, and terrain.",
    )

with row3_col2:
    page_card(
        "Data and Methods",
        "Planned",
        "Document data sources, processing steps, prediction target, model validation, "
        "limitations, and citations.",
    )

st.divider()

st.header("Data sources and modeling plan")

st.write(
    "The current live page uses NASA FIRMS VIIRS active-fire detections. The first "
    "prediction page will use a county-month target: whether at least one wildfire "
    "occurred in a county during a selected month. The model will start with logistic "
    "regression and a tree-based model, then report probability calibration and "
    "time-based validation metrics."
)

st.caption("Use the sidebar to open the available pages.")