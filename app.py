from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Wildfire Activity Dashboard",
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
        <div style="
            border: 1px solid rgba(250, 250, 250, 0.18);
            border-radius: 0.75rem;
            padding: 1rem;
            min-height: 11rem;
        ">
            <h3 style="margin-top: 0;">{title}</h3>
            <p style="font-size: 0.9rem; opacity: 0.75;">
                <b>Status:</b> {status}
            </p>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("Wildfire Activity Dashboard — U.S. West")

st.write(
    "This dashboard maps recent and historical satellite-detected wildfire activity "
    "across the western United States using NASA FIRMS VIIRS active-fire detections. "
    "The current version includes live fire activity, selected historical fire events, "
    "and long-term fire trend summaries."
)

st.warning(
    "Research and visualization use only. This dashboard is not an official fire "
    "forecast, evacuation tool, emergency warning system, fire perimeter map, or "
    "burned-area product."
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
    col3.metric("Maximum observed FRP", f"{max_frp:.0f} MW" if pd.notna(max_frp) else "—")
    col4.metric("Snapshot status", format_snapshot_age(latest))

    st.caption(f"Loaded recent active-fire snapshot: `{recent_path}`")

st.divider()

st.header("Dashboard sections")

row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    page_card(
        "Live Fire Activity",
        "Available",
        "Map recent VIIRS active-fire detections. Filter by date, fire radiative "
        "power, confidence, day/night flag, and FIRMS source.",
    )

with row1_col2:
    page_card(
        "Historical Fire Events",
        "Available",
        "Explore selected historical wildfire event windows using FIRMS detections. "
        "View FRP heatmaps, detection points, daily counts, and event tables.",
    )

with row1_col3:
    page_card(
        "Fire Trends",
        "Available",
        "Analyze annual, seasonal, year-month, and daily summaries built from "
        "standard-processed historical FIRMS detections.",
    )

st.divider()

st.header("What the data mean")

show_col, not_col = st.columns(2)

with show_col:
    st.subheader("Included")
    st.write(
        "Satellite-detected active-fire pixels, recent activity maps, selected "
        "historical event windows, detection-count summaries, and observed FRP "
        "summaries."
    )

with not_col:
    st.subheader("Not included")
    st.write(
        "Official fire perimeters, evacuation guidance, structure-level risk, "
        "fire spread forecasts, or official burned-area estimates. FIRMS detections "
        "are thermal anomaly pixels, not complete fire boundaries."
    )

st.divider()

st.header("Current data workflow")

st.write(
    "The live page uses a recent near-real-time FIRMS snapshot. Historical event and "
    "trend pages use standard-processed VIIRS files and smaller derived summary "
    "datasets. Full raw historical files are used as local processing inputs and "
    "should not be committed directly unless they are intentionally needed by a page."
)

st.caption("Use the sidebar to open the available dashboard pages.")