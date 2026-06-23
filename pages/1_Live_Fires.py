import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

import pydeck.bindings.json_tools as _pdk_json


_pdk_json.serialize = lambda obj: json.dumps(
    obj,
    sort_keys=True,
    default=_pdk_json.default_serialize,
    separators=(",", ":"),
)


st.set_page_config(page_title="Live Fire Activity", layout="wide")


RECENT_DATA_PATHS = [
    Path("data/active_fire/firms_viirs_snpp_nrt_recent.parquet"),
    Path("data/fires/viirs_west_recent.parquet"),
]

WEST_VIEW = pdk.ViewState(
    latitude=39.5,
    longitude=-112.0,
    zoom=4.0,
    pitch=0,
    bearing=0,
)

CONFIDENCE_LABELS = {
    "l": "Low",
    "n": "Nominal",
    "h": "High",
}

DAYNIGHT_LABELS = {
    "D": "Day",
    "N": "Night",
}


def find_recent_data_path() -> Path | None:
    """Find the first supported recent active-fire file path."""
    for path in RECENT_DATA_PATHS:
        if path.exists():
            return path
    return None


@st.cache_data
def load_recent(path_str: str, file_mtime: float) -> pd.DataFrame:
    """Load recent active-fire detections."""
    df = pd.read_parquet(path_str)

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    if "source" not in df.columns:
        df["source"] = "VIIRS_SNPP_NRT"

    if "satellite" not in df.columns:
        df["satellite"] = "—"

    if "daynight" not in df.columns:
        df["daynight"] = "—"

    if "confidence" not in df.columns:
        df["confidence"] = "n"

    if "acq_time" not in df.columns:
        df["acq_time"] = "—"

    df["confidence"] = df["confidence"].fillna("—").astype(str)
    df["daynight"] = df["daynight"].fillna("—").astype(str)
    df["source"] = df["source"].fillna("—").astype(str)
    df["satellite"] = df["satellite"].fillna("—").astype(str)

    return df.dropna(subset=["acq_date", "latitude", "longitude"])


def days_ago_color(days: int) -> list[int]:
    """Map age in days to a fire-like color ramp."""
    stops = [
        (0, [255, 255, 200]),
        (1, [255, 220, 120]),
        (3, [255, 140, 40]),
        (6, [200, 40, 20]),
        (10, [120, 20, 20]),
    ]

    for (lo_d, lo_c), (hi_d, hi_c) in zip(stops, stops[1:]):
        if days <= hi_d:
            t = (days - lo_d) / max(hi_d - lo_d, 1)
            return [int(lo_c[i] + t * (hi_c[i] - lo_c[i])) for i in range(3)]

    return stops[-1][1]


def format_confidence(code: str) -> str:
    return CONFIDENCE_LABELS.get(str(code), str(code))


def format_daynight(code: str) -> str:
    return DAYNIGHT_LABELS.get(str(code), str(code))


def normalize_date_range(date_range_value, default_start, default_end):
    """Handle Streamlit date_input output when users select one or two dates."""
    if isinstance(date_range_value, tuple) and len(date_range_value) == 2:
        return date_range_value

    if isinstance(date_range_value, tuple) and len(date_range_value) == 1:
        return date_range_value[0], default_end

    return default_start, default_end


def build_map(filtered: pd.DataFrame, view_mode: str, latest_detection: pd.Timestamp):
    """Build the PyDeck map object."""
    plot_data = filtered.copy()

    plot_data["days_old"] = (latest_detection - plot_data["acq_date"]).dt.days
    plot_data["color"] = plot_data["days_old"].apply(days_ago_color)
    plot_data["radius"] = np.sqrt(plot_data["frp"].clip(lower=1)) * 250
    plot_data["date_str"] = plot_data["acq_date"].dt.strftime("%Y-%m-%d")
    plot_data["conf_label"] = plot_data["confidence"].map(format_confidence)
    plot_data["daynight_label"] = plot_data["daynight"].map(format_daynight)
    plot_data["source_label"] = plot_data["source"].astype(str)
    plot_data["satellite_label"] = plot_data["satellite"].astype(str)

    records = plot_data.to_dict(orient="records")

    if view_mode == "Glowing points":
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=records,
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=2,
            radius_max_pixels=40,
            opacity=0.8,
            pickable=True,
            stroked=False,
        )

        tooltip = {
            "html": (
                "<b>{date_str}</b><br/>"
                "Fire radiative power: {frp} MW<br/>"
                "Confidence: {conf_label}<br/>"
                "Day/night: {daynight_label}<br/>"
                "Satellite: {satellite_label}<br/>"
                "Source: {source_label}"
            )
        }

    else:
        layer = pdk.Layer(
            "HeatmapLayer",
            data=records,
            get_position=["longitude", "latitude"],
            get_weight="frp",
            radius_pixels=40,
            intensity=1,
            threshold=0.05,
        )
        tooltip = None

    return pdk.Deck(
        layers=[layer],
        initial_view_state=WEST_VIEW,
        map_style="dark",
        tooltip=tooltip,
    )


def filtered_csv_download(filtered: pd.DataFrame) -> bytes:
    """Convert filtered detections to CSV bytes."""
    export_columns = [
        "acq_date",
        "acq_time",
        "latitude",
        "longitude",
        "frp",
        "confidence",
        "daynight",
        "satellite",
        "source",
    ]

    export_columns = [column for column in export_columns if column in filtered.columns]

    return filtered[export_columns].to_csv(index=False).encode("utf-8")


st.title("Live Fire Activity — U.S. West")

st.write(
    "This page shows recent NASA FIRMS VIIRS active-fire detections across the "
    "western United States. Each point represents a satellite-detected thermal "
    "anomaly pixel from a recent overpass."
)

st.warning(
    "Research and visualization use only. FIRMS detections are not official fire "
    "perimeters, burned-area estimates, evacuation guidance, or emergency warnings."
)

DATA_PATH = find_recent_data_path()

if DATA_PATH is None:
    st.error(
        "No recent fire data found. Run `python fetch_firms.py --recent` to create "
        "`data/active_fire/firms_viirs_snpp_nrt_recent.parquet`."
    )
    st.stop()

data = load_recent(str(DATA_PATH), DATA_PATH.stat().st_mtime)

if data.empty:
    st.error(
        "The recent FIRMS file was found, but it does not contain valid detections. "
        "Fetch the recent snapshot again with `python fetch_firms.py --recent`."
    )
    st.stop()

latest = data["acq_date"].max()
earliest = data["acq_date"].min()

days_since = (date.today() - latest.date()).days

if days_since <= 1:
    freshness = "Updated within the last day."
elif days_since <= 7:
    freshness = f"Most recent detection {days_since} days ago."
else:
    freshness = (
        f"Most recent detection {days_since} days ago — "
        "this snapshot may be out of date."
    )

st.info(
    f"**Data snapshot:** {earliest.date()} to {latest.date()}  \n"
    f"**Status:** {freshness}"
)

st.caption(
    f"Loaded file: `{DATA_PATH}`. The date range is based on detection dates in the "
    "current local FIRMS snapshot."
)

st.sidebar.header("Filters")

view_mode = st.sidebar.radio(
    "Map style",
    ["Glowing points", "Heatmap"],
)

default_start = earliest.date()
default_end = latest.date()

date_range_value = st.sidebar.date_input(
    "Date range",
    value=(default_start, default_end),
    min_value=default_start,
    max_value=default_end,
)

start_date, end_date = normalize_date_range(
    date_range_value,
    default_start=default_start,
    default_end=default_end,
)

min_frp = st.sidebar.slider(
    "Minimum fire radiative power (FRP, MW)",
    0.0,
    float(max(data["frp"].max(), 1.0)),
    0.0,
    step=1.0,
)

available_confidence = [
    code
    for code in ["h", "n", "l"]
    if code in set(data["confidence"].dropna().astype(str))
]

other_confidence = [
    code
    for code in sorted(data["confidence"].dropna().astype(str).unique().tolist())
    if code not in available_confidence
]

available_confidence = available_confidence + other_confidence

confidence_filter = st.sidebar.multiselect(
    "Confidence",
    options=available_confidence,
    default=available_confidence,
    format_func=format_confidence,
)

available_daynight = sorted(data["daynight"].dropna().astype(str).unique().tolist())

daynight_filter = st.sidebar.multiselect(
    "Day/night",
    options=available_daynight,
    default=available_daynight,
    format_func=format_daynight,
)

available_sources = sorted(data["source"].dropna().astype(str).unique().tolist())

source_filter = st.sidebar.multiselect(
    "FIRMS source",
    options=available_sources,
    default=available_sources,
)

st.sidebar.caption(
    "State filtering will be added after county or state boundary data is joined."
)

filtered = data[
    (data["acq_date"].dt.date >= start_date)
    & (data["acq_date"].dt.date <= end_date)
    & (data["frp"] >= min_frp)
    & (data["confidence"].astype(str).isin([str(code) for code in confidence_filter]))
    & (data["daynight"].astype(str).isin([str(code) for code in daynight_filter]))
    & (data["source"].astype(str).isin([str(code) for code in source_filter]))
].copy()

st.caption(
    f"Showing {len(filtered):,} of {len(data):,} detections "
    f"({start_date} to {end_date})."
)

if filtered.empty:
    st.info("No detections match the current filters.")
    st.stop()

high_count = int((filtered["confidence"] == "h").sum())
night_count = int((filtered["daynight"] == "N").sum())

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

metric_col1.metric("Displayed detections", f"{len(filtered):,}")
metric_col2.metric("Maximum observed FRP", f"{filtered['frp'].max():.0f} MW")
metric_col3.metric("Median observed FRP", f"{filtered['frp'].median():.1f} MW")
metric_col4.metric("High-confidence detections", f"{high_count:,}")
metric_col5.metric("Nighttime detections", f"{night_count:,}")

with st.expander("How to read this map", expanded=True):
    st.markdown(
        """
        **Point color** shows recency within the current snapshot. Brighter yellow-white
        points are newer detections, while darker red points are older detections.

        **Point size** is based on fire radiative power, or FRP. FRP measures radiant
        energy from active burning within a satellite pixel. It should not be read as
        total fire size or burned area.

        **Confidence** comes from FIRMS. High-confidence detections are generally more
        reliable. Low-confidence detections should be read with more caution.

        **Day/night** matters because satellite viewing conditions, background
        temperature, and fire behavior can differ between daytime and nighttime
        overpasses.
        """
    )

deck = build_map(filtered, view_mode=view_mode, latest_detection=latest)
st.pydeck_chart(deck, height=640)

st.caption(
    "Source: NASA FIRMS VIIRS active-fire detections. A detection marks where a "
    "satellite overpass sensed active burning or a thermal anomaly. Fire radiative "
    "power is measured in megawatts. Point color shows days since detection."
)

st.divider()

st.header("Detection summaries")

plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.subheader("FRP distribution")
    fig_frp = px.histogram(
        filtered,
        x="frp",
        nbins=40,
        labels={"frp": "Fire radiative power (MW)", "count": "Detections"},
    )
    fig_frp.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Detections",
    )
    st.plotly_chart(fig_frp, use_container_width=True)

with plot_col2:
    st.subheader("Daily detection count")
    daily_counts = (
        filtered.assign(date=filtered["acq_date"].dt.date)
        .groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "detections"})
    )

    fig_daily = px.bar(
        daily_counts,
        x="date",
        y="detections",
        labels={"date": "Date", "detections": "Detections"},
    )
    fig_daily.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_daily, use_container_width=True)

st.subheader("Top high-FRP detections")

table = filtered.copy()
table["date"] = table["acq_date"].dt.strftime("%Y-%m-%d")
table["confidence_label"] = table["confidence"].map(format_confidence)
table["daynight_label"] = table["daynight"].map(format_daynight)

table_columns = [
    "date",
    "acq_time",
    "latitude",
    "longitude",
    "frp",
    "confidence_label",
    "daynight_label",
    "satellite",
    "source",
]

table_columns = [column for column in table_columns if column in table.columns]

top_table = (
    table.sort_values("frp", ascending=False)
    .loc[:, table_columns]
    .head(25)
)

st.dataframe(
    top_table,
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    label="Download filtered detections as CSV",
    data=filtered_csv_download(filtered),
    file_name="filtered_live_fire_detections.csv",
    mime="text/csv",
)

st.caption(
    "The table is sorted by FRP. High FRP can indicate intense active burning in a "
    "satellite pixel, but it should not be read as total fire size."
)