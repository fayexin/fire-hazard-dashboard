import json
from datetime import datetime, timedelta
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


st.set_page_config(page_title="Historical Fire Events", layout="wide")


DATA_DIR = Path("data/active_fire")
DEFAULT_SOURCE = "VIIRS_SNPP_SP"

CONFIDENCE_LABELS = {
    "l": "Low",
    "n": "Nominal",
    "h": "High",
}

DAYNIGHT_LABELS = {
    "D": "Day",
    "N": "Night",
}


EVENTS = [
    {
        "name": "Camp Fire (2018)",
        "year": 2018,
        "center_date": "2018-11-08",
        "latitude": 39.78,
        "longitude": -121.55,
        "zoom": 8.7,
        "radius_degrees_lat": 1.1,
        "radius_degrees_lon": 1.4,
        "description": "Northern California event window centered near Paradise, California.",
    },
    {
        "name": "Carr Fire (2018)",
        "year": 2018,
        "center_date": "2018-07-27",
        "latitude": 40.65,
        "longitude": -122.60,
        "zoom": 8.5,
        "radius_degrees_lat": 1.2,
        "radius_degrees_lon": 1.5,
        "description": "Northern California event window centered near Redding, California.",
    },
    {
        "name": "August Complex (2020)",
        "year": 2020,
        "center_date": "2020-08-19",
        "latitude": 39.80,
        "longitude": -122.80,
        "zoom": 7.8,
        "radius_degrees_lat": 1.8,
        "radius_degrees_lon": 2.2,
        "description": "Northern California event window centered around the August Complex region.",
    },
    {
        "name": "Dixie Fire (2021)",
        "year": 2021,
        "center_date": "2021-07-24",
        "latitude": 40.00,
        "longitude": -121.20,
        "zoom": 7.8,
        "radius_degrees_lat": 1.8,
        "radius_degrees_lon": 2.2,
        "description": "Northern Sierra event window centered in northeastern California.",
    },
    {
        "name": "Bootleg Fire (2021)",
        "year": 2021,
        "center_date": "2021-07-14",
        "latitude": 42.70,
        "longitude": -121.30,
        "zoom": 7.8,
        "radius_degrees_lat": 1.8,
        "radius_degrees_lon": 2.2,
        "description": "Southern Oregon event window centered around the Bootleg Fire region.",
    },
]


def source_to_slug(source: str) -> str:
    return source.lower()


def historical_path(year: int, source: str = DEFAULT_SOURCE) -> Path:
    return DATA_DIR / f"firms_{source_to_slug(source)}_{year}.parquet"


@st.cache_data
def load_year(path_str: str, file_mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(path_str)

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    if "confidence" not in df.columns:
        df["confidence"] = "n"

    if "daynight" not in df.columns:
        df["daynight"] = "—"

    if "satellite" not in df.columns:
        df["satellite"] = "—"

    if "source" not in df.columns:
        df["source"] = DEFAULT_SOURCE

    if "acq_time" not in df.columns:
        df["acq_time"] = "—"

    df["confidence"] = df["confidence"].fillna("—").astype(str)
    df["daynight"] = df["daynight"].fillna("—").astype(str)
    df["satellite"] = df["satellite"].fillna("—").astype(str)
    df["source"] = df["source"].fillna("—").astype(str)

    return df.dropna(subset=["acq_date", "latitude", "longitude"])


def format_confidence(code: str) -> str:
    return CONFIDENCE_LABELS.get(str(code), str(code))


def format_daynight(code: str) -> str:
    return DAYNIGHT_LABELS.get(str(code), str(code))


def frp_color(value: float, low: float, high: float) -> list[int]:
    if high <= low:
        t = 1.0
    else:
        t = float(np.clip((value - low) / (high - low), 0, 1))

    stops = [
        (0.0, [120, 20, 20]),
        (0.35, [220, 70, 25]),
        (0.70, [255, 170, 60]),
        (1.0, [255, 255, 210]),
    ]

    for (lo_t, lo_c), (hi_t, hi_c) in zip(stops, stops[1:]):
        if t <= hi_t:
            f = (t - lo_t) / max(hi_t - lo_t, 1e-9)
            return [int(lo_c[i] + f * (hi_c[i] - lo_c[i])) for i in range(3)]

    return stops[-1][1]


def build_event_map(event_data: pd.DataFrame, event: dict, map_style: str):
    plot_data = event_data.copy()

    base_view = pdk.ViewState(
        latitude=event["latitude"],
        longitude=event["longitude"],
        zoom=event["zoom"],
        min_zoom=5,
        max_zoom=12,
    )

    if plot_data.empty:
        return pdk.Deck(
            layers=[],
            initial_view_state=base_view,
            map_style="dark",
        )

    low = float(np.nanpercentile(plot_data["frp"], 5))
    high = max(float(np.nanpercentile(plot_data["frp"], 95)), low + 1e-6)

    plot_data["date_str"] = plot_data["acq_date"].dt.strftime("%Y-%m-%d")
    plot_data["confidence_label"] = plot_data["confidence"].map(format_confidence)
    plot_data["daynight_label"] = plot_data["daynight"].map(format_daynight)
    plot_data["color"] = plot_data["frp"].apply(lambda x: frp_color(x, low, high))

    # Fixed radius keeps historical event points readable.
    # FRP is shown by color, not by huge circle size.
    plot_data["radius"] = 375

    if map_style == "FRP points":
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=plot_data.to_dict("records"),
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=2,
            radius_max_pixels=7,
            opacity=0.65,
            pickable=True,
            stroked=True,
            get_line_color=[255, 255, 255, 80],
            line_width_min_pixels=0.3,
        )

        tooltip = {
            "html": (
                "<b>{date_str}</b><br/>"
                "Fire radiative power: {frp} MW<br/>"
                "Confidence: {confidence_label}<br/>"
                "Day/night: {daynight_label}<br/>"
                "Satellite: {satellite}<br/>"
                "Source: {source}"
            )
        }

    else:
        layer = pdk.Layer(
            "HeatmapLayer",
            data=plot_data.to_dict("records"),
            get_position=["longitude", "latitude"],
            get_weight="frp",
            radius_pixels=28,
            intensity=1,
            threshold=0.05,
        )

        tooltip = None

    return pdk.Deck(
        layers=[layer],
        initial_view_state=base_view,
        map_style="dark",
        tooltip=tooltip,
    )


st.title("Historical Fire Events — U.S. West")

st.write(
    "This page explores selected historical wildfire event windows using NASA FIRMS "
    "VIIRS active-fire detections. It shows where satellite thermal anomaly pixels "
    "were detected during a short event window."
)

st.warning(
    "FIRMS detections are not official fire perimeters, burned-area estimates, or "
    "complete event boundaries. The map shows satellite-detected active-fire pixels "
    "within a simple event-centered spatial and temporal window."
)

st.sidebar.header("Event controls")

event_names = [event["name"] for event in EVENTS]
selected_name = st.sidebar.selectbox("Historical event", event_names)

event = next(item for item in EVENTS if item["name"] == selected_name)

window_days_before = st.sidebar.slider(
    "Days before center date",
    min_value=0,
    max_value=14,
    value=2,
)

window_days_after = st.sidebar.slider(
    "Days after center date",
    min_value=1,
    max_value=21,
    value=7,
)

map_style = st.sidebar.radio(
    "Map style",
    ["FRP heatmap", "FRP points"],
)

path = historical_path(event["year"])

if not path.exists():
    st.error(
        f"Historical FIRMS file not found for {event['year']}: `{path}`. "
        f"Run `python fetch_firms.py --year {event['year']} --source {DEFAULT_SOURCE}` "
        "from the repository root."
    )
    st.stop()

data = load_year(str(path), path.stat().st_mtime)

center_date = datetime.strptime(event["center_date"], "%Y-%m-%d").date()
start_date = center_date - timedelta(days=window_days_before)
end_date = center_date + timedelta(days=window_days_after)

event_data = data[
    (data["acq_date"].dt.date >= start_date)
    & (data["acq_date"].dt.date <= end_date)
    & data["latitude"].between(
        event["latitude"] - event["radius_degrees_lat"],
        event["latitude"] + event["radius_degrees_lat"],
    )
    & data["longitude"].between(
        event["longitude"] - event["radius_degrees_lon"],
        event["longitude"] + event["radius_degrees_lon"],
    )
].copy()

st.subheader(event["name"])

st.write(event["description"])

st.info(
    f"**Event window:** {start_date} to {end_date}  \n"
    f"**Historical file:** `{path}`"
)

if event_data.empty:
    st.info("No FIRMS detections were found in this event window.")
    st.stop()

high_count = int((event_data["confidence"] == "h").sum())
night_count = int((event_data["daynight"] == "N").sum())

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric("Detections in window", f"{len(event_data):,}")
metric_col2.metric("Maximum observed FRP", f"{event_data['frp'].max():.0f} MW")
metric_col3.metric("High-confidence detections", f"{high_count:,}")
metric_col4.metric("Nighttime detections", f"{night_count:,}")

with st.expander("How to read this event map", expanded=True):
    st.markdown(
        """
        **FRP heatmap** shows a smoothed concentration of active-fire detections,
        weighted by fire radiative power. It is useful for seeing the main detection
        cluster, but it is not a fire perimeter.

        **FRP points** show individual FIRMS detections. Point color shows observed
        fire radiative power. Point size is fixed so the map does not become visually
        dominated by a few high-FRP detections.

        **Event window** is a simple date range and geographic box around the selected
        event. It may include nearby detections that are not part of the named fire.
        """
    )

deck = build_event_map(event_data, event=event, map_style=map_style)
st.pydeck_chart(deck, height=640)

st.caption(
    "Source: NASA FIRMS VIIRS standard-processed active-fire detections. Detections "
    "are satellite thermal anomaly pixels and should not be interpreted as official "
    "fire boundaries."
)

st.divider()

st.header("Event detection summaries")

plot_col1, plot_col2 = st.columns(2)

daily = (
    event_data.assign(date=event_data["acq_date"].dt.date)
    .groupby("date", as_index=False)
    .agg(
        detections=("frp", "size"),
        max_frp=("frp", "max"),
        median_frp=("frp", "median"),
        sum_observed_frp=("frp", "sum"),
    )
)

with plot_col1:
    st.subheader("Daily detection count")
    fig_count = px.bar(
        daily,
        x="date",
        y="detections",
        labels={"date": "Date", "detections": "Detections"},
    )
    fig_count.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_count, use_container_width=True)

with plot_col2:
    st.subheader("Daily maximum observed FRP")
    fig_frp = px.line(
        daily,
        x="date",
        y="max_frp",
        markers=True,
        labels={"date": "Date", "max_frp": "Maximum observed FRP (MW)"},
    )
    fig_frp.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_frp, use_container_width=True)

st.subheader("Top high-FRP detections in this event window")

table = event_data.copy()
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

st.caption(
    "The table is sorted by FRP. High FRP indicates intense observed active burning "
    "within a satellite pixel, not total fire size."
)