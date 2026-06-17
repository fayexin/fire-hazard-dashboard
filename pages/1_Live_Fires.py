import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
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

CONTINENTAL_VIEW = pdk.ViewState(
    latitude=39.5,
    longitude=-112.0,
    zoom=4.0,
    pitch=0,
    bearing=0,
)

CONFIDENCE_LABELS = {"l": "Low", "n": "Nominal", "h": "High"}


def find_recent_data_path() -> Path | None:
    """Find the newest supported recent active-fire file path."""
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

    if "source" not in df.columns:
        df["source"] = "VIIRS_SNPP_NRT"

    if "satellite" not in df.columns:
        df["satellite"] = "—"

    if "daynight" not in df.columns:
        df["daynight"] = "—"

    if "confidence" not in df.columns:
        df["confidence"] = "n"

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


st.title("Live Fire Activity — U.S. West")

st.write(
    "This page maps recent VIIRS satellite active-fire detections across the western "
    "United States. Each point is a satellite-detected thermal anomaly pixel. "
    "Brighter points are more recent; larger points have higher fire radiative power."
)

st.warning(
    "FIRMS detections are not official fire perimeters or burned-area estimates. "
    "Point size is based on fire radiative power, not fire size."
)

DATA_PATH = find_recent_data_path()

if DATA_PATH is None:
    st.error(
        "No recent fire data found. Run `python fetch_firms.py --recent` to create "
        "`data/active_fire/firms_viirs_snpp_nrt_recent.parquet`."
    )
    st.stop()

data = load_recent(str(DATA_PATH), DATA_PATH.stat().st_mtime)

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
    f"Showing the latest available snapshot, {earliest.date()} to "
    f"{latest.date()}. {freshness}"
)

st.caption(f"Loaded data file: `{DATA_PATH}`")

st.sidebar.header("Display")

view_mode = st.sidebar.radio(
    "Style",
    ["Glowing points", "Heatmap"],
)

min_frp = st.sidebar.slider(
    "Minimum fire power (FRP, MW)",
    0.0,
    float(max(data["frp"].max(), 1.0)),
    0.0,
    step=1.0,
)

available_confidence = [
    code for code in ["h", "n", "l"] if code in set(data["confidence"].dropna().astype(str))
]

if not available_confidence:
    available_confidence = sorted(data["confidence"].dropna().astype(str).unique().tolist())

confidence_filter = st.sidebar.multiselect(
    "Confidence",
    options=available_confidence,
    default=available_confidence,
    format_func=lambda code: CONFIDENCE_LABELS.get(code, str(code)),
)

filtered = data[
    (data["frp"] >= min_frp)
    & (data["confidence"].astype(str).isin([str(code) for code in confidence_filter]))
].copy()

st.caption(
    f"Showing {len(filtered):,} of {len(data):,} detections "
    f"({earliest.date()} to {latest.date()})."
)

col1, col2, col3 = st.columns(3)

col1.metric("Detections", f"{len(filtered):,}")
col2.metric("Max fire power", f"{filtered['frp'].max():.0f} MW" if len(filtered) else "—")
col3.metric("Most recent", f"{latest.date()}")

if filtered.empty:
    st.info("No detections match the current filters.")
    st.stop()

filtered["days_old"] = (latest - filtered["acq_date"]).dt.days
filtered["color"] = filtered["days_old"].apply(days_ago_color)
filtered["radius"] = np.sqrt(filtered["frp"].clip(lower=1)) * 250
filtered["date_str"] = filtered["acq_date"].dt.strftime("%Y-%m-%d")
filtered["conf_label"] = filtered["confidence"].map(CONFIDENCE_LABELS).fillna(
    filtered["confidence"].astype(str)
)
filtered["source_label"] = filtered["source"].fillna("—").astype(str)
filtered["satellite_label"] = filtered["satellite"].fillna("—").astype(str)
filtered["daynight_label"] = filtered["daynight"].fillna("—").astype(str)

records = filtered.to_dict(orient="records")

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
            "Fire power: {frp} MW<br/>"
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

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=CONTINENTAL_VIEW,
    map_style="dark",
    tooltip=tooltip,
)

st.pydeck_chart(deck, height=640)

st.caption(
    "Source: NASA FIRMS VIIRS active-fire detections. A detection marks where a "
    "satellite overpass sensed active burning or a thermal anomaly. Fire radiative "
    "power is measured in megawatts. Point color shows days since detection."
)