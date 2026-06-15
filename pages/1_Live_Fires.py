import json
from datetime import date, timedelta
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


st.set_page_config(page_title="Live Fires", layout="wide")


DATA_PATH = Path("data/fires/viirs_west_recent.parquet")

CONTINENTAL_VIEW = pdk.ViewState(
    latitude=39.5, longitude=-98.0, zoom=3.3, pitch=0, bearing=0
)

CONFIDENCE_LABELS = {"l": "Low", "n": "Nominal", "h": "High"}


@st.cache_data
def load_recent(file_mtime):
    df = pd.read_parquet(DATA_PATH)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    return df


def days_ago_color(days):
    # Fresh = bright yellow-white, older = deep red.
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


st.title("Live Fires — US West")

st.write(
    "VIIRS satellite fire detections from the last several days across the western "
    "United States. Each point is a 375 m pixel where the satellite detected active "
    "burning. Brighter points are more recent; larger points burned more intensely."
)

if not DATA_PATH.exists():
    st.error(
        "No recent fire data found. Run fetch_firms.py --recent to create "
        "data/fires/viirs_west_recent.parquet."
    )
    st.stop()

data = load_recent(DATA_PATH.stat().st_mtime)

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
    f"Showing the latest committed snapshot, {earliest.date()} to "
    f"{latest.date()}. {freshness}"
)


st.sidebar.header("Display")

view_mode = st.sidebar.radio(
    "Style", ["Glowing points", "Heatmap"]
)

min_frp = st.sidebar.slider(
    "Minimum fire power (FRP, MW)",
    0.0,
    float(max(data["frp"].max(), 1.0)),
    0.0,
    step=1.0,
)

confidence_filter = st.sidebar.multiselect(
    "Confidence",
    options=["h", "n", "l"],
    default=["h", "n", "l"],
    format_func=lambda code: CONFIDENCE_LABELS.get(code, code),
)


filtered = data[
    (data["frp"] >= min_frp) & (data["confidence"].isin(confidence_filter))
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
filtered["conf_label"] = filtered["confidence"].map(CONFIDENCE_LABELS).fillna("—")

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
        "html": "<b>{date_str}</b><br/>Fire power: {frp} MW<br/>"
                "Confidence: {conf_label}"
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
    "Source: NASA FIRMS, VIIRS 375 m active fire product. A detection marks where a "
    "satellite overpass sensed active burning; it is not an official fire perimeter. "
    "Fire power (FRP) is measured in megawatts. Point color shows days since detection."
)
