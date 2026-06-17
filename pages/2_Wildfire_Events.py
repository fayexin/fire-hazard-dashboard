import json
from datetime import datetime, timedelta
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


st.set_page_config(page_title="Wildfire Events", layout="wide")


DATA_DIR = Path("data/fires")

CONFIDENCE_LABELS = {"l": "Low", "n": "Nominal", "h": "High"}

FIRE_STOPS = [
    (0.0, [40, 0, 0]),
    (0.25, [150, 20, 0]),
    (0.5, [240, 90, 0]),
    (0.75, [255, 180, 40]),
    (1.0, [255, 250, 220]),
]

# Sequential (viridis-like) ramp for the burn-time progression view.
PROG_STOPS = [
    (0.0, [68, 1, 84]),
    (0.25, [59, 82, 139]),
    (0.5, [33, 145, 140]),
    (0.75, [94, 201, 98]),
    (1.0, [253, 231, 37]),
]

HEATMAP_COLORS = [
    [40, 0, 0],
    [150, 20, 0],
    [240, 90, 0],
    [255, 180, 40],
    [255, 250, 220],
]


# name, peak date, lat, lon, zoom, blurb
EVENTS = [
    ("Dixie Fire (2021)", "2021-07-24", 40.0, -121.2, 8.2,
     "The largest single wildfire in California's recorded history, burning "
     "nearly one million acres across the northern Sierra."),
    ("Camp Fire (2018)", "2018-11-08", 39.78, -121.55, 9.2,
     "Paradise, CA — the deadliest and most destructive wildfire in "
     "California history."),
    ("August Complex (2020)", "2020-08-19", 39.8, -122.8, 8.2,
     "California's first 'gigafire', exceeding one million acres in the "
     "2020 record season."),
    ("Caldor Fire (2021)", "2021-08-29", 38.6, -120.3, 8.5,
     "Burned from the Sierra foothills across the crest, threatening the "
     "Lake Tahoe basin."),
    ("Creek Fire (2020)", "2020-09-06", 37.2, -119.3, 8.7,
     "An explosive Sierra National Forest fire that generated a "
     "fire-driven thunderstorm."),
    ("Carr Fire (2018)", "2018-07-27", 40.65, -122.6, 9.0,
     "Produced a destructive fire whirl ('firenado') near Redding."),
    ("Thomas Fire (2017)", "2017-12-12", 34.5, -119.2, 8.7,
     "A massive December fire across Ventura and Santa Barbara counties."),
    ("Bootleg Fire (2021)", "2021-07-14", 42.7, -121.3, 8.5,
     "The largest Oregon wildfire of 2021, large enough to influence local "
     "weather."),
]


@st.cache_data
def available_years():
    years = []
    for file in sorted(DATA_DIR.glob("viirs_west_[0-9]*.parquet")):
        stem = file.stem.split("_")[-1]
        if stem.isdigit():
            years.append(int(stem))
    return years


@st.cache_data
def load_year(year):
    df = pd.read_parquet(DATA_DIR / f"viirs_west_{year}.parquet")
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    return df


@st.cache_data
def build_ramp(stops):
    lut = []
    for i in range(256):
        t = i / 255
        placed = False
        for (lo, lc), (hi, hc) in zip(stops, stops[1:]):
            if t <= hi:
                f = (t - lo) / max(hi - lo, 1e-9)
                lut.append([int(lc[j] + f * (hc[j] - lc[j])) for j in range(3)])
                placed = True
                break
        if not placed:
            lut.append(stops[-1][1])
    return lut


def fire_lut():
    return build_ramp(FIRE_STOPS)


def legend_html(lut, lo, hi):
    stops = ", ".join(
        f"rgb({c[0]},{c[1]},{c[2]})" for c in lut[:: max(len(lut) // 20, 1)]
    )
    return f"""
    <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
      <span style="font-size:12px; color:#888;">{lo:.0f}</span>
      <div style="width:160px; height:12px; border-radius:3px;
                  background:linear-gradient(to right, {stops});"></div>
      <span style="font-size:12px; color:#888;">{hi:.0f}+</span>
      <span style="font-size:12px; color:#666; margin-left:6px;">
        Fire intensity (FRP, MW)</span>
    </div>
    """


def time_legend_html(lut, first_label, last_label):
    stops = ", ".join(
        f"rgb({c[0]},{c[1]},{c[2]})" for c in lut[:: max(len(lut) // 20, 1)]
    )
    return f"""
    <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
      <span style="font-size:12px; color:#888;">{first_label}</span>
      <div style="width:160px; height:12px; border-radius:3px;
                  background:linear-gradient(to right, {stops});"></div>
      <span style="font-size:12px; color:#888;">{last_label}</span>
      <span style="font-size:12px; color:#666; margin-left:6px;">
        Day burned (earlier → later)</span>
    </div>
    """


st.title("Wildfire Events")

st.write(
    "Major wildfires of the past decade, each shown at its own location. Detections "
    "are colored by fire intensity — deep red is cooler, white-hot is the most "
    "intense burning. Use the day slider beneath the map to step through the fire "
    "one day at a time and watch it spread."
)


years = available_years()

if not years:
    st.error("No fire data found. Run fetch_firms.py --all to create the year files.")
    st.stop()


st.sidebar.header("Controls")

choice = st.sidebar.selectbox("Fire", [event[0] for event in EVENTS])
event = next(e for e in EVENTS if e[0] == choice)
_, date_str, lat, lon, zoom, blurb = event
peak_date = datetime.strptime(date_str, "%Y-%m-%d").date()
year = peak_date.year

style = st.sidebar.radio(
    "Map style",
    ["Burn time (progression)", "Burn area (heatmap)", "Intensity points"],
)

event_length = st.sidebar.slider("Event length (days)", 3, 21, 7)


st.subheader(f"{choice}")
st.markdown(f"*{blurb}*")

if year not in years:
    st.warning(f"Detection data for {year} is not available.")
    st.stop()


data = load_year(year)

# Detections near this event's location, so distant fires don't intrude.
near = data[
    data["latitude"].between(lat - 2.0, lat + 2.0)
    & data["longitude"].between(lon - 2.5, lon + 2.5)
].copy()

span_start = peak_date - timedelta(days=2)
day_options = [span_start + timedelta(days=i) for i in range(event_length)]

# Color scale fixed across the whole span so days are comparable.
span = near[
    (near["acq_date"].dt.date >= day_options[0])
    & (near["acq_date"].dt.date <= day_options[-1])
]
lut = fire_lut()
if len(span):
    lo = float(np.nanpercentile(span["frp"], 5))
    hi = max(float(np.nanpercentile(span["frp"], 95)), lo + 1e-6)
else:
    lo, hi = 0.0, 1.0


# Reserve the layout: metrics on top, map in the middle, slider beneath the map.
metrics_area = st.container()
map_area = st.container()

default_day = peak_date if peak_date in day_options else day_options[len(day_options) // 2]
selected_day = st.select_slider(
    "Burned through",
    options=day_options,
    value=default_day,
    format_func=lambda d: d.strftime("%b %d, %Y"),
)

# Cumulative: every detection from the start of the event through the chosen day.
cumulative = near[
    (near["acq_date"].dt.date >= day_options[0])
    & (near["acq_date"].dt.date <= selected_day)
].copy()
cumulative = cumulative.sort_values("acq_date")  # latest drawn on top

layer = None

if len(cumulative):
    cumulative["date_str"] = cumulative["acq_date"].dt.strftime("%Y-%m-%d")
    if "confidence" in cumulative.columns:
        cumulative["conf_label"] = (
            cumulative["confidence"].map(CONFIDENCE_LABELS).fillna("—")
        )
    else:
        cumulative["conf_label"] = "—"

    if style == "Burn area (heatmap)":
        layer = pdk.Layer(
            "HeatmapLayer",
            data=cumulative[["latitude", "longitude", "frp"]].to_dict("records"),
            get_position=["longitude", "latitude"],
            get_weight="frp",
            radius_pixels=35,
            intensity=1,
            threshold=0.05,
            color_range=HEATMAP_COLORS,
        )

    elif style == "Burn time (progression)":
        prog = build_ramp(PROG_STOPS)
        day_index = (
            (cumulative["acq_date"].dt.normalize()
             - pd.Timestamp(day_options[0])).dt.days.to_numpy()
        )
        t = np.clip(day_index / max(event_length - 1, 1), 0, 1)
        cumulative["color"] = [prog[int(v * 255)] for v in t]
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=cumulative.to_dict("records"),
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius=375,
            radius_min_pixels=2,
            radius_max_pixels=6,
            opacity=0.9,
            pickable=True,
            stroked=False,
        )

    else:  # Intensity points
        idx = (np.clip((cumulative["frp"].to_numpy() - lo) / (hi - lo), 0, 1) * 255).astype(int)
        base_colors = [lut[i] for i in idx]
        age = (pd.Timestamp(selected_day) - cumulative["acq_date"]).dt.days.to_numpy()
        alpha = np.clip(255 - age * 45, 80, 255).astype(int)
        cumulative["color"] = [c + [int(a)] for c, a in zip(base_colors, alpha)]
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=cumulative.to_dict("records"),
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius=375,
            radius_min_pixels=2,
            radius_max_pixels=6,
            opacity=0.9,
            pickable=True,
            stroked=False,
        )

deck = pdk.Deck(
    layers=[layer] if layer else [],
    initial_view_state=pdk.ViewState(
        latitude=lat, longitude=lon, zoom=zoom, min_zoom=5, max_zoom=12
    ),
    map_style="dark",
    tooltip={
        "html": "<b>{date_str}</b><br/>Intensity: {frp} MW<br/>"
                "Confidence: {conf_label}"
    },
)

with metrics_area:
    col1, col2, col3 = st.columns(3)
    col1.metric("Detections so far", f"{len(cumulative):,}")
    col2.metric(
        "Peak intensity",
        f"{cumulative['frp'].max():.0f} MW" if len(cumulative) else "—",
    )
    col3.metric("Burned through", selected_day.strftime("%b %d, %Y"))

with map_area:
    st.pydeck_chart(deck, height=600)
    if not len(cumulative):
        st.caption("No detections yet by this day — slide right.")

if style == "Burn time (progression)":
    st.markdown(
        time_legend_html(
            build_ramp(PROG_STOPS),
            day_options[0].strftime("%b %d"),
            day_options[-1].strftime("%b %d"),
        ),
        unsafe_allow_html=True,
    )
else:
    st.markdown(legend_html(lut, lo, hi), unsafe_allow_html=True)

st.caption(
    "Source: NASA FIRMS VIIRS 375 m active fire detections, accumulated through the "
    "selected day. Burn time colors each detection by when it burned; burn area shows "
    "intensity-weighted density; intensity points color by fire radiative power. "
    "Detections are not official fire perimeters."
)