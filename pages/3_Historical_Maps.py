import json
from calendar import month_name
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


st.set_page_config(page_title="Historical Fire Maps", layout="wide")


DATA_DIR = Path("data/fires")

# Whole US West (the downloaded bounding box: -125,31,-102,49).
WEST_VIEW = pdk.ViewState(
    latitude=41.0, longitude=-114.0, zoom=4.2, min_zoom=3, max_zoom=10
)

MAX_RENDER = 80000  # cap points sent to the browser; sample if exceeded

FIRE_STOPS = [
    (0.0, [40, 0, 0]),
    (0.25, [150, 20, 0]),
    (0.5, [240, 90, 0]),
    (0.75, [255, 180, 40]),
    (1.0, [255, 250, 220]),
]

HEATMAP_COLORS = [
    [40, 0, 0],
    [150, 20, 0],
    [240, 90, 0],
    [255, 180, 40],
    [255, 250, 220],
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
    df["month"] = df["acq_date"].dt.month
    return df


@st.cache_data
def fire_lut():
    lut = []
    for i in range(256):
        t = i / 255
        placed = False
        for (lo, lc), (hi, hc) in zip(FIRE_STOPS, FIRE_STOPS[1:]):
            if t <= hi:
                f = (t - lo) / max(hi - lo, 1e-9)
                lut.append([int(lc[j] + f * (hc[j] - lc[j])) for j in range(3)])
                placed = True
                break
        if not placed:
            lut.append(FIRE_STOPS[-1][1])
    return lut


def legend_html(lut, lo, hi):
    stops = ", ".join(
        f"rgb({c[0]},{c[1]},{c[2]})" for c in lut[:: max(len(lut) // 20, 1)]
    )
    return f"""
    <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
      <span style="font-size:12px; color:#888;">{lo:.0f}</span>
      <div style="width:160px; height:12px; border-radius:3px;
                  background:linear-gradient(to right, {stops});"></div>
      <span style="font-size:12px; color:#888;">{hi:.0f}+</span>
      <span style="font-size:12px; color:#666; margin-left:6px;">
        Fire intensity (FRP, MW)</span>
    </div>
    """


st.title("Historical Fire Maps — US West")

st.write(
    "Every wildfire detection across the western United States, for any month since "
    "2012. Step through the years and months to see where and how intensely fires "
    "burned. Choose a heatmap to read the overall footprint, or thermal points to "
    "see individual fires colored by intensity."
)


years = available_years()

if not years:
    st.error("No fire data found. Run fetch_firms.py --all to create the year files.")
    st.stop()


st.sidebar.header("Controls")

style = st.sidebar.radio("Style", ["Heatmap", "Thermal points"])

selected_year = st.sidebar.select_slider("Year", options=years, value=years[-1])

selected_month = st.sidebar.select_slider(
    "Month",
    options=list(range(1, 13)),
    value=8,
    format_func=lambda m: month_name[m][:3],
)


data = load_year(selected_year)
subset = data[data["month"] == selected_month].copy()


st.subheader(f"{month_name[selected_month]} {selected_year}")

col1, col2, col3 = st.columns(3)
col1.metric("Detections", f"{len(subset):,}")
col2.metric("Peak intensity", f"{subset['frp'].max():.0f} MW" if len(subset) else "—")
col3.metric(
    "Total fire power",
    f"{subset['frp'].sum() / 1000:.0f} GW" if len(subset) else "—",
)

if subset.empty:
    st.info("No detections in this month.")
    st.stop()


# Cap points to keep the browser payload reasonable.
sampled = subset
note = ""
if len(subset) > MAX_RENDER:
    sampled = subset.sample(MAX_RENDER, random_state=0)
    note = f" (showing a {MAX_RENDER:,}-point sample of {len(subset):,})"


lut = fire_lut()
lo = float(np.nanpercentile(subset["frp"], 5))
hi = max(float(np.nanpercentile(subset["frp"], 95)), lo + 1e-6)

if style == "Thermal points":
    idx = (np.clip((sampled["frp"].to_numpy() - lo) / (hi - lo), 0, 1) * 255).astype(int)
    sampled = sampled.assign(color=[lut[i] for i in idx])
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=sampled[["latitude", "longitude", "color", "frp"]].to_dict("records"),
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius=1500,
        radius_min_pixels=1,
        radius_max_pixels=6,
        opacity=0.7,
        pickable=False,
    )
else:
    layer = pdk.Layer(
        "HeatmapLayer",
        data=sampled[["latitude", "longitude", "frp"]].to_dict("records"),
        get_position=["longitude", "latitude"],
        get_weight="frp",
        radius_pixels=30,
        intensity=1,
        threshold=0.05,
        color_range=HEATMAP_COLORS,
    )

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=WEST_VIEW,
    map_style="dark",
)

st.pydeck_chart(deck, height=640)

if style == "Thermal points":
    st.markdown(legend_html(lut, lo, hi), unsafe_allow_html=True)

st.caption(
    f"Source: NASA FIRMS VIIRS active fire detections, US West{note}. Heatmap weight "
    "and point color both reflect fire radiative power (intensity)."
)
