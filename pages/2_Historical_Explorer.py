import json
from datetime import date, datetime, timedelta
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


st.set_page_config(page_title="Historical Fire Explorer", layout="wide")


DATA_DIR = Path("data/fires")

# GIBS true-color satellite imagery, dated. VIIRS covers the 2012+ archive.
GIBS_TEMPLATE = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
    "VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{date}/"
    "GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
)

CONFIDENCE_LABELS = {"l": "Low", "n": "Nominal", "h": "High"}


# name, peak date, lat, lon, blurb
EVENTS = [
    ("Camp Fire (2018)", "2018-11-08", 39.8, -121.6,
     "Paradise, CA — the deadliest and most destructive wildfire in "
     "California history."),
    ("Dixie Fire (2021)", "2021-07-24", 40.1, -121.2,
     "The largest single wildfire in California's recorded history, burning "
     "nearly one million acres."),
    ("August Complex (2020)", "2020-08-17", 39.8, -122.9,
     "California's first 'gigafire', exceeding one million acres in the "
     "2020 record season."),
    ("Caldor Fire (2021)", "2021-08-29", 38.6, -120.2,
     "Burned from the Sierra foothills across the crest, threatening the "
     "Lake Tahoe basin."),
    ("Creek Fire (2020)", "2020-09-06", 37.2, -119.3,
     "An explosive Sierra National Forest fire that trapped campers and "
     "generated a fire-driven thunderstorm."),
    ("Carr Fire (2018)", "2018-07-27", 40.6, -122.5,
     "Produced a destructive fire whirl ('firenado') near Redding."),
    ("Thomas Fire (2017)", "2017-12-12", 34.5, -119.2,
     "A massive December fire across Ventura and Santa Barbara counties."),
    ("Bootleg Fire (2021)", "2021-07-14", 42.7, -121.4,
     "The largest Oregon wildfire of 2021, large enough to influence local "
     "weather."),
]


@st.cache_data
def available_years():
    files = sorted(DATA_DIR.glob("viirs_west_[0-9]*.parquet"))
    years = []
    for file in files:
        stem = file.stem.split("_")[-1]
        if stem.isdigit():
            years.append(int(stem))
    return years


@st.cache_data
def load_year(year):
    path = DATA_DIR / f"viirs_west_{year}.parquet"
    df = pd.read_parquet(path)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    return df


def gibs_layer(image_date):
    return pdk.Layer(
        "TileLayer",
        data=GIBS_TEMPLATE.format(date=image_date),
        min_zoom=0,
        max_zoom=9,
        tile_size=256,
        opacity=1.0,
    )


st.title("Historical Fire Explorer")

st.write(
    "Major wildfires of the past decade, shown over the true-color satellite image "
    "of the day. The background is the actual NASA satellite view on the selected "
    "date, so smoke plumes are often visible beneath the fire detections. Pick a "
    "famous fire, or explore any date in the record."
)


years = available_years()

if not years:
    st.error(
        "No historical fire data found. Run fetch_firms.py --all (or --year YYYY) "
        "to create data/fires/viirs_west_<year>.parquet files."
    )
    st.stop()


st.sidebar.header("Controls")

event_names = [event[0] for event in EVENTS] + ["Custom date"]

choice = st.sidebar.selectbox("Fire event", event_names)

if choice == "Custom date":
    selected_year = st.sidebar.select_slider(
        "Year", options=years, value=years[-1]
    )
    available_dates = sorted(
        load_year(selected_year)["acq_date"].dt.date.unique()
    )
    selected_date = st.sidebar.select_slider(
        "Date",
        options=available_dates,
        value=available_dates[len(available_dates) // 2],
        format_func=lambda d: d.strftime("%b %d, %Y"),
    )
    center_lat, center_lon = 39.5, -120.5
    blurb = None
    window_days = st.sidebar.slider("Days of detections to show", 1, 14, 3)
else:
    event = next(e for e in EVENTS if e[0] == choice)
    _, date_str, center_lat, center_lon, blurb = event
    selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    selected_year = selected_date.year
    window_days = st.sidebar.slider("Days of detections to show", 1, 14, 3)


show_imagery = st.sidebar.checkbox("Satellite imagery background", value=True)


if selected_year not in years:
    st.warning(
        f"Detection data for {selected_year} is not available yet. The satellite "
        "image will still display."
    )
    detections = pd.DataFrame(columns=["latitude", "longitude", "frp"])
else:
    year_data = load_year(selected_year)
    start = pd.Timestamp(selected_date)
    end = start + pd.Timedelta(days=window_days)
    detections = year_data[
        (year_data["acq_date"] >= start) & (year_data["acq_date"] < end)
    ].copy()


st.subheader(f"{choice if choice != 'Custom date' else 'Custom date'} — "
             f"{selected_date.strftime('%B %d, %Y')}")

if blurb:
    st.markdown(f"*{blurb}*")

col1, col2, col3 = st.columns(3)
col1.metric("Detections shown", f"{len(detections):,}")
col2.metric(
    "Max fire power",
    f"{detections['frp'].max():.0f} MW" if len(detections) else "—",
)
col3.metric("Window", f"{window_days} days")


layers = []

if show_imagery:
    layers.append(gibs_layer(selected_date.isoformat()))

if len(detections):
    detections["radius"] = np.sqrt(detections["frp"].clip(lower=1)) * 220
    detections["date_str"] = detections["acq_date"].dt.strftime("%Y-%m-%d")
    if "confidence" in detections.columns:
        detections["conf_label"] = (
            detections["confidence"].map(CONFIDENCE_LABELS).fillna("—")
        )
    else:
        detections["conf_label"] = "—"

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=detections.to_dict(orient="records"),
            get_position=["longitude", "latitude"],
            get_fill_color=[255, 90, 30, 220],
            get_radius="radius",
            radius_min_pixels=2,
            radius_max_pixels=40,
            opacity=0.85,
            pickable=True,
            stroked=False,
        )
    )

deck = pdk.Deck(
    layers=layers,
    initial_view_state=pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=7.5,
        min_zoom=4,
        max_zoom=12,
    ),
    map_style=None if show_imagery else "dark",
    tooltip={
        "html": "<b>{date_str}</b><br/>Fire power: {frp} MW<br/>"
                "Confidence: {conf_label}"
    },
)

st.pydeck_chart(deck, height=640)

st.caption(
    "Background: NASA GIBS true-color imagery (VIIRS) for the selected date. Points: "
    "NASA FIRMS VIIRS active fire detections, sized by fire radiative power. Imagery "
    "is daily; cloud cover or satellite timing can affect how clearly a plume shows."
)
