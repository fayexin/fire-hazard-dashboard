import json
from calendar import month_abbr
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Fire Trends", layout="wide")


DATA_DIR = Path("data/derived")

ANNUAL_PATH = DATA_DIR / "fire_annual_summary.parquet"
MONTHLY_PATH = DATA_DIR / "fire_monthly_climatology.parquet"
YEAR_MONTH_PATH = DATA_DIR / "fire_year_month_summary.parquet"
DAILY_PATH = DATA_DIR / "fire_daily_summary.parquet"
METADATA_PATH = DATA_DIR / "fire_trend_summary_metadata.json"


REQUIRED_FILES = [
    ANNUAL_PATH,
    MONTHLY_PATH,
    YEAR_MONTH_PATH,
    DAILY_PATH,
]


@st.cache_data
def load_parquet(path_str: str, file_mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path_str)


@st.cache_data
def load_metadata(path_str: str, file_mtime: float) -> dict:
    with open(path_str, "r", encoding="utf-8") as file:
        return json.load(file)


def require_files() -> None:
    missing = [path for path in REQUIRED_FILES if not path.exists()]

    if missing:
        missing_text = "\n".join(f"- `{path}`" for path in missing)
        st.error(
            "Trend summary files are missing. Run "
            "`python scripts/build_fire_trends.py` from the repository root.\n\n"
            f"Missing files:\n{missing_text}"
        )
        st.stop()


def add_month_name(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month_name"] = out["month"].astype(int).map(lambda value: month_abbr[value])
    return out


def format_count(value) -> str:
    return f"{int(value):,}"


require_files()

annual = load_parquet(str(ANNUAL_PATH), ANNUAL_PATH.stat().st_mtime)
monthly = load_parquet(str(MONTHLY_PATH), MONTHLY_PATH.stat().st_mtime)
year_month = load_parquet(str(YEAR_MONTH_PATH), YEAR_MONTH_PATH.stat().st_mtime)
daily = load_parquet(str(DAILY_PATH), DAILY_PATH.stat().st_mtime)

annual["year"] = annual["year"].astype(int)
monthly["month"] = monthly["month"].astype(int)
year_month["year"] = year_month["year"].astype(int)
year_month["month"] = year_month["month"].astype(int)
daily["year"] = daily["year"].astype(int)
daily["date"] = pd.to_datetime(daily["date"])

metadata = {}
if METADATA_PATH.exists():
    metadata = load_metadata(str(METADATA_PATH), METADATA_PATH.stat().st_mtime)


st.title("Fire Trends — U.S. West")

st.write(
    "This page summarizes long-term wildfire activity using NASA FIRMS VIIRS "
    "standard-processed active-fire detections. The summaries are derived from "
    "yearly FIRMS files and saved as smaller trend datasets."
)

st.warning(
    "FIRMS detections are satellite thermal anomaly pixels, not official fire "
    "perimeters, burned-area estimates, or fire-size measurements. Detection counts "
    "can also be affected by satellite overpass timing, cloud cover, smoke, and "
    "sensor or processing differences."
)

if metadata:
    st.info(
        f"**Source:** {metadata.get('source', 'Unknown')}  \n"
        f"**Years:** {metadata.get('start_year', '—')} to {metadata.get('end_year', '—')}  \n"
        f"**Total detections in summaries:** {metadata.get('total_detections', '—'):,}"
    )

st.sidebar.header("Trend controls")

all_years = sorted(annual["year"].unique().tolist())

exclude_partial_years = st.sidebar.checkbox(
    "Exclude first and latest year from annual plots",
    value=True,
    help=(
        "The first VIIRS year and the current/latest year may be partial. "
        "Excluding them makes annual comparisons cleaner."
    ),
)

if exclude_partial_years and len(all_years) > 2:
    plot_annual = annual[
        ~annual["year"].isin([min(all_years), max(all_years)])
    ].copy()
else:
    plot_annual = annual.copy()

selected_daily_year = st.sidebar.select_slider(
    "Daily detail year",
    options=all_years,
    value=max(all_years),
)

st.header("Overall summary")

total_detections = int(annual["detections"].sum())
peak_year_row = annual.loc[annual["detections"].idxmax()]
peak_month_row = year_month.loc[year_month["detections"].idxmax()]

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric("Years summarized", f"{min(all_years)}–{max(all_years)}")
metric_col2.metric("Total detections", format_count(total_detections))
metric_col3.metric(
    "Peak detection year",
    f"{int(peak_year_row['year'])} ({format_count(peak_year_row['detections'])})",
)
metric_col4.metric(
    "Peak month",
    f"{month_abbr[int(peak_month_row['month'])]} {int(peak_month_row['year'])}",
)

st.caption(
    "The first and latest years may be partial depending on the available FIRMS "
    "record and the date when the archive was downloaded."
)

st.divider()

st.header("Annual fire activity")

annual_col1, annual_col2 = st.columns(2)

with annual_col1:
    st.subheader("Annual detection count")
    fig_annual_count = px.bar(
        plot_annual,
        x="year",
        y="detections",
        labels={
            "year": "Year",
            "detections": "FIRMS detections",
        },
    )
    fig_annual_count.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_annual_count, use_container_width=True)

with annual_col2:
    st.subheader("Annual sum of observed FRP")
    fig_annual_frp = px.line(
        plot_annual,
        x="year",
        y="sum_observed_frp",
        markers=True,
        labels={
            "year": "Year",
            "sum_observed_frp": "Sum of observed FRP (MW-observations)",
        },
    )
    fig_annual_frp.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_annual_frp, use_container_width=True)

st.caption(
    "The sum of observed FRP is a summary of satellite observations. It is not total "
    "fire energy, burned area, or fire size."
)

st.divider()

st.header("Seasonal pattern")

monthly_named = add_month_name(monthly)

season_col1, season_col2 = st.columns(2)

with season_col1:
    st.subheader("Monthly detection climatology")
    fig_monthly = px.bar(
        monthly_named,
        x="month_name",
        y="detections",
        labels={
            "month_name": "Month",
            "detections": "FIRMS detections",
        },
        category_orders={
            "month_name": [month_abbr[i] for i in range(1, 13)],
        },
    )
    fig_monthly.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_monthly, use_container_width=True)

with season_col2:
    st.subheader("Monthly median observed FRP")
    fig_monthly_frp = px.line(
        monthly_named,
        x="month_name",
        y="median_observed_frp",
        markers=True,
        labels={
            "month_name": "Month",
            "median_observed_frp": "Median observed FRP (MW)",
        },
        category_orders={
            "month_name": [month_abbr[i] for i in range(1, 13)],
        },
    )
    fig_monthly_frp.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_monthly_frp, use_container_width=True)

st.divider()

st.header("Year-month fire activity heatmap")

heatmap_data = (
    year_month.pivot_table(
        index="year",
        columns="month",
        values="detections",
        aggfunc="sum",
        fill_value=0,
    )
    .sort_index()
)

fig_heatmap = px.imshow(
    heatmap_data,
    aspect="auto",
    labels={
        "x": "Month",
        "y": "Year",
        "color": "Detections",
    },
)

fig_heatmap.update_xaxes(
    tickmode="array",
    tickvals=list(range(1, 13)),
    ticktext=[month_abbr[i] for i in range(1, 13)],
)

fig_heatmap.update_layout(margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_heatmap, use_container_width=True)

st.caption(
    "The heatmap shows the number of FIRMS detections for each year and month. "
    "Bright cells indicate months with more satellite-detected active-fire pixels."
)

st.divider()

st.header(f"Daily activity in {selected_daily_year}")

daily_year = daily[daily["year"] == selected_daily_year].copy()

daily_col1, daily_col2 = st.columns(2)

with daily_col1:
    st.subheader("Daily detection count")
    fig_daily_count = px.line(
        daily_year,
        x="date",
        y="detections",
        labels={
            "date": "Date",
            "detections": "FIRMS detections",
        },
    )
    fig_daily_count.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_daily_count, use_container_width=True)

with daily_col2:
    st.subheader("Daily maximum observed FRP")
    fig_daily_frp = px.line(
        daily_year,
        x="date",
        y="max_observed_frp",
        labels={
            "date": "Date",
            "max_observed_frp": "Maximum observed FRP (MW)",
        },
    )
    fig_daily_frp.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_daily_frp, use_container_width=True)

st.divider()

st.header("Highest-activity year-months")

top_months = (
    year_month.sort_values("detections", ascending=False)
    .head(20)
    .copy()
)

top_months["month_name"] = top_months["month"].astype(int).map(
    lambda value: month_abbr[value]
)

display_cols = [
    "year",
    "month_name",
    "detections",
    "max_observed_frp",
    "median_observed_frp",
    "sum_observed_frp",
    "high_confidence_detections",
    "nighttime_detections",
]

display_cols = [column for column in display_cols if column in top_months.columns]

st.dataframe(
    top_months[display_cols],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "This table ranks year-month combinations by FIRMS detection count. Large values "
    "show months with more detected active-fire pixels, not official burned area."
)