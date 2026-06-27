import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Wildfire Activity Probability",
    layout="wide",
)


SCORES_PATH = Path(
    "data/model_outputs/fire_probability_deployment_scores_v1.parquet"
)

LATEST_SCORES_PATH = Path(
    "data/model_outputs/fire_probability_latest_scores_v1.parquet"
)

IMPORTANCE_PATH = Path(
    "data/model_outputs/xgboost_deployment_feature_importance_v1.parquet"
)

SUMMARY_PATH = Path(
    "data/model_outputs/fire_probability_deployment_summary_v1.json"
)

GEOJSON_PATH = Path(
    "data/context/western_counties_simplified.geojson"
)


WEST_MAP_CENTER = {"lat": 40.0, "lon": -115.0}
WEST_MAP_ZOOM = 4.2
MAP_HEIGHT = 720


BAND_ORDER = [
    "Below threshold",
    "Very low",
    "Low",
    "Moderate",
    "High",
    "Very high",
]

BAND_COLORS = {
    "Below threshold": "#e5e7eb",
    "Very low": "#fff7bc",
    "Low": "#fec44f",
    "Moderate": "#fe9929",
    "High": "#ec7014",
    "Very high": "#993404",
}


@st.cache_data
def load_parquet(
    path_str: str,
    file_mtime: float,
) -> pd.DataFrame:
    return pd.read_parquet(path_str)


@st.cache_data
def load_json(
    path_str: str,
    file_mtime: float,
) -> dict:
    with open(
        path_str,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def require_files() -> None:
    required = [
        SCORES_PATH,
        IMPORTANCE_PATH,
        SUMMARY_PATH,
        GEOJSON_PATH,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        missing_text = "\n".join(
            f"- `{path}`"
            for path in missing
        )

        st.error(
            "Probability-page files are missing.\n\n"
            f"{missing_text}\n\n"
            "Run the final-model and county-geometry scripts "
            "from the repository root."
        )

        st.stop()


def format_probability(value) -> str:
    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):.1%}"


def cleaned_feature_name(value: str) -> str:
    return (
        str(value)
        .replace("numeric__", "")
        .replace("categorical__", "")
        .replace("state_", "State: ")
        .replace("_", " ")
    )


def filtered_csv(frame: pd.DataFrame) -> bytes:
    columns = [
        "month_start",
        "state",
        "county_name",
        "county_geoid",
        "predicted_probability",
        "probability_band",
        "score_role",
    ]

    columns = [
        column
        for column in columns
        if column in frame.columns
    ]

    return (
        frame[columns]
        .to_csv(index=False)
        .encode("utf-8")
    )


def blank_probability_map():
    """Show the western U.S. basemap without county probability overlays."""
    dummy = pd.DataFrame(
        {
            "latitude": [WEST_MAP_CENTER["lat"]],
            "longitude": [WEST_MAP_CENTER["lon"]],
        }
    )

    if hasattr(px, "scatter_map"):
        figure = px.scatter_map(
            dummy,
            lat="latitude",
            lon="longitude",
            map_style="carto-positron",
            center=WEST_MAP_CENTER,
            zoom=WEST_MAP_ZOOM,
            height=MAP_HEIGHT,
        )
    else:
        figure = px.scatter_mapbox(
            dummy,
            lat="latitude",
            lon="longitude",
            mapbox_style="carto-positron",
            center=WEST_MAP_CENTER,
            zoom=WEST_MAP_ZOOM,
            height=MAP_HEIGHT,
        )

    figure.update_traces(
        marker_size=0,
        marker_opacity=0,
        hoverinfo="skip",
        showlegend=False,
    )

    figure.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return figure


def county_probability_map(
    map_scores: pd.DataFrame,
    county_geojson: dict,
):
    """Create the county probability map with a light basemap."""
    if hasattr(px, "choropleth_map"):
        figure = px.choropleth_map(
            map_scores,
            geojson=county_geojson,
            locations="county_geoid",
            featureidkey="properties.county_geoid",
            color="map_band",
            category_orders={"map_band": BAND_ORDER},
            color_discrete_map=BAND_COLORS,
            hover_name="county_name",
            hover_data={
                "state": True,
                "county_geoid": False,
                "predicted_probability": ":.1%",
                "probability_band": True,
                "map_band": False,
            },
            labels={
                "state": "State",
                "predicted_probability": "Probability",
                "probability_band": "Probability band",
            },
            map_style="carto-positron",
            center=WEST_MAP_CENTER,
            zoom=WEST_MAP_ZOOM,
            opacity=0.78,
            height=MAP_HEIGHT,
        )
    else:
        figure = px.choropleth_mapbox(
            map_scores,
            geojson=county_geojson,
            locations="county_geoid",
            featureidkey="properties.county_geoid",
            color="map_band",
            category_orders={"map_band": BAND_ORDER},
            color_discrete_map=BAND_COLORS,
            hover_name="county_name",
            hover_data={
                "state": True,
                "county_geoid": False,
                "predicted_probability": ":.1%",
                "probability_band": True,
                "map_band": False,
            },
            labels={
                "state": "State",
                "predicted_probability": "Probability",
                "probability_band": "Probability band",
            },
            mapbox_style="carto-positron",
            center=WEST_MAP_CENTER,
            zoom=WEST_MAP_ZOOM,
            opacity=0.78,
            height=MAP_HEIGHT,
        )

    figure.update_traces(
        marker_line_width=0.35,
        marker_line_color="rgba(60, 60, 60, 0.55)",
    )

    figure.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            title="Probability band",
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )

    return figure


require_files()

scores = load_parquet(
    str(SCORES_PATH),
    SCORES_PATH.stat().st_mtime,
)

importance = load_parquet(
    str(IMPORTANCE_PATH),
    IMPORTANCE_PATH.stat().st_mtime,
)

summary = load_json(
    str(SUMMARY_PATH),
    SUMMARY_PATH.stat().st_mtime,
)

county_geojson = load_json(
    str(GEOJSON_PATH),
    GEOJSON_PATH.stat().st_mtime,
)


scores["county_geoid"] = (
    scores["county_geoid"]
    .astype(str)
    .str.zfill(5)
)

scores["month_start"] = pd.to_datetime(
    scores["month_start"],
    errors="coerce",
)

scores["predicted_probability"] = pd.to_numeric(
    scores["predicted_probability"],
    errors="coerce",
)

scores = scores.dropna(
    subset=[
        "month_start",
        "predicted_probability",
    ]
).copy()


st.title(
    "Wildfire Activity Probability — U.S. West"
)

st.write(
    "This page shows county-month probabilities of meaningful "
    "NASA FIRMS VIIRS fire activity. The model combines prior "
    "fire activity, seasonality, county geography, and lagged "
    "gridMET weather conditions."
)

st.warning(
    "Research and visualization use only. These probabilities "
    "are not official fire-danger ratings, ignition forecasts, "
    "evacuation guidance, emergency warnings, or structure-level "
    "risk estimates."
)

with st.expander(
    "Prediction target and timing",
    expanded=False,
):
    st.markdown(
        """
        The target is positive when a county-month has at least
        **5 nominal-or-high-confidence FIRMS detections** across
        at least **2 distinct days**.

        For prediction month t, weather predictors use only
        observations from earlier months. The model does not use
        target-month FIRMS detections as predictors.

        Probability bands on this page are descriptive display
        categories. They are not official fire-danger classes.
        """
    )


available_months = sorted(
    pd.to_datetime(
        scores["month_start"]
        .dropna()
        .unique()
    ).tolist()
)

available_states = sorted(
    scores["state"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


st.sidebar.header(
    "Probability controls"
)

selected_month = st.sidebar.selectbox(
    "Prediction month",
    options=available_months,
    index=len(available_months) - 1,
    format_func=lambda value: pd.Timestamp(
        value
    ).strftime("%B %Y"),
)

st.sidebar.markdown("**States**")

# Initialize all states as selected on the first page load.
if "probability_states_initialized" not in st.session_state:
    for state in available_states:
        st.session_state[f"probability_state_{state}"] = True
    st.session_state["probability_states_initialized"] = True

button_col1, button_col2 = st.sidebar.columns(2)

with button_col1:
    if st.button("Select all", use_container_width=True):
        for state in available_states:
            st.session_state[f"probability_state_{state}"] = True

with button_col2:
    if st.button("Clear", use_container_width=True):
        for state in available_states:
            st.session_state[f"probability_state_{state}"] = False

with st.sidebar.expander("Choose states", expanded=True):
    state_cols = st.columns(3)

    for index, state in enumerate(available_states):
        with state_cols[index % 3]:
            st.checkbox(
                state,
                key=f"probability_state_{state}",
            )

selected_states = [
    state
    for state in available_states
    if st.session_state.get(
        f"probability_state_{state}",
        False,
    )
]

if not selected_states:
    st.sidebar.info(
        "No states selected. Showing blank basemap."
    )

minimum_probability_percent = st.sidebar.slider(
    "Minimum displayed probability",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
    format="%d%%",
)

minimum_probability = (
    minimum_probability_percent / 100
)


selected_month = pd.Timestamp(
    selected_month
)

if selected_states:
    month_scores = scores[
        (
            scores["month_start"]
            == selected_month
        )
        & (
            scores["state"]
            .astype(str)
            .isin(selected_states)
        )
    ].copy()
else:
    month_scores = scores.iloc[0:0].copy()


if month_scores.empty:
    st.divider()

    st.header(
        f"County probability map — "
        f"{selected_month.strftime('%B %Y')}"
    )

    st.info(
        "No states are selected. The basemap is shown without county probability overlays."
    )

    st.plotly_chart(
        blank_probability_map(),
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": False,
        },
    )

    st.stop()


score_roles = set(
    month_scores["score_role"]
    .dropna()
    .astype(str)
)

if score_roles == {
    "deployment_training_period"
}:
    st.info(
        "The selected month is inside the deployment-model "
        "training period. These are in-sample deployment scores "
        "and must not be read as model evaluation results."
    )
else:
    st.info(
        "The selected month is after the final deployment "
        "training year. It is a post-training model score."
    )


high_count = int(
    (
        month_scores["predicted_probability"]
        >= 0.60
    ).sum()
)

very_high_count = int(
    (
        month_scores["predicted_probability"]
        >= 0.80
    ).sum()
)

top_row = month_scores.sort_values(
    "predicted_probability",
    ascending=False,
).iloc[0]


metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "Counties scored",
    f"{len(month_scores):,}",
)

metric_col2.metric(
    "Median probability",
    format_probability(
        month_scores[
            "predicted_probability"
        ].median()
    ),
)

metric_col3.metric(
    "Counties at 60%+",
    f"{high_count:,}",
)

metric_col4.metric(
    "Highest probability",
    format_probability(
        top_row["predicted_probability"]
    ),
    help=(
        f"{top_row['county_name']}, "
        f"{top_row['state']}"
    ),
)


st.divider()

st.header(
    f"County probability map — "
    f"{selected_month.strftime('%B %Y')}"
)

map_scores = month_scores.copy()

map_scores["map_band"] = (
    map_scores["probability_band"]
    .astype(str)
)

map_scores.loc[
    map_scores["predicted_probability"]
    < minimum_probability,
    "map_band",
] = "Below threshold"

map_figure = county_probability_map(
    map_scores,
    county_geojson,
)

st.plotly_chart(
    map_figure,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": False,
    },
)

st.caption(
    "Counties below the minimum-probability threshold are shown in gray. "
    "Probability is modeled at county-month scale and does not indicate "
    "where within a county fire activity may occur."
)


st.divider()

ranking_col, distribution_col = st.columns(2)

with ranking_col:
    st.subheader(
        "Highest-probability counties"
    )

    ranking = (
        month_scores.sort_values(
            "predicted_probability",
            ascending=False,
        )
        .head(25)
        .copy()
    )

    ranking["probability"] = ranking[
        "predicted_probability"
    ].map(
        lambda value: f"{value:.1%}"
    )

    st.dataframe(
        ranking[
            [
                "state",
                "county_name",
                "probability",
                "probability_band",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with distribution_col:
    st.subheader(
        "Probability distribution"
    )

    distribution_figure = px.histogram(
        month_scores,
        x="predicted_probability",
        nbins=30,
        labels={
            "predicted_probability": (
                "Predicted probability"
            ),
            "count": "Counties",
        },
    )

    distribution_figure.update_xaxes(
        tickformat=".0%",
    )

    distribution_figure.update_layout(
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
        yaxis_title="Counties",
    )

    st.plotly_chart(
        distribution_figure,
        use_container_width=True,
    )


st.download_button(
    label=(
        "Download selected county probabilities"
    ),
    data=filtered_csv(
        month_scores.sort_values(
            "predicted_probability",
            ascending=False,
        )
    ),
    file_name=(
        "wildfire_activity_probability_"
        f"{selected_month.strftime('%Y_%m')}.csv"
    ),
    mime="text/csv",
)


st.divider()

st.header(
    "Locked model evaluation"
)

locked_metrics = (
    summary
    .get("locked_evaluation", {})
    .get("test_metrics", {})
)

evaluation_col1, evaluation_col2, evaluation_col3, evaluation_col4 = st.columns(4)

evaluation_col1.metric(
    "Test PR-AUC",
    (
        f"{locked_metrics.get('pr_auc'):.3f}"
        if locked_metrics.get("pr_auc") is not None
        else "—"
    ),
)

evaluation_col2.metric(
    "Test ROC-AUC",
    (
        f"{locked_metrics.get('roc_auc'):.3f}"
        if locked_metrics.get("roc_auc") is not None
        else "—"
    ),
)

evaluation_col3.metric(
    "Test Brier score",
    (
        f"{locked_metrics.get('brier_score'):.3f}"
        if locked_metrics.get("brier_score") is not None
        else "—"
    ),
)

evaluation_col4.metric(
    "Calibration error",
    (
        f"{locked_metrics.get('expected_calibration_error_10_bins'):.3f}"
        if locked_metrics.get(
            "expected_calibration_error_10_bins"
        ) is not None
        else "—"
    ),
)

st.caption(
    "The locked test period is 2023–2025. These metrics come "
    "from the earlier evaluation model trained on 2013–2021 "
    "and selected using 2022 validation. They were not recomputed "
    "after the deployment model was refit through 2025."
)


st.divider()

st.header(
    "Model feature importance"
)

importance_display = importance.copy()

importance_display["feature_label"] = importance_display[
    "feature"
].map(
    cleaned_feature_name
)

importance_display = (
    importance_display.head(20)
    .sort_values(
        "importance",
        ascending=True,
    )
)

importance_figure = px.bar(
    importance_display,
    x="importance",
    y="feature_label",
    orientation="h",
    labels={
        "importance": (
            "XGBoost split importance"
        ),
        "feature_label": "Feature",
    },
)

importance_figure.update_layout(
    height=650,
    margin=dict(
        l=10,
        r=10,
        t=30,
        b=10,
    ),
)

st.plotly_chart(
    importance_figure,
    use_container_width=True,
)

st.caption(
    "XGBoost feature importance summarizes how often and how "
    "usefully features contributed to tree splits. It does not "
    "show causal effects and does not indicate whether a feature "
    "raises or lowers probability."
)


with st.expander(
    "Model limitations",
    expanded=False,
):
    st.markdown(
        """
        - The target is based on FIRMS thermal-anomaly detections,
          not confirmed wildfire ignitions or official perimeters.
        - County weather uses the nearest gridMET cell to one
          representative point, not a full county-area average.
        - Most predictive power comes from season, geography, and
          prior fire activity; lagged weather adds a smaller gain.
        - Human ignitions, lightning, fuels, vegetation, suppression,
          and exposure are not fully represented.
        - A county probability does not identify a specific ignition
          location or predict fire spread.
        """
    )