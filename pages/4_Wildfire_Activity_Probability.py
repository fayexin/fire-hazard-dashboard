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


def format_probability(
    value: float | None,
) -> str:
    if value is None or pd.isna(value):
        return "—"

    return f"{value:.1%}"


def cleaned_feature_name(
    value: str,
) -> str:
    return (
        str(value)
        .replace("numeric__", "")
        .replace("categorical__", "")
        .replace("state_", "State: ")
        .replace("_", " ")
    )


def filtered_csv(
    frame: pd.DataFrame,
) -> bytes:
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
    scores["month_start"]
    .dropna()
    .unique()
    .tolist()
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

selected_states = st.sidebar.multiselect(
    "States",
    options=available_states,
    default=available_states,
)

minimum_probability = st.sidebar.slider(
    "Minimum displayed probability",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    format="%.0f%%",
)


selected_month = pd.Timestamp(
    selected_month
)

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

display_scores = month_scores[
    month_scores["predicted_probability"]
    >= minimum_probability
].copy()


if month_scores.empty:
    st.info(
        "No county scores match the selected month and states."
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

if display_scores.empty:
    st.info(
        "No counties meet the minimum displayed probability."
    )
else:
    band_order = [
        "Below threshold",
        "Very low",
        "Low",
        "Moderate",
        "High",
        "Very high",
    ]

    band_colors = {
        "Below threshold": "#e5e7eb",
        "Very low": "#fff7bc",
        "Low": "#fec44f",
        "Moderate": "#fe9929",
        "High": "#ec7014",
        "Very high": "#993404",
    }
    
    map_scores = month_scores.copy()

    map_scores["map_band"] = map_scores["probability_band"].astype(str)

    map_scores.loc[
        map_scores["predicted_probability"] < minimum_probability,
        "map_band",
    ] = "Below threshold"
    
    
    map_figure = px.choropleth_map(
        map_scores,
        geojson=county_geojson,
        locations="county_geoid",
        featureidkey="properties.county_geoid",
        color="map_band",
        category_orders={"map_band": band_order},
        color_discrete_map=band_colors,
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
        center={"lat": 40.0, "lon": -115.0},
        zoom=3.35,
        opacity=0.78,
        height=720,
    )
    
    map_figure.update_traces(
        marker_line_width=0.35,
        marker_line_color="rgba(60, 60, 60, 0.55)",
    )
    
    map_figure.update_layout(
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
    
    st.plotly_chart(
        map_figure,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": False,
        },
    )

st.caption(
    "Counties hidden by the minimum-probability control are "
    "not shown. Probability is modeled at county-month scale "
    "and does not indicate where within a county fire activity "
    "may occur."
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
