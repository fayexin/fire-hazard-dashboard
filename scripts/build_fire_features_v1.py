import json
from pathlib import Path

import numpy as np
import pandas as pd


try:
    from build_fire_labels import (
        load_western_counties,
        require_geopandas,
    )
except ImportError:
    from scripts.build_fire_labels import (
        load_western_counties,
        require_geopandas,
    )


TARGET_PATH = Path(
    "data/labels/firms_county_month_targets_v2.parquet"
)

OUT_DIR = Path("data/features")

FEATURE_PATH = (
    OUT_DIR / "county_month_fire_features_v1.parquet"
)

SUMMARY_PATH = (
    OUT_DIR / "county_month_fire_features_v1_summary.json"
)


IDENTIFIER_COLUMNS = [
    "county_geoid",
    "county_name",
    "state_fips",
    "state",
    "year",
    "month",
    "month_start",
    "split",
]

TARGET_COLUMNS = [
    "any_fire_activity",
    "meaningful_fire_activity",
]

CURRENT_MONTH_ACTIVITY_COLUMNS = [
    "detection_count",
    "distinct_detection_days",
    "nominal_or_high_detection_count",
    "nominal_or_high_distinct_days",
    "high_confidence_count",
    "nominal_confidence_count",
    "low_confidence_count",
    "nighttime_count",
    "max_observed_frp",
    "median_observed_frp",
    "sum_observed_frp",
    "nominal_or_high_sum_observed_frp",
]


def load_targets() -> pd.DataFrame:
    """Load and validate the county-month target table."""
    if not TARGET_PATH.exists():
        raise SystemExit(
            f"Target file not found: {TARGET_PATH}\n"
            "Run `python scripts/build_fire_targets_v2.py` first."
        )

    df = pd.read_parquet(TARGET_PATH)

    required_columns = [
        "county_geoid",
        "county_name",
        "state_fips",
        "state",
        "year",
        "month",
        "any_fire_activity",
        "meaningful_fire_activity",
        "detection_count",
        "nominal_or_high_detection_count",
        "high_confidence_count",
        "max_observed_frp",
        "sum_observed_frp",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{TARGET_PATH} is missing columns: {missing}"
        )

    df["county_geoid"] = (
        df["county_geoid"]
        .astype(str)
        .str.zfill(5)
    )

    df["state_fips"] = (
        df["state_fips"]
        .astype(str)
        .str.zfill(2)
    )

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    df["any_fire_activity"] = (
        df["any_fire_activity"].astype(int)
    )

    df["meaningful_fire_activity"] = (
        df["meaningful_fire_activity"].astype(int)
    )

    if "month_start" in df.columns:
        df["month_start"] = pd.to_datetime(
            df["month_start"],
            errors="coerce",
        )
    else:
        df["month_start"] = pd.to_datetime(
            {
                "year": df["year"],
                "month": df["month"],
                "day": 1,
            }
        )

    if df["month_start"].isna().any():
        raise ValueError(
            "Some target rows have invalid month_start values."
        )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "county_geoid",
                "year",
                "month",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count:,} duplicate county-month rows."
        )

    df = df.sort_values(
        [
            "county_geoid",
            "month_start",
        ]
    ).reset_index(drop=True)

    return df


def validate_monthly_panel(df: pd.DataFrame) -> None:
    """Confirm that each county has a continuous monthly sequence."""
    month_ordinal = (
        df["year"] * 12
        + df["month"]
    )

    differences = (
        month_ordinal
        .groupby(df["county_geoid"])
        .diff()
    )

    invalid = differences[
        differences.notna()
        & differences.ne(1)
    ]

    if not invalid.empty:
        raise ValueError(
            "The target panel contains missing or repeated months "
            f"for {len(invalid):,} county transitions."
        )


def build_county_static_features() -> pd.DataFrame:
    """Create county area and centroid features."""
    gpd = require_geopandas()
    counties = load_western_counties()

    counties["county_geoid"] = (
        counties["county_geoid"]
        .astype(str)
        .str.zfill(5)
    )

    projected = counties.to_crs("EPSG:5070")

    centroid_projected = projected.geometry.centroid

    centroid_geo = gpd.GeoSeries(
        centroid_projected,
        crs="EPSG:5070",
    ).to_crs("EPSG:4326")

    static = counties[
        [
            "county_geoid",
        ]
    ].copy()

    static["county_area_km2"] = (
        projected.geometry.area
        / 1_000_000
    )

    static["county_centroid_longitude"] = (
        centroid_geo.x.to_numpy()
    )

    static["county_centroid_latitude"] = (
        centroid_geo.y.to_numpy()
    )

    return static


def assign_split(year: pd.Series) -> pd.Series:
    """Assign fixed temporal development splits."""
    conditions = [
        year.between(2013, 2021),
        year.eq(2022),
        year.between(2023, 2025),
    ]

    choices = [
        "train",
        "validation",
        "test",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="excluded",
        ),
        index=year.index,
    )


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add known-at-prediction-time temporal features."""
    out = df.copy()

    angle = (
        2
        * np.pi
        * (out["month"] - 1)
        / 12
    )

    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)

    out["month_ordinal"] = (
        out["year"] * 12
        + out["month"]
    )

    out["months_since_panel_start"] = (
        out["month_ordinal"]
        - out["month_ordinal"].min()
    )

    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create county-level features using only earlier months."""
    out = df.copy()

    grouped = out.groupby(
        "county_geoid",
        sort=False,
    )

    lag_columns = {
        "any_fire_activity": "any_activity",
        "meaningful_fire_activity": "meaningful_activity",
        "detection_count": "detection_count",
        "nominal_or_high_detection_count": (
            "eligible_detection_count"
        ),
        "high_confidence_count": (
            "high_confidence_count"
        ),
        "max_observed_frp": "max_observed_frp",
        "sum_observed_frp": "sum_observed_frp",
    }

    lag_months = [
        1,
        2,
        3,
        6,
        12,
    ]

    for source_column, output_prefix in lag_columns.items():
        for lag in lag_months:
            out[
                f"{output_prefix}_lag_{lag}m"
            ] = grouped[source_column].shift(lag)

    rolling_windows = [
        3,
        6,
        12,
    ]

    for window in rolling_windows:
        out[
            f"meaningful_positive_months_prev_{window}m"
        ] = grouped[
            "meaningful_fire_activity"
        ].transform(
            lambda values: (
                values
                .shift(1)
                .rolling(
                    window=window,
                    min_periods=1,
                )
                .sum()
            )
        )

        out[
            f"any_positive_months_prev_{window}m"
        ] = grouped[
            "any_fire_activity"
        ].transform(
            lambda values: (
                values
                .shift(1)
                .rolling(
                    window=window,
                    min_periods=1,
                )
                .sum()
            )
        )

        out[
            f"eligible_detections_prev_{window}m"
        ] = grouped[
            "nominal_or_high_detection_count"
        ].transform(
            lambda values: (
                values
                .shift(1)
                .rolling(
                    window=window,
                    min_periods=1,
                )
                .sum()
            )
        )

        out[
            f"total_detections_prev_{window}m"
        ] = grouped[
            "detection_count"
        ].transform(
            lambda values: (
                values
                .shift(1)
                .rolling(
                    window=window,
                    min_periods=1,
                )
                .sum()
            )
        )

        out[
            f"mean_eligible_detections_prev_{window}m"
        ] = grouped[
            "nominal_or_high_detection_count"
        ].transform(
            lambda values: (
                values
                .shift(1)
                .rolling(
                    window=window,
                    min_periods=1,
                )
                .mean()
            )
        )

    prior_month_count = grouped.cumcount()

    prior_meaningful_count = (
        grouped[
            "meaningful_fire_activity"
        ].cumsum()
        - out["meaningful_fire_activity"]
    )

    prior_any_count = (
        grouped[
            "any_fire_activity"
        ].cumsum()
        - out["any_fire_activity"]
    )

    out["county_prior_meaningful_rate"] = (
        prior_meaningful_count
        / prior_month_count.replace(0, np.nan)
    )

    out["county_prior_any_rate"] = (
        prior_any_count
        / prior_month_count.replace(0, np.nan)
    )

    out["prior_observed_months"] = (
        prior_month_count.astype(int)
    )

    return out


def add_months_since_previous_activity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate months since prior meaningful activity without leakage."""
    out = df.copy()

    result = pd.Series(
        np.nan,
        index=out.index,
        dtype=float,
    )

    for _, index_values in out.groupby(
        "county_geoid",
        sort=False,
    ).groups.items():
        indexes = list(index_values)

        month_ordinals = (
            out.loc[indexes, "month_ordinal"]
            .to_numpy()
        )

        target_values = (
            out.loc[
                indexes,
                "meaningful_fire_activity",
            ]
            .to_numpy()
        )

        county_result = np.full(
            len(indexes),
            np.nan,
            dtype=float,
        )

        last_positive_month = None

        for position, (
            month_ordinal,
            target_value,
        ) in enumerate(
            zip(
                month_ordinals,
                target_values,
            )
        ):
            if last_positive_month is not None:
                county_result[position] = (
                    month_ordinal
                    - last_positive_month
                )

            if target_value == 1:
                last_positive_month = month_ordinal

        result.loc[indexes] = county_result

    out[
        "months_since_previous_meaningful_activity"
    ] = result

    out[
        "no_previous_meaningful_activity"
    ] = result.isna().astype(int)

    return out


def build_model_table(
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a leakage-safe model table."""
    validate_monthly_panel(targets)

    features = add_seasonal_features(targets)
    features = add_lag_features(features)
    features = add_months_since_previous_activity(
        features
    )

    county_static = build_county_static_features()

    features = features.merge(
        county_static,
        on="county_geoid",
        how="left",
        validate="many_to_one",
    )

    features["split"] = assign_split(
        features["year"]
    )

    feature_columns = [
        "month_sin",
        "month_cos",
        "months_since_panel_start",
        "county_area_km2",
        "county_centroid_latitude",
        "county_centroid_longitude",
        "prior_observed_months",
        "county_prior_meaningful_rate",
        "county_prior_any_rate",
        "months_since_previous_meaningful_activity",
        "no_previous_meaningful_activity",
    ]

    feature_columns.extend(
        sorted(
            column
            for column in features.columns
            if (
                "_lag_" in column
                or "_prev_" in column
            )
        )
    )

    feature_columns = list(
        dict.fromkeys(feature_columns)
    )

    output_columns = (
        IDENTIFIER_COLUMNS
        + TARGET_COLUMNS
        + feature_columns
    )

    model_table = features[
        output_columns
    ].copy()

    leaked_columns = [
        column
        for column in CURRENT_MONTH_ACTIVITY_COLUMNS
        if column in model_table.columns
    ]

    if leaked_columns:
        raise ValueError(
            "Current-month activity columns leaked into "
            f"the model table: {leaked_columns}"
        )

    if model_table[
        [
            "county_area_km2",
            "county_centroid_latitude",
            "county_centroid_longitude",
        ]
    ].isna().any().any():
        raise ValueError(
            "Some county static features are missing."
        )

    return model_table, feature_columns


def target_summary(
    df: pd.DataFrame,
    target: str,
) -> dict:
    """Summarize a binary target."""
    rows = len(df)

    if rows == 0:
        return {
            "rows": 0,
            "positive_count": 0,
            "positive_rate": None,
        }

    positives = int(df[target].sum())

    return {
        "rows": int(rows),
        "positive_count": positives,
        "positive_rate": float(
            positives / rows
        ),
    }


def build_summary(
    model_table: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """Create model-table metadata and validation summaries."""
    split_summary = {}

    for split_name in [
        "train",
        "validation",
        "test",
        "excluded",
    ]:
        split_data = model_table[
            model_table["split"] == split_name
        ]

        split_summary[split_name] = {
            "rows": int(len(split_data)),
            "start_year": (
                int(split_data["year"].min())
                if len(split_data)
                else None
            ),
            "end_year": (
                int(split_data["year"].max())
                if len(split_data)
                else None
            ),
            "any_fire_activity": target_summary(
                split_data,
                "any_fire_activity",
            ),
            "meaningful_fire_activity": target_summary(
                split_data,
                "meaningful_fire_activity",
            ),
        }

    missing_rates = {
        column: float(
            model_table[column].isna().mean()
        )
        for column in feature_columns
    }

    return {
        "input": str(TARGET_PATH),
        "output": str(FEATURE_PATH),
        "rows": int(len(model_table)),
        "counties": int(
            model_table["county_geoid"].nunique()
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "feature_columns": feature_columns,
        "split_definition": {
            "train": "2013-2021",
            "validation": "2022",
            "test": "2023-2025",
            "excluded": "2012 and 2026",
        },
        "split_summary": split_summary,
        "feature_missing_rates": missing_rates,
        "leakage_control": (
            "All fire-history features are shifted by at least "
            "one month. Current-month detection statistics are "
            "not included as predictors."
        ),
        "note": (
            "This is a fire-history, seasonality, and county-"
            "geography feature table. Weather variables are not "
            "included yet."
        ),
    }


def main() -> None:
    targets = load_targets()

    model_table, feature_columns = build_model_table(
        targets
    )

    summary = build_summary(
        model_table,
        feature_columns,
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_table.to_parquet(
        FEATURE_PATH,
        index=False,
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        f"Saved {len(model_table):,} rows "
        f"to {FEATURE_PATH}"
    )

    print(
        f"Saved {len(feature_columns):,} features "
        f"to {SUMMARY_PATH}"
    )

    for split_name, values in summary[
        "split_summary"
    ].items():
        meaningful = values[
            "meaningful_fire_activity"
        ]

        print(
            f"{split_name}: "
            f"{values['rows']:,} rows, "
            "meaningful positive rate = "
            f"{meaningful['positive_rate']}"
        )


if __name__ == "__main__":
    main()