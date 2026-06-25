import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd


# Support both:
#   python scripts/build_fire_targets_v2.py
# and:
#   python -m scripts.build_fire_targets_v2
try:
    from build_fire_labels import (
        RAW_DIR,
        RAW_PATTERN,
        SOURCE,
        load_western_counties,
        require_geopandas,
    )
except ImportError:
    from scripts.build_fire_labels import (
        RAW_DIR,
        RAW_PATTERN,
        SOURCE,
        load_western_counties,
        require_geopandas,
    )


OUT_DIR = Path("data/labels")

GROUP_COLUMNS = [
    "county_geoid",
    "county_name",
    "state_fips",
    "state",
    "year",
    "month",
]

COUNT_COLUMNS = [
    "detection_count",
    "distinct_detection_days",
    "nominal_or_high_detection_count",
    "nominal_or_high_distinct_days",
    "high_confidence_count",
    "nominal_confidence_count",
    "low_confidence_count",
    "nighttime_count",
]

FLOAT_COLUMNS = [
    "max_observed_frp",
    "median_observed_frp",
    "sum_observed_frp",
    "nominal_or_high_sum_observed_frp",
]


def load_firms_file(path: Path) -> pd.DataFrame:
    """Load one historical FIRMS file and prepare target-building fields."""
    df = pd.read_parquet(path)

    required_columns = [
        "acq_date",
        "latitude",
        "longitude",
        "frp",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    optional_columns = [
        "confidence",
        "daynight",
        "source",
    ]

    keep_columns = required_columns + [
        column
        for column in optional_columns
        if column in df.columns
    ]

    df = df[keep_columns].copy()

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )
    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )
    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )
    df["frp"] = pd.to_numeric(
        df["frp"],
        errors="coerce",
    ).fillna(0.0)

    if "confidence" not in df.columns:
        df["confidence"] = "unknown"

    if "daynight" not in df.columns:
        df["daynight"] = "unknown"

    if "source" not in df.columns:
        df["source"] = SOURCE

    df["confidence"] = (
        df["confidence"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["daynight"] = (
        df["daynight"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["source"] = (
        df["source"]
        .fillna(SOURCE)
        .astype(str)
    )

    df = df.dropna(
        subset=[
            "acq_date",
            "latitude",
            "longitude",
        ]
    ).copy()

    df["year"] = df["acq_date"].dt.year.astype(int)
    df["month"] = df["acq_date"].dt.month.astype(int)
    df["acq_day"] = df["acq_date"].dt.normalize()

    df["is_high_confidence"] = (
        df["confidence"] == "h"
    ).astype(int)

    df["is_nominal_confidence"] = (
        df["confidence"] == "n"
    ).astype(int)

    df["is_low_confidence"] = (
        df["confidence"] == "l"
    ).astype(int)

    df["is_nominal_or_high"] = (
        df["confidence"].isin(["n", "h"])
    ).astype(int)

    df["is_nighttime"] = (
        df["daynight"] == "N"
    ).astype(int)

    # Missing eligible days are ignored by nunique().
    df["nominal_or_high_day"] = df["acq_day"].where(
        df["is_nominal_or_high"] == 1
    )

    # Used to summarize FRP only among nominal/high detections.
    df["nominal_or_high_frp"] = df["frp"].where(
        df["is_nominal_or_high"] == 1
    )

    return df


def process_year_file(
    path: Path,
    counties,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join one FIRMS file to counties and aggregate county-month activity."""
    gpd = require_geopandas()

    df = load_firms_file(path)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    year_months = (
        df[["year", "month"]]
        .drop_duplicates()
        .copy()
    )

    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["longitude"],
            df["latitude"],
        ),
        crs="EPSG:4326",
    )

    county_columns = [
        "county_geoid",
        "county_name",
        "state_fips",
        "state",
        "geometry",
    ]

    joined = gpd.sjoin(
        points,
        counties[county_columns],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        return pd.DataFrame(), year_months

    aggregated = (
        joined.groupby(
            GROUP_COLUMNS,
            as_index=False,
        )
        .agg(
            detection_count=(
                "frp",
                "size",
            ),
            distinct_detection_days=(
                "acq_day",
                "nunique",
            ),
            nominal_or_high_detection_count=(
                "is_nominal_or_high",
                "sum",
            ),
            nominal_or_high_distinct_days=(
                "nominal_or_high_day",
                "nunique",
            ),
            high_confidence_count=(
                "is_high_confidence",
                "sum",
            ),
            nominal_confidence_count=(
                "is_nominal_confidence",
                "sum",
            ),
            low_confidence_count=(
                "is_low_confidence",
                "sum",
            ),
            nighttime_count=(
                "is_nighttime",
                "sum",
            ),
            max_observed_frp=(
                "frp",
                "max",
            ),
            median_observed_frp=(
                "frp",
                "median",
            ),
            sum_observed_frp=(
                "frp",
                "sum",
            ),
            nominal_or_high_sum_observed_frp=(
                "nominal_or_high_frp",
                "sum",
            ),
        )
    )

    return aggregated, year_months


def build_county_month_panel(
    counties,
    year_months: pd.DataFrame,
) -> pd.DataFrame:
    """Create every western county and observed archive month combination."""
    county_table = (
        counties.drop(columns="geometry")
        .drop_duplicates()
        .copy()
    )

    month_table = (
        year_months
        .drop_duplicates()
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

    return county_table.merge(
        month_table,
        how="cross",
    )


def empty_activity_table() -> pd.DataFrame:
    """Return an empty activity table with the expected schema."""
    return pd.DataFrame(
        columns=(
            GROUP_COLUMNS
            + COUNT_COLUMNS
            + FLOAT_COLUMNS
        )
    )


def target_statistics(
    df: pd.DataFrame,
    target_column: str,
) -> dict:
    """Return count and rate statistics for one binary target."""
    rows = int(len(df))

    if rows == 0:
        return {
            "rows": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_rate": None,
        }

    positive_count = int(df[target_column].sum())
    negative_count = rows - positive_count

    return {
        "rows": rows,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": float(positive_count / rows),
    }


def validate_targets(targets: pd.DataFrame) -> None:
    """Run basic integrity checks before writing output."""
    duplicate_count = int(
        targets.duplicated(
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

    if targets["county_geoid"].isna().any():
        raise ValueError("Some rows have missing county GEOIDs.")

    if targets["year"].isna().any():
        raise ValueError("Some rows have missing years.")

    if targets["month"].isna().any():
        raise ValueError("Some rows have missing months.")

    for column in COUNT_COLUMNS:
        if (targets[column] < 0).any():
            raise ValueError(
                f"{column} contains negative values."
            )

    invalid_meaningful = targets[
        (targets["meaningful_fire_activity"] == 1)
        & (targets["any_fire_activity"] == 0)
    ]

    if not invalid_meaningful.empty:
        raise ValueError(
            "Meaningful activity must be a subset of any activity."
        )


def build_targets(
    any_min_detections: int,
    meaningful_min_detections: int,
    meaningful_min_days: int,
    model_start_year: int,
    model_end_year: int,
) -> tuple[pd.DataFrame, dict]:
    """Build broad and stricter county-month FIRMS activity targets."""
    files = sorted(RAW_DIR.glob(RAW_PATTERN))

    if not files:
        raise SystemExit(
            f"No raw FIRMS files found in {RAW_DIR} "
            f"matching {RAW_PATTERN}.\n"
            "Run `python fetch_firms.py --all "
            "--source VIIRS_SNPP_SP` first."
        )

    counties = load_western_counties()

    activity_frames: list[pd.DataFrame] = []
    year_month_frames: list[pd.DataFrame] = []

    for path in files:
        print(f"Processing {path}")

        activity, year_months = process_year_file(
            path,
            counties,
        )

        if not activity.empty:
            activity_frames.append(activity)

        if not year_months.empty:
            year_month_frames.append(year_months)

    if not year_month_frames:
        raise SystemExit(
            "No valid year-month records were found."
        )

    year_months = (
        pd.concat(
            year_month_frames,
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

    panel = build_county_month_panel(
        counties,
        year_months,
    )

    if activity_frames:
        activity = pd.concat(
            activity_frames,
            ignore_index=True,
        )
    else:
        activity = empty_activity_table()

    targets = panel.merge(
        activity,
        on=GROUP_COLUMNS,
        how="left",
    )

    for column in COUNT_COLUMNS:
        targets[column] = (
            targets[column]
            .fillna(0)
            .astype(int)
        )

    for column in FLOAT_COLUMNS:
        targets[column] = (
            targets[column]
            .fillna(0.0)
            .astype(float)
        )

    targets["any_fire_activity"] = (
        targets["detection_count"]
        >= any_min_detections
    ).astype(int)

    targets["meaningful_fire_activity"] = (
        (
            targets["nominal_or_high_detection_count"]
            >= meaningful_min_detections
        )
        & (
            targets["nominal_or_high_distinct_days"]
            >= meaningful_min_days
        )
    ).astype(int)

    targets["month_start"] = pd.to_datetime(
        {
            "year": targets["year"],
            "month": targets["month"],
            "day": 1,
        }
    )

    targets["model_eligible"] = (
        (targets["year"] >= model_start_year)
        & (targets["year"] <= model_end_year)
    ).astype(int)

    targets = targets.sort_values(
        [
            "state",
            "county_geoid",
            "year",
            "month",
        ]
    ).reset_index(drop=True)

    validate_targets(targets)

    modeling_rows = targets[
        targets["model_eligible"] == 1
    ].copy()

    summary = {
        "source": SOURCE,
        "raw_pattern": RAW_PATTERN,
        "raw_files": [
            path.name
            for path in files
        ],
        "rows": int(len(targets)),
        "counties": int(
            targets["county_geoid"].nunique()
        ),
        "year_months": int(len(year_months)),
        "start_year": int(targets["year"].min()),
        "end_year": int(targets["year"].max()),
        "model_start_year": model_start_year,
        "model_end_year": model_end_year,
        "target_definitions": {
            "any_fire_activity": {
                "minimum_total_detections": (
                    any_min_detections
                ),
                "description": (
                    "At least the specified number of "
                    "FIRMS detections in a county-month."
                ),
            },
            "meaningful_fire_activity": {
                "eligible_confidence_codes": [
                    "n",
                    "h",
                ],
                "minimum_eligible_detections": (
                    meaningful_min_detections
                ),
                "minimum_distinct_detection_days": (
                    meaningful_min_days
                ),
                "description": (
                    "At least the specified number of "
                    "nominal-or-high-confidence FIRMS "
                    "detections occurring across the "
                    "specified number of distinct days."
                ),
            },
        },
        "all_period": {
            "any_fire_activity": target_statistics(
                targets,
                "any_fire_activity",
            ),
            "meaningful_fire_activity": target_statistics(
                targets,
                "meaningful_fire_activity",
            ),
        },
        "modeling_period": {
            "any_fire_activity": target_statistics(
                modeling_rows,
                "any_fire_activity",
            ),
            "meaningful_fire_activity": target_statistics(
                modeling_rows,
                "meaningful_fire_activity",
            ),
        },
        "total_detection_count": int(
            targets["detection_count"].sum()
        ),
        "total_nominal_or_high_detection_count": int(
            targets[
                "nominal_or_high_detection_count"
            ].sum()
        ),
        "note": (
            "These targets are based on FIRMS thermal-anomaly "
            "detections. They are not official wildfire ignition, "
            "perimeter, burned-area, or emergency-risk labels."
        ),
    }

    return targets, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build broad and stricter county-month "
            "FIRMS activity targets."
        )
    )

    parser.add_argument(
        "--any-min-detections",
        type=int,
        default=1,
        help=(
            "Minimum total FIRMS detections for "
            "any_fire_activity."
        ),
    )

    parser.add_argument(
        "--meaningful-min-detections",
        type=int,
        default=5,
        help=(
            "Minimum nominal-or-high-confidence detections "
            "for meaningful_fire_activity."
        ),
    )

    parser.add_argument(
        "--meaningful-min-days",
        type=int,
        default=2,
        help=(
            "Minimum distinct detection days for "
            "meaningful_fire_activity."
        ),
    )

    parser.add_argument(
        "--model-start-year",
        type=int,
        default=2013,
        help="First year eligible for model development.",
    )

    parser.add_argument(
        "--model-end-year",
        type=int,
        default=date.today().year - 1,
        help=(
            "Last complete year eligible for model development."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.any_min_detections < 1:
        raise SystemExit(
            "--any-min-detections must be at least 1."
        )

    if args.meaningful_min_detections < 1:
        raise SystemExit(
            "--meaningful-min-detections must be at least 1."
        )

    if args.meaningful_min_days < 1:
        raise SystemExit(
            "--meaningful-min-days must be at least 1."
        )

    if args.model_start_year > args.model_end_year:
        raise SystemExit(
            "--model-start-year cannot be later than "
            "--model-end-year."
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets, summary = build_targets(
        any_min_detections=args.any_min_detections,
        meaningful_min_detections=(
            args.meaningful_min_detections
        ),
        meaningful_min_days=args.meaningful_min_days,
        model_start_year=args.model_start_year,
        model_end_year=args.model_end_year,
    )

    targets_path = (
        OUT_DIR
        / "firms_county_month_targets_v2.parquet"
    )

    summary_path = (
        OUT_DIR
        / "firms_target_v2_summary.json"
    )

    targets.to_parquet(
        targets_path,
        index=False,
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    meaningful_all = summary[
        "all_period"
    ]["meaningful_fire_activity"]

    meaningful_modeling = summary[
        "modeling_period"
    ]["meaningful_fire_activity"]

    print(
        f"Saved {len(targets):,} rows to {targets_path}"
    )
    print(f"Saved summary to {summary_path}")
    print(
        "Meaningful positive rate, all period: "
        f"{meaningful_all['positive_rate']:.4f}"
    )
    print(
        "Meaningful positive rate, modeling period: "
        f"{meaningful_modeling['positive_rate']:.4f}"
    )


if __name__ == "__main__":
    main()