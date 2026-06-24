import argparse
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


RAW_DIR = Path("data/active_fire")
CONTEXT_DIR = Path("data/context")
OUT_DIR = Path("data/labels")

RAW_PATTERN = "firms_viirs_snpp_sp_*.parquet"
SOURCE = "VIIRS_SNPP_SP"

COUNTY_ZIP_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2023/shp/"
    "cb_2023_us_county_500k.zip"
)
COUNTY_ZIP_PATH = CONTEXT_DIR / "cb_2023_us_county_500k.zip"
COUNTY_EXTRACT_DIR = CONTEXT_DIR / "cb_2023_us_county_500k"

WEST_STATE_FIPS = {
    "04": "AZ",
    "06": "CA",
    "08": "CO",
    "16": "ID",
    "30": "MT",
    "32": "NV",
    "35": "NM",
    "41": "OR",
    "49": "UT",
    "53": "WA",
    "56": "WY",
}


def require_geopandas():
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit(
            "This script needs geopandas. Install it first:\n\n"
            "    pip install geopandas pyogrio\n"
        ) from exc

    return gpd


def download_counties() -> Path:
    """Download and extract Census county boundaries if missing."""
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    COUNTY_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    shp_files = list(COUNTY_EXTRACT_DIR.glob("*.shp"))
    if shp_files:
        return shp_files[0]

    if not COUNTY_ZIP_PATH.exists():
        print(f"Downloading county boundaries to {COUNTY_ZIP_PATH}")
        urlretrieve(COUNTY_ZIP_URL, COUNTY_ZIP_PATH)

    print(f"Extracting {COUNTY_ZIP_PATH}")
    with zipfile.ZipFile(COUNTY_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(COUNTY_EXTRACT_DIR)

    shp_files = list(COUNTY_EXTRACT_DIR.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(
            f"No shapefile was found after extracting {COUNTY_ZIP_PATH}"
        )

    return shp_files[0]


def load_western_counties():
    """Load county polygons for the western states used by this dashboard."""
    gpd = require_geopandas()
    shp_path = download_counties()

    counties = gpd.read_file(shp_path)
    counties = counties.to_crs("EPSG:4326")

    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)
    counties = counties[counties["STATEFP"].isin(WEST_STATE_FIPS.keys())].copy()

    counties["county_geoid"] = counties["GEOID"].astype(str)
    counties["county_name"] = counties["NAME"].astype(str)
    counties["state_fips"] = counties["STATEFP"].astype(str)
    counties["state"] = counties["state_fips"].map(WEST_STATE_FIPS)

    keep_columns = [
        "county_geoid",
        "county_name",
        "state_fips",
        "state",
        "geometry",
    ]

    return counties[keep_columns].copy()


def load_firms_file(path: Path) -> pd.DataFrame:
    """Load one yearly FIRMS file and keep fields needed for labels."""
    df = pd.read_parquet(path)

    required_columns = ["acq_date", "latitude", "longitude", "frp"]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    optional_columns = ["confidence", "daynight", "source"]
    keep_columns = required_columns + [
        column for column in optional_columns if column in df.columns
    ]

    df = df[keep_columns].copy()

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)

    if "confidence" not in df.columns:
        df["confidence"] = "unknown"

    if "daynight" not in df.columns:
        df["daynight"] = "unknown"

    if "source" not in df.columns:
        df["source"] = SOURCE

    df["confidence"] = df["confidence"].fillna("unknown").astype(str)
    df["daynight"] = df["daynight"].fillna("unknown").astype(str)
    df["source"] = df["source"].fillna(SOURCE).astype(str)

    df = df.dropna(subset=["acq_date", "latitude", "longitude"])

    df["year"] = df["acq_date"].dt.year.astype(int)
    df["month"] = df["acq_date"].dt.month.astype(int)

    return df


def point_to_county_counts(path: Path, counties) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spatially join one FIRMS year to counties and return county-month counts."""
    gpd = require_geopandas()

    df = load_firms_file(path)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    year_months = df[["year", "month"]].drop_duplicates().copy()

    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    county_join = counties[
        ["county_geoid", "county_name", "state_fips", "state", "geometry"]
    ].copy()

    joined = gpd.sjoin(
        points,
        county_join,
        how="inner",
        predicate="within",
    )

    if joined.empty:
        return pd.DataFrame(), year_months

    counts = (
        joined.groupby(
            [
                "county_geoid",
                "county_name",
                "state_fips",
                "state",
                "year",
                "month",
            ],
            as_index=False,
        )
        .agg(
            detection_count=("frp", "size"),
            max_observed_frp=("frp", "max"),
            median_observed_frp=("frp", "median"),
            sum_observed_frp=("frp", "sum"),
            high_confidence_count=("confidence", lambda x: int((x == "h").sum())),
            nighttime_count=("daynight", lambda x: int((x == "N").sum())),
        )
    )

    return counts, year_months


def build_full_panel(counties: pd.DataFrame, year_months: pd.DataFrame) -> pd.DataFrame:
    """Create every county-month row so zero-detection months are retained."""
    county_table = counties.drop(columns="geometry").drop_duplicates().copy()
    year_month_table = year_months.drop_duplicates().sort_values(["year", "month"])

    panel = county_table.merge(year_month_table, how="cross")

    return panel


def build_labels(min_detections: int) -> tuple[pd.DataFrame, dict]:
    """Build county-month FIRMS occurrence labels."""
    files = sorted(RAW_DIR.glob(RAW_PATTERN))

    if not files:
        raise SystemExit(
            f"No raw FIRMS files found in {RAW_DIR} matching {RAW_PATTERN}.\n"
            "Run `python fetch_firms.py --all --source VIIRS_SNPP_SP` first."
        )

    counties = load_western_counties()

    all_counts = []
    all_year_months = []

    for path in files:
        print(f"Processing {path}")
        counts, year_months = point_to_county_counts(path, counties)

        if not counts.empty:
            all_counts.append(counts)

        if not year_months.empty:
            all_year_months.append(year_months)

    if not all_year_months:
        raise SystemExit("No valid year-month records were found in raw FIRMS files.")

    year_months = (
        pd.concat(all_year_months, ignore_index=True)
        .drop_duplicates()
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

    panel = build_full_panel(counties, year_months)

    if all_counts:
        counts = pd.concat(all_counts, ignore_index=True)
    else:
        counts = pd.DataFrame(
            columns=[
                "county_geoid",
                "year",
                "month",
                "detection_count",
                "max_observed_frp",
                "median_observed_frp",
                "sum_observed_frp",
                "high_confidence_count",
                "nighttime_count",
            ]
        )

    labels = panel.merge(
        counts,
        on=[
            "county_geoid",
            "county_name",
            "state_fips",
            "state",
            "year",
            "month",
        ],
        how="left",
    )

    numeric_columns = [
        "detection_count",
        "max_observed_frp",
        "median_observed_frp",
        "sum_observed_frp",
        "high_confidence_count",
        "nighttime_count",
    ]

    for column in numeric_columns:
        labels[column] = labels[column].fillna(0)

    count_columns = [
        "detection_count",
        "high_confidence_count",
        "nighttime_count",
    ]

    for column in count_columns:
        labels[column] = labels[column].astype(int)

    labels["fire_occurrence"] = (
        labels["detection_count"] >= min_detections
    ).astype(int)

    labels = labels.sort_values(
        ["state", "county_geoid", "year", "month"]
    ).reset_index(drop=True)

    summary = {
        "source": SOURCE,
        "raw_pattern": RAW_PATTERN,
        "western_states": sorted(WEST_STATE_FIPS.values()),
        "min_detections_for_positive_label": min_detections,
        "rows": int(len(labels)),
        "counties": int(labels["county_geoid"].nunique()),
        "year_months": int(year_months.shape[0]),
        "start_year": int(labels["year"].min()),
        "end_year": int(labels["year"].max()),
        "positive_count": int(labels["fire_occurrence"].sum()),
        "negative_count": int((labels["fire_occurrence"] == 0).sum()),
        "positive_rate": float(labels["fire_occurrence"].mean()),
        "total_detection_count": int(labels["detection_count"].sum()),
        "note": (
            "fire_occurrence is a FIRMS-detection-based county-month label. "
            "It is not an official wildfire ignition, fire perimeter, burned-area, "
            "or emergency risk label."
        ),
    }

    return labels, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build county-month fire occurrence labels from FIRMS detections."
    )

    parser.add_argument(
        "--min-detections",
        type=int,
        default=1,
        help="Minimum FIRMS detections required to mark a county-month positive.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_detections < 1:
        raise SystemExit("--min-detections must be at least 1.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels, summary = build_labels(min_detections=args.min_detections)

    labels_path = OUT_DIR / "firms_county_month_labels.parquet"
    summary_path = OUT_DIR / "firms_label_summary.json"

    labels.to_parquet(labels_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved {len(labels):,} rows to {labels_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Positive labels: {summary['positive_count']:,}")
    print(f"Positive rate: {summary['positive_rate']:.4f}")


if __name__ == "__main__":
    main()