"""
Fetch active-fire detections for the western United States from NASA FIRMS.

Examples
--------
Fetch recent near-real-time S-NPP VIIRS detections:

    python fetch_firms.py --recent

Fetch recent NOAA-20 VIIRS detections:

    python fetch_firms.py --recent --source VIIRS_NOAA20_NRT

Fetch one historical year from standard-processed S-NPP VIIRS:

    python fetch_firms.py --year 2020

Fetch one historical year from NOAA-20 standard-processed VIIRS:

    python fetch_firms.py --year 2020 --source VIIRS_NOAA20_SP

Fetch all years using the default historical source:

    python fetch_firms.py --all

Setup
-----
Get a free FIRMS map key:
https://firms.modaps.eosdis.nasa.gov/api/map_key/

Then set the key in the current Anaconda Prompt session:

    set FIRMS_MAP_KEY=your_key_here

Output
------
Recent files are saved as:

    data/active_fire/firms_viirs_snpp_nrt_recent.parquet

Historical files are saved as:

    data/active_fire/firms_viirs_snpp_sp_<year>.parquet
"""

import argparse
import os
import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Western United States bounding box:
# west, south, east, north
# roughly CA, OR, WA, NV, AZ, UT, ID, MT, WY, CO, NM
DEFAULT_BBOX = "-125,31,-102,49"

DEFAULT_OUTPUT_DIR = Path("data/active_fire")

DEFAULT_RECENT_SOURCE = "VIIRS_SNPP_NRT"
DEFAULT_ARCHIVE_SOURCE = "VIIRS_SNPP_SP"

FIRST_YEAR = 2012

# Keep this conservative to avoid request-size and rate-limit problems.
DEFAULT_WINDOW_DAYS = 5

SUPPORTED_SOURCES = {
    "VIIRS_SNPP_NRT",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_NRT",
    "MODIS_NRT",
    "MODIS_SP",
    "LANDSAT_NRT",
    "GOES_NRT",
}

KEEP_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "frp",
    "confidence",
    "daynight",
    "satellite",
    "instrument",
    "version",
    "bright_ti4",
    "bright_ti5",
    "brightness",
    "bright_t31",
    "scan",
    "track",
    "type",
]


def get_map_key() -> str:
    """Read the FIRMS map key from the environment."""
    map_key = os.environ.get("FIRMS_MAP_KEY", "").strip()

    if not map_key:
        raise SystemExit(
            "Set FIRMS_MAP_KEY first. Example:\n\n"
            "    set FIRMS_MAP_KEY=your_key_here\n\n"
            "Then run the fetch command again."
        )

    return map_key


def source_to_slug(source: str) -> str:
    """Convert a FIRMS source name into a lowercase filename component."""
    return source.lower()


def output_path(output_dir: Path, source: str, mode: str, year: int | None = None) -> Path:
    """Create the output path for a recent or historical FIRMS file."""
    source_slug = source_to_slug(source)

    if mode == "recent":
        return output_dir / f"firms_{source_slug}_recent.parquet"

    if year is None:
        raise ValueError("year is required for historical output")

    return output_dir / f"firms_{source_slug}_{year}.parquet"


def fetch_window(
    map_key: str,
    source: str,
    bbox: str,
    day_count: int,
    start: date | None = None,
) -> pd.DataFrame:
    """Fetch one FIRMS window."""
    if start is None:
        url = f"{API_BASE}/{map_key}/{source}/{bbox}/{day_count}"
        label = f"recent {day_count}d"
    else:
        url = f"{API_BASE}/{map_key}/{source}/{bbox}/{day_count}/{start.isoformat()}"
        label = f"{start.isoformat()} +{day_count}d"

    response = requests.get(url, timeout=120)

    if response.status_code != 200:
        print(f"  {label}: HTTP {response.status_code}, skipped")
        print(f"    Response: {response.text.strip()[:300]}")
        return pd.DataFrame()

    text = response.text.strip()

    if not text:
        return pd.DataFrame()

    if text.lower().startswith("invalid"):
        raise SystemExit(f"FIRMS API error: {text[:300]}")

    try:
        frame = pd.read_csv(StringIO(text))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if frame.empty or "latitude" not in frame.columns or "longitude" not in frame.columns:
        return pd.DataFrame()

    return frame


def add_acquisition_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Add an approximate UTC acquisition datetime from acq_date and acq_time."""
    if "acq_date" not in df.columns or "acq_time" not in df.columns:
        return df

    time_text = (
        df["acq_time"]
        .fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )

    date_text = df["acq_date"].dt.strftime("%Y-%m-%d")

    df["acq_datetime_utc"] = pd.to_datetime(
        date_text + " " + time_text.str.slice(0, 2) + ":" + time_text.str.slice(2, 4),
        errors="coerce",
        utc=True,
    )

    return df


def tidy(frames: list[pd.DataFrame], source: str, bbox: str) -> pd.DataFrame:
    """Clean, standardize, and annotate FIRMS detections."""
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    keep = [column for column in KEEP_COLUMNS if column in df.columns]
    df = df[keep].copy()

    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

    if "frp" in df.columns:
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce")

    numeric_columns = [
        "latitude",
        "longitude",
        "bright_ti4",
        "bright_ti5",
        "brightness",
        "bright_t31",
        "scan",
        "track",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "acq_date"])

    df["source"] = source
    df["bbox"] = bbox
    df["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()

    df = add_acquisition_datetime(df)

    sort_columns = [
        column
        for column in ["acq_date", "acq_time", "latitude", "longitude"]
        if column in df.columns
    ]

    if sort_columns:
        df = df.sort_values(sort_columns)

    return df.reset_index(drop=True)


def fetch_recent(source: str, bbox: str, output_dir: Path, window_days: int) -> None:
    """Fetch recent near-real-time detections."""
    map_key = get_map_key()

    frame = fetch_window(
        map_key=map_key,
        source=source,
        bbox=bbox,
        day_count=window_days,
        start=None,
    )

    if frame.empty:
        print("No recent detections returned.")
        return

    df = tidy([frame], source=source, bbox=bbox)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_path(output_dir, source=source, mode="recent")
    df.to_parquet(out_path, index=False)

    print(f"Saved {len(df):,} recent detections to {out_path}")


def fetch_year(
    year: int,
    source: str,
    bbox: str,
    output_dir: Path,
    window_days: int,
    sleep_seconds: float,
) -> None:
    """Fetch one historical year in short FIRMS windows."""
    map_key = get_map_key()

    start_date = date(year, 1, 1)
    year_end = date(year, 12, 31)
    today = date.today()
    end_date = min(year_end, today)

    if start_date > today:
        print(f"{year}: skipped because it is in the future")
        return

    frames = []
    current = start_date

    while current <= end_date:
        days = min(window_days, (end_date - current).days + 1)

        frame = fetch_window(
            map_key=map_key,
            source=source,
            bbox=bbox,
            day_count=days,
            start=current,
        )

        if not frame.empty:
            frames.append(frame)

        print(f"  {current.isoformat()} +{days}d: {len(frame):,} detections")

        current += timedelta(days=days)
        time.sleep(sleep_seconds)

    if not frames:
        print(f"{year}: no detections returned")
        return

    df = tidy(frames, source=source, bbox=bbox)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_path(output_dir, source=source, mode="year", year=year)
    df.to_parquet(out_path, index=False)

    print(f"{year}: saved {len(df):,} detections to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch active-fire detections from NASA FIRMS."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--recent", action="store_true", help="Fetch the recent NRT window.")
    group.add_argument("--year", type=int, help="Fetch one historical year.")
    group.add_argument(
        "--all",
        action="store_true",
        help="Fetch all years from FIRST_YEAR to current year.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "FIRMS source. Defaults to VIIRS_SNPP_NRT for --recent and "
            "VIIRS_SNPP_SP for --year/--all."
        ),
    )

    parser.add_argument(
        "--bbox",
        type=str,
        default=DEFAULT_BBOX,
        help="Bounding box as west,south,east,north.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output Parquet files.",
    )

    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="Number of days per FIRMS request window.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Seconds to wait between historical API requests.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.window_days < 1 or args.window_days > 10:
        raise SystemExit("--window-days must be between 1 and 10.")

    if args.recent:
        source = args.source or DEFAULT_RECENT_SOURCE
    else:
        source = args.source or DEFAULT_ARCHIVE_SOURCE

    source = source.upper()

    if source not in SUPPORTED_SOURCES:
        print(f"Warning: {source} is not in the local supported-source list.")
        print("The request will still be attempted.")

    print(f"FIRMS source: {source}")
    print(f"Bounding box: {args.bbox}")
    print(f"Output directory: {args.output_dir}")

    if args.recent:
        fetch_recent(
            source=source,
            bbox=args.bbox,
            output_dir=args.output_dir,
            window_days=args.window_days,
        )

    elif args.year:
        fetch_year(
            year=args.year,
            source=source,
            bbox=args.bbox,
            output_dir=args.output_dir,
            window_days=args.window_days,
            sleep_seconds=args.sleep_seconds,
        )

    else:
        for year in range(FIRST_YEAR, datetime.now().year + 1):
            print(f"=== {year} ===")
            fetch_year(
                year=year,
                source=source,
                bbox=args.bbox,
                output_dir=args.output_dir,
                window_days=args.window_days,
                sleep_seconds=args.sleep_seconds,
            )


if __name__ == "__main__":
    main()