"""
Fetch VIIRS active-fire detections for the US West from NASA FIRMS.

Two modes:
    python fetch_firms.py --recent          # last 7 days (near-real-time)
    python fetch_firms.py --year 2018       # one historical year
    python fetch_firms.py --all             # full archive 2012-present

Historical data is pulled in 10-day windows (the API maximum per request)
and saved as one Parquet per year. Recent data is saved separately and is
meant to be re-fetched whenever the live page should update.

Setup:
    1. Get a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
    2. set FIRMS_MAP_KEY=your-key        (Windows cmd)
    3. Run one of the commands above from the repo root.

Output:
    data/fires/viirs_west_<year>.parquet
    data/fires/viirs_west_recent.parquet
"""

import argparse
import os
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")

# US West bounding box: west, south, east, north
# (CA, OR, WA, NV, AZ, UT, ID, MT, WY, CO, NM)
BBOX = "-125,31,-102,49"

# VIIRS S-NPP: standard processing for the archive, NRT for recent days.
SOURCE_ARCHIVE = "VIIRS_SNPP_SP"
SOURCE_RECENT = "VIIRS_SNPP_NRT"

FIRST_YEAR = 2012  # VIIRS S-NPP record begins 20 January 2012

OUTPUT_DIR = Path("data/fires")

WINDOW_DAYS = 5  # API archive maximum day range per request for this key

KEEP_COLUMNS = [
    "latitude", "longitude", "acq_date", "acq_time",
    "frp", "confidence", "daynight", "satellite",
]


def fetch_window(source, day_count, start=None):
    """Recent mode omits the date; historical mode passes a start date."""
    if start is None:
        url = f"{API_BASE}/{MAP_KEY}/{source}/{BBOX}/{day_count}"
        tag = f"recent {day_count}d"
    else:
        url = (
            f"{API_BASE}/{MAP_KEY}/{source}/{BBOX}/"
            f"{day_count}/{start.isoformat()}"
        )
        tag = f"{start} +{day_count}d"

    response = requests.get(url, timeout=120)

    if response.status_code != 200:
        print(f"  {tag}: HTTP {response.status_code}, skipped")
        print(f"    URL: {url}")
        print(f"    Response: {response.text.strip()[:300]}")
        return pd.DataFrame()

    text = response.text

    if text.startswith("Invalid"):
        raise SystemExit(f"FIRMS API error: {text.strip()[:200]}")

    frame = pd.read_csv(StringIO(text))

    if frame.empty or "latitude" not in frame.columns:
        return pd.DataFrame()

    return frame


def tidy(frames):
    df = pd.concat(frames, ignore_index=True)

    keep = [column for column in KEEP_COLUMNS if column in df.columns]
    df = df[keep]

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])

    return df.sort_values(["acq_date", "acq_time"]).reset_index(drop=True)


def fetch_year(year):
    start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    today = date.today()
    end = min(year_end, today)

    frames = []
    current = start

    while current <= end:
        days = min(WINDOW_DAYS, (end - current).days + 1)
        frame = fetch_window(SOURCE_ARCHIVE, days, start=current)

        if not frame.empty:
            frames.append(frame)

        print(f"  {current} +{days}d: {len(frame):,} detections")
        current += timedelta(days=days)
        time.sleep(1)  # stay well inside the rate limit

    if not frames:
        print(f"{year}: no detections returned")
        return

    df = tidy(frames)
    out_path = OUTPUT_DIR / f"viirs_west_{year}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"{year}: saved {len(df):,} detections to {out_path}")


def fetch_recent():
    frame = fetch_window(SOURCE_RECENT, WINDOW_DAYS)

    if frame.empty:
        print("No recent detections returned.")
        return

    df = tidy([frame])
    out_path = OUTPUT_DIR / "viirs_west_recent.parquet"
    df.to_parquet(out_path, index=False)

    print(
        f"Saved {len(df):,} detections from the last {WINDOW_DAYS} days "
        f"to {out_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--recent", action="store_true")
    group.add_argument("--year", type=int)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not MAP_KEY:
        raise SystemExit(
            "Set FIRMS_MAP_KEY first. Get a free key at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.recent:
        fetch_recent()
    elif args.year:
        fetch_year(args.year)
    else:
        for year in range(FIRST_YEAR, datetime.now().year + 1):
            print(f"=== {year} ===")
            fetch_year(year)


if __name__ == "__main__":
    main()