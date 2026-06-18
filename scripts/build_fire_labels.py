from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


QUERY_URL = (
    "https://apps.fs.usda.gov/arcx/rest/services/EDW/"
    "EDW_FireOccurrence6thEdition_01/MapServer/29/query"
)

FIRST_YEAR = 1992
LAST_YEAR = 2020
PAGE_SIZE = 2000

WESTERN_STATES = (
    "AZ",
    "CA",
    "CO",
    "ID",
    "MT",
    "NM",
    "NV",
    "OR",
    "UT",
    "WA",
    "WY",
)

STATE_FIPS = {
    "AZ": "04",
    "CA": "06",
    "CO": "08",
    "ID": "16",
    "MT": "30",
    "NM": "35",
    "NV": "32",
    "OR": "41",
    "UT": "49",
    "WA": "53",
    "WY": "56",
}

FIELDS = (
    "objectid",
    "fod_id",
    "discovery_date",
    "fire_size",
    "state",
    "county",
    "fips_code",
    "fips_name",
)

DEFAULT_CACHE_DIR = Path("data/raw/fpa_fod")
DEFAULT_OUTPUT = Path(
    "data/labels/fpa_fod_county_month_labels.parquet"
)
DEFAULT_SUMMARY = Path(
    "data/labels/fpa_fod_label_summary.json"
)


def make_session() -> requests.Session:
    """Create a requests session with retry handling."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {"User-Agent": "fire-hazard-dashboard/1.0"}
    )

    return session


def fetch_state(
    session: requests.Session,
    state: str,
    start_year: int,
    end_year: int,
    cache_dir: Path,
    refresh: bool,
    pause_seconds: float,
) -> pd.DataFrame:
    """Download and cache FPA-FOD records for one state."""
    cache_path = (
        cache_dir
        / f"fpa_fod_{state.lower()}_{start_year}_{end_year}.parquet"
    )

    if cache_path.exists() and not refresh:
        print(f"{state}: loading {cache_path}")
        return pd.read_parquet(cache_path)

    where = (
        f"state = '{state}' AND "
        f"fire_year >= {start_year} "
        f"AND fire_year <= {end_year}"
    )

    pages: list[pd.DataFrame] = []
    offset = 0

    print(f"{state}: fetching {start_year}-{end_year}")

    while True:
        form = {
            "where": where,
            "outFields": ",".join(FIELDS),
            "returnGeometry": "false",
            "orderByFields": "objectid ASC",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "f": "json",
        }

        response = session.post(
            QUERY_URL,
            data=form,
            timeout=180,
        )
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "FPA-FOD returned non-JSON content: "
                f"{response.text[:300]}"
            ) from exc

        if "error" in payload:
            raise RuntimeError(
                f"FPA-FOD service error: {payload['error']}"
            )

        features = payload.get("features", [])

        if not features:
            break

        page = pd.DataFrame(
            [item["attributes"] for item in features]
        )
        page.columns = [
            str(column).lower()
            for column in page.columns
        ]

        pages.append(page)

        offset += len(page)
        print(
            f"  {state}: {offset:,} records",
            flush=True,
        )

        exceeded_limit = payload.get(
            "exceededTransferLimit",
            False,
        )

        if not exceeded_limit and len(page) < PAGE_SIZE:
            break

        time.sleep(max(pause_seconds, 0.0))

    if not pages:
        raise RuntimeError(
            f"No records returned for "
            f"{state}, {start_year}-{end_year}."
        )

    data = pd.concat(
        pages,
        ignore_index=True,
    )

    data = data.drop_duplicates("objectid")

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        cache_path,
        index=False,
    )

    print(
        f"{state}: cached {len(data):,} records"
    )

    return data


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse ArcGIS dates stored as epoch milliseconds."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    from_epoch = pd.to_datetime(
        numeric,
        unit="ms",
        errors="coerce",
        utc=True,
    )

    from_text = pd.to_datetime(
        values.where(numeric.isna()),
        errors="coerce",
        utc=True,
    )

    return (
        from_epoch
        .fillna(from_text)
        .dt.tz_convert(None)
    )


def clean_fips(values: pd.Series) -> pd.Series:
    """Convert county FIPS values to five-character strings."""
    result = (
        values.astype("string")
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.replace(
            r"\D",
            "",
            regex=True,
        )
        .str.zfill(5)
    )

    return result.where(
        result.str.len() == 5
    )


def clean_events(
    raw: pd.DataFrame,
    states: list[str],
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean records used to build county-month labels."""
    data = raw.copy()

    data.columns = [
        str(column).lower()
        for column in data.columns
    ]

    required = {
        "objectid",
        "fod_id",
        "discovery_date",
        "fire_size",
        "state",
        "fips_code",
    }

    missing = sorted(
        required.difference(data.columns)
    )

    if missing:
        raise KeyError(
            "Missing required FPA-FOD fields: "
            f"{missing}"
        )

    data["state"] = (
        data["state"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    data["discovery_date"] = parse_dates(
        data["discovery_date"]
    )

    data["fire_size"] = pd.to_numeric(
        data["fire_size"],
        errors="coerce",
    ).fillna(0.0)

    data["county_fips"] = clean_fips(
        data["fips_code"]
    )

    stats = {
        "downloaded_rows": int(len(data)),
        "missing_date_rows": int(
            data["discovery_date"].isna().sum()
        ),
        "missing_fips_rows": int(
            data["county_fips"].isna().sum()
        ),
    }

    data = data[
        data["state"].isin(states)
        & data["discovery_date"].notna()
        & data["county_fips"].notna()
    ].copy()

    expected_prefix = data["state"].map(
        STATE_FIPS
    )

    mismatch = (
        data["county_fips"].str[:2]
        != expected_prefix
    )

    stats["state_fips_mismatch_rows"] = int(
        mismatch.sum()
    )

    data = data[~mismatch].copy()

    data["year"] = (
        data["discovery_date"]
        .dt.year
        .astype("int16")
    )

    data["month"] = (
        data["discovery_date"]
        .dt.month
        .astype("int8")
    )

    data = data[
        data["year"].between(
            start_year,
            end_year,
        )
    ].copy()

    data["fire_size"] = (
        data["fire_size"]
        .clip(lower=0.0)
    )

    fips_name = (
        data.get(
            "fips_name",
            pd.Series(
                index=data.index,
                dtype="string",
            ),
        )
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    county_name = (
        data.get(
            "county",
            pd.Series(
                index=data.index,
                dtype="string",
            ),
        )
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    data["county_name"] = (
        fips_name
        .fillna(county_name)
        .fillna("Unknown county")
    )

    if data["fod_id"].notna().all():
        dedupe_field = "fod_id"
    else:
        dedupe_field = "objectid"

    data = data.drop_duplicates(
        dedupe_field
    )

    stats["usable_rows"] = int(len(data))

    return data, stats


def most_common(values: pd.Series) -> str:
    """Return the most common nonempty county name."""
    cleaned = (
        values.dropna()
        .astype(str)
        .str.strip()
    )

    cleaned = cleaned[
        cleaned.ne("")
    ]

    if cleaned.empty:
        return "Unknown county"

    return cleaned.value_counts().index[0]


def build_labels(
    events: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Build the complete county-month label table."""
    counties = (
        events.groupby(
            "county_fips",
            as_index=False,
        )
        .agg(
            state=("state", "first"),
            county_name=(
                "county_name",
                most_common,
            ),
        )
    )

    counties["state_fips"] = (
        counties["state"]
        .map(STATE_FIPS)
    )

    observed = (
        events.groupby(
            [
                "county_fips",
                "state",
                "year",
                "month",
            ],
            as_index=False,
        )
        .agg(
            num_fires=(
                "fod_id",
                "nunique",
            ),
            total_fire_size_acres=(
                "fire_size",
                "sum",
            ),
            max_fire_size_acres=(
                "fire_size",
                "max",
            ),
            mean_fire_size_acres=(
                "fire_size",
                "mean",
            ),
        )
    )

    months = pd.DataFrame(
        {
            "month_start": pd.date_range(
                start=f"{start_year}-01-01",
                end=f"{end_year}-12-01",
                freq="MS",
            )
        }
    )

    months["year"] = (
        months["month_start"]
        .dt.year
        .astype("int16")
    )

    months["month"] = (
        months["month_start"]
        .dt.month
        .astype("int8")
    )

    panel = counties.merge(
        months,
        how="cross",
    )

    labels = panel.merge(
        observed,
        on=[
            "county_fips",
            "state",
            "year",
            "month",
        ],
        how="left",
        validate="one_to_one",
    )

    labels["num_fires"] = (
        labels["num_fires"]
        .fillna(0)
        .astype("int32")
    )

    outcome_columns = (
        "total_fire_size_acres",
        "max_fire_size_acres",
        "mean_fire_size_acres",
    )

    for column in outcome_columns:
        labels[column] = (
            labels[column]
            .fillna(0.0)
        )

    labels["fire_occurred"] = (
        labels["num_fires"]
        .gt(0)
        .astype("int8")
    )

    labels["label_source"] = (
        "FPA_FOD_6th_1992_2020"
    )

    columns = [
        "county_fips",
        "county_name",
        "state",
        "state_fips",
        "year",
        "month",
        "month_start",
        "fire_occurred",
        "num_fires",
        "total_fire_size_acres",
        "max_fire_size_acres",
        "mean_fire_size_acres",
        "label_source",
    ]

    return (
        labels[columns]
        .sort_values(
            [
                "state",
                "county_fips",
                "year",
                "month",
            ]
        )
        .reset_index(drop=True)
    )


def write_summary(
    path: Path,
    labels: pd.DataFrame,
    events: pd.DataFrame,
    cleaning: dict[str, int],
    states: list[str],
    start_year: int,
    end_year: int,
) -> None:
    """Write a JSON summary of the generated labels."""
    state_event_counts = {
        str(state): int(count)
        for state, count
        in (
            events.groupby("state")
            .size()
            .sort_index()
            .items()
        )
    }

    summary = {
        "generated_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "source": (
            "USDA Forest Service "
            "FPA-FOD 6th Edition"
        ),
        "target": "fire_occurred",
        "target_definition": (
            "1 if at least one FPA-FOD fire "
            "occurred in a county-month; "
            "otherwise 0"
        ),
        "states": states,
        "start_year": start_year,
        "end_year": end_year,
        "county_count": int(
            labels["county_fips"].nunique()
        ),
        "county_month_rows": int(
            len(labels)
        ),
        "event_rows_used": int(
            len(events)
        ),
        "positive_county_months": int(
            labels["fire_occurred"].sum()
        ),
        "positive_rate": float(
            labels["fire_occurred"].mean()
        ),
        "state_event_counts": (
            state_event_counts
        ),
        "cleaning": cleaning,
        "county_universe_note": (
            "Initial county coverage includes "
            "counties with at least one valid "
            "FPA-FOD county FIPS record in the "
            "selected period. County boundaries "
            "will be used later to check for "
            "missing zero-event counties."
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build county-month wildfire "
            "labels from FPA-FOD."
        )
    )

    parser.add_argument(
        "--states",
        nargs="+",
        default=list(WESTERN_STATES),
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=FIRST_YEAR,
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=LAST_YEAR,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
    )

    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.1,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    states = sorted(
        {
            state.upper()
            for state in args.states
        }
    )

    unsupported = sorted(
        set(states).difference(
            WESTERN_STATES
        )
    )

    if unsupported:
        raise SystemExit(
            f"Unsupported states: {unsupported}. "
            f"Choose from {WESTERN_STATES}."
        )

    valid_year_range = (
        FIRST_YEAR
        <= args.start_year
        <= args.end_year
        <= LAST_YEAR
    )

    if not valid_year_range:
        raise SystemExit(
            f"Use years between "
            f"{FIRST_YEAR} and {LAST_YEAR}."
        )

    session = make_session()

    state_frames = []

    for state in states:
        state_data = fetch_state(
            session=session,
            state=state,
            start_year=args.start_year,
            end_year=args.end_year,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            pause_seconds=args.pause_seconds,
        )

        state_frames.append(
            state_data
        )

    raw = pd.concat(
        state_frames,
        ignore_index=True,
    )

    events, cleaning = clean_events(
        raw=raw,
        states=states,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    if events.empty:
        raise RuntimeError(
            "No usable events remained "
            "after cleaning."
        )

    labels = build_labels(
        events=events,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels.to_parquet(
        args.output,
        index=False,
    )

    write_summary(
        path=args.summary_output,
        labels=labels,
        events=events,
        cleaning=cleaning,
        states=states,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    print()
    print(
        f"Saved {len(labels):,} "
        f"county-month rows to "
        f"{args.output}"
    )

    print(
        f"Saved summary to "
        f"{args.summary_output}"
    )

    print(
        f"Counties: "
        f"{labels['county_fips'].nunique():,}"
    )

    print(
        "Positive rate: "
        f"{labels['fire_occurred'].mean():.3f}"
    )


if __name__ == "__main__":
    main()