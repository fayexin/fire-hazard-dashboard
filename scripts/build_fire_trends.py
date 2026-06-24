from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/active_fire")
OUT_DIR = Path("data/derived")

SOURCE = "VIIRS_SNPP_SP"
RAW_PATTERN = "firms_viirs_snpp_sp_*.parquet"


def load_year_file(path: Path) -> pd.DataFrame:
    """Load one yearly FIRMS file and keep only columns needed for trends."""
    df = pd.read_parquet(path)

    required = ["acq_date", "latitude", "longitude", "frp"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[["acq_date", "latitude", "longitude", "frp", "confidence", "daynight", "source"]].copy()

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["acq_date", "latitude", "longitude"])

    df["year"] = df["acq_date"].dt.year
    df["month"] = df["acq_date"].dt.month
    df["date"] = df["acq_date"].dt.date

    if "confidence" not in df.columns:
        df["confidence"] = "unknown"

    if "daynight" not in df.columns:
        df["daynight"] = "unknown"

    if "source" not in df.columns:
        df["source"] = SOURCE

    df["confidence"] = df["confidence"].fillna("unknown").astype(str)
    df["daynight"] = df["daynight"].fillna("unknown").astype(str)
    df["source"] = df["source"].fillna(SOURCE).astype(str)

    return df


def build_summaries(all_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build annual, monthly, and year-month FIRMS trend summaries."""

    annual = (
        all_data.groupby("year", as_index=False)
        .agg(
            detections=("frp", "size"),
            max_observed_frp=("frp", "max"),
            median_observed_frp=("frp", "median"),
            sum_observed_frp=("frp", "sum"),
            high_confidence_detections=("confidence", lambda x: (x == "h").sum()),
            nighttime_detections=("daynight", lambda x: (x == "N").sum()),
        )
        .sort_values("year")
    )

    monthly_climatology = (
        all_data.groupby("month", as_index=False)
        .agg(
            detections=("frp", "size"),
            max_observed_frp=("frp", "max"),
            median_observed_frp=("frp", "median"),
            sum_observed_frp=("frp", "sum"),
            high_confidence_detections=("confidence", lambda x: (x == "h").sum()),
            nighttime_detections=("daynight", lambda x: (x == "N").sum()),
        )
        .sort_values("month")
    )

    year_month = (
        all_data.groupby(["year", "month"], as_index=False)
        .agg(
            detections=("frp", "size"),
            max_observed_frp=("frp", "max"),
            median_observed_frp=("frp", "median"),
            sum_observed_frp=("frp", "sum"),
            high_confidence_detections=("confidence", lambda x: (x == "h").sum()),
            nighttime_detections=("daynight", lambda x: (x == "N").sum()),
        )
        .sort_values(["year", "month"])
    )

    daily = (
        all_data.groupby(["year", "date"], as_index=False)
        .agg(
            detections=("frp", "size"),
            max_observed_frp=("frp", "max"),
            sum_observed_frp=("frp", "sum"),
        )
        .sort_values(["year", "date"])
    )

    return {
        "fire_annual_summary.parquet": annual,
        "fire_monthly_climatology.parquet": monthly_climatology,
        "fire_year_month_summary.parquet": year_month,
        "fire_daily_summary.parquet": daily,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob(RAW_PATTERN))

    if not files:
        raise SystemExit(
            f"No raw FIRMS files found in {RAW_DIR} matching {RAW_PATTERN}."
        )

    frames = []

    for path in files:
        print(f"Reading {path}")
        frame = load_year_file(path)
        frames.append(frame)

    all_data = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(all_data):,} FIRMS detections")
    print(f"Years: {all_data['year'].min()} to {all_data['year'].max()}")

    summaries = build_summaries(all_data)

    for filename, summary in summaries.items():
        out_path = OUT_DIR / filename
        summary.to_parquet(out_path, index=False)
        print(f"Saved {len(summary):,} rows to {out_path}")

    metadata = {
        "source": SOURCE,
        "raw_pattern": RAW_PATTERN,
        "raw_files": [path.name for path in files],
        "start_year": int(all_data["year"].min()),
        "end_year": int(all_data["year"].max()),
        "total_detections": int(len(all_data)),
        "outputs": list(summaries.keys()),
        "note": (
            "Summaries are based on FIRMS active-fire detections. "
            "Detection counts and observed FRP are not official burned area, "
            "fire perimeter, or fire size estimates."
        ),
    }

    pd.Series(metadata).to_json(
        OUT_DIR / "fire_trend_summary_metadata.json",
        indent=2,
    )

    print(f"Saved metadata to {OUT_DIR / 'fire_trend_summary_metadata.json'}")


if __name__ == "__main__":
    main()