import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


try:
    from build_fire_labels import load_western_counties, require_geopandas
except ImportError:
    from scripts.build_fire_labels import load_western_counties, require_geopandas


BASE_URL = "https://www.northwestknowledge.net/metdata/data"

FIRE_FEATURE_PATH = Path(
    "data/features/county_month_fire_features_v1.parquet"
)
WEATHER_DIR = Path("data/weather")
RAW_DIR = WEATHER_DIR / "gridmet_raw"
PARTS_DIR = WEATHER_DIR / "gridmet_monthly_parts"
OUT_DIR = Path("data/features")

WEATHER_PATH = OUT_DIR / "county_month_gridmet_features_v1.parquet"
MODEL_TABLE_PATH = OUT_DIR / "county_month_model_table_v1.parquet"
SUMMARY_PATH = OUT_DIR / "county_month_model_table_v1_summary.json"

# Compact first version: heat, moisture input, atmospheric dryness,
# and an integrated fire-danger index.
VARIABLES = {
    "tmmx": ["mean", "max"],
    "pr": ["sum"],
    "vpd": ["mean", "max"],
    "erc": ["mean", "max"],
}


def http_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download(url: str, path: Path, session: requests.Session) -> None:
    """Download one annual gridMET NetCDF file atomically."""
    if path.exists() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)

    print(f"Downloading {url}")
    with session.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with open(temporary, "wb") as file:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    file.write(chunk)

    if temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Empty download: {url}")

    os.replace(temporary, path)


def county_sample_points() -> pd.DataFrame:
    """Use one point guaranteed to lie inside each county polygon."""
    gpd = require_geopandas()
    counties = load_western_counties().reset_index(drop=True)

    projected = counties.to_crs("EPSG:5070")
    points_projected = projected.geometry.representative_point()
    points_geo = gpd.GeoSeries(
        points_projected,
        crs="EPSG:5070",
    ).to_crs("EPSG:4326")

    result = counties[
        ["county_geoid", "county_name", "state"]
    ].copy()
    result["county_geoid"] = (
        result["county_geoid"].astype(str).str.zfill(5)
    )
    result["longitude"] = points_geo.x.to_numpy()
    result["latitude"] = points_geo.y.to_numpy()
    return result


def coordinate_name(dataset: xr.Dataset, choices: list[str]) -> str:
    for name in choices:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise ValueError(
        f"None of {choices} found. Coordinates: {list(dataset.coords)}"
    )


def main_data_array(
    dataset: xr.Dataset,
    latitude_name: str,
    longitude_name: str,
) -> tuple[xr.DataArray, str]:
    """Find the primary daily raster variable and its time dimension."""
    for _, data_array in dataset.data_vars.items():
        if (
            latitude_name in data_array.dims
            and longitude_name in data_array.dims
        ):
            time_dimensions = [
                dim
                for dim in data_array.dims
                if dim not in {latitude_name, longitude_name}
            ]
            if len(time_dimensions) == 1:
                return data_array, time_dimensions[0]

    raise ValueError(
        f"No daily grid variable found. Variables: {list(dataset.data_vars)}"
    )


def aggregate_monthly(
    daily: pd.DataFrame,
    statistic: str,
) -> pd.DataFrame:
    if statistic == "mean":
        return daily.resample("MS").mean()
    if statistic == "max":
        return daily.resample("MS").max()
    if statistic == "sum":
        return daily.resample("MS").sum(min_count=1)
    raise ValueError(f"Unsupported statistic: {statistic}")


def extract_part(
    netcdf_path: Path,
    variable: str,
    points: pd.DataFrame,
) -> pd.DataFrame:
    """Extract one nearest grid cell per county and aggregate daily values."""
    with xr.open_dataset(
        netcdf_path,
        engine="netcdf4",
        decode_cf=True,
        mask_and_scale=True,
    ) as dataset:
        lat_name = coordinate_name(dataset, ["lat", "latitude", "y"])
        lon_name = coordinate_name(dataset, ["lon", "longitude", "x"])
        data_array, time_name = main_data_array(
            dataset,
            lat_name,
            lon_name,
        )

        county_ids = points["county_geoid"].to_numpy()
        longitudes = points["longitude"].to_numpy(dtype=float)

        source_longitudes = np.asarray(dataset[lon_name].values)
        if np.nanmin(source_longitudes) >= 0 and np.nanmin(longitudes) < 0:
            longitudes = longitudes % 360

        lat_indexer = xr.DataArray(
            points["latitude"].to_numpy(dtype=float),
            dims="county",
            coords={"county": county_ids},
        )
        lon_indexer = xr.DataArray(
            longitudes,
            dims="county",
            coords={"county": county_ids},
        )

        selected = data_array.sel(
            {lat_name: lat_indexer, lon_name: lon_indexer},
            method="nearest",
        ).transpose(time_name, "county")

        values = np.asarray(selected.load().values, dtype=float)

        # gridMET tmmx is normally Kelvin.
        if variable == "tmmx":
            finite = values[np.isfinite(values)]
            if finite.size and np.nanmedian(finite) > 150:
                values = values - 273.15

        dates = pd.to_datetime(selected[time_name].values)
        daily = pd.DataFrame(values, index=dates, columns=county_ids)
        daily.index.name = "date"

        output = None
        for statistic in VARIABLES[variable]:
            monthly = aggregate_monthly(daily, statistic)
            column = f"gridmet_{variable}_{statistic}"
            wide_frame = (
                monthly.rename_axis(
                    index="observed_month_start",
                    columns="county_geoid",
                )
                .reset_index()
            )
            
            long_frame = wide_frame.melt(
                id_vars="observed_month_start",
                var_name="county_geoid",
                value_name=column,
            )
            if output is None:
                output = long_frame
            else:
                output = output.merge(
                    long_frame,
                    on=["county_geoid", "observed_month_start"],
                    how="outer",
                    validate="one_to_one",
                )

    if output is None:
        raise RuntimeError(f"No output produced for {variable}")

    output["county_geoid"] = (
        output["county_geoid"].astype(str).str.zfill(5)
    )
    return output


def process_part(
    variable: str,
    year: int,
    points: pd.DataFrame,
    session: requests.Session,
    delete_raw: bool,
    force: bool,
) -> pd.DataFrame:
    """Download and cache one variable-year monthly table."""
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    part_path = PARTS_DIR / f"{variable}_{year}_monthly.parquet"

    if part_path.exists() and not force:
        print(f"Using cached {part_path}")
        return pd.read_parquet(part_path)

    raw_path = RAW_DIR / f"{variable}_{year}.nc"
    url = f"{BASE_URL}/{variable}_{year}.nc"

    download(url, raw_path, session)
    frame = extract_part(raw_path, variable, points)
    frame.to_parquet(part_path, index=False)

    if delete_raw:
        raw_path.unlink(missing_ok=True)

    print(f"Saved {part_path}")
    return frame


def weather_observations(
    parts: dict[str, list[pd.DataFrame]],
) -> pd.DataFrame:
    """Concatenate years within each variable and merge all variables."""
    variable_frames = []

    for variable, frames in parts.items():
        frame = pd.concat(frames, ignore_index=True)
        duplicates = frame.duplicated(
            ["county_geoid", "observed_month_start"]
        ).sum()
        if duplicates:
            raise ValueError(
                f"{variable} has {duplicates:,} duplicate county-month rows"
            )
        variable_frames.append(frame)

    combined = variable_frames[0]
    for frame in variable_frames[1:]:
        combined = combined.merge(
            frame,
            on=["county_geoid", "observed_month_start"],
            how="outer",
            validate="one_to_one",
        )

    combined["observed_month_start"] = pd.to_datetime(
        combined["observed_month_start"]
    )
    return combined.sort_values(
        ["county_geoid", "observed_month_start"]
    ).reset_index(drop=True)


def prediction_month_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Assign month t-1 and months t-3:t-1 weather to prediction month t."""
    output = weather.copy()
    grouped = output.groupby("county_geoid", sort=False)

    output["gridmet_pr_prev3m_sum"] = grouped[
        "gridmet_pr_sum"
    ].transform(lambda s: s.rolling(3, min_periods=3).sum())

    for column in [
        "gridmet_tmmx_mean",
        "gridmet_vpd_mean",
        "gridmet_erc_mean",
    ]:
        output[f"{column}_prev3m"] = grouped[column].transform(
            lambda s: s.rolling(3, min_periods=3).mean()
        )

    output["month_start"] = (
        output["observed_month_start"] + pd.offsets.MonthBegin(1)
    )

    output = output.rename(
        columns={
            "gridmet_tmmx_mean": "gridmet_tmax_prev1m_mean_c",
            "gridmet_tmmx_max": "gridmet_tmax_prev1m_max_c",
            "gridmet_tmmx_mean_prev3m": "gridmet_tmax_prev3m_mean_c",
            "gridmet_pr_sum": "gridmet_precip_prev1m_sum_mm",
            "gridmet_pr_prev3m_sum": "gridmet_precip_prev3m_sum_mm",
            "gridmet_vpd_mean": "gridmet_vpd_prev1m_mean_kpa",
            "gridmet_vpd_max": "gridmet_vpd_prev1m_max_kpa",
            "gridmet_vpd_mean_prev3m": "gridmet_vpd_prev3m_mean_kpa",
            "gridmet_erc_mean": "gridmet_erc_prev1m_mean",
            "gridmet_erc_max": "gridmet_erc_prev1m_max",
            "gridmet_erc_mean_prev3m": "gridmet_erc_prev3m_mean",
        }
    )

    feature_columns = [
        column for column in output.columns if column.startswith("gridmet_")
    ]

    return output[
        ["county_geoid", "month_start"] + feature_columns
    ].copy()


def merge_model_table(
    weather_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if not FIRE_FEATURE_PATH.exists():
        raise SystemExit(
            f"Missing {FIRE_FEATURE_PATH}. "
            "Run scripts/build_fire_features_v1.py first."
        )

    fire_features = pd.read_parquet(FIRE_FEATURE_PATH)
    fire_features["county_geoid"] = (
        fire_features["county_geoid"].astype(str).str.zfill(5)
    )
    fire_features["month_start"] = pd.to_datetime(
        fire_features["month_start"]
    )

    weather_columns = [
        column
        for column in weather_features.columns
        if column.startswith("gridmet_")
    ]

    model_table = fire_features.merge(
        weather_features,
        on=["county_geoid", "month_start"],
        how="left",
        validate="one_to_one",
    )

    model_table["weather_complete"] = (
        model_table[weather_columns].notna().all(axis=1).astype(int)
    )

    return model_table, weather_columns


def build_summary(
    model_table: pd.DataFrame,
    weather_columns: list[str],
    start_year: int,
    end_year: int,
) -> dict:
    split_summary = {}

    for split_name in ["train", "validation", "test", "excluded"]:
        subset = model_table[model_table["split"] == split_name]
        split_summary[split_name] = {
            "rows": int(len(subset)),
            "weather_complete_rate": (
                float(subset["weather_complete"].mean())
                if len(subset)
                else None
            ),
            "missing_rates": {
                column: float(subset[column].isna().mean())
                for column in weather_columns
            },
        }

    return {
        "source": "gridMET",
        "source_url": BASE_URL,
        "sampling_method": (
            "Nearest gridMET cell to a representative point inside each county"
        ),
        "observation_years": [start_year, end_year],
        "prediction_timing": (
            "Prediction month t uses weather from t-1 and rolling weather "
            "from t-3 through t-1."
        ),
        "weather_features": weather_columns,
        "split_summary": split_summary,
        "limitation": (
            "Representative-point weather is a low-compute approximation; "
            "it is not a county-wide area average."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete each NetCDF after its monthly cache is written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing monthly cache files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_year > args.end_year:
        raise SystemExit("start year must not exceed end year")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    points = county_sample_points()
    session = http_session()

    parts = {variable: [] for variable in VARIABLES}
    for variable in VARIABLES:
        for year in range(args.start_year, args.end_year + 1):
            parts[variable].append(
                process_part(
                    variable,
                    year,
                    points,
                    session,
                    delete_raw=args.delete_raw,
                    force=args.force,
                )
            )

    observed = weather_observations(parts)
    weather_features = prediction_month_features(observed)
    model_table, weather_columns = merge_model_table(weather_features)

    weather_features.to_parquet(WEATHER_PATH, index=False)
    model_table.to_parquet(MODEL_TABLE_PATH, index=False)

    result_summary = build_summary(
        model_table,
        weather_columns,
        args.start_year,
        args.end_year,
    )
    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(result_summary, file, indent=2)

    print(f"Saved {WEATHER_PATH}")
    print(f"Saved {MODEL_TABLE_PATH}")
    for split_name, values in result_summary["split_summary"].items():
        print(
            f"{split_name}: weather complete rate = "
            f"{values['weather_complete_rate']}"
        )


if __name__ == "__main__":
    main()
