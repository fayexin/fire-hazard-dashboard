import json
from pathlib import Path


try:
    from build_fire_labels import load_western_counties
except ImportError:
    from scripts.build_fire_labels import load_western_counties


OUTPUT_PATH = Path(
    "data/context/western_counties_simplified.geojson"
)

SIMPLIFY_TOLERANCE_METERS = 1500


def main() -> None:
    counties = load_western_counties().copy()

    counties["county_geoid"] = (
        counties["county_geoid"]
        .astype(str)
        .str.zfill(5)
    )

    projected = counties.to_crs("EPSG:5070")

    projected["geometry"] = projected.geometry.simplify(
        tolerance=SIMPLIFY_TOLERANCE_METERS,
        preserve_topology=True,
    )

    output = projected.to_crs("EPSG:4326")[
        [
            "county_geoid",
            "county_name",
            "state",
            "geometry",
        ]
    ].copy()

    geojson = json.loads(output.to_json())

    for feature in geojson["features"]:
        feature["id"] = feature["properties"]["county_geoid"]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            geojson,
            file,
            separators=(",", ":"),
        )

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)

    print(
        f"Saved {len(output):,} counties to {OUTPUT_PATH} "
        f"({size_mb:.2f} MB)"
    )


if __name__ == "__main__":
    main()
