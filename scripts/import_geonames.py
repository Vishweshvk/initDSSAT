"""Prepare a searchable India places CSV from the official GeoNames extracts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "external" / "geonames"
GEONAMES_COLUMNS = (
    "geoname_id", "name", "ascii_name", "alternate_names", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code", "admin2_code",
    "admin3_code", "admin4_code", "population", "elevation_m", "dem_m", "timezone",
    "modified_date",
)
OUTPUT_COLUMNS = (
    "geoname_id", "name", "ascii_name", "alternate_names", "latitude", "longitude",
    "feature_code", "admin1_code", "state_name", "admin2_code", "population", "timezone",
    "modified_date", "source",
)


def read_admin1(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) >= 2:
                names[row[0].strip()] = row[1].strip()
    if not names:
        raise ValueError(f"No admin1 state names found in {path}")
    return names


def convert(places_zip: Path, admin1_path: Path, output_path: Path) -> tuple[int, int]:
    state_names = read_admin1(admin1_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with zipfile.ZipFile(places_zip) as archive:
        member = next((name for name in archive.namelist()
                      if Path(name).name.casefold() == "in.txt"), None)
        if member is None:
            raise ValueError(f"{places_zip} does not contain IN.txt")
        with archive.open(member) as source, output_path.open("w", encoding="utf-8", newline="") as target:
            reader = csv.reader((line.decode("utf-8") for line in source), delimiter="\t")
            writer = csv.DictWriter(target, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for row in reader:
                if len(row) < len(GEONAMES_COLUMNS):
                    skipped += 1
                    continue
                record = dict(zip(GEONAMES_COLUMNS, row))
                # P = populated places (cities, towns, villages, and similar).
                if record["country_code"] != "IN" or record["feature_class"] != "P":
                    skipped += 1
                    continue
                try:
                    latitude = float(record["latitude"])
                    longitude = float(record["longitude"])
                    geoname_id = int(record["geoname_id"])
                    population = int(record["population"] or 0)
                except ValueError:
                    skipped += 1
                    continue
                admin1_code = record["admin1_code"].strip()
                writer.writerow({
                    "geoname_id": geoname_id,
                    "name": record["name"],
                    "ascii_name": record["ascii_name"],
                    "alternate_names": record["alternate_names"],
                    "latitude": latitude,
                    "longitude": longitude,
                    "feature_code": record["feature_code"],
                    "admin1_code": admin1_code,
                    "state_name": state_names.get(f"IN.{admin1_code}", ""),
                    "admin2_code": record["admin2_code"],
                    "population": population,
                    "timezone": record["timezone"],
                    "modified_date": record["modified_date"],
                    "source": "GeoNames (CC BY 4.0)",
                })
                written += 1
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GeoNames India country extract (IN.zip), CC BY 4.0",
        "places_archive": places_zip.name,
        "admin1_file": admin1_path.name,
        "filter": "country_code=IN and feature_class=P (populated places)",
        "rows_written": written,
        "rows_skipped": skipped,
        "output": str(output_path.relative_to(PROJECT_ROOT)),
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--places-zip", type=Path, default=DATA_DIR / "downloads" / "IN.zip")
    parser.add_argument("--admin1", type=Path, default=DATA_DIR / "downloads" / "admin1CodesASCII.txt")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "processed" / "india_places.csv")
    args = parser.parse_args()
    for label, path in (("India archive", args.places_zip), ("state-code table", args.admin1)):
        if not path.is_file():
            parser.error(f"{label} not found: {path}")
    try:
        written, skipped = convert(args.places_zip, args.admin1, args.output.resolve())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(f"Prepared {written:,} populated places; skipped {skipped:,} non-place or invalid rows.")
    print(f"CSV: {args.output.resolve()}")
    print(f"Metadata: {args.output.resolve().with_suffix('.metadata.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
