"""Offline state and populated-place lookup backed by GeoNames India data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3


def _default_paths(project_data_dir: Path) -> tuple[Path, Path, Path]:
    root = Path(project_data_dir).parent / "data" / "external" / "geonames"
    return (root / "processed" / "india_places.csv",
            root / "processed" / "india_places.sqlite3",
            root / "processed" / "india_places.metadata.json")


def _ensure_index(project_data_dir: Path) -> Path:
    source, database, _ = _default_paths(project_data_dir)
    if not source.is_file():
        raise FileNotFoundError(f"GeoNames place data is missing: {source}")
    if database.is_file() and database.stat().st_mtime >= source.stat().st_mtime:
        return database
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    try:
        connection.execute("""CREATE TABLE places (
            geoname_id INTEGER PRIMARY KEY, name TEXT NOT NULL, ascii_name TEXT,
            alternate_names TEXT, latitude REAL NOT NULL, longitude REAL NOT NULL,
            feature_code TEXT, admin1_code TEXT, state_name TEXT NOT NULL,
            admin2_code TEXT, population INTEGER, timezone TEXT, modified_date TEXT,
            source TEXT NOT NULL)""")
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            batch = []
            for row in reader:
                if not row.get("state_name"):
                    continue
                batch.append((int(row["geoname_id"]), row["name"], row["ascii_name"],
                              row["alternate_names"], float(row["latitude"]), float(row["longitude"]),
                              row["feature_code"], row["admin1_code"], row["state_name"],
                              row["admin2_code"], int(row["population"] or 0), row["timezone"],
                              row["modified_date"], row["source"]))
                if len(batch) >= 5000:
                    connection.executemany("INSERT INTO places VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO places VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        connection.execute("CREATE INDEX places_state_name ON places(state_name, name COLLATE NOCASE)")
        connection.execute("CREATE INDEX places_state_ascii ON places(state_name, ascii_name COLLATE NOCASE)")
        connection.commit()
    finally:
        connection.close()
    return database


def list_states(project_data_dir: Path) -> list[str]:
    database = _ensure_index(project_data_dir)
    with sqlite3.connect(database) as connection:
        return [row[0] for row in connection.execute(
            "SELECT DISTINCT state_name FROM places WHERE state_name != '' ORDER BY state_name COLLATE NOCASE")]


def search_places(project_data_dir: Path, state: str, query: str, limit: int = 20) -> list[dict]:
    database = _ensure_index(project_data_dir)
    query = query.strip()
    if not state or len(query) < 2:
        return []
    pattern = f"%{query}%"
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("""SELECT geoname_id,name,ascii_name,latitude,longitude,
            admin1_code,admin2_code,population,modified_date,source
            FROM places WHERE state_name=? AND
            (name LIKE ? COLLATE NOCASE OR ascii_name LIKE ? COLLATE NOCASE
             OR alternate_names LIKE ? COLLATE NOCASE)
            ORDER BY CASE WHEN name LIKE ? COLLATE NOCASE THEN 0
                          WHEN ascii_name LIKE ? COLLATE NOCASE THEN 1 ELSE 2 END,
                     population DESC, name COLLATE NOCASE LIMIT ?""",
            (state, pattern, pattern, pattern, f"{query}%", f"{query}%", limit)).fetchall()
    return [dict(row) for row in rows]


def get_place(project_data_dir: Path, state: str, geoname_id: int) -> dict | None:
    database = _ensure_index(project_data_dir)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("""SELECT geoname_id,name,ascii_name,latitude,longitude,
            admin1_code,state_name,admin2_code,population,modified_date,source
            FROM places WHERE state_name=? AND geoname_id=?""", (state, geoname_id)).fetchone()
    return dict(row) if row else None


def location_provenance(place: dict, experiment, weather_path: Path,
                        weather_start: str | None, weather_end: str | None,
                        soil_depth_cm: float | None, coordinates_used: bool = False,
                        nearest_distance_km: float | None = None,
                        weather_metadata: dict | None = None,
                        soil_metadata: dict | None = None,
                        soil_match: dict | None = None,
                        soil_file: Path | None = None,
                        soil_id: str | None = None) -> dict:
    """Describe user-place coordinates and the DSSAT inputs used."""
    weather_metadata = weather_metadata or {}
    soil_metadata = soil_metadata or {}
    soil_match = soil_match or {}
    soil_file = soil_file or experiment.soil_file
    return {
        "selected_location": {
            "geoname_id": int(place["geoname_id"]), "name": place["name"],
            "state": place["state_name"], "latitude": float(place["latitude"]),
            "longitude": float(place["longitude"]), "source": "GeoNames India extract",
            "license": "CC BY 4.0", "source_modified_date": place.get("modified_date"),
            "resolution": "populated-place settlement point; not an exact farm coordinate",
            "nearest_installed_experiment_distance_km": nearest_distance_km,
            "coordinate_match_radius_km": 100,
        },
        "weather_used": {
            "source": weather_metadata.get("source", "DSSAT installed weather file"),
            "file": ", ".join(weather_metadata.get("weather_files", [])) or weather_path.name,
            "station": experiment.weather_station,
            "resolution": weather_metadata.get("resolution", "weather-station record"),
            "date_start": weather_start, "date_end": weather_end,
            "coordinates_used_to_select_experiment": coordinates_used,
            "validated": weather_metadata.get("validated", True),
            "crop_window_coverage": weather_metadata.get("crop_window_coverage"),
            "fallback_reason": weather_metadata.get("fallback_reason"),
            "latitude": weather_metadata.get("latitude"),
            "longitude": weather_metadata.get("longitude"),
            "source_metadata": weather_metadata,
        },
        "soil_used": {
            "source": soil_metadata.get("source", "DSSAT experiment's installed soil profile"),
            "file": soil_file.name,
            "profile_id": soil_id or experiment.soil_id, "resolution": "georeferenced profile layers",
            "profile_depth_cm": soil_depth_cm,
            "coordinates_used_to_select_profile": soil_metadata.get("status") == "applied",
            "profile_latitude": soil_metadata.get("latitude"),
            "profile_longitude": soil_metadata.get("longitude"),
            "profile_distance_km": soil_metadata.get("distance_km"),
            "matching_limit_km": soil_metadata.get("matching_limit_km", soil_match.get("max_distance_km")),
            "match_status": soil_match.get("status", "not_checked"),
            "fallback_reason": soil_match.get("reason"),
            "nearest_candidate_profile_id": soil_match.get("nearest_profile_id"),
            "nearest_candidate_distance_km": soil_match.get("nearest_distance_km"),
            "source_metadata": soil_match,
        },
    }


def metadata(project_data_dir: Path) -> dict:
    _, _, path = _default_paths(project_data_dir)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"source": "GeoNames India extract", "license": "CC BY 4.0"}
