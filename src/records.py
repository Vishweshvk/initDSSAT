"""Append auditable DSSAT and residual-model results to a flat CSV file."""

import csv
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil


FIELDS = [
    "timestamp_utc", "run_id", "crop", "experiment_id", "treatment_id",
    "state", "city", "season", "soil_type", "planting_date", "dssat_planting_date", "experiment_location", "cultivar",
    "input_application_json", "experiment_latitude", "experiment_longitude",
    "geoname_id", "selected_latitude", "selected_longitude", "location_provenance_json",
    "weather_date_start", "weather_date_end",
    "treatment_name", "weather_station", "soil_profile",
    "irrigation", "irrigation_interval_days", "irrigation_amount_mm", "nitrogen_rate_kg_ha", "nitrogen_timing", "plant_density_ha",
    "area_input", "area_unit", "area_hectares", "dssat_yield_kg_ha",
    "dssat_yield_variable", "dssat_yield_definition", "dssat_yield_unit",
    "dssat_yield_basis", "yield_definition_verified",
    "observed_yield_kg_ha", "observed_residual_kg_ha", "predicted_residual_kg_ha",
    "ml_adjusted_yield_kg_ha", "dssat_harvest_kg", "ml_adjusted_harvest_kg",
    "model_evaluation_mae_kg_ha", "residual_formula", "yield_formula", "harvest_formula",
    "ml_model_outputs_json", "management_suggestions_json",
    "model_version", "dssat_version", "dssat_executable_sha256",
    "weather_source", "weather_metadata_json",
    "soil_source", "soil_metadata_json", "input_file_sha256_json",
    "residual_model_sha256", "run_metadata_json", "record_schema_version",
]


def append_prediction_record(project_data_dir: Path, values: dict) -> Path:
    """Append one simulation result, creating the CSV and header as needed."""
    path = Path(project_data_dir) / "predictions" / "scenario_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(FIELDS)
    if path.is_file() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader) if not set(FIELDS).issubset(existing_fields) else None
        if existing_rows is not None:
            fields.extend(field for field in existing_fields if field not in fields)
            backup = path.with_name(f"{path.stem}.schema-before-audit{path.suffix}")
            if not backup.exists():
                shutil.copy2(path, backup)
            migrated = path.with_suffix(".migrating.csv")
            with migrated.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_rows)
            os.replace(migrated, path)
        else:
            fields = existing_fields or list(FIELDS)
    if path.is_file() and path.stat().st_size and fields == list(FIELDS):
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            fields = csv.DictReader(stream).fieldnames or list(FIELDS)
    write_header = not path.exists() or path.stat().st_size == 0
    record = {field: "" for field in FIELDS}
    record.update(values)
    record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if not record["yield_formula"]:
        record["yield_formula"] = "ML-adjusted yield = DSSAT yield + model-predicted residual"
    if not record["residual_formula"]:
        record["residual_formula"] = "Measured residual = observed yield - DSSAT yield"
    if not record["harvest_formula"]:
        record["harvest_formula"] = "Total harvest (kg) = yield (kg/ha) x area (ha)"
    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return path
