"""Compact offline Flask interface for multi-crop DSSAT simulations and evaluation."""

import logging
import hashlib
import json
import csv
import math
from dataclasses import replace
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from markupsafe import Markup

from src.dashboard import dashboard_charts
from src.dssat.config import DssatConfig, project_directories
from src.dssat.data.repository import discover_experiments, validate_simulation
from src.dssat.data.observations import OBSERVED_YIELD_TARGETS, simulation_yield_target
from src.dssat.exceptions import DssatError
from src.dssat.runner import _weather_stations_from_experiment, run_experiment
from src.prediction import (HybridNotSupported, build_experiment_features,
                            load_latest_benchmark, predict_residual)
from src.records import append_prediction_record
from src.location import get_place, list_states, location_provenance, search_places
from src.validation_regions import india_state_region
from src.dssat.data.weather import read_weather
from src.dssat.data.power_weather import fetch_power_icasa
from src.dssat.data.soil import find_nearest_soil_profile
from src.features.soil_features import aggregate_soil_profile
from src.dssat.parser import parse_summary
from src.external_outcomes import load_des_apy_context

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


def _coordinate(value, lower, upper):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if lower <= number <= upper else None


def _sha256_if_file(path: Path) -> str | None:
    """Fingerprint an input file when it is present; preserve missing as missing."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _experiment_coordinates(config, experiment, audit=None):
    try:
        if audit is None:
            audit_path = config.project_data_dir / "diagnostics" / "experiment_support_audit.json"
            if not audit_path.is_file():
                return None
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        entry = next(item for item in audit.get("experiments", [])
                     if item.get("experiment_id") == experiment.experiment_id)
        probes = entry.get("probes", [])
        for probe in probes:
            if probe.get("status") != "simulation_passed" or not probe.get("run_id"):
                continue
            run_dir = config.project_data_dir / "runs" / probe["run_id"] / "output"
            outputs = {path.name.upper(): path for path in run_dir.rglob("*") if path.is_file()}
            output = next((outputs[name] for name in
                           ("SUMMARY.OUT", "PLANTSUM.OUT", f"{experiment.experiment_id}.OSU".upper())
                           if name in outputs), None)
            if output is None:
                continue
            row = parse_summary(output).iloc[0]
            latitude, longitude = _coordinate(row.get("XLAT"), -90, 90), _coordinate(row.get("LONG"), -180, 180)
            if latitude is not None and longitude is not None:
                return latitude, longitude
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError):
        return None
    return None


def _distance_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    value = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1, math.sqrt(value)))


def _evaluation_config():
    _, project_data_dir, model_dir = project_directories()
    return SimpleNamespace(project_data_dir=project_data_dir, model_dir=model_dir)


def _successful_experiments(config, crop=None):
    """Limit the UI to treatments verified by the latest DSSAT support audit."""
    experiments = discover_experiments(config.data_dir, crop=crop,
                                       profile_path=config.home / "DSSATPRO.v48")
    audit_path = config.project_data_dir / "diagnostics" / "experiment_support_audit.json"
    if not audit_path.is_file():
        return experiments
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return experiments
    successful = {}
    for item in audit.get("experiments", []):
        treatment_ids = {str(probe["treatment_id"]) for probe in item.get("probes", [])
                         if probe.get("status") == "simulation_passed"
                         and float(probe.get("positive_yield_candidate_values", {}).get("HWAM", 0) or 0) > 0}
        if treatment_ids:
            successful[item["experiment_id"]] = treatment_ids
    filtered = []
    for experiment in experiments:
        if experiment.experiment_id not in successful:
            continue
        treatment_ids = tuple(treatment for treatment in experiment.treatments
                              if str(treatment) in successful[experiment.experiment_id])
        if not treatment_ids:
            continue
        details = {int(treatment): experiment.treatment_details[int(treatment)]
                   for treatment in treatment_ids if int(treatment) in experiment.treatment_details}
        filtered.append(replace(experiment, treatments=treatment_ids, treatment_details=details))
    return filtered


def _pooled_comparison(metrics: dict | None) -> dict | None:
    if not metrics or "pooled_oof" not in metrics:
        return None
    pooled = metrics["pooled_oof"]
    dssat_mae = pooled["dssat_only"]["mae_kg_ha"]
    hybrid_mae = pooled["hybrid"]["mae_kg_ha"]
    status = "beats" if hybrid_mae < dssat_mae else "is worse than" if hybrid_mae > dssat_mae else "matches"
    return {"status": status, "dssat_mae": dssat_mae, "hybrid_mae": hybrid_mae, "pooled": pooled}


@app.get("/")
def index():
    try:
        config = DssatConfig.discover()
        experiments = _successful_experiments(config)
        crops = sorted({experiment.crop for experiment in experiments})
        error = None
    except DssatError as exc:
        experiments, crops, error = [], [], str(exc)
    try:
        _, location_data_dir, _ = project_directories()
        states = list_states(location_data_dir)
        location_error = None
    except (OSError, ValueError) as exc:
        states, location_error = [], str(exc)
    seasons = ["Kharif / Monsoon", "Rabi / Winter", "Zaid / Summer", "Spring", "Autumn", "Other"]
    soil_types = ["Sandy", "Sandy loam", "Loam", "Silt loam", "Clay loam", "Clay", "Other / local soil"]
    return render_template("index.html", experiments=experiments, crops=crops, error=error,
                           states=states, seasons=seasons, soil_types=soil_types,
                           location_error=location_error)


@app.get("/api/places")
def places_api():
    _, project_data_dir, _ = project_directories()
    state = request.args.get("state", "").strip()
    query = request.args.get("q", "").strip()
    try:
        matches = search_places(project_data_dir, state, query)
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify([{
        "geoname_id": item["geoname_id"], "name": item["name"],
        "label": f"{item['name']} · admin area {item['admin2_code']}" if item["admin2_code"] else item["name"],
        "latitude": item["latitude"], "longitude": item["longitude"],
    } for item in matches])


@app.post("/predict")
def predict():
    try:
        config = DssatConfig.discover()
        crop = request.form.get("crop")
        experiment_id = request.form.get("experiment", "")
        field_inputs = {
            "state": request.form.get("state", "").strip(),
            "city": request.form.get("city", "").strip(),
            "geoname_id": request.form.get("geoname_id", "").strip(),
            "season": request.form.get("season", "").strip(),
            "planting_date": request.form.get("planting_date", "").strip(),
            "soil_type": request.form.get("soil_type", "").strip(),
            "irrigation": request.form.get("irrigation", "rainfed").strip(),
            "irrigation_interval_days": request.form.get("irrigation_interval_days", "7").strip(),
            "irrigation_amount_mm": request.form.get("irrigation_amount_mm", "50").strip(),
            "nitrogen_rate_kg_ha": request.form.get("nitrogen_rate", "0").strip(),
            "nitrogen_timing": request.form.get("nitrogen_timing", "planting").strip(),
            "plant_density_ha": request.form.get("plant_density", "").strip(),
        }
        if not all(field_inputs[key] for key in ("state", "city", "geoname_id", "season", "planting_date", "soil_type", "plant_density_ha")):
            raise ValueError("Complete each required field before running the simulation.")
        try:
            geoname_id = int(field_inputs["geoname_id"])
        except ValueError as exc:
            raise ValueError("Choose a city or town from the location suggestions.") from exc
        place = get_place(config.project_data_dir, field_inputs["state"], geoname_id)
        if place is None:
            raise ValueError("That location is not in the local GeoNames list for the selected state. Choose a suggestion again.")
        field_inputs["city"] = place["name"]
        datetime.strptime(field_inputs["planting_date"], "%Y-%m-%d")
        try:
            field_inputs["nitrogen_rate_kg_ha"] = float(field_inputs["nitrogen_rate_kg_ha"])
            field_inputs["plant_density_ha"] = float(field_inputs["plant_density_ha"])
            field_inputs["irrigation_interval_days"] = int(field_inputs["irrigation_interval_days"])
            field_inputs["irrigation_amount_mm"] = float(field_inputs["irrigation_amount_mm"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Nitrogen, plant density, and irrigation schedule must be valid numbers.") from exc
        if field_inputs["irrigation"] not in {"rainfed", "irrigated"}:
            raise ValueError("Choose rainfed or irrigated for irrigation.")
        eligible_experiments = _successful_experiments(config, crop=crop)
        experiment = next((item for item in eligible_experiments
                           if item.experiment_id == experiment_id), None)
        if experiment is None and eligible_experiments:
            experiment = eligible_experiments[0]
        if experiment is None:
            raise ValueError("No DSSAT experiment with a completed treatment is available for this crop.")
        coordinate_matches = []
        try:
            experiment_audit = json.loads((config.project_data_dir / "diagnostics" /
                                           "experiment_support_audit.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            experiment_audit = None
        for item in eligible_experiments:
            coordinates = _experiment_coordinates(config, item, experiment_audit)
            if coordinates:
                distance = _distance_km(place["latitude"], place["longitude"], *coordinates)
                coordinate_matches.append((distance, item, coordinates))
        coordinate_matches.sort(key=lambda match: match[0])
        location_matched = bool(coordinate_matches and coordinate_matches[0][0] <= 100)
        location_distance_km = coordinate_matches[0][0] if coordinate_matches else None
        if location_matched:
            experiment = coordinate_matches[0][1]
        available_treatments = experiment.treatment_details
        requested_irrigated = field_inputs["irrigation"] == "irrigated"
        treatment = min(
            experiment.treatments,
            key=lambda treatment_id: (
                (abs((available_treatments[int(treatment_id)].get("nitrogen_rate_kg_ha") or 0)
                     - field_inputs["nitrogen_rate_kg_ha"]))
                + (10000 if available_treatments[int(treatment_id)].get("irrigated") != requested_irrigated else 0)
                if available_treatments[int(treatment_id)].get("nitrogen_rate_kg_ha") is not None
                else 100000
            ),
        )
        treatment = int(treatment)
        validate_simulation(experiment)
        weather_metadata = {
            "source": "DSSAT installed weather file", "validated": None,
            "weather_file": f"{experiment.experiment_id}.WTH",
            "resolution": "installed station record",
        }
        run_scenario = dict(field_inputs)
        soil_match = find_nearest_soil_profile(
            config.soil_dir, place["latitude"], place["longitude"], max_distance_km=100.0)
        if soil_match.get("status") == "matched":
            run_scenario.update({
                "soil_override_path": soil_match["file"],
                "soil_override_id": soil_match["profile_id"],
                "soil_source": soil_match["source"],
                "soil_latitude": soil_match["latitude"],
                "soil_longitude": soil_match["longitude"],
                "soil_distance_km": soil_match["distance_km"],
                "soil_matching_limit_km": soil_match["max_distance_km"],
                "soil_depth_cm": soil_match["depth_cm"],
                "soil_texture_code": soil_match["texture_code"],
            })
        stations = _weather_stations_from_experiment(experiment)
        if len(stations) == 1:
            try:
                weather_override, weather_metadata = fetch_power_icasa(
                    config.project_data_dir, place["latitude"], place["longitude"],
                    field_inputs["planting_date"], experiment.weather_station,
                    f"{experiment.experiment_id}.WTH")
                run_scenario["weather_override_path"] = str(weather_override)
                run_scenario["weather_source_metadata"] = weather_metadata
                run_scenario["weather_simulation_start_date"] = weather_metadata["simulation_start"]
            except (OSError, ValueError, RuntimeError) as exc:
                weather_metadata = {
                    "source": "DSSAT installed weather file", "validated": None,
                    "weather_file": f"{experiment.experiment_id}.WTH",
                    "resolution": "installed station record",
                    "fallback_reason": str(exc), "requested_source": "NASA POWER Daily API",
                }
                logging.warning("Using installed DSSAT weather after NASA POWER fallback: %s", exc)
        else:
            weather_metadata["fallback_reason"] = (
                "The experiment references multiple weather stations; a single location-based station override would be ambiguous.")
            weather_metadata["requested_source"] = "NASA POWER Daily API"
        result = run_experiment(config, experiment, treatment, scenario=run_scenario)
        input_application = dict(result.get("scenario_application") or {})
        input_application["location"] = {
            "status": "matched_nearby_installed_experiment" if location_matched else "coordinates_recorded_no_nearby_installed_experiment",
            "experiment_id": experiment.experiment_id,
            "nearest_experiment_distance_km": location_distance_km,
            "matching_radius_km": 100,
        }
        input_application["weather"] = {
            "status": "applied" if run_scenario.get("weather_override_path") else "installed_weather_fallback",
            **weather_metadata,
        }
        input_application["season"] = {"status": "recorded_not_used", "reason": "Season does not currently select weather or DSSAT management."}
        soil_application = input_application.get("soil") or {}
        soil_match_provenance = dict(soil_match)
        if soil_match_provenance.get("file"):
            soil_match_provenance["file"] = Path(soil_match_provenance["file"]).name
        input_application["soil_profile_match"] = soil_match_provenance
        input_application["soil_type"] = {
            "status": "recorded_not_used",
            "reason": "The broad soil-texture choice is retained as context; DSSAT uses a complete georeferenced profile when one is available within 100 km, and does not invent missing profile values.",
        }
        field_inputs["input_application"] = input_application
        soil_used_file = (Path(soil_match["file"]) if soil_application.get("status") == "applied"
                          else experiment.soil_file)
        soil_used_id = soil_application.get("profile_id", experiment.soil_id)
        row = result["summary"].iloc[0]
        dssat_yield = float(row["HWAM"])
        yield_target = simulation_yield_target(experiment.crop_code)
        planting_yyddd = int(row["PDAT"])
        planting_year, planting_day = divmod(planting_yyddd, 1000)
        if planting_year < 100:
            planting_year += 1900 if planting_year >= 50 else 2000
        planting_date = datetime.strptime(f"{planting_year:04d}{planting_day:03d}", "%Y%j").date().isoformat()
        field_inputs["dssat_planting_date"] = planting_date
        if run_scenario.get("weather_override_path"):
            weather_start = weather_metadata.get("simulation_start")
            weather_end = weather_metadata.get("crop_window_end")
            weather_path_used = Path(run_scenario["weather_override_path"])
        else:
            weather_files = list(config.weather_dir.glob(f"{experiment.weather_station.upper()}*.WTH"))
            weather_dates = []
            for weather_file in weather_files:
                try:
                    dates = read_weather(weather_file)["DATE"].dropna().astype(int).tolist()
                    weather_dates.extend(dates)
                except (OSError, ValueError, KeyError):
                    continue
            def weather_iso_date(value):
                yy, day = divmod(int(value), 1000)
                year = yy + (1900 if yy >= 50 else 2000)
                return datetime.strptime(f"{year:04d}{day:03d}", "%Y%j").date().isoformat()
            weather_start = weather_iso_date(min(weather_dates)) if weather_dates else None
            weather_end = weather_iso_date(max(weather_dates)) if weather_dates else None
            weather_path_used = config.weather_dir / f"{experiment.experiment_id}.WTH"
        try:
            soil_summary = aggregate_soil_profile(soil_used_file, soil_used_id)
            soil_depth_cm = soil_summary.get("soil_depth_cm")
        except (OSError, ValueError, KeyError):
            soil_depth_cm = None
        location_details = location_provenance(place, experiment, weather_path_used,
                                               weather_start, weather_end, soil_depth_cm,
                                               coordinates_used=location_matched,
                                               nearest_distance_km=location_distance_km,
                                               weather_metadata=weather_metadata,
                                               soil_metadata=soil_application,
                                               soil_match=soil_match_provenance,
                                               soil_file=soil_used_file,
                                               soil_id=soil_used_id)
        treatment_details = experiment.treatment_details[treatment]

        hybrid, model_error, model_note = None, None, None
        model_path = config.model_dir / "residual_model.joblib"
        model_metadata_path = config.model_dir / "model_metadata.json"
        has_model_files = model_path.is_file() and model_metadata_path.is_file()
        try:
            saved_model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8")) if has_model_files else {}
        except (OSError, ValueError):
            saved_model_metadata = {}
        model_available = has_model_files and saved_model_metadata.get("estimator_name") == "HistGradientBoostingRegressor"
        model_needs_retrain = has_model_files and not model_available
        if not yield_target["verified"]:
            model_note = "The crop's harvested-product definition is not reviewed; showing the raw DSSAT output without residual adjustment."
        elif model_available:
            try:
                features = build_experiment_features(config, experiment, row, treatment,
                                                    soil_file=soil_used_file, soil_id=soil_used_id)
                hybrid = predict_residual(config, features, dssat_yield,
                                          validation_region=india_state_region(place["state_name"]))
                if hybrid is None:
                    model_error = "Model artifacts became unavailable during this request."
            except HybridNotSupported as exc:
                model_note = str(exc)
            except Exception as exc:
                logging.exception("Residual model prediction failed for run_id=%s", result["run_id"])
                model_error = "Hybrid calculation failed; the DSSAT result below is not a hybrid prediction. Check the application log for details."

        benchmark = load_latest_benchmark(config)
        comparison = _pooled_comparison(benchmark)
        # The custom planting and management overrides create a new scenario;
        # an observation attached to the unmodified source treatment is not a
        # valid paired target for this run.
        observed_yield = observed_residual = None
        provenance = {
            "dssat_version": config.version(),
            "model_version": hybrid["model_version"] if hybrid else None,
            "run_id": result["run_id"],
            "experiment": experiment.experiment_id,
            "treatment": treatment,
            "treatment_name": experiment.treatment_details[treatment].get("treatment_name", "not recorded"),
            "cultivar": experiment.cultivar_name,
            "weather_station": experiment.weather_station,
            "soil_profile": soil_used_id,
            "input_files": [experiment.experiment_file.name,
                            *(weather_metadata.get("weather_files") or [experiment.weather_file.name]),
                            soil_used_file.name, experiment.cultivar_file.name],
        }
        run_dir = Path(result.get("run_dir", ""))
        input_hashes = {
            relative_path.as_posix(): digest
            for filename in (result.get("metadata", {}).get("input_files") or [])
            if (relative_path := Path(filename)).is_relative_to("input")
            and (digest := _sha256_if_file(run_dir / relative_path)) is not None
        }
        csv_path = append_prediction_record(config.project_data_dir, {
            "run_id": result["run_id"], "crop": getattr(experiment, "crop", "Unknown crop"),
            "experiment_id": experiment.experiment_id, "treatment_id": treatment,
            "experiment_location": experiment.location,
            "dssat_planting_date": planting_date,
            "cultivar": experiment.cultivar_name,
            "treatment_name": treatment_details.get("treatment_name", "not recorded"),
            "weather_station": experiment.weather_station,
            "irrigation": treatment_details.get("irrigated"),
            "nitrogen_rate_kg_ha": treatment_details.get("nitrogen_rate_kg_ha"),
            "soil_profile": soil_used_id,
            **field_inputs,
            "input_application_json": json.dumps(input_application, ensure_ascii=False),
            "geoname_id": place["geoname_id"],
            "selected_latitude": place["latitude"], "selected_longitude": place["longitude"],
            "location_provenance_json": json.dumps(location_details, ensure_ascii=False),
            "weather_date_start": weather_start, "weather_date_end": weather_end,
            "experiment_latitude": _coordinate(row.get("XLAT"), -90, 90),
            "experiment_longitude": _coordinate(row.get("LONG"), -180, 180),
            "dssat_yield_kg_ha": dssat_yield,
            "dssat_yield_variable": yield_target["column"],
            "dssat_yield_definition": yield_target["definition"],
            "dssat_yield_unit": yield_target["unit"],
            "dssat_yield_basis": yield_target["basis"],
            "yield_definition_verified": yield_target["verified"],
            "observed_yield_kg_ha": observed_yield, "observed_residual_kg_ha": observed_residual,
            "predicted_residual_kg_ha": hybrid["predicted_residual_kg_ha"] if hybrid else None,
            "ml_adjusted_yield_kg_ha": hybrid["hybrid_yield_kg_ha"] if hybrid else None,
            "residual_formula": "Measured residual = observed yield - DSSAT yield; ML predicts residual from paired observations",
            "yield_formula": "ML-adjusted yield = DSSAT yield + model-predicted residual",
            "model_version": hybrid["model_version"] if hybrid else None,
            "model_evaluation_mae_kg_ha": hybrid["evaluation_mae_kg_ha"] if hybrid else None,
            "dssat_version": config.version(),
            "weather_source": weather_metadata.get("source", "DSSAT installed weather file"),
            "weather_metadata_json": json.dumps(weather_metadata, ensure_ascii=False),
            "soil_source": soil_application.get("source", "DSSAT experiment soil profile"),
            "soil_metadata_json": json.dumps(soil_application, ensure_ascii=False),
            "input_file_sha256_json": json.dumps(input_hashes, ensure_ascii=False, sort_keys=True),
            "residual_model_sha256": _sha256_if_file(model_path) if hybrid else None,
            "run_metadata_json": json.dumps(result.get("metadata", {}), ensure_ascii=False, default=str),
            "record_schema_version": "2",
        })
        return render_template("result.html", experiment=experiment, treatment=treatment,
                               yield_kg_ha=dssat_yield, field_inputs=field_inputs,
                               yield_target=yield_target,
                               hybrid=hybrid, model_error=model_error, model_note=model_note,
                               model_available=model_available,
                               provenance=provenance, growth_rows=len(result["growth"]) if result["growth"] is not None else 0,
                               comparison=comparison, benchmark=benchmark, observed_yield=observed_yield,
                               observed_residual=observed_residual, csv_path=csv_path,
                               model_needs_retrain=model_needs_retrain,
                               location_details=location_details)
    except (DssatError, ValueError, KeyError, TypeError) as exc:
        return render_template("error.html", message=str(exc)), 400


@app.get("/evaluation")
def evaluation():
    config = _evaluation_config()
    charts = dashboard_charts(config)
    metrics = load_latest_benchmark(config)
    metadata_path = config.project_data_dir / "evaluation" / "metadata.json"
    metadata = None
    if metadata_path.is_file():
        try:
            import json
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = None
    if metrics is None:
        metadata = None
    return render_template("evaluation.html", charts={key: Markup(value) for key, value in charts.items()} if charts else None,
                           metrics=metrics, metadata=metadata)


@app.get("/research")
def research():
    _, project_data_dir, _ = project_directories()
    outcomes, _ = load_des_apy_context(project_data_dir)
    paired_summary = {"records": 0, "crops": 0}
    paired_path = project_data_dir / "processed" / "dssat_hybrid_dataset.csv"
    if paired_path.is_file():
        try:
            with paired_path.open("r", newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    paired_summary["records"] += 1
                    if row.get("crop"):
                        paired_summary.setdefault("crop_names", set()).add(row["crop"])
            paired_summary["crops"] = len(paired_summary.pop("crop_names", set()))
        except (OSError, UnicodeError, csv.Error):
            paired_summary = {"records": 0, "crops": 0}
    outcome_summary = {
        "rows": int(len(outcomes)),
        "crops": int(outcomes["crop"].nunique()) if not outcomes.empty else 0,
        "states": int(outcomes["state"].nunique()) if not outcomes.empty else 0,
    }
    return render_template("research.html", outcome_summary=outcome_summary,
                           paired_summary=paired_summary,
                           reviewed_yield_crops=len(OBSERVED_YIELD_TARGETS))


@app.get("/outcomes")
def outcomes():
    _, project_data_dir, _ = project_directories()
    frame, manifests = load_des_apy_context(project_data_dir)
    if not frame.empty:
        frame = frame.sort_values(["year", "state", "crop", "district"], ascending=[False, True, True, True])
    records = frame.head(250).to_dict("records")
    summary = {
        "rows": int(len(frame)),
        "district_rows": int(frame["aggregation_level"].eq("district").sum()) if not frame.empty else 0,
        "state_rows": int(frame["aggregation_level"].eq("state").sum()) if not frame.empty else 0,
        "crops": int(frame["crop"].nunique()) if not frame.empty else 0,
        "states": int(frame["state"].nunique()) if not frame.empty else 0,
    }
    return render_template("outcomes.html", rows=records, summary=summary,
                           manifests=manifests, total_rows=len(frame))


if __name__ == "__main__":
    app.run(debug=False)
