"""Shared experiment prediction inputs and harvest-area arithmetic."""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .dssat.data.weather import read_weather
from .features.soil_features import aggregate_soil_profile as soil_features
from .features.weather_features import aggregate_weather

ACRE_TO_HECTARES = 0.40468564224


def build_experiment_features(config, experiment, summary_row, treatment: int,
                              soil_file=None, soil_id: str | None = None) -> dict:
    """Build the documented historical-season feature set for one DSSAT result."""
    start, end = int(summary_row["PDAT"]), int(summary_row["HDAT"])
    weather_files = sorted(config.data_dir.joinpath("Weather").glob(f"{experiment.weather_station}*.WTH"))
    if not weather_files:
        raise ValueError(f"No weather series was found for station {experiment.weather_station}.")
    weather = pd.concat([read_weather(path) for path in weather_files], ignore_index=True)
    features = {
        "crop": experiment.crop,
        "cultivar": experiment.cultivar_name,
        "weather_station": experiment.weather_station,
        "soil_id": soil_id or experiment.soil_id,
        "dssat_yield_kg_ha": float(summary_row["HWAM"]),
        **aggregate_weather(weather, start, end),
        **soil_features(soil_file or experiment.soil_file, soil_id or experiment.soil_id),
        **experiment.treatment_details[int(treatment)],
    }
    return features


class HybridNotSupported(ValueError):
    """Raised when held-out evidence does not cover the requested crop-region."""


def predict_residual_models(config, features: dict, dssat_yield: float,
                            validation_region: str) -> dict:
    """Return separate residual-model estimates and their validation support status."""
    from .modeling import MODEL_SPECS

    metadata_path = config.model_dir / "model_metadata.json"
    if not metadata_path.is_file():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    evaluation_dir = Path(config.project_data_dir) / "evaluation"
    evaluation_metadata_path = evaluation_dir / "metadata.json"
    evaluation_metrics_path = evaluation_dir / "metrics.json"
    if not evaluation_metadata_path.is_file() or not evaluation_metrics_path.is_file():
        raise ValueError("Run grouped held-out evaluation before using residual models.")
    evaluation_metadata = json.loads(evaluation_metadata_path.read_text(encoding="utf-8"))
    evaluation_metrics = json.loads(evaluation_metrics_path.read_text(encoding="utf-8"))
    if evaluation_metadata.get("dataset_sha256") != metadata.get("training_dataset_sha256"):
        raise ValueError("Residual models and grouped evaluation do not match the current training dataset.")

    support = evaluation_metrics.get("supported_regions_by_model", {})
    crop = str(features.get("crop", "")).strip().casefold()
    outputs = {}
    for key, spec in MODEL_SPECS.items():
        model_meta = metadata.get("models", {}).get(key)
        artifact_name = model_meta.get("artifact") if model_meta else None
        model_path = config.model_dir / artifact_name if artifact_name else None
        if not model_path or not model_path.is_file():
            continue
        feature_names = metadata.get("features", [])
        missing = [name for name in feature_names if name not in features]
        if not feature_names or missing:
            raise ValueError(f"Trained model inputs do not match this experiment; missing: {missing or 'feature list'}.")
        model = joblib.load(model_path)
        frame = pd.DataFrame([{name: features[name] for name in feature_names}])
        residual = float(model.predict(frame)[0])
        if not math.isfinite(residual):
            raise ValueError(f"{spec['label']} returned a non-finite residual.")
        output_key = {"hist_gradient_boosting": "hybrid", "extra_trees": "extra_trees", "ridge": "ridge"}[key]
        is_supported = any(str(item.get("crop", "")).strip().casefold() == crop
                           and item.get("region") == validation_region
                           for item in support.get(output_key, []))
        metrics = evaluation_metrics.get("pooled_oof", {}).get(output_key, {})
        outputs[key] = {
            "label": spec["label"], "estimator": spec["estimator"],
            "predicted_residual_kg_ha": residual,
            "estimated_yield_kg_ha": float(dssat_yield) + residual,
            "supported_for_crop_region": is_supported,
            "evaluation_mae_kg_ha": metrics.get("mae_kg_ha"),
        }
    return outputs


def management_scenarios(config, features: dict, dssat_yield: float,
                         applied_inputs: dict | None = None) -> list[dict]:
    """Suggest dataset-bounded ML what-if settings; these are screening ideas, not DSSAT reruns."""
    metadata_path = config.model_dir / "model_metadata.json"
    if not metadata_path.is_file():
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_path = Path(config.project_data_dir) / "processed" / metadata.get("training_dataset", "")
    if not dataset_path.is_file():
        return []
    training = pd.read_csv(dataset_path)
    crop_rows = training.loc[training["crop"].astype(str).str.casefold()
                             == str(features.get("crop", "")).casefold()]
    if crop_rows.empty:
        return []
    applied_inputs = applied_inputs or {}
    # Offer a setting only when it was actually applied in the DSSAT run.
    usable = {"nitrogen_rate_kg_ha": applied_inputs.get("nitrogen", {}).get("status") == "applied",
              "irrigated": applied_inputs.get("irrigation", {}).get("status") == "applied"}
    if not any(usable.values()):
        return []
    current = {name: features.get(name) for name in ("nitrogen_rate_kg_ha", "irrigated")}
    candidate_rows = []
    if usable["nitrogen_rate_kg_ha"] and "nitrogen_rate_kg_ha" in crop_rows:
        observed_n = pd.to_numeric(crop_rows["nitrogen_rate_kg_ha"], errors="coerce").dropna()
        if not observed_n.empty:
            for quantile in (0.25, 0.5, 0.75):
                value = float(observed_n.quantile(quantile))
                if abs(value - float(current["nitrogen_rate_kg_ha"] or 0)) > 1:
                    candidate_rows.append(("nitrogen_rate_kg_ha", value,
                                           f"N at {int(quantile * 100)}th percentile of paired {features['crop']} records"))
    if usable["irrigated"]:
        candidate_rows.append(("irrigated", 1 - int(bool(current["irrigated"])),
                               "Switch irrigation category to compare against paired records"))

    baseline = predict_residual_models(config, features, dssat_yield, validation_region="")
    baseline_values = [item["estimated_yield_kg_ha"] for item in baseline.values()]
    if not baseline_values:
        return []
    baseline_mean = float(np.mean(baseline_values))
    scenarios = []
    for feature_name, value, description in candidate_rows:
        candidate = dict(features)
        candidate[feature_name] = value
        estimates = predict_residual_models(config, candidate, dssat_yield, validation_region="")
        values = [item["estimated_yield_kg_ha"] for item in estimates.values()]
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        improvements = sum(estimate > baseline_mean for estimate in values)
        if mean > baseline_mean and improvements >= 2:
            scenarios.append({"input": "Nitrogen" if feature_name == "nitrogen_rate_kg_ha" else "Irrigation",
                              "value": (f"{value:.0f} kg N/ha" if feature_name == "nitrogen_rate_kg_ha"
                                        else ("Irrigated" if value else "Rainfed")),
                              "description": description,
                              "ensemble_estimate_kg_ha": mean,
                              "change_vs_current_ml_estimate_kg_ha": mean - baseline_mean,
                              "model_agreement": f"{improvements}/{len(values)} models higher"})
    return sorted(scenarios, key=lambda item: item["change_vs_current_ml_estimate_kg_ha"], reverse=True)[:3]


def predict_residual(config, features: dict, dssat_yield: float,
                     validation_region: str) -> dict | None:
    model_path = config.model_dir / "residual_model.joblib"
    metadata_path = config.model_dir / "model_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("estimator_name") != "HistGradientBoostingRegressor":
        raise ValueError("The saved residual model uses an older estimator. Retrain it before requesting an ML-adjusted estimate.")
    evaluation_dir = Path(config.project_data_dir) / "evaluation"
    evaluation_metadata_path = evaluation_dir / "metadata.json"
    evaluation_metrics_path = evaluation_dir / "metrics.json"
    if not evaluation_metadata_path.is_file() or not evaluation_metrics_path.is_file():
        raise ValueError("Run a grouped held-out evaluation before using the residual model.")
    evaluation_metadata = json.loads(evaluation_metadata_path.read_text(encoding="utf-8"))
    evaluation_metrics = json.loads(evaluation_metrics_path.read_text(encoding="utf-8"))
    residual_eval = evaluation_metadata.get("models", {}).get("residual", {})
    if (evaluation_metadata.get("dataset_sha256") != metadata.get("training_dataset_sha256")
            or residual_eval.get("estimator") != "HistGradientBoostingRegressor"):
        raise ValueError("The held-out evaluation does not match this model and dataset. Re-evaluate before using it.")
    support = evaluation_metrics.get("hybrid_supported_regions", [])
    crop = str(features.get("crop", "")).strip().casefold()
    approved = any(str(item.get("crop", "")).strip().casefold() == crop
                   and item.get("region") == validation_region for item in support)
    if not approved:
        raise HybridNotSupported(
            f"No held-out validation supports a hybrid estimate for {features.get('crop', 'this crop')} "
            f"in {validation_region.replace('|', ', ')}. Showing the DSSAT-only yield."
        )
    feature_names = metadata.get("features", [])
    missing = [name for name in feature_names if name not in features]
    if not feature_names or missing:
        raise ValueError(f"Trained model inputs do not match this experiment; missing: {missing or 'feature list'}.")
    model = joblib.load(model_path)
    frame = pd.DataFrame([{name: features[name] for name in feature_names}])
    residual = float(model.predict(frame)[0])
    if not math.isfinite(residual):
        raise ValueError("The residual model returned a non-finite prediction.")
    return {"predicted_residual_kg_ha": residual,
            "hybrid_yield_kg_ha": float(dssat_yield) + residual,
            "model_version": str(metadata.get("model_version", "unknown")),
            "evaluation_mae_kg_ha": evaluation_metrics["pooled_oof"]["hybrid"]["mae_kg_ha"]}


def area_estimate(yield_kg_ha: float, area: float, unit: str) -> dict:
    """Return validated farm-area conversion and total-harvest arithmetic."""
    try:
        area_value = float(area)
        yield_value = float(yield_kg_ha)
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter a numeric farm area.") from exc
    unit_key = str(unit).strip().lower()
    if not math.isfinite(area_value) or area_value <= 0:
        raise ValueError("Farm area must be a positive finite number.")
    if not math.isfinite(yield_value):
        raise ValueError("Yield must be finite to calculate total harvest.")
    if unit_key in {"ha", "hectare", "hectares"}:
        hectares, label = area_value, "hectares"
    elif unit_key in {"acre", "acres", "ac"}:
        hectares, label = area_value * ACRE_TO_HECTARES, "acres"
    else:
        raise ValueError("Choose hectares or acres for farm area.")
    total_kg = yield_value * hectares
    if not math.isfinite(total_kg):
        raise ValueError("Total harvest is non-finite; check the area and yield values.")
    return {"area_input": area_value, "area_unit": label, "area_hectares": hectares,
            "total_harvest_kg": total_kg, "total_harvest_tonnes": total_kg / 1000.0}


def load_latest_benchmark(config) -> dict | None:
    metrics_path = config.project_data_dir / "evaluation" / "metrics.json"
    metadata_path = config.project_data_dir / "evaluation" / "metadata.json"
    if not metrics_path.is_file():
        return None
    try:
        if not metadata_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("models", {}).get("residual", {}).get("estimator") != "HistGradientBoostingRegressor":
            return None
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
