"""Grouped five-fold retrospective benchmark with out-of-fold predictions."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from .modeling import (ELIGIBLE_FEATURES, ML_ONLY_FEATURES, MODEL_SPECS, RANDOM_SEED,
                       RESIDUAL_FEATURES, make_regressor, validate_training_frame)
from .validation_regions import dssat_location_region, dssat_validation_group

METHOD_COLUMNS = {
    "dssat_only": "dssat_prediction_kg_ha",
    "ml_only": "ml_only_prediction_kg_ha",
    "hybrid": "hybrid_prediction_kg_ha",
    "extra_trees": "extra_trees_prediction_kg_ha",
    "ridge": "ridge_prediction_kg_ha",
}


def regression_metrics(observed, predicted) -> dict:
    actual = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    error = estimate - actual
    positive = actual > 0
    r2 = None
    if len(actual) >= 2 and np.var(actual) > 0:
        r2 = float(r2_score(actual, estimate))
    within = None
    if positive.any():
        within = 100.0 * float((np.abs(error[positive]) <= 0.10 * actual[positive]).sum()) / int(positive.sum())
    return {
        "n": int(len(actual)),
        "mae_kg_ha": float(mean_absolute_error(actual, estimate)),
        "rmse_kg_ha": float(mean_squared_error(actual, estimate) ** 0.5),
        "mean_signed_error_kg_ha": float(error.mean()),
        "r2": r2,
        "within_10_percent_pct": within,
        "within_10_percent_denominator": int(positive.sum()),
        "excluded_nonpositive_observed_count": int((~positive).sum()),
    }


def _dependency_versions() -> dict:
    result = {}
    for package in ("pandas", "numpy", "scikit-learn", "joblib"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def evaluate_dataset(dataset_path: Path, output_dir: Path) -> dict:
    """Recompute grouped five-fold predictions and write audit artifacts."""
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    frame = pd.read_csv(dataset_path)
    validate_training_frame(frame)
    if "location" not in frame.columns:
        raise ValueError("Dataset needs source location values before crop-and-region validation can run.")
    frame["validation_region"] = frame["location"].map(dssat_location_region)
    grouped = frame.apply(
        lambda row: dssat_validation_group(row.get("location", ""), row["experiment_id"]),
        axis=1, result_type="expand",
    )
    frame["validation_group"] = grouped[0]
    frame["validation_group_type"] = grouped[1]
    groups = sorted(frame["validation_group"].astype(str).unique().tolist())
    if len(groups) < 2:
        raise ValueError("Grouped evaluation requires at least two distinct source groups.")
    known_region_frame = frame.loc[frame["validation_region"] != "UNKNOWN|UNKNOWN"]
    region_counts_by_crop = known_region_frame.groupby("crop", dropna=False)["validation_region"].nunique().to_dict()

    rows = []
    folds = []
    group_types = frame.groupby("validation_group")["validation_group_type"].first().to_dict()
    n_splits = min(5, len(groups))
    splitter = GroupKFold(n_splits=n_splits)
    for fold_number, (train_indices, test_indices) in enumerate(
            splitter.split(frame, groups=frame["validation_group"]), start=1):
        train = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        held_out_groups = sorted(test["validation_group"].astype(str).unique().tolist())
        if train.empty or test.empty:
            raise ValueError(f"Cannot form held-out group fold {fold_number}.")
        # Each pipeline fits its own encoder only on this fold's training data.
        ml_model = make_regressor(ML_ONLY_FEATURES, seed=RANDOM_SEED)
        ml_model.fit(train[ML_ONLY_FEATURES], train["observed_yield_kg_ha"])
        ml_prediction = ml_model.predict(test[ML_ONLY_FEATURES])

        residual_predictions = {}
        for model_name in MODEL_SPECS:
            residual_model = make_regressor(RESIDUAL_FEATURES, seed=RANDOM_SEED,
                                            model_name=model_name)
            residual_model.fit(train[RESIDUAL_FEATURES], train["residual_kg_ha"])
            residual_predictions[model_name] = residual_model.predict(test[RESIDUAL_FEATURES])
        dssat_prediction = test["dssat_yield_kg_ha"].to_numpy(dtype=float)
        for index, (_, item) in enumerate(test.iterrows()):
            prediction_row = {
                "experiment_id": str(item["experiment_id"]),
                "treatment_id": int(item["treatment_id"]),
                "crop": str(item["crop"]),
                "validation_region": str(item["validation_region"]),
                "validation_group": str(item["validation_group"]),
                "validation_group_type": str(item["validation_group_type"]),
                "held_out_region": str(item["validation_region"]),
                "held_out_groups": ";".join(held_out_groups),
                "held_out_experiment": str(item["experiment_id"]),
                "fold_number": fold_number,
                "n_train": int(len(train)), "n_test": int(len(test)),
                "observed_yield_kg_ha": float(item["observed_yield_kg_ha"]),
                "dssat_prediction_kg_ha": float(dssat_prediction[index]),
                "ml_only_prediction_kg_ha": float(ml_prediction[index]),
            }
            for model_name, residual_values in residual_predictions.items():
                column_key = {"hist_gradient_boosting": "hybrid_prediction_kg_ha",
                              "extra_trees": "extra_trees_prediction_kg_ha",
                              "ridge": "ridge_prediction_kg_ha"}[model_name]
                prediction_row[column_key] = float(dssat_prediction[index] + residual_values[index])
            prediction_row["predicted_residual_kg_ha"] = prediction_row["hybrid_prediction_kg_ha"] - float(dssat_prediction[index])
            rows.append(prediction_row)
        fold_predictions = [r for r in rows if r["fold_number"] == fold_number]
        observed = [r["observed_yield_kg_ha"] for r in fold_predictions]
        fold_metrics = {name: regression_metrics(observed, [r[column] for r in fold_predictions])
                        for name, column in METHOD_COLUMNS.items()}
        folds.append({"fold_number": fold_number, "held_out_groups": held_out_groups,
                      "held_out_group_types": sorted({group_types[group] for group in held_out_groups}),
                      "held_out_experiments": sorted(test["experiment_id"].astype(str).unique().tolist()),
                      "n_held_out_experiments": int(test["experiment_id"].nunique()),
                      "training_experiments": sorted(train["experiment_id"].astype(str).unique().tolist()),
                      "n_train": int(len(train)), "n_test": int(len(test)), "metrics": fold_metrics})

    predictions = pd.DataFrame(rows).sort_values(["fold_number", "experiment_id", "treatment_id"]).reset_index(drop=True)
    if predictions.duplicated(["experiment_id", "treatment_id"]).any() or len(predictions) != len(frame):
        raise RuntimeError("Benchmark integrity failure: each dataset record must receive exactly one OOF prediction.")
    observed = predictions["observed_yield_kg_ha"].to_numpy(dtype=float)
    pooled = {name: regression_metrics(observed, predictions[column]) for name, column in METHOD_COLUMNS.items()}
    per_experiment = []
    for group_name, subset in predictions.groupby("experiment_id", sort=True):
        per_experiment.append({
            "experiment_id": str(group_name), "n": int(len(subset)),
            "metrics": {name: regression_metrics(subset["observed_yield_kg_ha"], subset[column])
                        for name, column in METHOD_COLUMNS.items()},
        })
    crop_region = []
    supported_regions = []
    regional_predictions = predictions.loc[predictions["validation_region"] != "UNKNOWN|UNKNOWN"]
    for (crop, region), subset in regional_predictions.groupby(["crop", "validation_region"], sort=True):
        region_metrics = {name: regression_metrics(subset["observed_yield_kg_ha"], subset[column])
                          for name, column in METHOD_COLUMNS.items()}
        independent_region_count = int(region_counts_by_crop.get(crop, 0))
        enough_regions = independent_region_count >= 3
        model_support = {name: bool(enough_regions and region_metrics[name]["mae_kg_ha"]
                                    < region_metrics["dssat_only"]["mae_kg_ha"])
                         for name in ("hybrid", "extra_trees", "ridge")}
        approved = model_support["hybrid"]
        decision = ("one_or_more_models_supported" if any(model_support.values()) else
                    "insufficient_independent_regions" if not enough_regions else
                    "dssat_only_lower_or_equal_mae")
        item = {
            "crop": str(crop), "region": str(region), "n": int(len(subset)),
            "n_independent_regions_for_crop": independent_region_count,
            "held_out_experiments": sorted(subset["experiment_id"].astype(str).unique().tolist()),
            "metrics": region_metrics, "hybrid_supported": bool(approved),
            "model_support": model_support, "decision": decision,
        }
        crop_region.append(item)
        if approved:
            supported_regions.append({"crop": str(crop), "region": str(region)})
    hybrid_beats_dssat = pooled["hybrid"]["mae_kg_ha"] < pooled["dssat_only"]["mae_kg_ha"]
    supported_by_model = {
        model_key: [{"crop": item["crop"], "region": item["region"]}
                    for item in crop_region if item["model_support"].get(model_key, False)]
        for model_key in ("hybrid", "extra_trees", "ridge")
    }
    metrics = {
        "benchmark_type": "retrospective_historical_full_season_weather",
        "validation_method": "5_fold_grouped_oof",
        "group_column": "validation_group",
        "n_records": int(len(predictions)), "n_experiments": int(frame["experiment_id"].nunique()),
        "n_regions": int(frame.loc[frame["validation_region"] != "UNKNOWN|UNKNOWN", "validation_region"].nunique()),
        "n_validation_groups": len(groups),
        "n_site_label_groups": sum(group_types[group] == "site_label_only" for group in groups),
        "n_experiment_only_groups": sum(group_types[group] == "experiment_only" for group in groups),
        "n_records_without_canonical_region": int(frame["validation_region"].eq("UNKNOWN|UNKNOWN").sum()),
        "folds": folds, "per_experiment": per_experiment,
        "crop_region_validation": crop_region,
        "hybrid_supported_regions": supported_regions,
        "supported_regions_by_model": supported_by_model,
        "minimum_independent_regions_per_crop": 3,
        "pooled_oof": pooled,
        "pooled_hybrid_beats_dssat_on_mae": bool(hybrid_beats_dssat),
        "model_comparison": {key: {"label": MODEL_SPECS[model_key]["label"],
                                    "estimator": MODEL_SPECS[model_key]["estimator"],
                                    "target": "measured yield minus DSSAT yield"}
                             for key, model_key in (("hybrid", "hist_gradient_boosting"),
                                                    ("extra_trees", "extra_trees"),
                                                    ("ridge", "ridge"))},
        "interpretation": "Five-fold out-of-fold evaluation keeps each canonical country/admin-1 region together when available. Rows without one are grouped by exact DSSAT site label, or by experiment when the site label is missing. Fallback groups are included in pooled metrics but never count as geographic evidence for hybrid approval. A crop-region is supported only when that crop has at least three canonical regions and lower held-out-region hybrid MAE. Historical full-season weather makes this retrospective, not sowing-time forecast validation.",
    }
    try:
        relative_dataset = dataset_path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        relative_dataset = dataset_path.name
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": relative_dataset,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "record_count": int(len(frame)), "experiment_groups": sorted(frame["experiment_id"].astype(str).unique().tolist()),
        "validation_regions": sorted(frame.loc[frame["validation_region"] != "UNKNOWN|UNKNOWN", "validation_region"].unique().tolist()),
        "validation_groups": groups,
        "grouping_strategy": "Five-fold GroupKFold: canonical country/admin-1 groups where available; otherwise exact DSSAT location labels; rows with no usable location are grouped by experiment. Fallback groups are not geographic validation evidence. Preprocessing is fitted separately inside every fold.",
        "random_seeds": {"grouped_oof": RANDOM_SEED},
        "models": {"ml_only": {"estimator": "HistGradientBoostingRegressor", "max_iter": 150,
                                 "features": ML_ONLY_FEATURES},
                   **{name: {"estimator": spec["estimator"], "label": spec["label"],
                             "features": RESIDUAL_FEATURES,
                             "target": "observed_yield_kg_ha - dssat_yield_kg_ha"}
                      for name, spec in MODEL_SPECS.items()},
                   "residual": {"estimator": MODEL_SPECS["hist_gradient_boosting"]["estimator"],
                                "features": RESIDUAL_FEATURES}},
        "excluded_ml_only_features": ["dssat_yield_kg_ha", "observed_yield_kg_ha", "residual_kg_ha",
                                      "experiment_id", "treatment_id", "run_id", "held_out_experiment"],
        "eligible_environment_and_management_features": ELIGIBLE_FEATURES,
        "weather_limitation": "Features summarize the completed crop season; this benchmark is retrospective, not an operational sowing-time forecast.",
        "dependency_versions": _dependency_versions(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return {"metrics": metrics, "metadata": metadata, "predictions": predictions}
