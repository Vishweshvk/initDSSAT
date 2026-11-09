"""Leave-one-experiment-out retrospective benchmark with out-of-fold predictions."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .modeling import (ELIGIBLE_FEATURES, ML_ONLY_FEATURES, RANDOM_SEED,
                       RESIDUAL_FEATURES, make_regressor, validate_training_frame)

METHOD_COLUMNS = {
    "dssat_only": "dssat_prediction_kg_ha",
    "ml_only": "ml_only_prediction_kg_ha",
    "hybrid": "hybrid_prediction_kg_ha",
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
    """Recompute all leave-one-experiment-out predictions and write audit artifacts."""
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    frame = pd.read_csv(dataset_path)
    validate_training_frame(frame)
    groups = sorted(frame["experiment_id"].astype(str).unique().tolist())
    if len(groups) < 2:
        raise ValueError("Leave-one-experiment-out evaluation requires at least two experiment groups.")

    rows = []
    folds = []
    raw_groups = frame["experiment_id"].astype(str)
    for fold_number, held_out in enumerate(groups, start=1):
        test_mask = raw_groups.eq(held_out)
        train = frame.loc[~test_mask]
        test = frame.loc[test_mask]
        if train.empty or test.empty:
            raise ValueError(f"Cannot form held-out fold for {held_out}.")
        # Each pipeline fits its own encoder only on this fold's training data.
        ml_model = make_regressor(ML_ONLY_FEATURES, seed=RANDOM_SEED)
        ml_model.fit(train[ML_ONLY_FEATURES], train["observed_yield_kg_ha"])
        ml_prediction = ml_model.predict(test[ML_ONLY_FEATURES])

        residual_model = make_regressor(RESIDUAL_FEATURES, seed=RANDOM_SEED)
        residual_model.fit(train[RESIDUAL_FEATURES], train["residual_kg_ha"])
        predicted_residual = residual_model.predict(test[RESIDUAL_FEATURES])
        dssat_prediction = test["dssat_yield_kg_ha"].to_numpy(dtype=float)
        hybrid_prediction = dssat_prediction + predicted_residual
        for index, (_, item) in enumerate(test.iterrows()):
            rows.append({
                "experiment_id": str(item["experiment_id"]),
                "treatment_id": int(item["treatment_id"]),
                "held_out_experiment": held_out,
                "fold_number": fold_number,
                "n_train": int(len(train)), "n_test": int(len(test)),
                "observed_yield_kg_ha": float(item["observed_yield_kg_ha"]),
                "dssat_prediction_kg_ha": float(dssat_prediction[index]),
                "ml_only_prediction_kg_ha": float(ml_prediction[index]),
                "predicted_residual_kg_ha": float(predicted_residual[index]),
                "hybrid_prediction_kg_ha": float(hybrid_prediction[index]),
            })
        fold_predictions = [r for r in rows if r["held_out_experiment"] == held_out]
        observed = [r["observed_yield_kg_ha"] for r in fold_predictions]
        fold_metrics = {name: regression_metrics(observed, [r[column] for r in fold_predictions])
                        for name, column in METHOD_COLUMNS.items()}
        folds.append({"fold_number": fold_number, "held_out_experiment": held_out,
                      "training_experiments": sorted(train["experiment_id"].astype(str).unique().tolist()),
                      "n_train": int(len(train)), "n_test": int(len(test)), "metrics": fold_metrics})

    predictions = pd.DataFrame(rows).sort_values(["held_out_experiment", "treatment_id"]).reset_index(drop=True)
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
    hybrid_beats_dssat = pooled["hybrid"]["mae_kg_ha"] < pooled["dssat_only"]["mae_kg_ha"]
    metrics = {
        "benchmark_type": "retrospective_historical_full_season_weather",
        "validation_method": "leave_one_experiment_out",
        "group_column": "experiment_id",
        "n_records": int(len(predictions)), "n_experiments": int(len(groups)),
        "folds": folds, "per_experiment": per_experiment, "pooled_oof": pooled,
        "pooled_hybrid_beats_dssat_on_mae": bool(hybrid_beats_dssat),
        "interpretation": "Historical full-season weather is used; these results are not evidence of sowing-time or in-season forecast performance.",
    }
    try:
        relative_dataset = dataset_path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        relative_dataset = dataset_path.name
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": relative_dataset,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "record_count": int(len(frame)), "experiment_groups": groups,
        "grouping_strategy": "LeaveOneGroupOut, with experiment_id as the held-out group; preprocessing fitted separately inside every training fold.",
        "random_seeds": {"ml_only_gradient_boosting": RANDOM_SEED, "residual_gradient_boosting": RANDOM_SEED},
        "models": {"ml_only": {"estimator": "HistGradientBoostingRegressor", "max_iter": 150,
                                 "features": ML_ONLY_FEATURES},
                   "residual": {"estimator": "HistGradientBoostingRegressor", "max_iter": 150,
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
