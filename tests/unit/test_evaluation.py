import json

import pandas as pd

from src.evaluation import evaluate_dataset, regression_metrics
from src.modeling import CATEGORICAL_FEATURES, ELIGIBLE_FEATURES, ML_ONLY_FEATURES


def _benchmark_frame():
    rows = []
    for group_index, experiment in enumerate(("EXP_A", "EXP_B", "EXP_C")):
        for treatment in range(1, 4):
            dssat = 2000 + group_index * 400 + treatment * 120
            observed = dssat + (group_index - 1) * 100 + treatment * 15
            row = {
                "experiment_id": experiment, "treatment_id": treatment,
                "cultivar": f"CULT_{group_index}", "weather_station": f"STN_{group_index}",
                "soil_id": f"SOIL_{group_index}", "dssat_yield_kg_ha": dssat,
                "observed_yield_kg_ha": observed, "residual_kg_ha": observed - dssat,
                "total_rainfall_mm": 280 + group_index * 20 + treatment,
                "mean_tmax_c": 21 + group_index, "mean_tmin_c": 4 + group_index,
                "mean_solar_radiation_mj_m2_day": 16, "rain_days": 45 + treatment,
                "crop_cycle_days": 210, "soil_depth_cm": 150,
                "soil_mean_slll_cm3_cm3": 0.10, "soil_mean_sdul_cm3_cm3": 0.20,
                "soil_mean_ssat_cm3_cm3": 0.35, "soil_mean_sbdm_g_cm3": 1.2,
                "soil_mean_sloc_pct": 0.8, "nitrogen_rate_kg_ha": treatment * 30,
                "irrigated": treatment % 2,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_leave_one_experiment_out_makes_one_prediction_per_record(tmp_path):
    dataset = tmp_path / "observations.csv"
    output = tmp_path / "evaluation"
    _benchmark_frame().to_csv(dataset, index=False)

    result = evaluate_dataset(dataset, output)
    predictions = pd.read_csv(output / "oof_predictions.csv")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))

    assert len(predictions) == 9
    assert predictions.groupby(["experiment_id", "treatment_id"]).size().eq(1).all()
    assert set(predictions["held_out_experiment"]) == {"EXP_A", "EXP_B", "EXP_C"}
    assert all(row["n_train"] + row["n_test"] == 9 for row in predictions.to_dict("records"))
    assert set(ML_ONLY_FEATURES) == set(ELIGIBLE_FEATURES) - {"dssat_yield_kg_ha"}
    assert not ({"observed_yield_kg_ha", "residual_kg_ha", "experiment_id", "treatment_id", "run_id"} & set(metadata["models"]["ml_only"]["features"]))
    assert len(metrics["folds"]) == 3
    for fold in metrics["folds"]:
        assert fold["held_out_experiment"] not in fold["training_experiments"]
        assert set(fold["training_experiments"]) == {"EXP_A", "EXP_B", "EXP_C"} - {fold["held_out_experiment"]}
    assert metrics["pooled_oof"]["dssat_only"]["mae_kg_ha"] == regression_metrics(
        predictions["observed_yield_kg_ha"], predictions["dssat_prediction_kg_ha"]
    )["mae_kg_ha"]
    assert metrics == result["metrics"]
    assert set(metadata["models"]["ml_only"]["features"]) == set(ML_ONLY_FEATURES)
    assert not set(CATEGORICAL_FEATURES) - set(metadata["models"]["ml_only"]["features"])


def test_metrics_report_nonpositive_yield_exclusion_and_undefined_r2():
    metrics = regression_metrics([0], [10])
    assert metrics["r2"] is None
    assert metrics["within_10_percent_denominator"] == 0
    assert metrics["excluded_nonpositive_observed_count"] == 1
    assert metrics["within_10_percent_pct"] is None
