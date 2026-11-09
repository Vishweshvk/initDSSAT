import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from cli import main as cli_main


def test_residual_training_saves_model_and_metrics(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "pairs.csv"
    rows = []
    for experiment, cultivar in (("A", "C1"), ("B", "C2"), ("C", "C3")):
        for treatment in range(1, 5):
            baseline = 2000 + treatment * 100
            observed = baseline + treatment * 20
            rows.append({"experiment_id": experiment, "treatment_id": treatment,
                         "observed_yield_kg_ha": observed, "dssat_yield_kg_ha": baseline,
                         "residual_kg_ha": observed - baseline, "cultivar": cultivar,
                         "weather_station": experiment, "soil_id": f"S{experiment}",
                         "total_rainfall_mm": 300 + treatment, "mean_tmax_c": 20 + treatment,
                         "mean_tmin_c": 5 + treatment, "mean_solar_radiation_mj_m2_day": 15,
                         "rain_days": 50, "crop_cycle_days": 200, "soil_depth_cm": 150,
                         "soil_mean_slll_cm3_cm3": 0.1, "soil_mean_sdul_cm3_cm3": 0.2,
                         "soil_mean_ssat_cm3_cm3": 0.3, "soil_mean_sbdm_g_cm3": 1.2,
                         "soil_mean_sloc_pct": 0.5, "nitrogen_rate_kg_ha": treatment * 10,
                         "irrigated": treatment % 2 == 0})
    pd.DataFrame(rows).to_csv(dataset, index=False)
    model_dir = tmp_path / "models"
    fake_config = SimpleNamespace(model_dir=model_dir, version=lambda: "4.8.6.0")
    monkeypatch.setattr(cli_main, "_config_or_report", lambda: fake_config)
    args = SimpleNamespace(dataset=str(dataset), crop="wheat")
    assert cli_main.command_train(args) == 0
    assert (model_dir / "residual_model.joblib").is_file()
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_record_count"] == 12
    assert metadata["fit_scope"].startswith("All dataset rows")
    assert "metrics" not in metadata
    assert "soil_mean_sloc_pct" in metadata["features"]
    assert not (model_dir / "model_metrics.json").exists()
