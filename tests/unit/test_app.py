from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import app as webapp
from src.dssat.exceptions import DssatRunError


def _web_fixture(tmp_path, monkeypatch):
    input_dir = tmp_path / "DSSAT"
    input_dir.mkdir()
    experiment_file = input_dir / "TRIAL0001.WHX"
    weather_file = input_dir / "STN.WTH"
    soil_file = input_dir / "PROFILE.SOL"
    cultivar_file = input_dir / "WHCER048.CUL"
    for path in (experiment_file, weather_file, soil_file, cultivar_file):
        path.write_text("fixture", encoding="utf-8")
    experiment = SimpleNamespace(
        experiment_id="TRIAL0001", experiment_file=experiment_file, observed_file=None,
        weather_file=weather_file, soil_file=soil_file, cultivar_file=cultivar_file,
        cultivar_name="TEST CULTIVAR", cultivar_code="CUL1", weather_station="STN",
        soil_id="PROFILE1", location="Test site", treatments=("1",),
        treatment_details={1: {"treatment_name": "control"}},
    )
    config = SimpleNamespace(home=input_dir, executable=input_dir / "DSCSM048.EXE", data_dir=input_dir,
                             project_data_dir=tmp_path / "project-data", model_dir=tmp_path / "models",
                             version=lambda: "4.8.6.0")
    config.model_dir.mkdir()
    monkeypatch.setattr(webapp.DssatConfig, "discover", lambda: config)
    monkeypatch.setattr(webapp, "project_directories", lambda: (tmp_path, config.project_data_dir, config.model_dir))
    monkeypatch.setattr(webapp, "discover_wheat", lambda _data_dir: [experiment])
    monkeypatch.setattr(webapp, "get_experiment", lambda _data_dir, _experiment_id: experiment)
    result = {"run_id": "run-abc123", "run_dir": tmp_path / "run-abc123",
              "summary": pd.DataFrame([{"HWAM": 3000.0, "PDAT": 2020001, "HDAT": 2020200, "TRNO": 1}]),
              "growth": None}
    monkeypatch.setattr(webapp, "run_experiment", lambda *_args, **_kwargs: result)
    return config, experiment, result


def test_flask_pages_and_area_harvest_result(tmp_path, monkeypatch):
    _config, _experiment, _result = _web_fixture(tmp_path, monkeypatch)
    client = webapp.app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/evaluation").status_code == 200
    assert b"No benchmark artifacts" in client.get("/evaluation").data
    assert client.get("/research").status_code == 200
    assert b"Research direction" in client.get("/research").data

    response = client.post("/predict", data={"experiment": "TRIAL0001", "treatment": "1", "area": "2", "area_unit": "acres"})
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "DSSAT yield" in page
    assert "Estimated total harvest under the selected experimental conditions" in page
    assert "No trained residual model is available" in page
    assert "historical full-season weather" in page.lower()
    assert "run-abc123" in page


def test_flask_rejects_invalid_area_before_simulation(tmp_path, monkeypatch):
    _config, _experiment, _result = _web_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(webapp, "run_experiment", lambda *_args, **_kwargs: calls.append(True))
    response = webapp.app.test_client().post("/predict", data={
        "experiment": "TRIAL0001", "treatment": "1", "area": "inf", "area_unit": "hectares",
    })
    assert response.status_code == 400
    assert b"positive finite number" in response.data
    assert calls == []


def test_flask_does_not_hide_invalid_treatment_or_model_failure(tmp_path, monkeypatch):
    config, _experiment, _result = _web_fixture(tmp_path, monkeypatch)
    client = webapp.app.test_client()
    monkeypatch.setattr(webapp, "run_experiment", lambda *_args, **_kwargs: (_ for _ in ()).throw(DssatRunError("Treatment 9 is not listed.")))
    bad_treatment = client.post("/predict", data={"experiment": "TRIAL0001", "treatment": "9", "area": "1", "area_unit": "hectares"})
    assert bad_treatment.status_code == 400
    assert b"Treatment 9 is not listed" in bad_treatment.data

    monkeypatch.setattr(webapp, "run_experiment", lambda *_args, **_kwargs: _result)
    (config.model_dir / "residual_model.joblib").write_text("not deserialized by the test", encoding="utf-8")
    (config.model_dir / "model_metadata.json").write_text('{"features":["field"]}', encoding="utf-8")
    monkeypatch.setattr(webapp, "build_experiment_features", lambda *_args, **_kwargs: {"field": 1})
    monkeypatch.setattr(webapp, "predict_residual", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("test prediction error")))
    model_error = client.post("/predict", data={"experiment": "TRIAL0001", "treatment": "1", "area": "1", "area_unit": "hectares"})
    assert model_error.status_code == 200
    assert b"Hybrid calculation failed" in model_error.data
    assert b"not a hybrid prediction" in model_error.data
    assert b"Hybrid yield" not in model_error.data
