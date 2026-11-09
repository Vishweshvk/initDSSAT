import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from src.dssat.config import DssatConfig
from src.dssat.exceptions import DssatRunError
from src.dssat.runner import run_experiment


def _inputs(tmp_path: Path):
    home = tmp_path / "DSSAT"
    home.mkdir()
    executable = home / "DSCSM048.EXE"
    executable.write_bytes(b"trusted test fixture executable")
    (home / "DSSATPRO.v48").write_text(
        "*** DSSAT PROFILE ***\nWED C: \\DSSAT48\\Weather\nSLD C: \\DSSAT48\\Soil\nCRD C: \\DSSAT48\\Genotype\n",
        encoding="ascii",
    )
    experiment_file = tmp_path / "trial.WHX"
    weather = tmp_path / "station.WTH"
    soil = tmp_path / "profile.SOL"
    cultivar = tmp_path / "WHCER048.CUL"
    for path in (experiment_file, weather, soil, cultivar):
        path.write_text("fixture", encoding="utf-8")
    experiment = SimpleNamespace(
        experiment_id="TRIAL0001", experiment_file=experiment_file, observed_file=None,
        growth_observations_file=None, weather_file=weather, soil_file=soil,
        weather_station="STATION", cultivar_file=cultivar, cultivar_code="CUL1", treatments=("1",),
        treatment_details={1: {}},
    )
    config = DssatConfig(home, executable, home, tmp_path / "project-data", tmp_path / "models")
    return config, experiment, executable


def test_runner_archives_outputs_and_points_metadata_to_final_paths(tmp_path, monkeypatch):
    config, experiment, executable = _inputs(tmp_path)
    original_bytes = executable.read_bytes()

    def fake_run(command, cwd, **kwargs):
        Path(cwd, "summary.csv").write_text("RUNNO,TRNO,HWAM,PDAT,HDAT\n1,1,4200,2020001,2020100\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="completed", stderr="")

    monkeypatch.setattr("src.dssat.runner.subprocess.run", fake_run)
    result = run_experiment(config, experiment, treatment=1)
    assert result["run_id"] == result["run_dir"].name
    metadata = json.loads((result["run_dir"] / "metadata.json").read_text(encoding="utf-8"))
    assert float(result["summary"].iloc[0]["HWAM"]) == 4200
    assert metadata["output_files"] == ["output/summary.csv"]
    assert (result["run_dir"] / "output" / "summary.csv").is_file()
    assert metadata["log_files"] == ["logs/stdout.txt", "logs/stderr.txt"]
    assert not (result["run_dir"] / executable.name).exists()
    assert (result["run_dir"] / "input" / "Genotype" / "WHCER048.CUL").is_file()
    local_profile = (result["run_dir"] / "DSSATPRO.v48").read_text(encoding="latin-1")
    assert "CRD C: \\DSSAT48\\Genotype" in local_profile
    profile_overrides = json.loads((result["run_dir"] / "profile_overrides.json").read_text(encoding="utf-8"))
    assert set(profile_overrides) == {"WED", "SLD", "CRD"}
    assert executable.read_bytes() == original_bytes


def test_runner_retains_timeout_diagnostics(tmp_path, monkeypatch):
    config, experiment, _ = _inputs(tmp_path)

    def fake_timeout(command, cwd, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr("src.dssat.runner.subprocess.run", fake_timeout)
    with pytest.raises(DssatRunError, match="timed out"):
        run_experiment(config, experiment, treatment=1)
    run_dirs = list((config.project_data_dir / "runs").iterdir())
    run_dir = run_dirs[0]
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["success"] is False
    assert metadata["error_type"] == "TimeoutExpired"
    assert "partial stdout" in (run_dir / "logs" / "stdout.txt").read_text(encoding="utf-8")
    assert "timed out" in (run_dir / "logs" / "stderr.txt").read_text(encoding="utf-8")
