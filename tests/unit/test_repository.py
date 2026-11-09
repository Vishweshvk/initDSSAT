from pathlib import Path

from types import SimpleNamespace

import pytest

from src.dssat.data.repository import (discover_wheat, validate_experiment,
                                      validate_observed_dataset, validate_simulation)
from src.dssat.exceptions import ExperimentValidationError


def test_discovery_links_wheat_files(tmp_path: Path):
    (tmp_path / "Wheat").mkdir()
    (tmp_path / "Weather").mkdir()
    (tmp_path / "Soil").mkdir()
    (tmp_path / "Genotype").mkdir()
    filex = tmp_path / "Wheat" / "ABCD0001.WHX"
    filex.write_text(
        "*SITE\n Test, ZZ\n*TREATMENTS\n@N R O C TNAME CU FL SA IC MP MI MF MR MC MT ME MH SM\n"
        " 1 1 0 0 control 1 1 0 1 1 0 1 0 0 0 0 0 1\n"
        "*CULTIVARS\n@C CR INGENO CNAME\n 1 WH IB0001 TESTVAR\n"
        "*FIELDS\n@L ID_FIELD WSTA.... FLSA FLOB FLDT FLDD FLDS FLST SLTX SLDP ID_SOIL FLNAME\n"
        " 1 ABCD0001 ABCD -99 0 DR000 0 0 00000 -99 100 SOIL000001 -99\n",
        encoding="latin-1",
    )
    (tmp_path / "Wheat" / "ABCD0001.WHA").write_text("@TRNO HWAM\n 1 1\n")
    (tmp_path / "Weather" / "ABCD8101.WTH").write_text("*WEATHER\n")
    (tmp_path / "Soil" / "SOIL.SOL").write_text("*SOIL000001 profile\n")
    (tmp_path / "Genotype" / "WHCER048.CUL").write_text("IB0001 TESTVAR 1 1\n")
    experiment = discover_wheat(tmp_path)[0]
    assert experiment.experiment_id == "ABCD0001"
    assert experiment.weather_station == "ABCD"
    assert experiment.soil_id == "SOIL000001"
    assert experiment.cultivar_name == "TESTVAR"
    assert experiment.treatments == ("1",)
    assert experiment.location == "Test, ZZ"
    assert experiment.treatment_details[1]["treatment_name"] == "control"
    assert experiment.cultivar_file.name == "WHCER048.CUL"
    try:
        validate_experiment(experiment)
    except Exception as exc:
        assert "weather file" not in str(exc)
        assert "soil file" not in str(exc)


def test_simulation_does_not_require_observed_yield_file(tmp_path: Path):
    files = {name: tmp_path / name for name in ("trial.WHX", "station.WTH", "profile.SOL", "cultivar.CUL")}
    for path in files.values():
        path.write_text("fixture", encoding="utf-8")
    without_observations = SimpleNamespace(
        experiment_id="TRIAL", experiment_file=files["trial.WHX"], observed_file=None,
        weather_file=files["station.WTH"], soil_file=files["profile.SOL"],
        cultivar_code="CUL1", cultivar_file=files["cultivar.CUL"], treatments=("1",),
    )
    assert validate_simulation(without_observations) == []
    with pytest.raises(ExperimentValidationError, match=r"no observed-yield \.WHA"):
        validate_observed_dataset(without_observations)
