from pathlib import Path

import pandas as pd

from src.dssat.data.observations import calculate_residual, read_wheat_yield_observations
from src.dssat.parser import parse_summary
from src.features.weather_features import aggregate_weather
from src.features.soil_features import aggregate_soil_profile


def test_observation_yield_and_residual_sign(tmp_path: Path):
    observation_file = tmp_path / "fixture.WHA"
    observation_file.write_text(
        "*EXP. DATA (A)\n@TRNO HWAM HWUM\n 1 4321 30.0\n 2 5000 31.0\n",
        encoding="latin-1",
    )
    values = read_wheat_yield_observations(observation_file)
    assert values.to_dict("records") == [
        {"treatment_id": 1, "observed_yield_kg_ha": 4321.0},
        {"treatment_id": 2, "observed_yield_kg_ha": 5000.0},
    ]
    assert calculate_residual(4321, 4100) == 221
    assert calculate_residual(4100, 4321) == -221


def test_fixed_width_summary_parses_hwam(tmp_path: Path):
    summary = tmp_path / "Summary.OUT"
    summary.write_text(
        "*SUMMARY\n\n@ RUNNO TRTNO HWAM\n      1     2  4321\n",
        encoding="latin-1",
    )
    frame = parse_summary(summary)
    assert int(frame.iloc[0]["TRTNO"]) == 2
    assert float(frame.iloc[0]["HWAM"]) == 4321


def test_summary_csv_parses_hwam(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    summary.write_text("RUNNO,TRNO,HWAM\n1,3,5678\n", encoding="utf-8")
    frame = parse_summary(summary)
    assert int(frame.iloc[0]["TRNO"]) == 3
    assert float(frame.iloc[0]["HWAM"]) == 5678


def test_weather_aggregates_have_documented_values():
    weather = pd.DataFrame({
        "DATE": [20001, 20002], "SRAD": [10, 20], "TMAX": [10, 12],
        "TMIN": [0, 2], "RAIN": [0, 5],
    })
    features = aggregate_weather(weather, 2020001, 2020002)
    assert features["weather_days"] == 2
    assert features["crop_cycle_days"] == 2
    assert features["total_rainfall_mm"] == 5
    assert features["mean_tmax_c"] == 11
    assert features["mean_tmin_c"] == 1
    assert features["mean_solar_radiation_mj_m2_day"] == 15
    assert features["rain_days"] == 1


def test_soil_profile_uses_depth_weighted_layer_values(tmp_path: Path):
    profile = tmp_path / "SOIL.SOL"
    profile.write_text(
        "*TEST000001 profile\n@  SLB  SLLL  SDUL  SSAT  SBDM  SLOC\n"
        "    10  0.10  0.20  0.30  1.0  2.0\n"
        "    30  0.20  0.30  0.40  1.5  4.0\n",
        encoding="latin-1",
    )
    values = aggregate_soil_profile(profile, "TEST000001")
    assert values["soil_depth_cm"] == 30
    assert values["soil_mean_slll_cm3_cm3"] == 0.16666666666666666
    assert values["soil_mean_sbdm_g_cm3"] == 1.3333333333333333
