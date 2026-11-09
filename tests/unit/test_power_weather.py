from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.dssat.data import power_weather
from src.dssat.data.weather import read_weather


def _icasa_text(start: date, days: int = 365) -> str:
    rows = [
        "$WEATHER DATA: NASA POWER",
        "@ INSI LAT LONG ELEV TAV AMP REFHT WNDHT",
        " NASA 18.5204 73.8567 560 25.0 8.0 2.0 2.0",
        "@ DATE T2M TMIN TMAX TDEW RH2M RAIN WIND SRAD",
    ]
    for offset in range(days):
        current = start + timedelta(days=offset)
        doy = current.timetuple().tm_yday
        rows.append(f"{current.year}{doy:03d} 25 20 30 18 60 1 2 15")
    return "\n".join(rows) + "\n"


def _mock_urlopen(monkeypatch, text: str) -> None:
    @contextmanager
    def fake_urlopen(request, timeout):
        class Response:
            def read(self):
                return text.encode("utf-8")

        yield Response()

    monkeypatch.setattr(power_weather, "urlopen", fake_urlopen)


def test_fetch_writes_dssat_annual_files_and_reuses_cache(tmp_path, monkeypatch):
    planting = date(2020, 1, 1)
    _mock_urlopen(monkeypatch, _icasa_text(planting - timedelta(days=1), days=366))

    weather_path, metadata = power_weather.fetch_power_icasa(
        tmp_path, 18.5204, 73.8567, planting.isoformat(), "KSAS", "KSAS8101.WTH")

    assert weather_path.name == "KSAS8101.WTH"
    assert metadata["validated"] is True
    assert metadata["crop_window_coverage"] == 1.0
    parsed = read_weather(weather_path)
    assert len(parsed) == 366
    assert parsed.iloc[0]["DATE"] == 19365
    assert parsed.iloc[-1]["DATE"] == 20365
    assert weather_path.read_text(encoding="latin-1").startswith("*WEATHER DATA")
    assert list(parsed.columns) == ["DATE", "SRAD", "TMAX", "TMIN", "RAIN"]
    assert parsed.iloc[0]["SRAD"] == 15
    assert parsed.iloc[0]["TMAX"] == 30
    assert parsed.iloc[0]["TMIN"] == 20
    assert parsed.iloc[0]["RAIN"] == 1
    assert " KSAS " in weather_path.read_text(encoding="latin-1")

    def fail_if_requested(request, timeout):
        raise AssertionError("cached weather should avoid another API request")

    monkeypatch.setattr(power_weather, "urlopen", fail_if_requested)
    cached_path, cached_metadata = power_weather.fetch_power_icasa(
        tmp_path, 18.5204, 73.8567, planting.isoformat(), "KSAS", "KSAS8101.WTH")
    assert cached_path == weather_path
    assert cached_metadata["validated"] is True


def test_station_id_is_part_of_weather_cache_key(tmp_path, monkeypatch):
    planting = date(2020, 1, 1)
    _mock_urlopen(monkeypatch, _icasa_text(planting - timedelta(days=1), days=366))

    first, _ = power_weather.fetch_power_icasa(
        tmp_path, 18.5204, 73.8567, planting.isoformat(), "KSAS", "KSAS8101.WTH")
    second, _ = power_weather.fetch_power_icasa(
        tmp_path, 18.5204, 73.8567, planting.isoformat(), "PUNE", "PUNE2001.WTH")

    assert first.parent != second.parent
    assert second.name == "PUNE2001.WTH"
    assert " PUNE " in second.read_text(encoding="latin-1")


def test_incomplete_crop_window_is_rejected_and_recorded(tmp_path, monkeypatch):
    planting = date(2020, 1, 1)
    _mock_urlopen(monkeypatch, _icasa_text(planting - timedelta(days=1), days=365))

    with pytest.raises(ValueError, match="covers only"):
        power_weather.fetch_power_icasa(
            tmp_path, 18.5204, 73.8567, planting.isoformat(), "KSAS", "KSAS8101.WTH")

    metadata_files = list(Path(tmp_path).rglob("metadata.json"))
    assert len(metadata_files) == 1
    assert '"validated": false' in metadata_files[0].read_text(encoding="utf-8")
