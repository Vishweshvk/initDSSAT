"""Fetch, cache, and validate NASA POWER daily ICASA weather for DSSAT."""

from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .weather import read_weather


BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
REQUIRED_COLUMNS = ("SRAD", "TMAX", "TMIN", "RAIN")
MINIMUM_CROP_COVERAGE = 1.0
FORMAT_VERSION = 7
PREPLANT_WEATHER_DAYS = 1


def _date_to_yyddd(value: date) -> int:
    return value.year * 1000 + value.timetuple().tm_yday


def _cache_key(latitude: float, longitude: float, start: date, end: date,
               station_id: str) -> str:
    material = f"{latitude:.5f}|{longitude:.5f}|{start:%Y%m%d}|{end:%Y%m%d}|{station_id[:4].upper()}"
    return hashlib.sha256(material.encode("ascii")).hexdigest()[:20]


def _set_station_id(text: str, station_id: str) -> str:
    lines = text.splitlines()
    header_index = next((i for i, line in enumerate(lines)
                         if line.lstrip().upper().startswith("@ INSI")), None)
    if header_index is None:
        raise ValueError("NASA POWER ICASA response has no station metadata header.")
    row_index = next((i for i in range(header_index + 1, len(lines))
                      if lines[i].strip() and not lines[i].lstrip().startswith(("@", "!", "*"))), None)
    if row_index is None:
        raise ValueError("NASA POWER ICASA response has no station metadata row.")
    tokens = lines[row_index].split()
    if not tokens or tokens[0].upper() != "NASA":
        raise ValueError("NASA POWER ICASA station metadata row is malformed.")
    tokens[0] = station_id[:4]
    lines[row_index] = " " + " ".join(tokens)
    return "\n".join(lines) + "\n"


def _write_yearly_weather(text: str, cache_dir: Path, key: str, station_id: str) -> list[Path]:
    """Write DSSAT's station + two-digit-year annual files from ICASA rows."""
    lines = text.splitlines()
    header_index = next(i for i, line in enumerate(lines)
                        if line.startswith("@") and [part.upper() for part in line[1:].split()[:1]] == ["DATE"])
    headers = lines[:header_index + 1]
    annual_rows: dict[int, list[str]] = {}
    for line in lines[header_index + 1:]:
        tokens = line.split()
        if not tokens or not tokens[0].isdigit():
            continue
        date_value = int(tokens[0])
        year, day_of_year = divmod(date_value, 1000)
        if year < 1000 or not 1 <= day_of_year <= 366:
            raise ValueError(f"Invalid NASA POWER ICASA weather date: {tokens[0]}")
        tokens[0] = f"{year:04d}{day_of_year:03d}"
        annual_rows.setdefault(year, []).append(" ".join(tokens))
    outputs = []
    for year, rows in sorted(annual_rows.items()):
        # This cache directory is unique to the location and date window, so
        # the filename can follow DSSAT's station + year + sequence convention.
        output = cache_dir / f"{station_id[:4]}{year % 100:02d}01.WTH"
        output.write_text("\n".join(headers + rows) + "\n", encoding="latin-1", errors="replace")
        outputs.append(output)
    if not outputs:
        raise ValueError("NASA POWER ICASA response contains no annual daily records.")
    return outputs


def _write_experiment_weather(text: str, output: Path) -> Path:
    """Write classic DSSAT WTH dates under the exact filename referenced by FileX."""
    lines = text.splitlines()
    station_header = next((i for i, line in enumerate(lines)
                           if line.lstrip().upper().startswith("@ INSI")), None)
    date_header = next((i for i, line in enumerate(lines)
                        if line.startswith("@") and [p.upper() for p in line[1:].split()[:1]] == ["DATE"]), None)
    if station_header is None or date_header is None:
        raise ValueError("NASA POWER ICASA response is missing DSSAT station or date headers.")
    station_row = next((line.split() for line in lines[station_header + 1:date_header]
                        if line.strip() and not line.lstrip().startswith(("@", "!", "*"))), None)
    if not station_row or len(station_row) < 8:
        raise ValueError("NASA POWER ICASA response has incomplete station metadata.")
    source_columns = [part.upper() for part in lines[date_header][1:].split()]
    required_indexes = {}
    for name in REQUIRED_COLUMNS:
        if name not in source_columns:
            raise ValueError(f"NASA POWER ICASA response lacks {name}.")
        required_indexes[name] = source_columns.index(name)
    try:
        station_row_values = [
            station_row[0], float(station_row[1]), float(station_row[2]),
            float(station_row[3]), float(station_row[4]), float(station_row[5]),
            float(station_row[6]), float(station_row[7]),
        ]
    except (IndexError, ValueError) as exc:
        raise ValueError("NASA POWER ICASA station metadata values are malformed.") from exc
    station, latitude, longitude, elevation, tav, amp, refht, wndht = station_row_values
    output_lines = [
        "*WEATHER DATA : NASA POWER / ICASA",
        "! Source values from NASA POWER Daily API, Agroclimatology community.",
        "@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT",
        f"  {station:<4s} {latitude:8.3f} {longitude:8.3f} {elevation:5.0f} {tav:5.1f} {amp:5.1f} {refht:5.1f} {wndht:5.1f}",
        "@DATE  SRAD  TMAX  TMIN  RAIN",
    ]
    for line in lines[date_header + 1:]:
        tokens = line.split()
        if not tokens or not tokens[0].isdigit():
            continue
        year, day_of_year = divmod(int(tokens[0]), 1000)
        if year < 1000 or not 1 <= day_of_year <= 366:
            raise ValueError(f"Invalid NASA POWER ICASA weather date: {tokens[0]}")
        if len(tokens) <= max(required_indexes.values()):
            raise ValueError(f"NASA POWER ICASA weather row is incomplete for {tokens[0]}.")
        try:
            srad, tmax, tmin, rain = (float(tokens[required_indexes[name]])
                                       for name in ("SRAD", "TMAX", "TMIN", "RAIN"))
        except ValueError as exc:
            raise ValueError(f"NASA POWER ICASA weather values are malformed for {tokens[0]}.") from exc
        output_lines.append(
            f"{year % 100:02d}{day_of_year:03d}  {srad:5.1f} {tmax:5.1f} {tmin:5.1f} {rain:5.1f}")
    output.write_text("\n".join(output_lines) + "\n", encoding="latin-1", errors="replace")
    return output


def _validate_weather(path: Path, planting_date: date, simulation_start: date) -> dict:
    frame = read_weather(path)
    columns = {str(column).upper(): column for column in frame.columns}
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"Weather file lacks required DSSAT variables: {', '.join(missing)}")
    values = frame.copy()
    values["DATE"] = values[columns["DATE"]].astype(int)
    values = values.drop_duplicates("DATE").set_index("DATE")
    required_end = planting_date + timedelta(days=364)
    expected = [_date_to_yyddd(simulation_start + timedelta(days=offset))
                for offset in range((required_end - simulation_start).days + 1)]
    available = values.reindex(expected)
    valid = available[[columns[name] for name in REQUIRED_COLUMNS]].notna().all(axis=1)
    for name, lower, upper in (("SRAD", 0, 60), ("TMAX", -80, 65),
                               ("TMIN", -90, 60), ("RAIN", 0, 2000)):
        valid &= available[columns[name]].astype(float).between(lower, upper)
    valid &= available[columns["TMIN"]].astype(float) <= available[columns["TMAX"]].astype(float)
    coverage = float(valid.mean()) if len(expected) else 0.0
    if coverage < MINIMUM_CROP_COVERAGE:
        raise ValueError(f"NASA POWER weather covers only {coverage:.1%} of the simulation/crop window; at least {MINIMUM_CROP_COVERAGE:.0%} is required.")
    dates = values.index.to_list()
    return {
        "date_start": f"{min(dates)}", "date_end": f"{max(dates)}",
        "crop_window_start": planting_date.isoformat(),
        "crop_window_end": required_end.isoformat(),
        "simulation_start": simulation_start.isoformat(),
        "simulation_window_coverage": coverage,
        "crop_window_coverage": coverage,
        "daily_rows": int(len(values)), "parameters": list(REQUIRED_COLUMNS),
        "validation": "required daily weather values present for 100% of the pre-plant and 365-day crop window",
    }


def _validate_yearly_files(paths: list[Path], planting_date: date,
                           simulation_start: date) -> None:
    """Check the generated DSSAT year files still represent every required day."""
    observed = []
    for path in paths:
        frame = read_weather(path)
        columns = {str(column).upper(): column for column in frame.columns}
        missing = [name for name in (*REQUIRED_COLUMNS, "DATE") if name not in columns]
        if missing:
            raise ValueError(f"Generated DSSAT weather file {path.name} lacks {', '.join(missing)}.")
        dates = frame[columns["DATE"]].dropna().astype(int).tolist()
        if not dates:
            raise ValueError(f"Generated DSSAT weather file {path.name} contains no daily records.")
        observed.extend(dates)
    end = planting_date + timedelta(days=364)
    expected = [((simulation_start + timedelta(days=offset)).year % 100) * 1000
                + (simulation_start + timedelta(days=offset)).timetuple().tm_yday
                for offset in range((end - simulation_start).days + 1)]
    if sorted(observed) != sorted(expected):
        raise ValueError("Generated DSSAT weather file does not contain exactly the validated simulation and crop window.")


def fetch_power_icasa(project_data_dir: Path, latitude: float, longitude: float,
                      planting_date: str, station_id: str,
                      target_filename: str) -> tuple[Path, dict]:
    """Get historical ICASA weather; cache source response and validated WTH."""
    planting = date.fromisoformat(planting_date)
    simulation_start = planting - timedelta(days=PREPLANT_WEATHER_DAYS)
    end = planting + timedelta(days=364)
    if Path(target_filename).name != target_filename or not target_filename.upper().endswith(".WTH"):
        raise ValueError("DSSAT weather target must be a .WTH filename without a directory path.")
    key = _cache_key(latitude, longitude, planting, end, station_id)
    cache_dir = Path(project_data_dir) / "weather_cache" / "nasa_power" / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "response.icasa.txt"
    metadata_path = cache_dir / "metadata.json"

    if raw_path.is_file():
        try:
            cached_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            fresh = time.time() - raw_path.stat().st_mtime < 86400
            cached_output = cache_dir / cached_meta.get("weather_file", "")
            if (cached_output.is_file() and cached_meta.get("weather_file") == target_filename
                    and cached_meta.get("validated")
                    and cached_meta.get("format_version") == FORMAT_VERSION):
                return cached_output, cached_meta
            if not cached_meta.get("validated") and fresh:
                raise ValueError(cached_meta.get("validation_error", "Cached NASA POWER data did not pass coverage checks."))
            response_text = raw_path.read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            response_text = raw_path.read_text(encoding="utf-8")
    else:
        query = urlencode({
            "parameters": "T2M", "community": "AG", "longitude": f"{longitude:.5f}",
            "latitude": f"{latitude:.5f}", "start": simulation_start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"), "format": "ICASA", "time-standard": "LST",
        })
        request = Request(f"{BASE_URL}?{query}", headers={"User-Agent": "DSSAT-Field-Scenario/1.0"})
        try:
            with urlopen(request, timeout=40) as response:
                response_text = response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"NASA POWER weather request failed: {exc}") from exc
        if not response_text.lstrip().startswith(("$WEATHER DATA", "*WEATHER DATA")):
            raise RuntimeError("NASA POWER returned an error instead of an ICASA weather file.")
        raw_path.write_text(response_text, encoding="utf-8")

    try:
        wth_text = _set_station_id(response_text, station_id)
        validation_path = cache_dir / "validation.WTH"
        validation_path.write_text(wth_text, encoding="latin-1", errors="replace")
        validation = _validate_weather(validation_path, planting, simulation_start)
        validation_path.unlink(missing_ok=True)
        annual_paths = _write_yearly_weather(wth_text, cache_dir, key, station_id)
        output_path = _write_experiment_weather(wth_text, cache_dir / target_filename)
        _validate_yearly_files([output_path], planting, simulation_start)
    except (OSError, ValueError) as exc:
        for path in (*cache_dir.glob(f"{station_id[:4]}*.WTH"), cache_dir / target_filename):
            path.unlink(missing_ok=True)
        failure = {"source": "NASA POWER Daily API (AG/ICASA)", "validated": False,
                   "validation_error": str(exc), "latitude": latitude,
                   "longitude": longitude, "requested_start": simulation_start.isoformat(),
                   "requested_end": end.isoformat(),
                   "resolution": "NASA POWER global grid; variable-specific spatial resolution"}
        metadata_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise ValueError(str(exc)) from exc
    metadata = {
        "source": "NASA POWER Daily API", "community": "AG (Agroclimatology)",
        "format": "ICASA", "license": "NASA POWER data access; cite NASA POWER",
        "format_version": FORMAT_VERSION,
        "validated": True, "latitude": latitude, "longitude": longitude,
        "station_id": station_id[:4],
        "resolution": "NASA POWER global grid; variable-specific spatial resolution",
        "requested_start": simulation_start.isoformat(), "requested_end": end.isoformat(),
        "cached_response": raw_path.name, "weather_file": output_path.name,
        "annual_cache_files": [path.name for path in annual_paths],
        "weather_files": [output_path.name],
        **validation,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata
