"""Run one DSSAT experiment in an isolated scratch workspace and archive it."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

from .exceptions import DssatRunError
from .parser import parse_plantgro, parse_summary
from .data.repository import dssat_module_for_crop, validate_simulation


def _short_path(path: Path) -> Path:
    """Use Windows 8.3 paths when available to preserve DSSAT profile widths."""
    if os.name != "nt":
        return path
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    return Path(buffer.value) if length and length < len(buffer) else path


def _rewrite_profile(source: Path, destination: Path, paths: dict[str, Path]) -> None:
    lines = source.read_text(encoding="latin-1", errors="replace").splitlines()
    found = set()
    for index, line in enumerate(lines):
        key = line[:3].strip().upper()
        if key in paths:
            target = _short_path(paths[key]).resolve()
            drive, tail = os.path.splitdrive(str(target))
            if not drive:
                raise DssatRunError(f"DSSAT profile paths must be on a drive: {target}")
            # DSSAT's profile convention is "KEY C: \directory".
            lines[index] = f"{key} {drive} {tail.replace('/', chr(92))}"
            found.add(key)
    missing = set(paths) - found
    if missing:
        raise DssatRunError(f"DSSAT profile is missing path entries: {', '.join(sorted(missing))}")
    destination.write_text("\n".join(lines) + "\n", encoding="latin-1")


def _copy_required_inputs(experiment, input_dir: Path, weather_dir: Path, soil_dir: Path, genotype_dir: Path) -> list[str]:
    validate_simulation(experiment)
    input_dir.mkdir(parents=True, exist_ok=True)
    for directory in (weather_dir, soil_dir, genotype_dir):
        directory.mkdir(parents=True, exist_ok=True)
    staged = []
    copies = (
        (experiment.experiment_file, input_dir),
        (experiment.weather_file, weather_dir),
        (experiment.soil_file, soil_dir),
        (experiment.cultivar_file, genotype_dir),
    )
    for source, target_dir in copies:
        assert source is not None
        shutil.copy2(source, target_dir / source.name)
        staged.append(source.name)
    # Several DSSAT experiments need same-stem sidecar files in addition to
    # FileX (for example MOW/BR* forage inputs). Preserve these in the input
    # folder so the crop model can resolve every file referenced by FileX.
    for source in experiment.experiment_file.parent.glob(f"{experiment.experiment_file.stem}.*"):
        if source.is_file() and source.resolve() != experiment.experiment_file.resolve():
            shutil.copy2(source, input_dir / source.name)
            if source.name not in staged:
                staged.append(source.name)
    return staged


def _weather_stations_from_experiment(experiment) -> set[str]:
    """Find weather stations used by every field record in FileX sidecars."""
    stations = {str(experiment.weather_station).strip().upper()}
    sources = [experiment.experiment_file]
    sources.extend(experiment.experiment_file.parent.glob(f"{experiment.experiment_file.stem}.*"))
    for source in dict.fromkeys(sources):
        try:
            lines = source.read_text(encoding="latin-1", errors="replace").splitlines()
        except OSError:
            continue
        in_fields = False
        for line in lines:
            if line.startswith("*"):
                in_fields = False
            if line.upper().startswith("@L ID_FIELD") and "WSTA" in line.upper():
                in_fields = True
                continue
            if in_fields:
                tokens = line.split()
                if len(tokens) >= 3 and tokens[0].isdigit():
                    station = tokens[2].upper()
                    if station not in {"-99", "-99.0"}:
                        stations.add(station)
    return stations


def _cultivar_tables_from_experiment(experiment, genotype_dir: Path) -> set[Path]:
    """Find cultivar tables required by every crop record in FileX sidecars."""
    sources = [experiment.experiment_file]
    sources.extend(experiment.experiment_file.parent.glob(f"{experiment.experiment_file.stem}.*"))
    requested: set[tuple[str, str]] = set()
    for source in dict.fromkeys(sources):
        try:
            lines = source.read_text(encoding="latin-1", errors="replace").splitlines()
        except OSError:
            continue
        cultivar_header: list[str] = []
        in_cultivars = False
        for line in lines:
            if line.startswith("*"):
                in_cultivars = line.upper().startswith("*CULTIVARS")
                cultivar_header = []
                continue
            if not in_cultivars:
                continue
            if line.startswith("@"):
                cultivar_header = line[1:].split()
                if cultivar_header and cultivar_header[0].upper() == "C":
                    cultivar_header = cultivar_header[1:]
                continue
            tokens = line.split()
            if not tokens or not tokens[0].isdigit() or not cultivar_header:
                continue
            row = dict(zip(cultivar_header, tokens[1:]))
            crop_code, cultivar_code = row.get("CR"), row.get("INGENO")
            if crop_code and cultivar_code:
                requested.add((crop_code.upper(), cultivar_code.upper()))

    tables: set[Path] = set()
    for crop_code, cultivar_code in requested:
        module = dssat_module_for_crop(genotype_dir.parent, crop_code)
        expected = genotype_dir / f"{crop_code}{module[2:]}.CUL" if module else None
        candidates = [expected] if expected and expected.is_file() else []
        if not candidates:
            candidates = [path for path in genotype_dir.glob("*.CUL")
                          if any(line.strip().split()[:1] == [cultivar_code]
                                 for line in path.read_text(encoding="latin-1", errors="replace").splitlines())]
        for cultivar_file in candidates:
            if not any(line.strip().split()[:1] == [cultivar_code]
                       for line in cultivar_file.read_text(encoding="latin-1", errors="replace").splitlines()):
                continue
            tables.update(cultivar_file.parent.glob(f"{cultivar_file.stem}.*"))
    return {path for path in tables if path.is_file()}


def _yyddd(iso_date: str) -> str:
    parsed = datetime.strptime(iso_date, "%Y-%m-%d").date()
    return f"{parsed.year % 100:02d}{parsed.timetuple().tm_yday:03d}"


def _apply_scenario_to_filex(path: Path, treatment: int, scenario: dict) -> dict:
    """Apply common FileX values and return what the selected format accepted."""
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    planting_date = _yyddd(scenario["planting_date"])
    simulation_start_date = _yyddd(scenario.get("weather_simulation_start_date", scenario["planting_date"]))
    density_per_m2 = float(scenario["plant_density_ha"]) / 10000.0
    section = ""
    headers: dict[str, int] = {}
    fertilizer_header: dict[str, int] = {}
    fertilizer_rows: list[tuple[int, list[str]]] = []
    irrigation_header_index = None
    irrigation_rows = []
    application = {
        "planting_date": {"status": "not_applied", "reason": "FileX has no PDATE field."},
        "simulation_start": {"status": "not_applied", "reason": "FileX has no SDATE field."},
        "initial_conditions_date": {"status": "not_applied", "reason": "FileX has no ICDAT field."},
        "plant_density": {"status": "not_applied", "reason": "FileX has no PPOP/PPOE field."},
        "nitrogen": {"status": "not_applied", "reason": "FileX has no inorganic fertilizer events."},
        "irrigation": {"status": "not_applied", "reason": "FileX has no dated irrigation event section."},
    }
    planting_rows = 0
    density_rows = 0
    simulation_start_rows = 0
    initial_condition_rows = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("*"):
            section = stripped.upper()
            headers = {}
            continue
        if stripped.startswith("@"):
            tokens = stripped.split()
            headers = {name.upper(): position for position, name in enumerate(tokens[1:], start=1)}
            if section.startswith("*FERTILIZERS (INORGANIC)") and "FAMN" in headers:
                fertilizer_header = headers.copy()
            if section.startswith("*IRRIGATION AND WATER MANAGEMENT") and "IDATE" in headers:
                irrigation_header_index = index
            continue
        tokens = stripped.split()
        if not tokens or not tokens[0].isdigit():
            continue
        if section.startswith("*PLANTING DETAILS") and "PDATE" in headers:
            row = tokens
            position = headers.get("PDATE")
            if position is not None and position < len(row):
                row[position] = planting_date
                planting_rows += 1
            for key in ("PPOP", "PPOE"):
                position = headers.get(key)
                if position is not None and position < len(row):
                    row[position] = f"{density_per_m2:.4f}"
                    density_rows += 1
            lines[index] = " " + " ".join(row)
        elif section.startswith("*SIMULATION CONTROLS") and "SDATE" in headers:
            row = tokens
            position = headers["SDATE"]
            if position < len(row):
                row[position] = simulation_start_date
                lines[index] = " " + " ".join(row)
                simulation_start_rows += 1
        elif section.startswith("*INITIAL CONDITIONS") and "ICDAT" in headers:
            row = tokens
            position = headers["ICDAT"]
            if position < len(row):
                row[position] = simulation_start_date
                lines[index] = " " + " ".join(row)
                initial_condition_rows += 1
        elif section.startswith("*FERTILIZERS (INORGANIC)") and "FAMN" in headers:
            fertilizer_rows.append((index, tokens))
        elif section.startswith("*IRRIGATION AND WATER MANAGEMENT") and "IDATE" in headers:
            irrigation_rows.append(index)
    if fertilizer_rows:
        total_n = float(scenario["nitrogen_rate_kg_ha"])
        split = scenario["nitrogen_timing"] == "split"
        if split and len(fertilizer_rows) < 2:
            application["nitrogen"] = {"status": "recorded_not_used", "reason": "Split timing needs at least two fertilizer events in this FileX."}
        else:
            event_count = 2 if split else 1
            for event_index, (line_index, row) in enumerate(fertilizer_rows):
                amount = total_n / event_count if split and event_index < event_count else (total_n if not split and event_index == 0 else 0.0)
                if "FDATE" in fertilizer_header:
                    row[fertilizer_header["FDATE"]] = planting_date
                if split and event_index == 1 and "FDATE" in fertilizer_header:
                    split_date = datetime.strptime(scenario["planting_date"], "%Y-%m-%d") + timedelta(days=30)
                    row[fertilizer_header["FDATE"]] = _yyddd(split_date.date().isoformat())
                row[fertilizer_header["FAMN"]] = f"{amount:.2f}"
                lines[line_index] = " " + " ".join(row)
            application["nitrogen"] = {"status": "applied", "event_count": event_count,
                                       "amount_kg_n_ha": total_n, "timing": scenario["nitrogen_timing"]}
    # Replace DSSAT's dated irrigation schedule with the requested regular
    # schedule. DSSAT stops processing events when its simulated crop ends.
    for line_index in reversed(irrigation_rows):
        del lines[line_index]
    if scenario["irrigation"] == "irrigated" and irrigation_header_index is not None:
        start = datetime.strptime(scenario["planting_date"], "%Y-%m-%d").date()
        event_lines = []
        for offset in range(int(scenario["irrigation_interval_days"]), 241,
                            int(scenario["irrigation_interval_days"])):
            event_date = start + timedelta(days=offset)
            event_lines.append(f" 1 {_yyddd(event_date.isoformat())} IR001 {float(scenario['irrigation_amount_mm']):.2f}")
        insert_at = irrigation_header_index + 1
        lines[insert_at:insert_at] = event_lines
        application["irrigation"] = {"status": "applied", "interval_days": int(scenario["irrigation_interval_days"]),
                                      "amount_mm_per_event": float(scenario["irrigation_amount_mm"]),
                                      "event_count": len(event_lines)}
    elif scenario["irrigation"] == "rainfed" and irrigation_header_index is not None:
        application["irrigation"] = {"status": "applied", "mode": "rainfed", "event_count": 0}
    if planting_rows:
        application["planting_date"] = {"status": "applied", "date": scenario["planting_date"], "rows_updated": planting_rows}
        if simulation_start_rows:
            application["simulation_start"] = {
                "status": "applied",
                "date": scenario.get("weather_simulation_start_date", scenario["planting_date"]),
                "reason": "The run starts one day before planting so DSSAT has pre-plant weather for initialization.",
            }
        if initial_condition_rows:
            application["initial_conditions_date"] = {
                "status": "applied",
                "date": scenario.get("weather_simulation_start_date", scenario["planting_date"]),
                "reason": "The installed initial soil/crop state was carried forward to the selected simulation start.",
            }
    if density_rows:
        application["plant_density"] = {"status": "applied", "plants_per_hectare": float(scenario["plant_density_ha"]),
                                         "filex_value_plants_per_m2": density_per_m2, "fields_updated": density_rows}
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return application


def _apply_soil_profile_override(path: Path, soil_id: str) -> int:
    """Set the DSSAT field records to a selected installed soil profile ID."""
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    section = ""
    headers: dict[str, int] = {}
    updated = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("*"):
            section = stripped.upper()
            headers = {}
        elif stripped.startswith("@") and section.startswith("*FIELDS"):
            tokens = stripped.split()
            headers = {name.upper(): position for position, name in enumerate(tokens[1:], start=1)}
        elif (section.startswith("*FIELDS") and "ID_SOIL" in headers
              and stripped and stripped.split()[0].isdigit()):
            row = stripped.split()
            position = headers["ID_SOIL"]
            if position < len(row):
                row[position] = soil_id
                lines[index] = " " + " ".join(row)
                updated += 1
    if updated:
        path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return updated


def run_experiment(config, experiment, treatment: int, timeout: int = 180,
                   scenario: dict | None = None, yield_column: str | None = "HWAM") -> dict:
    """Execute DSSAT without changing its installation; persist logs and outputs."""
    validate_simulation(experiment)
    if str(treatment) not in experiment.treatments:
        raise DssatRunError(f"Treatment {treatment} is not available for {experiment.experiment_id}.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    run_dir = Path(config.project_data_dir) / "runs" / run_id
    archived_input = run_dir / "input"
    archived_output = run_dir / "output"
    logs_dir = run_dir / "logs"
    for directory in (archived_input, archived_output, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # DSSAT's profile reader uses fixed-width path records. Choose a writable,
    # short ancestor so staged paths remain inside the profile's supported line width.
    candidates = [parent for parent in Path(config.project_data_dir).parents if len(str(parent)) <= 35]
    scratch_parent = next((parent for parent in candidates if parent.is_dir()), Path(config.project_data_dir))
    scratch_parent.mkdir(parents=True, exist_ok=True)
    # Keep the directory name to two characters; DSSAT builds some absolute
    # input filenames in a 64-character Fortran buffer.
    scratch = scratch_parent / f"d{uuid.uuid4().hex[0]}"
    while scratch.exists():
        scratch = scratch_parent / f"d{uuid.uuid4().hex[0]}"
    scratch.mkdir()
    scratch = _short_path(scratch)
    staged = _copy_required_inputs(
        experiment,
        scratch / "input",
        scratch / "Weather",
        scratch / "Soil",
        scratch / "Genotype",
    )
    _copy_required_inputs(
        experiment,
        archived_input,
        archived_input / "Weather",
        archived_input / "Soil",
        archived_input / "Genotype",
    )
    scenario_application = {}
    if scenario:
        scenario_application = _apply_scenario_to_filex(
            scratch / "input" / experiment.experiment_file.name, treatment, scenario)
        _apply_scenario_to_filex(archived_input / experiment.experiment_file.name, treatment, scenario)
    if scenario and scenario.get("soil_override_path"):
        soil_source = Path(scenario["soil_override_path"])
        if not soil_source.is_file():
            raise DssatRunError(f"Selected soil profile file is missing: {soil_source}")
        profile_id = str(scenario["soil_override_id"]).strip()
        for destination in (scratch / "Soil" / soil_source.name,
                            archived_input / "Soil" / soil_source.name):
            shutil.copy2(soil_source, destination)
        updated_scratch = _apply_soil_profile_override(
            scratch / "input" / experiment.experiment_file.name, profile_id)
        updated_archive = _apply_soil_profile_override(
            archived_input / experiment.experiment_file.name, profile_id)
        if not updated_scratch or not updated_archive:
            raise DssatRunError("The selected soil profile was staged, but this FileX has no editable ID_SOIL field record.")
        staged.append(soil_source.name)
        scenario_application["soil"] = {
            "status": "applied", "source": scenario.get("soil_source", "Installed DSSAT soil profile library"),
            "profile_id": profile_id, "file": soil_source.name,
            "latitude": scenario.get("soil_latitude"), "longitude": scenario.get("soil_longitude"),
            "distance_km": scenario.get("soil_distance_km"),
            "matching_limit_km": scenario.get("soil_matching_limit_km"),
            "profile_depth_cm": scenario.get("soil_depth_cm"),
            "texture_code": scenario.get("soil_texture_code"),
            "rows_updated": updated_archive,
        }
    # Cultivar records reference matching ECO and SPE tables. Stage all files
    # sharing the cultivar table stem (for wheat, WHCER048.*).
    assert experiment.cultivar_file is not None
    for source in experiment.cultivar_file.parent.glob(f"{experiment.cultivar_file.stem}.*"):
        if source.is_file():
            shutil.copy2(source, scratch / "Genotype" / source.name)
            shutil.copy2(source, archived_input / "Genotype" / source.name)
            if source.name not in staged:
                staged.append(source.name)
    for source in _cultivar_tables_from_experiment(experiment, experiment.cultivar_file.parent):
        shutil.copy2(source, scratch / "Genotype" / source.name)
        shutil.copy2(source, archived_input / "Genotype" / source.name)
        if source.name not in staged:
            staged.append(source.name)
    # DSSAT may span a calendar year and request the next annual file for the
    # same station, so include every station-year weather file in the run.
    assert experiment.weather_file is not None
    for station_prefix in _weather_stations_from_experiment(experiment):
        for source in experiment.weather_file.parent.glob(f"{station_prefix}*.WTH"):
            if source.is_file():
                shutil.copy2(source, scratch / "Weather" / source.name)
                shutil.copy2(source, archived_input / "Weather" / source.name)
                if source.name not in staged:
                    staged.append(source.name)
    if scenario and scenario.get("weather_override_path"):
        override = Path(scenario["weather_override_path"])
        if not override.is_file():
            raise DssatRunError(f"Selected weather override is missing: {override}")
        # Drop copied station years from this run only, so the requested NASA
        # records cannot collide with a different installed record for the same dates.
        for station_prefix in _weather_stations_from_experiment(experiment):
            for target_dir in (scratch / "Weather", archived_input / "Weather"):
                for old_weather in target_dir.glob(f"{station_prefix}*.WTH"):
                    old_weather.unlink(missing_ok=True)
        weather_filename = f"{experiment.experiment_id}.WTH"
        shutil.copy2(override, scratch / "Weather" / weather_filename)
        shutil.copy2(override, archived_input / "Weather" / weather_filename)
        staged.append(weather_filename)
        scenario_application["weather"] = {
            "status": "applied", "source": "NASA POWER Daily API (ICASA)",
            "weather_files": [weather_filename],
            "source_metadata": scenario.get("weather_source_metadata", {}),
        }

    profile_source = Path(config.home) / "DSSATPRO.v48"
    if not profile_source.is_file():
        shutil.rmtree(scratch, ignore_errors=True)
        raise DssatRunError(f"DSSAT profile was not found: {profile_source}")
    profile = scratch / "DSSATPRO.v48"
    profile_paths = {
        "WED": scratch / "Weather",
        "SLD": scratch / "Soil",
        "CRD": scratch / "Genotype",
    }
    _rewrite_profile(profile_source, profile, profile_paths)
    shutil.copy2(profile_source, run_dir / "DSSATPRO.v48")
    (run_dir / "profile_overrides.json").write_text(
        json.dumps({key: str(value) for key, value in profile_paths.items()}, indent=2),
        encoding="utf-8",
    )

    batch_name = "DSSBatch.v48"
    batch_text = (
        "$BATCH@FILEX                                                                                        TRTNO     RP     SQ     OP     CO\n"
        f".\\input\\{experiment.experiment_file.name:<86}{int(treatment):>5}      1      0      0      0\n"
    )
    (scratch / batch_name).write_text(batch_text, encoding="ascii")
    (run_dir / batch_name).write_text(batch_text, encoding="ascii")
    command = [str(config.executable), getattr(experiment, "dssat_module", "CSCER048") or "CSCER048", "B", batch_name]
    started = time.monotonic()
    stdout = ""
    stderr = ""
    return_code = None
    error = None
    error_type = None
    try:
        completed = subprocess.run(command, cwd=scratch, capture_output=True, text=True, timeout=timeout, check=False)
        stdout, stderr, return_code = completed.stdout or "", completed.stderr or "", completed.returncode
        if return_code != 0:
            error = f"DSSAT exited with status {return_code}."
            error_type = "DssatNonzeroExit"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        error = f"DSSAT timed out after {timeout} seconds."
        error_type = type(exc).__name__
    except OSError as exc:
        error, error_type = str(exc), type(exc).__name__

    (logs_dir / "stdout.txt").write_text(stdout, encoding="utf-8", errors="replace")
    diagnostic_stderr = stderr
    if error and error not in diagnostic_stderr:
        diagnostic_stderr = f"{diagnostic_stderr.rstrip()}\n{error}".strip()
    (logs_dir / "stderr.txt").write_text(diagnostic_stderr, encoding="utf-8", errors="replace")
    output_names = []
    for path in scratch.rglob("*"):
        if path.is_file() and path.suffix.upper() in {".OUT", ".CSV", ".ERR", ".LST", ".INP", ".INH", ".OSU"}:
            relative = path.relative_to(scratch)
            destination = archived_output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            output_names.append(f"output/{relative.as_posix()}")

    archived_outputs = {path.name.upper(): path for path in archived_output.rglob("*") if path.is_file()}
    summary_path = next((archived_outputs[name.upper()] for name in
                         ("Summary.OUT", "summary.csv", "PlantSum.OUT", f"{experiment.experiment_id}.OSU")
                         if name.upper() in archived_outputs), None)
    growth_path = next((archived_outputs[name.upper()] for name in ("PlantGro.OUT", "plantgro.csv")
                        if name.upper() in archived_outputs), None)
    summary = growth = None
    if error is None and summary_path is not None:
        try:
            summary = parse_summary(summary_path)
            if growth_path:
                growth = parse_plantgro(growth_path)
        except Exception as exc:
            error, error_type = str(exc), type(exc).__name__
    if error is None and summary is None:
        error = "DSSAT did not produce Summary.OUT."
        error_type = "MissingSummaryOutput"
    if error is None and summary is not None and yield_column is not None:
        try:
            yield_values = summary[yield_column].astype(float)
            if yield_values.empty or not yield_values.map(lambda value: value > 0 and value < float("inf")).all():
                error = f"DSSAT produced no valid positive {yield_column} yield; inspect WARNING.OUT and ERROR.OUT."
                error_type = "InvalidYieldOutput"
        except (KeyError, TypeError, ValueError):
            error = f"DSSAT output {yield_column} yield could not be validated."
            error_type = "InvalidYieldOutput"

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment_id": experiment.experiment_id,
        "treatment_id": int(treatment),
        "dssat_version": config.version(),
        "command": command,
        "working_directory": "DSSAT scratch directory (removed after run)",
        "runtime_seconds": round(time.monotonic() - started, 3),
        "input_files": [
            f"input/{path.relative_to(archived_input).as_posix()}"
            for path in sorted(archived_input.rglob("*")) if path.is_file()
        ],
        "output_files": output_names,
        "log_files": ["logs/stdout.txt", "logs/stderr.txt"],
        "return_code": return_code,
        "success": error is None,
        "yield_validation_column": yield_column,
        "error": error,
        "error_type": error_type,
        "staged_input_names": staged,
        "scenario_application": scenario_application,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.rmtree(scratch, ignore_errors=True)
    if error:
        raise DssatRunError(f"DSSAT run failed for {experiment.experiment_id}: {error} See {run_dir} for diagnostics.")
    return {"run_id": run_id, "run_dir": run_dir, "summary": summary, "growth": growth,
            "metadata": metadata, "scenario_application": scenario_application}
