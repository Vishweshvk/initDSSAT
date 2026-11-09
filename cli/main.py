"""CLI for multi-crop DSSAT experiments and paired-yield residual modeling."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import shutil
import sys

import pandas as pd

from src.dssat.config import DssatConfig
from src.dssat.data.linking import build_manifest
from src.dssat.data.observations import (calculate_residual, read_dssat_a_file,
                                         read_observed_yield_values,
                                         observed_yield_target,
                                         read_wheat_yield_observations)
from src.dssat.data.repository import (discover_experiments, get_experiment,
                                       validate_observed_dataset, validate_simulation)
from src.dssat.exceptions import DssatError
from src.dssat.runner import run_experiment
from src.dssat.parser import parse_summary
from src.evaluation import evaluate_dataset
from src.modeling import RANDOM_SEED, RESIDUAL_FEATURES, make_regressor, validate_training_frame
from src.prediction import area_estimate, build_experiment_features, predict_residual


def _config_or_report() -> DssatConfig:
    return DssatConfig.discover()


def _dataset_path(config: DssatConfig, argument: str | None = None,
                  crop: str = "wheat") -> Path:
    default_name = ("dssat_hybrid_dataset.csv" if str(crop).lower() == "all"
                    else f"{str(crop).lower()}_hybrid_dataset.csv")
    path = Path(argument).expanduser() if argument else config.project_data_dir / "processed" / default_name
    if argument and not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archived_run_index(config) -> dict:
    """Index recent successful runs so dataset rebuilds can reuse exact pairs."""
    index = {}
    runs_dir = config.project_data_dir / "runs"
    if not runs_dir.is_dir():
        return index
    recent_cutoff = datetime.now().timestamp() - 24 * 60 * 60
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        try:
            if run_dir.stat().st_mtime < recent_cutoff:
                continue
        except OSError:
            continue
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            treatment = int(metadata.get("treatment_id", -1))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if (metadata.get("success") and not metadata.get("scenario_application")
                and metadata.get("yield_validation_column") == "HWAM"):
            index.setdefault((str(metadata.get("experiment_id", "")).upper(), treatment), []).append(
                (run_dir, metadata))
    return index


def _reuse_archived_run(config, experiment, treatment: int, run_index: dict):
    """Reuse a prior run only when staged experiment, weather, soil and cultivar bytes match."""
    key = (experiment.experiment_id.upper(), int(treatment))
    for run_dir, metadata in run_index.get(key, []):
        staged = set(metadata.get("staged_input_names", []))
        expected = (experiment.experiment_file, experiment.weather_file,
                    experiment.soil_file, experiment.cultivar_file)
        if any(path is None or path.name not in staged for path in expected):
            continue
        input_dir = run_dir / "input"
        matches = True
        for source in expected:
            archived = next((path for path in input_dir.rglob(source.name) if path.is_file()), None)
            if archived is None or _file_sha256(source) != _file_sha256(archived):
                matches = False
                break
        summary_path = run_dir / "output" / "Summary.OUT"
        if not matches or not summary_path.is_file():
            continue
        try:
            summary = parse_summary(summary_path)
        except Exception:
            continue
        return {"summary": summary, "growth": None, "run_id": metadata.get("run_id", run_dir.name),
                "run_dir": run_dir}
    return None


def command_doctor(_args) -> int:
    try:
        config = _config_or_report()
    except DssatError as exc:
        print("[OK] Python", sys.version.split()[0])
        print("[FAIL] DSSAT executable")
        print(f"Configuration: {exc}")
        return 1
    experiments = discover_experiments(config.data_dir, profile_path=config.home / "DSSATPRO.v48")
    print(f"[OK] Python {sys.version.split()[0]}")
    print(f"[OK] DSSAT executable: {config.executable}")
    print(f"[OK] DSSAT version: {config.version()}")
    print(f"[OK] DSSAT data root: {config.data_dir}")
    print(f"[{'OK' if experiments else 'FAIL'}] Supported crop experiments discovered: {len(experiments)}")
    print(f"[OK] Project data directory: {config.project_data_dir}")
    print(f"[OK] Model directory: {config.model_dir}")
    failed = False
    for package in ("pandas", "sklearn", "flask", "joblib"):
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package} (install requirements.txt)")
            failed = True
    return 0 if experiments and not failed else 1


def command_list(args) -> int:
    config = _config_or_report()
    experiments = discover_experiments(config.data_dir,
                                       None if args.crop.lower() == "all" else args.crop,
                                       profile_path=config.home / "DSSATPRO.v48")
    if not experiments:
        print(f"No supported FileX experiments found for crop '{args.crop}' under {config.data_dir}")
        return 1
    for exp in experiments:
        try:
            validate_simulation(exp)
            status = "SIMULATION READY"
        except DssatError as exc:
            status = f"SIMULATION INVALID: {exc}"
        observed_status = "observed-yield data available" if exp.observed_file else "no .WHA (not eligible for observed-yield dataset)"
        print(f"{exp.experiment_id}\t{status}\t{observed_status}\tweather={exp.weather_station}\tsoil={exp.soil_id}\tcultivar={exp.cultivar_name}")
    return 0


def command_audit_crops(args) -> int:
    """Inventory each installed crop and optionally run one DSSAT probe per crop."""
    config = _config_or_report()
    ignored = {"weather", "soil", "genotype", "pest", "standarddata", "documentation",
               "economic", "help", "images", "climatechange", "seasonal", "sequence",
               "skins", "spatial", "tools", "yieldforecast"}
    crop_folders = [folder for folder in config.data_dir.iterdir()
                    if folder.is_dir() and folder.name.casefold() not in ignored]
    experiments = discover_experiments(config.data_dir,
                                       profile_path=config.home / "DSSATPRO.v48")
    rows = []
    for folder in sorted(crop_folders, key=lambda item: item.name.casefold()):
        raw_files = sorted(path for path in folder.glob("*.??X") if path.is_file())
        parsed_here = [exp for exp in experiments if exp.experiment_file.parent == folder]
        parsed_ids = {exp.experiment_file.resolve() for exp in parsed_here}
        unparsed_files = [path.name for path in raw_files if path.resolve() not in parsed_ids]
        observed_candidates = sorted(path for path in folder.glob("*.??A") if path.is_file())
        matched_observations = [exp.experiment_file.with_suffix(exp.experiment_file.suffix[:-1] + "A")
                                for exp in parsed_here]
        matched_observation_records = [
            {"experiment_id": exp.experiment_id, "file": observation.name,
             "extension": observation.suffix.upper()}
            for exp, observation in zip(parsed_here, matched_observations) if observation.is_file()
        ]
        crop_codes = sorted({exp.crop_code for exp in parsed_here})
        crop_code = crop_codes[0] if crop_codes else ""
        module = next((exp.dssat_module for exp in parsed_here if exp.dssat_module), None)
        ready = []
        invalid = []
        for exp in parsed_here:
            try:
                validate_simulation(exp)
                ready.append(exp)
            except DssatError as exc:
                invalid.append({"experiment_id": exp.experiment_id, "reason": str(exc)})
        row = {
            "crop_folder": folder.name,
            "crop": parsed_here[0].crop if parsed_here else folder.name,
            "crop_code": crop_code,
            "model_module": module,
            "filex_files_found": len(raw_files),
            "experiments_discovered": len(parsed_here),
            "experiments_input_ready": len(ready),
            "experiments_with_observed_wha": sum(bool(exp.observed_file) for exp in parsed_here),
            "experiments_with_matching_observation_file": sum(path.is_file() for path in matched_observations),
            "observation_file_count": len(observed_candidates),
            "observation_files": [path.name for path in observed_candidates],
            "observation_extensions": sorted({path.suffix.upper() for path in observed_candidates}),
            "matching_observation_files": matched_observation_records,
            "discovery_status": "discovered" if parsed_here else ("unparsed_filex" if raw_files else "no_filex"),
            "unparsed_filex_files": unparsed_files,
            "probe_status": "not_run",
            "probe_experiment": None,
            "probe_treatment": None,
            "probe_run_id": None,
            "probe_error": None,
            "probe_attempts": [],
            "invalid_experiments": invalid,
        }
        if args.run_probes:
            if not ready:
                row["probe_status"] = "skipped_no_input_ready_experiment"
            else:
                all_yield_unrecognized = True
                for probe in ready:
                    treatment = int(probe.treatments[0])
                    attempt = {"experiment_id": probe.experiment_id, "treatment_id": treatment}
                    try:
                        result = run_experiment(config, probe, treatment)
                        attempt.update({"status": "passed", "run_id": result["run_id"]})
                        row["probe_status"] = "passed"
                        row["probe_experiment"] = probe.experiment_id
                        row["probe_treatment"] = treatment
                        row["probe_run_id"] = result["run_id"]
                        row["probe_attempts"].append(attempt)
                        break
                    except Exception as exc:
                        is_output_issue = ("valid positive HWAM yield" in str(exc)
                                           or "HWAM yield could not be validated" in str(exc))
                        attempt.update({"status": "yield_output_unrecognized" if is_output_issue else "failed",
                                        "error": f"{type(exc).__name__}: {exc}"})
                        row["probe_attempts"].append(attempt)
                        if not is_output_issue:
                            all_yield_unrecognized = False
                else:
                    row["probe_status"] = ("yield_output_unrecognized" if all_yield_unrecognized
                                            else "failed")
                    row["probe_error"] = "No input-ready experiment produced a validated positive HWAM yield. See probe_attempts for details."
        rows.append(row)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dssat_home": str(config.home),
        "dssat_executable": str(config.executable),
        "dssat_data_root": str(config.data_dir),
        "dssat_version": config.version(),
        "probe_runs_requested": bool(args.run_probes),
        "crop_folder_count": len(rows),
        "crop_folders_with_experiments": sum(row["experiments_discovered"] > 0 for row in rows),
        "experiments_discovered": sum(row["experiments_discovered"] for row in rows),
        "experiments_input_ready": sum(row["experiments_input_ready"] for row in rows),
        "crop_probes_passed": sum(row["probe_status"] == "passed" for row in rows),
        "crop_probes_failed": sum(row["probe_status"] == "failed" for row in rows),
        "crop_probes_output_unrecognized": sum(row["probe_status"] == "yield_output_unrecognized" for row in rows),
        "crop_probes_skipped": sum(row["probe_status"] == "skipped_no_input_ready_experiment" for row in rows),
        "crops": rows,
    }
    output_dir = config.project_data_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "crop_support_audit.json"
    csv_path = output_dir / "crop_support_audit.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([{key: value for key, value in row.items() if key != "invalid_experiments"}
                  for row in rows]).to_csv(csv_path, index=False)
    print(f"DSSAT installation: {config.home}")
    print(f"Data root: {config.data_dir}")
    print(f"Crop folders: {report['crop_folder_count']}; with experiments: {report['crop_folders_with_experiments']}")
    print(f"Experiments discovered: {report['experiments_discovered']}; input-ready: {report['experiments_input_ready']}")
    if args.run_probes:
        print(f"Representative DSSAT runs passed with HWAM: {report['crop_probes_passed']}; execution failed: {report['crop_probes_failed']}; HWAM unsupported/missing: {report['crop_probes_output_unrecognized']}")
    print("Crop\tCode\tModule\tExperiments\tInput-ready\tA-files\tMatched A-files\tProbe")
    for row in rows:
        print(f"{row['crop_folder']}\t{row['crop_code'] or '-'}\t{row['model_module'] or '-'}\t"
              f"{row['experiments_discovered']}\t{row['experiments_input_ready']}\t{row['observation_file_count']}\t"
              f"{row['experiments_with_matching_observation_file']}\t{row['probe_status']}")
    print(f"Detailed report: {json_path}")
    print(f"Spreadsheet report: {csv_path}")
    return 0


def command_audit_experiments(args) -> int:
    """Check every discovered experiment and optionally probe each treatment."""
    config = _config_or_report()
    experiments = discover_experiments(config.data_dir,
                                       profile_path=config.home / "DSSATPRO.v48")
    yield_candidates = ("HWAM", "HWAH", "HWAHF", "FHWAM", "FWAH", "FHWAH",
                        "PWAM", "PWAH", "UWAH", "UYAH", "SWAH", "TWAH", "CWAM", "BWAH")
    checkpoint_path = config.project_data_dir / "diagnostics" / "experiment_support_audit.checkpoint.json"
    checkpoint = {"runs": {}}
    if args.resume:
        if not checkpoint_path.is_file():
            raise DssatError(f"No experiment audit checkpoint exists at {checkpoint_path}.")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    saved_runs = checkpoint.setdefault("runs", {})
    rows = []
    total_treatments = 0
    if args.all_treatments:
        for item in experiments:
            try:
                validate_simulation(item)
                total_treatments += len(item.treatments)
            except DssatError:
                continue
    treatments_started = 0
    for experiment in experiments:
        try:
            validate_simulation(experiment)
            ready, validation_error = True, None
        except DssatError as exc:
            ready, validation_error = False, str(exc)
        observation = experiment.experiment_file.with_suffix(
            experiment.experiment_file.suffix[:-1] + "A")
        treatments = [int(value) for value in experiment.treatments]
        selected_treatments = treatments if args.all_treatments else treatments[:1]
        probes = []
        if (args.run_probes or args.all_treatments) and ready:
            for treatment in selected_treatments:
                treatments_started += 1
                if args.all_treatments and (treatments_started == 1 or treatments_started % 25 == 0):
                    print(f"Treatment sweep: {treatments_started}/{total_treatments} "
                          f"({experiment.crop} / {experiment.experiment_id})", flush=True)
                checkpoint_key = f"{experiment.experiment_id}:{treatment}"
                if args.resume and checkpoint_key in saved_runs:
                    attempt = saved_runs[checkpoint_key]
                else:
                    attempt = {"treatment_id": treatment}
                    try:
                        result = run_experiment(config, experiment, treatment, yield_column=None)
                        summary_row = result["summary"].iloc[0]
                        positive_candidates = {}
                        for column in yield_candidates:
                            if column not in summary_row.index:
                                continue
                            try:
                                value = float(summary_row[column])
                            except (TypeError, ValueError):
                                continue
                            if value > 0 and value < float("inf"):
                                positive_candidates[column] = value
                        attempt.update({"status": "simulation_passed", "run_id": result["run_id"],
                                        "summary_columns": list(result["summary"].columns),
                                        "positive_yield_candidate_values": positive_candidates})
                    except Exception as exc:
                        match = re.search(r"data[\\/]runs[\\/]([0-9_a-f]+)", str(exc), flags=re.IGNORECASE)
                        attempt.update({
                            "status": ("yield_output_unrecognized"
                                       if "valid positive HWAM yield" in str(exc)
                                       or "HWAM yield could not be validated" in str(exc)
                                       else "failed"),
                            "error": f"{type(exc).__name__}: {exc}",
                            "run_id": match.group(1) if match else None,
                        })
                    saved_runs[checkpoint_key] = attempt
                    checkpoint.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                                       "probe_mode": "all_treatments" if args.all_treatments else "first_treatment_per_experiment",
                                       "runs": saved_runs})
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
                probes.append(attempt)
        successful = sum(probe["status"] == "simulation_passed" for probe in probes)
        failed = sum(probe["status"] == "failed" for probe in probes)
        yield_unrecognized = sum(probe["status"] == "yield_output_unrecognized" for probe in probes)
        rows.append({
            "experiment_id": experiment.experiment_id,
            "crop": experiment.crop,
            "crop_code": experiment.crop_code,
            "crop_folder": experiment.experiment_file.parent.name,
            "filex": experiment.experiment_file.name,
            "model_module": experiment.dssat_module,
            "treatment_count": len(treatments),
            "weather_file": experiment.weather_file.name if experiment.weather_file else None,
            "soil_file": experiment.soil_file.name if experiment.soil_file else None,
            "cultivar_file": experiment.cultivar_file.name if experiment.cultivar_file else None,
            "observation_file": observation.name if observation.is_file() else None,
            "input_ready": ready,
            "validation_error": validation_error,
            "probe_count": len(probes),
            "probes_passed": successful,
            "probes_failed": failed,
            "probes_yield_unrecognized": yield_unrecognized,
            "probe_status": ("simulation_passed" if successful and successful == len(probes) else
                             "partial" if successful else
                             "yield_output_unrecognized" if yield_unrecognized and not failed else
                             "failed" if failed else "not_run" if ready else "not_input_ready"),
            "probes": probes,
        })
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dssat_home": str(config.home),
        "dssat_data_root": str(config.data_dir),
        "dssat_version": config.version(),
        "probe_mode": "all_treatments" if args.all_treatments else "first_treatment_per_experiment",
        "probe_runs_requested": bool(args.run_probes or args.all_treatments),
        "experiment_count": len(rows),
        "input_ready_experiments": sum(row["input_ready"] for row in rows),
        "input_ready_treatment_count": sum(row["treatment_count"] for row in rows if row["input_ready"]),
        "experiments_probed": sum(row["probe_count"] > 0 for row in rows),
        "treatments_probed": sum(row["probe_count"] for row in rows),
        "probe_simulations_completed": sum(row["probes_passed"] for row in rows),
        "probe_failures": sum(row["probes_failed"] for row in rows),
        "probe_yield_unrecognized": sum(row["probes_yield_unrecognized"] for row in rows),
        "experiments": rows,
    }
    output_dir = config.project_data_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "experiment_support_audit.json"
    csv_path = output_dir / "experiment_support_audit.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([{key: value for key, value in row.items() if key != "probes"}
                  for row in rows]).to_csv(csv_path, index=False)
    print(f"Experiments discovered: {report['experiment_count']}")
    print(f"Input-ready experiments: {report['input_ready_experiments']}; eligible treatments: {report['input_ready_treatment_count']}")
    if args.run_probes or args.all_treatments:
        print(f"Treatment probes: {report['treatments_probed']}; DSSAT runs completed: {report['probe_simulations_completed']}; execution/output failures: {report['probe_failures']}")
    print(f"Detailed report: {json_path}")
    print(f"Spreadsheet report: {csv_path}")
    return 0


def command_audit_observations(_args) -> int:
    """Inspect all matched crop A-files and compare candidate fields to probes."""
    config = _config_or_report()
    experiments = discover_experiments(config.data_dir,
                                       profile_path=config.home / "DSSATPRO.v48")
    previous_audit_path = config.project_data_dir / "diagnostics" / "experiment_support_audit.json"
    previous_audit = json.loads(previous_audit_path.read_text(encoding="utf-8")) if previous_audit_path.is_file() else {}
    experiment_results = {item["experiment_id"]: item for item in previous_audit.get("experiments", [])}
    candidate_fields = ("HWAM", "HWAH", "HWAHF", "FHWAM", "FWAH", "FHWAH", "PWAM", "PWAH",
                        "UWAH", "UYAH", "SWAH", "TWAH", "CFAH", "CFAD", "LIWAM", "LWAH",
                        "EWAM", "HYAH", "HERB", "YDWAH", "CWAM", "BWAH")
    dictionary_path = config.data_dir / "DATA.CDE"
    dictionary_lines = {}
    if dictionary_path.is_file():
        for line in dictionary_path.read_text(encoding="latin-1", errors="replace").splitlines():
            tokens = line.split()
            if tokens and tokens[0].upper() in candidate_fields:
                dictionary_lines[tokens[0].upper()] = line.strip()
    rows = []
    for experiment in experiments:
        observation_path = experiment.experiment_file.with_suffix(
            experiment.experiment_file.suffix[:-1] + "A")
        if not observation_path.is_file():
            continue
        row = {"experiment_id": experiment.experiment_id, "crop": experiment.crop,
               "crop_code": experiment.crop_code, "file": observation_path.name,
               "extension": observation_path.suffix.upper(), "parse_status": "unparsed",
               "parse_error": None, "treatment_rows": 0, "duplicate_treatment_rows": 0,
               "fields": [], "candidate_fields": [], "candidate_valid_counts": {},
               "probe_treatment": None, "probe_run_id": None,
               "matched_positive_fields": {}, "definitions": {}}
        try:
            observed = read_dssat_a_file(observation_path)
            row["parse_status"] = "parsed"
            row["treatment_rows"] = int(len(observed))
            row["duplicate_treatment_rows"] = int(observed["TRNO"].duplicated().sum())
            row["fields"] = list(observed.columns)
            row["candidate_fields"] = [name for name in candidate_fields if name in observed.columns]
            row["candidate_valid_counts"] = {
                name: int((pd.to_numeric(observed[name], errors="coerce").notna()
                           & pd.to_numeric(observed[name], errors="coerce").ge(0)
                           & pd.to_numeric(observed[name], errors="coerce").ne(-99)).sum())
                for name in row["candidate_fields"]
            }
            row["definitions"] = {name: dictionary_lines.get(name, "not found in DATA.CDE")
                                   for name in row["candidate_fields"]}
            experiment_audit = experiment_results.get(experiment.experiment_id, {})
            successful_probe = next((probe for probe in experiment_audit.get("probes", [])
                                     if probe.get("status") == "simulation_passed"), None)
            if successful_probe:
                row["probe_treatment"] = successful_probe.get("treatment_id")
                row["probe_run_id"] = successful_probe.get("run_id")
                summary_path = config.project_data_dir / "runs" / successful_probe["run_id"] / "output" / "Summary.OUT"
                if summary_path.is_file():
                    summary = parse_summary(summary_path)
                    matching = summary.loc[summary.get("TRNO", pd.Series(index=summary.index, dtype=float))
                                           == row["probe_treatment"]]
                    if not matching.empty:
                        simulated = matching.iloc[0]
                        observed_rows = observed.loc[observed["TRNO"] == row["probe_treatment"]]
                        if not observed_rows.empty:
                            measured = observed_rows.iloc[0]
                            for name in row["candidate_fields"]:
                                try:
                                    actual_value, predicted_value = float(measured[name]), float(simulated[name])
                                except (TypeError, ValueError, KeyError):
                                    continue
                                if actual_value >= 0 and actual_value != -99 and predicted_value > 0:
                                    row["matched_positive_fields"][name] = {
                                        "observed": actual_value, "simulated": predicted_value,
                                        "definition": dictionary_lines.get(name, "not found in DATA.CDE"),
                                    }
        except Exception as exc:
            row["parse_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dssat_data_root": str(config.data_dir),
        "candidate_definition_source": str(dictionary_path) if dictionary_path.is_file() else None,
        "matched_observation_files": len(rows),
        "parsed_files": sum(row["parse_status"] == "parsed" for row in rows),
        "unparsed_files": sum(row["parse_status"] != "parsed" for row in rows),
        "files_with_simulation_probe": sum(bool(row["probe_run_id"]) for row in rows),
        "files_with_positive_field_pairs": sum(bool(row["matched_positive_fields"]) for row in rows),
        "files": rows,
    }
    output_dir = config.project_data_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "observation_file_audit.json"
    csv_path = output_dir / "observation_file_audit.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([{key: value for key, value in row.items() if key not in {"definitions", "matched_positive_fields"}}
                  for row in rows]).to_csv(csv_path, index=False)
    print(f"Matched A-files: {report['matched_observation_files']}; parsed: {report['parsed_files']}; unparsed: {report['unparsed_files']}")
    print(f"Files with a saved DSSAT probe: {report['files_with_simulation_probe']}; with positive measured/simulated field matches: {report['files_with_positive_field_pairs']}")
    print(f"Detailed report: {json_path}")
    print(f"Spreadsheet report: {csv_path}")
    return 0


def command_validate(args) -> int:
    config = _config_or_report()
    experiment = get_experiment(config.data_dir, args.experiment, profile_path=config.home / "DSSATPRO.v48")
    if args.for_dataset:
        validate_observed_dataset(experiment)
        mode = "SIMULATION + OBSERVED-YIELD DATASET READY"
    else:
        validate_simulation(experiment)
        mode = "SIMULATION READY (a .WHA file is not required)"
    print(f"Experiment: {experiment.experiment_id}\nCrop: {experiment.crop}\nWeather: {experiment.weather_file.name}\nSoil: {experiment.soil_id}\nCultivar: {experiment.cultivar_name}\nTreatments: {', '.join(experiment.treatments)}\nStatus: {mode}")
    if experiment.observed_file:
        print(f"Observed yield file: {experiment.observed_file.name}")
    elif args.for_dataset:
        print("Observed yield file: missing")
    return 0


def command_run(args) -> int:
    config = _config_or_report()
    experiment = get_experiment(config.data_dir, args.experiment, profile_path=config.home / "DSSATPRO.v48")
    validate_simulation(experiment)
    result = run_experiment(config, experiment, args.treatment)
    row = result["summary"].iloc[0]
    simulated = float(row["HWAM"])
    observed_value = None
    if experiment.observed_file:
        try:
            observations = read_wheat_yield_observations(experiment.observed_file)
            matching = observations.loc[observations.treatment_id == args.treatment, "observed_yield_kg_ha"]
            if len(matching):
                observed_value = float(matching.iloc[0])
        except (ValueError, OSError):
            observed_value = None
    print(f"Experiment: {experiment.experiment_id}\nCrop: {experiment.crop}\nModel: DSSAT-CSM / {experiment.dssat_module}\nWeather: {experiment.weather_file.name}\nSoil: {experiment.soil_id}\nCultivar: {experiment.cultivar_name}\nTreatment: {args.treatment}\nStatus: SUCCESS")
    print(f"DSSAT HWAM: {simulated:.2f} kg/ha")
    if observed_value is None:
        print("Observed HWAM: unavailable for this treatment (the simulation is still valid).")
    else:
        print(f"Observed HWAM: {observed_value:.2f} kg/ha; residual observed - simulated: {calculate_residual(observed_value, simulated):+.2f} kg/ha")
    if result["growth"] is not None:
        print(f"PlantGro daily rows: {len(result['growth'])}")
    print(f"Run ID: {result['run_id']}\nRun archive: {result['run_dir']}")
    return 0


def command_build_dataset(args) -> int:
    config = _config_or_report()
    output_dir = config.project_data_dir / "processed"
    crop_key = None if args.crop.lower() == "all" else args.crop
    manifest_path = config.project_data_dir / "manifests" / f"{args.crop.lower()}_experiments.json"
    manifest = build_manifest(config.data_dir, manifest_path, require_observed=True, crop=crop_key,
                              profile_path=config.home / "DSSATPRO.v48")
    rows, rejected = [], []
    run_index = _archived_run_index(config)
    reused_run_count = 0
    weather_frame_cache, soil_feature_cache = {}, {}
    print(f"Indexed {sum(len(entries) for entries in run_index.values())} reusable DSSAT run records across {len(run_index)} experiment/treatment keys.", flush=True)
    from src.dssat.data.weather import read_weather
    from src.features.weather_features import aggregate_weather
    from src.features.soil_features import aggregate_soil_profile
    for item in manifest:
        if item["dataset_status"] != "valid":
            rejected.append({"experiment_id": item["experiment_id"], "status": "rejected",
                             "reason": item["dataset_validation_error"]})
            continue
        experiment = get_experiment(config.data_dir, item["experiment_id"], profile_path=config.home / "DSSATPRO.v48")
        target = observed_yield_target(experiment.crop_code)
        if target is None:
            rejected.append({"experiment_id": experiment.experiment_id, "status": "rejected_unmapped_crop_yield",
                             "reason": f"No reviewed yield mapping exists for crop code {experiment.crop_code}."})
            continue
        try:
            observed = read_observed_yield_values(experiment.observed_file, target["column"], target["unit"])
        except Exception as exc:
            rejected.append({"experiment_id": experiment.experiment_id, "status": "rejected_observation_file",
                             "reason": f"{type(exc).__name__}: {exc}"})
            continue
        duplicate_treatments = set(observed.loc[observed["treatment_id"].duplicated(keep=False), "treatment_id"].astype(int))
        for duplicate in sorted(duplicate_treatments):
            rejected.append({"experiment_id": experiment.experiment_id, "treatment_id": int(duplicate),
                             "status": "rejected_duplicate_observation",
                             "reason": "The source A-file has more than one measured-yield row for this treatment."})
        for observed_row in observed.to_dict("records"):
            treatment = int(observed_row["treatment_id"])
            if treatment in duplicate_treatments:
                continue
            if str(treatment) not in experiment.treatments:
                rejected.append({"experiment_id": experiment.experiment_id, "treatment_id": treatment,
                                 "status": "rejected_unmatched_treatment",
                                 "reason": "Observed treatment ID is absent from the paired FileX experiment."})
                continue
            try:
                result = _reuse_archived_run(config, experiment, int(treatment), run_index)
                if result is None:
                    result = run_experiment(config, experiment, int(treatment))
                else:
                    reused_run_count += 1
                summary = result["summary"]
                if "TRNO" not in summary:
                    raise ValueError("DSSAT Summary.OUT has no treatment ID; exact treatment pairing cannot be verified.")
                matching_summary = summary.loc[pd.to_numeric(summary["TRNO"], errors="coerce").eq(treatment)]
                if len(matching_summary) != 1:
                    raise ValueError(f"DSSAT returned {len(matching_summary)} summary rows for treatment {treatment}; exactly one is required.")
                simulation = matching_summary.iloc[0]
                run_metadata = json.loads((result["run_dir"] / "metadata.json").read_text(encoding="utf-8"))
                if (not run_metadata.get("success") or run_metadata.get("experiment_id") != experiment.experiment_id
                        or int(run_metadata.get("treatment_id", -1)) != treatment
                        or run_metadata.get("scenario_application")):
                    raise ValueError("Archived DSSAT run does not prove an unmodified same-experiment, same-treatment simulation.")
                planting_yyddd, harvest_yyddd = int(simulation["PDAT"]), int(simulation["HDAT"])
                staged_weather_dir = result["run_dir"] / "input" / "Weather"
                weather_files = sorted(staged_weather_dir.glob(f"{experiment.weather_station}*.WTH"))
                if not weather_files:
                    raise ValueError("Archived run has no staged weather files for the paired station.")
                weather_key = (experiment.weather_station.upper(), tuple(path.name for path in weather_files))
                if weather_key not in weather_frame_cache:
                    weather_frame_cache[weather_key] = pd.concat(
                        [read_weather(path) for path in weather_files], ignore_index=True)
                weather_frame = weather_frame_cache[weather_key]
                weather_features = aggregate_weather(weather_frame, planting_yyddd, harvest_yyddd)
                planting_year, planting_day = divmod(planting_yyddd, 1000)
                if planting_year < 100:
                    planting_year += 1900 if planting_year >= 50 else 2000
                planting_date = (datetime.strptime(f"{planting_year:04d}{planting_day:03d}", "%Y%j").date().isoformat())
                observed_yield = float(observed_row["observed_value"])
                dssat_yield = float(simulation["HWAM"])
                if not (pd.notna(observed_yield) and observed_yield > 0 and pd.notna(dssat_yield) and dssat_yield > 0):
                    raise ValueError("Measured and DSSAT yields must both be finite positive harvested yields.")
                summary_file = result["run_dir"] / "output" / "Summary.OUT"
                if not summary_file.is_file():
                    raise ValueError("Archived DSSAT Summary.OUT is missing; this treatment cannot be audited as a pair.")
                soil_file = result["run_dir"] / "input" / "Soil" / experiment.soil_file.name
                if not soil_file.is_file():
                    raise ValueError("Archived DSSAT soil input is missing; this treatment cannot be audited as a pair.")
                relative_summary = summary_file.relative_to(config.project_data_dir.parent).as_posix()
                source_experiment_id = experiment.experiment_file.relative_to(config.data_dir).with_suffix("").as_posix()
                soil_key = (experiment.soil_id, soil_file.name)
                if soil_key not in soil_feature_cache:
                    soil_feature_cache[soil_key] = aggregate_soil_profile(soil_file, experiment.soil_id)
                rows.append({"experiment_id": f"{experiment.crop_code}:{source_experiment_id}", "source_experiment_id": source_experiment_id, "crop_code": experiment.crop_code, "treatment_id": int(treatment), "crop": experiment.crop,
                             "cultivar": experiment.cultivar_name, "cultivar_code": experiment.cultivar_code,
                             **experiment.treatment_details[int(treatment)], "weather_station": experiment.weather_station,
                             "soil_id": experiment.soil_id, "location": experiment.location,
                             "year": int(simulation.get("WYEAR", planting_year)), "planting_date": planting_date,
                             "observed_yield_kg_ha": observed_yield, "observed_yield_variable": target["column"],
                             "observed_yield_definition": target["definition"],
                             "observed_yield_unit": target["unit"],
                             "observed_yield_basis": target["basis"],
                             "dssat_yield_kg_ha": dssat_yield,
                             "residual_kg_ha": calculate_residual(observed_yield, dssat_yield), **weather_features,
                             **soil_feature_cache[soil_key],
                             "run_id": result["run_id"], "dssat_version": config.version(),
                             "weather_basis": "historical_completed_season",
                             "pairing_status": "eligible",
                             "pairing_basis": "same_experiment_same_treatment",
                             "observation_level": "experiment_treatment_measurement",
                             "observed_source_file": experiment.observed_file.name,
                             "observed_source_row": int(observed_row["source_row"]),
                             "observed_treatment_id": treatment,
                             "simulation_filex": experiment.experiment_file.name,
                             "simulation_yield_variable": target["column"],
                             "dssat_run_id": result["run_id"],
                             "dssat_summary_file": relative_summary,
                             "dssat_treatment_id": treatment,
                             "management_source_file": experiment.experiment_file.name,
                             "management_treatment_id": treatment,
                             "weather_files_used": ";".join(path.name for path in weather_files),
                             "soil_file_used": soil_file.name})
                if len(rows) % 50 == 0:
                    print(f"Paired {len(rows)} rows so far across {len({row['crop_code'] for row in rows})} crops; reused {reused_run_count} archived runs.", flush=True)
            except Exception as exc:
                rejected.append({"experiment_id": experiment.experiment_id, "treatment_id": int(treatment),
                                 "status": "rejected_run_or_pair_validation",
                                 "reason": f"{type(exc).__name__}: {exc}"})
                if not args.continue_on_error:
                    break
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = output_dir / ("dssat_hybrid_dataset.csv" if args.crop.lower() == "all" else f"{args.crop.lower()}_hybrid_dataset.csv")
    if not frame.empty:
        validate_training_frame(frame.copy())
    if csv_path.exists():
        backup_dir = output_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(csv_path, backup_dir / f"{csv_path.stem}_{stamp}{csv_path.suffix}")
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    frame.to_csv(temporary_csv, index=False)
    temporary_csv.replace(csv_path)
    report = {
        "pairing_policy": "Only exact same-stem crop A-file and FileX records with matching treatment IDs and one successful unmodified DSSAT run are eligible.",
        "residual_formula": "observed_yield_kg_ha - dssat_yield_kg_ha",
        "eligible_rows": len(frame),
        "eligible_experiments": int(frame["experiment_id"].nunique()) if not frame.empty else 0,
        "reused_archived_runs": reused_run_count,
        "eligible_rows_by_crop_code": ({str(code): int(count) for code, count in frame["crop_code"].value_counts().sort_index().items()}
                                        if not frame.empty else {}),
        "eligible_pairs": frame[["experiment_id", "treatment_id", "observed_source_file", "dssat_run_id"]].to_dict("records") if not frame.empty else [],
        "rejected_count": len(rejected),
        "rejected_records": rejected,
    }
    (output_dir / f"{args.crop.lower()}_pairing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(frame)} observed/simulated pairs to {csv_path}")
    print(f"Rejected or unmapped records: {len(rejected)}")
    return 0 if len(frame) else 1


def _dependency_versions() -> dict:
    result = {}
    for package in ("pandas", "scikit-learn", "joblib"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _backup_model_artifacts(model_dir: Path) -> Path | None:
    filenames = ("residual_model.joblib", "model_metadata.json", "model_metrics.json")
    existing = [model_dir / name for name in filenames if (model_dir / name).is_file()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = model_dir / "backups" / stamp
    suffix = 1
    while backup.exists():
        backup = model_dir / "backups" / f"{stamp}-{suffix}"
        suffix += 1
    backup.mkdir(parents=True)
    for path in existing:
        shutil.copy2(path, backup / path.name)
    return backup


def command_train(args) -> int:
    config = _config_or_report()
    source = _dataset_path(config, args.dataset, args.crop)
    if not source.is_file():
        raise FileNotFoundError(f"Training dataset not found: {source}")
    frame = pd.read_csv(source)
    validate_training_frame(frame)
    model = make_regressor(RESIDUAL_FEATURES, seed=RANDOM_SEED)
    model.fit(frame[RESIDUAL_FEATURES], frame["residual_kg_ha"])
    config.model_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = _backup_model_artifacts(config.model_dir)
    model_path = config.model_dir / "residual_model.joblib"
    temporary_model = config.model_dir / "residual_model.joblib.tmp"
    import joblib
    joblib.dump(model, temporary_model)
    temporary_model.replace(model_path)
    metadata = {
        "model_version": "4.0", "model_type": "Histogram gradient boosting residual regression",
        "estimator_name": "HistGradientBoostingRegressor",
        "training_dataset": source.name, "training_dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "training_record_count": int(len(frame)), "training_experiments": sorted(frame["experiment_id"].astype(str).unique().tolist()),
        "dssat_version": config.version(), "features": RESIDUAL_FEATURES,
        "categorical_features": ["crop", "cultivar", "weather_station", "soil_id"],
        "numeric_features": [name for name in RESIDUAL_FEATURES if name not in {"crop", "cultivar", "weather_station", "soil_id"}],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(), "random_seed": RANDOM_SEED,
        "estimator_parameters": {"max_iter": 150, "l2_regularization": 1.0},
        "fit_scope": "All dataset rows for a deployable artifact. Benchmark performance is reported separately from out-of-fold evaluation.",
        "pairing_policy": "Training rows are accepted only when measured yield and DSSAT yield share the crop experiment and treatment, carry a successful unmodified run ID, and include weather/soil/management source provenance.",
        "residual_definition": "measured field yield - DSSAT yield, both in the reviewed crop-specific unit",
        "observation_level": "experiment_treatment_measurement; aggregate district/state averages excluded",
        "weather_basis": "historical_completed_season; retrospective only",
        "backup_directory": str(backup_dir) if backup_dir else None,
        "dependency_versions": _dependency_versions(),
    }
    (config.model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # A former single-holdout metrics file is backed up above; current metrics
    # live only in the explicitly recomputed evaluation directory.
    legacy_metrics = config.model_dir / "model_metrics.json"
    if legacy_metrics.exists():
        legacy_metrics.unlink()
    print(f"Saved final all-data residual model: {model_path}")
    print(f"Training rows: {len(frame)}; experiment groups: {frame['experiment_id'].nunique()}")
    print(f"Prior model backup: {backup_dir if backup_dir else 'none'}")
    print("This training operation does not calculate or claim benchmark performance; run `evaluate` separately.")
    return 0


def command_evaluate(args) -> int:
    config = _config_or_report()
    source = _dataset_path(config, args.dataset, args.crop)
    if not source.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {source}")
    result = evaluate_dataset(source, config.project_data_dir / "evaluation")
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))
    print(f"OOF predictions: {config.project_data_dir / 'evaluation' / 'oof_predictions.csv'}")
    print(f"Evaluation metadata: {config.project_data_dir / 'evaluation' / 'metadata.json'}")
    return 0


def command_show_evaluation(_args) -> int:
    config = _config_or_report()
    path = config.project_data_dir / "evaluation" / "metrics.json"
    metadata_path = config.project_data_dir / "evaluation" / "metadata.json"
    if not path.is_file():
        print("No saved benchmark is available. Recompute with: python -m cli.main evaluate --crop wheat")
        return 1
    if not metadata_path.is_file() or json.loads(metadata_path.read_text(encoding="utf-8")).get("models", {}).get("residual", {}).get("estimator") != "HistGradientBoostingRegressor":
        print("The saved benchmark uses an older model. Recompute it with: python -m cli.main evaluate --crop wheat")
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


def command_predict(args) -> int:
    config = _config_or_report()
    experiment = get_experiment(config.data_dir, args.experiment, profile_path=config.home / "DSSATPRO.v48")
    validate_simulation(experiment)
    result = run_experiment(config, experiment, args.treatment)
    summary = result["summary"].iloc[0]
    dssat_yield = float(summary["HWAM"])
    area = area_estimate(dssat_yield, args.area, args.area_unit)
    print(f"Experiment: {experiment.experiment_id}\nCrop: {experiment.crop}\nTreatment: {args.treatment}\nCultivar: {experiment.cultivar_name}")
    print("Prediction basis: historical full-season weather; retrospective simulation, not a sowing-time forecast.")
    print(f"DSSAT yield: {dssat_yield:.2f} kg/ha")
    print(f"Selected experimental area: {area['area_input']:g} {area['area_unit']} ({area['area_hectares']:.6g} ha)")
    print(f"Estimated total harvest under the selected experimental conditions (DSSAT): {area['total_harvest_kg']:.2f} kg ({area['total_harvest_tonnes']:.4f} tonnes)")
    model_path = config.model_dir / "residual_model.joblib"
    metadata_path = config.model_dir / "model_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        print("Hybrid status: no trained residual model is installed; train one with `python -m cli.main train-residual --crop wheat`.")
        print(f"Run ID: {result['run_id']}\nRun archive: {result['run_dir']}")
        return 0
    try:
        features = build_experiment_features(config, experiment, summary, args.treatment)
        hybrid = predict_residual(config, features, dssat_yield)
    except Exception as exc:
        print(f"Hybrid status: FAILED ({type(exc).__name__}: {exc}); DSSAT-only result shown above is not a hybrid result.")
        print(f"Run ID: {result['run_id']}\nRun archive: {result['run_dir']}")
        return 1
    if hybrid is None:
        print("Hybrid status: no trained residual model is installed; train one with `python -m cli.main train-residual --crop wheat`.")
    else:
        hybrid_area = area_estimate(hybrid["hybrid_yield_kg_ha"], args.area, args.area_unit)
        print(f"Predicted ML residual: {hybrid['predicted_residual_kg_ha']:+.2f} kg/ha")
        print(f"Hybrid yield: {hybrid['hybrid_yield_kg_ha']:.2f} kg/ha (model {hybrid['model_version']})")
        print(f"Estimated total harvest under the selected experimental conditions (hybrid): {hybrid_area['total_harvest_kg']:.2f} kg ({hybrid_area['total_harvest_tonnes']:.4f} tonnes)")
    benchmark = config.project_data_dir / "evaluation" / "metrics.json"
    print(f"Benchmark status: {'available' if benchmark.is_file() else 'not yet evaluated; run `python -m cli.main evaluate --crop wheat`'}")
    print(f"Run ID: {result['run_id']}\nRun archive: {result['run_dir']}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cli.main")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor").set_defaults(func=command_doctor)
    audit = commands.add_parser("audit-crops", help="inventory supported DSSAT crop folders and optionally run one probe per crop")
    audit.add_argument("--run-probes", action="store_true", help="run one real DSSAT experiment for each crop with complete inputs")
    audit.set_defaults(func=command_audit_crops)
    experiment_audit = commands.add_parser("audit-experiments", help="inventory every DSSAT experiment and optionally run every treatment")
    experiment_audit.add_argument("--run-probes", action="store_true", help="run the first treatment of each input-ready experiment")
    experiment_audit.add_argument("--all-treatments", action="store_true", help="probe every treatment (also enables --run-probes)")
    experiment_audit.add_argument("--resume", action="store_true", help="reuse saved treatments from the experiment audit checkpoint")
    experiment_audit.set_defaults(func=command_audit_experiments)
    observations_audit = commands.add_parser("audit-observations", help="inspect crop-specific measured A-files and compare fields with saved probes")
    observations_audit.set_defaults(func=command_audit_observations)
    listing = commands.add_parser("list-experiments"); listing.add_argument("--crop", default="all", help="DSSAT crop folder name, or 'all'"); listing.set_defaults(func=command_list)
    validation = commands.add_parser("validate-experiment"); validation.add_argument("--experiment", required=True); validation.add_argument("--for-dataset", action="store_true", help="also require parseable observed yields (.WHA)"); validation.set_defaults(func=command_validate)
    run = commands.add_parser("run-experiment"); run.add_argument("--experiment", required=True); run.add_argument("--treatment", type=int, default=1); run.set_defaults(func=command_run)
    build = commands.add_parser("build-dataset"); build.add_argument("--crop", default="all", help="DSSAT crop folder name, or 'all'"); build.add_argument("--continue-on-error", action="store_true"); build.set_defaults(func=command_build_dataset)
    train = commands.add_parser("train-residual"); train.add_argument("--crop", default="all", help="DSSAT crop folder name, or 'all'"); train.add_argument("--dataset"); train.set_defaults(func=command_train)
    evaluate = commands.add_parser("evaluate"); evaluate.add_argument("--crop", default="all", help="DSSAT crop folder name, or 'all'"); evaluate.add_argument("--dataset"); evaluate.set_defaults(func=command_evaluate)
    show = commands.add_parser("show-evaluation", help="print saved evaluation results without recomputing"); show.set_defaults(func=command_show_evaluation)
    predict = commands.add_parser("predict"); predict.add_argument("--experiment", required=True); predict.add_argument("--treatment", type=int, default=1); predict.add_argument("--area", type=float, default=1.0); predict.add_argument("--area-unit", choices=("hectares", "acres"), default="hectares"); predict.set_defaults(func=command_predict)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return args.func(args)
    except (DssatError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
