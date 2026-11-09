"""Build validated, traceable experiment manifests."""

from pathlib import Path
import json

from .repository import discover_experiments, validate_observed_dataset, validate_simulation


def build_manifest(data_dir: Path, output_path: Path, require_observed: bool = False,
                   crop: str | None = None, profile_path: Path | None = None) -> list[dict]:
    manifest = []
    for experiment in discover_experiments(data_dir, crop, profile_path):
        record = experiment.to_dict()
        try:
            validate_simulation(experiment)
            record["simulation_status"] = "valid"
            record["simulation_validation_error"] = ""
        except Exception as exc:
            record["simulation_status"] = "invalid"
            record["simulation_validation_error"] = str(exc)
        try:
            validate_observed_dataset(experiment)
            record["dataset_status"] = "valid"
            record["dataset_validation_error"] = ""
        except Exception as exc:
            record["dataset_status"] = "invalid"
            record["dataset_validation_error"] = str(exc)
        record["status"] = record["dataset_status"] if require_observed else record["simulation_status"]
        record["validation_error"] = record["dataset_validation_error"] if require_observed else record["simulation_validation_error"]
        manifest.append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
