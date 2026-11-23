"""Create a clearly synthetic residual-model demonstration dataset.

This is a bootstrap teaching artifact, not new field evidence. It deliberately
omits the strict DSSAT/A-file pairing provenance required by production training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.modeling import ELIGIBLE_FEATURES


DEFAULT_SOURCE = ROOT / "data/processed/dssat_hybrid_dataset.csv"
DEFAULT_OUTPUT = ROOT / "data/synthetic/demo_residual_dataset.csv"
DEFAULT_METADATA = ROOT / "data/synthetic/demo_residual_dataset_metadata.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(source: Path, output: Path, metadata: Path, rows: int, seed: int) -> None:
    source_frame = pd.read_csv(source)
    if not set(ELIGIBLE_FEATURES + ["observed_yield_kg_ha", "residual_kg_ha"]).issubset(source_frame):
        raise ValueError("Source dataset does not contain the model's current feature and target columns.")
    if source_frame.empty or (source_frame["dssat_yield_kg_ha"] <= 0).any():
        raise ValueError("Source must contain positive, observed/DSSAT-paired records.")

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(source_frame), size=rows)
    synthetic = source_frame.iloc[sampled][ELIGIBLE_FEATURES].reset_index(drop=True).copy()
    synthetic["source_record_id"] = [f"{source_frame.iloc[i]['experiment_id']}:{int(source_frame.iloc[i]['treatment_id'])}" for i in sampled]
    synthetic["source_crop"] = source_frame.iloc[sampled]["crop"].astype(str).to_numpy()

    # Resample residuals within crop where possible, preserving empirical units
    # and rough crop-specific error distributions without claiming new labels.
    residuals_by_crop = {
        str(crop): group["residual_kg_ha"].to_numpy(dtype=float)
        for crop, group in source_frame.groupby("crop", sort=False)
    }
    residual = np.empty(rows, dtype=float)
    for crop in synthetic["crop"].astype(str).unique():
        mask = synthetic["crop"].astype(str).eq(crop).to_numpy()
        pool = residuals_by_crop.get(crop, source_frame["residual_kg_ha"].to_numpy(dtype=float))
        residual[mask] = rng.choice(pool, size=int(mask.sum()), replace=True)
    synthetic["observed_yield_kg_ha"] = np.maximum(
        1.0, synthetic["dssat_yield_kg_ha"].to_numpy(dtype=float) + residual
    )
    synthetic["residual_kg_ha"] = (
        synthetic["observed_yield_kg_ha"] - synthetic["dssat_yield_kg_ha"]
    )
    synthetic["record_type"] = "SYNTHETIC_BOOTSTRAP_NOT_FIELD_OBSERVATION"
    synthetic["generation_seed"] = seed
    synthetic["synthetic_row_id"] = [f"SYN-{seed}-{i + 1:06d}" for i in range(rows)]

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(output, index=False, float_format="%.6f")
    manifest = {
        "dataset_status": "synthetic_bootstrap_demo_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(synthetic),
        "seed": seed,
        "source_file": source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source),
        "source_sha256": sha256(source),
        "source_row_count": len(source_frame),
        "source_experiment_count": int(source_frame["experiment_id"].nunique()),
        "source_crop_count": int(source_frame["crop"].nunique()),
        "feature_columns": ELIGIBLE_FEATURES,
        "target_columns": ["residual_kg_ha", "observed_yield_kg_ha"],
        "generation_method": (
            "Stratified bootstrap of authentic DSSAT feature vectors, followed by within-crop "
            "bootstrap sampling of observed-minus-DSSAT residuals. Rows are intentionally pseudo-data."
        ),
        "external_open_data_references": [
            {"name": "DSSAT-CSM experimental data repository", "url": "https://github.com/DSSAT/dssat-csm-data"},
            {"name": "NASA POWER daily API documentation", "url": "https://power.larc.nasa.gov/docs/services/api/temporal/daily/"},
            {"name": "SoilGrids documentation", "url": "https://docs.isric.org/globaldata/soilgrids/"},
            {"name": "CIMMYT eastern India wheat field dataset (not Maharashtra ground truth)", "url": "https://doi.org/10.71682/10549424"},
        ],
        "limitations": [
            "Synthetic rows are resamples of the source dataset and do not increase independent sample size.",
            "Residuals are bootstrapped from existing paired observations; they are not new measured outcomes.",
            "No synthetic row has a verified DSSAT run for its complete row-specific inputs.",
            "This file intentionally lacks production pairing provenance and must not be used for production training or accuracy claims.",
            "It may be used for UI demonstrations, schema checks, and clearly labeled pipeline exercises only.",
        ],
        "production_training_dataset": "data/processed/dssat_hybrid_dataset.csv",
        "production_model_uses_synthetic_rows": False,
    }
    metadata.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(synthetic)} clearly synthetic demonstration rows to {output}")
    print(f"Manifest: {metadata}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    create(args.source.resolve(), args.output.resolve(), args.metadata.resolve(), args.rows, args.seed)


if __name__ == "__main__":
    main()
