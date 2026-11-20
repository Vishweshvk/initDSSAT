"""Shared feature definitions and model factory for evaluation and deployment."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from pathlib import PurePath
import pandas as pd

CATEGORICAL_FEATURES = ["crop", "cultivar", "weather_station", "soil_id"]
NUMERIC_FEATURES = [
    "dssat_yield_kg_ha", "total_rainfall_mm", "mean_tmax_c", "mean_tmin_c",
    "mean_solar_radiation_mj_m2_day", "rain_days", "crop_cycle_days", "soil_depth_cm",
    "soil_mean_slll_cm3_cm3", "soil_mean_sdul_cm3_cm3", "soil_mean_ssat_cm3_cm3",
    "soil_mean_sbdm_g_cm3", "soil_mean_sloc_pct", "nitrogen_rate_kg_ha", "irrigated",
]
ELIGIBLE_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
RESIDUAL_FEATURES = ELIGIBLE_FEATURES
ML_ONLY_FEATURES = CATEGORICAL_FEATURES + [name for name in NUMERIC_FEATURES if name != "dssat_yield_kg_ha"]
# Keep unobserved management, weather, and soil values missing instead of
# inventing values. HistGradientBoostingRegressor supports missing numeric
# feature values natively; DSSAT and measured outcomes must remain present.
MISSING_ALLOWED_NUMERIC_FEATURES = set(NUMERIC_FEATURES) - {"dssat_yield_kg_ha"}
RANDOM_SEED = 42
PAIRING_PROVENANCE_COLUMNS = {
    "pairing_status", "pairing_basis", "source_experiment_id",
    "observed_source_file", "observed_yield_variable", "observed_yield_unit",
    "observed_yield_basis", "observed_treatment_id",
    "simulation_filex", "dssat_run_id", "dssat_summary_file",
    "simulation_yield_variable", "dssat_treatment_id", "management_source_file",
    "management_treatment_id", "weather_files_used", "soil_file_used",
    "observation_level", "crop_code", "simulation_yield_definition",
    "simulation_yield_unit", "simulation_yield_basis", "yield_definition_verified",
}


def make_regressor(features: list[str], *, seed: int = RANDOM_SEED) -> Pipeline:
    categorical = [name for name in CATEGORICAL_FEATURES if name in features]
    numeric = [name for name in features if name not in categorical]
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ], sparse_threshold=0)
    return Pipeline([
        ("preprocess", preprocess),
        ("regressor", HistGradientBoostingRegressor(max_iter=150, l2_regularization=1.0,
                                                     random_state=seed)),
    ])


def validate_training_frame(frame) -> None:
    # Older single-crop datasets predate the crop feature; they represent Wheat.
    if "crop" not in frame.columns:
        frame["crop"] = "Wheat"
    required = {"experiment_id", "treatment_id", "observed_yield_kg_ha", "residual_kg_ha",
                *ELIGIBLE_FEATURES, *PAIRING_PROVENANCE_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    # Trial labels and local resource files may omit feature values. Keep those
    # values missing instead of guessing; HistGradientBoostingRegressor can
    # learn with missing numeric features natively.
    for column in MISSING_ALLOWED_NUMERIC_FEATURES:
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & raw.astype(str).str.strip().ne("") & numeric.isna()
        if invalid.any():
            raise ValueError(f"{column} contains values that are neither numeric nor missing.")
        frame[column] = numeric.astype(float)
    if len(frame) < 2 or frame["experiment_id"].nunique() < 2:
        raise ValueError("Need at least two records from at least two experiments for experiment-grouped evaluation.")
    if frame.duplicated(["experiment_id", "treatment_id"]).any():
        raise ValueError("Dataset contains duplicate experiment/treatment observations.")
    if not frame["pairing_status"].astype(str).eq("eligible").all():
        raise ValueError("Training accepts only records explicitly marked pairing_status='eligible'.")
    if not frame["pairing_basis"].astype(str).eq("same_experiment_same_treatment").all():
        raise ValueError("Training accepts only measured/DSSAT pairs from the same experiment and treatment.")
    if not frame["observation_level"].astype(str).eq("experiment_treatment_measurement").all():
        raise ValueError("Regional averages and other aggregate observations are not eligible field-level residual targets.")
    for column in ("source_experiment_id", "observed_source_file", "observed_yield_variable",
                   "observed_yield_unit", "observed_yield_basis",
                   "simulation_filex", "dssat_run_id", "dssat_summary_file",
                   "simulation_yield_variable", "management_source_file",
                   "weather_files_used", "soil_file_used"):
        if not frame[column].map(lambda value: bool(str(value).strip()) and str(value).strip().casefold() != "nan").all():
            raise ValueError(f"Dataset has missing pairing provenance in {column}.")
    if not frame["observed_yield_variable"].astype(str).str.upper().equals(
            frame["simulation_yield_variable"].astype(str).str.upper()):
        raise ValueError("Measured and simulated yield variables must use the same reviewed crop target.")
    if not frame["observed_yield_unit"].astype(str).eq("kg/ha").all() or not frame["observed_yield_basis"].astype(str).eq("dry_matter").all():
        raise ValueError("This residual model requires yields normalized to kg/ha dry matter.")
    if (not frame["simulation_yield_unit"].astype(str).eq("kg/ha").all()
            or not frame["simulation_yield_basis"].astype(str).eq("dry_matter").all()
            or not frame["yield_definition_verified"].astype(str).str.casefold().isin({"true", "1"}).all()):
        raise ValueError("Residual training requires a reviewed harvested-product definition normalized to kg/ha dry matter.")
    for column in ("observed_treatment_id", "dssat_treatment_id", "management_treatment_id"):
        if not pd.to_numeric(frame[column], errors="coerce").eq(
                pd.to_numeric(frame["treatment_id"], errors="coerce")).all():
            raise ValueError(f"{column} must match the paired treatment_id for every row.")
    expected_groups = frame["crop_code"].astype(str) + ":" + frame["source_experiment_id"].astype(str)
    if not expected_groups.eq(frame["experiment_id"].astype(str)).all():
        raise ValueError("Experiment groups must identify the crop and exact source experiment.")
    filex_stems = frame["simulation_filex"].map(lambda value: PurePath(str(value)).stem.casefold())
    observed_stems = frame["observed_source_file"].map(lambda value: PurePath(str(value)).stem.casefold())
    if not filex_stems.eq(observed_stems).all():
        raise ValueError("Measured A-files must share a filename stem with the paired DSSAT FileX.")
    values = frame[list(required)].select_dtypes(include="number")
    required_values = values.drop(columns=[column for column in MISSING_ALLOWED_NUMERIC_FEATURES if column in values],
                                  errors="ignore")
    if not required_values.empty and not np.isfinite(required_values.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite numeric outcomes or features.")
    for column in MISSING_ALLOWED_NUMERIC_FEATURES:
        present = frame[column].dropna().to_numpy(dtype=float)
        if present.size and not np.isfinite(present).all():
            raise ValueError(f"{column} contains non-finite values.")
    required_without_optional = frame[list(required)].drop(
        columns=[column for column in MISSING_ALLOWED_NUMERIC_FEATURES if column in frame], errors="ignore")
    if required_without_optional.isna().any().any():
        raise ValueError("Dataset contains missing identifiers, outcomes, or model features.")
    expected = frame["observed_yield_kg_ha"] - frame["dssat_yield_kg_ha"]
    if not expected.sub(frame["residual_kg_ha"]).abs().le(1e-6).all():
        raise ValueError("Residual values must equal observed_yield_kg_ha - dssat_yield_kg_ha.")
