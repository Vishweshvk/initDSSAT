"""Depth-weighted soil summaries from real DSSAT profile layers."""

import pandas as pd

from src.dssat.data.soil import read_soil_profile


SOIL_PROPERTIES = {
    "SLLL": "soil_mean_slll_cm3_cm3",
    "SDUL": "soil_mean_sdul_cm3_cm3",
    "SSAT": "soil_mean_ssat_cm3_cm3",
    "SBDM": "soil_mean_sbdm_g_cm3",
    "SLOC": "soil_mean_sloc_pct",
}


def aggregate_soil_profile(path, soil_id: str) -> dict:
    """Return layer-depth-weighted means; DSSAT -99 missing codes are excluded."""
    layers = read_soil_profile(path, soil_id)
    boundaries = pd.to_numeric(layers["SLB"], errors="coerce")
    thickness = boundaries.diff().fillna(boundaries.iloc[0]).clip(lower=0)
    result = {"soil_depth_cm": float(boundaries.max())}
    for dssat_code, feature in SOIL_PROPERTIES.items():
        values = pd.to_numeric(layers[dssat_code], errors="coerce")
        valid = values.notna() & values.ne(-99)
        denominator = float(thickness[valid].sum())
        result[feature] = float((values[valid] * thickness[valid]).sum() / denominator) if denominator else None
    return result
