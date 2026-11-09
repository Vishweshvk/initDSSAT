"""Load and apply the trained residual model."""
import os

import joblib
import pandas as pd

MODEL_PATH = "residual_model.pkl"


def apply_residual_correction(
    state, crop, season, district, year, twso_simulated, soil_type=None, irrigated=None
):
    """Apply a trained residual correction using only training-time features."""
    if not os.path.exists(MODEL_PATH):
        return float(twso_simulated), False

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    features = pd.DataFrame([{
        "state": state,
        "crop": crop,
        "season": season,
        "district": district,
        "year": int(year),
        "TWSO_simulated": float(twso_simulated),
    }])
    predicted_residual = float(model.predict(features)[0])
    corrected_yield = max(0.0, float(twso_simulated) + predicted_residual)
    return corrected_yield, True
