"""Train the hybrid residual ML model.

Real-world data provides observed yield. WOFOST provides a mechanistic yield
for the same state/district/crop/season/year. The ML model learns the residual:

    residual = observed_yield_kg_ha - TWSO_simulated

The live model deliberately uses only features available at prediction time:
state, district, crop, season, year and WOFOST yield. Soil/irrigation are kept
inside the mechanistic simulation rather than pretending that the public
production dataset contains field-level soil/irrigation observations.
"""
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REAL_CSV = "real_world_dataset.csv"
SIM_CSV = "simulated_output_dataset.csv"
MODEL_OUT = "residual_model.pkl"

CATEGORICAL = ["state", "district", "crop", "season"]
NUMERIC = ["year", "TWSO_simulated"]
FEATURES = CATEGORICAL + NUMERIC
KEYS = ["state", "district", "crop", "season", "year"]


def _normalize(df):
    df = df.copy()
    for col in ["state", "district"]:
        df[col] = df[col].astype(str).str.strip()
    df["crop"] = df["crop"].astype(str).str.strip().str.lower()
    df["season"] = df["season"].astype(str).str.strip().str.title()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def main():
    if not os.path.exists(REAL_CSV):
        raise FileNotFoundError(f"Missing {REAL_CSV}. Run download_real_world_dataset.py first.")
    if not os.path.exists(SIM_CSV):
        raise FileNotFoundError(f"Missing {SIM_CSV}. Run generate_simulated_dataset.py first.")

    real = _normalize(pd.read_csv(REAL_CSV))
    sim = _normalize(pd.read_csv(SIM_CSV))

    required_real = set(KEYS + ["observed_yield_kg_ha"])
    required_sim = set(KEYS + ["TWSO_simulated"])
    if missing := required_real - set(real.columns):
        raise ValueError(f"real_world_dataset.csv is missing columns: {sorted(missing)}")
    if missing := required_sim - set(sim.columns):
        raise ValueError(f"simulated_output_dataset.csv is missing columns: {sorted(missing)}")
    if "status" not in sim.columns:
        sim["status"] = sim["TWSO_simulated"].apply(
            lambda v: "NON_ZERO" if pd.notna(v) and float(v) > 0 else "ZERO_YIELD"
        )

    # A real observation can match several WOFOST runs because the simulator
    # explores multiple soil/irrigation scenarios. Aggregate those simulations
    # to one reference mechanistic baseline per real observation, avoiding
    # duplicate weighting of the same field-year observation.
    sim = sim[sim["status"].eq("NON_ZERO")].copy()
    sim["TWSO_simulated"] = pd.to_numeric(sim["TWSO_simulated"], errors="coerce")
    sim = sim.dropna(subset=["TWSO_simulated"])
    sim_reference = sim.groupby(KEYS, as_index=False)["TWSO_simulated"].median()

    merged = real.merge(sim_reference, on=KEYS, how="inner")
    merged = merged.dropna(subset=["observed_yield_kg_ha", "TWSO_simulated"])
    merged["observed_yield_kg_ha"] = pd.to_numeric(merged["observed_yield_kg_ha"], errors="coerce")
    merged = merged[merged["observed_yield_kg_ha"] >= 0].copy()
    if len(merged) < 30:
        raise ValueError(
            f"Only {len(merged)} matched observations are available. "
            "Generate WOFOST simulations across more real-data years before training."
        )

    merged["residual"] = merged["observed_yield_kg_ha"] - merged["TWSO_simulated"]
    X = merged[FEATURES]
    y = merged["residual"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    model = Pipeline([
        ("preprocess", pre),
        ("regressor", RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2
        )),
    ])
    model.fit(X_train, y_train)

    residual_pred = model.predict(X_test)
    final_pred = X_test["TWSO_simulated"].to_numpy() + residual_pred
    observed = X_test["TWSO_simulated"].to_numpy() + y_test.to_numpy()

    print(f"Matched real observations: {len(merged)}")
    print(f"Residual MAE: {mean_absolute_error(y_test, residual_pred):.2f} kg/ha")
    print(f"Final-yield MAE: {mean_absolute_error(observed, final_pred):.2f} kg/ha")
    print(f"Final-yield RMSE: {mean_squared_error(observed, final_pred) ** 0.5:.2f} kg/ha")
    print(f"Final-yield R2: {r2_score(observed, final_pred):.3f}")

    joblib.dump({"model": model, "features": FEATURES}, MODEL_OUT)
    merged.to_csv("training_pairs_used.csv", index=False)
    print(f"Saved {MODEL_OUT} and training_pairs_used.csv")


if __name__ == "__main__":
    main()
