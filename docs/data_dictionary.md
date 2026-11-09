# Data and evaluation dictionary

## Experiment and source fields

| Field | Meaning | Source / caution |
|---|---|---|
| `experiment_id` | DSSAT crop FileX stem and evaluation group | Crop FileX; entire experiment held out in one evaluation fold |
| `treatment_id` | Treatment number within the experiment | `.WHX` and `.WHA`; not globally unique |
| `treatment_name` | Human-readable treatment label | `.WHX` TNAME |
| `nitrogen_rate_kg_ha` | Nitrogen rate stated in treatment label | Label-derived where stated; not a parse of all timed fertilizer events |
| `irrigated` | Boolean flag inferred from treatment label | Label-derived; does not describe dates or irrigation amounts |
| `cultivar`, `cultivar_code` | Variety name and genotype code | Crop FileX; code checked against matching crop cultivar table |
| `weather_station` | Weather file station identifier | `.WHX` WSTA; DSSAT `.WTH` |
| `soil_id` / `soil_profile` | DSSAT profile identifier actually used for the run | `.WHX` by default; replaced by a complete georeferenced installed `.SOL` profile only when it is within 100 km of the selected settlement. The run's `location_provenance_json` records source, coordinates, distance, and fallback status. |
| `location`, `year`, `planting_date` | Experimental site and planting time | `.WHX` and DSSAT Summary output |
| `run_id` | Unique local simulation run | Run metadata; excluded from ML features |
| `dssat_version` | DSSAT release used for run | Installation log/output; provenance only |

## Outcomes

| Field | Definition | Unit / source |
|---|---|---|
| `observed_yield_kg_ha` | Measured maturity yield | kg/ha dry matter, crop-specific A-file `HWAM` |
| `dssat_yield_kg_ha` | Simulated maturity yield | kg/ha, `Summary.OUT` `HWAM` |
| `residual_kg_ha` | `observed_yield_kg_ha − dssat_yield_kg_ha` | kg/ha; positive means observed exceeded DSSAT |
| `predicted_residual_kg_ha` | Model estimate of the residual for a new run | kg/ha; only present when a trusted trained model loads successfully |
| `hybrid_prediction_kg_ha` | `dssat_yield_kg_ha + predicted_residual_kg_ha` | kg/ha; no clipping is applied |

## Pair eligibility and provenance

Training rows must have `pairing_status=eligible`, `pairing_basis=same_experiment_same_treatment`, and `observation_level=experiment_treatment_measurement`. The row also records the same-stem observation A-file and FileX, observed/DSSAT/management treatment IDs, reviewed yield variable, DSSAT run ID and Summary.OUT path, plus the weather files and soil profile staged for that run. Training rejects aggregate regional averages, incomplete provenance, duplicate experiment/treatment pairs, or residuals not equal to measured minus simulated yield.

Residual training is multi-crop. `src/dssat/data/observations.py` lists the currently admitted crop codes; each must have measured `HWAM` paired with simulated `HWAM` in kg/ha dry matter. This is deliberately narrower than DSSAT simulation support. Crops with only biomass or unreviewed harvested-product observations remain excluded until their output meaning is checked.

## Environmental and management features

`src/features/weather_features.py` summarizes `RAIN`, `TMAX`, `TMIN`, and `SRAD` over the simulated crop-cycle window: total rainfall (mm), mean temperature (°C), mean solar radiation (MJ/m²/day), number of rainy days, number of weather records, and crop-cycle length in days. These are **realized historical full-season values**, so they are retrospective features. They must not be represented as known at sowing or day 30/60.

`src/features/soil_features.py` computes thickness-weighted means from DSSAT soil layers for SLLL, SDUL, SSAT, SBDM, and SLOC, plus deepest layer depth. `-99` soil sentinels are excluded rather than imputed.

## Candidate model features

Shared categorical features: `crop`, `cultivar`, `weather_station`, `soil_id`.

Shared environmental/management numeric features: weather aggregates, soil profile summaries, `nitrogen_rate_kg_ha`, and `irrigated`.

- Hybrid residual model additionally sees `dssat_yield_kg_ha`.
- ML-only yield model uses the same eligible feature set but excludes `dssat_yield_kg_ha`.
- Both candidates exclude observed outcome, residual target, treatment/run identifiers, and held-out group identifiers.
- One-hot encoding is fitted inside the training fold by a scikit-learn Pipeline; unknown held-out categories are ignored safely.

## Leave-one-experiment-out artifacts

`data/evaluation/oof_predictions.csv` has one row per original record:

| Column | Meaning |
|---|---|
| `experiment_id`, `treatment_id` | Original observation keys |
| `held_out_experiment`, `fold_number` | Experiment excluded from this row's model fit |
| `n_train`, `n_test` | Training and held-out record counts for its fold |
| `observed_yield_kg_ha` | Measured reference |
| `dssat_prediction_kg_ha` | DSSAT baseline prediction |
| `ml_only_prediction_kg_ha` | Fold-trained direct-yield histogram-gradient-boosting prediction |
| `predicted_residual_kg_ha` | Fold-trained gradient-boosting residual prediction |
| `hybrid_prediction_kg_ha` | DSSAT yield plus predicted residual |

`metrics.json` reports per-fold, per-experiment, and pooled OOF metrics by candidate: MAE, RMSE, mean signed error (prediction minus observation), R² when at least two observations and target variation make it meaningful, percent within ±10%, denominator for that percent, and count of nonpositive observations excluded from that denominator. Pooled metrics are calculated from the full pooled OOF vector, not an average of group scores.

`metadata.json` records timestamp, dataset filename/hash, fold groups, grouping policy, random seeds, model definitions, feature lists, exclusions, and pandas/NumPy/scikit-learn/joblib versions.

## Farm area

User-entered area must be finite and strictly positive. For acres, hectares = acres × 0.40468564224. Total harvest estimate = yield kg/ha × area ha; tonnes = kg ÷ 1000. This scales the selected experimental scenario; it does not establish that the selected experiment represents the entered field.
