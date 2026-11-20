# Data and evaluation dictionary

## Experiment and source fields

| Field | Meaning | Source / caution |
|---|---|---|
| `experiment_id` | DSSAT crop FileX stem and evaluation key | Crop FileX; validation groups are held out together across five folds |
| `treatment_id` | Treatment number within the experiment | Crop FileX and matching crop observation file; not globally unique |
| `treatment_name` | Human-readable treatment label | `.WHX` TNAME |
| `nitrogen_rate_kg_ha` | Nitrogen rate used as a model feature or requested scenario value | Dataset values may come from treatment labels; web-run application details are in `input_application_json` and `run_metadata_json` |
| `irrigated` | Irrigation indicator used as a model feature or requested scenario value | Dataset values may be inferred from treatment labels; web-run schedule/application status is recorded separately |
| `cultivar`, `cultivar_code` | Variety name and genotype code | Crop FileX; code checked against matching crop cultivar table |
| `weather_station` | Weather file station identifier | `.WHX` WSTA; DSSAT `.WTH` |
| `soil_id` / `soil_profile` | DSSAT profile identifier actually used for the run | `.WHX` by default; replaced by a complete georeferenced installed `.SOL` profile only when it is within 100 km of the selected settlement. The run's `location_provenance_json` records source, coordinates, distance, and fallback status. |
| `location`, `year`, `planting_date` | Experimental site and planting time | `.WHX` and DSSAT Summary output |
| `run_id` | Unique local simulation run | Run metadata; excluded from ML features |
| `dssat_version` | DSSAT release used for run | Installation log/output; provenance only |
| `weather_source` / `weather_metadata_json` | Weather provider and retrieval/validation details, including fallback status | NASA POWER metadata or installed DSSAT station record; JSON remains empty when unavailable |
| `soil_source` / `soil_metadata_json` | Soil profile source and matching/application details | Installed DSSAT profile or matched local profile; JSON remains empty when unavailable |
| `input_file_sha256_json` | SHA-256 fingerprints of every actual staged DSSAT input file, including the scenario FileX and its weather, soil, and genotype dependencies | Hashes are computed from the archived input snapshot; absent files are not listed |
| `residual_model_sha256` | SHA-256 fingerprint of the residual model artifact used for a hybrid result | Empty for DSSAT-only runs |
| `run_metadata_json` | Archived runner metadata, applied input statuses, and DSSAT run details | Serialized from the completed DSSAT run |
| `record_schema_version` | CSV record layout version | Current version: `2`; older rows are migrated with blank values for new fields |

## Outcomes

| Field | Definition | Unit / source |
|---|---|---|
| `observed_yield_kg_ha` | Measured maturity yield | kg/ha dry matter, crop-specific A-file `HWAM` |
| `dssat_yield_kg_ha` | Simulated harvested-product value, after unit normalization | kg/ha, `Summary.OUT` `HWAM` |
| `dssat_yield_variable` | Source DSSAT yield variable | `HWAM` for the currently reviewed crop mappings |
| `dssat_yield_definition` | Crop-aware harvested product represented by the source field | Explicitly mapped by crop; unreviewed crops are labeled as raw HWAM |
| `dssat_yield_unit` / `dssat_yield_basis` | Canonical unit and mass basis used for comparison | Currently `kg/ha` and `dry_matter` for reviewed targets; no conversion is needed for HWAM |
| `yield_definition_verified` | Whether this crop's harvested-product interpretation is reviewed | False for unmapped crops; raw output must not be treated as yield, area-scaled, or hybrid-adjusted |
| `residual_kg_ha` | `observed_yield_kg_ha − dssat_yield_kg_ha` | kg/ha; positive means observed exceeded DSSAT |
| `predicted_residual_kg_ha` | Model estimate of the residual for a new run | kg/ha; only present when a trusted trained model loads successfully |
| `hybrid_prediction_kg_ha` | `dssat_yield_kg_ha + predicted_residual_kg_ha` | kg/ha; no clipping is applied |

## Pair eligibility and provenance

Training rows must have `pairing_status=eligible`, `pairing_basis=same_experiment_same_treatment`, and `observation_level=experiment_treatment_measurement`. The row also records the same-stem observation A-file and FileX, observed/DSSAT/management treatment IDs, reviewed yield variable, DSSAT run ID and Summary.OUT path, plus the weather files and soil profile staged for that run. Training rejects aggregate regional averages, incomplete provenance, duplicate experiment/treatment pairs, or residuals not equal to measured minus simulated yield.

Residual training is multi-crop. `src/dssat/data/observations.py` lists the currently admitted crop codes; each must have measured `HWAM` paired with simulated `HWAM` in kg/ha dry matter. The result UI and CSV now carry a crop-specific product description, source variable, unit, mass basis, and review flag. For these 17 mapped crops, both sides already use kg/ha, so the canonical conversion is the identity (`canonical kg/ha = source kg/ha × 1`). Peanut `HWAM` is specifically the dry seed yield; `PWAM` is pod yield and is not substituted. DSSAT simulations for other crops remain possible, but their raw `HWAM` is explicitly unverified and cannot be area-scaled or hybrid-adjusted. This remains narrower than DSSAT simulation support; biomass and unreviewed harvest fields are not treated as yield.

## Environmental and management features

Validation derives a `validation_region` when a source location includes country and first-level administrative area. Five-fold grouped evaluation keeps canonical regions together; for records without canonical geography it groups by exact DSSAT site label, then by experiment if the location is missing. Those fallback records count in pooled out-of-fold metrics, but never as geographic evidence for hybrid approval. A crop-region receives hybrid approval only if that crop has at least three distinct canonical regions and its held-out regional MAE is lower than DSSAT's. Runtime location must match an approved canonical region exactly; absent coverage or insufficient regional evidence means DSSAT-only output.

`src/features/weather_features.py` summarizes `RAIN`, `TMAX`, `TMIN`, and `SRAD` over the simulated crop-cycle window: total rainfall (mm), mean temperature (°C), mean solar radiation (MJ/m²/day), number of rainy days, number of weather records, and crop-cycle length in days. These are **realized historical full-season values**, so they are retrospective features. They must not be represented as known at sowing or day 30/60.

`src/features/soil_features.py` computes thickness-weighted means from DSSAT soil layers for SLLL, SDUL, SSAT, SBDM, and SLOC, plus deepest layer depth. `-99` soil sentinels are excluded rather than imputed.

## Candidate model features

Shared categorical features: `crop`, `cultivar`, `weather_station`, `soil_id`.

Shared environmental/management numeric features: weather aggregates, soil profile summaries, `nitrogen_rate_kg_ha`, and `irrigated`.

Unknown numeric feature values remain missing; the dataset builder does not substitute zero or infer values from district averages. The histogram gradient boosting residual model handles missing numeric features natively. Measured and DSSAT yields, pairing identifiers, categorical identifiers, and provenance must still be present and valid.

- Hybrid residual model additionally sees `dssat_yield_kg_ha`.
- ML-only yield model uses the same eligible feature set but excludes `dssat_yield_kg_ha`.
- Both candidates exclude observed outcome, residual target, treatment/run identifiers, and held-out group identifiers.
- One-hot encoding is fitted inside the training fold by a scikit-learn Pipeline; unknown held-out categories are ignored safely.

## Five-fold grouped evaluation artifacts

`data/evaluation/oof_predictions.csv` has one row per original record:

| Column | Meaning |
|---|---|
| `experiment_id`, `treatment_id` | Original observation keys |
| `validation_region`, `validation_group`, `validation_group_type` | Canonical geography when available and the group used to keep records together during validation |
| `held_out_region`, `held_out_groups`, `held_out_experiment`, `fold_number` | Region/groups/experiments held out for this row's model fit |
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
