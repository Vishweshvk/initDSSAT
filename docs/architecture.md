# Architecture and research workflow

## Current program boundary

```text
Web form or CLI
    |
    +-- src/dssat/config.py
    |     project-root paths + optional project-root .env
    |
    +-- repository.py: discover supported crop FileX, treatment/cultivar, weather, soil
    |       | simulation validation does not require observed .WHA
    |       + observed-dataset validation additionally requires parseable .WHA
    |
    +-- runner.py: copy experiment input + DSSAT profile to unique data/runs/<id>/
    |       | execute installed DSCSM048.EXE with the crop-specific model module
    |       + archive Summary/PlantGro/ERR outputs under output/ and stdout/stderr under logs/
    |         metadata.json points at those final relative output paths
    |
    +-- parser.py: Summary.OUT HWAM (kg/ha); optional PlantGro.OUT
    |
    +-- prediction.py: historical-season features + evaluated gradient-boosting residual model + farm-area arithmetic
    |
    +-- evaluation.py: fresh five-fold grouped out-of-fold validation
            + each location group stays within a single fold
            + DSSAT-only, ML-only yield, and DSSAT + residual predictions
            + exactly one out-of-fold prediction per record
            + data/evaluation/{oof_predictions.csv,metrics.json,metadata.json}
```

## Input and validation flow

- The experiment FileX (`.WHX`) holds experimental design, treatment descriptions, cultivar, soil identifier, and weather station.
- The linked `.WTH`, `.SOL`, and crop-specific cultivar table support the DSSAT simulation.
- A runnable simulation needs FileX, weather, soil, a valid cultivar, and treatments. It does not require observed yield.
- A measured-versus-simulated training row additionally needs a reviewed crop-specific measured-yield mapping and a same-stem DSSAT A-file. Reviewed outputs carry a crop-specific harvested-product definition, canonical unit/basis, and verification flag; unmapped crops expose only raw DSSAT HWAM and are not area-scaled or ML-adjusted.
- Dataset builder pairs exactly one observed row with one successful DSSAT run for the same crop experiment and treatment, verifies the treatment in Summary.OUT, and records the source A-file, FileX, run, weather, soil, and management inputs.
- The residual is `observed_yield_kg_ha - dssat_yield_kg_ha`; training and evaluation reject records without this exact-pair provenance, and aggregate district/state averages are ineligible.
- Residual training is multi-crop. The explicit registry in `src/dssat/data/observations.py` currently admits only crops with measured `HWAM` and matching DSSAT `HWAM` output in kg/ha dry matter. This is a conservative data-availability gate, not a claim that every DSSAT crop has paired labels. Crops with only `CWAM` biomass or unreviewed harvest variables remain eligible for DSSAT runs but not residual training.
- Weather and soil feature functions are in `src/features/`. Weather aggregates span the crop cycle in the recorded historical season.

`data/manifests/<crop>_experiments.json` records separate `simulation_status` and `dataset_status` fields. The old `status` field remains as a compatibility alias for the requested validation mode.

## Evaluation versus deployment model

The evaluation code in `src/evaluation.py` runs five-fold grouped out-of-fold validation. Canonical country/admin-1 groups stay together; when that metadata is absent, records are grouped by exact DSSAT site label or, if needed, experiment. These fallback groups contribute to pooled metrics but are not geographic evidence for hybrid approval. A crop-region is approved only when that crop has at least three independent canonical regions and the held-out regional hybrid MAE is lower than DSSAT's. `src/prediction.py` requires an exact crop-region approval before serving a hybrid; otherwise the page explains why it shows DSSAT alone. Pipelines from `src/modeling.py` fit category encoders and gradient boosting models inside each training fold.

| Candidate | Features | Target |
|---|---|---|
| DSSAT only | DSSAT run | Observed yield used only for scoring |
| ML only | Environment/soil/management; no DSSAT yield | Observed yield |
| Hybrid | Same eligible features plus DSSAT yield | Residual = observed minus DSSAT |

The output CSV stores actual held-out predictions. `metrics.json` reports fold and per-experiment scores as well as pooled metrics calculated directly over all out-of-fold records. `metadata.json` stores dataset hash, feature lists, experiment groups, fold policy, seeds, library versions, and timestamp. The offline dashboard (`src/dashboard.py`, `/evaluation`) reads these artifacts; the `evaluate` command always recomputes them. `show-evaluation` is the explicit load-only CLI command.

`train-residual` is separate and fits a final deployable residual model on all valid data only after previous model files are backed up. It must not be used to produce benchmark predictions. The saved model metadata describes fit scope and training data but contains no held-out performance claim.

## Farm area and serving flow

`src/prediction.py` validates finite positive farm area, converts acres using `1 acre = 0.40468564224 hectares`, and applies:

```text
total_harvest_kg = yield_kg_ha × area_hectares
total_harvest_tonnes = total_harvest_kg ÷ 1000
```

The Flask `POST /predict` route validates area before launching DSSAT; shows DSSAT output even without a compatible model; and labels ML-adjusted results only when a matching grouped evaluation shows lower MAE than DSSAT alone. Each web run is appended to `data/predictions/scenario_results.csv`, including input-application statuses, weather/soil source metadata, SHA-256 fingerprints for available simulation inputs and any residual model used, and the archived run metadata. Missing files stay absent from the fingerprint map. Schema upgrades preserve older/custom columns and retain a pre-migration backup. The result page reports identifiers, formulas, input files, historic-weather basis, and benchmark comparison when available. It does not expose arbitrary logs or filesystem browsing.

The website SVG charts are generated locally by Python and work without external plotting assets. `/research` summarizes current multi-crop capabilities, validation limits, and the currently imported DES context; it does not present unverified candidate-dataset counts as project records.

## Configuration, run archive, and errors

`DssatConfig.project_directories()` resolves `data/` and `models/` from the actual project root (the directory above `src/`), or from configured paths. Relative paths resolve against that root, not the shell. A simple `.env` reader loads `KEY=VALUE` lines from the project-root `.env`, preserving existing process variables and ignoring blank optional values.

Each `data/runs/<run-id>/` contains staged input, copied `DSSATPRO.v48`, batch file, `output/`, `logs/`, and `metadata.json`. The installed executable is called directly with the run folder as its working directory; original FileX and installed inputs are not edited. Failed startup, nonzero exit, parse failure, and timeout preserve diagnostics. `metadata.json` uses final `output/<filename>` names after files are moved from the DSSAT working root.

## Current field-scenario behavior and remaining forecast work

The current app accepts location and field-management inputs for installed, runnable crop experiments. GeoNames settlement coordinates guide experiment and soil-profile matching; NASA POWER weather is requested when one weather station is unambiguous and complete input coverage validates. Planting date, nitrogen, irrigation, and plant density are written to a per-run FileX copy when the required records exist, and each application status is recorded. Season and broad soil texture remain context-only. If no suitable soil profile is found within 100 km, the experiment's original profile stays in use. These steps make the simulation location-aware, but they do not make the selected settlement an exact farm location or guarantee that an experiment represents the user's field. Forecasts at sowing or during the season still need date-appropriate weather and repeated crop observations. Uncertainty is future work; the 90%-within-±10% independent-field target remains unachieved. See `/research` and `crop_and_location_data.md` for current gaps.
