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
    +-- evaluation.py: fresh Leave-One-Experiment-Out folds
            + each fold fits its own preprocessing/model on other experiments
            + DSSAT-only, ML-only yield, and DSSAT + residual predictions
            + exactly one out-of-fold prediction per record
            + data/evaluation/{oof_predictions.csv,metrics.json,metadata.json}
```

## Input and validation flow

- The experiment FileX (`.WHX`) holds experimental design, treatment descriptions, cultivar, soil identifier, and weather station.
- The linked `.WTH`, `.SOL`, and crop-specific cultivar table support the DSSAT simulation.
- A runnable simulation needs FileX, weather, soil, a valid cultivar, and treatments. It does not require observed yield.
- A measured-versus-simulated training row additionally needs a reviewed crop-specific measured-yield mapping and a same-stem DSSAT A-file.
- Dataset builder pairs exactly one observed row with one successful DSSAT run for the same crop experiment and treatment, verifies the treatment in Summary.OUT, and records the source A-file, FileX, run, weather, soil, and management inputs.
- The residual is `observed_yield_kg_ha - dssat_yield_kg_ha`; training and evaluation reject records without this exact-pair provenance, and aggregate district/state averages are ineligible.
- Residual training is multi-crop. The explicit registry in `src/dssat/data/observations.py` currently admits only crops with measured `HWAM` and matching DSSAT `HWAM` output in kg/ha dry matter. This is a conservative data-availability gate, not a claim that every DSSAT crop has paired labels. Crops with only `CWAM` biomass or unreviewed harvest variables remain eligible for DSSAT runs but not residual training.
- Weather and soil feature functions are in `src/features/`. Weather aggregates span the crop cycle in the recorded historical season.

`data/manifests/<crop>_experiments.json` records separate `simulation_status` and `dataset_status` fields. The old `status` field remains as a compatibility alias for the requested validation mode.

## Evaluation versus deployment model

The evaluation code in `src/evaluation.py` loops through all observed experiment IDs. One whole ID is excluded as test data per fold. Pipelines from `src/modeling.py` contain the one-hot encoder and histogram gradient boosting model so category encoding is fit only on each training fold.

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

The Flask `POST /predict` route validates area before launching DSSAT; shows DSSAT output even without a compatible model; and labels ML-adjusted results only when a matching grouped evaluation shows lower MAE than DSSAT alone. Each web run is appended to `data/predictions/scenario_results.csv`. The result page reports identifiers, formulas, input files, historic-weather basis, and benchmark comparison when available. It does not expose arbitrary logs or filesystem browsing.

The website SVG charts are generated locally by Python and work without external plotting assets. `/research` presents confirmed direction, planned features, candidate datasets pending audit, and unresolved measurement issues without integrating candidate rows or implying unbuilt forecast actions.

## Configuration, run archive, and errors

`DssatConfig.project_directories()` resolves `data/` and `models/` from the actual project root (the directory above `src/`), or from configured paths. Relative paths resolve against that root, not the shell. A simple `.env` reader loads `KEY=VALUE` lines from the project-root `.env`, preserving existing process variables and ignoring blank optional values.

Each `data/runs/<run-id>/` contains staged input, copied `DSSATPRO.v48`, batch file, `output/`, `logs/`, and `metadata.json`. The installed executable is called directly with the run folder as its working directory; original FileX and installed inputs are not edited. Failed startup, nonzero exit, parse failure, and timeout preserve diagnostics. `metadata.json` uses final `output/<filename>` names after files are moved from the DSSAT working root.

## Intended future, not implemented in the current simulator

The current product direction is multi-crop support and location-aware field simulation. The interface writes planting date, plant population, nitrogen events, and irrigation events into a per-run FileX copy when those fields exist, and records an applied/not-used status. State/city coordinates come from the local GeoNames index. The app can match complete, georeferenced installed DSSAT soil profiles to a selected settlement within 100 km; the FileX soil ID and staged `.SOL` file are changed together and provenance is recorded. If no suitable profile is close enough, the experiment's soil profile stays in use. The selected broad texture remains context only, and gridded texture maps are not converted into hydraulic profiles without validated assumptions. Weather known at each issue date and repeated in-season crop observations must be acquired before sowing/day-30/day-60 forecast points can be validated. Region selection follows audited candidate data. Uncertainty is future work. The 90%-within-±10% independent-field target is unachieved. See the `/research` page and `crop_and_location_data.md` for data needs and gaps.
