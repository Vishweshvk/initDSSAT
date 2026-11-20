# Multi-crop DSSAT–ML Research Console

This is a **historical DSSAT simulation and research benchmark**. It discovers crop experiments present in the configured DSSAT installation, runs them with the matching DSSAT-CSM module, pairs simulated yields with measured outcomes where crop-specific targets are reviewed, and evaluates a histogram-gradient-boosting residual model. The current local paired dataset contains 640 treatment records across 17 reviewed crops. Other DSSAT crops can still be simulated, but their harvest outputs remain unverified until their crop-specific observed and simulated yield definitions are reviewed. It is not a validated farm forecast at sowing or during the season.

## What the project currently does

1. Discovers supported DSSAT FileX experiments and their linked weather, soil, cultivar, and management information.
2. Runs a selected treatment with the installed DSSAT executable in a unique project run folder. The installed DSSAT data are read from their configured location; run inputs and output logs are kept in the project archive.
3. Reads the configured crop-specific DSSAT harvest output and pairs it with the matching crop observation file when a reviewed yield mapping is available.
4. Builds a verified paired dataset with residual `observed yield − DSSAT yield` and documented historical season weather and soil features.
5. Recomputes a five-fold grouped retrospective benchmark for DSSAT alone, ML-only yield, and DSSAT plus a learned gradient-boosting residual.
6. Lets a user scale an experimental yield per hectare to a selected area. That arithmetic does not make an experiment representative of an arbitrary farm.

The web form asks for state, a GeoNames settlement, crop, season, planting date, soil texture context, irrigation, nitrogen, and plant density. It uses the place coordinates to select a nearby installed experiment, request validated NASA POWER weather when possible, and match a complete georeferenced DSSAT soil profile when one is available within 100 km. Planting date and field-management values are applied to a staged FileX only when the required records exist; the result page labels each input as applied or context-only. Season and broad soil texture are currently context-only. The app selects from discovered installed experiments; a crop is available only when its linked DSSAT data can run. Evaluation uses completed-season historical weather, so its metrics are not sowing-time or day-30/day-60 forecast accuracy.

## Setup on Windows PowerShell

Use Python 3.10 or newer and install DSSAT-CSM v4.8.6 with each target crop’s experiment, Weather, Soil, and Genotype data. From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The app checks a `DSSAT48` folder in the project first, then `C:\DSSAT48` and `C:\Program Files\DSSAT48`. Set `.env` if your installation or data folder is elsewhere. For the common installation:

```dotenv
DSSAT_HOME=C:\DSSAT48
DSSAT_DATA=C:\DSSAT48
PROJECT_DATA_DIR=data
MODEL_DIR=models
```

The app reads `.env` from the project root without overriding values already set in the PowerShell environment. Relative paths in `.env` are resolved from the project root, regardless of the shell's current directory. Blank optional values in `.env.example` are ignored.

## Commands

```powershell
# Check configuration and dependencies
python -m cli.main doctor

# Inventory installed crop folders and their linked input files
python -m cli.main audit-crops

# Also run one representative DSSAT simulation for every crop with complete inputs
python -m cli.main audit-crops --run-probes

# Check one treatment in every discovered experiment
python -m cli.main audit-experiments --run-probes

# Exhaustively probe every treatment (large run; archives every simulation)
python -m cli.main audit-experiments --all-treatments

# Resume an interrupted treatment audit from its saved checkpoint
python -m cli.main audit-experiments --all-treatments --resume

# Parse crop observation A-files and compare candidate fields to saved probes
python -m cli.main audit-observations

# List experiments; status distinguishes runnable simulations and observed data
python -m cli.main list-experiments --crop all

# Validate resources needed for a simulation (does not require .WHA)
python -m cli.main validate-experiment --experiment RORO7401

# Also require a parseable observed-yield file for dataset building
python -m cli.main validate-experiment --experiment RORO7401 --for-dataset

# Run one installed treatment
python -m cli.main run-experiment --experiment RORO7401 --treatment 1

# Rebuild measured-versus-simulated research rows from installed examples
python -m cli.main build-dataset --crop all --continue-on-error

# Recompute grouped out-of-fold evaluation; never uses the all-data model
python -m cli.main evaluate --crop all --dataset data/processed/dssat_hybrid_dataset.csv

# Print previously saved evaluation without recomputing it
python -m cli.main show-evaluation

# Fit a separate deployable residual model on all validated dataset rows
python -m cli.main train-residual --crop all --dataset data/processed/dssat_hybrid_dataset.csv

# Simulate and estimate plot total; area units can be hectares or acres
python -m cli.main predict --experiment RORO7401 --treatment 1 --area 2 --area-unit acres

# Launch the offline Flask demonstration
python app.py
```

Open `http://127.0.0.1:5000`. The interface has Prediction, Evaluation, Real-world outcomes, and Research readiness pages. It does not offer unimplemented sowing or in-season forecasting controls. If an evaluation run fails, inspect its message and the relevant run's `logs/` and `metadata.json`.

`audit-crops --run-probes` writes `data/diagnostics/crop_support_audit.json` and `.csv`. A probe is a compatibility check for one installed experiment per crop; it does not validate crop yield definitions or field-level accuracy. See [crop support audit](docs/crop_support_audit.md) for interpreting the report and preparing crop-specific observation parsing.

## Benchmark design and artifacts

`evaluate` uses five-fold grouped out-of-fold validation. Canonical country/admin-1 groups stay together; where that metadata is missing it groups by the exact DSSAT site label, then by experiment if no location is available. Fallback groups contribute to pooled metrics but never count as geographic evidence for hybrid approval. Fold-local scikit-learn pipelines fit `OneHotEncoder` only on each training fold. Every observation gets exactly one out-of-fold prediction. Crop-region decisions are reported separately; the hybrid is approved only when a crop has at least three independent canonical regions and its held-out regional MAE is below DSSAT's. The three candidates are:

- **DSSAT alone:** use `dssat_yield_kg_ha` directly.
- **ML-only:** histogram gradient boosting predicts observed yield from eligible crop/environment/management features, excluding DSSAT yield and any outcome, residual, run, or group identifier.
- **Hybrid:** histogram gradient boosting learns `observed − DSSAT` from paired records; its predicted residual is applied only when the exact crop-region is approved by held-out evaluation.

Artifacts are written to `data/evaluation/`:

- `oof_predictions.csv` — one held-out prediction row per dataset record.
- `metrics.json` — grouped folds, canonical crop-region decisions, and pooled out-of-fold metrics calculated over all predictions directly.
- `metadata.json` — dataset SHA-256, features, fold/grouping design, random seeds, dependency versions, and evaluation timestamp.

Reported metrics include MAE, RMSE, signed bias (prediction minus observation), mathematically meaningful R², and percent within ±10% of positive measured yields. Nonpositive measured yields are counted and excluded from the percent-within-tolerance denominator. `evaluate` always recomputes. `show-evaluation` only reads the saved report.

`train-residual` is deliberately separate: it first backs up prior model artifacts under `models/backups/`, then fits the final model on all validated rows. It does not create evaluation metrics. Use out-of-fold evaluation—not the all-data fitted artifact—to compare methods.

### Import DES aggregate outcomes for context

The `/outcomes` page displays imported DES Area, Production & Yield values as district/state context. Download a DES report and import it with `python -m cli.main import-des-apy --source <file> --yield-unit kg/ha` (or pass explicit area/production units when yield must be derived). The aggregate records are kept separate and are never residual-training targets. See [DES APY import instructions](data/external/des_apy/README.md).

## Two-minute professor demonstration

From PowerShell in the project directory, run the prepared walkthrough:

```powershell
.\demo.ps1
```

It checks the local installation, lists available experiments, validates the selected experiment, refreshes the benchmark, and starts the local web app. Override the defaults if needed, for example `& .\demo.ps1 -Experiment RORO7401 -Treatment 1 -Area 3 -AreaUnit hectares`.

1. Run `python -m cli.main doctor` and show that Python, DSSAT, and crop data are detected.
2. Run `python -m cli.main list-experiments --crop all`; explain that treatment IDs belong to each experiment.
3. Open the web app and submit an experiment/treatment with a small area. Show DSSAT yield per hectare, the residual/hybrid if available, and the area-scaled estimated harvest.
4. Point out the historic full-season weather warning and explain that the area calculation does not transfer the experiment to another field.
5. Open Evaluation. Compare the pooled held-out DSSAT, ML-only, and hybrid results. Say explicitly whether hybrid MAE improved; explain that the benchmark uses five grouped folds and keeps each validation group together.
6. Open Research readiness to show the future sowing/day-30/day-60 direction, data gaps, and the 90%-within-10% target as **unachieved**.

## Research and product limitations

- The current paired benchmark has 640 experiment-treatment records across 17 crops, but it is not independent field validation and does not support a broad farm-level forecast claim.
- Current validation is retrospective, using full-season historical weather unavailable at sowing.
- Installed examples do not establish performance in an eventual target region; region selection depends on audited data.
- ML-only performance may be poor on unseen experiments; show measured results without tuning against held-out groups.
- Some crop-management mappings depend on the FileX records available in a selected experiment; the result page reports which requested values DSSAT actually received.
- Imported DES district/state outcomes provide aggregate context only. They are not included in residual-model training; independent field-level outcomes remain needed for operational validation.
- No calibrated uncertainty, recommendation engine, or real-time seasonal forecast is implemented.
- A lower MAE or any target tolerance must be confirmed on independent, suitable field data before being presented as operational performance.

See [architecture](docs/architecture.md), [data dictionary](docs/data_dictionary.md), [DSSAT setup](docs/dssat_setup.md), and [FINAL_CHECK.md](FINAL_CHECK.md) for further detail.
See [crop and location data requirements](docs/crop_and_location_data.md) before preparing custom state/city scenarios.
