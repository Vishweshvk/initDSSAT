# Multi-crop DSSAT–ML Research Console

This is a **historical DSSAT simulation and research benchmark**, not a validated farm forecast. The command-line tools can discover and run compatible experiments in the configured DSSAT installation. The web form is currently restricted to Maharashtra and shows only crops on its Maharashtra shortlist that also have a successful local DSSAT support-audit record. A crop module being installed does not by itself mean a runnable experiment or a reviewed yield definition is available.

The app runs DSSAT first. A histogram-gradient-boosting model may then estimate a residual (`measured yield − DSSAT yield`) from eligible paired observations. It displays a corrected value only when the crop's harvested-product mapping, model artifact, and matching held-out evaluation pass the project's checks. Otherwise, it shows the DSSAT result alone. A fresh checkout does not include the local DSSAT installation, generated training/evaluation files, or fitted model artifacts; follow the setup and data instructions below before expecting a hybrid result.

## What the project currently does

1. Discovers supported DSSAT FileX experiments and their linked weather, soil, cultivar, and management information.
2. Runs a selected treatment with the installed DSSAT executable in a unique project run folder. The installed DSSAT data are read from their configured location; run inputs and output logs are kept in the project archive.
3. Reads the configured crop-specific DSSAT harvest output and pairs it with the matching crop observation file when a reviewed yield mapping is available.
4. Builds a verified paired dataset with residual `observed yield − DSSAT yield` and documented historical season weather and soil features.
5. Recomputes a five-fold grouped retrospective benchmark for DSSAT alone, ML-only yield, and DSSAT plus a learned gradient-boosting residual.
6. Lets a user scale an experimental yield per hectare to a selected area. That arithmetic does not make an experiment representative of an arbitrary farm.

The web form fixes the state to Maharashtra and offers GeoNames cities/towns with population at least 5,000. It asks for crop, season, planting month/day, soil-texture context, irrigation, nitrogen, and plant density. **It does not ask for a year.** It maps the selected month/day to the planting year recorded in the selected DSSAT FileX experiment, so DSSAT uses a year present in that experiment's historical weather rather than a typed current year. For a single-station experiment, it requests NASA POWER weather for the mapped date when possible; otherwise it uses installed experiment weather and records the fallback. It matches a complete georeferenced DSSAT soil profile only when one is within 100 km of the selected settlement. The result page reports which values were applied and which were context only. Season and the broad soil-texture selection are currently context only. The selected city is a settlement point, not an exact field location. Evaluation uses realized full-season historical weather, so its metrics are not sowing-time or day-30/day-60 forecast accuracy.

## Data sources, external files, and APIs

| Need | Source / file | How this project uses it |
| --- | --- | --- |
| DSSAT crop simulations | Local DSSAT-CSM 4.8.6 install: `DSCSM048.EXE`, `DSSATPRO.v48`, target crop experiment/FileX files, matching crop model module, `Weather/*.WTH`, `Soil/*.SOL`, and `Genotype` cultivar files | Required locally. The app stages copies for a run; it does not edit the installed originals. The `DSSAT48/` folder is gitignored: install DSSAT separately and set `DSSAT_HOME` / `DSSAT_DATA`. Do not push DSSAT installation data unless its license permits redistribution. |
| City names and coordinates | GeoNames `IN.zip` and `admin1CodesASCII.txt` from [GeoNames downloads](https://download.geonames.org/export/dump/) | Offline lookup. If the processed lookup is missing, download both files to `data/external/geonames/downloads/`, then run `python scripts/import_geonames.py`. The resulting processed CSV/database are local generated files. Attribution: [GeoNames](https://www.geonames.org/export/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Coordinates represent settlements. |
| Location weather | [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), endpoint `https://power.larc.nasa.gov/api/temporal/daily/point` | Runtime internet request in ICASA format for a single-station experiment; no API key is configured. The app validates daily DSSAT inputs (`SRAD`, `TMAX`, `TMIN`, `RAIN`) over the required simulation window, caches accepted responses under `data/weather_cache/`, and falls back to installed DSSAT weather when the request or coverage is unsuitable. POWER is gridded weather, not a field station. |
| Soil profiles | Georeferenced `.SOL` profiles in the local DSSAT installation (including WISE profiles); for additional Indian profiles see [ICAR-NBSS&LUP BHOOMI](https://bhoomigeoportal-nbsslup.in/IndianNationalSoilGrid/) | The app applies a complete profile within 100 km of the chosen settlement, when available. The selected profile and source are recorded. Soil texture alone is not converted into a DSSAT hydraulic profile; SoilGrids is not an active source for building `.SOL` files in this project. |
| Residual-model observations | DSSAT crop observation A-files (`.WHA` and crop-specific equivalents such as `.CHA`) or independently measured field-trial data paired to the exact crop, experiment/treatment, weather, soil, and management | Required to train and evaluate a meaningful residual model. A DSSAT installation may include observation files, but these are experimental measurements, not automatically paired labels for arbitrary user-entered scenarios. Inspect and build the eligible dataset with the CLI; do not use unpaired or district-average yields as run residuals. [GARDIAN](https://gardian.bigdata.cgiar.org/) is a possible source to search for field trials, but its records require manual review and this app does not automatically import them. |
| Regional outcome context | [DES Area, Production & Yield portal](https://data.desagri.gov.in/website/crops-apy-report-web); optional [OGD district-wise crop production catalog](https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0) | Optional downloaded CSV/Excel report. Place it under `data/external/des_apy/downloads/` and import using the instructions in [`data/external/des_apy/README.md`](data/external/des_apy/README.md). These state/district aggregates are context only; they are **not** residual-training labels. |

No API credentials are currently required. NASA POWER is the only live data API called by a prediction request; GeoNames and DES data are locally downloaded/imported. IMD gridded weather and SoilGrids are documented alternatives/research sources, not active simulation inputs. See [crop and location data requirements](docs/crop_and_location_data.md) for coverage limits and source details.

### What to provide on a new machine

1. Install DSSAT-CSM 4.8.6 and the crop experiment, Weather, Soil, and Genotype files needed for the crops you want to run; configure `.env` as shown below.
2. If the city list is absent, download `IN.zip` and `admin1CodesASCII.txt` as described above and run the GeoNames importer.
3. For ML correction, make sure matching crop observation A-files or compatible measured field-trial records exist, then rebuild and validate the paired dataset and train/evaluate the model using the commands below. `data/processed/`, `data/evaluation/`, and `models/` are gitignored local artifacts; without compatible artifacts, the app intentionally returns DSSAT-only output.
4. DES APY reports are optional and are only needed for the `/outcomes` context page. They do not enable ML correction.

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

Open `http://127.0.0.1:5000`. The interface has Prediction, Evaluation, Real-world outcomes, and Research readiness pages. If port 5000 is occupied, start the app on another port, for example:

```powershell
python -c "from app import app; app.run(host='127.0.0.1', port=5058, debug=False)"
```

It does not offer unimplemented sowing or in-season forecasting controls. If a run fails, inspect its message and the relevant run's `logs/`, `output/WARNING.OUT`, and `metadata.json`.

Each completed web run adds one row to `<project-data-dir>/predictions/scenario_results.csv` (`data/` by default) and archives its staged inputs, outputs, logs, and metadata under `<project-data-dir>/runs/<run-id>/`. These generated run records are local and gitignored. The CSV records the mapped DSSAT planting date, requested/applied inputs, crop-specific yield definition and units, location and weather/soil provenance, DSSAT output, and hybrid details when available. SHA-256 fingerprints cover archived DSSAT inputs and run-control files, the DSSAT executable, and the residual model when used. Missing values stay blank; older rows retain available information when the CSV schema is upgraded.

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
