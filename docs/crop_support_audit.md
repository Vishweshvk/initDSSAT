# DSSAT crop support audit

Run this from the project root after adding or changing a DSSAT installation:

```powershell
python -m cli.main audit-crops
python -m cli.main audit-crops --run-probes
python -m cli.main audit-experiments --run-probes
python -m cli.main audit-experiments --all-treatments --resume
python -m cli.main audit-observations
```

The first command inventories crop folders, parseable FileX experiments, and linked inputs. The second also tries experiments with complete inputs until it finds one DSSAT run that produces a positive `HWAM`, or exhausts that crop's input-ready experiments. Runs are isolated and archived under `data/runs/`; the DSSAT installation itself is not edited.

The audit creates:

- `data/diagnostics/crop_support_audit.json` — detailed resource status, probe attempts, FileX files that could not be parsed, and observation-file candidates.
- `data/diagnostics/crop_support_audit.csv` — one summary row per crop folder.
- `data/diagnostics/experiment_support_audit.json` and `.csv` — one row per discovered experiment, linked input status, treatment count, observation-file match, and optional DSSAT probe result. Probe mode runs the first treatment of every input-ready experiment; `--all-treatments` checks all treatment IDs and can create a large run archive.
- `data/diagnostics/observation_file_audit.json` and `.csv` — parses same-stem crop A-files, records measured fields and valid treatment counts, and compares measured/simulated values for experiments with saved probes. Positive overlaps are candidate fields only: a shared biomass or by-product field is not automatically the crop's harvest-yield target.

## Status meanings

- `discovered`: at least one FileX experiment was parsed.
- `experiments_input_ready`: experiment has a weather file, soil profile, cultivar table, model module, and treatment according to the current project validator.
- `passed`: one representative simulation produced a valid positive `HWAM` in `Summary.OUT`.
- `yield_output_unrecognized`: DSSAT completed, but the run did not produce a valid positive `HWAM`; the crop may use a different yield field or may not be a harvest-yield experiment (for example, Fallow).
- `failed`: every candidate run for the crop failed in DSSAT; inspect each probe's archived `logs/`, `ERROR.OUT`, and `WARNING.OUT`.
- `skipped_no_input_ready_experiment`: the crop folder has experiments, but none has all resources required by the current validator.

Observation candidates are counted by matching each FileX stem to its same-stem DSSAT A-file (for example, `ABC123.WHX` and `ABC123.WHA`). Dataset training is multi-crop and uses the explicit registry in `src/dssat/data/observations.py`. The current conservative registry allows measured `HWAM` paired with simulated `HWAM`, both in kg/ha dry matter; it does not treat biomass (`CWAM`) or other candidate fields as crop yield. Crop runs without a reviewed target remain supported for DSSAT simulation and are excluded only from residual training. The builder pairs exact treatment IDs with a successful unmodified DSSAT run and records the source/run/input provenance; records without that evidence do not enter training.

## Current workspace audit

The project's bundled DSSAT installation has 48 crop/model data folders (including Fallow) and 270 parseable crop experiments. Of those experiments, 173 pass the current linked-input check. The representative probes found 32 crop models that produced a positive `HWAM`; six completed but did not provide a valid `HWAM` under the current generic output check; and ten folders had no experiment with all required inputs. The audit found 210 crop-specific observation A-files, with 208 matching discovered experiment stems. An A-file match is only an observation candidate: the current dataset builder admits a record only when the crop has a reviewed yield mapping and the exact treatment has a successful DSSAT run. The current paired dataset contains 640 records across 17 reviewed crops.

The broader experiment inventory also includes ClimateChange, Sequence, and YieldForecast examples: 298 experiments total, 195 with the current required inputs, and 1,244 logical run selections across those input-ready experiments. Sequence (`.SQX`) files are treated as one run per sequence treatment number; their rotation components are not independent treatments. `audit-experiments --run-probes` runs the first valid treatment of every input-ready experiment and records positive output candidates without assuming that every crop's yield is `HWAM`. Use `audit-experiments --all-treatments` for full treatment coverage. Every attempted treatment is checkpointed in `data/diagnostics/experiment_support_audit.checkpoint.json`; resume an interrupted sweep with `audit-experiments --all-treatments --resume`. The per-crop audit remains a quicker summary; the experiment-level report is the detailed source for diagnosing failures and linking A-files.

The previous sweep reported 63 failures because the parser exposed rotation components as treatments 110–156. DSSAT sequence inputs use one treatment number for the full rotation, with each crop step represented by its rotation-component number. The parser now presents one logical sequence treatment for each `.SQX` file. The rebuilt audit contains 1,244 logical run selections and no failed selections; fresh DSSAT runs of Cotton `CHWC0012` and `CHWC0014` and Maize `MSKB8902` all completed under treatment 1. This confirms sequence-run compatibility only; a multi-crop rotation summary is not automatically a single crop's harvest yield or an eligible residual-training label. The observation-file audit parsed all 211 same-stem A-files; 145 have a saved representative probe and 122 have positive measured/simulated overlaps among candidate fields. These overlaps are evidence for candidate output variables only, not crop-specific yield approval.

The runner now stages weather stations and cultivar tables referenced by all crop/treatment records, archives nested outputs, and parses DSSAT `.OSU` seasonal summaries. The full audit results are in `data/diagnostics/experiment_support_audit.json` and `.csv`. Do not treat this compatibility sweep as scientific validation: crop-specific harvested-product variables, units, management mapping, and calibration still require review before residual model training.

Treat the audit as a compatibility inventory, not as proof that every crop's management inputs, yield variable, or scientific calibration is correct. After adding missing linked files, rerun the audit. After adding crop-specific output parsers, update the probe to validate the appropriate harvested-product variable for each crop.

## Crop-aware yield definitions (Task 10)

The current reviewed residual-training roster contains 17 crop codes. For these, the paired A-file and `Summary.OUT` both provide `HWAM`, stored as kg/ha dry matter, with an explicit harvested-product description in `src/dssat/data/observations.py`. Peanut was added after checking its measured `.PNA` records against successful simulation output and confirming from DSSAT's CROPGRO harvest code that `HWAM` is dry seed yield while `PWAM` is dry pod yield. The yield normalization is therefore `kg/ha canonical = HWAM kg/ha × 1`; no unit conversion is applied. The processed paired dataset records each crop-specific definition, unit, basis, and review flag. Scenario CSV records preserve the corresponding output metadata, and the result page names the crop-specific product. DSSAT may still return raw `HWAM` for other crops, but without a reviewed mapping the app marks it unverified and suppresses hybrid and area-scaled harvest calculations. Extending the roster requires checking that crop's A-file measurement and model output semantics first; DSSAT simulation availability alone is not yield validation.
