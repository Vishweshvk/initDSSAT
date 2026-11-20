# Verification record

## Current saved benchmark — 2026-11-20

The current saved evaluation uses 640 paired treatment records from 93 experiments, grouped into 62 validation groups and 17 canonical regions. It was generated from dataset SHA-256 `d22daaba18c7beb4ab13283053cbf4e89975fd115939e49f86cafd05716f731a`.

- Five-fold grouped out-of-fold MAE: DSSAT-only **522.09 kg/ha**, ML-only **1,307.78 kg/ha**, hybrid **599.05 kg/ha**.
- The hybrid does **not** improve pooled MAE over DSSAT-only. Deployment remains subject to the crop-region approval list; pooled performance alone does not authorize a hybrid result.
- This is retrospective validation using completed-season weather. It is not a sowing-time or in-season forecast and is not independent farm-field validation.

`data/evaluation/metrics.json` and `metadata.json` are the detailed current artifacts. Re-run evaluation after changing the paired dataset; older notes below preserve the initial October verification and should not be read as the current benchmark.

## Historical integration verification — 2026-10-11

Recorded using the installed DSSAT-CSM v4.8.6.0 Wheat example library.

- Configuration check discovers the installed executable and the Wheat, Weather, Soil, and Genotype resources.
- A real DSSAT integration test ran `KSAS8101`, treatment 1, and produced a valid `HWAM` of **2,010 kg/ha**. The observed record is **2,317 kg/ha**, giving `observed − DSSAT = +307 kg/ha`.
- A separate real CLI run of `RORO7401`, treatment 1, completed with **1,982 kg/ha** simulated and **3,100 kg/ha** observed (`observed − DSSAT = +1,118 kg/ha`).
- The runner stages experiment inputs, all required station-year weather files, and related cultivar `.CUL`/`.ECO`/`.SPE` tables in a short-lived workspace. It archives inputs, outputs, logs, and metadata without modifying the DSSAT installation.
- Full verification: **30 tests passed**, including the installed-DSSAT integration test.
- The initial leave-one-experiment-out benchmark had 28 records across three experiment groups. Its metrics were superseded by the current saved benchmark above.
- This initial verification was retrospective and does not measure sowing-time or in-season forecast accuracy.
- The demo runner is `demo.ps1`; see README for invocation and the web app workflow.

The final residual model is a separate all-data training artifact. It is not used to calculate out-of-fold benchmark predictions. `data/evaluation/` stores the out-of-fold predictions, pooled/per-experiment metrics, and metadata.

The installed DSSAT software/library and its license are not bundled. The handoff zip omits `.venv`, generated `data/runs/`, caches, `.env`, and model backups; these are local/generated files.
