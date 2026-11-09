# Verification record

Recorded 2026-10-11 using the installed DSSAT-CSM v4.8.6.0 wheat example library.

- Configuration check discovers the installed executable and the Wheat, Weather, Soil, and Genotype resources.
- A real DSSAT integration test ran `KSAS8101`, treatment 1, and produced a valid `HWAM` of **2,010 kg/ha**. The observed record is **2,317 kg/ha**, giving `observed − DSSAT = +307 kg/ha`.
- A separate real CLI run of `RORO7401`, treatment 1, completed with **1,982 kg/ha** simulated and **3,100 kg/ha** observed (`observed − DSSAT = +1,118 kg/ha`).
- The runner stages experiment inputs, all required station-year weather files, and related cultivar `.CUL`/`.ECO`/`.SPE` tables in a short-lived workspace. It archives inputs, outputs, logs, and metadata without modifying the DSSAT installation.
- Full verification: **30 tests passed**, including the installed-DSSAT integration test.
- Recomputed leave-one-experiment-out benchmark: 28 records across three experiment groups. Pooled MAE was **516.11 kg/ha** for DSSAT-only, **1,962.83 kg/ha** for ML-only, and **559.08 kg/ha** for the hybrid. The hybrid did not improve pooled MAE over DSSAT-only.
- This is a retrospective benchmark based on completed-season historical weather. It does not measure sowing-time or in-season forecast accuracy, and three experiment groups are not enough for a broad farm-level performance claim.
- The demo runner is `demo.ps1`; see README for invocation and the web app workflow.

The final residual model is a separate all-data training artifact. It is not used to calculate the leave-one-experiment-out benchmark. `data/evaluation/` stores the out-of-fold predictions, pooled/per-experiment metrics, and metadata.

The installed DSSAT software/library and its license are not bundled. The handoff zip omits `.venv`, generated `data/runs/`, caches, `.env`, and model backups; these are local/generated files.
