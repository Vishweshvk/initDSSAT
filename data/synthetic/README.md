# Synthetic ML demonstration data

`demo_residual_dataset.csv` contains 2,000 deterministic bootstrap rows built
from the project's 640 authentic DSSAT/measured-yield pairs. Its columns include
the current model features, the residual and measured-yield targets, and fields
that label each row synthetic and identify its sampled source record.

This is useful for showing the expected data shape and for a clearly labeled
demo. It does not add new field observations, and it is not an accuracy dataset.
The production trainer intentionally rejects it because it has no verified,
row-specific DSSAT run and matching measured observation. Keep using
`data/processed/dssat_hybrid_dataset.csv` for production training and evaluation.

Regenerate it from the project root with:

```powershell
python scripts\create_synthetic_ml_demo_dataset.py
```

The generation seed, source hash, method, limitations, feature list, and public
data references are recorded in `demo_residual_dataset_metadata.json`.
