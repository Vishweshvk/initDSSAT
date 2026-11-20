# India DES APY context data

Use the official DES Area, Production & Yield reports portal:
<https://data.desagri.gov.in/website/crops-apy-report-web>

The OGD Platform also lists DES's district-wise, season-wise crop production statistics:
<https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0>
That dataset describes area in hectares and production in tonnes; it may not contain a yield field, so the importer can derive kg/ha from those two source columns. Check the source record's coverage and update date before using it for a particular year.

The portal provides report selection by state, district, crop, season, year range, and report format. If it offers CSV or supported Excel `.xlsx`/`.xlsm`, put that file in `downloads/`. The OGD catalog offers a CSV/ZIP alternative; unzip it and import its CSV. Keep original files locally; download and processed report folders are gitignored.

Import with an explicit yield unit when the report contains yield. If the source provides only area and production, supply both source units so yield can be derived transparently:

```powershell
python -m cli.main import-des-apy `
  --source "data/external/des_apy/downloads/report.csv" `
  --area-unit "1000 ha" `
  --production-unit "1000 tonnes"
```

Supported reported-yield units are `kg/ha`, `quintal/ha`, and `tonne/ha`. For area/production derivation, supported area units are `ha` and `1000 ha`; supported production units are `tonnes` and `1000 tonnes`. The conversion is `yield (kg/ha) = production (tonnes) × 1000 / area (ha)`. The importer writes normalized CSV, row-rejection details, and a provenance manifest to `processed/`. It preserves source values/units and identifies whether yield was reported or derived.

These DES records are district/state aggregates and always have `training_eligible=false` and `use_scope=aggregate_context_only`. They must not be merged into `data/processed/*_hybrid_dataset.csv` or used to calculate field-level residual targets. Training residuals still require measured field/experiment yields paired to the same DSSAT crop, treatment, weather, and soil.
