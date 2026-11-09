# GeoNames India place lookup data

Download the following two source files from the [GeoNames download server](https://download.geonames.org/export/dump/):

- `IN.zip` — India place records. Keep the ZIP intact; the importer reads `IN.txt` from inside it.
- `admin1CodesASCII.txt` — maps GeoNames state/administrative codes to readable state names.

Put both files in `data/external/geonames/downloads/`. Then, from the project root, run:

```powershell
python scripts/import_geonames.py
```

The importer writes `data/external/geonames/processed/india_places.csv` and a `.metadata.json` sidecar. It retains populated places (`feature_class=P`) with name variants, coordinates, state code/name, population, and timezone. It does not geocode an exact farm or replace a DSSAT soil profile; coordinates are the GeoNames point for the named settlement.

GeoNames data are licensed CC BY 4.0 and provided as-is. Keep the source attribution in any UI or exported dataset: [GeoNames](https://www.geonames.org/export/), [license](https://creativecommons.org/licenses/by/4.0/). Place coverage and spellings can be incomplete; confirm a user's actual field location before using the coordinate for soil or weather extraction.
