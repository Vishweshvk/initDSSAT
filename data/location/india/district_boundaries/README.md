# India district boundary data

Put the downloaded boundary dataset in `raw/`. Keep the original download and filename unchanged.

## Recommended download

From the National Water Data Portal page for **District Boundary**:

1. Open <https://nwdp.nwic.gov.in/dataset/district-boundary>.
2. Under **Data and Resources**, choose **District NWIC GeoJSON**.
3. Choose **Download** for that resource. The archive is named `district_nwic_geojson.zip`.
4. Save the ZIP to `data/location/india/district_boundaries/raw/district_nwic_geojson.zip`.
5. Keep the downloaded ZIP as the original source. Its extracted GeoJSON is stored at `raw/extracted/district_nwic.GeoJSON`.

The portal describes this resource as 733 district boundaries and identifies the Geological Survey of India as its producer. This copy was downloaded on 2026-11-06. It contains 733 features and 36 state/UT names. The GeoJSON properties include `state`, `stcode`, `district`, and `dtcode`; the recorded CRS is EPSG:7755. District names are not unique on their own (728 distinct names), so the state name/code must be part of every lookup key.

The coordinates are projected, not latitude/longitude. Before using a district centroid for SoilGrids or weather lookup, reproject the geometry to WGS84 (EPSG:4326). Do not read the raw coordinate numbers as latitudes and longitudes. The portal identifies the producer but its page does not state a clear reuse license; preserve attribution and confirm usage terms before distributing this data with the app.

## Alternative Survey of India download

The Survey of India Administrative Boundary Data Base page offers **State/District/Sub District Boundary Data-Base of India**. Its archive is named `State_District_Subdistrict_PAN INDIA.rar`.

1. Open <https://surveyofindia.gov.in/pages/administrative-boundary-data-base-abdb->.
2. Select **State/District/Sub District Boundary Data-Base of India**.
3. Save the archive to `raw/State_District_Subdistrict_PAN INDIA.rar`.

Use one boundary source at a time for the initial integration. Do not download both formats unless the GeoJSON lacks district names or state identifiers needed by the app.

## Expected project layout

```text
data/location/india/district_boundaries/
  README.md
  raw/
    district_nwic_geojson.zip        # original download
    extracted/
      district_nwic.GeoJSON          # extracted source, EPSG:7755
  processed/                         # WGS84 and normalized lookup goes here
```

The processed lookup should eventually contain a stable state name/code, district name/code, and geometry or centroid coordinates. The boundary data will locate a district; it does not provide a DSSAT-ready soil profile by itself.
