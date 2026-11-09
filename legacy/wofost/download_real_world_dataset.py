"""Download and prepare the Indian district/season/crop production dataset.

Expected source fields are state, district, year, season, crop, area and
production. The current public mirror uses state_name, district_name and
crop_year; those aliases are normalized automatically.

The source dataset used by the project reports area in hectares and production
in tonnes, so yield is derived as:
    observed_yield_kg_ha = production_tonnes / area_hectares * 1000
"""
import argparse
import io
import re
from pathlib import Path

import pandas as pd
import requests

DEFAULT_URL = (
    "https://raw.githubusercontent.com/JitendraAmbekar/india-crop-production-analysis/"
    "main/crop_production.csv"
)
CROPS = {"wheat", "rice", "maize"}
SEASONS = {"kharif", "rabi", "zaid"}


def load_csv(path_or_url: str) -> pd.DataFrame:
    if re.match(r"^https?://", str(path_or_url), flags=re.I):
        response = requests.get(path_or_url, timeout=60)
        response.raise_for_status()
        return pd.read_csv(io.BytesIO(response.content))
    path = Path(path_or_url)
    if not path.exists():
        raise FileNotFoundError(f"Source CSV not found: {path.resolve()}")
    return pd.read_csv(path)


def norm_col(column) -> str:
    value = str(column).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def find_column(columns, choices):
    lookup = {str(c).strip().lower(): c for c in columns}
    for choice in choices:
        if choice in lookup:
            return lookup[choice]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_URL, help="CSV URL or local CSV path")
    parser.add_argument("--out", default="real_world_dataset.csv")
    args = parser.parse_args()

    df = load_csv(args.source)
    df.columns = [norm_col(c) for c in df.columns]

    aliases = {
        "state": ["state", "state_name", "statename"],
        "district": ["district", "district_name", "districtname"],
        "year": ["year", "crop_year", "cropyear"],
        "season": ["season", "crop_season", "cropseason"],
        "crop": ["crop", "crop_name", "cropname"],
        "area": [
            "area", "area_hectare", "area_hectares", "area_in_hectare",
            "area_hactare", "area_hactares",
        ],
        "production": [
            "production", "production_tonne", "production_tonnes", "production_tonnes",
            "production_ton", "production_tons", "production_in_tonnes",
            "production_in_tons",
        ],
    }

    rename = {}
    missing = []
    for target, choices in aliases.items():
        found = find_column(df.columns, choices)
        if found is None:
            missing.append(target)
        else:
            rename[found] = target
    if missing:
        raise ValueError(
            "Could not find required column(s): " + ", ".join(missing)
            + f". Available columns: {list(df.columns)}"
        )

    df = df.rename(columns=rename)
    for col in ["state", "district"]:
        df[col] = df[col].astype(str).str.strip()
    df["crop"] = df["crop"].astype(str).str.strip().str.lower()
    df["season"] = df["season"].astype(str).str.strip().str.title()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["production"] = pd.to_numeric(df["production"], errors="coerce")

    df = df[
        df["crop"].isin(CROPS)
        & df["season"].str.lower().isin(SEASONS)
        & df["area"].gt(0)
        & df["production"].ge(0)
        & df["year"].notna()
    ].copy()
    df["year"] = df["year"].astype(int)

    # If a source copy contains duplicate rows for the same crop/location/year,
    # aggregate area and production before deriving yield. This avoids silently
    # creating duplicate observations during ML training.
    keys = ["state", "district", "year", "season", "crop"]
    df = df.groupby(keys, as_index=False, dropna=False).agg(
        area=("area", "sum"), production=("production", "sum")
    )
    df["observed_yield_kg_ha"] = (df["production"] / df["area"]) * 1000.0

    out = df[
        ["state", "district", "year", "season", "crop", "area", "production", "observed_yield_kg_ha"]
    ]
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} real-world rows to {args.out}")
    print(f"Years: {out['year'].min()}–{out['year'].max()}")
    print(f"Crops retained: {', '.join(sorted(out['crop'].unique()))}")


if __name__ == "__main__":
    main()
