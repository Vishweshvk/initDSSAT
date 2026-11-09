"""Generate WOFOST outputs for crop/season/soil/irrigation combinations.

Examples:
  python generate_simulated_dataset.py --year 2015
  python generate_simulated_dataset.py --year 2015 --district Kanpur
  python generate_simulated_dataset.py --real-match real_world_dataset.csv
  python generate_simulated_dataset.py --real-match real_world_dataset.csv --year 2015

With --real-match and no --year, every matching year in the real dataset is
simulated. With --year and no --real-match, the full configured district matrix
is simulated for that year.
"""
import argparse
import itertools

import pandas as pd

from constants import STATE_DISTRICTS, CROPS, SEASONS, SOIL_TYPES
from wofost_runner import run_wofost

OUT = "simulated_output_dataset.csv"


def state_for_district(district):
    for state, districts in STATE_DISTRICTS.items():
        if district in districts:
            return state
    return ""


def normalize_real_frame(real):
    required = {"state", "district", "crop", "season", "year"}
    missing = required - set(real.columns)
    if missing:
        raise ValueError(
            f"Real dataset is missing {sorted(missing)}. Run download_real_world_dataset.py first."
        )
    real = real.copy()
    real["crop"] = real["crop"].astype(str).str.lower().str.strip()
    real["season"] = real["season"].astype(str).str.title().str.strip()
    real["district"] = real["district"].astype(str).str.strip()
    real["state"] = real["state"].astype(str).str.strip()
    real["year"] = pd.to_numeric(real["year"], errors="coerce")
    return real.dropna(subset=["year"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--district", default=None)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--real-match", default=None)
    args = parser.parse_args()

    if args.real_match:
        real = normalize_real_frame(pd.read_csv(args.real_match))
        real = real[real["crop"].isin(CROPS) & real["season"].isin(SEASONS)]
        if args.year is not None:
            real = real[real["year"].astype(int) == args.year]
        if args.district:
            real = real[real["district"].eq(args.district)]
        if real.empty:
            raise ValueError("No real-world rows match the requested year/district/crop/season filters.")

        combos = real[["state", "district", "crop", "season", "year"]].drop_duplicates()
        cases = [
            {**row, "soil_type": soil, "irrigated": irrigated}
            for row in combos.to_dict("records")
            for soil, irrigated in itertools.product(SOIL_TYPES, [False, True])
        ]
    else:
        if args.year is None:
            raise ValueError("--year is required unless --real-match is supplied.")
        districts = [args.district] if args.district else [d for ds in STATE_DISTRICTS.values() for d in ds]
        cases = [
            {
                "state": state_for_district(district), "district": district,
                "crop": crop, "season": season, "year": args.year,
                "soil_type": soil, "irrigated": irrigated,
            }
            for district, crop, season, soil, irrigated in itertools.product(
                districts, CROPS, SEASONS, SOIL_TYPES, [False, True]
            )
        ]

    results = []
    for i, case in enumerate(cases, 1):
        print(
            f"[{i}/{len(cases)}] {case['district']} | {case['crop']} | "
            f"{case['season']} | {case['soil_type']} | irrigated={case['irrigated']} | {case['year']}"
        )
        try:
            sim = run_wofost(**case)
            twso = sim["TWSO"]
            status = "NON_ZERO" if twso > 0 else "ZERO_YIELD"
            results.append({
                **case,
                "sowing_date": sim["sowing_date"].isoformat(),
                "standard_crop_season": sim["standard_crop_season"],
                "season_label": sim["season_label"],
                "TWSO_simulated": round(twso, 3),
                "TAGP_simulated": round(sim["TAGP"], 3),
                "status": status,
                "error": "",
            })
        except Exception as exc:
            results.append({
                **case,
                "sowing_date": "", "standard_crop_season": "", "season_label": "",
                "TWSO_simulated": None, "TAGP_simulated": None,
                "status": "ERROR", "error": str(exc),
            })

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")
    print("Status counts:")
    print(df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
