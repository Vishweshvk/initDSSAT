"""Read measured observations from DSSAT crop-specific A-files."""

from pathlib import Path
import pandas as pd


# Conservative Task 7 registry. A crop is eligible only when its measured A-file
# supplies HWAM and DSSAT Summary supplies the same variable. The generic
# definition and dry-matter units are documented by DSSAT; crop-specific
# harvested-product meaning still needs review before expanding this registry.
# Biomass (CWAM), by-product, and differently defined harvest fields are not
# substituted for HWAM.
OBSERVED_YIELD_TARGETS = {
    "BA": {"column": "HWAM", "unit": "kg/ha", "definition": "Barley harvested grain yield at maturity"},
    "BN": {"column": "HWAM", "unit": "kg/ha", "definition": "Dry bean harvested seed yield at maturity"},
    "CH": {"column": "HWAM", "unit": "kg/ha", "definition": "Chickpea harvested seed yield at maturity"},
    "CP": {"column": "HWAM", "unit": "kg/ha", "definition": "Cowpea harvested seed yield at maturity"},
    "CS": {"column": "HWAM", "unit": "kg/ha", "definition": "Cassava harvested storage-root yield at maturity"},
    "FB": {"column": "HWAM", "unit": "kg/ha", "definition": "Faba bean harvested seed yield at maturity"},
    "GY": {"column": "HWAM", "unit": "kg/ha", "definition": "Guar harvested seed yield at maturity"},
    "ML": {"column": "HWAM", "unit": "kg/ha", "definition": "Pearl millet harvested grain yield at maturity"},
    "MZ": {"column": "HWAM", "unit": "kg/ha", "definition": "Maize harvested grain yield at maturity"},
    "PP": {"column": "HWAM", "unit": "kg/ha", "definition": "Pigeon pea harvested seed yield at maturity"},
    "RI": {"column": "HWAM", "unit": "kg/ha", "definition": "Rice harvested grain yield at maturity"},
    "SB": {"column": "HWAM", "unit": "kg/ha", "definition": "Soybean harvested seed yield at maturity"},
    "SF": {"column": "HWAM", "unit": "kg/ha", "definition": "Safflower harvested seed yield at maturity"},
    "SG": {"column": "HWAM", "unit": "kg/ha", "definition": "Sorghum harvested grain yield at maturity"},
    "SU": {"column": "HWAM", "unit": "kg/ha", "definition": "Sunflower harvested seed yield at maturity"},
    "WH": {"column": "HWAM", "unit": "kg/ha", "definition": "Wheat harvested grain yield at maturity"},
}


def observed_yield_target(crop_code: str) -> dict | None:
    """Return a reviewed measured-yield field mapping for a DSSAT crop code."""
    target = OBSERVED_YIELD_TARGETS.get(str(crop_code).strip().upper())
    return {"basis": "dry_matter", **target} if target else None


def calculate_residual(observed_yield: float, mechanistic_yield: float) -> float:
    """Return observed minus mechanistic yield, in kg/ha."""
    return float(observed_yield) - float(mechanistic_yield)


def read_dssat_a_file(path: Path) -> pd.DataFrame:
    """Read an experimental A-file without assuming a crop or yield field.

    A-files use crop-specific suffixes (for example `.WHA`, `.MZA`, `.SCA`)
    but share a treatment table structure. Values are kept as supplied here;
    callers must choose a crop-appropriate measured variable and unit.
    """
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    header = next((line[1:].split() for line in lines if line.startswith("@TRNO")), None)
    if not header or "TRNO" not in header:
        raise ValueError(f"{path.name} has no @TRNO treatment table.")
    rows = []
    for source_row, line in enumerate(lines, start=1):
        tokens = line.split()
        if not tokens or not tokens[0].isdigit() or len(tokens) < len(header):
            continue
        row = dict(zip(header, tokens))
        try:
            parsed = {key: value for key, value in row.items()}
            parsed["TRNO"] = int(row["TRNO"])
            parsed["source_row"] = source_row
            rows.append(parsed)
        except (KeyError, ValueError):
            continue
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No treatment rows parsed from {path.name}.")
    for column in header:
        if column != "TRNO":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def read_observed_yield_values(path: Path, yield_column: str, unit: str) -> pd.DataFrame:
    """Extract one explicitly selected measured field; never guesses its unit."""
    frame = read_dssat_a_file(path)
    column = yield_column.strip().upper()
    if column not in frame:
        raise ValueError(f"{path.name} has no measured field {column}.")
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna() & values.isin([-99, -99.0]).eq(False) & values.ge(0)
    selected = frame.loc[valid, ["TRNO", "source_row"]].copy()
    selected = selected.rename(columns={"TRNO": "treatment_id"})
    selected["observed_value"] = values.loc[valid].astype(float)
    selected["observed_variable"] = column
    selected["observed_unit"] = unit
    if selected.empty:
        raise ValueError(f"{path.name} has no valid non-negative values in {column}.")
    return selected.reset_index(drop=True)


def read_wheat_yield_observations(path: Path) -> pd.DataFrame:
    """Backward-compatible Wheat reader for HWAM observations in kg/ha."""
    selected = read_observed_yield_values(path, "HWAM", "kg/ha")
    return selected.rename(columns={"observed_value": "observed_yield_kg_ha"})[
        ["treatment_id", "observed_yield_kg_ha"]]
