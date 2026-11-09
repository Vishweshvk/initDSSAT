"""Parse DSSAT seasonal and daily outputs without guessing variable codes."""

from pathlib import Path
import pandas as pd

from .exceptions import DssatRunError


OUTPUT_DICTIONARY = {
    "HWAM": {"description": "Yield at harvest maturity", "unit": "kg/ha", "file": "Summary.OUT"},
    "HWUM": {"description": "Unit weight at maturity", "unit": "g", "file": "Summary.OUT"},
    "LAIX": {"description": "Maximum leaf area index", "unit": "m2/m2", "file": "Summary.OUT"},
    "CWAD": {"description": "Total crop weight minus roots", "unit": "kg dry matter/ha", "file": "PlantGro.OUT"},
    "LAID": {"description": "Leaf area index", "unit": "m2/m2", "file": "PlantGro.OUT"},
    "DAP": {"description": "Days after planting", "unit": "days", "file": "PlantGro.OUT"},
}


def _read_dssat_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("@")), None)
    if header_index is None:
        raise DssatRunError(f"No DSSAT column header found in {path.name}.")
    # Summary.OUT includes free-text site/treatment values; read_fwf honors
    # DSSAT's fixed column widths. The same reader also handles whitespace
    # separated PlantGro tables. The @ marker is represented as an empty col.
    try:
        frame = pd.read_fwf(path, skiprows=header_index, encoding="latin-1")
    except (ValueError, OSError) as exc:
        raise DssatRunError(f"Could not parse DSSAT table {path.name}: {exc}") from exc
    frame = frame.loc[:, [not str(column).startswith("Unnamed") and str(column) != "@" for column in frame.columns]]
    frame.columns = [str(column).strip().rstrip(".") for column in frame.columns]
    if frame.empty:
        raise DssatRunError(f"No data records found in {path.name}.")
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            frame[column] = converted
    return frame


def parse_summary(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip().upper() for column in frame.columns]
    else:
        frame = _read_dssat_table(path)
    if "HWAM" not in frame:
        raise DssatRunError("Summary.OUT lacks HWAM, the DSSAT yield-at-harvest-maturity variable (kg/ha).")
    return frame


def parse_plantgro(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip().upper() for column in frame.columns]
        return frame
    return _read_dssat_table(path)
