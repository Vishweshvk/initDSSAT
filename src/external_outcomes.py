"""Import DES Area/Production/Yield reports as context-only observations."""

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


DES_APY_URL = "https://data.desagri.gov.in/website/crops-apy-report-web"
CANONICAL_COLUMNS = [
    "state", "district", "crop", "season", "year", "aggregation_level",
    "area_value", "area_unit", "production_value", "production_unit",
    "yield_value_source", "yield_unit_source", "yield_method", "yield_kg_ha",
    "source_name", "source_url", "source_file", "source_row_number",
    "retrieved_at_utc", "training_eligible", "use_scope",
]

_HEADER_ALIASES = {
    "state": {"state", "statename", "stateut", "stateunionterritory"},
    "district": {"district", "districtname"},
    "crop": {"crop", "cropname"},
    "season": {"season", "cropseason"},
    "year": {"year", "cropyear", "agriculturalyear", "agriyear"},
    "area": {"area", "area000ha", "areahectare", "areahectares"},
    "production": {"production", "production000tonnes", "productiontonnes", "quantity"},
    "yield": {"yield", "yieldrate", "productivity", "yieldkgperha", "yieldkgha"},
}
_YIELD_CONVERSIONS = {
    "kg/ha": ("kg/ha", 1.0), "kg ha-1": ("kg/ha", 1.0), "kg per ha": ("kg/ha", 1.0),
    "q/ha": ("quintal/ha", 100.0), "qtl/ha": ("quintal/ha", 100.0),
    "quintal/ha": ("quintal/ha", 100.0), "quintals/ha": ("quintal/ha", 100.0),
    "t/ha": ("tonne/ha", 1000.0), "tonne/ha": ("tonne/ha", 1000.0),
    "tonnes/ha": ("tonne/ha", 1000.0), "metric ton/ha": ("tonne/ha", 1000.0),
}


def _header_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    plain = decomposed.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", "", plain)


def _find_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    mapped: dict[str, str | None] = {}
    keyed = {_header_key(column): str(column) for column in frame.columns}
    for field, aliases in _HEADER_ALIASES.items():
        mapped[field] = next((keyed[alias] for alias in aliases if alias in keyed), None)
        if mapped[field] is None and field in {"area", "production", "yield"}:
            prefixes = {"area": ("area",), "production": ("production",),
                        "yield": ("yield", "productivity")}[field]
            mapped[field] = next((column for key, column in keyed.items()
                                  if key.startswith(prefixes)), None)
    required = [field for field in ("state", "crop", "season", "year") if not mapped[field]]
    if not mapped["yield"] and not (mapped["area"] and mapped["production"]):
        required.append("yield or both area and production")
    if required:
        raise ValueError(
            "Could not identify required DES columns: " + ", ".join(required)
            + ". Expected headers for state, crop, season, year, and yield; rename columns or extend the importer mapping."
        )
    return mapped


def _read_table(source: Path) -> pd.DataFrame:
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            sample = stream.read(4096)
            stream.seek(0)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
            records = list(csv.reader(stream, delimiter=delimiter))
        width = max((len(row) for row in records), default=0)
        raw = pd.DataFrame([row + [""] * (width - len(row)) for row in records], dtype=str)
    if suffix in {".xlsx", ".xlsm"}:
        raw = pd.read_excel(source, dtype=str, header=None)
    elif suffix != ".csv":
        raise ValueError("DES report must be a CSV or Excel .xlsx/.xlsm file.")
    header_index = None
    for index in range(min(30, len(raw))):
        candidate = {_header_key(value) for value in raw.iloc[index].tolist() if _clean_text(value)}
        has_core = any(key in candidate for key in _HEADER_ALIASES["state"]) \
            and any(key in candidate for key in _HEADER_ALIASES["crop"]) \
            and any(key in candidate for key in _HEADER_ALIASES["year"])
        has_outcome = (any(key in candidate for key in _HEADER_ALIASES["yield"]) or
                       (any(key in candidate for key in _HEADER_ALIASES["area"]) and
                        any(key in candidate for key in _HEADER_ALIASES["production"])))
        if has_core and has_outcome:
            header_index = index
            break
    if header_index is None:
        raise ValueError("Could not find a DES table header in the first 30 rows.")
    headers = [_clean_text(value) or f"unnamed_{index}" for index, value in enumerate(raw.iloc[header_index].tolist())]
    counts, unique_headers = {}, []
    for header in headers:
        counts[header] = counts.get(header, 0) + 1
        unique_headers.append(header if counts[header] == 1 else f"{header}_{counts[header]}")
    table = raw.iloc[header_index + 1:].copy()
    table.columns = unique_headers
    return table.dropna(how="all")


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _number(value) -> float | None:
    text = _clean_text(value).replace(",", "")
    if not text or text in {"-", "--", "NA", "N/A", "nan"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if np.isfinite(number) else None


def _unit(value: str) -> tuple[str, float]:
    key = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    key = key.replace("per hectare", "/ha").replace("hectare", "ha")
    try:
        return _YIELD_CONVERSIONS[key]
    except KeyError as exc:
        raise ValueError("Specify a supported yield unit: kg/ha, quintal/ha, or tonne/ha.") from exc


def _area_to_hectares(value: float, unit: str) -> float:
    key = re.sub(r"\s+", " ", str(unit or "").strip().casefold())
    aliases = {"ha": 1.0, "hectare": 1.0, "hectares": 1.0,
               "1000 ha": 1000.0, "thousand ha": 1000.0,
               "1000 hectares": 1000.0, "thousand hectares": 1000.0}
    if key not in aliases:
        raise ValueError("Specify area units as ha or 1000 ha when deriving yield from production and area.")
    return value * aliases[key]


def _production_to_tonnes(value: float, unit: str) -> float:
    key = re.sub(r"\s+", " ", str(unit or "").strip().casefold())
    aliases = {"tonne": 1.0, "tonnes": 1.0, "metric tonne": 1.0,
               "1000 tonnes": 1000.0, "thousand tonnes": 1000.0,
               "1000 tonne": 1000.0, "thousand tonne": 1000.0}
    if key not in aliases:
        raise ValueError("Specify production units as tonnes or 1000 tonnes when deriving yield from production and area.")
    return value * aliases[key]


def normalize_des_apy(source: Path, *, yield_unit: str | None = None, source_url: str = DES_APY_URL,
                      retrieved_at_utc: str | None = None, area_unit: str = "",
                      production_unit: str = "") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Normalize a DES table while preserving provenance and aggregate-only status."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"DES source report not found: {source}")
    raw = _read_table(source)
    columns = _find_columns(raw)
    derive_yield = not columns["yield"]
    if derive_yield:
        area_unit_factor = _area_to_hectares(1.0, area_unit)
        production_unit_factor = _production_to_tonnes(1.0, production_unit)
        canonical_yield_unit, kg_per_unit = "kg/ha", 1.0
    else:
        if not yield_unit:
            raise ValueError("The report has a yield column; specify its printed unit with --yield-unit.")
        canonical_yield_unit, kg_per_unit = _unit(yield_unit)
    retrieved = retrieved_at_utc or datetime.now(timezone.utc).isoformat()
    rows, rejected = [], []
    for source_index, source_row in raw.iterrows():
        row_number = int(source_index) + 1
        state = _clean_text(source_row[columns["state"]])
        district = _clean_text(source_row[columns["district"]]) if columns["district"] else ""
        crop = _clean_text(source_row[columns["crop"]])
        season = _clean_text(source_row[columns["season"]])
        year = _clean_text(source_row[columns["year"]])
        yield_value = (_number(source_row[columns["yield"]]) if columns["yield"] else None)
        area_value = _number(source_row[columns["area"]]) if columns["area"] else None
        production_value = _number(source_row[columns["production"]]) if columns["production"] else None
        if derive_yield and area_value is not None and production_value is not None and area_value > 0:
            area_ha = area_value * area_unit_factor
            production_tonnes = production_value * production_unit_factor
            yield_value = production_tonnes * 1000.0 / area_ha
        problems = []
        if not state: problems.append("missing_state")
        if not crop: problems.append("missing_crop")
        if not season: problems.append("missing_season")
        if not year: problems.append("missing_year")
        if yield_value is None or yield_value < 0: problems.append("invalid_yield_or_area_production")
        if problems:
            rejected.append({"source_row_number": row_number, "reason": ";".join(problems)})
            continue
        rows.append({
            "state": state, "district": district, "crop": crop, "season": season, "year": year,
            "aggregation_level": "district" if district else "state",
            "area_value": area_value,
            "area_unit": area_unit,
            "production_value": production_value,
            "production_unit": production_unit,
            "yield_value_source": yield_value,
            "yield_unit_source": "kg/ha (derived)" if derive_yield else canonical_yield_unit,
            "yield_method": "derived_from_area_and_production" if derive_yield else "reported_by_source",
            "yield_kg_ha": yield_value * kg_per_unit if not derive_yield else yield_value,
            "source_name": "India DES Area, Production & Yield",
            "source_url": source_url or DES_APY_URL, "source_file": source.name,
            "source_row_number": row_number, "retrieved_at_utc": retrieved,
            "training_eligible": False, "use_scope": "aggregate_context_only",
        })
    if not rows:
        raise ValueError(f"No valid DES outcome rows were found in {source.name}.")
    normalized = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    rejected_frame = pd.DataFrame(rejected, columns=["source_row_number", "reason"])
    duplicate_keys = ["state", "district", "crop", "season", "year"]
    duplicate_count = int(normalized.duplicated(duplicate_keys, keep=False).sum())
    manifest = {
        "source_name": "India DES Area, Production & Yield",
        "source_url": source_url or DES_APY_URL,
        "source_file": source.name,
        "retrieved_at_utc": retrieved,
        "source_rows": int(len(raw)), "accepted_rows": len(normalized),
        "rejected_rows": len(rejected_frame), "duplicate_key_rows": duplicate_count,
        "yield_unit_source": "kg/ha (derived)" if derive_yield else canonical_yield_unit,
        "yield_method": "derived_from_area_and_production" if derive_yield else "reported_by_source",
        "canonical_yield_unit": "kg/ha",
        "aggregation_levels": normalized["aggregation_level"].value_counts().to_dict(),
        "training_eligible": False,
        "use_scope": "aggregate_context_only",
        "normalization_columns": columns,
    }
    return normalized, rejected_frame, manifest


def load_des_apy_context(project_data_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Load imported reports for the context page; never returns ML labels."""
    processed = Path(project_data_dir) / "external" / "des_apy" / "processed"
    tables, manifests = [], []
    if not processed.is_dir():
        return pd.DataFrame(columns=CANONICAL_COLUMNS), manifests
    for path in sorted(processed.glob("*_normalized.csv")):
        try:
            frame = pd.read_csv(path)
            missing = set(CANONICAL_COLUMNS) - set(frame.columns)
            if missing or not frame["training_eligible"].astype(str).str.casefold().eq("false").all():
                continue
            tables.append(frame[CANONICAL_COLUMNS])
            manifest_path = path.with_name(path.name.removesuffix("_normalized.csv") + "_manifest.json")
            if manifest_path.is_file():
                manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not tables:
        return pd.DataFrame(columns=CANONICAL_COLUMNS), manifests
    combined = pd.concat(tables, ignore_index=True)
    return combined.drop_duplicates(
        ["state", "district", "crop", "season", "year", "yield_value_source", "yield_unit_source"],
        keep="last",
    ), manifests
