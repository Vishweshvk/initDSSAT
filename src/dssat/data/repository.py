"""Discover DSSAT experiments and their linked crop resources."""

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import re

from ..exceptions import ExperimentValidationError

TREATMENT_FACTOR_COLUMNS = ("CU", "FL", "SA", "IC", "MP", "MI", "MF", "MR", "MC", "MT", "ME", "MH", "SM")

CROP_MODULES = {
    "WH": "CSCER048", "BA": "CSCER048", "MZ": "MZCER048",
    "SG": "SGCER048", "ML": "MLCER048", "RI": "RICER048",
    "SB": "CRGRO048", "PN": "CRGRO048", "BN": "CRGRO048",
    "CH": "CRGRO048", "CP": "CRGRO048", "TM": "CRGRO048",
    "CB": "CRGRO048", "PP": "CRGRO048", "SC": "SCCAN048",
    "CS": "CSYCA048", "BS": "BSCER048", "SW": "SWCER048",
    "PT": "PTSUB048", "PI": "PIALO048", "TN": "TNARO048",
    "TR": "TRARO048", "TF": "TFAPS048",
}


def dssat_module_for_crop(data_dir: Path, crop_code: str,
                          profile_path: Path | None = None) -> str | None:
    """Resolve the active model module from the installed DSSAT profile."""
    key = f"M{crop_code.upper()}"
    candidates = [Path(profile_path)] if profile_path else []
    candidates.append(Path(data_dir) / "DSSATPRO.v48")
    for profile in dict.fromkeys(candidates):
        if not profile.is_file():
            continue
        for line in profile.read_text(encoding="latin-1", errors="replace").splitlines():
            tokens = line.split()
            if len(tokens) >= 4 and tokens[0].upper() == key:
                # DSSAT profile paths are written as `D: \folder`; splitting
                # on whitespace can therefore shift the model module column.
                # The active module is the final token in the crop record.
                module = tokens[-1].upper()
                if module.endswith("048"):
                    return module
                return f"{module}048"
    return CROP_MODULES.get(crop_code.upper())


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    experiment_file: Path
    observed_file: Path | None
    growth_observations_file: Path | None
    weather_station: str
    weather_file: Path | None
    soil_id: str
    soil_file: Path | None
    cultivar_code: str
    cultivar_name: str
    cultivar_file: Path | None
    crop: str
    treatments: tuple[str, ...]
    treatment_details: dict[int, dict]
    location: str
    crop_code: str = "WH"
    dssat_module: str | None = "CSCER048"

    def to_dict(self) -> dict:
        return {key: (str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value)
                for key, value in asdict(self).items()}


def _field_record(text: str, marker: str) -> list[str] | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(marker):
            for candidate in lines[index + 1:]:
                stripped = candidate.strip()
                if stripped and not stripped.startswith(("!", "@", "*")):
                    return stripped.split()
    return None


def crop_display_name(crop_code: str, data_dir: Path) -> str:
    """Return DSSAT's human-readable crop name when its code table is installed."""
    codes_path = Path(data_dir) / "crop_codes.csv"
    if codes_path.is_file():
        try:
            with codes_path.open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            matches = [row for row in records if row.get("DSSAT_code", "").strip().upper() == crop_code.upper()]
            preferred = next((row for row in matches
                              if row.get("Common_name", "").strip().casefold() == "wheat"), None)
            if preferred is None:
                preferred = next((row for row in matches if row.get("Default_module", "").strip()), None)
            if preferred is None and matches:
                preferred = matches[0]
            if preferred and preferred.get("Common_name", "").strip():
                return preferred["Common_name"].strip()
        except (OSError, UnicodeError, csv.Error):
            pass
    return {"WH": "Wheat", "BA": "Barley", "MZ": "Maize", "SG": "Sorghum",
            "ML": "Pearl millet", "RI": "Rice", "SB": "Soybean", "PN": "Peanut",
            "BN": "Dry bean", "CH": "Chickpea", "CP": "Cowpea", "TM": "Tomato",
            "CB": "Cabbage", "PP": "Pigeon pea", "SC": "Sugarcane", "CS": "Cassava",
            "BS": "Sugar beet", "SW": "Sweet corn", "PT": "Potato", "PI": "Pineapple",
            "TN": "Tanier", "TR": "Taro", "TF": "Teff"}.get(crop_code.upper(), crop_code)


def parse_experiment_file(path: Path, data_dir: Path,
                          profile_path: Path | None = None) -> Experiment:
    text = path.read_text(encoding="latin-1", errors="replace")
    field_lines = [line for line in text.splitlines() if line.startswith(" 1 ")]
    field_line = next((line for line in field_lines if len(line.split()) >= 12 and line.split()[1].endswith("0001")), None)
    if field_line is None:
        field_line = next((line for line in text.splitlines() if line.startswith(" 1 ") and "ID_SOIL" not in line), "")
    field_tokens = field_line.split()
    if len(field_tokens) < 12:
        raise ExperimentValidationError(f"Cannot parse field/weather/soil links in {path.name}.")
    station = field_tokens[2]
    soil_id = field_tokens[11]
    cultivar = _field_record(text, "@C CR INGENO CNAME")
    if not cultivar or len(cultivar) < 4:
        raise ExperimentValidationError(f"No DSSAT crop/cultivar record in {path.name}.")
    crop_code, cultivar_code = cultivar[1], cultivar[2]
    cultivar_name = " ".join(cultivar[3:])
    treatments = []
    treatment_details = {}
    sequence_component_counts = {}
    is_sequence = path.suffix.casefold() == ".sqx"
    treatment_section = False
    for line in text.splitlines():
        if line.startswith("*TREATMENTS"):
            treatment_section = True
            continue
        if treatment_section and line.startswith("*"):
            break
        if treatment_section and re.match(r"\s*\d+\s+\d+", line):
            tokens = line.split()
            raw_treatment = tokens[0]
            # In SQX files, N identifies the whole rotation and R identifies
            # each crop component. DSSAT's fixed-width N/R fields can run
            # together once R reaches two digits (for example "110" means
            # sequence N=1, component R=10), so whitespace splitting alone
            # invents treatments 110, 111, etc. All components are one run.
            treatment_id = (int(raw_treatment[0])
                            if is_sequence and len(raw_treatment) >= 3
                            else int(raw_treatment))
            if is_sequence:
                if treatment_id not in treatment_details:
                    treatments.append(str(treatment_id))
                    treatment_details[treatment_id] = {
                        "treatment_name": f"Sequence {treatment_id}",
                        "nitrogen_rate_kg_ha": None,
                        "irrigated": None,
                    }
                sequence_component_counts[treatment_id] = sequence_component_counts.get(treatment_id, 0) + 1
                continue
            treatments.append(str(treatment_id))
            factor_count = len(TREATMENT_FACTOR_COLUMNS)
            treatment_name = " ".join(tokens[4:-factor_count]).strip()
            nitrogen_match = re.search(r"(\d+(?:\.\d+)?)\s*KG\s+N/HA", treatment_name, flags=re.IGNORECASE)
            treatment_details[treatment_id] = {
                "treatment_name": treatment_name,
                "nitrogen_rate_kg_ha": float(nitrogen_match.group(1)) if nitrogen_match else None,
                "irrigated": "IRRIG" in treatment_name.upper(),
            }
    for treatment_id, component_count in sequence_component_counts.items():
        treatment_details[treatment_id]["treatment_name"] = (
            f"Sequence {treatment_id} ({component_count} rotation components)"
        )
    lines = text.splitlines()
    site_index = next((i for i, line in enumerate(lines) if line.startswith(("@SITE", "*SITE"))), None)
    location = ""
    if site_index is not None:
        location = next((line.strip() for line in lines[site_index + 1:]
                         if line.strip() and not line.startswith(("*", "!", "@"))), "")
    weather_dir = data_dir / "Weather"
    soil_dir = data_dir / "Soil"
    genotype_dir = data_dir / "Genotype"
    cultivar_file = None
    expected_cultivar_table = (genotype_dir / f"{crop_code}{dssat_module_for_crop(data_dir, crop_code, profile_path)[2:]}.CUL"
                               if dssat_module_for_crop(data_dir, crop_code, profile_path) else None)
    cultivar_tables = []
    if expected_cultivar_table is not None and expected_cultivar_table.is_file():
        cultivar_tables.append(expected_cultivar_table)
    cultivar_tables.extend(sorted(genotype_dir.glob(f"*{crop_code}*.CUL")))
    cultivar_tables.extend(sorted(genotype_dir.glob("*.CUL")))
    for genotype_file in dict.fromkeys(cultivar_tables):
        if any(
            line.split()[:1] == [cultivar_code]
            for line in genotype_file.read_text(encoding="latin-1", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith(("!", "*", "@"))
        ):
            cultivar_file = genotype_file
            break
    crop_code = crop_code.upper()
    observed_path = path.with_suffix(f".{crop_code}A")
    return Experiment(
        experiment_id=path.stem.upper(), experiment_file=path,
        observed_file=observed_path if observed_path.is_file() else None,
        growth_observations_file=path.with_suffix(".WHT") if path.with_suffix(".WHT").is_file() else None,
        weather_station=station, weather_file=next(iter(weather_dir.glob(f"{station}*.WTH")), None),
        soil_id=soil_id, soil_file=next((p for p in soil_dir.glob("*.SOL") if f"*{soil_id}" in p.read_text(encoding="latin-1", errors="replace")), None),
        cultivar_code=cultivar_code, cultivar_name=cultivar_name, cultivar_file=cultivar_file,
        crop_code=crop_code.upper(), dssat_module=dssat_module_for_crop(data_dir, crop_code, profile_path),
        crop=crop_display_name(crop_code, data_dir),
        treatments=tuple(treatments), treatment_details=treatment_details, location=location,
    )


def discover_experiments(data_dir: Path, crop: str | None = None,
                         profile_path: Path | None = None) -> list[Experiment]:
    """Discover supported FileX experiments in installed DSSAT crop folders."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    ignored = {"weather", "soil", "genotype", "pest", "standarddata"}
    folders = [folder for folder in Path(data_dir).iterdir()
               if folder.is_dir() and folder.name.casefold() not in ignored]
    target_crop = crop.casefold().strip() if crop else None
    aliases = {"maize": {"maize", "corn", "mz"}, "soybean": {"soybean", "soy", "sb"},
               "wheat": {"wheat", "wh"}, "rice": {"rice", "ri"},
               "barley": {"barley", "ba"}, "sorghum": {"sorghum", "sg"},
               "millet": {"millet", "mi"}, "peanut": {"peanut", "pn"},
               "dry bean": {"dry bean", "drybean", "bean", "bn"},
               "chickpea": {"chickpea", "ch"}, "cowpea": {"cowpea", "cp"},
               "sugarcane": {"sugarcane", "su"}, "cassava": {"cassava", "cs"}}
    accepted = aliases.get(target_crop, {target_crop}) if target_crop else None
    experiments = []
    for folder in sorted(folders):
        for path in sorted(folder.glob("*.??X")):
            try:
                experiment = parse_experiment_file(path, data_dir, profile_path)
            except ExperimentValidationError:
                continue
            if experiment.dssat_module and (accepted is None or
                    experiment.crop.casefold() in accepted or
                    experiment.crop_code.casefold() in accepted or
                    folder.name.casefold() in accepted):
                experiments.append(experiment)
    return experiments


def discover_wheat(data_dir: Path, profile_path: Path | None = None) -> list[Experiment]:
    """Backward-compatible Wheat discovery helper."""
    return discover_experiments(data_dir, "wheat", profile_path)


def get_experiment(data_dir: Path, experiment_id: str, crop: str | None = None,
                   profile_path: Path | None = None) -> Experiment:
    target = experiment_id.upper().strip()
    matches = [exp for exp in discover_experiments(data_dir, crop, profile_path) if exp.experiment_id == target]
    if not matches:
        raise ExperimentValidationError(f"DSSAT experiment '{experiment_id}' was not found in supported crop folders under {data_dir}.")
    return matches[0]


def validate_simulation(experiment: Experiment) -> list[str]:
    """Validate resources needed to run DSSAT; observed yields are not required."""
    missing = []
    for label, path in (("experiment FileX", experiment.experiment_file), ("weather file", experiment.weather_file),
                        ("soil file", experiment.soil_file)):
        if path is None or not path.is_file():
            missing.append(label)
    if not getattr(experiment, "dssat_module", "CSCER048"):
        missing.append(f"supported DSSAT model module for crop code {getattr(experiment, 'crop_code', 'unknown')}")
    if not experiment.cultivar_code or experiment.cultivar_file is None:
        missing.append(f"cultivar code in {experiment.crop_code} DSSAT genotype file")
    if not experiment.treatments:
        missing.append("treatments")
    if missing:
        raise ExperimentValidationError(f"{experiment.experiment_id} has missing resources: {', '.join(missing)}")
    return []


def validate_observed_dataset(experiment: Experiment) -> list[str]:
    """Validate simulation inputs plus a reviewed crop-specific measured-yield target."""
    validate_simulation(experiment)
    if experiment.observed_file is None or not experiment.observed_file.is_file():
        suffix = f".{experiment.crop_code}A"
        raise ExperimentValidationError(f"{experiment.experiment_id} has no observed-yield {suffix} file; simulation can still run, but it cannot enter the observed-yield dataset.")
    from .observations import observed_yield_target, read_observed_yield_values
    target = observed_yield_target(experiment.crop_code)
    if target is None:
        raise ExperimentValidationError(
            f"{experiment.experiment_id} has no reviewed measured-yield field mapping for crop code {experiment.crop_code}; "
            "it is excluded until its harvested-product variable and unit are verified.")
    read_observed_yield_values(experiment.observed_file, target["column"], target["unit"])
    return []


def validate_experiment(experiment: Experiment) -> list[str]:
    """Backward-compatible name for the resources required by a simulation."""
    return validate_simulation(experiment)
