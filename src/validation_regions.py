"""Canonical region keys used only to scope residual-model validation."""

import re
import unicodedata


_COUNTRY_ALIASES = {
    "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US", "USA": "US", "US": "US",
    "UNITED KINGDOM": "UK", "GREAT BRITAIN": "UK", "UK": "UK", "GB": "UK",
    "CANADA": "CA", "INDIA": "IN",
}


def _token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+", "_", ascii_value.upper()).strip("_")


def validation_region(country: str, state_or_region: str) -> str:
    """Return COUNTRY|ADMIN1, or an explicit unknown key when either is absent."""
    country_token = _token(country)
    country_key = _COUNTRY_ALIASES.get(country_token.replace("_", " "), country_token)
    region_key = _token(state_or_region)
    if not country_key or not region_key:
        return "UNKNOWN|UNKNOWN"
    return f"{country_key}|{region_key}"


def dssat_location_region(location: str) -> str:
    """Extract country and first-level area from a DSSAT location label.

    Common DSSAT labels place coordinates after the comma-separated site,
    admin-1, and country fields (for example ``SITE,STATE,COUNTRY lat;long``).
    Missing/ambiguous admin-1 values stay unknown so they cannot approve a run.
    """
    label = str(location or "").strip()
    coordinate_match = re.search(r"\s+[+-]?\d+(?:\.\d+)?\s*;", label)
    if coordinate_match:
        label = label[:coordinate_match.start()].strip()
    parts = [part.strip() for part in label.split(",") if part.strip()]
    if len(parts) < 3:
        return "UNKNOWN|UNKNOWN"
    return validation_region(parts[-1], parts[-2])


def dssat_validation_group(location: str, experiment_id: str) -> tuple[str, str]:
    """Return a holdout key and its evidence level without inventing geography.

    Use canonical country/admin-1 when DSSAT provides it. Otherwise group by
    the exact DSSAT location label; if that label is missing, keep the whole
    experiment together. These fallback groups support leakage-resistant
    out-of-fold scoring but never count as geographic validation evidence.
    """
    region = dssat_location_region(location)
    if region != "UNKNOWN|UNKNOWN":
        return region, "country_admin1"
    label = _token(location)
    if label and label not in {"99", "UNKNOWN", "NA", "N_A", "NONE"}:
        return f"SITE_LABEL|{label}", "site_label_only"
    experiment = _token(experiment_id)
    if not experiment:
        raise ValueError("A row without a DSSAT location needs an experiment ID for grouped validation.")
    return f"EXPERIMENT|{experiment}", "experiment_only"


def india_state_region(state: str) -> str:
    """The local GeoNames index bundled with the app contains Indian places."""
    return validation_region("India", state)
