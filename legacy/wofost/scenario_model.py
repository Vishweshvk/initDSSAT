"""Proportional scenario layer for natural and intentionally extreme inputs.

WOFOST remains the mechanistic baseline. This module lets a user explore
continuous values (temperature, rainfall, humidity, radiation, soil quality,
nitrogen and irrigation) without forcing those values into dropdown buckets.

Important: this is a prototype sensitivity layer. Its coefficients should be
calibrated against WOFOST/DSSAT runs and observed field data before being used
for real agronomic decisions.
"""

import math
from constants import SCENARIO_REFERENCE, SCENARIO_WEIGHTS


class ScenarioInputError(Exception):
    pass


def _finite_number(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ScenarioInputError(f"{name} must be a numeric value.")
    if not math.isfinite(value):
        raise ScenarioInputError(f"{name} must be a finite numeric value.")
    # Deliberately do not enforce agronomic min/max limits. The project
    # requirement is to let users explore natural as well as intentionally
    # unrealistic scenarios. Extreme values are reported as warnings rather
    # than silently clipped or rejected.
    return value


def parse_scenario(form):
    """Read optional continuous inputs. Blank means use the mechanistic baseline."""
    raw = {}
    for name in SCENARIO_REFERENCE:
        value = form.get(name, "").strip()
        raw[name] = None if value == "" else _finite_number(name, value)
    return raw


def _bounded_response(ratio, weight):
    """Smooth response centred on the reference proportion.

    A reference ratio of 1.0 gives a neutral factor of 1.0. Moving far below
    or above the reference reduces the factor, which is more appropriate for
    stress variables such as temperature, rainfall and nutrient availability
    than allowing unlimited inputs to increase yield forever.
    """
    return math.exp(-weight * (ratio - 1.0) ** 2)


def calculate_scenario(baseline_yield, scenario):
    """Return adjusted yield, multiplicative factor, warnings and details."""
    if all(v is None for v in scenario.values()):
        return {
            "yield": float(baseline_yield),
            "factor": 1.0,
            "warnings": [],
            "details": {},
            "used": False,
        }

    factor = 1.0
    details = {}
    warnings = []

    for name, reference in SCENARIO_REFERENCE.items():
        value = scenario[name]
        if value is None:
            value = reference
        ratio = value / reference if reference else 1.0
        response = _bounded_response(ratio, SCENARIO_WEIGHTS[name])
        factor *= response
        details[name] = {
            "value": value,
            "reference": reference,
            "ratio": ratio,
            "response": response,
        }

        # These are warnings, not rejection rules: unusual inputs are a
        # supported scenario-analysis feature.
        if name == "temperature" and (value < 5 or value > 40):
            warnings.append("Temperature is outside a common crop-growing range; treat the result as an extreme scenario.")
        elif name == "rainfall" and (value < 100 or value > 2000):
            warnings.append("Rainfall is unusually low/high for the prototype reference scenario.")
        elif name == "humidity" and (value < 20 or value > 95):
            warnings.append("Humidity is outside a common field range; disease/evapotranspiration effects are simplified here.")
        elif name == "solar_radiation" and (value < 5 or value > 35):
            warnings.append("Solar radiation is outside a common field range; treat the result as an extreme scenario.")
        elif name in {"soil_quality", "nitrogen", "irrigation"} and (value < 0 or value > 150):
            warnings.append(f"{name.replace('_', ' ').title()} is outside the prototype reference range; this is an intentionally extreme scenario.")

    adjusted = max(0.0, float(baseline_yield) * factor)
    if factor < 1e-6:
        warnings.append(
            "The combined scenario factor is extremely small; the displayed yield may round to 0.0 kg/ha. "
            "This is an extreme mathematical scenario, not a calibrated agronomic prediction."
        )
    return {
        "yield": adjusted,
        "factor": factor,
        "warnings": list(dict.fromkeys(warnings)),
        "details": details,
        "used": True,
    }
