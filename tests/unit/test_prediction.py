import math

import pytest

from src.prediction import ACRE_TO_HECTARES, area_estimate


def test_area_estimate_hectares_scales_yield():
    result = area_estimate(4200, 2.5, "hectares")
    assert result["area_hectares"] == 2.5
    assert result["total_harvest_kg"] == 10500
    assert result["total_harvest_tonnes"] == 10.5


def test_area_estimate_uses_exact_acre_conversion():
    result = area_estimate(1000, 2, "acres")
    assert result["area_hectares"] == 2 * ACRE_TO_HECTARES
    assert result["total_harvest_kg"] == 2 * ACRE_TO_HECTARES * 1000


@pytest.mark.parametrize("area", [0, -1, float("nan"), float("inf"), "not a number", None])
def test_area_estimate_rejects_invalid_area(area):
    with pytest.raises(ValueError, match="area"):
        area_estimate(1000, area, "hectares")


def test_area_estimate_does_not_clip_negative_model_output():
    result = area_estimate(-100, 2, "hectares")
    assert result["total_harvest_kg"] == -200
    assert math.isfinite(result["total_harvest_kg"])


def test_area_estimate_rejects_unknown_units():
    with pytest.raises(ValueError, match="hectares or acres"):
        area_estimate(1000, 1, "square_feet")
