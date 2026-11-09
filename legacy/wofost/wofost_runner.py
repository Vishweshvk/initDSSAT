"""Run one WOFOST/PCSE simulation with explicit crop-season dates."""

from datetime import date, timedelta

from pcse.base import ParameterProvider
from pcse.input import (
    NASAPowerWeatherDataProvider,
    YAMLCropDataProvider,
    WOFOST72SiteDataProvider,
)
from pcse.models import Wofost72_WLP_FD

from constants import (
    CROP_SEASON_CONFIG,
    CROP_VARIETY,
    DISTRICT_COORDS,
    SOIL_PARAMS,
    STATE_DISTRICTS,
)

MAX_CROP_CYCLE_DAYS = 300


class SimulationInputError(Exception):
    pass


def _derive_sowing_date(crop: str, season: str, year: int) -> date:
    try:
        cfg = CROP_SEASON_CONFIG[(crop, season)]
    except KeyError as exc:
        raise SimulationInputError(
            f"Unsupported crop/season combination: {crop}/{season}"
        ) from exc
    try:
        return date(int(year), cfg["month"], cfg["day"])
    except ValueError as exc:
        raise SimulationInputError(f"Invalid year: {year}") from exc


def crop_season_info(crop: str, season: str) -> dict:
    cfg = CROP_SEASON_CONFIG.get((crop, season))
    if cfg is None:
        raise SimulationInputError(f"Unsupported crop/season combination: {crop}/{season}")
    return cfg


def _validate_location(state: str, district: str):
    districts = STATE_DISTRICTS.get(state)
    if districts is None:
        raise SimulationInputError(f"Unknown state: {state}")
    if district not in districts:
        raise SimulationInputError(f"District '{district}' is not configured under {state}.")
    if district not in DISTRICT_COORDS:
        raise SimulationInputError(f"No coordinates configured for district: {district}")


def _build_soil_provider(soil_type: str) -> dict:
    try:
        base = SOIL_PARAMS[soil_type]
    except KeyError as exc:
        raise SimulationInputError(f"Unknown soil type: {soil_type}") from exc
    return {
        "SMW": base["SMW"],
        "SMFCF": base["SMFCF"],
        "SM0": base["SM0"],
        "CRAIRC": base["CRAIRC"],
        "RDMSOL": 100.0,
        "K0": 10.0,
        "SOPE": 10.0,
        "KSUB": 10.0,
    }


def _irrigation_events(sowing_date: date, irrigated: bool):
    """Return a simple prototype irrigation schedule for irrigated runs.

    PCSE's AgroManager supports timed irrigation events. The amounts here are
    deliberately labelled as prototype management assumptions, not calibrated
    recommendations. Rainfed runs contain no irrigation events.
    """
    if not irrigated:
        return None
    schedule = ((30, 3.0), (60, 4.0), (90, 3.0))  # cm of water
    return [
        {sowing_date + timedelta(days=offset): {"irrigation_amount": amount}}
        for offset, amount in schedule
    ]


def run_wofost(
    state: str,
    district: str,
    crop: str,
    season: str,
    year: int,
    soil_type: str,
    irrigated: bool,
) -> dict:
    _validate_location(state, district)
    if crop not in CROP_VARIETY:
        raise SimulationInputError(f"Unknown crop: {crop}")
    if soil_type not in SOIL_PARAMS:
        raise SimulationInputError(f"Unknown soil type: {soil_type}")

    lat, lon = DISTRICT_COORDS[district]
    sowing_date = _derive_sowing_date(crop, season, year)
    season_info = crop_season_info(crop, season)

    try:
        weatherdataprovider = NASAPowerWeatherDataProvider(latitude=lat, longitude=lon)
    except Exception as exc:
        raise SimulationInputError(
            "Could not retrieve NASA POWER weather data. Check internet access and the district coordinates."
        ) from exc

    last_available = getattr(weatherdataprovider, "last_date", None)
    if last_available is not None:
        if hasattr(last_available, "date"):
            last_available = last_available.date()
        if sowing_date > last_available:
            raise SimulationInputError(
                f"Sowing date {sowing_date} is after the last available weather date "
                f"{last_available}. Choose an earlier year."
            )
        available_days = (last_available - sowing_date).days
        if available_days < MAX_CROP_CYCLE_DAYS:
            raise SimulationInputError(
                f"Only {available_days} days of weather are available after sowing. "
                f"This prototype allows up to {MAX_CROP_CYCLE_DAYS} days for the crop cycle. "
                "Choose an earlier year."
            )

    variety = CROP_VARIETY[crop]
    try:
        cropdataprovider = YAMLCropDataProvider()
        cropdataprovider.set_active_crop(crop, variety)
        soildataprovider = _build_soil_provider(soil_type)
        sitedataprovider = WOFOST72SiteDataProvider(WAV=50)
        parameters = ParameterProvider(
            cropdata=cropdataprovider,
            soildata=soildataprovider,
            sitedata=sitedataprovider,
        )

        irrigation_events = _irrigation_events(sowing_date, irrigated)
        agro_yaml = {
            "Version": "1.0.0",
            "AgroManagement": [{
                sowing_date: {
                    "CropCalendar": {
                        "crop_name": crop,
                        "variety_name": variety,
                        "crop_start_date": sowing_date,
                        "crop_start_type": "sowing",
                        "crop_end_date": None,
                        "crop_end_type": "maturity",
                        "max_duration": MAX_CROP_CYCLE_DAYS,
                    },
                    "TimedEvents": (
                        [{
                            "event_signal": "irrigate",
                            "name": "Prototype timed irrigation",
                            "comment": "Prototype irrigation amounts in cm",
                            "events_table": irrigation_events,
                        }]
                        if irrigation_events else None
                    ),
                    "StateEvents": None,
                }
            }],
        }

        # PCSE accepts date keys and datetime.date values in its agromanagement
        # structure. YAML is retained in the project for readability, but using
        # the native dictionary avoids date/string conversion surprises.
        wofost = Wofost72_WLP_FD(parameters, weatherdataprovider, agro_yaml)
        wofost.run_till_terminate()
        output = wofost.get_summary_output()
    except SimulationInputError:
        raise
    except Exception as exc:
        raise SimulationInputError(
            f"WOFOST simulation failed for {crop}/{season}, {district}, {year}: {exc}"
        ) from exc

    if not output or output[0].get("TWSO") is None:
        raise SimulationInputError(
            "WOFOST completed but returned no TWSO. Check crop variety, soil parameters, "
            "weather coverage and the selected year."
        )

    result = output[0]
    return {
        "TWSO": float(result["TWSO"]),
        "TAGP": float(result["TAGP"]),
        "sowing_date": sowing_date,
        "standard_crop_season": season_info["standard"],
        "season_label": season_info["label"],
        "irrigated": bool(irrigated),
    }
