"""Seasonal weather aggregates derived from DSSAT daily weather records."""

import pandas as pd


def aggregate_weather(weather: pd.DataFrame, start_jdate: int, end_jdate: int) -> dict:
    """Aggregate DSSAT YYDDD weather records between inclusive crop dates."""
    from datetime import date
    def decode(value: int) -> date:
        year, day = divmod(int(value), 1000)
        if year < 100:
            year += 1900 if year >= 50 else 2000
        return date(year, 1, 1) + pd.Timedelta(days=day - 1)
    start, end = decode(start_jdate), decode(end_jdate)
    # Cache decoded dates on the parsed weather frame. Dataset construction
    # aggregates many treatments against the same station file; decoding the
    # entire file once per treatment made multi-crop pairing unnecessarily slow.
    if "calendar_date" not in weather.columns:
        weather["calendar_date"] = weather["DATE"].map(decode)
    frame = weather
    frame = frame[frame["calendar_date"].between(start, end)]
    result = {"weather_days": int(len(frame)), "crop_cycle_days": int((end - start).days + 1)}
    for col, feature, operation in (("RAIN", "total_rainfall_mm", "sum"), ("TMAX", "mean_tmax_c", "mean"),
                                    ("TMIN", "mean_tmin_c", "mean"), ("SRAD", "mean_solar_radiation_mj_m2_day", "mean")):
        if col not in frame:
            result[feature] = None
        else:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            result[feature] = float(values.sum() if operation == "sum" else values.mean()) if not values.empty else None
    valid_rain = pd.to_numeric(frame.get("RAIN"), errors="coerce") if "RAIN" in frame else pd.Series(dtype=float)
    result["rain_days"] = int((valid_rain > 0).sum())
    return result
