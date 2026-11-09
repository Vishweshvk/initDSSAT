"""Find DSSAT soil profiles, parse layers, and match profiles to coordinates."""

import math
from pathlib import Path
import pandas as pd


def read_soil_profile(path: Path, soil_id: str) -> pd.DataFrame:
    """Read one profile's @ depth rows; values retain DSSAT-native units."""
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"*{soil_id}")), None)
    if start is None:
        raise ValueError(f"Soil ID {soil_id} is not present in {path.name}.")
    header = None
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("*"):
            break
        if line.startswith("@  SLB") or line.startswith("@SLB"):
            header = line[1:].split()
            continue
        tokens = line.split()
        if header and tokens and tokens[0].isdigit() and len(tokens) >= len(header):
            rows.append(dict(zip(header, tokens)))
    if not rows:
        raise ValueError(f"No soil layers parsed for {soil_id} in {path.name}.")
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(min(1.0, a)))


def _valid_profile_layers(rows: list[dict[str, float]]) -> tuple[bool, float]:
    """Require usable DSSAT water and density layers; never fill missing values."""
    if len(rows) < 2:
        return False, 0.0
    previous_depth = 0.0
    for row in rows:
        try:
            depth = float(row["SLB"])
            slll, sdul, ssat, sbdm = (float(row[key]) for key in ("SLLL", "SDUL", "SSAT", "SBDM"))
        except (KeyError, TypeError, ValueError):
            return False, 0.0
        if (not all(math.isfinite(value) for value in (depth, slll, sdul, ssat, sbdm))
                or depth <= previous_depth or depth < 0
                or min(slll, sdul, ssat, sbdm) < 0
                or not (slll <= sdul < ssat <= 0.8)
                or not (0.5 <= sbdm <= 2.7)):
            return False, 0.0
        previous_depth = depth
    return previous_depth >= 30, previous_depth


def _soil_records(path: Path):
    """Yield coordinate-bearing DSSAT profiles with complete water layers."""
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    source_header = next((line.strip() for line in lines if line.upper().startswith("*SOILS")), "")
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("*") or line.upper().startswith(("*SOILS:", "*SOIL ")):
            index += 1
            continue
        header_tokens = line[1:].split()
        if not header_tokens:
            index += 1
            continue
        soil_id = header_tokens[0]
        texture_code = header_tokens[2] if len(header_tokens) > 2 else ""
        site = country = ""
        latitude = longitude = None
        layer_header = None
        layer_rows: list[dict[str, float]] = []
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].startswith("*"):
            current = lines[cursor]
            if current.lstrip().startswith("@"):
                fields = current.lstrip()[1:].split()
                if fields and fields[0].upper() == "SITE":
                    site_header = [field.upper() for field in fields]
                    data = []
                    for candidate_index in range(cursor + 1, len(lines)):
                        candidate = lines[candidate_index]
                        if candidate.startswith("*"):
                            break
                        if candidate.strip() and not candidate.lstrip().startswith(("@", "!")):
                            data = candidate.split()
                            break
                    try:
                        site = data[site_header.index("SITE")]
                        country = data[site_header.index("COUNTRY")]
                        latitude = float(data[site_header.index("LAT")])
                        longitude = float(data[site_header.index("LONG")])
                    except (ValueError, IndexError):
                        latitude = longitude = None
                elif fields and fields[0].upper() == "SLB":
                    candidate_header = [field.upper() for field in fields]
                    if {"SLLL", "SDUL", "SSAT", "SBDM"}.issubset(candidate_header):
                        layer_header = candidate_header
                        layer_rows = []
                    else:
                        layer_header = None
                elif layer_header:
                    layer_header = None
            elif layer_header and current.strip() and not current.lstrip().startswith(("!", "@")):
                tokens = current.split()
                if tokens and tokens[0].lstrip("+-").isdigit() and len(tokens) >= len(layer_header):
                    row = {}
                    for key, token in zip(layer_header, tokens):
                        try:
                            row[key] = float(token)
                        except ValueError:
                            continue
                    if row:
                        layer_rows.append(row)
                elif layer_rows:
                    layer_header = None
            cursor += 1
        if (latitude is not None and longitude is not None
                and math.isfinite(latitude) and math.isfinite(longitude)
                and -90 <= latitude <= 90 and -180 <= longitude <= 180
                and abs(latitude) != 99 and abs(longitude) != 99):
            valid_layers, depth = _valid_profile_layers(layer_rows)
            if valid_layers:
                yield {
                    "profile_id": soil_id, "file": path,
                    "source": source_header.split(":", 1)[-1].strip() or "Installed DSSAT soil file",
                    "texture_code": texture_code, "site": site, "country": country,
                    "latitude": latitude, "longitude": longitude, "depth_cm": depth,
                }
        index = cursor


def find_nearest_soil_profile(soil_dir: Path, latitude: float, longitude: float,
                              max_distance_km: float = 100.0) -> dict:
    """Find the nearest georeferenced installed DSSAT profile with complete water layers.

    A profile is eligible only when its DSSAT file provides usable SLLL, SDUL,
    SSAT, and SBDM values through at least 30 cm. No gridded values or guesses
    are substituted into a measured/profiled DSSAT record.
    """
    candidates = []
    for path in sorted(Path(soil_dir).glob("*.SOL")):
        try:
            records = _soil_records(path)
            for record in records:
                record["distance_km"] = _distance_km(latitude, longitude,
                                                       record["latitude"], record["longitude"])
                candidates.append(record)
        except (OSError, UnicodeError):
            continue
    candidates.sort(key=lambda item: item["distance_km"])
    base = {
        "source": "Installed DSSAT soil profile library",
        "max_distance_km": float(max_distance_km),
        "candidate_count": len(candidates),
    }
    if not candidates:
        return {**base, "status": "no_eligible_profile",
                "reason": "No georeferenced installed DSSAT soil profile has complete water-retention and bulk-density layers."}
    nearest = candidates[0]
    if nearest["distance_km"] > max_distance_km:
        return {**base, "status": "no_nearby_profile",
                "reason": f"Nearest eligible installed profile is {nearest['distance_km']:.1f} km away, beyond the {max_distance_km:.0f} km matching limit.",
                "nearest_profile_id": nearest["profile_id"],
                "nearest_file": nearest["file"].name,
                "nearest_distance_km": round(nearest["distance_km"], 2),
                "nearest_source": nearest["source"],
                "nearest_latitude": nearest["latitude"],
                "nearest_longitude": nearest["longitude"],
                "nearest_texture_code": nearest["texture_code"],
                "nearest_depth_cm": nearest["depth_cm"]}
    return {**base, **nearest, "status": "matched",
            "distance_km": round(nearest["distance_km"], 2),
            "file": str(nearest["file"])}
