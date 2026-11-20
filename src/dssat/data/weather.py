"""Parse DSSAT daily weather files (.WTH)."""

from pathlib import Path
import pandas as pd


def read_weather(path: Path) -> pd.DataFrame:
    """Read daily weather records using the DSSAT @DATE column header."""
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    header = None
    for line in lines:
        if not line.startswith("@"):
            continue
        tokens = line[1:].split()
        if tokens and tokens[0].upper() == "DATE":
            header = [token.upper() for token in tokens]
            break
    if not header:
        raise ValueError(f"No @DATE weather header in {path.name}.")
    records = []
    for line in lines:
        tokens = line.split()
        if not tokens or not tokens[0].isdigit():
            continue
        # DSSAT permits optional weather columns at the end of a row to be
        # blank. split() removes those blanks, so pad them back to keep the
        # required DATE/SRAD/TMAX/TMIN/RAIN columns aligned and available.
        if len(tokens) < len(header):
            tokens.extend([""] * (len(header) - len(tokens)))
        row = dict(zip(header, tokens[:len(header)]))
        records.append(row)
    if not records:
        raise ValueError(f"No daily weather data parsed from {path.name}.")
    frame = pd.DataFrame(records)
    for name in header:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["DATE"])
    if frame.empty:
        raise ValueError(f"No daily weather data parsed from {path.name}.")
    return frame
