"""Application constants and crop/season configuration."""

STATE_DISTRICTS = {
    "Maharashtra": ["Pune", "Nashik", "Nagpur", "Aurangabad"],
    "Punjab": ["Ludhiana", "Amritsar", "Patiala"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi"],
}

DISTRICT_COORDS = {
    "Pune": (18.52, 73.85), "Nashik": (20.00, 73.79), "Nagpur": (21.15, 79.09),
    "Aurangabad": (19.88, 75.34), "Ludhiana": (30.90, 75.85), "Amritsar": (31.63, 74.87),
    "Patiala": (30.34, 76.38), "Lucknow": (26.85, 80.95), "Kanpur": (26.45, 80.35),
    "Varanasi": (25.32, 82.97),
}

CROPS = ["wheat", "rice", "maize"]
CROP_VARIETY = {
    "wheat": "Winter_wheat_101",
    "rice": "Rice_501",
    "maize": "Maize_VanHeemst_1988",
}
SEASONS = ["Kharif", "Rabi", "Zaid"]
SOIL_TYPES = ["Sandy", "Loam", "Clay"]

# Every crop x season combination now has an explicit sowing date.
# These are prototype calendar dates used to make the 3x3 matrix explicit.
# The UI marks the less-common combinations as non-standard scenarios.
CROP_SEASON_CONFIG = {
    ("wheat", "Kharif"): {"month": 6, "day": 15, "standard": False, "label": "non-standard scenario"},
    ("wheat", "Rabi"):   {"month": 11, "day": 15, "standard": True,  "label": "standard"},
    ("wheat", "Zaid"):   {"month": 3, "day": 15, "standard": False, "label": "non-standard scenario"},
    ("rice", "Kharif"):  {"month": 6, "day": 20, "standard": True,  "label": "standard"},
    ("rice", "Rabi"):    {"month": 11, "day": 15, "standard": False, "label": "non-standard scenario"},
    ("rice", "Zaid"):    {"month": 3, "day": 15, "standard": True,  "label": "standard"},
    ("maize", "Kharif"): {"month": 6, "day": 15, "standard": True,  "label": "standard"},
    ("maize", "Rabi"):   {"month": 10, "day": 15, "standard": True,  "label": "standard"},
    ("maize", "Zaid"):   {"month": 3, "day": 15, "standard": False, "label": "non-standard scenario"},
}

SOIL_PARAMS = {
    "Sandy": {"SMW": 0.05, "SMFCF": 0.15, "SM0": 0.30, "CRAIRC": 0.05},
    "Loam":  {"SMW": 0.10, "SMFCF": 0.30, "SM0": 0.40, "CRAIRC": 0.05},
    "Clay":  {"SMW": 0.15, "SMFCF": 0.40, "SM0": 0.50, "CRAIRC": 0.05},
}

CSV_LOG_PATH = "user_inputs_log.csv"
SCENARIO_REFERENCE = {
    "temperature": 25.0, "rainfall": 600.0, "humidity": 65.0,
    "solar_radiation": 18.0, "soil_quality": 70.0, "nitrogen": 70.0, "irrigation": 60.0,
}
SCENARIO_WEIGHTS = {
    "temperature": 0.18, "rainfall": 0.22, "humidity": 0.08,
    "solar_radiation": 0.15, "soil_quality": 0.15, "nitrogen": 0.14, "irrigation": 0.16,
}

CSV_COLUMNS = [
    "timestamp", "state", "district", "crop", "season", "year", "soil_type", "irrigated", "sowing_date",
    "temperature", "rainfall", "humidity", "solar_radiation", "soil_quality", "nitrogen", "irrigation",
    "scenario_factor", "scenario_yield_kg_ha", "TWSO_simulated", "TAGP_simulated", "residual_applied", "final_yield_kg_ha",
]
