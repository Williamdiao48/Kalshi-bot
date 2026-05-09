"""Open-Meteo forecast bias calibration table.

Mean bias (forecast − actual, °F) per (source, city, month) derived from the
10-year Open-Meteo Historical Forecast API vs Iowa State Mesonet ASOS backtest
(scripts/backtest_openmeteo_bias.py, 2016-05-08 → 2026-05-08, 3,415 city-date
pairs per city, 21 cities).

Only entries with |bias| >= 1.0°F are included.  Biases < 1.0°F are treated
as noise relative to model MAE (~3-4°F day-ahead).

Interpretation:
    bias < 0  → model ran cold (underestimated peak temperature)
    bias > 0  → model ran warm (overestimated peak temperature)

Usage in strike_arb.find_forecast_nos():
    corrected_value = raw_value - BIAS_F.get((source, city, month), 0.0)
    # Cold model (bias < 0): shifts forecast up → more NO_HIGH edge, less NO_LOW edge.
    # Warm model (bias > 0): shifts forecast down → less NO_HIGH edge, more NO_LOW edge.

Keys: (source_name: str, city_suffix: str, month: int)
    city_suffix — matches the suffix in the metric key (e.g. "atl" for temp_high_atl)
    month       — 1=Jan … 12=Dec (local calendar month)
"""

# (source, city, month) → mean bias °F (forecast - actual)
BIAS_F: dict[tuple[str, str, int], float] = {

    # ── open_meteo (blended GFS; open_meteo_gfs is identical and excluded from
    # ── _FORECAST_NO_SOURCES to prevent double-counting the same signal) ──────

    # Atlanta — summer/fall cold bias
    ("open_meteo", "atl", 7):  -1.2,
    ("open_meteo", "atl", 8):  -1.4,
    ("open_meteo", "atl", 9):  -1.4,
    ("open_meteo", "atl", 10): -1.3,
    ("open_meteo", "atl", 12): -1.0,

    # Austin
    ("open_meteo", "aus", 8):  -1.0,
    ("open_meteo", "aus", 9):  -1.0,

    # Chicago
    ("open_meteo", "chi", 9):  -1.0,
    ("open_meteo", "chi", 10): -1.0,
    ("open_meteo", "chi", 12): -1.1,

    # Dallas — warm summer bias
    ("open_meteo", "dal", 7):  +1.0,

    # Denver — cold in warm months
    ("open_meteo", "den", 4):  -1.0,
    ("open_meteo", "den", 5):  -1.1,
    ("open_meteo", "den", 6):  -1.0,
    ("open_meteo", "den", 7):  -1.5,
    ("open_meteo", "den", 8):  -1.6,
    ("open_meteo", "den", 9):  -1.3,
    ("open_meteo", "den", 10): -1.2,

    # Miami — fall cold bias
    ("open_meteo", "mia", 9):  -1.1,
    ("open_meteo", "mia", 10): -1.1,

    # Minneapolis
    ("open_meteo", "msp", 7):  -1.0,
    ("open_meteo", "msp", 8):  -1.0,

    # New Orleans — summer cold bias
    ("open_meteo", "msy", 6):  -1.1,
    ("open_meteo", "msy", 7):  -1.3,
    ("open_meteo", "msy", 8):  -1.4,
    ("open_meteo", "msy", 9):  -1.2,

    # New York — summer warm bias (coastal sea breeze / urban heat suppressed in model)
    ("open_meteo", "ny",  6):  +1.1,
    ("open_meteo", "ny",  7):  +1.4,
    ("open_meteo", "ny",  8):  +1.0,
    ("open_meteo", "ny",  10): +1.0,

    # Seattle — consistently cold year-round
    ("open_meteo", "sea", 1):  -1.7,
    ("open_meteo", "sea", 2):  -1.4,
    ("open_meteo", "sea", 3):  -1.1,
    ("open_meteo", "sea", 7):  -1.1,
    ("open_meteo", "sea", 8):  -1.1,
    ("open_meteo", "sea", 10): -1.3,
    ("open_meteo", "sea", 11): -1.7,
    ("open_meteo", "sea", 12): -1.9,

    # San Francisco — summer/fall warm bias (marine layer dynamics)
    ("open_meteo", "sfo", 7):  +1.8,
    ("open_meteo", "sfo", 8):  +2.4,
    ("open_meteo", "sfo", 9):  +2.1,
    ("open_meteo", "sfo", 10): +1.1,

    # ── open_meteo_ecmwf — globally cold, worst in humid/tropical cities ──────

    # Atlanta
    ("open_meteo_ecmwf", "atl", 1):  -1.7,
    ("open_meteo_ecmwf", "atl", 2):  -1.8,
    ("open_meteo_ecmwf", "atl", 3):  -2.4,
    ("open_meteo_ecmwf", "atl", 4):  -1.7,
    ("open_meteo_ecmwf", "atl", 5):  -1.0,
    ("open_meteo_ecmwf", "atl", 6):  -1.5,
    ("open_meteo_ecmwf", "atl", 7):  -2.5,
    ("open_meteo_ecmwf", "atl", 8):  -2.2,
    ("open_meteo_ecmwf", "atl", 9):  -2.1,
    ("open_meteo_ecmwf", "atl", 10): -2.1,
    ("open_meteo_ecmwf", "atl", 11): -2.0,
    ("open_meteo_ecmwf", "atl", 12): -1.7,

    # Austin
    ("open_meteo_ecmwf", "aus", 1):  -1.5,
    ("open_meteo_ecmwf", "aus", 5):  -1.0,
    ("open_meteo_ecmwf", "aus", 6):  -1.3,
    ("open_meteo_ecmwf", "aus", 7):  -1.1,
    ("open_meteo_ecmwf", "aus", 8):  -1.3,
    ("open_meteo_ecmwf", "aus", 9):  -2.3,
    ("open_meteo_ecmwf", "aus", 10): -1.5,
    ("open_meteo_ecmwf", "aus", 11): -1.5,
    ("open_meteo_ecmwf", "aus", 12): -1.7,

    # Boston — only November
    ("open_meteo_ecmwf", "bos", 11): -1.0,

    # Chicago
    ("open_meteo_ecmwf", "chi", 4):  -1.1,
    ("open_meteo_ecmwf", "chi", 5):  -1.9,
    ("open_meteo_ecmwf", "chi", 6):  -1.6,
    ("open_meteo_ecmwf", "chi", 7):  -1.9,
    ("open_meteo_ecmwf", "chi", 8):  -1.5,
    ("open_meteo_ecmwf", "chi", 9):  -1.4,
    ("open_meteo_ecmwf", "chi", 10): -1.4,
    ("open_meteo_ecmwf", "chi", 11): -1.3,

    # Dallas
    ("open_meteo_ecmwf", "dal", 1):  -1.6,
    ("open_meteo_ecmwf", "dal", 2):  -1.2,
    ("open_meteo_ecmwf", "dal", 3):  -1.4,
    ("open_meteo_ecmwf", "dal", 4):  -1.0,
    ("open_meteo_ecmwf", "dal", 5):  -1.1,
    ("open_meteo_ecmwf", "dal", 6):  -1.6,
    ("open_meteo_ecmwf", "dal", 7):  -1.1,
    ("open_meteo_ecmwf", "dal", 8):  -1.4,
    ("open_meteo_ecmwf", "dal", 9):  -1.3,
    ("open_meteo_ecmwf", "dal", 11): -1.1,
    ("open_meteo_ecmwf", "dal", 12): -1.5,

    # Washington DC
    ("open_meteo_ecmwf", "dca", 1):  -1.8,
    ("open_meteo_ecmwf", "dca", 2):  -1.4,
    ("open_meteo_ecmwf", "dca", 3):  -1.6,
    ("open_meteo_ecmwf", "dca", 4):  -1.2,
    ("open_meteo_ecmwf", "dca", 5):  -1.0,
    ("open_meteo_ecmwf", "dca", 7):  -1.2,
    ("open_meteo_ecmwf", "dca", 8):  -2.0,
    ("open_meteo_ecmwf", "dca", 9):  -1.5,
    ("open_meteo_ecmwf", "dca", 10): -1.7,
    ("open_meteo_ecmwf", "dca", 11): -2.0,
    ("open_meteo_ecmwf", "dca", 12): -1.6,

    # Denver
    ("open_meteo_ecmwf", "den", 1):  -1.4,
    ("open_meteo_ecmwf", "den", 6):  -1.1,
    ("open_meteo_ecmwf", "den", 7):  -2.1,
    ("open_meteo_ecmwf", "den", 8):  -2.0,
    ("open_meteo_ecmwf", "den", 9):  -1.1,
    ("open_meteo_ecmwf", "den", 11): -1.1,
    ("open_meteo_ecmwf", "den", 12): -1.7,

    # DFW
    ("open_meteo_ecmwf", "dfw", 1):  -1.3,
    ("open_meteo_ecmwf", "dfw", 3):  -1.0,
    ("open_meteo_ecmwf", "dfw", 6):  -1.0,
    ("open_meteo_ecmwf", "dfw", 11): -1.1,
    ("open_meteo_ecmwf", "dfw", 12): -1.0,

    # Houston — extremely cold-biased for humid Gulf Coast
    ("open_meteo_ecmwf", "hou", 1):  -1.9,
    ("open_meteo_ecmwf", "hou", 2):  -1.2,
    ("open_meteo_ecmwf", "hou", 3):  -2.2,
    ("open_meteo_ecmwf", "hou", 4):  -1.8,
    ("open_meteo_ecmwf", "hou", 5):  -2.3,
    ("open_meteo_ecmwf", "hou", 6):  -2.5,
    ("open_meteo_ecmwf", "hou", 7):  -3.2,
    ("open_meteo_ecmwf", "hou", 8):  -3.2,
    ("open_meteo_ecmwf", "hou", 9):  -3.2,
    ("open_meteo_ecmwf", "hou", 10): -2.7,
    ("open_meteo_ecmwf", "hou", 11): -2.1,
    ("open_meteo_ecmwf", "hou", 12): -1.9,

    # Las Vegas
    ("open_meteo_ecmwf", "las", 11): -1.0,
    ("open_meteo_ecmwf", "las", 12): -1.4,

    # Los Angeles — cold in winter, neutral/warm in summer
    ("open_meteo_ecmwf", "lax", 1):  -2.4,
    ("open_meteo_ecmwf", "lax", 2):  -1.5,
    ("open_meteo_ecmwf", "lax", 3):  -1.3,
    ("open_meteo_ecmwf", "lax", 11): -1.3,
    ("open_meteo_ecmwf", "lax", 12): -1.9,

    # Miami — summer massive cold bias
    ("open_meteo_ecmwf", "mia", 1):  -1.2,
    ("open_meteo_ecmwf", "mia", 2):  -1.1,
    ("open_meteo_ecmwf", "mia", 5):  -1.2,
    ("open_meteo_ecmwf", "mia", 6):  -2.7,
    ("open_meteo_ecmwf", "mia", 7):  -2.8,
    ("open_meteo_ecmwf", "mia", 8):  -2.6,
    ("open_meteo_ecmwf", "mia", 9):  -2.7,
    ("open_meteo_ecmwf", "mia", 10): -2.3,
    ("open_meteo_ecmwf", "mia", 11): -1.4,
    ("open_meteo_ecmwf", "mia", 12): -1.3,

    # Minneapolis
    ("open_meteo_ecmwf", "msp", 5):  -1.3,
    ("open_meteo_ecmwf", "msp", 6):  -1.0,
    ("open_meteo_ecmwf", "msp", 9):  -1.0,
    ("open_meteo_ecmwf", "msp", 10): -1.0,

    # New Orleans — summer worst
    ("open_meteo_ecmwf", "msy", 1):  -1.8,
    ("open_meteo_ecmwf", "msy", 2):  -2.1,
    ("open_meteo_ecmwf", "msy", 3):  -2.1,
    ("open_meteo_ecmwf", "msy", 4):  -1.4,
    ("open_meteo_ecmwf", "msy", 5):  -2.0,
    ("open_meteo_ecmwf", "msy", 6):  -3.1,
    ("open_meteo_ecmwf", "msy", 7):  -3.7,
    ("open_meteo_ecmwf", "msy", 8):  -3.4,
    ("open_meteo_ecmwf", "msy", 9):  -3.2,
    ("open_meteo_ecmwf", "msy", 10): -2.6,
    ("open_meteo_ecmwf", "msy", 11): -2.1,
    ("open_meteo_ecmwf", "msy", 12): -2.0,

    # New York
    ("open_meteo_ecmwf", "ny",  2):  -1.5,
    ("open_meteo_ecmwf", "ny",  3):  -1.2,
    ("open_meteo_ecmwf", "ny",  4):  -1.3,
    ("open_meteo_ecmwf", "ny",  5):  -1.0,
    ("open_meteo_ecmwf", "ny",  12): -1.0,

    # Oklahoma City
    ("open_meteo_ecmwf", "okc", 5):  +1.0,
    ("open_meteo_ecmwf", "okc", 6):  +1.0,

    # Phoenix
    ("open_meteo_ecmwf", "phx", 1):  -1.8,
    ("open_meteo_ecmwf", "phx", 2):  -1.3,
    ("open_meteo_ecmwf", "phx", 3):  -1.0,
    ("open_meteo_ecmwf", "phx", 7):  -1.4,
    ("open_meteo_ecmwf", "phx", 8):  -1.2,
    ("open_meteo_ecmwf", "phx", 9):  -1.1,
    ("open_meteo_ecmwf", "phx", 10): -1.0,
    ("open_meteo_ecmwf", "phx", 11): -1.0,
    ("open_meteo_ecmwf", "phx", 12): -1.6,

    # San Antonio
    ("open_meteo_ecmwf", "sat", 1):  -1.2,
    ("open_meteo_ecmwf", "sat", 5):  -1.3,
    ("open_meteo_ecmwf", "sat", 6):  -1.1,
    ("open_meteo_ecmwf", "sat", 7):  -1.5,
    ("open_meteo_ecmwf", "sat", 8):  -1.4,
    ("open_meteo_ecmwf", "sat", 9):  -1.9,
    ("open_meteo_ecmwf", "sat", 10): -1.3,
    ("open_meteo_ecmwf", "sat", 11): -1.6,
    ("open_meteo_ecmwf", "sat", 12): -1.6,

    # Seattle
    ("open_meteo_ecmwf", "sea", 1):  -1.5,
    ("open_meteo_ecmwf", "sea", 2):  -1.4,
    ("open_meteo_ecmwf", "sea", 3):  -1.6,
    ("open_meteo_ecmwf", "sea", 4):  -1.9,
    ("open_meteo_ecmwf", "sea", 5):  -2.1,
    ("open_meteo_ecmwf", "sea", 6):  -1.4,
    ("open_meteo_ecmwf", "sea", 9):  -1.1,
    ("open_meteo_ecmwf", "sea", 10): -1.4,
    ("open_meteo_ecmwf", "sea", 11): -1.3,
    ("open_meteo_ecmwf", "sea", 12): -1.4,

    # San Francisco
    ("open_meteo_ecmwf", "sfo", 2):  -1.1,
    ("open_meteo_ecmwf", "sfo", 3):  -1.2,
    ("open_meteo_ecmwf", "sfo", 4):  -1.5,
    ("open_meteo_ecmwf", "sfo", 9):  +1.3,
    ("open_meteo_ecmwf", "sfo", 10): +1.0,

    # ── open_meteo_icon — n≈1269 (≈3.7yr); notable outliers at LAX/PHX/SEA ──

    # Atlanta — very cold all year
    ("open_meteo_icon", "atl", 1):  -1.8,
    ("open_meteo_icon", "atl", 2):  -2.1,
    ("open_meteo_icon", "atl", 3):  -1.9,
    ("open_meteo_icon", "atl", 4):  -1.1,
    ("open_meteo_icon", "atl", 5):  -1.5,
    ("open_meteo_icon", "atl", 6):  -1.6,
    ("open_meteo_icon", "atl", 7):  -1.9,
    ("open_meteo_icon", "atl", 8):  -1.9,
    ("open_meteo_icon", "atl", 9):  -1.9,
    ("open_meteo_icon", "atl", 10): -2.2,
    ("open_meteo_icon", "atl", 11): -2.1,
    ("open_meteo_icon", "atl", 12): -2.1,

    # Austin
    ("open_meteo_icon", "aus", 9):  -2.0,
    ("open_meteo_icon", "aus", 10): -1.4,
    ("open_meteo_icon", "aus", 11): -1.5,
    ("open_meteo_icon", "aus", 12): -1.3,

    # Boston
    ("open_meteo_icon", "bos", 1):  -1.1,
    ("open_meteo_icon", "bos", 8):  +1.0,

    # Chicago
    ("open_meteo_icon", "chi", 5):  -1.0,
    ("open_meteo_icon", "chi", 7):  -1.1,
    ("open_meteo_icon", "chi", 11): -1.1,
    ("open_meteo_icon", "chi", 12): -1.1,

    # Denver
    ("open_meteo_icon", "den", 8):  -1.6,
    ("open_meteo_icon", "den", 9):  -1.1,
    ("open_meteo_icon", "den", 10): -1.5,
    ("open_meteo_icon", "den", 12): -1.3,

    # DFW
    ("open_meteo_icon", "dfw", 12): -1.3,

    # Los Angeles — massively warm (most harmful source×city combination)
    ("open_meteo_icon", "lax", 3):  +1.4,
    ("open_meteo_icon", "lax", 4):  +2.0,
    ("open_meteo_icon", "lax", 5):  +2.4,
    ("open_meteo_icon", "lax", 6):  +1.5,
    ("open_meteo_icon", "lax", 7):  +4.1,
    ("open_meteo_icon", "lax", 8):  +3.6,
    ("open_meteo_icon", "lax", 9):  +2.3,
    ("open_meteo_icon", "lax", 10): +1.8,

    # Miami
    ("open_meteo_icon", "mia", 3):  -1.0,
    ("open_meteo_icon", "mia", 5):  -1.0,
    ("open_meteo_icon", "mia", 6):  -1.9,
    ("open_meteo_icon", "mia", 7):  -1.8,
    ("open_meteo_icon", "mia", 8):  -1.7,
    ("open_meteo_icon", "mia", 9):  -1.7,
    ("open_meteo_icon", "mia", 10): -1.3,
    ("open_meteo_icon", "mia", 11): -1.4,

    # Minneapolis
    ("open_meteo_icon", "msp", 11): -1.0,

    # New Orleans
    ("open_meteo_icon", "msy", 1):  -1.0,
    ("open_meteo_icon", "msy", 2):  -1.1,
    ("open_meteo_icon", "msy", 9):  -1.3,
    ("open_meteo_icon", "msy", 10): -1.3,
    ("open_meteo_icon", "msy", 12): -1.2,

    # New York
    ("open_meteo_icon", "ny",  1):  -1.2,
    ("open_meteo_icon", "ny",  2):  -1.2,
    ("open_meteo_icon", "ny",  3):  -1.3,
    ("open_meteo_icon", "ny",  4):  -2.0,
    ("open_meteo_icon", "ny",  5):  -1.0,
    ("open_meteo_icon", "ny",  12): -1.0,

    # Philadelphia
    ("open_meteo_icon", "phl", 3):  -1.0,
    ("open_meteo_icon", "phl", 4):  -1.0,
    ("open_meteo_icon", "phl", 9):  -1.4,
    ("open_meteo_icon", "phl", 10): -1.1,
    ("open_meteo_icon", "phl", 12): -1.2,

    # Phoenix — very cold all year
    ("open_meteo_icon", "phx", 1):  -2.0,
    ("open_meteo_icon", "phx", 2):  -1.5,
    ("open_meteo_icon", "phx", 3):  -1.5,
    ("open_meteo_icon", "phx", 4):  -1.4,
    ("open_meteo_icon", "phx", 5):  -1.8,
    ("open_meteo_icon", "phx", 6):  -1.7,
    ("open_meteo_icon", "phx", 7):  -1.2,
    ("open_meteo_icon", "phx", 8):  -1.3,
    ("open_meteo_icon", "phx", 9):  -2.0,
    ("open_meteo_icon", "phx", 10): -1.9,
    ("open_meteo_icon", "phx", 11): -2.0,
    ("open_meteo_icon", "phx", 12): -2.2,

    # Seattle — very cold in summer
    ("open_meteo_icon", "sea", 1):  -1.6,
    ("open_meteo_icon", "sea", 2):  -1.2,
    ("open_meteo_icon", "sea", 3):  -1.5,
    ("open_meteo_icon", "sea", 4):  -1.5,
    ("open_meteo_icon", "sea", 5):  -1.9,
    ("open_meteo_icon", "sea", 6):  -2.2,
    ("open_meteo_icon", "sea", 7):  -2.9,
    ("open_meteo_icon", "sea", 8):  -2.4,
    ("open_meteo_icon", "sea", 9):  -1.9,
    ("open_meteo_icon", "sea", 10): -2.0,
    ("open_meteo_icon", "sea", 11): -1.3,
    ("open_meteo_icon", "sea", 12): -1.3,

    # San Francisco
    ("open_meteo_icon", "sfo", 2):  -1.2,
    ("open_meteo_icon", "sfo", 3):  -2.5,
    ("open_meteo_icon", "sfo", 4):  -1.8,
    ("open_meteo_icon", "sfo", 6):  -1.6,
    ("open_meteo_icon", "sfo", 9):  -1.5,
    ("open_meteo_icon", "sfo", 10): -2.1,
    ("open_meteo_icon", "sfo", 11): -1.8,

    # ── open_meteo_gem — n≈1248 (≈3.7yr); highest variance, worst city outliers ─

    # Atlanta — strongly cold all year
    ("open_meteo_gem", "atl", 1):  -2.0,
    ("open_meteo_gem", "atl", 2):  -1.5,
    ("open_meteo_gem", "atl", 3):  -1.9,
    ("open_meteo_gem", "atl", 4):  -2.1,
    ("open_meteo_gem", "atl", 5):  -2.6,
    ("open_meteo_gem", "atl", 6):  -2.8,
    ("open_meteo_gem", "atl", 7):  -2.9,
    ("open_meteo_gem", "atl", 8):  -2.3,
    ("open_meteo_gem", "atl", 9):  -2.4,
    ("open_meteo_gem", "atl", 10): -2.8,
    ("open_meteo_gem", "atl", 11): -2.3,
    ("open_meteo_gem", "atl", 12): -1.8,

    # Austin
    ("open_meteo_gem", "aus", 4):  +1.3,
    ("open_meteo_gem", "aus", 6):  -1.2,
    ("open_meteo_gem", "aus", 8):  -1.1,
    ("open_meteo_gem", "aus", 9):  -1.9,

    # Boston
    ("open_meteo_gem", "bos", 6):  +1.3,
    ("open_meteo_gem", "bos", 7):  +1.3,
    ("open_meteo_gem", "bos", 8):  +1.5,
    ("open_meteo_gem", "bos", 9):  +1.0,

    # Chicago — warm bias spring/summer
    ("open_meteo_gem", "chi", 3):  +1.0,
    ("open_meteo_gem", "chi", 4):  +1.5,
    ("open_meteo_gem", "chi", 5):  +1.5,
    ("open_meteo_gem", "chi", 6):  +2.1,
    ("open_meteo_gem", "chi", 7):  +1.8,
    ("open_meteo_gem", "chi", 8):  +2.0,

    # Dallas
    ("open_meteo_gem", "dal", 2):  -1.1,
    ("open_meteo_gem", "dal", 7):  -1.2,
    ("open_meteo_gem", "dal", 8):  -1.8,
    ("open_meteo_gem", "dal", 9):  -1.5,

    # Washington DC — strongly cold spring
    ("open_meteo_gem", "dca", 1):  -1.2,
    ("open_meteo_gem", "dca", 2):  -1.8,
    ("open_meteo_gem", "dca", 3):  -3.2,
    ("open_meteo_gem", "dca", 4):  -2.4,
    ("open_meteo_gem", "dca", 5):  -1.1,
    ("open_meteo_gem", "dca", 6):  -1.7,
    ("open_meteo_gem", "dca", 9):  -1.1,
    ("open_meteo_gem", "dca", 10): -1.5,
    ("open_meteo_gem", "dca", 11): -1.7,
    ("open_meteo_gem", "dca", 12): -1.4,

    # Denver
    ("open_meteo_gem", "den", 8):  -1.4,

    # DFW
    ("open_meteo_gem", "dfw", 2):  -1.1,
    ("open_meteo_gem", "dfw", 7):  -1.2,
    ("open_meteo_gem", "dfw", 8):  -1.8,
    ("open_meteo_gem", "dfw", 9):  -1.5,

    # Houston
    ("open_meteo_gem", "hou", 4):  +1.3,
    ("open_meteo_gem", "hou", 5):  +1.3,

    # Las Vegas — warm spring and fall
    ("open_meteo_gem", "las", 2):  +1.7,
    ("open_meteo_gem", "las", 3):  +1.9,
    ("open_meteo_gem", "las", 4):  +1.5,
    ("open_meteo_gem", "las", 9):  +1.7,
    ("open_meteo_gem", "las", 10): +1.2,

    # Los Angeles — catastrophically warm (worst source×city in dataset)
    ("open_meteo_gem", "lax", 2):  +1.2,
    ("open_meteo_gem", "lax", 3):  +2.3,
    ("open_meteo_gem", "lax", 4):  +2.4,
    ("open_meteo_gem", "lax", 5):  +2.7,
    ("open_meteo_gem", "lax", 6):  +2.9,
    ("open_meteo_gem", "lax", 7):  +5.3,
    ("open_meteo_gem", "lax", 8):  +4.8,
    ("open_meteo_gem", "lax", 9):  +3.8,
    ("open_meteo_gem", "lax", 10): +2.8,
    ("open_meteo_gem", "lax", 11): +1.2,
    ("open_meteo_gem", "lax", 12): +1.1,

    # Miami
    ("open_meteo_gem", "mia", 4):  +1.0,
    ("open_meteo_gem", "mia", 5):  +1.2,

    # Minneapolis — warm summer
    ("open_meteo_gem", "msp", 5):  +1.0,
    ("open_meteo_gem", "msp", 6):  +1.6,
    ("open_meteo_gem", "msp", 7):  +2.5,
    ("open_meteo_gem", "msp", 8):  +2.4,
    ("open_meteo_gem", "msp", 9):  +1.4,

    # New Orleans
    ("open_meteo_gem", "msy", 7):  -1.0,
    ("open_meteo_gem", "msy", 8):  -1.4,
    ("open_meteo_gem", "msy", 9):  -1.5,
    ("open_meteo_gem", "msy", 10): -1.7,

    # New York — massively warm spring/summer
    ("open_meteo_gem", "ny",  5):  +2.6,
    ("open_meteo_gem", "ny",  6):  +3.6,
    ("open_meteo_gem", "ny",  7):  +3.5,
    ("open_meteo_gem", "ny",  8):  +3.2,
    ("open_meteo_gem", "ny",  9):  +1.7,
    ("open_meteo_gem", "ny",  10): +1.8,

    # Oklahoma City
    ("open_meteo_gem", "okc", 6):  +1.1,
    ("open_meteo_gem", "okc", 8):  +1.1,

    # Philadelphia
    ("open_meteo_gem", "phl", 6):  +1.7,
    ("open_meteo_gem", "phl", 7):  +2.0,
    ("open_meteo_gem", "phl", 8):  +2.2,

    # Phoenix
    ("open_meteo_gem", "phx", 5):  -1.1,
    ("open_meteo_gem", "phx", 6):  -1.7,
    ("open_meteo_gem", "phx", 7):  -1.5,
    ("open_meteo_gem", "phx", 8):  -2.2,
    ("open_meteo_gem", "phx", 9):  -1.6,
    ("open_meteo_gem", "phx", 10): -1.3,

    # San Antonio
    ("open_meteo_gem", "sat", 2):  +1.3,
    ("open_meteo_gem", "sat", 8):  -1.1,

    # Seattle
    ("open_meteo_gem", "sea", 4):  +1.4,
    ("open_meteo_gem", "sea", 5):  +2.0,
    ("open_meteo_gem", "sea", 6):  +1.6,
    ("open_meteo_gem", "sea", 8):  +1.4,

    # San Francisco — catastrophically cold spring/summer (worst cold outlier)
    ("open_meteo_gem", "sfo", 2):  -1.4,
    ("open_meteo_gem", "sfo", 3):  -3.2,
    ("open_meteo_gem", "sfo", 4):  -4.3,
    ("open_meteo_gem", "sfo", 5):  -5.5,
    ("open_meteo_gem", "sfo", 6):  -5.3,
    ("open_meteo_gem", "sfo", 7):  -5.8,
    ("open_meteo_gem", "sfo", 8):  -5.6,
    ("open_meteo_gem", "sfo", 9):  -4.5,
    ("open_meteo_gem", "sfo", 10): -5.0,
    ("open_meteo_gem", "sfo", 11): -2.5,
}
