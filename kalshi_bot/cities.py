"""Canonical city registry for Kalshi temperature markets. Single source of truth for city coordinates, timezones, station IDs, and NWS CLI location codes. To add a new city, edit only this file."""

from zoneinfo import ZoneInfo

_ET  = ZoneInfo("America/New_York")
_CT  = ZoneInfo("America/Chicago")
_MT  = ZoneInfo("America/Denver")
_PT  = ZoneInfo("America/Los_Angeles")
_PHX = ZoneInfo("America/Phoenix")    # no DST observed in Arizona

# Cities with Kalshi KXHIGH* temperature markets.
# metric key → (display name, latitude, longitude)
#
# Verified 2026-03-09 by exhaustive probe of all plausible KXHIGH{CODE}
# ticker prefixes across 30-day horizon — Kalshi currently offers daily
# high-temp markets only for these 9 cities.  DAL/BOS/HOU are inactive
# at time of writing but included so forecasts are ready if they return.
# To add a new city: add a row here AND add the ticker prefix mapping in
# market_parser.py → TICKER_TO_METRIC and _NUMERIC_PATTERN_PREFIXES.
CITIES: dict[str, tuple[str, float, float, ZoneInfo]] = {
    # Coordinates are the official NWS ASOS airport stations that Kalshi
    # uses for temperature market settlement.  Using city-centre coordinates
    # caused station mismatch (e.g. downtown LA vs. coastal KLAX), producing
    # systematic forecast errors of 3–8°F.
    # Timezone is used to compute local midnight for the observation window
    # (NWS API start= parameter) so the full calendar day's readings are
    # captured, not just the hours since midnight UTC.
    #
    # Consistently active (confirmed 2026-03-09)
    "temp_high_lax": ("Los Angeles", 33.9425, -118.4081, _PT),  # KLAX airport
    "temp_high_den": ("Denver",      39.8561, -104.6737, _MT),  # KDEN airport
    "temp_high_chi": ("Chicago",     41.7868,  -87.7522, _CT),  # KMDW Midway
    "temp_high_ny":  ("New York",    40.7789,  -73.9692, _ET),  # KNYC Central Park
    "temp_high_mia": ("Miami",       25.7959,  -80.2870, _ET),  # KMIA airport
    "temp_high_aus": ("Austin",      30.1975,  -97.6664, _CT),  # KAUS airport
    # Previously inactive; kept for old KXHIGHDAL series (Love Field settlement)
    "temp_high_dal": ("Dallas",      32.8479,  -96.8514, _CT),  # KDAL Love Field
    "temp_high_bos": ("Boston",      42.3643,  -71.0052, _ET),  # KBOS airport
    "temp_high_hou": ("Houston",     29.6454,  -95.2789, _CT),  # KHOU Hobby
    # KXHIGHTDAL settles against DFW (confirmed by rules_primary), not Love Field
    "temp_high_dfw": ("Dallas/Fort Worth", 32.8998,  -97.0403, _CT),  # KDFW
    # New cities — active from 2026-04
    "temp_high_sfo": ("San Francisco", 37.6190, -122.3750, _PT),   # KSFO
    "temp_high_sea": ("Seattle",       47.4502, -122.3088, _PT),   # KSEA
    "temp_high_phx": ("Phoenix",       33.4373, -112.0078, _PHX),  # KPHX (no DST)
    "temp_high_phl": ("Philadelphia",  39.8729,  -75.2437, _ET),   # KPHL
    "temp_high_atl": ("Atlanta",       33.6407,  -84.4277, _ET),   # KATL
    "temp_high_msp": ("Minneapolis",   44.8848,  -93.2223, _CT),   # KMSP
    "temp_high_dca": ("Washington DC", 38.8512,  -77.0402, _ET),   # KDCA
    "temp_high_las": ("Las Vegas",     36.0840, -115.1537, _PT),   # KLAS
    "temp_high_okc": ("Oklahoma City", 35.3931,  -97.6007, _CT),   # KOKC
    "temp_high_sat": ("San Antonio",   29.5337,  -98.4698, _CT),   # KSAT
    "temp_high_msy": ("New Orleans",   29.9934,  -90.2580, _CT),   # KMSY
}

# Cities with Kalshi KXLOWT* daily low temperature markets.
# Same coordinates/timezones as CITIES (same settlement stations).
# Do NOT add temp_low_* entries to KALSHI_STATION_IDS — that breaks METAR's
# reverse map.  Low-temp fetchers derive the station ID from the corresponding
# temp_high_* key.
LOW_CITIES: dict[str, tuple[str, float, float, ZoneInfo]] = {
    "temp_low_lax": ("Los Angeles",      33.9425, -118.4081, _PT),   # KLAX
    "temp_low_den": ("Denver",           39.8561, -104.6737, _MT),   # KDEN
    "temp_low_chi": ("Chicago",          41.7868,  -87.7522, _CT),   # KMDW
    "temp_low_ny":  ("New York",         40.7789,  -73.9692, _ET),   # KNYC
    "temp_low_mia": ("Miami",            25.7959,  -80.2870, _ET),   # KMIA
    "temp_low_aus": ("Austin",           30.1975,  -97.6664, _CT),   # KAUS
    "temp_low_bos": ("Boston",           42.3643,  -71.0052, _ET),   # KBOS
    "temp_low_hou": ("Houston",          29.6454,  -95.2789, _CT),   # KHOU
    "temp_low_dfw": ("Dallas/Fort Worth",32.8998,  -97.0403, _CT),   # KDFW
    "temp_low_sfo": ("San Francisco",    37.6190, -122.3750, _PT),   # KSFO
    "temp_low_sea": ("Seattle",          47.4502, -122.3088, _PT),   # KSEA
    "temp_low_phx": ("Phoenix",          33.4373, -112.0078, _PHX),  # KPHX (no DST)
    "temp_low_phl": ("Philadelphia",     39.8729,  -75.2437, _ET),   # KPHL
    "temp_low_atl": ("Atlanta",          33.6407,  -84.4277, _ET),   # KATL
    "temp_low_msp": ("Minneapolis",      44.8848,  -93.2223, _CT),   # KMSP
    "temp_low_dca": ("Washington DC",    38.8512,  -77.0402, _ET),   # KDCA
    "temp_low_las": ("Las Vegas",        36.0840, -115.1537, _PT),   # KLAS
    "temp_low_okc": ("Oklahoma City",    35.3931,  -97.6007, _CT),   # KOKC
    "temp_low_sat": ("San Antonio",      29.5337,  -98.4698, _CT),   # KSAT
    "temp_low_msy": ("New Orleans",      29.9934,  -90.2580, _CT),   # KMSY
}

# Kalshi's authoritative NWS ASOS station identifiers for settlement.
#
# These stations define which physical thermometer Kalshi uses to resolve
# each market.  Fetching NWS observations for ANY other station would
# produce systematic 3–8°F mismatch errors.
#
# These hard-coded station identifiers bypass the dynamic resolution entirely.
# They were verified against Kalshi market rules_primary text and NWS ASOS
# records.  Update if Kalshi changes its resolution station for any city.
KALSHI_STATION_IDS: dict[str, str] = {
    "temp_high_lax": "KLAX",   # Los Angeles International Airport
    "temp_high_den": "KDEN",   # Denver International Airport (5,431 ft)
    "temp_high_chi": "KMDW",   # Chicago Midway International Airport
    "temp_high_ny":  "KNYC",   # New York — Central Park (per Kalshi market rules)
    "temp_high_mia": "KMIA",   # Miami International Airport
    "temp_high_aus": "KAUS",   # Austin-Bergstrom International Airport
    "temp_high_dal": "KDAL",   # Dallas Love Field Airport
    "temp_high_bos": "KBOS",   # Boston Logan International Airport
    "temp_high_hou": "KHOU",   # Houston William P. Hobby Airport
    "temp_high_dfw": "KDFW",   # Dallas/Fort Worth International Airport (KXHIGHTDAL)
    # New cities — active from 2026-04
    "temp_high_sfo": "KSFO",   # San Francisco International Airport
    "temp_high_sea": "KSEA",   # Seattle-Tacoma International Airport
    "temp_high_phx": "KPHX",   # Phoenix Sky Harbor International Airport
    "temp_high_phl": "KPHL",   # Philadelphia International Airport
    "temp_high_atl": "KATL",   # Hartsfield-Jackson Atlanta International Airport
    "temp_high_msp": "KMSP",   # Minneapolis-Saint Paul International Airport
    "temp_high_dca": "KDCA",   # Ronald Reagan Washington National Airport
    "temp_high_las": "KLAS",   # Harry Reid International Airport (Las Vegas)
    "temp_high_okc": "KOKC",   # Will Rogers World Airport (Oklahoma City)
    "temp_high_sat": "KSAT",   # San Antonio International Airport
    "temp_high_msy": "KMSY",   # Louis Armstrong New Orleans International Airport
}

# Derived look-ups used by open_meteo.py, backtest scripts, and audit scripts.
CITY_TZ: dict[str, ZoneInfo] = {k: v[3] for k, v in CITIES.items()}
CITY_TZ.update({k: v[3] for k, v in LOW_CITIES.items()})
CITY_TZ_STRINGS: dict[str, str] = {k: str(v[3]) for k, v in CITIES.items()}
CITY_TZ_STRINGS.update({k: str(v[3]) for k, v in LOW_CITIES.items()})

# NWS 3-letter location IDs for the CLI product, keyed by Kalshi metric.
# These match the station IDs in KALSHI_STATION_IDS (same settlement stations).
CLIMO_LOCATIONS: dict[str, tuple[str, str, ZoneInfo]] = {
    # metric              location  display     timezone
    "temp_high_ny":  ("NYC", "New York/Central Park",    ZoneInfo("America/New_York")),
    "temp_high_bos": ("BOS", "Boston",                   ZoneInfo("America/New_York")),
    "temp_high_mia": ("MIA", "Miami",                    ZoneInfo("America/New_York")),
    "temp_high_chi": ("MDW", "Chicago Midway",           ZoneInfo("America/Chicago")),
    "temp_high_dal": ("DAL", "Dallas Love Field",        ZoneInfo("America/Chicago")),
    "temp_high_dfw": ("DFW", "Dallas/Fort Worth",        ZoneInfo("America/Chicago")),
    "temp_high_aus": ("AUS", "Austin",                   ZoneInfo("America/Chicago")),
    "temp_high_hou": ("HOU", "Houston Hobby",            ZoneInfo("America/Chicago")),
    "temp_high_den": ("DEN", "Denver",                   ZoneInfo("America/Denver")),
    "temp_high_lax": ("LAX", "Los Angeles",              ZoneInfo("America/Los_Angeles")),
    # New cities — active from 2026-04
    "temp_high_sfo": ("SFO", "San Francisco",            ZoneInfo("America/Los_Angeles")),
    "temp_high_sea": ("SEA", "Seattle",                  ZoneInfo("America/Los_Angeles")),
    "temp_high_phx": ("PHX", "Phoenix",                  ZoneInfo("America/Phoenix")),
    "temp_high_phl": ("PHL", "Philadelphia",             ZoneInfo("America/New_York")),
    "temp_high_atl": ("ATL", "Atlanta",                  ZoneInfo("America/New_York")),
    "temp_high_msp": ("MSP", "Minneapolis",              ZoneInfo("America/Chicago")),
    "temp_high_dca": ("DCA", "Washington DC",            ZoneInfo("America/New_York")),
    "temp_high_las": ("LAS", "Las Vegas",                ZoneInfo("America/Los_Angeles")),
    "temp_high_okc": ("OKC", "Oklahoma City",            ZoneInfo("America/Chicago")),
    "temp_high_sat": ("SAT", "San Antonio",              ZoneInfo("America/Chicago")),
    "temp_high_msy": ("MSY", "New Orleans",              ZoneInfo("America/Chicago")),
}

# Same NWS 3-letter CLI location codes, but for daily low temperature metrics.
LOW_CLIMO_LOCATIONS: dict[str, tuple[str, str, ZoneInfo]] = {
    "temp_low_ny":  ("NYC", "New York/Central Park",    ZoneInfo("America/New_York")),
    "temp_low_bos": ("BOS", "Boston",                   ZoneInfo("America/New_York")),
    "temp_low_mia": ("MIA", "Miami",                    ZoneInfo("America/New_York")),
    "temp_low_chi": ("MDW", "Chicago Midway",           ZoneInfo("America/Chicago")),
    "temp_low_dfw": ("DFW", "Dallas/Fort Worth",        ZoneInfo("America/Chicago")),
    "temp_low_aus": ("AUS", "Austin",                   ZoneInfo("America/Chicago")),
    "temp_low_hou": ("HOU", "Houston Hobby",            ZoneInfo("America/Chicago")),
    "temp_low_den": ("DEN", "Denver",                   ZoneInfo("America/Denver")),
    "temp_low_lax": ("LAX", "Los Angeles",              ZoneInfo("America/Los_Angeles")),
    "temp_low_sfo": ("SFO", "San Francisco",            ZoneInfo("America/Los_Angeles")),
    "temp_low_sea": ("SEA", "Seattle",                  ZoneInfo("America/Los_Angeles")),
    "temp_low_phx": ("PHX", "Phoenix",                  ZoneInfo("America/Phoenix")),
    "temp_low_phl": ("PHL", "Philadelphia",             ZoneInfo("America/New_York")),
    "temp_low_atl": ("ATL", "Atlanta",                  ZoneInfo("America/New_York")),
    "temp_low_msp": ("MSP", "Minneapolis",              ZoneInfo("America/Chicago")),
    "temp_low_dca": ("DCA", "Washington DC",            ZoneInfo("America/New_York")),
    "temp_low_las": ("LAS", "Las Vegas",                ZoneInfo("America/Los_Angeles")),
    "temp_low_okc": ("OKC", "Oklahoma City",            ZoneInfo("America/Chicago")),
    "temp_low_sat": ("SAT", "San Antonio",              ZoneInfo("America/Chicago")),
    "temp_low_msy": ("MSY", "New Orleans",              ZoneInfo("America/Chicago")),
}
