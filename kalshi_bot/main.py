"""Main async polling loop for the Kalshi information-alpha bot.

Each cycle concurrently fetches:
  - Kalshi open markets
  - Federal Register documents  (keyword matching)
  - RSS feeds (AP, Reuters, BBC, NPR, ESPN, Billboard, Politico, The Hill)
  - SEC EDGAR 8-K filings       (keyword matching)
  - NOAA / OWM city forecasts   (numeric matching)
  - Binance crypto prices       (numeric matching)
  - Frankfurter forex rates     (numeric matching)
  - BLS economic releases       (numeric matching, new releases only)
  - FRED interest rates         (numeric matching)
  - EIA energy prices           (numeric matching)
  - Polymarket / Metaculus / Manifold forecasts (divergence matching)

Results are deduplicated, matched against markets, and printed to stdout.
"""

import asyncio
import collections
from datetime import datetime, timezone, timedelta, date
import logging
import os
import re
import time
from zoneinfo import ZoneInfo

import aiohttp

from .markets import fetch_all_markets, fetch_market_detail, fetch_markets_by_series, NUMERIC_SERIES, TEXT_SERIES
from .market_parser import scan_unknown_series, parse_market, TICKER_TO_METRIC
from .matcher import find_opportunities, Opportunity
from .data import DataPoint
from .numeric_matcher import find_numeric_opportunities, NumericOpportunity
from .spread_matcher import find_spread_opportunities, SpreadOpportunity
from .arb_detector import (
    find_arb_opportunities, ArbOpportunity, ARB_MIN_PROFIT_CENTS,
    find_crossed_book_opportunities, CrossedBookArb, CROSSED_BOOK_MIN_PROFIT,
)
from .bracket_arb import find_bracket_set_opportunities, BracketSetArb, BRACKET_ARB_MIN_PROFIT, BRACKET_ARB_ENABLED
from .strike_arb import find_band_arbs, find_forecast_nos, BAND_ARB_EXECUTION_ENABLED, FORECAST_NO_ENABLED
from .polymarket_matcher import match_poly_to_kalshi, match_metaculus_to_kalshi, match_manifold_to_kalshi, match_predictit_to_kalshi, PolyOpportunity
from .news import federal_register
from .news import noaa, owm, open_meteo, nws_hourly, weatherapi, binance, coinbase, frankfurter, yahoo_forex, bls, rss, nws_alerts, fred, eia, eia_inventory, cme_fedwatch, hrrr, congress, whitehouse, equity_index, nws_climo, adp, chicago_pmi, metar, wti_futures
from .news import polymarket, metaculus, manifold, edgar, predictit
from .news import box_office
from .box_office_matcher import match_box_office_to_kalshi
from .dry_run_ledger import DryRunLedger
from .opportunity_log import OpportunityLog
from .portfolio import fetch_positions, build_position_index, summarise_portfolio
from .trade_executor import TRADE_DRY_RUN, TradeExecutor, set_drawdown_factor, POLY_ENABLED
from .scoring import (
    score_text_opportunity,
    score_numeric_opportunity,
    score_poly_opportunity,
    METRIC_EDGE_SCALES,
)
from .release_schedule import is_within_release_window, next_release
from .state import SeenDocuments
from .win_rate_tracker import WinRateTracker, WIN_RATE_REPORT_INTERVAL
from .analytics import run_attribution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS: int = int(os.environ.get("POLL_INTERVAL", "60"))

# Faster poll interval used during high-opportunity windows.  Must be ≤
# POLL_INTERVAL_SECONDS to have any effect; if larger, it is ignored.
# Default 20s: catches EIA repricing and band arb crossings 3× more often.
POLL_INTERVAL_FAST: int = int(os.environ.get("POLL_INTERVAL_FAST", "20"))

# Number of minutes after each EIA/natgas release to use POLL_INTERVAL_FAST.
# EIA KXWTI markets can reprice within 1–2 minutes of the 10:30 AM release —
# a fast poll window of 10 minutes captures the full edge window.
POLL_INTERVAL_EIA_WINDOW_MINUTES: int = int(
    os.environ.get("POLL_INTERVAL_EIA_WINDOW_MINUTES", "10")
)

# ET hour range during which METAR band-arb crossings are most likely.
# 13:00–21:00 ET covers the afternoon peak-heating window for all US time zones:
#   Eastern cities: peak by ~4:30 PM ET
#   Central cities: peak by ~4:30 PM CT = 5:30 PM ET
#   Mountain cities: peak by ~4:30 PM MT = 6:30 PM ET
#   Pacific cities:  peak by ~4:30 PM PT = 7:30 PM ET
# Expressed as integer hours (start inclusive, end exclusive).
POLL_BAND_ARB_START_ET_HOUR: int = int(os.environ.get("POLL_BAND_ARB_START_ET_HOUR", "13"))
POLL_BAND_ARB_END_ET_HOUR:   int = int(os.environ.get("POLL_BAND_ARB_END_ET_HOUR",   "21"))

# Fast inner loop: METAR + near-threshold KXHIGH price refresh between full cycles.
# WATCH_THRESHOLD_F: cities within this many °F of a band ceiling go on the watchlist.
# FAST_LOOP_INTERVAL: seconds between fast loop iterations during the sleep window.
WATCH_THRESHOLD_F: float = float(os.environ.get("WATCH_THRESHOLD_F", "3.0"))
FAST_LOOP_INTERVAL: float = float(os.environ.get("FAST_LOOP_INTERVAL", "10.0"))

# Shared state between _poll() and _fast_loop().
# _near_threshold_cities: metric keys (e.g. "temp_high_den") whose METAR reading
#   was within WATCH_THRESHOLD_F of a band ceiling in the last full cycle.
# _last_noaa_obs: NOAA observed values cached from the last full cycle for
#   use as corroboration in _fast_loop() without re-fetching NOAA.
# _last_noaa_obs_time: monotonic timestamp of when _last_noaa_obs was populated.
#   Used to detect stale cache after a poll gap; if older than NOAA_OBS_MAX_AGE_S
#   the fast loop falls back to NOAA-None mode (market-price confirmation only).
_near_threshold_cities: set[str] = set()
_last_noaa_obs: dict[str, float] = {}
_last_noaa_obs_time: float = 0.0  # time.monotonic(); 0 = never populated

# Maximum age (seconds) of the cached NOAA observed values before the fast loop
# treats them as absent (falls back to NOAA-None mode).  After a poll gap > 30 min
# the cached reading may no longer represent the current observed daily max.
NOAA_OBS_MAX_AGE_S: float = float(os.environ.get("NOAA_OBS_MAX_AGE_S", "1800"))

# How often to re-fetch the NUMERIC series markets (KXHIGH, KXBTCD, etc.).
# These are the markets the bot actually trades; stale prices cause bad fills.
# Default 60s: ~5s fetch cost, ensures prices are never more than ~90s old.
NUMERIC_MARKET_REFRESH_INTERVAL: int = int(os.environ.get("NUMERIC_MARKET_REFRESH_INTERVAL", "60"))

# How often to re-fetch the general (text/keyword) market list.
# Used only for Federal Register / RSS keyword matching; these markets move
# slowly so a 5-minute cache is fine.  ~20s fetch cost, infrequent.
GENERAL_MARKET_REFRESH_INTERVAL: int = int(os.environ.get("GENERAL_MARKET_REFRESH_INTERVAL", "300"))

# Legacy alias — if set, overrides both intervals for backwards compatibility.
_MARKET_REFRESH_OVERRIDE: int | None = (
    int(os.environ["MARKET_REFRESH_INTERVAL"])
    if "MARKET_REFRESH_INTERVAL" in os.environ
    else None
)
if _MARKET_REFRESH_OVERRIDE is not None:
    NUMERIC_MARKET_REFRESH_INTERVAL = _MARKET_REFRESH_OVERRIDE
    GENERAL_MARKET_REFRESH_INTERVAL = _MARKET_REFRESH_OVERRIDE

# Federal Register agencies to monitor.
# Each slug maps to an agency's Federal Register publication stream.
# Documents from all agencies are routed to the politics_economics source group.
AGENCIES: list[str] = [
    "environmental-protection-agency",          # EPA — climate, emissions, env rules
    "food-and-drug-administration",             # FDA — drug approvals, recalls, guidance
    "federal-trade-commission",                 # FTC — antitrust, consumer protection
    "commodity-futures-trading-commission",     # CFTC — derivatives, crypto regulation
    "department-of-the-treasury",              # Treasury — sanctions, debt, tax rules
    "department-of-health-and-human-services", # HHS — Medicare, Medicaid, drug pricing
    "securities-and-exchange-commission",       # SEC — earnings, disclosures, IPOs
    "consumer-financial-protection-bureau",     # CFPB — consumer finance rules
    "department-of-labor",                      # DOL — jobs, wages, OSHA
    "federal-communications-commission",        # FCC — telecom, broadband rules
]

# ---------------------------------------------------------------------------
# Source-scoped topic groups
#
# Each group routes a specific set of feed IDs to a curated topic list and a
# filtered subset of Kalshi markets. This prevents cross-category false
# positives (e.g. "Houston" in a Texas politics article matching NBA markets,
# or "Boston" in an economics article matching Celtics player props).
#
# Routing key per doc: doc.get("feed_id") for RSS articles,
#                      doc.get("_source") for Federal Register docs.
# ---------------------------------------------------------------------------

_SOURCE_GROUPS: list[dict] = [
    {
        # ESPN + Motorsport feeds → ball-sport and F1 markets
        "name": "sports",
        "feed_ids": {"espn_nba", "espn_top", "espn_nhl", "espn_mlb", "motorsport_f1"},
        "topics": [
            # --- NBA ---
            # "NBA" intentionally omitted: too broad, hits every KXNBA ticker.
            # Player names and team cities provide specific matching instead.
            "LeBron", "Giannis", "Curry", "Durant",
            "Antetokounmpo", "Doncic", "Embiid", "Jokic",
            "Houston", "Milwaukee", "Golden State",
            "Los Angeles Lakers", "Los Angeles Clippers",
            # --- NHL ---
            "NHL", "McDavid", "MacKinnon", "Draisaitl", "Ovechkin",
            "Pastrnak", "Tkachuk", "Crosby", "Kucherov",
            "Matthews", "Marchessault", "Hughes", "Caufield",
            # --- NCAA March Madness ---
            "NCAA", "March Madness", "Final Four", "Elite Eight", "Sweet Sixteen",
            # --- MLB ---
            "MLB", "Ohtani", "Trout", "Judge", "Soto",
            "Betts", "Freeman", "Schwarber",
            # --- Formula 1 ---
            "Formula 1", "Grand Prix", "Verstappen", "Hamilton", "Norris",
            "Leclerc", "Sainz", "Russell", "Alonso", "Piastri",
            "drivers championship", "constructors championship",
        ],
        "include_prefixes": ("KXNBA", "KXNHL", "KXNCAAMB", "KXMLB", "KXF1"),
        "exclude_prefixes": (),
    },
    # NOTE: crypto price markets (KXBTC, KXDOGE, etc.) are intentionally excluded
    # from text matching — they are covered by the numeric matcher via live
    # Binance/Coinbase prices.  Crypto news articles cannot reliably predict
    # short-duration price strikes and were generating only noise here.
    {
        # Billboard → song chart markets only
        "name": "entertainment",
        "feed_ids": {"billboard"},
        "topics": [
            "Billboard", "Hot 100", "#1 song", "number one",
            "Morgan Wallen", "Taylor Swift", "Drake", "Beyonce",
            # Multi-word phrases for higher specificity scores
            "Billboard Hot 100", "top song", "number one song",
        ],
        "include_prefixes": ("KXTOPSONG", "KXTOP10BIL"),
        "exclude_prefixes": (),
    },
    {
        # Variety + Hollywood Reporter → Rotten Tomatoes score markets (KXRT)
        # KXRT market titles always contain "Rotten Tomatoes" verbatim, so the
        # keyword matches both the article and the market title automatically.
        "name": "film_reviews",
        "feed_ids": {"variety", "hollywood_reporter"},
        "topics": [
            "Rotten Tomatoes", "Tomatometer", "critics score",
            "certified fresh", "box office", "opening weekend",
        ],
        "include_prefixes": ("KXRT",),
        "exclude_prefixes": (),
        "require_title_match": True,
    },
    {
        # Politics / economics feeds + Federal Register →
        # excludes all sports, entertainment, and weather series
        # (weather handled exclusively by numeric matcher)
        #
        # Topic design rules:
        #   - Use NOUN PHRASES, not verb phrases.  Kalshi market titles ask
        #     "Will X [verb]?" so they use base verbs ("sign", "fire") while
        #     news headlines use conjugated forms ("signs", "fired").  Nouns
        #     appear verbatim in BOTH article text AND market titles.
        #   - Keep to 1–3 words.  Long compound phrases rarely appear as a
        #     consecutive verbatim substring in either headline or market title.
        #   - require_title_match=False lets abstract/body content match even
        #     when the headline uses a paraphrase (e.g. headline: "Senate
        #     Advances Spending Bill"; abstract: "…amid a government shutdown
        #     deadline…").  Scoring + TRADE_MIN_SCORE gate false positives.
        "name": "politics_economics",
        "feed_ids": {
            "politico_congress", "politico_healthcare", "politico_defense",
            "politico_economy", "politico_energy", "politico_politics",
            "thehill", "npr", "bbc", "ap_top", "ap_politics", "reuters",
            "federal_register",
            "cnbc_top", "marketwatch",
        },
        "topics": [
            # Environment / EPA
            "EPA", "carbon emissions", "methane",
            # FDA — drug approvals, recalls, emergency authorizations
            "FDA", "drug approval", "drug recall", "vaccine", "PDUFA",
            "clinical trial", "emergency use", "NDA", "BLA", "EUA",
            # FTC / DOJ — antitrust
            "FTC", "antitrust", "merger", "monopoly",
            # CFTC — derivatives, crypto regulation
            "CFTC", "crypto regulation",
            # Treasury — sanctions, debt, fiscal
            "sanctions", "debt ceiling", "OFAC", "Treasury",
            # HHS — healthcare
            "Medicare", "Medicaid", "drug pricing",
            # Economics — data releases
            "Federal Reserve", "interest rate", "inflation",
            "CPI", "nonfarm payroll", "unemployment", "GDP", "recession",
            "rate hike", "rate cut", "jobs report",
            # Trump — NOUN phrases that appear in both news and market titles
            "executive order", "pardon", "national emergency",
            "tariff", "trade deal", "trade war",
            "Fed Chair", "Supreme Court",
            "government shutdown", "continuing resolution",
            # Congress / legislation — noun phrases only
            "reconciliation", "filibuster", "cloture",
            "spending bill", "appropriations",
            "impeachment", "Senate confirmation",
            # Foreign policy
            "ceasefire", "Ukraine", "NATO", "Iran nuclear",
            # Trump Cabinet — person names map directly to open KXCABOUT markets
            # ("Will [Name] be the next to leave the Trump Cabinet?")
            "Gabbard", "Hegseth", "Rubio", "Bessent", "Kennedy",
            "Wiles", "Vought", "Duffy", "Turner", "Waltz",
            "Mullin", "Kratsios", "Zeldin", "McMahon", "Loeffler",
            "Ratcliffe", "Greer", "Lutnick", "Burgum", "Wright",
            "Rollins", "Collins", "Chavez-DeRemer",
            # Crypto (supplements numeric matching)
            "Bitcoin", "Ethereum", "crypto",
            # Equities / financial markets (CNBC, MarketWatch)
            "S&P 500", "Nasdaq", "earnings", "stock market", "IPO", "Fed rate",
            # SEC / DOL / FCC (new Federal Register agencies)
            "SEC", "securities", "CFPB", "OSHA", "FCC", "broadband",
        ],
        "include_prefixes": (),
        # Exclude all sports, entertainment, weather, and esports series
        "exclude_prefixes": (
            "KXNBA", "KXNHL", "KXNCAAMB", "KXMLB", "KXATP", "KXWBC",
            "KXLOL", "KXVALORANT", "KXMVE",
            "KXTOPSONG", "KXTOP10BIL", "KXRT", "KXMAMDANIM",
            "KXHIGH",
        ),
        # Abstract-level matching: article headlines often paraphrase the
        # specific topic that appears verbatim in the body/abstract.
        "require_title_match": False,
    },
    {
        # SEC EDGAR 8-K filings → company/stock and macro markets
        # 8-K events: earnings surprises, CEO/CFO changes, M&A, bankruptcies,
        # FDA decisions, restatements — all market-moving hard-news events.
        "name": "edgar_filings",
        "feed_ids": {"edgar"},
        "topics": [
            # Company actions
            "acquisition", "merger", "takeover", "bankruptcy", "restructuring",
            "restatement", "settlement", "dividend", "buyback", "spinoff",
            # Executive changes
            "CEO", "CFO", "chairman", "resign", "appoint", "executive",
            # Earnings / financial
            "earnings", "revenue", "guidance", "profit", "loss", "outlook",
            # Regulatory
            "FDA", "approval", "recall", "investigation", "lawsuit", "fine",
            "SEC", "DOJ", "FTC", "antitrust",
            # Macro / market indices
            "S&P 500", "Nasdaq", "Dow Jones", "Russell",
            # Notable companies Kalshi tends to have markets on
            "Apple", "Tesla", "Microsoft", "Nvidia", "Amazon", "Google",
            "Meta", "Alphabet", "OpenAI", "SpaceX",
        ],
        # Target company/stock, index, and macro markets; exclude sports/entertainment/weather
        "include_prefixes": (),
        "exclude_prefixes": (
            "KXNBA", "KXNHL", "KXNCAAMB", "KXMLB", "KXATP", "KXWBC",
            "KXLOL", "KXVALORANT", "KXMVE",
            "KXTOPSONG", "KXTOP10BIL", "KXRT", "KXMAMDANIM",
            "KXHIGH",
        ),
        # EDGAR titles are structured stubs ("COMPANY filed 8-K"), not prose.
        # Skip Phase-2 title confirmation so abstract content (event keywords)
        # is matched directly.
        "require_title_match": False,
    },
    {
        # NWS active weather alerts → weather and natural-disaster markets
        "name": "weather_alerts",
        "feed_ids": {"nws_alerts"},
        "topics": [
            # Heat
            "Excessive Heat Warning", "Excessive Heat Watch", "Heat Advisory",
            # Cold / winter
            "Winter Storm Warning", "Winter Storm Watch",
            "Blizzard Warning", "Ice Storm Warning",
            "Freeze Warning", "Freeze Watch", "Frost Advisory",
            "Wind Chill Warning",
            # Tropical / hurricane
            "Hurricane Warning", "Hurricane Watch",
            "Tropical Storm Warning", "Tropical Storm Watch",
            # Severe / tornado
            "Tornado Warning", "Tornado Watch",
            "Severe Thunderstorm Warning",
            # High wind / fire
            "High Wind Warning", "Red Flag Warning",
        ],
        # Target weather, hurricane, and general prediction markets
        "include_prefixes": ("KXHIGH", "KXHURRICANE", "KXWEATHER", "KXTORNADO"),
        "exclude_prefixes": (),
    },
]

# Minimum distance from strike for numeric opportunities (0 = show all).
# This global default applies to all non-weather numeric metrics.
NUMERIC_MIN_EDGE: float = float(os.environ.get("NUMERIC_MIN_EDGE", "0"))

# Per-category minimum edge expressed as a fraction of each metric's reference
# scale (METRIC_EDGE_SCALES in scoring.py).  Applied after find_numeric_opportunities
# as a noise filter: any non-weather opportunity whose edge is below
# (NUMERIC_MIN_EDGE_FRACTION × metric_scale) is dropped.
# 0.0 = disabled (default — keeps existing behaviour).
# Recommended starting value: 0.05 (5% of the reference scale per metric).
# Examples at 0.05: BTC $250, ETH $15, EUR/USD 0.002, CPI 0.025pp, SPX 2.5pts.
NUMERIC_MIN_EDGE_FRACTION: float = float(os.environ.get("NUMERIC_MIN_EDGE_FRACTION", "0.0"))

# Weather-specific signal quality gates (applied after OWM consensus filter).
#
# TEMP_FORECAST_MIN_EDGE — Minimum edge (°F) required for raw NWS forecast
#   signals (source="noaa").  NWS day-1 forecasts have MAE ≈ 3–4°F, so a
#   7°F edge (≈ 2σ) is the threshold where the signal likely reflects genuine
#   information rather than forecast noise.  Downstream gates (corroboration
#   requiring 2+ sources, score gate at 0.75) handle residual false positives;
#   a higher threshold is no longer needed to maintain trade quality.
TEMP_FORECAST_MIN_EDGE: float = float(os.environ.get("TEMP_FORECAST_MIN_EDGE", "5.0"))

# Minimum edge (°F) for day-2+ NWS extended forecasts (source="noaa_day2" etc.).
# Day-2 NWS MAE ≈ 5–6°F (1.4× day-1); the same 7°F edge represents only ~1.1σ
# vs ~1.6σ for day-1. Requiring 9°F was calibrated for winter cold extremes but
# kills all day-ahead signals in spring/summer when NWS forecasts are 4–7°F from
# strikes. Lowered to 5.0°F — equivalent to same-day TEMP_FORECAST_MIN_EDGE.
# Quality is guarded downstream by FORECAST_CORROBORATION_MIN=2 (requires an
# independent model to agree) and NOAA_DAY2_MIN_SCORE=0.90 (score gate).
# Set to 0 to use TEMP_FORECAST_MIN_EDGE for all forecast sources uniformly.
TEMP_DAY2_MIN_EDGE: float = float(os.environ.get("TEMP_DAY2_MIN_EDGE", "5.0"))

# Per-source minimum edge overrides (°F), split by market direction.
#
# OVER thresholds — used when the market resolves YES if actual > strike.
#   Sources with a cold bias (forecast < actual on average) are more conservative
#   on over signals: a 5°F raw edge already implies ~10°F true edge for open_meteo.
#
# UNDER thresholds — used when the market resolves YES if actual < strike.
#   Cold-biased sources overstate apparent under-edge: their forecast is already
#   below actual, so a 7°F apparent edge may represent ~0°F true edge.
#   Set to float("inf") to disable under signals from that source entirely.
#
# Justified by:
#   noaa       — published NWS day-1 MAE ≈ 3–4°F, bias ≈ 0 → 5°F is ~1.4σ
#   open_meteo — ERA5 backtest bias ≈ -5°F cold → over gains buffer, under loses it
#   weatherapi — ERA5 backtest bias ≈ -8°F cold → under signals have ~-1°F true edge
#
# Override any value via env var (e.g. TEMP_EDGE_OVER_NOAA=4.0).
TEMP_FORECAST_MIN_EDGE_OVER: dict[str, float] = {
    "noaa":        5.0,
    "noaa_day1":   5.0,
    "nws_hourly":  5.0,  # NWS HRRR-driven hourly — same agency/station as settlement
    "open_meteo":  5.0,
    "weatherapi":  5.0,  # rarely fires; cold bias makes any over signal very reliable
}

TEMP_FORECAST_MIN_EDGE_UNDER: dict[str, float] = {
    "noaa":        5.0,
    "noaa_day1":   5.0,
    "nws_hourly":  5.0,  # NWS product, near-zero bias expected
    "open_meteo":  7.0,  # -5°F cold bias → 7°F raw nets ~2°F true edge; FORECAST_CORROBORATION_MIN=2
                          # prevents open_meteo from trading solo; its role is corroboration only.
                          # Lowered from 12°F (was too aggressive; suppressed all spring signals).
    "weatherapi":  float("inf"),  # -8°F cold bias → under signals have ~-1°F true edge
}

# Load env-var overrides: TEMP_EDGE_OVER_{SOURCE} / TEMP_EDGE_UNDER_{SOURCE}
for _src, _over_env, _under_env in [
    ("noaa",       "TEMP_EDGE_OVER_NOAA",       "TEMP_EDGE_UNDER_NOAA"),
    ("open_meteo", "TEMP_EDGE_OVER_OPEN_METEO",  "TEMP_EDGE_UNDER_OPEN_METEO"),
    ("weatherapi", "TEMP_EDGE_OVER_WEATHERAPI",  "TEMP_EDGE_UNDER_WEATHERAPI"),
    ("noaa_day2",  "TEMP_EDGE_OVER_NOAA_DAY2",   "TEMP_EDGE_UNDER_NOAA_DAY2"),
    ("hrrr",       "TEMP_EDGE_OVER_HRRR",        "TEMP_EDGE_UNDER_HRRR"),
    ("nws_hourly", "TEMP_EDGE_OVER_NWS_HOURLY",  "TEMP_EDGE_UNDER_NWS_HOURLY"),
]:
    for _d, _env in [("over", _over_env), ("under", _under_env)]:
        _v = os.environ.get(_env)
        if _v is not None:
            try:
                (_d == "over" and TEMP_FORECAST_MIN_EDGE_OVER or TEMP_FORECAST_MIN_EDGE_UNDER)[_src] = float(_v)
            except ValueError:
                pass
del _src, _over_env, _under_env, _d, _env, _v  # type: ignore[name-defined]

# ---------------------------------------------------------------------------
# Intraday temperature trajectory (obs_trajectory) configuration
# ---------------------------------------------------------------------------
# The obs_trajectory source uses individual METAR observations (already fetched
# by metar.py) to compute the warming slope and project the day's likely peak
# using a parabolic diurnal model. It fires only when the city is actively
# warming (slope > TRAJ_MIN_SLOPE_FPH) and the projected peak meaningfully
# exceeds the current temperature (by at least TRAJ_MIN_EDGE_F).
#
# Backtest (7-day, 20 cities):
#   Parabolic MAE = 1.50°F (vs naive linear = 3.34°F)
#   Mean error = -1.29°F (consistent underestimation → favorable for YES side)
#   Reliable hours: 11–15 local. Unreliable: none in the 10–16 window.
#   Recommended TRAJ_MIN_EDGE_F = 3.0°F (2× MAE).
TRAJ_START_LOCAL_HOUR:  int   = int(os.environ.get("TRAJ_START_LOCAL_HOUR", "10"))
TRAJ_END_LOCAL_HOUR:    int   = int(os.environ.get("TRAJ_END_LOCAL_HOUR", "16"))
TRAJ_LOOKBACK_HOURS:    float = float(os.environ.get("TRAJ_LOOKBACK_HOURS", "2.0"))
TRAJ_MIN_OBS:           int   = int(os.environ.get("TRAJ_MIN_OBS", "3"))
TRAJ_MIN_SLOPE_FPH:     float = float(os.environ.get("TRAJ_MIN_SLOPE_FPH", "0.3"))
TRAJ_MIN_HOURS_TO_PEAK: float = float(os.environ.get("TRAJ_MIN_HOURS_TO_PEAK", "0.5"))
TRAJ_MIN_EDGE_F:        float = float(os.environ.get("TRAJ_MIN_EDGE_F", "3.0"))

# City-specific typical dawn/peak hours (local). Dawn = daily minimum;
# peak = daily maximum. Define the phase u in the parabolic diurnal model.
# Default: dawn=6, peak=16 for all cities not listed here.
TRAJ_DAWN_LOCAL_HOUR: dict[str, int] = {
    "temp_high_mia": 7,   # Miami: ocean thermal lag
    "temp_high_lax": 7,   # LA: marine layer delays warming start
}
TRAJ_PEAK_LOCAL_HOUR: dict[str, int] = {
    "temp_high_phx": 15,  # Phoenix: desert heating peaks earlier
    "temp_high_mia": 14,  # Miami: sea breeze onset ~2 PM
    "temp_high_lax": 15,  # LA: peaks before marine layer returns
}


def _resolve_min_edge(source: str, direction: str) -> float:
    """Return the calibrated minimum edge (°F) for a source + market direction.

    Priority:
      1. Per-source direction override (TEMP_FORECAST_MIN_EDGE_OVER/UNDER)
      2. TEMP_DAY2_MIN_EDGE for noaa_day2+ (direction-agnostic legacy gate)
      3. TEMP_FORECAST_MIN_EDGE global default
    """
    if direction == "over" and source in TEMP_FORECAST_MIN_EDGE_OVER:
        return TEMP_FORECAST_MIN_EDGE_OVER[source]
    if direction == "under" and source in TEMP_FORECAST_MIN_EDGE_UNDER:
        return TEMP_FORECAST_MIN_EDGE_UNDER[source]
    _day2 = source.startswith("noaa_day") and source != "noaa_day1"
    if _day2 and TEMP_DAY2_MIN_EDGE > 0:
        return TEMP_DAY2_MIN_EDGE
    return TEMP_FORECAST_MIN_EDGE


# Minimum edge (°F) for observed-temperature YES signals, split by direction:
#
# TEMP_OBSERVED_MIN_EDGE_OVER — direction=over (observed already exceeds strike).
#   We read from the exact same ASOS airport station Kalshi uses for settlement
#   (confirmed from market rules), so station mismatch is ~0°F.  The only risk
#   is NOAA observation lag (typically 30–60 min); 0.5°F covers that comfortably.
#
# TEMP_OBSERVED_MIN_EDGE_BETWEEN — direction=between YES (observed is inside range).
#   Lower bound is confirmed (observed ≥ strike_lo).  Upper bound is only safe
#   AFTER the afternoon gate fires (same 4:30 PM local threshold as under YES).
#   Before 4:30 PM the temperature could still rise above strike_hi, so the gate
#   prevents entry.  After 4:30 PM the day's peak has passed; 0.2°F clearance
#   is sufficient — same logic as direction=over where outcome is locked.
#
# TEMP_OBSERVED_MIN_EDGE_UNDER — direction=under (observed still below strike).
#   The observed max is a lower bound; residual afternoon warming before the
#   4:30 PM gate is the main risk.  2°F provides adequate headroom given lag.
TEMP_OBSERVED_MIN_EDGE_OVER: float = float(os.environ.get("TEMP_OBSERVED_MIN_EDGE_OVER", "0.5"))
TEMP_OBSERVED_MIN_EDGE_BETWEEN: float = float(os.environ.get("TEMP_OBSERVED_MIN_EDGE_BETWEEN", "0.2"))
TEMP_OBSERVED_MIN_EDGE_UNDER: float = float(os.environ.get("TEMP_OBSERVED_MIN_EDGE_UNDER", "2.0"))

# TEMP_OBSERVED_MAX_EDGE — Upper edge cap for noaa_observed signals.
#
# Very large edges (>10°F) on an observed source indicate a faulty reading
# (e.g. sensor error or station reporting a stale value), not a genuine market
# mispricing: a real market rarely prices 10°F+ away from a temperature that has
# already been recorded at settlement.  Post-fix backtest shows all losses on
# noaa_observed were at edges ≥10°F (sensor errors), while all wins were ≤10°F.
# Cap set at 10.0°F; set to a large number to disable.
TEMP_OBSERVED_MAX_EDGE: float = float(os.environ.get("TEMP_OBSERVED_MAX_EDGE", "10.0"))

# TEMP_OBSERVED_MAX_HOURS — For noaa_observed NO signals (observed max is still
#   *below* the strike — the temperature may still rise to hit it), only surface
#   the opportunity within this many hours of market close.  Beyond this window
#   the remaining-day rise is too uncertain.  YES signals (observed max already
#   *exceeds* the strike) are always surfaced — the outcome is locked.
TEMP_OBSERVED_MAX_HOURS: float = float(os.environ.get("TEMP_OBSERVED_MAX_HOURS", "4.0"))

# HRRR_MAX_SPREAD_F — Maximum acceptable spread (°F) between the NWS daily
#   forecast high and the HRRR-derived hourly daytime high.  When the two
#   model products disagree by more than this threshold, the atmosphere is
#   unstable / convectively uncertain and the daily forecast is unreliable.
#   Signals from source="noaa" or "owm" are suppressed; noaa_observed and
#   nws_alert DataPoints are never gated (they are ground truth / official).
HRRR_MAX_SPREAD_F: float = float(os.environ.get("HRRR_MAX_SPREAD_F", "5.0"))

# MORNING_OBS_GAP_F — Maximum allowed gap (°F) between the Kalshi strike and
#   the current noaa_observed max (recorded since midnight) before a *forecast*
#   YES trade is blocked.  If the observed temperature is more than this many
#   degrees below the strike, a YES forecast is unlikely to be realized even if
#   the model says so — early-morning observations are a reality check.
#   Example: strike=92°F, gap=15°F → observed must be ≥ 77°F to trade YES.
#   Set to 0 to disable (allow any observed gap).
MORNING_OBS_GAP_F: float = float(os.environ.get("MORNING_OBS_GAP_F", "20.0"))

# NWS_HOURLY_MAX_AGE_HOURS — Maximum age of an nws_hourly DataPoint before it
#   is considered stale and dropped from corroboration.  nws_hourly updates every
#   hour; a reading older than 3 hours means at least 2 update cycles were missed,
#   indicating an API outage or network issue.  Stale readings can be dangerously
#   wrong on warm days when the temperature rises several degrees per hour.
#   The canonical failure is trade #224: nws_hourly last fetched at 2:15 AM ET
#   but was used to corroborate a 9:55 AM entry — 7+ hours stale.
#   Set to 0 to disable (allow DataPoints of any age).
NWS_HOURLY_MAX_AGE_HOURS: float = float(os.environ.get("NWS_HOURLY_MAX_AGE_HOURS", "3.0"))

# FORECAST_CORROBORATION_MIN — Minimum number of distinct forecast sources
#   (from {noaa, owm, open_meteo}) that must independently agree on a YES
#   direction before the signal is traded.  Lone signals (only one source)
#   are suppressed — they are more likely to be model noise than genuine alpha.
#   Set to 1 to disable (allow single-source signals).
FORECAST_CORROBORATION_MIN: int = int(os.environ.get("FORECAST_CORROBORATION_MIN", "2"))

# OBS_CONSENSUS_MIN — Minimum number of confirmed-observation sources
#   (noaa_observed, nws_climo) that must independently agree on a YES outcome
#   before the signal is traded.  Set to 1 to allow single-source observation
#   trades; the 4:30 PM peak-past gate on "below threshold" signals provides
#   the primary safety net in place of strict multi-source verification.
#   nws_climo is only available after ~5 PM local, so OBS_CONSENSUS_MIN=2
#   blocked all afternoon observation trades before that window.
OBS_CONSENSUS_MIN: int = int(os.environ.get("OBS_CONSENSUS_MIN", "1"))

# RELEASE_WINDOW_MINUTES — (Option D) Maximum minutes after a scheduled data
#   release during which BLS, FRED, and EIA numeric opportunities are allowed
#   to trigger trades.  Outside this window the data is already priced in.
#   Set to 0 to disable the gate (allow trades at any time).
RELEASE_WINDOW_MINUTES: int = int(os.environ.get("RELEASE_WINDOW_MINUTES", "30"))

# EIA_MAX_STALE_DAYS — Maximum age (calendar days) of EIA data before it is
# considered stale and dropped entirely.  EIA daily spot prices (RWTC for WTI,
# RNGWHHD for Henry Hub) are published with a 1-business-day lag.  Over a
# weekend that lag grows to 3 days (Friday's price available on Monday).
# Using Friday's price to trade Monday's Kalshi WTI market causes losses
# because WTI can move significantly over the weekend.
# Default: 1 (allow yesterday's data; drop anything older — effectively
# prevents trading over weekends and holidays when data is stale).
# Set to 0 to require same-day data; set to 7 to disable the gate.
EIA_MAX_STALE_DAYS: int = int(os.environ.get("EIA_MAX_STALE_DAYS", "1"))

# FOREX_MAX_STALE_DAYS — Maximum age (calendar days) of ECB/Frankfurter data
# before it is considered stale.  ECB reference rates publish once per day
# at ~16:00 CET (~10:00 AM ET).  Kalshi KXEURUSD / KXUSDJPY markets resolve
# at that same 10:00 AM ET fixing.  Before the daily ECB publication, only
# the previous day's rate is available — that rate is fully priced in and
# useless as a directional signal for today's resolution.
# Default: 0 (same-day data only; yesterday's rate is blocked).
# Set to 1 to allow yesterday's rate (e.g. while debugging); set to 7 to
# disable the gate entirely.
FOREX_MAX_STALE_DAYS: int = int(os.environ.get("FOREX_MAX_STALE_DAYS", "0"))

# GDPNOW_MAX_STALE_DAYS — Maximum age (calendar days) of the Atlanta Fed GDPNow
# estimate before it is considered stale and dropped.  GDPNow updates ~2-3x per
# week during the quarter (whenever major macro data arrives).  Between updates
# the reading can be 1-3 days old, which is still forward-looking.  The main
# risk is the inter-quarter gap after the BEA advance estimate releases but
# before GDPNow starts updating for the next quarter.
# Default: 5 (covers weekends + minor gaps; drops stale inter-quarter readings).
# Set to 30 to disable and always use the most recent available estimate.
GDPNOW_MAX_STALE_DAYS: int = int(os.environ.get("GDPNOW_MAX_STALE_DAYS", "5"))

# FOREX_CLOSE_HOURS — Gate for forex daily-fix markets (KXEURUSD, KXUSDJPY).
# Only surface opportunities when within this many hours of the market's
# close_time.  The ECB fixing and Kalshi resolution both land at ~10 AM ET;
# the rate is not predictive of the fixing until it IS the fixing.
# Set to 0 to disable.  Default: 1.0
FOREX_CLOSE_HOURS: float = float(os.environ.get("FOREX_CLOSE_HOURS", "2.0"))

# CRYPTO_DAILY_CLOSE_HOURS — Gate for daily-close crypto markets (KXBTCD).
# Binance returns the live intraday spot price, which can be several percent
# away from the 5 PM ET settlement price many hours before close.  Only
# surface KXBTCD opportunities when within this many hours of the market's
# close_time.  Set to 0 to disable.  Default: 2.0
CRYPTO_DAILY_CLOSE_HOURS: float = float(os.environ.get("CRYPTO_DAILY_CLOSE_HOURS", "6.0"))

# CONTRARIAN_MAX_ENTRY_CENTS — (Option E) Maximum cost (in cents) to enter a
#   numeric position.  For a YES trade this is yes_ask; for a NO trade it is
#   100 − yes_bid.  Only trades where the market currently disagrees with the
#   data signal (cheap entry) pass this gate.  Trades where the market already
#   agrees (expensive entry) offer no information edge.
#   Set to 0 to disable (allow any price).
CONTRARIAN_MAX_ENTRY_CENTS: int = int(os.environ.get("CONTRARIAN_MAX_ENTRY_CENTS", "65"))

# MARKET_MIN_PRICE_CENTS — Mirror of CONTRARIAN_MAX_ENTRY_CENTS.
#   The contrarian gate blocks expensive entries (market agrees → priced in).
#   This gate blocks near-zero entries (market is near-certain against us).
#   When a market prices our side at <N cents the collective orderbook is
#   telling us our signal is wrong — most often caused by the data source
#   monitoring a different station/price than the one Kalshi uses for
#   settlement (e.g. NOAA observed max from wrong station giving p=1.0 while
#   the market sits at 1¢ YES because the official resolution station is
#   actually far below the strike).
#   YES trade: blocked when yes_ask  < MARKET_MIN_PRICE_CENTS.
#   NO  trade: blocked when 100−yes_bid < MARKET_MIN_PRICE_CENTS.
#   Default: 10. Set to 0 to disable.
#   Data: trades with our-side cost ≥8¢ showed 46–54% win rates; trades at
#   ≤5¢ effective cost showed 7–15% win rates regardless of model confidence.
#   Raised from 8→10: a YES ask of 8¢ on a near-certain noaa_observed signal
#   is almost always a station mismatch (different NWS resolution station),
#   not a real arbitrage opportunity.
MARKET_MIN_PRICE_CENTS: int = int(os.environ.get("MARKET_MIN_PRICE_CENTS", "10"))

# Maximum yes_ask for noaa_observed YES direction=over entries.
# When the observed temperature ALREADY exceeds the strike (direction=over), the
# market should agree (high yes price).  Entering at a very high price has
# terrible risk:reward: at 90¢ entry you risk 90¢ to gain at most 10¢.
# If the observation is wrong (station mismatch, stale data) the loss is the
# full entry price.  Canonical data: trades #194 and #195 entered at 100¢ YES
# and both settled at 0¢ (total loss of 200¢ combined) because the observation
# was from the wrong NWS station.
# noaa_observed is exempt from CONTRARIAN_MAX_ENTRY_CENTS (direction=under
# locked-YES signals at 93¢ are still valid), but for direction=over the
# market-agreement risk is too high above this ceiling.
# Default: 85¢.  Set to 0 to disable.
NOAA_OBS_OVER_MAX_ENTRY_CENTS: int = int(
    os.environ.get("NOAA_OBS_OVER_MAX_ENTRY_CENTS", "85")
)

# Maximum days until market close to consider for text matching.
# Markets resolving further out than this are excluded — today's news
# rarely provides actionable alpha on a market that settles in 2 years.
MARKET_MAX_DAYS_OUT: int = int(os.environ.get("MARKET_MAX_DAYS_OUT", "30"))

# Minimum minutes until market close. Markets closing sooner than this
# (including already-closed markets still listed as "open") are dropped.
# Prevents betting on markets that have already resolved or are in their
# final minutes where information edge can't be acted on in time.
MARKET_MIN_MINUTES_TO_CLOSE: int = int(os.environ.get("MARKET_MIN_MINUTES_TO_CLOSE", "30"))

# Liquidity filters applied after orderbook enrichment.
# Set LIQUIDITY_MAX_SPREAD=0 to disable spread filtering.
LIQUIDITY_MAX_SPREAD: int = int(os.environ.get("LIQUIDITY_MAX_SPREAD", "10"))
LIQUIDITY_MIN_VOLUME: int = int(os.environ.get("LIQUIDITY_MIN_VOLUME", "0"))

# Cross-cycle display cooldown. If the same (ticker, signal) pair was surfaced
# within this many minutes, it is suppressed from stdout (but a fresh signal
# type on the same ticker will still surface). Set to 0 to disable suppression.
OPPORTUNITY_COOLDOWN_MINUTES: int = int(os.environ.get("OPPORTUNITY_COOLDOWN_MINUTES", "60"))

# Post-exit re-entry cooldown.
# After a stop_loss or trailing_stop exit on a ticker, block re-entry into
# that same ticker for this many minutes.  Prevents cascading losses from
# repeatedly entering a position that has already been stopped out.
# Historical impact: KXHIGHNY-26MAR27-T62 had 4 entries totalling -151¢;
# a 4h cooldown after the first trailing_stop would have limited it to -9¢.
# Profit-take exits are NOT blocked — a market that hit its profit target
# may still have actionable upside worth re-entering.
# Set to 0 to disable.
EXIT_REENTRY_COOLDOWN_MINUTES: int = int(os.environ.get("EXIT_REENTRY_COOLDOWN_MINUTES", "120"))

# Minimum absolute net position (in contracts) at which an opportunity in that
# market is suppressed from display. Prevents re-alerting on a market where we
# already hold a meaningful position. Set to 0 to always surface all markets
# regardless of existing exposure.
POSITION_SKIP_CONTRACTS: int = int(os.environ.get("POSITION_SKIP_CONTRACTS", "5"))

# Manifold consensus gate.  Manifold-only signals with divergence above this
# threshold are blocked unless Polymarket or Metaculus independently agrees on
# the same Kalshi ticker.  Real-money platforms rarely sustain 50%+ divergences
# — if only Manifold shows it, it is almost certainly stale/frozen data.
# Set to 0 to disable.
MANI_MAX_SOLO_DIVERGENCE: float = float(os.environ.get("MANI_MAX_SOLO_DIVERGENCE", "0.50"))

# Manifold hard cap.  Manifold opportunities with divergence above this threshold
# are ALWAYS blocked, even if Polymarket or Metaculus corroborates.  A 40%+
# divergence (Manifold 85%, Kalshi 45%) is almost certainly a stale Manifold
# market or a matching error — legitimate Kalshi mispricings rarely exceed 30–35pp.
# Set to 0 to disable.
MANI_HARD_CAP_DIVERGENCE: float = float(os.environ.get("MANI_HARD_CAP_DIVERGENCE", "0.40"))

# Maximum number of simultaneously open temperature (KXHIGH*) positions.
# When this many temperature positions are already open, new temperature numeric
# opportunities are skipped for the cycle.  Prevents a single weather system
# from flooding the portfolio with 15–20 correlated positions that all win or
# lose together.  Set to 0 to disable.  Default: 8.
TEMP_MAX_CONCURRENT_POSITIONS: int = int(os.environ.get("TEMP_MAX_CONCURRENT_POSITIONS", "8"))


# ---------------------------------------------------------------------------
# Market cache — populated by _get_markets(), shared across poll cycles
# ---------------------------------------------------------------------------

_numeric_cache: list[dict] = []
_numeric_cache_ts: float = 0.0
_general_cache: list[dict] = []
_general_cache_ts: float = 0.0


async def _get_markets(session: aiohttp.ClientSession) -> list[dict]:
    """Return the combined market list, with split-interval refresh.

    Numeric series (KXHIGH, KXBTCD, etc.) are refreshed every
    NUMERIC_MARKET_REFRESH_INTERVAL seconds (~5s fetch cost) so the bot
    always trades on prices that are at most ~90s old.

    General text/keyword markets are refreshed every
    GENERAL_MARKET_REFRESH_INTERVAL seconds (~20s fetch cost) since they
    move slowly and only matter for Federal Register / RSS matching.
    """
    global _numeric_cache, _numeric_cache_ts, _general_cache, _general_cache_ts

    now = time.monotonic()

    if now - _numeric_cache_ts > NUMERIC_MARKET_REFRESH_INTERVAL:
        logging.info("Numeric market cache stale — refreshing %d series …", len(NUMERIC_SERIES))
        fresh = await fetch_markets_by_series(session, status="open")
        if fresh:
            _numeric_cache = fresh
            _numeric_cache_ts = now
        elif not _numeric_cache:
            logging.warning("Numeric market sync returned empty.")

    if now - _general_cache_ts > GENERAL_MARKET_REFRESH_INTERVAL:
        logging.info(
            "General market cache stale — fetching %d text series …", len(TEXT_SERIES)
        )
        # Kalshi's default pagination returns thousands of KXMVE esports markets
        # before any politics/economics/entertainment markets appear.  Fetching
        # by series_ticker directly bypasses this and is far faster.
        fresh_general = await fetch_markets_by_series(
            session, series_tickers=TEXT_SERIES, status="open", limit_per_series=200
        )
        logging.info("General market cache: %d text-series markets fetched.", len(fresh_general))
        if fresh_general:
            _general_cache = fresh_general
            _general_cache_ts = now
            scan_unknown_series(fresh_general)
        elif not _general_cache:
            logging.warning("General market sync returned empty.")

    # Merge: numeric entries take precedence (fresher prices)
    seen = {m["ticker"] for m in _numeric_cache}
    return _numeric_cache + [m for m in _general_cache if m["ticker"] not in seen]


# ---------------------------------------------------------------------------
# Market filtering helper
# ---------------------------------------------------------------------------

def _select_markets(
    markets: list[dict],
    include_prefixes: tuple[str, ...],
    exclude_prefixes: tuple[str, ...],
) -> list[dict]:
    """Return markets matching the include/exclude ticker-prefix rules."""
    if include_prefixes:
        return [m for m in markets if any(m.get("ticker", "").startswith(p) for p in include_prefixes)]
    if exclude_prefixes:
        return [m for m in markets if not any(m.get("ticker", "").startswith(p) for p in exclude_prefixes)]
    return markets


def _filter_by_close_time(
    markets: list[dict], max_days: int, min_minutes: int = 0
) -> list[dict]:
    """Drop markets outside the [min_minutes, max_days] close-time window.

    Markets are dropped if their close_time is:
      - More than max_days from now (too far out; no alpha from today's news), OR
      - Less than min_minutes from now (too close; includes already-closed markets
        still appearing as "open" while awaiting official resolution).

    Markets with a missing or unparseable close_time are kept to avoid
    accidentally filtering valid markets during API inconsistencies.
    """
    now = datetime.now(timezone.utc)
    max_cutoff = now + timedelta(days=max_days) if max_days > 0 else None
    min_cutoff = now + timedelta(minutes=min_minutes) if min_minutes > 0 else None
    result = []
    for m in markets:
        ct = m.get("close_time") or m.get("expiration_time")
        if not ct:
            result.append(m)
            continue
        try:
            close_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            if max_cutoff is not None and close_dt > max_cutoff:
                continue
            if min_cutoff is not None and close_dt < min_cutoff:
                continue
            result.append(m)
        except (ValueError, AttributeError):
            result.append(m)
    return result


# ---------------------------------------------------------------------------
# Days-to-close helper  (used by scoring and display)
# ---------------------------------------------------------------------------

def _days_to_close(ticker: str, ticker_detail: dict[str, dict]) -> float:
    """Return fractional days until market close for a ticker.

    Reads close_time (or expiration_time as fallback) from the live market
    detail dict already fetched during orderbook enrichment.

    Returns float("inf") if the detail is missing or the date is unparseable,
    so callers can treat it as "far future" without special-casing.
    """
    detail = ticker_detail.get(ticker)
    if not detail:
        return float("inf")
    ct = detail.get("close_time") or detail.get("expiration_time")
    if not ct:
        return float("inf")
    try:
        close_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        delta = close_dt - datetime.now(timezone.utc)
        return max(0.0, delta.total_seconds() / 86_400)
    except (ValueError, AttributeError):
        return float("inf")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SEP  = "-" * 64
_WIDE = "=" * 64


def _fmt_liquidity(detail: dict | None) -> str:
    """Format bid/ask/spread/volume from a market detail dict."""
    if not detail:
        return "  Liquidity: (unavailable)"
    bid  = detail.get("yes_bid")
    ask  = detail.get("yes_ask")
    vol  = detail.get("volume")
    if bid is not None and ask is not None:
        spread = ask - bid
        price_str = f"{bid}¢ bid / {ask}¢ ask  (spread: {spread}¢)"
    else:
        last = detail.get("last_price", "N/A")
        price_str = f"{last}¢ last  (no bid/ask)"
    vol_str = f"  |  Volume: {vol:,}" if vol is not None else ""
    return f"  Liquidity: {price_str}{vol_str}"


def _fmt_position(net_position: int) -> str:
    """Format an existing position for inline display (returns empty string if flat)."""
    if net_position == 0:
        return ""
    side = "YES" if net_position > 0 else "NO"
    return f"  Position: {abs(net_position)} {side} contracts held"


def _print_text_opportunity(
    idx: int, opp: Opportunity, detail: dict | None, score: float,
    existing_position: int = 0,
) -> None:
    print(_SEP)
    alts = f"  (+{opp.n_alternatives} similar markets)" if opp.n_alternatives else ""
    print(f"  [TEXT #{idx}  score={score:.2f}  display-only]  {opp.topic}  |  {opp.market_ticker}{alts}")
    print(f"  Market:   {opp.market_title}")
    print(_fmt_liquidity(detail) + f"  |  Source: {opp.source}")
    pos_line = _fmt_position(existing_position)
    if pos_line:
        print(pos_line)
    print(f"  Article:  {opp.doc_title}")
    print(f"  URL:      {opp.doc_url}")


def _print_numeric_opportunity(
    idx: int, opp: NumericOpportunity, detail: dict | None, score: float,
    existing_position: int = 0,
) -> None:
    if opp.direction == "between":
        strike_str = f"{opp.strike_lo}–{opp.strike_hi}"
    elif opp.strike is not None:
        strike_str = str(opp.strike)
    else:
        strike_str = "N/A"

    print(_SEP)
    print(f"  [DATA #{idx}  score={score:.2f}]  {opp.metric}  |  {opp.market_ticker}")
    print(f"  Market:   {opp.market_title}")
    print(f"  Live:     {opp.data_value}{opp.unit}  (as of {opp.as_of})")
    print(
        f"  Strike:   {opp.direction.upper()} {strike_str}"
        f"  →  implied {opp.implied_outcome}  (edge {opp.edge:.3f})"
    )
    print(_fmt_liquidity(detail) + f"  |  Source: {opp.source}")
    pos_line = _fmt_position(existing_position)
    if pos_line:
        print(pos_line)


def _print_poly_opportunity(
    idx: int, opp: PolyOpportunity, detail: dict | None, score: float,
    existing_position: int = 0,
) -> None:
    _SOURCE_LABELS = {
        "polymarket": "Polymarket",
        "metaculus":  "Metaculus",
        "manifold":   "Manifold",
        "predictit":  "PredictIt",
    }
    source_label = _SOURCE_LABELS.get(opp.source, opp.source.capitalize())
    # Liquidity label varies by platform
    if opp.source == "metaculus":
        liq_str = f"{opp.poly_liquidity:.0f} forecasters"
    elif opp.source == "predictit":
        liq_str = f"vol=${opp.poly_liquidity:,.0f}"
    else:
        liq_str = f"liq=${opp.poly_liquidity:,.0f}"
    print(_SEP)
    print(f"  [EXT #{idx}  score={score:.2f}  src={source_label}]  divergence={opp.divergence:.2%}  |  {opp.kalshi_ticker}")
    print(f"  Kalshi:   {opp.kalshi_title}")
    print(f"  Kalshi p: {opp.kalshi_mid:.0f}¢  →  side={opp.implied_side.upper()}")
    print(f"  {source_label} ({opp.poly_p_yes:.1%}  {liq_str}):")
    print(f"    {opp.poly_question}")
    print(f"  Match:    {opp.match_score:.2f}  |  " + _fmt_liquidity(detail))
    pos_line = _fmt_position(existing_position)
    if pos_line:
        print(pos_line)


# ---------------------------------------------------------------------------
# Helpers: collect and deduplicate text documents from a single source
# ---------------------------------------------------------------------------

def _drain_text_source(
    label: str,
    docs: list[dict],
    seen: SeenDocuments,
    id_field: str,
    source_tag: str,
    accumulator: list[dict],
) -> None:
    """Filter unseen docs from one source and append to accumulator."""
    new = seen.filter_new(docs, id_field=id_field)
    skipped = len(docs) - len(new)
    if new:
        logging.info("%s: %d new item(s) (%d already seen).", label, len(new), skipped)
    # Tag each doc so the matcher can report which feed it came from
    for d in new:
        d.setdefault("_source", source_tag)
    accumulator.extend(new)


# ---------------------------------------------------------------------------
# Single poll cycle
# ---------------------------------------------------------------------------

_ET = ZoneInfo("America/New_York")

# City → local timezone, derived from nws_climo.CLIMO_LOCATIONS so there is a
# single source of truth for the metric → timezone mapping.
# Covers both temp_high_* and temp_low_* (same cities, same timezones).
_CITY_TZ: dict[str, ZoneInfo] = {
    metric: tz for metric, (_, _, tz) in nws_climo.CLIMO_LOCATIONS.items()
}
_CITY_TZ.update({
    metric: tz for metric, (_, _, tz) in nws_climo.LOW_CLIMO_LOCATIONS.items()
})
# Minimum local hour before a noaa_observed YES trade is allowed.
# Daily temp max is typically established 1–5 PM local; before 1 PM the
# "observed" max is still a morning partial reading — not a confirmed high.
NOAA_OBS_YES_MIN_LOCAL_HOUR: int = int(os.environ.get("NOAA_OBS_YES_MIN_LOCAL_HOUR", "13"))

# Stricter minimum local hour for "below threshold" YES signals (direction=under).
# Observed 66°F against a 76°F strike is only a locked signal AFTER the daily
# temperature peak has passed — at 1 PM there is still 3–4 hours of potential
# warming.  Daily highs are typically established by 4 PM local; requiring
# local hour >= 16 ensures we're already in the cooling-off phase before
# committing to "temp will stay below strike".
NOAA_OBS_PEAK_PAST_LOCAL_HOUR: int = int(os.environ.get("NOAA_OBS_PEAK_PAST_LOCAL_HOUR", "16"))
NOAA_OBS_PEAK_PAST_LOCAL_MINUTE: int = int(os.environ.get("NOAA_OBS_PEAK_PAST_LOCAL_MINUTE", "30"))

# Morning gate for low-temperature observed YES signals (direction=over on
# temp_low_* metrics).  The running daily minimum resets at local midnight and
# at midnight equals the current temperature — not the overnight low, which
# typically occurs at 4–6 AM.  Only allow noaa_observed YES trades on low-temp
# markets after this local hour, by which point the overnight trough has passed
# and the running min is a reliable confirmed floor.
# Default: 05:00 local (typically past the overnight low trough).
# Override via NOAA_OBS_LOW_PAST_LOCAL_HOUR / _MINUTE env vars.
NOAA_OBS_LOW_PAST_LOCAL_HOUR: int = int(os.environ.get("NOAA_OBS_LOW_PAST_LOCAL_HOUR", "5"))
NOAA_OBS_LOW_PAST_LOCAL_MINUTE: int = int(os.environ.get("NOAA_OBS_LOW_PAST_LOCAL_MINUTE", "0"))

# Forecast floor gate for noaa_observed temp_low YES signals.
# After the morning trough, temps rise during the day then drop again at sunset.
# The NWS daily minimum spans the full calendar day — evening cooling can push
# the final daily min below the strike even if the morning min was above it.
# Guard: if tonight's NWS forecast low (source="noaa", day1) is below
# (strike + TEMP_LOW_YES_FORECAST_BUFFER_F), suppress the YES signal.
# Buffer of 0.5°F mirrors NWS integer rounding used in band arb: a forecast
# of 47°F means the official low rounds to 47 (≥46.5) → safe above a 46°F
# strike; a forecast of 46°F could round to 46 = strike → YES won't resolve.
# Set TEMP_LOW_YES_FORECAST_BUFFER_F=0 for strict < strike comparison.
TEMP_LOW_YES_FORECAST_BUFFER_F: float = float(
    os.environ.get("TEMP_LOW_YES_FORECAST_BUFFER_F", "0.5")
)
# If True (default), suppress YES when no tonight forecast is available.
# Set to 'false' to fall back to pre-fix behavior (allow without forecast).
TEMP_LOW_YES_REQUIRE_FORECAST: bool = (
    os.environ.get("TEMP_LOW_YES_REQUIRE_FORECAST", "true").lower() == "true"
)
# Only enforce the TEMP_LOW_YES_REQUIRE_FORECAST gate at or after this local
# hour.  Before this hour the signal is about the confirmed overnight/morning
# trough, which is less sensitive to evening cooldown risk.  After this hour
# the temperature is likely still rising (or near peak) and an evening dip
# could invalidate the YES — so the forecast is required.
# Default 12 (noon local): require forecast in the afternoon when cooldown matters.
TEMP_LOW_YES_REQUIRE_FORECAST_AFTER_LOCAL_HOUR: int = int(
    os.environ.get("TEMP_LOW_YES_REQUIRE_FORECAST_AFTER_LOCAL_HOUR", "12")
)

# Regex for the date segment in KXHIGH tickers: e.g. "26MAR10" in KXHIGHAUS-26MAR10-T75
_TICKER_DATE_RE = re.compile(r"^(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})$")
_MONTH_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _ticker_date(ticker: str) -> date | None:
    """Parse the market date from a KXHIGH ticker segment (e.g. KXHIGHAUS-26MAR10-T75 → date(2026,3,10))."""
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    m = _TICKER_DATE_RE.match(parts[1])
    if not m:
        return None
    try:
        return date(2000 + int(m.group(1)), _MONTH_NUM[m.group(2)], int(m.group(3)))
    except (ValueError, KeyError):
        return None


def _compute_trajectory_projections(
    metar_points: list[DataPoint],
    now_utc: datetime,
) -> list[DataPoint]:
    """Compute temperature trajectory DataPoints from METAR time-series.

    Uses a parabolic diurnal model to project the expected daily peak, accounting
    for the natural deceleration of warming as temperatures approach their afternoon
    maximum.  Only fires during TRAJ_START_LOCAL_HOUR..TRAJ_END_LOCAL_HOUR local
    time when the city is actively warming (slope > TRAJ_MIN_SLOPE_FPH) and the
    projected rise exceeds TRAJ_MIN_EDGE_F.

    Model: T(t) = T_min + A*(2u - u²) where u = phase through warming phase.
    Remaining rise: slope_now × hours_to_peak × (1 - u_now) / 2.
    """
    points: list[DataPoint] = []
    for dp in metar_points:
        if dp.source != "metar":
            continue
        obs_series = dp.metadata.get("obs_series")
        if not obs_series or len(obs_series) < TRAJ_MIN_OBS:
            continue

        city_entry = noaa.CITIES.get(dp.metric)
        if city_entry is None:
            continue
        city_tz = city_entry[3]

        local_now = now_utc.astimezone(city_tz)
        if not (TRAJ_START_LOCAL_HOUR <= local_now.hour < TRAJ_END_LOCAL_HOUR):
            continue

        local_hour_frac = local_now.hour + local_now.minute / 60.0

        # Use only observations from the last TRAJ_LOOKBACK_HOURS
        cutoff = now_utc.timestamp() - TRAJ_LOOKBACK_HOURS * 3600
        recent = [(t, temp) for t, temp in obs_series if t >= cutoff]
        if len(recent) < TRAJ_MIN_OBS:
            continue

        # Linear regression to compute warming slope (°F/hour)
        n = len(recent)
        ts = [t for t, _ in recent]
        temps = [temp for _, temp in recent]
        t_mean = sum(ts) / n
        temp_mean = sum(temps) / n
        num = sum((ts[i] - t_mean) * (temps[i] - temp_mean) for i in range(n))
        den = sum((ts[i] - t_mean) ** 2 for i in range(n))
        if den < 1.0:
            continue
        slope_fph = (num / den) * 3600  # per-second → per-hour
        if slope_fph <= TRAJ_MIN_SLOPE_FPH:
            continue

        current_temp = temps[-1]

        dawn_hour = TRAJ_DAWN_LOCAL_HOUR.get(dp.metric, 6)
        peak_hour = TRAJ_PEAK_LOCAL_HOUR.get(dp.metric, 16)
        warming_duration_h = peak_hour - dawn_hour
        if warming_duration_h <= 0:
            continue

        u_now = max(0.0, min(1.0, (local_hour_frac - dawn_hour) / warming_duration_h))
        hours_to_peak = max(0.0, peak_hour - local_hour_frac)
        if hours_to_peak < TRAJ_MIN_HOURS_TO_PEAK:
            continue

        # Parabolic projected remaining rise
        tod_factor = (1.0 - u_now) / 2.0
        projected_rise = slope_fph * hours_to_peak * tod_factor
        if projected_rise < TRAJ_MIN_EDGE_F:
            continue

        projected_peak = current_temp + projected_rise
        logging.info(
            "Trajectory [%s]: current=%.1f°F  slope=+%.2f°F/h  "
            "phase=%.0f%%  projected_peak=%.1f°F  (+%.1f°F in %.1fh)",
            dp.metric, current_temp, slope_fph,
            u_now * 100, projected_peak, projected_rise, hours_to_peak,
        )

        points.append(DataPoint(
            source="obs_trajectory",
            metric=dp.metric,
            value=projected_peak,
            unit="°F",
            as_of=now_utc.isoformat(),
            metadata={
                "current_temp":   current_temp,
                "slope_fph":      round(slope_fph, 2),
                "u_now":          round(u_now, 3),
                "tod_factor":     round(tod_factor, 3),
                "hours_to_peak":  round(hours_to_peak, 2),
                "projected_rise": round(projected_rise, 2),
                "obs_count":      len(recent),
            },
        ))
    return points


def _filter_weather_opportunities(
    opps: list[NumericOpportunity],
    markets: list[dict],
    hrrr_hourly_highs: dict[str, float] | None = None,
    observed_values: dict[str, float] | None = None,
    fc_low_by_metric: dict[str, float] | None = None,
) -> list[NumericOpportunity]:
    """Apply source-specific quality gates to temperature (KXHIGH) opportunities.

    Three signal classes are handled:

    noaa_observed  (station max recorded since midnight)
        Ground truth — a hard lower bound on the day's high.
        YES: observed > strike → outcome locked; always surfaced (edge ≥ TEMP_OBSERVED_MIN_EDGE).
        NO:  only surfaced within TEMP_OBSERVED_MAX_HOURS of market close.

    nws_alert  (temperature parsed from an active NWS heat/cold warning)
        High-confidence directional signal — NWS issues warnings only when
        its own ensemble is highly confident.  Treated identically to
        noaa_observed: lower edge threshold, never gated by HRRR spread.

    noaa / owm / open_meteo  (raw model forecast)
        Require edge >= TEMP_FORECAST_MIN_EDGE (7°F ≈ 2σ of day-1 MAE).
        Additionally gated by HRRR spread for same-day forecasts: if
        |daily_forecast − hourly_hrrr| >= HRRR_MAX_SPREAD_F the atmosphere
        is convectively uncertain and the forecast is unreliable.
        Extended-day sources (noaa_day2+, open_meteo day1+) skip the HRRR
        gate since today's HRRR data is irrelevant to future days.

    Morning observation gate (MORNING_OBS_GAP_F)
        For raw forecast YES signals, if the current noaa_observed max (since
        midnight) is more than MORNING_OBS_GAP_F below the strike, the trade
        is blocked: the observed reality is too far from the forecast for it
        to be credible.

    All non-temperature metrics pass through unchanged.
    """
    now = datetime.now(timezone.utc)

    # Build close_time index from the market list for quick lookup.
    close_dt_by_ticker: dict[str, datetime | None] = {}
    for m in markets:
        ticker = m.get("ticker", "")
        ct = m.get("close_time") or m.get("expiration_time")
        if ct:
            try:
                close_dt_by_ticker[ticker] = datetime.fromisoformat(
                    ct.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                close_dt_by_ticker[ticker] = None
        else:
            close_dt_by_ticker[ticker] = None

    # Obs-consensus pre-pass: for each (metric, strike), count how many
    # confirmed-observation sources (noaa_observed, nws_climo) already show
    # YES *and* pass the afternoon hour gate.  Used below to gate YES trades.
    _OBS_CONFIRMED = frozenset({"noaa_observed", "nws_climo", "metar"})
    obs_yes_sources: dict[tuple[str, float | None], set[str]] = {}
    if OBS_CONSENSUS_MIN > 1:
        for _o in opps:
            if (
                _o.source in _OBS_CONFIRMED
                and _o.metric.startswith(("temp_high", "temp_low"))
                and _o.implied_outcome == "YES"
            ):
                # Apply the same afternoon hour gate used in the main loop.
                _city_tz = _CITY_TZ.get(_o.metric)
                if _city_tz is not None:
                    if now.astimezone(_city_tz).hour < NOAA_OBS_YES_MIN_LOCAL_HOUR:
                        continue
                obs_yes_sources.setdefault((_o.metric, _o.strike), set()).add(_o.source)

    result: list[NumericOpportunity] = []
    for opp in opps:
        # Non-temperature metrics are unaffected.
        if not opp.metric.startswith(("temp_high", "temp_low")):
            result.append(opp)
            continue

        # Date guard: NOAA DataPoints carry an `as_of` timestamp in UTC.
        # After midnight UTC (~7 PM ET), the UTC date is already the next
        # calendar day while the ET date is still "today".  Without this
        # check, today's observed temperature would match tomorrow's Kalshi
        # market, producing spurious near-certain signals on the wrong day.
        #
        # Rule: the market's date (parsed from the ticker, e.g. 26MAR10 →
        # 2026-03-10) must equal the ET date of the DataPoint's as_of
        # timestamp.  Opportunities that span a date boundary are dropped.
        if opp.as_of:
            try:
                as_of_utc = datetime.fromisoformat(opp.as_of.replace("Z", "+00:00"))
                as_of_et_date = as_of_utc.astimezone(_ET).date()
                market_date = _ticker_date(opp.market_ticker)
                if market_date is not None and market_date != as_of_et_date:
                    logging.debug(
                        "NOAA date guard: dropped %s — market date %s ≠ data ET date %s",
                        opp.market_ticker, market_date, as_of_et_date,
                    )
                    continue
            except (ValueError, AttributeError):
                pass  # malformed as_of: allow through, worse case is a bad signal
        elif opp.source in ("noaa_observed", "metar", "nws_climo"):
            # as_of is absent — fall back to a wall-clock city-local date check so
            # observed DataPoints with missing timestamps still get date-guarded.
            # Without this, a None as_of bypasses the crossover-bug guard entirely.
            _city_tz_dg = _CITY_TZ.get(opp.metric)
            if _city_tz_dg is not None:
                _local_date_dg = now.astimezone(_city_tz_dg).date()
                _mkt_date_dg = _ticker_date(opp.market_ticker)
                if _mkt_date_dg is not None and _mkt_date_dg != _local_date_dg:
                    logging.debug(
                        "NOAA date guard (no as_of): dropped %s — market %s ≠ local %s",
                        opp.market_ticker, _mkt_date_dg, _local_date_dg,
                    )
                    continue


        if opp.source in ("noaa_observed", "metar", "nws_climo", "nws_alert"):
            # High-confidence signals: ground truth (observed/climo) or NWS
            # warning.  Never gated by HRRR spread; direction-split edge threshold.
            if opp.direction == "over":
                min_edge = TEMP_OBSERVED_MIN_EDGE_OVER
            elif opp.direction == "between":
                min_edge = TEMP_OBSERVED_MIN_EDGE_BETWEEN
            else:
                min_edge = TEMP_OBSERVED_MIN_EDGE_UNDER
            if opp.edge < min_edge:
                continue
            # noaa_observed max-edge cap: very high edges signal a faulty sensor
            # reading (station reporting stale/wrong temp), not a genuine gap.
            if opp.source == "noaa_observed" and opp.edge > TEMP_OBSERVED_MAX_EDGE:
                logging.info(
                    "noaa_observed max-edge cap: edge=%.1f°F > %.1f°F on %s — "
                    "likely sensor error; suppressed",
                    opp.edge, TEMP_OBSERVED_MAX_EDGE, opp.market_ticker,
                )
                continue

            if opp.implied_outcome == "YES":
                # Time-of-day gate for confirmed-observation YES signals.
                #
                # direction=over  (observed already EXCEEDS strike): local-date guard only.
                #   The daily max is a running maximum — once above the strike it
                #   can only stay above or climb further.  Outcome is locked the
                #   moment it is observed.  The 5°F edge threshold absorbs any
                #   station-mismatch risk.
                #   BUT: _fetch_observed_max_today uses midnight UTC as its window
                #   start.  For CDT cities (UTC-5) midnight UTC = 7 PM local, so at
                #   04:00 UTC the window captures *yesterday evening's* readings.
                #   The ET date guard rolls to the next day at midnight ET (04:00 UTC),
                #   letting a March-25-evening observed high slip through as a March-26
                #   YES signal.  The local-date guard below closes this gap for both
                #   directions.
                #
                # direction=between YES (observed IS inside the range): 4:30 PM local.
                #   Lower bound is confirmed (observed ≥ strike_lo), but before 4:30 PM
                #   the temperature could still rise above strike_hi and make YES lose.
                #   Same afternoon gate as direction=under to ensure the day's peak
                #   has passed before committing.
                #
                # direction=under (observed still BELOW strike): 4:30 PM local.
                #   The observed max is a lower bound, not the confirmed daily
                #   high.  Before 4:30 PM there are still hours of potential
                #   warming; this gate ensures the peak has passed and the
                #   temperature is in the cooling-off phase.
                if opp.source in _OBS_CONFIRMED and opp.direction == "over":
                    city_tz = _CITY_TZ.get(opp.metric)
                    if city_tz is not None:
                        local_dt = now.astimezone(city_tz)
                        _mkt_date = _ticker_date(opp.market_ticker)
                        if _mkt_date is not None and _mkt_date != local_dt.date():
                            logging.info(
                                "Date guard (local): suppressed %s direction=over YES %s"
                                " — market date %s ≠ city local date %s",
                                opp.source, opp.market_ticker,
                                _mkt_date, local_dt.date(),
                            )
                            continue

                        # Morning gate for low-temp observed YES signals.
                        # The running daily min resets at local midnight and at
                        # that point equals the current temp — not the overnight
                        # low, which typically occurs at 4–6 AM.  Block until
                        # NOAA_OBS_LOW_PAST_LOCAL_HOUR (default 05:00) by which
                        # point the trough has passed and the running min is a
                        # confirmed floor.
                        if opp.metric.startswith("temp_low_"):
                            local_mins = local_dt.hour * 60 + local_dt.minute
                            min_mins = (NOAA_OBS_LOW_PAST_LOCAL_HOUR * 60
                                        + NOAA_OBS_LOW_PAST_LOCAL_MINUTE)
                            if local_mins < min_mins:
                                logging.info(
                                    "Morning gate: suppressed %s low-temp YES %s"
                                    " — local time %02d:%02d < %02d:%02d"
                                    " (overnight low not yet confirmed)",
                                    opp.source, opp.market_ticker,
                                    local_dt.hour, local_dt.minute,
                                    min_mins // 60, min_mins % 60,
                                )
                                continue

                            # Forecast floor gate: suppress YES if tonight's NWS
                            # forecast low is within striking distance of the
                            # strike.  Evening cooling after sunset can push the
                            # final daily min below the morning trough — the
                            # morning gate passes but the signal is still unsafe.
                            # 0.5°F buffer: NWS rounds to integer; need forecast
                            # ≥ strike+1 to guarantee official low > strike.
                            if fc_low_by_metric is not None and opp.strike is not None:
                                _fc_low = fc_low_by_metric.get(opp.metric)
                                if _fc_low is None and TEMP_LOW_YES_REQUIRE_FORECAST:
                                    _past_gate_hour = (
                                        local_dt.hour
                                        >= TEMP_LOW_YES_REQUIRE_FORECAST_AFTER_LOCAL_HOUR
                                    )
                                    if _past_gate_hour:
                                        logging.info(
                                            "Forecast floor gate: suppressed noaa_observed YES %s"
                                            " — no tonight forecast after %02d:00 local"
                                            " (TEMP_LOW_YES_REQUIRE_FORECAST=true)",
                                            opp.market_ticker,
                                            TEMP_LOW_YES_REQUIRE_FORECAST_AFTER_LOCAL_HOUR,
                                        )
                                        continue
                                    # Before the gate hour: morning trough is the primary
                                    # signal; allow through without forecast.
                                    logging.debug(
                                        "Forecast floor gate: allowing noaa_observed YES %s"
                                        " without forecast — before %02d:00 local (%02d:%02d)",
                                        opp.market_ticker,
                                        TEMP_LOW_YES_REQUIRE_FORECAST_AFTER_LOCAL_HOUR,
                                        local_dt.hour, local_dt.minute,
                                    )
                                if (
                                    _fc_low is not None
                                    and _fc_low < opp.strike + TEMP_LOW_YES_FORECAST_BUFFER_F
                                ):
                                    logging.info(
                                        "Forecast floor gate: suppressed noaa_observed YES %s"
                                        " — tonight forecast %.1f°F < strike+buf %.1f°F"
                                        " (strike=%.1f°F, buf=%.1f°F)",
                                        opp.market_ticker, _fc_low,
                                        opp.strike + TEMP_LOW_YES_FORECAST_BUFFER_F,
                                        opp.strike, TEMP_LOW_YES_FORECAST_BUFFER_F,
                                    )
                                    continue

                if opp.source in _OBS_CONFIRMED and opp.direction in ("under", "between"):
                    city_tz = _CITY_TZ.get(opp.metric)
                    if city_tz is not None:
                        local_dt = now.astimezone(city_tz)
                        local_mins = local_dt.hour * 60 + local_dt.minute
                        min_mins = (NOAA_OBS_PEAK_PAST_LOCAL_HOUR * 60
                                    + NOAA_OBS_PEAK_PAST_LOCAL_MINUTE)

                        # Local-date guard: after midnight local time an observed
                        # reading is the overnight floor, not a confirmed daily high.
                        # The market must be for *today* in the city's timezone —
                        # if it's already past midnight locally the date guard at the
                        # top of this loop uses the ET date which can roll ahead of
                        # the city's local date, letting tomorrow's market slip through.
                        _mkt_date = _ticker_date(opp.market_ticker)
                        if _mkt_date is not None and _mkt_date != local_dt.date():
                            logging.info(
                                "Date guard (local): suppressed %s direction=%s YES %s"
                                " — market date %s ≠ city local date %s",
                                opp.source, opp.direction, opp.market_ticker,
                                _mkt_date, local_dt.date(),
                            )
                            continue

                        if local_mins < min_mins:
                            logging.info(
                                "Afternoon gate: suppressed %s YES %s"
                                " — local time %02d:%02d < %02d:%02d"
                                " (peak not yet confirmed)",
                                opp.source, opp.market_ticker,
                                local_dt.hour, local_dt.minute,
                                min_mins // 60, min_mins % 60,
                            )
                            continue

                        # After 4:30 PM the daily max is established — the upper
                        # bound of a "between" band is now locked.  Flag peak_past
                        # so score_p_yes uses a one-sided CDF (lower-boundary risk
                        # only) and the edge scorer measures clearance from
                        # strike_lo only, not the min of both distances.
                        if opp.direction == "between":
                            opp.peak_past = True

                    # Obs-consensus gate: require OBS_CONSENSUS_MIN distinct
                    # confirmed-observation sources to both show YES before
                    # trading.  Station mismatches appear in a single feed —
                    # requiring 2 sources eliminates most false positives.
                    if OBS_CONSENSUS_MIN > 1:
                        confirming = obs_yes_sources.get((opp.metric, opp.strike), set())
                        if len(confirming) < OBS_CONSENSUS_MIN:
                            logging.info(
                                "Obs consensus gate: suppressed %s YES %s"
                                " — %d/%d obs sources confirm"
                                " (have: %s, need %d)",
                                opp.source, opp.market_ticker,
                                len(confirming), OBS_CONSENSUS_MIN,
                                confirming or "{none}", OBS_CONSENSUS_MIN,
                            )
                            continue

                    if opp.direction == "between" and getattr(opp, "peak_past", False):
                        logging.info(
                            "%s LOCKED-BETWEEN-YES: %s  observed %.1f°F inside"
                            " [%.1f, %.1f], peak confirmed past %02d:%02d"
                            "  (edge %.1f°F above lo)",
                            opp.source.upper(), opp.market_ticker, opp.data_value,
                            opp.strike_lo or 0.0, opp.strike_hi or 0.0,
                            min_mins // 60, min_mins % 60, opp.edge,
                        )
                    else:
                        logging.info(
                            "%s LOCKED-YES: %s  observed %.1f°F already exceeds strike"
                            "  (edge %.1f°F)",
                            opp.source.upper(), opp.market_ticker,
                            opp.data_value, opp.edge,
                        )
                else:
                    logging.info(
                        "NWS ALERT-YES: %s  alert temp %.1f°F above strike"
                        "  (edge %.1f°F)",
                        opp.market_ticker, opp.data_value, opp.edge,
                    )
                result.append(opp)
            else:
                # NO signal.
                #
                # direction=over after 4:30 PM local: the observed max is BELOW
                # the strike and the day's peak is established — this is a
                # locked NO (mirror of the direction=between YES lock).  Flag it
                # with peak_past=True so score_numeric_opportunity() can override
                # the uncertainty sub-score to 1.0 instead of penalising the
                # market's already-low YES price.
                #
                # All other NO signals: only surface within TEMP_OBSERVED_MAX_HOURS
                # of close (market must be nearly over before a non-certain NO is
                # worth acting on).
                if opp.direction == "over":
                    city_tz = _CITY_TZ.get(opp.metric)
                    if city_tz is not None:
                        local_dt = now.astimezone(city_tz)
                        local_mins = local_dt.hour * 60 + local_dt.minute
                        min_mins = (NOAA_OBS_PEAK_PAST_LOCAL_HOUR * 60
                                    + NOAA_OBS_PEAK_PAST_LOCAL_MINUTE)
                        _mkt_date = _ticker_date(opp.market_ticker)
                        if _mkt_date == local_dt.date() and local_mins >= min_mins:
                            opp.peak_past = True
                            logging.info(
                                "%s LOCKED-NO: %s  observed %.1f°F < strike %.1f°F"
                                ", peak confirmed past %02d:%02d  (edge %.1f°F)",
                                opp.source.upper(), opp.market_ticker,
                                opp.data_value, opp.strike or 0.0,
                                min_mins // 60, min_mins % 60, opp.edge,
                            )
                            result.append(opp)
                            continue
                # Standard NO gate: only surface within TEMP_OBSERVED_MAX_HOURS of close.
                close_dt = close_dt_by_ticker.get(opp.market_ticker)
                if close_dt is not None:
                    hours_remaining = (close_dt - now).total_seconds() / 3600
                    if hours_remaining <= TEMP_OBSERVED_MAX_HOURS:
                        result.append(opp)

        else:
            # Raw forecast sources. Edge gate is per-source + direction calibrated
            # via _resolve_min_edge() (falls back to TEMP_DAY2_MIN_EDGE /
            # TEMP_FORECAST_MIN_EDGE for sources not in the override dicts).
            if opp.edge < _resolve_min_edge(opp.source, opp.direction):
                continue

            # Morning observation gate: block forecast YES signals when the
            # current observed max is too far below the strike to be credible.
            # Extended forecasts (noaa_day2 … noaa_day7) are exempt — today's
            # observed max is irrelevant to a future day's expected high.
            if (
                opp.implied_outcome == "YES"
                and not opp.source.startswith("noaa_day")
                and MORNING_OBS_GAP_F > 0
                and observed_values is not None
                and opp.strike is not None
            ):
                obs_val = observed_values.get(opp.metric)
                if obs_val is not None and (opp.strike - obs_val) > MORNING_OBS_GAP_F:
                    logging.info(
                        "Morning gate: suppressed %s YES forecast — observed %.1f°F"
                        " is %.1f°F below strike %.1f°F (gap > %.0f°F threshold)",
                        opp.market_ticker, obs_val,
                        opp.strike - obs_val, opp.strike, MORNING_OBS_GAP_F,
                    )
                    continue

            # HRRR spread gate: suppress when hourly and daily forecasts disagree.
            # Only applies to same-day forecasts — extended-day sources
            # (noaa_day2 … noaa_day7, open_meteo day1+) target a future date,
            # so today's HRRR hourly peak is not a valid quality signal for them.
            is_future_day = (
                opp.source.startswith("noaa_day")
                or (
                    opp.source in ("open_meteo", "nws_hourly", "weatherapi")
                    and opp.metadata.get("forecast_offset", 0) > 0
                )
            )
            if hrrr_hourly_highs and not is_future_day:
                hourly_high = hrrr_hourly_highs.get(opp.metric)
                if hourly_high is not None:
                    spread = abs(opp.data_value - hourly_high)
                    if spread >= HRRR_MAX_SPREAD_F:
                        # Only block if the spread puts daily and hourly on
                        # opposite sides of the strike — if both forecasts still
                        # predict the same outcome, spread uncertainty in magnitude
                        # does not change the trade direction.
                        outcome_flips = True  # default: block when no strike info
                        if opp.strike is not None:
                            if opp.direction == "over":
                                daily_yes = opp.data_value >= opp.strike
                                hourly_yes = hourly_high >= opp.strike
                            else:  # "under"
                                daily_yes = opp.data_value <= opp.strike
                                hourly_yes = hourly_high <= opp.strike
                            outcome_flips = (daily_yes != hourly_yes)

                        if outcome_flips:
                            logging.info(
                                "HRRR gate: suppressed %s — daily %.1f°F vs hourly %.1f°F"
                                " (spread %.1f°F flips outcome across strike %.1f°F)",
                                opp.market_ticker, opp.data_value, hourly_high,
                                spread, opp.strike or 0.0,
                            )
                            continue
                        else:
                            logging.info(
                                "HRRR spread %.1f°F on %s — daily %.1f°F and"
                                " hourly %.1f°F agree on outcome; allowing through",
                                spread, opp.market_ticker,
                                opp.data_value, hourly_high,
                            )

            result.append(opp)

    return result


async def _poll(
    session: aiohttp.ClientSession,
    seen: SeenDocuments,
    opp_log: OpportunityLog,
    executor: TradeExecutor,
    ledger: DryRunLedger | None,
) -> None:
    """Run one full fetch-match-report cycle."""

    # ---- confirm fills / cancel stale resting orders (live mode only) ------
    # Runs before any new data is processed so fill state is current before
    # the position filter and circuit-breaker logic run this cycle.
    await executor.poll_open_orders(session)

    # ---- concurrent fetch of every source at once --------------------------
    # Tasks are named so unpacking is robust to future source additions.
    # To add a source: append a ("name", coroutine) pair and unpack by name below.
    _tasks: list[tuple[str, object]] = [
        ("markets",      _get_markets(session)),
        *[(f"fed_{a}",   federal_register.fetch_documents(session, agency_slug=a))
          for a in AGENCIES],
        ("rss",          rss.fetch_all_feeds(session)),
        ("nws",          nws_alerts.fetch_alerts(session)),
        ("nws_signals",  nws_alerts.fetch_city_alert_signals(session)),
        ("hrrr",         hrrr.fetch_hourly_highs(session)),
        ("edgar",        edgar.fetch_filings(session)),
        ("noaa",         noaa.fetch_city_forecasts(session)),
        ("nws_climo",    nws_climo.fetch_city_climo(session)),
        ("metar",        metar.fetch_city_forecasts(session)),
        ("owm",          owm.fetch_city_forecasts(session)),
        ("nws_hourly",   nws_hourly.fetch_city_forecasts(session)),
        ("weatherapi",   weatherapi.fetch_city_forecasts(session)),
        ("open_meteo",    open_meteo.fetch_city_forecasts(session)),
        ("equity_index",  equity_index.fetch_prices(session)),
        ("binance",      binance.fetch_prices(session)),
        ("coinbase",     coinbase.fetch_prices(session)),
        ("frankfurter",  frankfurter.fetch_rates(session)),
        ("yahoo_forex",  yahoo_forex.fetch_rates(session)),
        ("bls",          bls.fetch_latest(session, seen)),
        ("fred",         fred.fetch_rates(session)),
        ("eia",          eia.fetch_prices(session)),
        ("wti_futures",  wti_futures.fetch_futures(session)),
        ("fedwatch",     cme_fedwatch.fetch_next_meeting(session)),
        ("adp",          adp.fetch_datapoints(session)),
        ("chicago_pmi",  chicago_pmi.fetch_datapoints(session)),
        ("box_office",   box_office.fetch_weekend_chart(session, seen)),
        ("polymarket",   polymarket.fetch_markets(session)),
        ("metaculus",    metaculus.fetch_questions(session)),
        ("manifold",     manifold.fetch_markets(session)),
        ("predictit",    predictit.fetch_contracts(session)),
        ("positions",    fetch_positions(session)),
    ]
    _task_names, _task_coros = zip(*_tasks)
    _raw = await asyncio.gather(*_task_coros, return_exceptions=True)
    R: dict[str, object] = dict(zip(_task_names, _raw))

    # ---- unpack by name ----------------------------------------------------
    markets_result     = R["markets"]
    fed_results        = [R[f"fed_{a}"] for a in AGENCIES]
    rss_result         = R["rss"]
    nws_result         = R["nws"]
    nws_signals_result = R["nws_signals"]
    hrrr_result        = R["hrrr"]
    edgar_result       = R["edgar"]
    noaa_result        = R["noaa"]
    nws_climo_result   = R["nws_climo"]
    metar_result       = R["metar"]
    owm_result         = R["owm"]
    nws_hourly_result  = R["nws_hourly"]
    weatherapi_result  = R["weatherapi"]
    open_meteo_result    = R["open_meteo"]
    equity_index_result  = R["equity_index"]
    binance_result     = R["binance"]
    coinbase_result    = R["coinbase"]
    frankfurter_result = R["frankfurter"]
    yahoo_forex_result = R["yahoo_forex"]
    bls_result         = R["bls"]
    fred_result        = R["fred"]
    eia_result         = R["eia"]
    wti_futures_result = R["wti_futures"]
    fedwatch_result    = R["fedwatch"]
    adp_result         = R["adp"]
    chicago_pmi_result  = R["chicago_pmi"]
    box_office_result   = R["box_office"]
    poly_result         = R["polymarket"]
    metaculus_result   = R["metaculus"]
    manifold_result    = R["manifold"]
    predictit_result   = R["predictit"]
    positions_result   = R["positions"]

    if isinstance(fedwatch_result, Exception):
        logging.error("CME FedWatch fetch error: %s", fedwatch_result)

    if isinstance(markets_result, Exception):
        logging.error("Failed to fetch markets: %s", markets_result)
        return
    markets: list[dict] = markets_result  # type: ignore[assignment]

    # ---- portfolio positions -----------------------------------------------
    if isinstance(positions_result, Exception):
        logging.warning("Portfolio positions fetch failed: %s", positions_result)
        raw_positions: list[dict] = []
    else:
        raw_positions = positions_result or []  # type: ignore[assignment]
    positions = build_position_index(raw_positions)
    portfolio_summary = summarise_portfolio(raw_positions)

    # Temporal filter — drop markets closing too far out or too soon
    if MARKET_MAX_DAYS_OUT > 0 or MARKET_MIN_MINUTES_TO_CLOSE > 0:
        before = len(markets)
        markets = _filter_by_close_time(
            markets, MARKET_MAX_DAYS_OUT, MARKET_MIN_MINUTES_TO_CLOSE
        )
        dropped = before - len(markets)
        if dropped:
            logging.info(
                "Temporal filter: kept %d / %d markets "
                "(dropped %d outside [%dm, %dd] close-time window).",
                len(markets), before, dropped,
                MARKET_MIN_MINUTES_TO_CLOSE, MARKET_MAX_DAYS_OUT,
            )

    # ---- collect new text documents from all text sources ------------------
    all_new_docs: list[dict] = []

    # Federal Register
    for agency, doc_result in zip(AGENCIES, fed_results):
        if isinstance(doc_result, Exception):
            logging.error("Federal Register error [%s]: %s", agency, doc_result)
            continue
        _drain_text_source(
            f"Federal Register [{agency}]",
            doc_result,
            seen,
            id_field="document_number",
            source_tag="federal_register",
            accumulator=all_new_docs,
        )

    # RSS feeds
    if isinstance(rss_result, Exception):
        logging.error("RSS fetch error: %s", rss_result)
    else:
        _drain_text_source(
            "RSS feeds",
            rss_result,
            seen,
            id_field="document_number",
            source_tag="rss",
            accumulator=all_new_docs,
        )

    # NWS weather alerts
    if isinstance(nws_result, Exception):
        logging.error("NWS Alerts fetch error: %s", nws_result)
    else:
        _drain_text_source(
            "NWS Alerts",
            nws_result,
            seen,
            id_field="document_number",
            source_tag="nws_alerts",
            accumulator=all_new_docs,
        )

    # SEC EDGAR 8-K filings
    if isinstance(edgar_result, Exception):
        logging.error("EDGAR fetch error: %s", edgar_result)
    else:
        _drain_text_source(
            "EDGAR 8-K",
            edgar_result,
            seen,
            id_field="document_number",
            source_tag="edgar",
            accumulator=all_new_docs,
        )

    # ---- text / keyword matching (source-scoped) ---------------------------
    text_opps: list[Opportunity] = []
    if markets:
        for group in _SOURCE_GROUPS:
            group_docs = [
                d for d in all_new_docs
                if (d.get("feed_id") or d.get("_source", "")) in group["feed_ids"]
            ]
            if not group_docs:
                continue
            group_markets = _select_markets(
                markets, group["include_prefixes"], group["exclude_prefixes"]
            )
            if not group_markets:
                continue
            opps = find_opportunities(
                group_docs, group_markets, group["topics"],
                require_title_match=group.get("require_title_match", True),
            )
            if opps:
                logging.info(
                    "Group [%s]: %d opportunity(ies) from %d doc(s) vs %d market(s).",
                    group["name"], len(opps), len(group_docs), len(group_markets),
                )
            text_opps.extend(opps)

    # Fix 1: cycle-level dedup — keep only first occurrence of each (term, ticker) pair
    seen_pairs: set[tuple[str, str]] = set()
    deduped_opps: list[Opportunity] = []
    for opp in text_opps:
        key = (opp.matched_terms[0] if opp.matched_terms else opp.topic.lower(), opp.market_ticker)
        if key not in seen_pairs:
            seen_pairs.add(key)
            deduped_opps.append(opp)
    text_opps = deduped_opps

    # Mark all new docs as seen — tracked per source for accurate audit trail
    if all_new_docs:
        by_source: dict[str, list[str]] = collections.defaultdict(list)
        for d in all_new_docs:
            by_source[d.get("_source", "unknown")].append(str(d.get("document_number", "")))
        for src, ids in by_source.items():
            seen.mark_many(ids, source=src)
    else:
        logging.info("No new text items this cycle.")

    # ---- HRRR hourly highs (quality gate + forecast corroboration source) ---
    hrrr_hourly_highs: dict[str, float] = {}
    if isinstance(hrrr_result, Exception):
        logging.warning("HRRR fetch error: %s", hrrr_result)
    elif hrrr_result:
        hrrr_hourly_highs = hrrr_result  # type: ignore[assignment]

    # ---- numeric matching (NOAA, OWM, Binance, Frankfurter, BLS, FRED, EIA) -
    # Staleness gate for nws_hourly: drop DataPoints whose fetched_at timestamp
    # is older than NWS_HOURLY_MAX_AGE_HOURS.  nws_hourly updates every hour so
    # a 3-hour-old reading has missed ≥2 cycles and may be badly wrong on a
    # rapidly warming day.  Stale DataPoints are dropped entirely — they should
    # not count toward corroboration or trigger trades.
    if not isinstance(nws_hourly_result, Exception) and nws_hourly_result and NWS_HOURLY_MAX_AGE_HOURS > 0:
        _nws_cutoff = datetime.now(timezone.utc) - timedelta(hours=NWS_HOURLY_MAX_AGE_HOURS)
        _nws_fresh, _nws_stale = [], []
        for _dp in nws_hourly_result:
            _fetched_at_str = (_dp.metadata or {}).get("fetched_at")
            if _fetched_at_str:
                try:
                    _fetched_dt = datetime.fromisoformat(_fetched_at_str)
                    if _fetched_dt < _nws_cutoff:
                        _nws_stale.append(_dp)
                        continue
                except (ValueError, AttributeError):
                    pass
            _nws_fresh.append(_dp)
        if _nws_stale:
            logging.warning(
                "nws_hourly staleness gate: dropped %d DataPoint(s) older than %.1fh"
                " (cities: %s)",
                len(_nws_stale),
                NWS_HOURLY_MAX_AGE_HOURS,
                ", ".join({dp.metadata.get("city", dp.metric) for dp in _nws_stale}),
            )
        nws_hourly_result = _nws_fresh

    data_points = []
    for label, result in [
        ("NOAA",        noaa_result),
        ("NWS Climo",   nws_climo_result),
        ("METAR",       metar_result),
        ("OWM",         owm_result),
        ("NWS Hourly",  nws_hourly_result),
        ("WeatherAPI",  weatherapi_result),
        ("Open-Meteo",  open_meteo_result),
        ("Equity",      equity_index_result),
        ("Binance",     binance_result),
        ("Coinbase",    coinbase_result),
        ("Frankfurter", frankfurter_result),
        ("Yahoo Forex", yahoo_forex_result),
        ("BLS",         bls_result),
        ("FRED",        fred_result),
        ("EIA",         eia_result),
        ("WTI Futures", wti_futures_result),
    ]:
        if isinstance(result, Exception):
            logging.error("%s fetch error: %s", label, result)
        elif result:
            data_points.extend(result)

    # CME FedWatch — expected post-meeting Fed Funds rate (continuous signal,
    # not gated to FOMC release dates like FRED fred_fedfunds).
    fedwatch_dps = await cme_fedwatch.fetch_fedwatch_datapoints(session)
    if fedwatch_dps:
        data_points.extend(fedwatch_dps)

    # EIA Inventory — deferred to AFTER the staleness filter below so that
    # implied prices are computed from the same fresh spot the filter approves.
    # (Previously extracted from raw eia_result before filtering, which allowed
    # stale 2022-era WTI prints to slip through and produce wildly wrong signals.)

    # ADP Employment Report — pre-signal for KXNFP (Wed before BLS Friday).
    if isinstance(adp_result, Exception):
        logging.error("ADP fetch error: %s", adp_result)
    elif adp_result:
        data_points.extend(adp_result)

    # Chicago PMI — leading indicator for KXISMMFG (last biz day of month).
    if isinstance(chicago_pmi_result, Exception):
        logging.error("Chicago PMI fetch error: %s", chicago_pmi_result)
    elif chicago_pmi_result:
        data_points.extend(chicago_pmi_result)

    # Box office — weekend gross estimates from The Numbers chart.
    if isinstance(box_office_result, Exception):
        logging.error("Box office fetch error: %s", box_office_result)
    elif box_office_result:
        data_points.extend(box_office_result)

    # NWS alert signals: parsed temperatures from active heat/cold warnings.
    # Treated as high-confidence numeric triggers (same confidence tier as
    # noaa_observed) because NWS only issues warnings when the forecast is
    # certain enough to warrant a public advisory.
    if isinstance(nws_signals_result, Exception):
        logging.warning("NWS alert signals fetch error: %s", nws_signals_result)
    elif nws_signals_result:
        data_points.extend(nws_signals_result)  # type: ignore[arg-type]

    # ---- HRRR DataPoints for forecast corroboration -------------------------
    # Convert the HRRR hourly-high dict to DataPoints so source="hrrr" counts
    # toward FORECAST_CORROBORATION_MIN.  A lone NOAA signal confirmed by HRRR
    # satisfies the 2-source gate without requiring OWM or Open-Meteo.
    # The HRRR spread gate in _filter_weather_opportunities still applies to
    # all other forecast sources; HRRR opps self-consistently pass (spread=0).
    if hrrr_hourly_highs:
        data_points.extend(
            hrrr.to_data_points(
                hrrr_hourly_highs,
                datetime.now(timezone.utc).isoformat(),
            )
        )

    # ---- EIA staleness filter -----------------------------------------------
    # EIA daily spot prices carry a period_str equal to the data date (e.g.
    # "2026-03-07" for Friday's price published on Monday).  Drop any EIA
    # DataPoint whose period date is more than EIA_MAX_STALE_DAYS calendar
    # days before today.  This prevents Friday's WTI price from being used
    # to trade the following Monday's KXWTI market.
    if EIA_MAX_STALE_DAYS < 7 and any(dp.source == "eia" for dp in data_points):
        today_date = datetime.now(timezone.utc).date()
        fresh: list = []
        stale_count = 0
        for dp in data_points:
            if dp.source != "eia":
                fresh.append(dp)
                continue
            try:
                period_date = datetime.fromisoformat(
                    dp.as_of.split("T")[0].replace("Z", "")
                ).date()
                age_days = (today_date - period_date).days
                if age_days <= EIA_MAX_STALE_DAYS:
                    fresh.append(dp)
                else:
                    stale_count += 1
                    logging.debug(
                        "EIA staleness gate: dropped %s period=%s (age %d day(s),"
                        " max=%d)",
                        dp.metric, dp.as_of, age_days, EIA_MAX_STALE_DAYS,
                    )
            except (ValueError, AttributeError):
                fresh.append(dp)  # unparseable as_of: allow through
        if stale_count:
            logging.info(
                "EIA staleness gate: dropped %d stale EIA DataPoint(s)"
                " (data older than %d day(s)).",
                stale_count, EIA_MAX_STALE_DAYS,
            )
        data_points = fresh

    # ---- EIA Inventory (post-staleness-filter) ------------------------------
    # Extract spot prices from the already-filtered data_points so that stale
    # EIA periods (e.g. 2022-era WTI at $114) cannot reach eia_inventory.
    # Opens a Tuesday 16:30 ET (API report) and Wednesday 10:30 ET (AGA report)
    # pre-release window in addition to the standard Wednesday/Thursday EIA windows.
    _eia_wti_spot    = next((dp.value for dp in data_points if dp.source == "eia" and dp.metric == "eia_wti"),    None)
    _eia_natgas_spot = next((dp.value for dp in data_points if dp.source == "eia" and dp.metric == "eia_natgas"), None)
    if _eia_wti_spot is not None or _eia_natgas_spot is not None:
        try:
            _eia_inv_dps = await eia_inventory.fetch_signals(
                session, wti_spot=_eia_wti_spot, natgas_spot=_eia_natgas_spot
            )
            if _eia_inv_dps:
                data_points.extend(_eia_inv_dps)
        except Exception as _e:
            logging.error("EIA inventory fetch error: %s", _e)

    # ---- Frankfurter/ECB forex staleness filter -----------------------------
    # ECB reference rates publish once per day at ~16:00 CET (~10:00 AM ET).
    # Until that publication the only available rate is yesterday's — which is
    # fully priced into the market and worthless as a directional signal for
    # today's fixing.  Drop any Frankfurter DataPoint whose ecb_date is more
    # than FOREX_MAX_STALE_DAYS calendar days before today.
    if FOREX_MAX_STALE_DAYS < 7 and any(dp.source == "frankfurter" for dp in data_points):
        today_date = datetime.now(timezone.utc).date()
        fx_fresh: list = []
        fx_stale = 0
        for dp in data_points:
            if dp.source != "frankfurter":
                fx_fresh.append(dp)
                continue
            try:
                ecb_date = datetime.fromisoformat(
                    dp.as_of.split("T")[0].replace("Z", "")
                ).date()
                age_days = (today_date - ecb_date).days
                if age_days <= FOREX_MAX_STALE_DAYS:
                    fx_fresh.append(dp)
                else:
                    fx_stale += 1
                    logging.debug(
                        "Forex staleness gate: dropped %s ecb_date=%s (age %d day(s),"
                        " max=%d)",
                        dp.metric, dp.as_of, age_days, FOREX_MAX_STALE_DAYS,
                    )
            except (ValueError, AttributeError):
                fx_fresh.append(dp)  # unparseable date: allow through
        if fx_stale:
            logging.info(
                "Forex staleness gate: dropped %d stale Frankfurter DataPoint(s)"
                " (ECB data older than %d day(s)).",
                fx_stale, FOREX_MAX_STALE_DAYS,
            )
        data_points = fx_fresh

    # ---- GDPNow staleness filter --------------------------------------------
    # Atlanta Fed GDPNow updates ~2-3x per week during the quarter.  Between
    # updates the series date can be 1-3 days old, which is still useful.
    # The risky period is the inter-quarter gap: after the BEA advance estimate
    # is published the old quarter's GDPNOW stops updating, leaving a stale
    # reading until the new quarter's estimate begins.  Drop any fred_gdp_nowcast
    # DataPoint whose FRED series date exceeds GDPNOW_MAX_STALE_DAYS.
    # Note: filter is scoped to metric=="fred_gdp_nowcast" only — other FRED
    # series (DGS10, ICSA, etc.) reflect the most recent trading/report day
    # and must never be age-filtered here.
    if GDPNOW_MAX_STALE_DAYS < 30 and any(
        dp.source == "fred" and dp.metric == "fred_gdp_nowcast" for dp in data_points
    ):
        today_date = datetime.now(timezone.utc).date()
        gdp_fresh: list = []
        gdp_stale = 0
        for dp in data_points:
            if dp.source != "fred" or dp.metric != "fred_gdp_nowcast":
                gdp_fresh.append(dp)
                continue
            try:
                series_date = datetime.fromisoformat(
                    dp.as_of.split("T")[0].replace("Z", "")
                ).date()
                age_days = (today_date - series_date).days
                if age_days <= GDPNOW_MAX_STALE_DAYS:
                    gdp_fresh.append(dp)
                else:
                    gdp_stale += 1
                    logging.info(
                        "GDPNow staleness gate: dropped fred_gdp_nowcast"
                        " series_date=%s (age %d day(s) > max %d)",
                        dp.as_of, age_days, GDPNOW_MAX_STALE_DAYS,
                    )
            except (ValueError, AttributeError):
                gdp_fresh.append(dp)  # unparseable date: allow through
        if gdp_stale:
            logging.info(
                "GDPNow staleness gate: dropped %d stale GDPNow DataPoint(s)"
                " (series date older than %d day(s)).",
                gdp_stale, GDPNOW_MAX_STALE_DAYS,
            )
        data_points = gdp_fresh

    # ---- Intraday trajectory projections (obs_trajectory) -------------------
    # Extract METAR time-series already present in data_points and append
    # obs_trajectory DataPoints.  These flow through numeric_matcher exactly
    # like any other forecast source.  Zero extra API calls — the obs_series
    # is already fetched by metar.fetch_city_forecasts().
    _metar_dps = [dp for dp in data_points if dp.source == "metar"]
    if _metar_dps:
        _traj_dps = _compute_trajectory_projections(_metar_dps, datetime.now(timezone.utc))
        if _traj_dps:
            data_points.extend(_traj_dps)

    numeric_opps: list[NumericOpportunity] = []
    if data_points and markets:
        numeric_opps = find_numeric_opportunities(
            data_points, markets, min_edge=NUMERIC_MIN_EDGE
        )

    # Box office custom matcher — bypasses ticker-prefix system, matches by title.
    if box_office_result and not isinstance(box_office_result, Exception) and markets:
        bo_opps = match_box_office_to_kalshi(box_office_result, markets)
        if bo_opps:
            numeric_opps.extend(bo_opps)

    # ---- Per-category minimum edge filter (noise reduction) -----------------
    # Drops non-weather opportunities whose edge is below a fraction of each
    # metric's reference scale.  Disabled by default (NUMERIC_MIN_EDGE_FRACTION=0).
    if NUMERIC_MIN_EDGE_FRACTION > 0 and numeric_opps:
        _pre_fraction = len(numeric_opps)
        _filtered: list = []
        for _o in numeric_opps:
            if _o.metric.startswith(("temp_high", "temp_low")):
                _filtered.append(_o)  # weather has its own gates
                continue
            _scale = next(
                (v for k, v in METRIC_EDGE_SCALES.items() if _o.metric.startswith(k)),
                None,
            )
            if _scale is None or _o.edge >= NUMERIC_MIN_EDGE_FRACTION * _scale:
                _filtered.append(_o)
        numeric_opps = _filtered
        _dropped_fraction = _pre_fraction - len(numeric_opps)
        if _dropped_fraction:
            logging.debug(
                "Edge fraction filter: dropped %d opportunity(ies) "
                "(edge < %.0f%% of metric scale).",
                _dropped_fraction, NUMERIC_MIN_EDGE_FRACTION * 100,
            )

    # ---- Structured government event matching (binary, bypass numeric matcher)
    # These modules do their own title-based market matching internally and
    # return NumericOpportunity objects with implied_outcome set directly.
    if markets:
        congress_opps = await congress.find_congress_opportunities(session, markets)
        if congress_opps:
            numeric_opps.extend(congress_opps)

        wh_opps = await whitehouse.find_whitehouse_opportunities(session, markets)
        if wh_opps:
            numeric_opps.extend(wh_opps)

    # ---- OWM consensus filter (temp markets only) --------------------------
    # For each (metric, ticker) pair with multiple forecast sources, drop all
    # forecast-based opportunities if NOAA and OWM disagree on direction.
    # noaa_observed is always exempt — station readings are ground truth.
    #
    # Extended check: also suppress forecasts that conflict with any concurrent
    # noaa_observed data point, even if that observation is no longer tradeable
    # (e.g. contrarian cap already filled or market price moved past threshold).
    # This prevents the bot from trading a forecast against a station reading
    # that tells the opposite story — the observed station is ground truth.
    if numeric_opps:
        # Build a direct lookup of noaa_observed values from raw data_points
        # so we can detect conflicts even when the observed opp was filtered out.
        _obs_value: dict[str, float] = {}
        for _dp in data_points:
            if _dp.source in ("noaa_observed", "metar") and _dp.metric.startswith(("temp_high", "temp_low")):
                # metar updates faster — prefer it over noaa_observed when both present
                if _dp.source == "metar" or _dp.metric not in _obs_value:
                    _obs_value[_dp.metric] = _dp.value

        # NWS "Tonight" (day1) forecast low per temp_low metric.
        # source="noaa" carries the day-1 overnight low from fetch_city_forecasts().
        # Used by _filter_weather_opportunities to suppress noaa_observed YES signals
        # when tonight's forecast will push the daily minimum below the strike.
        _fc_low_by_metric: dict[str, float] = {}
        for _dp in data_points:
            if _dp.source == "noaa" and _dp.metric.startswith("temp_low_"):
                _fc_low_by_metric[_dp.metric] = _dp.value

        by_key: dict[tuple[str, str], list[NumericOpportunity]] = collections.defaultdict(list)
        for opp in numeric_opps:
            by_key[(opp.metric, opp.market_ticker)].append(opp)

        consensus_opps: list[NumericOpportunity] = []
        for (metric, ticker), group in by_key.items():
            if not metric.startswith(("temp_high", "temp_low")):
                consensus_opps.extend(group)
                continue
            # noaa_observed, nws_climo, nws_alert are ground-truth / high-confidence
            # — always pass through the consensus check unchanged.
            # owm is excluded from the unanimity check entirely: the OWM free-tier
            # "Current Weather" API returns the *current* temperature, not the
            # forecast day-high, so it will systematically disagree with NWS
            # forecast-high sources during morning hours.
            # open_meteo is also excluded: it uses raw GFS output which
            # systematically underestimates cold-front timing/intensity and lags
            # NWS human-adjusted forecasts by 6-18 hours.  NOAA day-2+ is already
            # a multi-model blend (NBM) superior to raw GFS, so GFS should not
            # veto it.  Open-Meteo still runs and its DataPoints still count toward
            # corroboration when it agrees with NOAA, but it cannot block a NOAA
            # signal by disagreeing.
            _PASS_THROUGH = ("noaa_observed", "metar", "nws_climo", "nws_alert")

            # Compute market date and same-day flag before building _EXCLUDE so
            # we can tune which sources participate in the vote.
            mkt_date = _ticker_date(ticker)
            today_et = datetime.now(_ET).date()
            is_same_day_mkt = (mkt_date is not None and mkt_date == today_et)

            # noaa_day2–7 draw from the same NWS forecast grid as nws_hourly
            # day+1/2, so counting both creates correlated double-counting.
            # Exception: for future-day markets (tomorrow+), only nws_hourly and
            # weatherapi vote, which makes a 1-1 tie structurally unbreakable.
            # Allowing noaa_day2 to join the vote gives 3 genuinely independent
            # data points (NWS hourly, WeatherAPI, NOAA NBM blend) so a majority
            # always emerges.  noaa_day2 on same-day markets fails date alignment
            # below anyway (as_of = tomorrow ≠ today's market), so this only
            # changes behaviour for tomorrow-closing markets.
            # owm stays excluded — its 3-hour slot aggregation is ±1-2°F less
            # precise than a daily forecast endpoint and it rarely adds signal.
            #
            # open_meteo is excluded for same-day markets (HRRR hourly is better
            # and avoids GFS model noise intra-day), but included for future-day
            # markets where it is a genuinely independent European-model (ECMWF)
            # source that can corroborate NWS extended forecasts.  Unlike owm/noaa,
            # open_meteo is NOT derived from the same NWS NBM blending pipeline.
            _EXCLUDE_BASE = ("owm",
                             "noaa_day3", "noaa_day4",
                             "noaa_day5", "noaa_day6", "noaa_day7")
            _EXCLUDE = _EXCLUDE_BASE + (("noaa_day2", "open_meteo") if is_same_day_mkt else ())
            observed = [o for o in group if o.source in _PASS_THROUGH]
            forecasts = [o for o in group if o.source not in _PASS_THROUGH and o.source not in _EXCLUDE]

            # Date-alignment pre-filter: remove forecast DataPoints whose as_of
            # date does not match the market's resolution date.  Without this,
            # NOAA day-1 (as_of = today) and NOAA day-2 (as_of = tomorrow) both
            # appear in the same (metric, ticker) group for a tomorrow-closing
            # market and disagree with each other, triggering a false "disagree"
            # block.  Pass-through (observed) signals are exempt — they reflect
            # the current day's running maximum and are always relevant.
            if mkt_date is not None:
                aligned: list[NumericOpportunity] = []
                for _fc in forecasts:
                    if not _fc.as_of:
                        aligned.append(_fc)
                        continue
                    try:
                        _fc_date = (
                            datetime.fromisoformat(_fc.as_of.replace("Z", "+00:00"))
                            .astimezone(_ET)
                            .date()
                        )
                        if _fc_date == mkt_date:
                            aligned.append(_fc)
                    except (ValueError, AttributeError):
                        aligned.append(_fc)
                forecasts = aligned

            # Multiple observed sources (noaa_observed, nws_observed) can fire on
            # the same (metric, ticker) in the same poll cycle.  Keep only the
            # highest-edge signal so downstream trade_executor sees one clean
            # opportunity rather than duplicate orders for the same market.
            seen_obs: dict[str, NumericOpportunity] = {}
            for _obs in observed:
                key = f"{_obs.metric}|{_obs.market_ticker}"
                if key not in seen_obs or abs(_obs.edge) > abs(seen_obs[key].edge):
                    seen_obs[key] = _obs
            consensus_opps.extend(seen_obs.values())

            if not forecasts:
                continue

            # Tiered consensus — replaces strict unanimity which became
            # intractable once 4 sources were in the blocking pool.
            #
            # Same-day markets: nws_hourly and hrrr update every hour and
            # track cold fronts in real-time.  NOAA updates only 4×/day and
            # can be 10-20°F stale.  When same-day real-time sources agree,
            # use their direction.  Only override if ALL slower daily sources
            # also oppose (i.e., unanimous disagreement across 2+ daily sources).
            #
            # Future-day markets: NOAA multi-model blend is most reliable at
            # longer horizons.  Use majority vote among all forecasts; a tie
            # is too uncertain to trade.
            _REALTIME_SRCS = frozenset(("nws_hourly", "hrrr"))
            # mkt_date, today_et, is_same_day_mkt already computed above.

            realtime_fc = [o for o in forecasts if o.source in _REALTIME_SRCS]
            daily_fc = [o for o in forecasts if o.source not in _REALTIME_SRCS]

            if is_same_day_mkt and realtime_fc:
                # Real-time sources must agree with each other first.
                rt_outcomes = {o.implied_outcome for o in realtime_fc}
                if len(rt_outcomes) > 1:
                    logging.info(
                        "Consensus filter: real-time sources split on %s %s"
                        " — skipping (sources: %s)",
                        metric, ticker, {o.source for o in realtime_fc},
                    )
                    continue
                forecast_outcome = next(iter(rt_outcomes))
                # Only block when ALL daily-update sources (e.g. NOAA + WeatherAPI)
                # oppose the real-time direction.  A lone NOAA disagreement (stale
                # zone forecast) should not override nws_hourly+hrrr consensus.
                daily_oppose = [o for o in daily_fc if o.implied_outcome != forecast_outcome]
                if len(daily_oppose) == len(daily_fc) and len(daily_fc) >= 2:
                    logging.info(
                        "Consensus filter: all daily sources %s oppose"
                        " real-time (%s) on %s %s — skipping",
                        {o.source for o in daily_oppose},
                        forecast_outcome, metric, ticker,
                    )
                    continue
            else:
                # Future-day (or no real-time source present): NWS-primary logic.
                #
                # nws_hourly and noaa_day2 are both NWS products (hourly zone
                # forecast and NBM extended blend respectively) and are the most
                # reliable multi-day forecast sources.  open_meteo (raw ECMWF/GFS)
                # and weatherapi (third-party model) act as independent validators.
                #
                # If NWS sources agree with each other, treat their direction as
                # primary and only block if ALL external validators unanimously
                # oppose — the same asymmetric rule used on same-day markets where
                # nws_hourly+hrrr are primary.  A 2-2 tie (NWS vs external) is
                # resolved in favour of NWS rather than leaving the market untraded.
                #
                # If NWS sources disagree or are absent, fall back to majority vote.
                _NWS_SRCS = frozenset(("nws_hourly", "noaa_day2"))
                _EXT_SRCS = frozenset(("open_meteo", "weatherapi"))
                nws_fc = [o for o in forecasts if o.source in _NWS_SRCS]
                ext_fc = [o for o in forecasts if o.source in _EXT_SRCS]
                nws_outcomes = {o.implied_outcome for o in nws_fc}

                if len(nws_fc) >= 2 and len(nws_outcomes) == 1:
                    # Both NWS sources agree → use NWS direction.
                    forecast_outcome = next(iter(nws_outcomes))
                    ext_oppose = [o for o in ext_fc if o.implied_outcome != forecast_outcome]
                    if ext_oppose and len(ext_oppose) == len(ext_fc) and len(ext_fc) >= 2:
                        # ALL external models unanimously oppose NWS → too uncertain.
                        logging.info(
                            "Consensus filter: all external models %s oppose"
                            " NWS primary (%s) on %s %s — skipping",
                            {o.source for o in ext_oppose},
                            forecast_outcome, metric, ticker,
                        )
                        continue
                    logging.info(
                        "Consensus filter: NWS primary (%s) on %s %s"
                        " — %d external validator(s) oppose but not unanimous; allowing",
                        forecast_outcome, metric, ticker, len(ext_oppose),
                    )
                else:
                    # NWS sources disagree or absent: plain majority vote.
                    yes_count = sum(1 for o in forecasts if o.implied_outcome == "YES")
                    no_count = sum(1 for o in forecasts if o.implied_outcome == "NO")
                    if yes_count == no_count:
                        logging.info(
                            "Consensus filter: forecasts tied on %s %s — skipping"
                            " (sources: %s yes=%d no=%d)",
                            metric, ticker,
                            {o.source for o in forecasts}, yes_count, no_count,
                        )
                        continue
                    forecast_outcome = "YES" if yes_count > no_count else "NO"
            # Cross-check against raw noaa_observed data even when that
            # observation is too expensive to trade (not in numeric_opps).
            #
            # IMPORTANT: only valid for same-day markets.  For future-day markets
            # (tomorrow or later), today's observed temperature has no bearing on
            # tomorrow's forecast — of course today's high is below tomorrow's
            # warm-day strike.  Applying this check to future-day markets silently
            # suppressed ALL future-day YES forecasts where today's temp hadn't
            # yet reached tomorrow's strike (e.g. Austin T86: today 72°F → "NO"
            # conflicts with noaa_day2+weatherapi "YES" for tomorrow → suppressed).
            obs_val = _obs_value.get(metric)
            if obs_val is not None and not observed and is_same_day_mkt:
                ref = forecasts[0]
                if ref.direction == "over" and ref.strike is not None:
                    obs_outcome = "YES" if obs_val >= ref.strike else "NO"
                elif ref.direction == "under" and ref.strike is not None:
                    obs_outcome = "YES" if obs_val <= ref.strike else "NO"
                else:
                    obs_outcome = None
                if obs_outcome is not None and obs_outcome != forecast_outcome:
                    logging.info(
                        "Consensus filter: noaa_observed (%.1f°F → %s) conflicts"
                        " with forecast (%s) on %s %s — suppressing forecast",
                        obs_val, obs_outcome, forecast_outcome, metric, ticker,
                    )
                    continue
            # Cross-source corroboration gate: for YES forecast signals,
            # require at least FORECAST_CORROBORATION_MIN distinct sources to
            # agree.  Lone signals (only one model sees it) are model noise.
            #
            # Special rule: open_meteo (raw GFS) may never trade solo —
            # it systematically underestimates cold-front timing/intensity and
            # is already embedded in NOAA's blended forecast.  A lone
            # open_meteo YES is suppressed regardless of FORECAST_CORROBORATION_MIN.
            # NOAA-family sources (noaa, noaa_day2 … noaa_day7) may trade solo
            # when FORECAST_CORROBORATION_MIN=1 since they are already a
            # multi-model blend with human forecaster adjustments.
            if forecast_outcome == "YES":
                yes_sources = {o.source for o in forecasts if o.implied_outcome == "YES"}
                # obs_trajectory is an observed-trend signal (derived from real METAR
                # readings) — treated alongside noaa/hrrr as a trusted source.
                noaa_yes = {
                    s for s in yes_sources
                    if s.startswith("noaa") or s in ("hrrr", "obs_trajectory")
                }
                # Sources that may never trade solo (unreliable standalone forecasts):
                # - open_meteo: raw GFS, systematically underestimates cold-front timing
                # - weatherapi: commercial aggregator with demonstrated day+0/+1 errors
                #   (7°F+ same-day, >16°F next-day). Can still vote in consensus.
                _SOLO_BLOCKED = {"open_meteo", "weatherapi"}
                gfs_only = not noaa_yes  # True when no NOAA/HRRR/obs_trajectory source is in YES set
                solo_unreliable = len(yes_sources) == 1 and yes_sources <= _SOLO_BLOCKED
                if (gfs_only or solo_unreliable) and len(yes_sources) < 2:
                    logging.info(
                        "Corroboration gate: lone %s YES on %s %s — "
                        "unreliable source may not trade solo; suppressed",
                        ", ".join(yes_sources), metric, ticker,
                    )
                    continue
                # obs_trajectory is self-corroborating: it derives from real observed
                # METAR readings so a lone signal is accepted without requiring an
                # additional model source.  All other trusted sources (noaa, hrrr)
                # still require FORECAST_CORROBORATION_MIN when trading solo.
                obs_traj_solo = yes_sources == {"obs_trajectory"}
                if not gfs_only and not obs_traj_solo and FORECAST_CORROBORATION_MIN > 1:
                    if len(yes_sources) < FORECAST_CORROBORATION_MIN:
                        logging.info(
                            "Corroboration gate: lone %s YES on %s %s"
                            " (need %d sources, have %d) — suppressed",
                            next(iter(yes_sources)), metric, ticker,
                            FORECAST_CORROBORATION_MIN, len(yes_sources),
                        )
                        continue
                # Hard gate: noaa_day2…day7 are all from the same NWS extended
                # product and are NOT independent of each other.  Even with
                # FORECAST_CORROBORATION_MIN=1, require at least one genuinely
                # independent source (noaa same-day, hrrr, owm, open_meteo)
                # before acting on a day-ahead NWS forecast signal.
                # These forecasts update infrequently and can be 12+ hours stale
                # by the time the market moves — a single stale model run is not
                # enough justification to trade.
                _NOAA_FUTURE_PREFIXES = (
                    "noaa_day2", "noaa_day3", "noaa_day4",
                    "noaa_day5", "noaa_day6", "noaa_day7",
                )
                _noaa_future = {
                    s for s in yes_sources
                    if any(s == p or s.startswith(p + "_") for p in _NOAA_FUTURE_PREFIXES)
                }
                if _noaa_future and not (yes_sources - _noaa_future):
                    logging.info(
                        "Corroboration gate: day-ahead NOAA-only YES on %s %s"
                        " (%s) — no independent model confirms; suppressed",
                        metric, ticker, ", ".join(sorted(_noaa_future)),
                    )
                    continue

            # NO-side corroboration gate (future-day markets).
            # YES signals require FORECAST_CORROBORATION_MIN sources; NO had no
            # equivalent, which is why all 3 noaa_day2 NO trades were disasters —
            # a lone NWS extended-forecast saying "temp won't reach the strike"
            # was allowed through with zero independent confirmation.
            # Canonical failures: trades #185 (-135¢), #188 (-69¢), #189 (-4¢).
            # Now require the same count as YES.  With open_meteo now in the
            # forecast pool for future-day markets, noaa_day2 + open_meteo
            # provides a genuinely independent NWS+ECMWF two-source signal.
            if forecast_outcome == "NO" and not is_same_day_mkt:
                no_forecasts_fday = [o for o in forecasts if o.implied_outcome == "NO"]
                no_sources_fday = {o.source for o in no_forecasts_fday}
                # weatherapi is demoted to corroboration-only (same as open_meteo).
                # A weatherapi-only NO on a future-day market is suppressed regardless
                # of FORECAST_CORROBORATION_MIN.
                if len(no_sources_fday) == 1 and no_sources_fday <= {"open_meteo", "weatherapi"}:
                    logging.info(
                        "Corroboration gate: lone %s NO on future-day %s %s — "
                        "unreliable source may not trade solo; suppressed",
                        ", ".join(no_sources_fday), metric, ticker,
                    )
                    continue
                if len(no_sources_fday) < FORECAST_CORROBORATION_MIN:
                    logging.info(
                        "Corroboration gate: lone %s NO on future-day %s %s"
                        " (need %d sources, have %d) — suppressed",
                        ", ".join(sorted(no_sources_fday)), metric, ticker,
                        FORECAST_CORROBORATION_MIN, len(no_sources_fday),
                    )
                    continue
                # For "between" band NO signals, sources can agree on NO for
                # opposite reasons: one says too hot (above band), another says
                # too cold (below band).  This is contradictory, not corroboration.
                # Require all NO sources to agree on which side of the band the
                # forecast falls (all above strike_hi, or all below strike_lo).
                ref_opp = next((o for o in no_forecasts_fday), None)
                if (
                    ref_opp is not None
                    and ref_opp.direction == "between"
                    and ref_opp.strike_lo is not None
                    and ref_opp.strike_hi is not None
                ):
                    above = [o for o in no_forecasts_fday if o.data_value > ref_opp.strike_hi]
                    below = [o for o in no_forecasts_fday if o.data_value < ref_opp.strike_lo]
                    if above and below:
                        logging.info(
                            "Corroboration gate: contradictory NO on between %s %s"
                            " — sources disagree on which side of band"
                            " (above=%s, below=%s); suppressed",
                            metric, ticker,
                            [o.source for o in above],
                            [o.source for o in below],
                        )
                        continue

            # NO-side corroboration gate (same-day markets).
            # A lone real-time source (nws_hourly only, no HRRR) calling NO
            # on a same-day market carries substantial late-afternoon risk:
            # hourly forecast grids lag reality by 1–3 hours and the daily
            # high may already be locked in.  Require at least one daily
            # source (noaa) to corroborate the NO direction. weatherapi is demoted
            # to corroboration-only and cannot act as a qualifying daily source here.
            # Trade #200 is the canonical failure: nws_hourly alone fired NO
            # at 4:11 PM ET; market resolved YES 7 minutes later.
            # HRRR is exempt: it runs hourly with data assimilation (no lag),
            # has scored 0.92 on overnight NO calls, and the comment above
            # explicitly describes the gate as covering nws_hourly only.
            if forecast_outcome == "NO" and is_same_day_mkt:
                no_sources = {o.source for o in forecasts if o.implied_outcome == "NO"}
                rt_no   = {s for s in no_sources if s in _REALTIME_SRCS}
                daily_no = {s for s in no_sources if s not in _REALTIME_SRCS}
                if (
                    rt_no and not daily_no and len(rt_no) < 2
                    and rt_no != frozenset({"hrrr"})  # HRRR alone is high-quality; exempt from gate
                ):
                    logging.info(
                        "Corroboration gate: lone real-time NO (%s) on same-day"
                        " %s %s with no daily-source agreement — suppressed",
                        rt_no, metric, ticker,
                    )
                    continue

            # Forecast consensus reached — select the most authoritative source
            # as the primary, rather than max-edge (which would elevate
            # weatherapi/open_meteo above NOAA/HRRR when their edge is higher).
            # weatherapi and open_meteo are corroboration-only; they should never
            # be the logged "source" that drove the trade.
            _SRC_PRIORITY: dict[str, int] = {
                "hrrr":       0,
                "noaa":       1,
                "nws_hourly": 2,
                "noaa_day2":  3,
                "noaa_day3":  4,
                "noaa_day4":  5,
                "noaa_day5":  6,
                "noaa_day6":  7,
                "noaa_day7":  8,
                "open_meteo": 9,
                "weatherapi": 10,
            }
            agreeing = [o for o in forecasts if o.implied_outcome == forecast_outcome]
            best = min(agreeing, key=lambda o: _SRC_PRIORITY.get(o.source, 99))
            best.corroborating_sources = [
                o.source for o in agreeing if o.source != best.source
            ]
            consensus_opps.append(best)
        numeric_opps = consensus_opps

    # ---- Log all pre-gate weather forecasts for calibration data -----------
    # Must happen BEFORE _filter_weather_opportunities so that opportunities
    # suppressed by the edge gate are still captured.  This feeds the backtest
    # in scripts/backtest_source_accuracy.py with the full population of
    # source forecasts, not just post-gate survivors.
    if numeric_opps:
        opp_log.log_raw_forecasts(numeric_opps)

    # ---- Weather-specific edge and time-to-close gates ---------------------
    # Applied after OWM consensus so both NOAA and OWM sources are already
    # merged / deduplicated before the per-source quality gates fire.
    if numeric_opps:
        before = len(numeric_opps)
        numeric_opps = _filter_weather_opportunities(
            numeric_opps, markets, hrrr_hourly_highs,
            observed_values=_obs_value if numeric_opps else None,
            fc_low_by_metric=_fc_low_by_metric if numeric_opps else None,
        )
        dropped = before - len(numeric_opps)
        if dropped:
            logging.info(
                "Weather gate: suppressed %d temp opportunity(ies) "
                "(forecast edge below per-source threshold, HRRR spread ≥ %.0f°F, or observed NO outside %.0fh window).",
                dropped, HRRR_MAX_SPREAD_F, TEMP_OBSERVED_MAX_HOURS,
            )

    # ---- Release window gate (Option D) ------------------------------------
    # Block BLS, FRED, and EIA numeric opportunities when we are outside the
    # configured window after their scheduled release.  Outside the window the
    # published value is already fully reflected in market prices; trading on
    # it offers no information edge.  Metrics with no known schedule (weather,
    # crypto, forex) always pass through unchanged.
    if RELEASE_WINDOW_MINUTES > 0 and numeric_opps:
        now_utc = datetime.now(timezone.utc)
        before_rw = len(numeric_opps)
        passed_rw: list[NumericOpportunity] = []
        for opp in numeric_opps:
            # These sources are leading indicators / proxies, not the actual
            # scheduled releases, so they are exempt from the release window gate.
            #   cme_fedwatch — continuous FOMC probability signal
            #   adp           — Wednesday pre-signal for Friday BLS NFP
            #   chicago_pmi   — last-biz-day pre-signal for ISM Manufacturing
            if opp.source in ("cme_fedwatch", "adp", "chicago_pmi", "yahoo_wti_futures"):
                passed_rw.append(opp)
                continue
            if is_within_release_window(opp.metric, now_utc, RELEASE_WINDOW_MINUTES):
                passed_rw.append(opp)
            else:
                nxt = next_release(opp.metric, now_utc)
                nxt_str = nxt.strftime("%Y-%m-%d %H:%M UTC") if nxt else "unknown"
                logging.info(
                    "Release gate: suppressed %s [%s] — outside %d-min window"
                    " (next release: %s)",
                    opp.market_ticker, opp.metric, RELEASE_WINDOW_MINUTES, nxt_str,
                )
        numeric_opps = passed_rw
        dropped_rw = before_rw - len(numeric_opps)
        if dropped_rw:
            logging.info(
                "Release gate: suppressed %d opportunity(ies) outside %d-min release window.",
                dropped_rw, RELEASE_WINDOW_MINUTES,
            )

    # ---- Crypto daily-close gate -------------------------------------------
    # Daily-close crypto markets resolve at a fixed settlement price (e.g.
    # 5 PM ET for KXBTCD).  Binance returns the live intraday spot price,
    # which can differ significantly from the close price when hours remain.
    # Only trade these within CRYPTO_DAILY_CLOSE_HOURS of the market's
    # close_time.  15-minute markets (KXBTC15M, KXETH15M, …) are always
    # exempt — they resolve so frequently the spot price IS the close price.
    #
    # Daily-close prefixes (no "15M" suffix): KXBTCD, KXDOGE, KXADA,
    # KXAVAX, KXLINK.  Note: KXDOGE15M / KXDOGE are distinguished by
    # checking that the ticker segment after the prefix is a date (starts
    # with a digit), not "15M".
    _DAILY_CLOSE_PREFIXES: tuple[str, ...] = (
        "KXBTCD", "KXDOGE", "KXADA", "KXAVAX", "KXLINK", "KXBNB",
    )
    _EXEMPT_15M_PREFIXES: tuple[str, ...] = (
        "KXDOGE15M", "KXADA15M", "KXAVAX15M", "KXLINK15M", "KXBNB15M",
    )

    def _is_daily_close_crypto(ticker: str) -> bool:
        """Return True if ticker is a daily-close crypto market (not a 15M market)."""
        if ticker.startswith(_EXEMPT_15M_PREFIXES):
            return False
        return ticker.startswith(_DAILY_CLOSE_PREFIXES)

    if CRYPTO_DAILY_CLOSE_HOURS > 0 and numeric_opps:
        close_dt_index: dict[str, str | None] = {
            m.get("ticker", ""): m.get("close_time") or m.get("expiration_time")
            for m in (markets or [])
        }
        now_utc = datetime.now(timezone.utc)
        before_cc = len(numeric_opps)
        passed_cc: list[NumericOpportunity] = []
        for opp in numeric_opps:
            if not _is_daily_close_crypto(opp.market_ticker):
                passed_cc.append(opp)
                continue
            ct_str = close_dt_index.get(opp.market_ticker)
            if ct_str:
                try:
                    close_dt = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
                    hours_to_close = (close_dt - now_utc).total_seconds() / 3600
                    if hours_to_close <= CRYPTO_DAILY_CLOSE_HOURS:
                        passed_cc.append(opp)
                    else:
                        logging.debug(
                            "Crypto close gate: dropped %s — %.1fh until close"
                            " (gate=%.1fh)",
                            opp.market_ticker, hours_to_close, CRYPTO_DAILY_CLOSE_HOURS,
                        )
                except (ValueError, AttributeError):
                    passed_cc.append(opp)  # unparseable close_time: allow through
            else:
                passed_cc.append(opp)  # no close_time: allow through
        numeric_opps = passed_cc
        dropped_cc = before_cc - len(numeric_opps)
        if dropped_cc:
            logging.info(
                "Crypto close gate: suppressed %d daily-close crypto opportunity(ies)"
                " (>%.1fh until close).",
                dropped_cc, CRYPTO_DAILY_CLOSE_HOURS,
            )

    # ---- Forex daily-fix close gate ----------------------------------------
    # KXEURUSD and KXUSDJPY markets resolve at the ECB daily fixing (~10 AM ET).
    # The Frankfurter rate is not a forward-looking signal — it IS the fixing
    # once published.  Only surface forex opportunities within FOREX_CLOSE_HOURS
    # of the market's close_time so we don't act on yesterday's stale rate that
    # survived the staleness gate (e.g. if FOREX_MAX_STALE_DAYS > 0).
    if FOREX_CLOSE_HOURS > 0 and numeric_opps:
        fx_close_dt_index: dict[str, str | None] = {
            m.get("ticker", ""): m.get("close_time") or m.get("expiration_time")
            for m in (markets or [])
        }
        now_utc = datetime.now(timezone.utc)
        before_fx = len(numeric_opps)
        passed_fx: list[NumericOpportunity] = []
        for opp in numeric_opps:
            if not opp.market_ticker.startswith(("KXEURUSD", "KXUSDJPY")):
                passed_fx.append(opp)
                continue
            ct_str = fx_close_dt_index.get(opp.market_ticker)
            if ct_str:
                try:
                    close_dt = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
                    hours_to_close = (close_dt - now_utc).total_seconds() / 3600
                    if hours_to_close <= FOREX_CLOSE_HOURS:
                        passed_fx.append(opp)
                    else:
                        logging.debug(
                            "Forex close gate: dropped %s — %.1fh until close"
                            " (gate=%.1fh)",
                            opp.market_ticker, hours_to_close, FOREX_CLOSE_HOURS,
                        )
                except (ValueError, AttributeError):
                    passed_fx.append(opp)
            else:
                passed_fx.append(opp)
        numeric_opps = passed_fx
        dropped_fx = before_fx - len(numeric_opps)
        if dropped_fx:
            logging.info(
                "Forex close gate: suppressed %d forex opportunity(ies)"
                " (>%.1fh until close).",
                dropped_fx, FOREX_CLOSE_HOURS,
            )

    # ---- External forecast matching (Polymarket, Metaculus, Manifold) ------
    poly_opps: list[PolyOpportunity] = []

    if isinstance(poly_result, Exception):
        logging.error("Polymarket fetch error: %s", poly_result)
    elif poly_result and markets:
        opps = match_poly_to_kalshi(poly_result, markets)
        if opps:
            logging.info("Polymarket: %d divergence(s) found vs Kalshi.", len(opps))
        poly_opps.extend(opps)

    if isinstance(metaculus_result, Exception):
        logging.error("Metaculus fetch error: %s", metaculus_result)
    elif metaculus_result and markets:
        opps = match_metaculus_to_kalshi(metaculus_result, markets)
        if opps:
            logging.info("Metaculus: %d divergence(s) found vs Kalshi.", len(opps))
        poly_opps.extend(opps)

    if isinstance(manifold_result, Exception):
        logging.error("Manifold fetch error: %s", manifold_result)
    elif manifold_result and markets:
        opps = match_manifold_to_kalshi(manifold_result, markets)
        if opps:
            logging.info("Manifold: %d divergence(s) found vs Kalshi.", len(opps))
        poly_opps.extend(opps)

    if isinstance(predictit_result, Exception):
        logging.error("PredictIt fetch error: %s", predictit_result)
    elif predictit_result and markets:
        opps = match_predictit_to_kalshi(predictit_result, markets)
        if opps:
            logging.info("PredictIt: %d divergence(s) found vs Kalshi.", len(opps))
        poly_opps.extend(opps)

    # ---- Block poly matching on numeric price-series tickers ---------------
    # Crypto, forex, and BLS/FRED price markets (KXBTCD, KXETH15M, KXEURUSD,
    # KXCPI, etc.) are purely quantitative and Polymarket/Metaculus/Manifold
    # have no corresponding markets.  Any text-similarity match is spurious
    # and the large edge produced by comparing a 5%/99% event probability to
    # a 40¢/60¢ crypto binary generates dangerously high scores.
    if poly_opps:
        _n_before_series_gate = len(poly_opps)
        poly_opps = [
            o for o in poly_opps
            if not any(o.kalshi_ticker.startswith(s) for s in NUMERIC_SERIES)
        ]
        _n_series_blocked = _n_before_series_gate - len(poly_opps)
        if _n_series_blocked:
            logging.info(
                "Poly series gate: blocked %d match(es) on numeric price-series tickers.",
                _n_series_blocked,
            )

    # ---- Manifold quality gates --------------------------------------------
    # Two-tier filter applied to all Manifold opportunities:
    #
    #   Hard cap  (MANI_HARD_CAP_DIVERGENCE, default 0.40):
    #     Always block — even with corroboration — because a 40%+ divergence
    #     from Manifold is implausibly large and almost always a frozen/stale
    #     market, not a real signal.
    #
    #   Consensus gate (MANI_MAX_SOLO_DIVERGENCE, default 0.50):
    #     Block Manifold-only divergences above 50% unless Polymarket or
    #     Metaculus independently shows an opportunity for the same ticker.
    if poly_opps:
        corroborated: set[str] = {
            o.kalshi_ticker for o in poly_opps
            if o.source in ("polymarket", "metaculus")
        }
        gated: list[PolyOpportunity] = []
        hard_blocked = 0
        gate_blocked = 0
        for opp in poly_opps:
            if opp.source == "manifold":
                if MANI_HARD_CAP_DIVERGENCE > 0 and opp.divergence > MANI_HARD_CAP_DIVERGENCE:
                    logging.info(
                        "Manifold hard cap: divergence %.0f%% > cap %.0f%% — blocked %s.",
                        opp.divergence * 100, MANI_HARD_CAP_DIVERGENCE * 100,
                        opp.kalshi_ticker,
                    )
                    hard_blocked += 1
                    continue
                if (
                    MANI_MAX_SOLO_DIVERGENCE > 0
                    and opp.divergence > MANI_MAX_SOLO_DIVERGENCE
                    and opp.kalshi_ticker not in corroborated
                ):
                    logging.info(
                        "Consensus gate: Manifold-only %.0f%% divergence blocked on %s "
                        "— no corroborating real-money source.",
                        opp.divergence * 100, opp.kalshi_ticker,
                    )
                    gate_blocked += 1
                    continue
            gated.append(opp)
        if hard_blocked:
            logging.info("Manifold hard cap: blocked %d opportunity(ies).", hard_blocked)
        if gate_blocked:
            logging.info("Consensus gate: blocked %d Manifold-only opportunity(ies).", gate_blocked)
        poly_opps = gated

    # ---- snapshot opps for counter-signal exits ----------------------------
    # Taken here — after all quality gates but before entry filters (liquidity,
    # contrarian, position).  Counter-signal exits are based on whether the
    # underlying data has flipped, not on whether we'd enter a new trade today.
    _exit_numeric_opps: list[NumericOpportunity] = list(numeric_opps)
    _exit_poly_opps:    list[PolyOpportunity]    = list(poly_opps)

    # ---- orderbook enrichment (only matched tickers, concurrent) -----------
    pre_liquidity = len(text_opps) + len(numeric_opps) + len(poly_opps)
    if pre_liquidity == 0:
        logging.info("No opportunities surfaced this cycle.")
        if ledger is not None:
            await ledger.refresh_and_write(
                session,
                numeric_opps=_exit_numeric_opps,
                poly_opps=_exit_poly_opps,
            )
        return

    unique_tickers = list({
        opp.market_ticker for opp in text_opps
    } | {
        opp.market_ticker for opp in numeric_opps
    } | {
        opp.kalshi_ticker for opp in poly_opps
    })

    # ---- build ticker_detail from the already-fetched market cache ----------
    # The market list endpoint (fetch_all_markets) returns yes_bid/yes_ask
    # alongside all other market fields, so we can satisfy the orderbook
    # enrichment step for free — no extra API calls needed for tickers
    # already in the cache.  Any ticker NOT in the cache (e.g. a general-
    # market poly match) falls back to the individual detail endpoint.
    _market_by_ticker: dict[str, dict] = {
        m["ticker"]: m for m in markets if m.get("ticker")
    }
    ticker_detail: dict[str, dict] = {}
    for t in unique_tickers:
        if t in _market_by_ticker:
            ticker_detail[t] = _market_by_ticker[t]

    missing_tickers = [t for t in unique_tickers if t not in ticker_detail]
    if missing_tickers:
        # Serial fetching with a 0.5s gap between requests to stay within
        # Kalshi rate limits.  Only called for tickers not in the market cache
        # (typically poly matches to general markets).
        logging.info(
            "Fetching live orderbook for %d ticker(s) not in market cache …",
            len(missing_tickers),
        )
        _detail_sem = asyncio.Semaphore(1)

        async def _fetch_detail(ticker: str):
            async with _detail_sem:
                result = await fetch_market_detail(session, ticker)
                await asyncio.sleep(0.5)
                return result

        detail_results = await asyncio.gather(
            *[_fetch_detail(t) for t in missing_tickers],
            return_exceptions=True,
        )
        for ticker, result in zip(missing_tickers, detail_results):
            if isinstance(result, Exception):
                logging.warning("Orderbook fetch failed for %s: %s", ticker, result)
            elif result:
                ticker_detail[ticker] = result

    logging.debug(
        "Orderbook enrichment: %d/%d ticker(s) resolved (cache=%d, detail=%d).",
        len(ticker_detail), len(unique_tickers),
        len(unique_tickers) - len(missing_tickers),
        sum(1 for t in missing_tickers if t in ticker_detail),
    )

    # ---- liquidity filter --------------------------------------------------
    def _passes_liquidity(ticker: str) -> bool:
        detail = ticker_detail.get(ticker)
        if not detail:
            return True  # no data → don't discard; surface with "(unavailable)"
        bid = detail.get("yes_bid")
        ask = detail.get("yes_ask")
        vol = detail.get("volume") or 0
        if LIQUIDITY_MIN_VOLUME > 0 and vol < LIQUIDITY_MIN_VOLUME:
            return False
        if LIQUIDITY_MAX_SPREAD > 0 and bid is not None and ask is not None:
            if (ask - bid) > LIQUIDITY_MAX_SPREAD:
                return False
        return True

    text_opps    = [o for o in text_opps    if _passes_liquidity(o.market_ticker)]
    numeric_opps = [o for o in numeric_opps if _passes_liquidity(o.market_ticker)]
    poly_opps    = [o for o in poly_opps    if _passes_liquidity(o.kalshi_ticker)]

    filtered = pre_liquidity - len(text_opps) - len(numeric_opps) - len(poly_opps)
    if filtered:
        logging.info(
            "Liquidity filter: dropped %d opportunity(ies) (spread >%d¢ or vol <%d).",
            filtered, LIQUIDITY_MAX_SPREAD, LIQUIDITY_MIN_VOLUME,
        )

    # ---- Contrarian pricing filter (Option E) -------------------------------
    # Only allow numeric opportunities where the market price significantly
    # *disagrees* with the data signal — i.e. the entry cost is cheap because
    # the market hasn't priced in our data yet.
    #
    # For a YES signal:  entry cost = yes_ask.  If yes_ask is already high,
    #   the market agrees with us and the edge is gone.
    # For a NO signal:   entry cost = 100 − yes_bid.  If yes_bid is already
    #   low (NO is cheap), the market agrees and the edge is gone.
    #
    # UNKNOWN direction markets and opportunities with no orderbook data are
    # passed through unchanged.
    #
    # Observed-data sources (noaa_observed, nws_climo, nws_alert) are exempt:
    # the observed station reading is ground truth — a "below 76°F" contract at
    # 93¢ still yields a real 7¢ profit per contract once the day's high is
    # confirmed locked.  The market floor gate (8¢ minimum on our side) already
    # prevents entry above ~92¢, providing the upper bound.
    _CONTRARIAN_EXEMPT = frozenset({"noaa_observed", "metar", "nws_climo", "nws_alert"})
    if CONTRARIAN_MAX_ENTRY_CENTS > 0 and numeric_opps:
        before_ct = len(numeric_opps)
        passed_ct: list[NumericOpportunity] = []
        for opp in numeric_opps:
            if opp.implied_outcome == "UNKNOWN" or opp.source in _CONTRARIAN_EXEMPT:
                passed_ct.append(opp)
                continue
            detail = ticker_detail.get(opp.market_ticker)
            if not detail:
                passed_ct.append(opp)  # no orderbook data → can't apply filter
                continue
            bid = detail.get("yes_bid")
            ask = detail.get("yes_ask")
            if bid is None or ask is None:
                passed_ct.append(opp)
                continue
            if opp.implied_outcome == "YES":
                entry_cost = ask
            else:  # "NO"
                entry_cost = 100 - bid
            if entry_cost <= CONTRARIAN_MAX_ENTRY_CENTS:
                passed_ct.append(opp)
            else:
                logging.debug(
                    "Contrarian gate: suppressed %s [%s→%s] — entry cost %d¢ > %d¢"
                    " (market agrees; no edge remaining)",
                    opp.market_ticker, opp.metric, opp.implied_outcome,
                    entry_cost, CONTRARIAN_MAX_ENTRY_CENTS,
                )
        numeric_opps = passed_ct
        dropped_ct = before_ct - len(numeric_opps)
        if dropped_ct:
            logging.info(
                "Contrarian gate: suppressed %d opportunity(ies) where market"
                " already agrees with signal (entry cost > %d¢).",
                dropped_ct, CONTRARIAN_MAX_ENTRY_CENTS,
            )

    # ---- Market minimum price gate -----------------------------------------
    # Complement to the contrarian gate above.  When the market prices our
    # side below MARKET_MIN_PRICE_CENTS, the collective orderbook is
    # near-certain we are wrong.  The most common cause is a data-source
    # station mismatch: e.g. NOAA observed a high of 68°F from the wrong NWS
    # station while Kalshi resolves against a different official station still
    # well below the strike, so market participants price YES at 1¢ while our
    # model says p=1.0.  A <3¢ market is a stronger signal than any single
    # data feed and should always override our model.
    if MARKET_MIN_PRICE_CENTS > 0 and numeric_opps:
        before_mp = len(numeric_opps)
        passed_mp: list[NumericOpportunity] = []
        for opp in numeric_opps:
            if opp.implied_outcome == "UNKNOWN":
                passed_mp.append(opp)
                continue
            detail = ticker_detail.get(opp.market_ticker)
            if not detail:
                passed_mp.append(opp)
                continue
            bid = detail.get("yes_bid")
            ask = detail.get("yes_ask")
            if bid is None or ask is None:
                passed_mp.append(opp)
                continue
            if opp.implied_outcome == "YES":
                our_side_price = ask
            else:  # "NO"
                our_side_price = 100 - bid
            if our_side_price >= MARKET_MIN_PRICE_CENTS:
                passed_mp.append(opp)
            else:
                logging.info(
                    "Market floor gate: suppressed %s [→%s] — market prices"
                    " our side at %d¢ (< %d¢ floor); collective orderbook"
                    " near-certain against signal.",
                    opp.market_ticker, opp.implied_outcome,
                    our_side_price, MARKET_MIN_PRICE_CENTS,
                )
        numeric_opps = passed_mp
        dropped_mp = before_mp - len(numeric_opps)
        if dropped_mp:
            logging.info(
                "Market floor gate: suppressed %d opportunity(ies) where"
                " market prices our side below %d¢.",
                dropped_mp, MARKET_MIN_PRICE_CENTS,
            )

    # Max-entry-price gate for noaa_observed direction=over/between YES signals.
    # Entering when yes_ask is very high (market already agrees) means we risk
    # a large loss for a tiny gain.  At 90¢ entry the risk:reward is 9:1 against
    # us — the slightest station-mismatch or data error wipes the position.
    # noaa_observed is exempt from the contrarian gate for direction=under
    # (locked-NO confirmed observations should be entered even at 93¢), but for
    # direction=over and direction=between the same argument does NOT hold: if the
    # observation is wrong, we lose the full entry price.  Canonical failures: #194
    # and #195 entered at 100¢ YES and both settled at 0¢ (total −200¢).
    # direction=between is included because the station-mismatch risk is identical:
    # the official settlement station could read outside the band even when our
    # ASOS station is inside it.
    _OBS_OVER_SRCS = frozenset({"noaa_observed", "metar"})
    if NOAA_OBS_OVER_MAX_ENTRY_CENTS > 0 and numeric_opps:
        before_oo = len(numeric_opps)
        passed_oo: list[NumericOpportunity] = []
        for opp in numeric_opps:
            if (
                opp.source in _OBS_OVER_SRCS
                and opp.direction in ("over", "between")
                and opp.implied_outcome == "YES"
            ):
                detail = ticker_detail.get(opp.market_ticker)
                if detail:
                    ask = detail.get("yes_ask")
                    if ask is not None and ask > NOAA_OBS_OVER_MAX_ENTRY_CENTS:
                        logging.info(
                            "Obs-over ceiling: suppressed %s %s"
                            " — yes_ask %d¢ > %d¢ ceiling"
                            " (risk:reward too poor for observed-%s signal)",
                            opp.source, opp.market_ticker,
                            ask, NOAA_OBS_OVER_MAX_ENTRY_CENTS,
                            opp.direction,
                        )
                        continue
            passed_oo.append(opp)
        numeric_opps = passed_oo
        dropped_oo = before_oo - len(numeric_opps)
        if dropped_oo:
            logging.info(
                "Obs-over ceiling: blocked %d opportunity(ies)"
                " (noaa_observed/metar direction=over/between YES with yes_ask > %d¢).",
                dropped_oo, NOAA_OBS_OVER_MAX_ENTRY_CENTS,
            )

    # Same gate for external-forecast (poly) opportunities.
    # A Polymarket/Manifold divergence can also produce a 1¢ trade when an
    # external platform sits at 99% while Kalshi is at 1¢ — the exact same
    # station-mismatch / information-asymmetry scenario applies.
    if MARKET_MIN_PRICE_CENTS > 0 and poly_opps:
        before_pmp = len(poly_opps)
        passed_pmp: list[PolyOpportunity] = []
        for opp in poly_opps:
            detail = ticker_detail.get(opp.kalshi_ticker)
            if not detail:
                passed_pmp.append(opp)
                continue
            bid = detail.get("yes_bid")
            ask = detail.get("yes_ask")
            if bid is None or ask is None:
                passed_pmp.append(opp)
                continue
            our_side_price = ask if opp.implied_side == "yes" else (100 - bid)
            if our_side_price >= MARKET_MIN_PRICE_CENTS:
                passed_pmp.append(opp)
            else:
                logging.info(
                    "Market floor gate [poly]: suppressed %s [%s→%s] —"
                    " Kalshi prices our side at %d¢ (< %d¢ floor).",
                    opp.kalshi_ticker, opp.source, opp.implied_side,
                    our_side_price, MARKET_MIN_PRICE_CENTS,
                )
        poly_opps = passed_pmp
        dropped_pmp = before_pmp - len(poly_opps)
        if dropped_pmp:
            logging.info(
                "Market floor gate [poly]: suppressed %d external-forecast"
                " opportunity(ies) below %d¢.",
                dropped_pmp, MARKET_MIN_PRICE_CENTS,
            )

    # ---- score and sort by composite score (highest = most actionable) ------
    scored_text = [
        (
            score_text_opportunity(
                opp,
                ticker_detail.get(opp.market_ticker),
                _days_to_close(opp.market_ticker, ticker_detail),
            ),
            opp,
        )
        for opp in text_opps
    ]
    scored_numeric = [
        (
            score_numeric_opportunity(
                opp,
                ticker_detail.get(opp.market_ticker),
                _days_to_close(opp.market_ticker, ticker_detail),
            ),
            opp,
        )
        for opp in numeric_opps
    ]
    # Poly score: same composite formula as numeric (spread/uncertainty/temporal/edge)
    # so scores are directly comparable across all opportunity types.
    scored_poly: list[tuple[float, PolyOpportunity]] = sorted(
        [
            (
                score_poly_opportunity(
                    o,
                    ticker_detail.get(o.kalshi_ticker),
                    _days_to_close(o.kalshi_ticker, ticker_detail),
                ),
                o,
            )
            for o in poly_opps
        ],
        key=lambda t: t[0],
        reverse=True,
    )

    scored_text.sort(key=lambda t: t[0], reverse=True)
    scored_numeric.sort(key=lambda t: t[0], reverse=True)

    # ---- cross-cycle cooldown filter ---------------------------------------
    suppressed = opp_log.recently_surfaced_pairs(OPPORTUNITY_COOLDOWN_MINUTES)

    def _signal_key_text(opp: Opportunity) -> str:
        return (opp.matched_terms[0] if opp.matched_terms else opp.topic).lower()

    scored_text = [
        (s, o) for s, o in scored_text
        if (o.market_ticker, _signal_key_text(o)) not in suppressed
    ]
    scored_numeric = [
        (s, o) for s, o in scored_numeric
        if (o.market_ticker, o.metric) not in suppressed
    ]
    scored_poly = [
        (s, o) for s, o in scored_poly
        if (o.kalshi_ticker, f"{o.source}:{o.poly_market_id}") not in suppressed
    ]

    if suppressed:
        logging.info(
            "Cooldown filter: %d unique (ticker, signal) pair(s) suppressed "
            "(already surfaced within last %d min).",
            len(suppressed), OPPORTUNITY_COOLDOWN_MINUTES,
        )

    # ---- position filter ---------------------------------------------------
    # In dry-run mode the Kalshi API never returns simulated positions, so we
    # supplement with the set of tickers that have an open (unsettled, unexited)
    # trade in the local DB.  This prevents the bot from re-entering the same
    # market every poll cycle while a dry-run position is still live.
    dry_run_held: set[str] = set()
    if TRADE_DRY_RUN and ledger is not None:
        set_drawdown_factor(ledger.current_drawdown_factor())
        for src, stats in ledger.source_performance_summary().items():
            if stats["win_rate"] < 0.25 and stats["net_pnl_cents"] < 0:
                logging.warning(
                    "Source %s: last-20 win rate %.0f%%, P&L $%.2f — consider blocking it.",
                    src, stats["win_rate"] * 100, stats["net_pnl_cents"] / 100,
                )
        dry_run_held = ledger.open_tickers()
        if dry_run_held:
            logging.info(
                "Position dedup: %d open dry-run position(s) will block re-entry.",
                len(dry_run_held),
            )
        if EXIT_REENTRY_COOLDOWN_MINUTES > 0:
            recently_exited = ledger.recently_exited_tickers(EXIT_REENTRY_COOLDOWN_MINUTES)
            if recently_exited:
                logging.info(
                    "Re-entry cooldown: %d ticker(s) blocked for %d min"
                    " after stop_loss/trailing_stop: %s",
                    len(recently_exited), EXIT_REENTRY_COOLDOWN_MINUTES,
                    ", ".join(sorted(recently_exited)),
                )
            dry_run_held |= recently_exited

        # forecast_no: block same-day re-entry for any exit reason (profit-take
        # included).  The `recently_exited_tickers` cooldown only covers stop-loss
        # exits, so profit-taken forecast_no positions could be re-entered at a
        # higher price with less remaining upside on the very next poll.
        _fno_exited_today: set[str] = ledger.forecast_no_exited_today()
        if _fno_exited_today:
            logging.info(
                "ForecastNO today-exited: %d ticker(s) blocked from same-day re-entry: %s",
                len(_fno_exited_today), ", ".join(sorted(_fno_exited_today)),
            )
    else:
        _fno_exited_today = set()

    if POSITION_SKIP_CONTRACTS > 0 or dry_run_held:
        before_pos = len(scored_text) + len(scored_numeric) + len(scored_poly)
        scored_text = [
            (s, o) for s, o in scored_text
            if abs(positions.get(o.market_ticker, {}).get("position", 0)) < POSITION_SKIP_CONTRACTS
            and o.market_ticker not in dry_run_held
        ]
        scored_numeric = [
            (s, o) for s, o in scored_numeric
            if abs(positions.get(o.market_ticker, {}).get("position", 0)) < POSITION_SKIP_CONTRACTS
            and o.market_ticker not in dry_run_held
        ]
        scored_poly = [
            (s, o) for s, o in scored_poly
            if abs(positions.get(o.kalshi_ticker, {}).get("position", 0)) < POSITION_SKIP_CONTRACTS
            and o.kalshi_ticker not in dry_run_held
        ]
        pos_dropped = before_pos - len(scored_text) - len(scored_numeric) - len(scored_poly)
        if pos_dropped:
            logging.info(
                "Position filter: dropped %d opportunity(ies) "
                "(existing position >= %d contracts or open dry-run trade).",
                pos_dropped, POSITION_SKIP_CONTRACTS,
            )

    # ---- Temperature position concentration cap ----------------------------
    # When a weather system triggers many simultaneous signals (e.g. 20+ KXHIGH
    # opportunities across 5 cities × 4 strikes each), don't flood the portfolio
    # with correlated positions.  Count currently-open temperature tickers and
    # skip new ones once TEMP_MAX_CONCURRENT_POSITIONS is reached.
    if TEMP_MAX_CONCURRENT_POSITIONS > 0:
        if TRADE_DRY_RUN and ledger is not None:
            temp_open = sum(1 for t in dry_run_held if t.startswith("KXHIGH"))
        else:
            temp_open = sum(
                1 for t, info in positions.items()
                if t.startswith("KXHIGH") and info.get("position", 0) != 0
            )
        if temp_open >= TEMP_MAX_CONCURRENT_POSITIONS:
            before_temp = len(scored_numeric)
            scored_numeric = [
                (s, o) for s, o in scored_numeric
                if not o.market_ticker.startswith("KXHIGH")
            ]
            temp_dropped = before_temp - len(scored_numeric)
            if temp_dropped:
                logging.info(
                    "Temp concentration cap: skipped %d temperature opportunity(ies) "
                    "(already hold %d/%d KXHIGH positions).",
                    temp_dropped, temp_open, TEMP_MAX_CONCURRENT_POSITIONS,
                )

    # ---- band-pass arbitrage must run even when all normal opportunities fail
    # the score gate.  The second early-return below (total == 0) was previously
    # skipping both band_arb and the end-of-poll ledger refresh, so those blocks
    # are now hoisted here to run unconditionally before either return path.
    # (The first early-return at line ~2472 still short-circuits before
    # enrichment when there are literally zero opportunities of any kind.)

    # ---- band-pass arbitrage (METAR observed high vs. KXHIGH band markets) --
    # Duplicate of the identical block lower in _poll; this copy runs even when
    # all text/numeric/poly signals fail the score gate.
    # Guard: metar_result may be an Exception if the fetch failed; iterating over
    # an Exception raises TypeError and would crash the entire _poll().
    _band_arb_obs_early: dict[str, float] = {}
    _band_arb_obs_dates: dict[str, date] = {}
    if not isinstance(metar_result, Exception) and metar_result:
        for _dp in metar_result:
            if _dp.metric.startswith(("temp_high", "temp_low")):
                _band_arb_obs_early[_dp.metric] = _dp.value
                _ld = (_dp.metadata or {}).get("local_date")
                if _ld:
                    _band_arb_obs_dates[_dp.metric] = date.fromisoformat(_ld)
    _band_arb_noaa_obs_early: dict[str, float] = {
        dp.metric: dp.value
        for dp in data_points
        if dp.source == "noaa_observed" and dp.metric.startswith(("temp_high", "temp_low"))
    }
    if _band_arb_obs_early:
        _early_band_arb_signals = find_band_arbs(
            markets,
            _band_arb_obs_early,
            noaa_obs_values=_band_arb_noaa_obs_early or None,
            obs_dates=_band_arb_obs_dates or None,
        )
        if _early_band_arb_signals:
            logging.info(
                "Band arb (pre-score-gate): %d signal(s) found (BAND_ARB_EXECUTION_ENABLED=%s).",
                len(_early_band_arb_signals), BAND_ARB_EXECUTION_ENABLED,
            )
            for _barb in _early_band_arb_signals:
                if _barb.ticker in dry_run_held:
                    logging.info(
                        "BandArb skip: %s in re-entry cooldown / held position.",
                        _barb.ticker,
                    )
                    continue
                await executor.maybe_trade_band_arb(session, _barb)

    # ---- forecast-driven NO signals (early, before METAR confirmation) ------
    if FORECAST_NO_ENABLED and data_points:
        _forecast_no_signals = find_forecast_nos(markets, data_points)
        if _forecast_no_signals:
            opp_log.log_forecast_no_sources(_forecast_no_signals)
            logging.info(
                "ForecastNO: %d signal(s) found.", len(_forecast_no_signals),
            )
            # Build set of city-date prefixes (e.g. "KXHIGHTSFO-26APR14") that
            # already have a held or today-exited forecast_no position.  Adjacent
            # strike bands in the same city are correlated — holding multiple
            # bands doubles exposure with no diversification benefit.
            _held_prefixes: set[str] = {
                "-".join(t.split("-")[:2])
                for t in (dry_run_held | _fno_exited_today)
                if "KXHIGH" in t or "KXLOWT" in t
            }
            # Process highest-edge signal first so the best-conviction band wins
            # when multiple bands from the same city qualify simultaneously.
            for _fno in sorted(_forecast_no_signals, key=lambda s: -s.min_edge_f):
                _fno_prefix = "-".join(_fno.ticker.split("-")[:2])
                if _fno.ticker in dry_run_held:
                    logging.info(
                        "ForecastNO skip: %s — open position held.", _fno.ticker,
                    )
                    continue
                if _fno.ticker in _fno_exited_today:
                    logging.info(
                        "ForecastNO skip: %s — already traded today (same-day re-entry blocked).",
                        _fno.ticker,
                    )
                    continue
                if _fno_prefix in _held_prefixes:
                    logging.info(
                        "ForecastNO skip: %s — city-date %s already has a position.",
                        _fno.ticker, _fno_prefix,
                    )
                    continue
                await executor.maybe_trade_forecast_no(session, _fno)
                # Block further signals for this city-date within the same poll
                _held_prefixes.add(_fno_prefix)

    # ---- report + log ------------------------------------------------------
    total = len(scored_text) + len(scored_numeric) + len(scored_poly)
    if total == 0:
        logging.info("No new opportunities to surface this cycle.")
        executor.stats.log_summary()
        executor.stats.reset()
        if ledger is not None:
            logging.info("DryRunLedger: refreshing overview (no-trade cycle).")
            await ledger.refresh_and_write(
                session,
                numeric_opps=_exit_numeric_opps,
                poly_opps=_exit_poly_opps,
            )
            logging.info("DryRunLedger: overview write complete.")
        return

    print(f"\n{_WIDE}")
    print(f"  MARKET DISCOVERY REPORT  —  {total} opportunity(ies)")
    print(f"  {portfolio_summary}")
    print(_WIDE)

    for idx, (score, opp) in enumerate(scored_text, 1):
        detail = ticker_detail.get(opp.market_ticker)
        dtc = _days_to_close(opp.market_ticker, ticker_detail)
        existing = positions.get(opp.market_ticker, {}).get("position", 0)
        _print_text_opportunity(idx, opp, detail, score, existing_position=existing)
        opp_log.log_text(opp, detail, score, dtc)
        await executor.maybe_trade_text(session, opp, detail, score)

    offset = len(scored_text)
    for idx, (score, opp) in enumerate(scored_numeric, offset + 1):
        detail = ticker_detail.get(opp.market_ticker)
        dtc = _days_to_close(opp.market_ticker, ticker_detail)
        existing = positions.get(opp.market_ticker, {}).get("position", 0)
        _print_numeric_opportunity(idx, opp, detail, score, existing_position=existing)
        opp_log.log_numeric(opp, detail, score, dtc)
        await executor.maybe_trade_numeric(session, opp, detail, score)

    offset += len(scored_numeric)
    for idx, (score, opp) in enumerate(scored_poly, offset + 1):
        detail = ticker_detail.get(opp.kalshi_ticker)
        dtc = _days_to_close(opp.kalshi_ticker, ticker_detail)
        existing = positions.get(opp.kalshi_ticker, {}).get("position", 0)
        _print_poly_opportunity(idx, opp, detail, score, existing_position=existing)
        opp_log.log_poly(opp, detail, score, dtc)
        if POLY_ENABLED:
            await executor.maybe_trade_poly_opportunity(session, opp, detail, score)

    # ---- spread detection (synthetic range positions) ----------------------
    # Run on the post-gate numeric opportunities so both legs have already
    # passed all quality filters (ensemble spread, release window, liquidity).
    spread_opps: list[SpreadOpportunity] = find_spread_opportunities(
        numeric_opps, ticker_detail
    )
    if spread_opps:
        logging.info("Spread detector: %d synthetic range pair(s) found.", len(spread_opps))
        for spread in spread_opps:
            # Score the spread as the minimum of the two leg scores — the weaker
            # leg is the binding constraint on edge quality.
            score_lo = score_numeric_opportunity(
                spread.leg_lo,
                ticker_detail.get(spread.leg_lo.market_ticker),
                _days_to_close(spread.leg_lo.market_ticker, ticker_detail),
            )
            score_hi = score_numeric_opportunity(
                spread.leg_hi,
                ticker_detail.get(spread.leg_hi.market_ticker),
                _days_to_close(spread.leg_hi.market_ticker, ticker_detail),
            )
            spread_score = min(score_lo, score_hi)
            await executor.maybe_trade_spread(session, spread, spread_score)

    # ---- combinatorial arbitrage detection ---------------------------------
    # Scans all parsed markets for monotonicity violations (prices not summing
    # correctly across same-underlying markets). Guaranteed risk-free profit
    # when found; no signal quality gate needed.
    arb_opps: list[ArbOpportunity] = find_arb_opportunities(markets, ticker_detail)
    if arb_opps:
        logging.info(
            "Arb detector: %d opportunity(ies) found (min_profit=%d¢).",
            len(arb_opps), ARB_MIN_PROFIT_CENTS,
        )
        for arb in arb_opps:
            await executor.execute_arb(session, arb)

    # ---- crossed-book arbitrage (YES_ask + NO_ask < 100¢) -------------------
    # A single market where the order book is crossed guarantees profit at
    # settlement regardless of outcome — no directional forecast needed.
    crossed_opps: list[CrossedBookArb] = find_crossed_book_opportunities(markets)
    if crossed_opps:
        logging.info(
            "Crossed-book arb: %d opportunity(ies) found (min_profit=%d¢).",
            len(crossed_opps), CROSSED_BOOK_MIN_PROFIT,
        )
        for _cb in crossed_opps:
            await executor.execute_crossed_book(session, _cb)

    # ---- series bracket arbitrage (sum of YES prices ≠ 100¢) ----------------
    # For mutually exclusive exhaustive "between" brackets (temperature markets),
    # sum(YES_ask) < 100 → buy all YES for guaranteed profit, or
    # sum(YES_bid) > 100 → buy all NO for guaranteed profit.
    bracket_opps: list[BracketSetArb] = find_bracket_set_opportunities(markets)
    if bracket_opps:
        logging.info(
            "Bracket arb: %d set(s) found (min_profit=%d¢, enabled=%s).",
            len(bracket_opps), BRACKET_ARB_MIN_PROFIT, BRACKET_ARB_ENABLED,
        )
        for _ba in bracket_opps:
            await executor.execute_bracket_set_arb(session, _ba)

    # (band_arb runs unconditionally earlier in _poll — see pre-score-gate block)

    print(_WIDE + "\n")

    # ---- trade funnel summary + reset --------------------------------------
    executor.stats.log_summary()
    executor.stats.reset()

    # ---- update live overview file (dry-run only) --------------------------
    if ledger is not None:
        await ledger.refresh_and_write(
            session,
            numeric_opps=_exit_numeric_opps,
            poly_opps=_exit_poly_opps,
        )

    # ---- populate fast-loop watchlist --------------------------------------
    # Identify cities within WATCH_THRESHOLD_F of a band ceiling so the fast
    # inner loop only refreshes series prices for cities that matter.
    global _near_threshold_cities, _last_noaa_obs, _last_noaa_obs_time
    _near_threshold_cities = set()
    for _wl_mkt in markets:
        _wl_parsed = parse_market(_wl_mkt)
        if _wl_parsed is None or not _wl_parsed.metric.startswith(("temp_high", "temp_low")):
            continue
        _wl_obs = _band_arb_obs_early.get(_wl_parsed.metric)
        if _wl_obs is None:
            continue
        _wl_ceil: float | None = None
        if _wl_parsed.direction == "between" and _wl_parsed.strike_hi is not None:
            _wl_ceil = _wl_parsed.strike_hi
        elif _wl_parsed.direction == "under" and _wl_parsed.strike is not None:
            _wl_ceil = _wl_parsed.strike
        if _wl_ceil is not None and abs(_wl_ceil - _wl_obs) <= WATCH_THRESHOLD_F:
            _near_threshold_cities.add(_wl_parsed.metric)
    _last_noaa_obs = _band_arb_noaa_obs_early.copy()
    _last_noaa_obs_time = time.monotonic()
    if _near_threshold_cities:
        logging.debug(
            "Fast-loop watchlist: %d near-threshold city(ies): %s",
            len(_near_threshold_cities), sorted(_near_threshold_cities),
        )


# ---------------------------------------------------------------------------
# Adaptive poll interval
# ---------------------------------------------------------------------------

def _adaptive_poll_interval(now_utc: datetime) -> int:
    """Return the sleep interval (seconds) appropriate for the current moment.

    Two windows trigger POLL_INTERVAL_FAST instead of POLL_INTERVAL_SECONDS:

    1. EIA/KXWTI release window
       Within POLL_INTERVAL_EIA_WINDOW_MINUTES of any EIA WTI or natgas
       release, KXWTI/KXNATGAS markets can reprice within 1–2 minutes.
       A fast poll captures the information edge before the market closes.

    2. Band-arb / METAR peak-heating window
       Between POLL_BAND_ARB_START_ET_HOUR and POLL_BAND_ARB_END_ET_HOUR
       (default 13:00–21:00 ET), METAR observed highs are most likely to
       cross a band threshold and trigger a locked-NO arb signal.
       Faster polling reduces the average delay from ~60s to ~20s.

    Outside these windows the base POLL_INTERVAL_SECONDS applies.
    POLL_INTERVAL_FAST is capped at POLL_INTERVAL_SECONDS so setting it
    larger than the base has no effect.
    """
    fast = min(POLL_INTERVAL_FAST, POLL_INTERVAL_SECONDS)

    # EIA release window
    if POLL_INTERVAL_EIA_WINDOW_MINUTES > 0 and fast < POLL_INTERVAL_SECONDS:
        if (
            is_within_release_window("eia_wti", now_utc, POLL_INTERVAL_EIA_WINDOW_MINUTES)
            or is_within_release_window("eia_natgas", now_utc, POLL_INTERVAL_EIA_WINDOW_MINUTES)
        ):
            logging.info(
                "Adaptive poll: EIA release window active — using %ds interval.",
                fast,
            )
            return fast

    # Band-arb / METAR peak-heating window
    if POLL_BAND_ARB_START_ET_HOUR < POLL_BAND_ARB_END_ET_HOUR and fast < POLL_INTERVAL_SECONDS:
        et_hour = now_utc.astimezone(_ET).hour
        if POLL_BAND_ARB_START_ET_HOUR <= et_hour < POLL_BAND_ARB_END_ET_HOUR:
            return fast

    return POLL_INTERVAL_SECONDS


# ---------------------------------------------------------------------------
# Fast inner loop
# ---------------------------------------------------------------------------

async def _fast_loop(
    session: aiohttp.ClientSession,
    executor: "TradeExecutor",
    ledger: "DryRunLedger | None" = None,
) -> None:
    """Lightweight band-arb check that runs between full poll cycles.

    Only fetches METAR (hits 30s cache most iterations) and refreshes KXHIGH
    series prices for cities that were within WATCH_THRESHOLD_F of a band
    ceiling in the last full cycle.  Uses cached NOAA observed values from
    the last full cycle for corroboration — NOAA updates every 5–60 min so a
    60s-stale value is always at least as fresh as NOAA's own cadence.
    """
    if not _near_threshold_cities:
        return

    # METAR fetch — usually returns from 30s cache, one HTTP call otherwise
    try:
        metar_result = await metar.fetch_city_forecasts(session)
    except Exception as exc:
        logging.debug("Fast loop: METAR fetch failed: %s", exc)
        return

    obs_values: dict[str, float] = {}
    _fast_obs_dates: dict[str, date] = {}
    for _dp in metar_result:
        if _dp.metric.startswith("temp_high"):
            obs_values[_dp.metric] = _dp.value
            _ld = (_dp.metadata or {}).get("local_date")
            if _ld:
                _fast_obs_dates[_dp.metric] = date.fromisoformat(_ld)
    if not obs_values:
        return

    # Refresh prices only for near-threshold city series
    _METRIC_TO_SERIES: dict[str, str] = {
        v: k for k, v in TICKER_TO_METRIC.items() if k.startswith("KXHIGH")
    }
    series_to_fetch = [
        _METRIC_TO_SERIES[m]
        for m in _near_threshold_cities
        if m in _METRIC_TO_SERIES
    ]
    if not series_to_fetch:
        return

    fresh_markets: list[dict] = []
    for series in series_to_fetch:
        try:
            batch = await fetch_markets_by_series(session, [series], status="open")
            fresh_markets.extend(batch)
            await asyncio.sleep(0.2)
        except Exception as exc:
            logging.debug("Fast loop: series fetch failed %s: %s", series, exc)

    if not fresh_markets:
        return

    # Use cached NOAA only if it's fresh enough.  After a poll gap > NOAA_OBS_MAX_AGE_S
    # the cached reading may no longer represent the current observed daily max; fall
    # back to NOAA-None mode so the market-price gate (BAND_ARB_NOAA_NONE_MAX_NO_ASK)
    # provides soft confirmation instead.
    _noaa_age = time.monotonic() - _last_noaa_obs_time
    _noaa_for_arb = (
        _last_noaa_obs
        if _last_noaa_obs and _noaa_age < NOAA_OBS_MAX_AGE_S
        else None
    )
    if _noaa_for_arb is None and _last_noaa_obs:
        logging.debug("Fast loop: NOAA cache stale (%.0fs > %.0fs), using None", _noaa_age, NOAA_OBS_MAX_AGE_S)
    signals = find_band_arbs(
        fresh_markets,
        obs_values,
        noaa_obs_values=_noaa_for_arb,
        obs_dates=_fast_obs_dates or None,
    )
    if signals:
        logging.info("Fast loop: %d band_arb signal(s) found.", len(signals))

    fast_held: set[str] = set()
    if TRADE_DRY_RUN and ledger is not None:
        fast_held = ledger.open_tickers()
        if EXIT_REENTRY_COOLDOWN_MINUTES > 0:
            fast_held |= ledger.recently_exited_tickers(EXIT_REENTRY_COOLDOWN_MINUTES)

    for signal in signals:
        if signal.ticker in fast_held:
            logging.info(
                "Fast loop BandArb skip: %s in re-entry cooldown / held position.",
                signal.ticker,
            )
            continue
        await executor.maybe_trade_band_arb(session, signal)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run(*, poll_interval: int = POLL_INTERVAL_SECONDS) -> None:
    """Start the bot and poll indefinitely."""
    logging.info(
        "Kalshi bot starting — interval=%ds  fed-agencies=%d  "
        "numeric-sources=7  external-forecast-sources=3",
        poll_interval,
        len(AGENCIES),
    )

    seen = SeenDocuments()
    opp_log = OpportunityLog()
    executor = TradeExecutor()
    win_tracker = WinRateTracker()
    ledger = DryRunLedger() if TRADE_DRY_RUN else None
    if ledger is not None:
        executor.set_ledger(ledger)
    connector = aiohttp.TCPConnector(limit=30)
    cycle = 0

    # Seed calibrated priors from any historical data already in the DB.
    executor.refresh_calibrated_priors(win_tracker)

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            while True:
                try:
                    await _poll(session, seen, opp_log, executor, ledger)
                except Exception as exc:
                    logging.error("Unhandled error in poll cycle: %s", exc, exc_info=True)

                cycle += 1
                if WIN_RATE_REPORT_INTERVAL > 0 and cycle % WIN_RATE_REPORT_INTERVAL == 0:
                    try:
                        await win_tracker.settle_and_report(session)
                        executor.refresh_calibrated_priors(win_tracker)
                        run_attribution()
                    except Exception as exc:
                        logging.error("Win-rate tracker error: %s", exc)

                _sleep = _adaptive_poll_interval(datetime.now(timezone.utc))
                logging.info("Next poll in %ds …", _sleep)
                # Interleave fast band-arb loops during the sleep window
                _elapsed = 0.0
                while _elapsed + FAST_LOOP_INTERVAL < _sleep:
                    await asyncio.sleep(FAST_LOOP_INTERVAL)
                    _elapsed += FAST_LOOP_INTERVAL
                    try:
                        await _fast_loop(session, executor, ledger=ledger)
                    except Exception as exc:
                        logging.debug("Fast loop error: %s", exc)
                _remaining = _sleep - _elapsed
                if _remaining > 0:
                    await asyncio.sleep(_remaining)
    finally:
        seen.close()
        opp_log.close()
        executor.close()
        win_tracker.close()
        if ledger is not None:
            ledger.close()
