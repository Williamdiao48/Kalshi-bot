"""Band-pass / strike consistency arbitrage for KXHIGH partition markets.

All Kalshi temperature-high markets for a given city/date form a complete
partition of the temperature space (mutually exclusive, collectively exhaustive).
When the FAA METAR observed daily maximum temperature exceeds the upper boundary
of a band ("between") or bottom-tier ("under") market, that contract resolves NO
with near-certainty.

This module scans open KXHIGH markets against METAR observed daily highs and
emits BandArbSignal objects for any market that has been definitively passed
through.  The 5–8 minute head-start METAR provides over NOAA's aggregated
feed is the edge window: during that window the market may not yet reflect
the new observed high.

Market types (parsed by market_parser.py)
------------------------------------------
  "between"  strike_lo, strike_hi   → YES if strike_lo ≤ high ≤ strike_hi
  "under"    strike                 → YES if high < strike  (bottom tier)
  "over"     strike                 → YES if high > strike  (top tier)

Arb signals generated here (NO-side only)
------------------------------------------
  "between" market:  observed_max > strike_hi  → band passed through → buy NO
  "under"   market:  observed_max >= strike     → high at/above top of bottom tier → buy NO

The top-tier "over" YES signal is already produced by the normal
numeric_matcher + metar pipeline (metar is a _PASS_THROUGH source).
No duplication — this module only generates NO signals on passed-through bands.

Min/max NO ask filter
---------------------
  BAND_ARB_MIN_NO_ASK (default 1¢): below this the market has already priced
    the band as near-certain NO — no edge remaining.  1¢ allows catching fully
    repriced markets (99¢ YES bid) after NOAA confirms; at p_win=0.97 the
    Kelly recommendation is still the contract-count cap.
  BAND_ARB_MAX_NO_ASK (default 99¢): above this the market still believes the
    band is live.  At 99¢ NO ask the entry_cost = 1¢ and the exit manager's
    profit-take at 50% fires after just 0.5¢ gain — acceptable for a locked signal.
    Set to 0 to disable the cap.

Environment variables
---------------------
  BAND_ARB_EXECUTION_ENABLED        'true'/'false'. Default: true.
  BAND_ARB_MIN_NO_ASK               Minimum NO ask in cents. Default: 1.
  BAND_ARB_MAX_NO_ASK               Maximum NO ask in cents. Default: 99.
  BAND_ARB_NOAA_NONE_MAX_NO_ASK     When NOAA has no data for this city, only
                                    act if NO ask ≤ this cap (market provides
                                    soft confirmation). Default: 40.
  BAND_ARB_MAX_SOURCE_DIVERGENCE_F  Max METAR vs noaa_observed divergence (°F)
                                    before suppressing the signal. Default: 4.0.
                                    Primary safety net after BLOCKER 3 removal.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any

from .market_parser import parse_market
from .news.noaa import CITIES  # city timezone lookup for date-alignment guard

BAND_ARB_EXECUTION_ENABLED: bool = (
    os.environ.get("BAND_ARB_EXECUTION_ENABLED", "true").lower() == "true"
)
BAND_ARB_MIN_NO_ASK: int = int(os.environ.get("BAND_ARB_MIN_NO_ASK", "1"))
BAND_ARB_MAX_NO_ASK: int = int(os.environ.get("BAND_ARB_MAX_NO_ASK", "85"))
# Maximum divergence between METAR and noaa_observed before suppressing a band
# arb signal.  A 27.5°F gap (DEN, APR06) indicates sensor failure; 4°F is
# a conservative threshold that catches gross errors while tolerating the
# typical 1–2°F normal inter-sensor variation.  Set to 0 to disable check.
# This is now the primary safety net — BLOCKER 3 (band-crossing confirmation)
# has been removed; the divergence check is the main guard against sensor spikes.
BAND_ARB_MAX_SOURCE_DIVERGENCE_F: float = float(
    os.environ.get("BAND_ARB_MAX_SOURCE_DIVERGENCE_F", "4.0")
)
# When NOAA has no data for this city yet (METAR is 5-8 min ahead), only act
# if the market price provides soft confirmation (NO ask ≤ this cap).  At 40¢
# NO ask the market is ~40% certain the band was crossed.  Set to 0 to block
# all signals when NOAA is absent (restores old BLOCKER 1 behaviour).
BAND_ARB_NOAA_NONE_MAX_NO_ASK: int = int(os.environ.get("BAND_ARB_NOAA_NONE_MAX_NO_ASK", "40"))

# --- Band-arb YES signal configuration ------------------------------------
BAND_ARB_YES_ENABLED: bool = os.environ.get("BAND_ARB_YES_ENABLED", "true").lower() == "true"
# Comma-separated city suffixes to skip for YES signals (e.g. "aus,bos"). Default: "aus"
BAND_ARB_YES_BLACKLIST_CITIES: frozenset[str] = frozenset(
    c.strip().lower() for c in os.environ.get("BAND_ARB_YES_BLACKLIST_CITIES", "aus").split(",")
    if c.strip()
)
# Pre-lock: only fire within this many hours of close
BAND_ARB_YES_MAX_HOURS_PRELOCK: float = float(os.environ.get("BAND_ARB_YES_MAX_HOURS_PRELOCK", "6.0"))
# Max YES ask to enter (market already priced in above this). Default: 85¢
BAND_ARB_YES_MAX_YES_ASK: int = int(os.environ.get("BAND_ARB_YES_MAX_YES_ASK", "85"))
# Min YES ask (no edge if market is already near-certain YES). Default: 10¢
BAND_ARB_YES_MIN_YES_ASK: int = int(os.environ.get("BAND_ARB_YES_MIN_YES_ASK", "50"))
# Buffer (°F) inside band edges before firing.  NWS rounding buffer.
BAND_ARB_YES_BUFFER_F: float = float(os.environ.get("BAND_ARB_YES_BUFFER_F", "1.0"))
# Max METAR vs NOAA divergence. NOAA is required for YES (no market-price fallback).
BAND_ARB_YES_MAX_DIVERGENCE_F: float = float(os.environ.get("BAND_ARB_YES_MAX_DIVERGENCE_F", "3.0"))
# Local hour + minute at which daily high is considered locked (matches NOAA_OBS_PEAK_PAST)
BAND_ARB_YES_LOCK_LOCAL_HOUR: int = int(os.environ.get("BAND_ARB_YES_LOCK_LOCAL_HOUR", "16"))
BAND_ARB_YES_LOCK_LOCAL_MINUTE: int = int(os.environ.get("BAND_ARB_YES_LOCK_LOCAL_MINUTE", "30"))

# --- Forecast-driven NO signal configuration --------------------------------
FORECAST_NO_ENABLED: bool = os.environ.get("FORECAST_NO_ENABLED", "true").lower() == "true"
# Minimum forecast-to-strike edge (°F) for a source's value to count toward
# corroboration.  With NOAA day-1 MAE ~3-4°F, 5°F means P(correct) > 85%.
FORECAST_NO_MIN_EDGE_F: float = float(os.environ.get("FORECAST_NO_MIN_EDGE_F", "6.0"))
# Number of independent sources required (noaa_observed counts as 2 if edge >= 2°F)
FORECAST_NO_MIN_SOURCES: int = int(os.environ.get("FORECAST_NO_MIN_SOURCES", "2"))
# Maximum NO ask to enter — market hasn't yet priced in the outcome.
# Backtest: NO-settling bands open ~55-65¢ avg on the trading day; 70¢ cap
# captures most early-entry opportunities while leaving meaningful upside.
FORECAST_NO_MAX_ASK: int = int(os.environ.get("FORECAST_NO_MAX_ASK", "70"))
# Minimum NO ask — skip markets where YES is near-certain (market priced it in).
# A 2¢ NO ask means the market is 98% confident YES wins; the forecast edge
# would need to be enormous to justify buying NO.  15¢ floor (85¢ YES bid cap)
# keeps entries in the zone where we still have meaningful information edge.
FORECAST_NO_MIN_ASK: int = int(os.environ.get("FORECAST_NO_MIN_ASK", "15"))
# City suffixes to skip (same default as band_arb YES — AUS has low hit rate)
FORECAST_NO_BLACKLIST_CITIES: frozenset[str] = frozenset(
    c.strip().lower() for c in
    os.environ.get("FORECAST_NO_BLACKLIST_CITIES", "aus").split(",") if c.strip()
)
# Qualifying forecast sources.
# noaa_observed is intentionally excluded: for HIGH markets the running daily
# max above the strike triggers band_arb (duplicating the signal); for LOW
# markets the running daily min data has proven unreliable (station returns
# current temperature rather than the true overnight low in some cities).
_FORECAST_NO_SOURCES: frozenset[str] = frozenset({
    "hrrr", "nws_hourly", "open_meteo", "noaa",
})
# Local hour (city-local) at or after which the overnight low window is considered
# closed and the observed running minimum is treated as the definitive daily low.
# noaa_observed for temp_low_* only queries midnight→5AM to avoid afternoon highs
# contaminating the reading, so by 6AM that window is complete.
FORECAST_NO_OVERNIGHT_LOCK_HOUR: int = int(
    os.environ.get("FORECAST_NO_OVERNIGHT_LOCK_HOUR", "6")
)
# Require at least one near-term model (hrrr or nws_hourly) in the qualifying sources.
# Day-ahead models (open_meteo, noaa) have 4-5°F MAE vs 2-3°F for HRRR/NWS.
# When neither near-term model confirms the edge, signal confidence is too low.
FORECAST_NO_REQUIRE_NEAR_TERM: bool = os.environ.get(
    "FORECAST_NO_REQUIRE_NEAR_TERM", "false"
).lower() != "false"
# For LOW "under" markets (NO wins if overnight low stays above strike), block
# entries after this local hour.  Afternoon cooling begins ~15:00 and forecast
# model accuracy for overnight minimums degrades sharply past this point.
FORECAST_NO_LOW_UNDER_MAX_HOUR: int = int(
    os.environ.get("FORECAST_NO_LOW_UNDER_MAX_HOUR", "15")
)
# HRRR dissent veto: if HRRR is present and forecasts the daily high/low on the
# WRONG side of the strike (i.e., HRRR thinks YES is likely), block the signal
# regardless of how many other sources qualify.  HRRR is the most accurate
# short-range terrain-aware model; its disagreement indicates the other models
# are missing local dynamics (e.g. Rocky Mountain front-range cooling in Denver).
FORECAST_NO_HRRR_VETO: bool = (
    os.environ.get("FORECAST_NO_HRRR_VETO", "true").lower() == "true"
)
# Observed-minimum proximity margin for LOW-temperature forecast_no signals (°F).
# If the running observed minimum (metar/noaa_observed) is within this distance
# of the strike, further overnight cooling makes YES resolution too likely.
# Default 2.0°F — covers the typical METAR ±0.5°F observation noise plus a
# meaningful safety margin before the overnight cooling window.
FORECAST_NO_LOW_OBS_MARGIN_F: float = float(
    os.environ.get("FORECAST_NO_LOW_OBS_MARGIN_F", "2.0")
)
# Late-day HIGH-market METAR gap veto.  If the local hour is at or past
# FORECAST_NO_HIGH_LATE_HOUR and the METAR running maximum is more than
# FORECAST_NO_HIGH_LATE_MARGIN_F below the strike, the temperature has
# already peaked too low to reach the strike — block the NO signal.
FORECAST_NO_HIGH_LATE_HOUR: int = int(
    os.environ.get("FORECAST_NO_HIGH_LATE_HOUR", "15")
)
FORECAST_NO_HIGH_LATE_MARGIN_F: float = float(
    os.environ.get("FORECAST_NO_HIGH_LATE_MARGIN_F", "2.0")
)
# Pre-dawn proximity veto for LOW "over" markets (T-type: YES = daily_low > strike).
# After FORECAST_NO_LOW_OVER_OBS_MIN_HOUR local time the overnight low is largely set;
# if the running observed minimum is still more than FORECAST_NO_LOW_OVER_OBS_MARGIN_F
# above the strike, the drop required to reach (and cross) the strike is implausible.
# Window: [OBS_MIN_HOUR, FORECAST_NO_OVERNIGHT_LOCK_HOUR).  Default: 3 AM, 4.0°F.
FORECAST_NO_LOW_OVER_OBS_MIN_HOUR: int = int(
    os.environ.get("FORECAST_NO_LOW_OVER_OBS_MIN_HOUR", "3")
)
FORECAST_NO_LOW_OVER_OBS_MARGIN_F: float = float(
    os.environ.get("FORECAST_NO_LOW_OVER_OBS_MARGIN_F", "4.0")
)
# Terrain-city HRRR minimum-edge gate.  For cities in FORECAST_NO_HRRR_TERRAIN_CITIES,
# HRRR must be present AND have edge ≥ this value (°F) before the signal fires.
# Denver and other Rocky Mountain front-range cities have terrain-driven reversals
# that NWS hourly and OpenMeteo both miss; a barely-positive HRRR edge is not
# sufficient confidence to bet NO.  Default 2.0°F.
FORECAST_NO_HRRR_TERRAIN_MIN_EDGE_F: float = float(
    os.environ.get("FORECAST_NO_HRRR_TERRAIN_MIN_EDGE_F", "2.0")
)
# Cities (metric-key suffixes) where the HRRR terrain-veto applies.
# Comma-separated; e.g. "den" matches temp_high_den and temp_low_den.
FORECAST_NO_HRRR_TERRAIN_CITIES: frozenset[str] = frozenset(
    c.strip().lower()
    for c in os.environ.get("FORECAST_NO_HRRR_TERRAIN_CITIES", "den").split(",")
    if c.strip()
)
# Global model-spread threshold (°F).  If HRRR disagrees with the highest
# NWS-hourly or OpenMeteo forecast by more than this amount AND HRRR
# edge < FORECAST_NO_HRRR_TERRAIN_MIN_EDGE_F, block regardless of city.
FORECAST_NO_HRRR_SPREAD_F: float = float(
    os.environ.get("FORECAST_NO_HRRR_SPREAD_F", "4.0")
)


@dataclass
class BandArbSignal:
    """A band/bottom-tier market identified by METAR observed data.

    NO signals: The YES side will settle to 0¢ because the observed daily
    maximum temperature has already passed through (or exceeded) this band.
    Buying NO at the current NO ask captures near-certain profit.

    YES signals: The observed temperature is currently inside the band.
    Buying YES captures value when the market underprices the probability
    the temperature stays within the band until settlement.

    Fields
    ------
    metric         Canonical metric key, e.g. "temp_high_ny".
    ticker         Kalshi market ticker.
    yes_bid        Current YES bid in cents (NO ask = 100 − yes_bid).
    no_ask         Cost to buy one NO contract = 100 − yes_bid cents.
    observed_max   METAR observed daily maximum (°F).
    band_ceil      Upper bound: strike_hi for "between", strike for "under".
    direction      Market type: "between" | "under".
    city           Human-readable label for logging.
    side           "no" or "yes" signal.
    yes_ask        YES ask price in cents (for YES signals).
    hours_to_close Hours until market closes (for YES pre-lock p_win scaling).
    is_locked      True when past 4:30 PM local (daily high confirmed in-band).
    strike_lo      Lower band boundary (for YES clearance computation).
    """

    metric: str
    ticker: str
    yes_bid: int
    no_ask: int
    observed_max: float
    band_ceil: float
    direction: str
    city: str
    side: str = "no"
    yes_ask: int = 0
    hours_to_close: float = 0.0
    is_locked: bool = False
    strike_lo: float = 0.0


@dataclass
class ForecastNoSignal:
    """A KXHIGH/KXLOWT market where multi-source forecast consensus indicates
    the temperature will be well outside the band, but the NO price is still
    cheap enough to offer meaningful upside.

    Unlike BandArbSignal (which requires METAR confirmation), this fires
    earlier — typically 5-10h before METAR crosses the band — based on
    forecast model agreement rather than observed temperature.

    Fields
    ------
    ticker        Kalshi market ticker.
    metric        Canonical metric key, e.g. "temp_high_mia".
    city          City suffix for logging, e.g. "mia".
    direction     Band direction: "between" | "under" | "over".
    no_ask        Current NO ask in cents (= 100 − yes_bid).
    yes_bid       Current YES bid in cents.
    min_edge_f    Smallest forecast-to-strike edge among contributing sources (°F).
    source_count  Number of qualifying sources that crossed the edge threshold.
    sources       List of source names that contributed.
    score         Signal confidence in [0, 1].
    p_estimate    Estimated win probability for sizing.
    opportunity_kind  Always "forecast_no".
    """
    ticker: str
    metric: str
    city: str
    direction: str
    no_ask: int
    yes_bid: int
    min_edge_f: float
    source_count: int
    sources: list[str]
    score: float
    p_estimate: float
    # Full qualifying details: list of (source, forecast_value, edge_F).
    # Used to log each source's contribution to raw_forecasts for visibility.
    source_details: list[tuple[str, float, float]] = field(default_factory=list)
    opportunity_kind: str = "forecast_no"


def _hours_to_close(close_time_str: str) -> float | None:
    """Return hours until market close, or None if unparseable."""
    from datetime import timezone
    if not close_time_str:
        return None
    try:
        close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        return max(0.0, (close_dt - datetime.now(timezone.utc)).total_seconds() / 3600)
    except (ValueError, TypeError):
        return None


def _is_past_lock(city_tz) -> bool:
    """Return True if local city time has passed the 4:30 PM daily-high lock point."""
    local_now = datetime.now(city_tz)
    lock_mins = BAND_ARB_YES_LOCK_LOCAL_HOUR * 60 + BAND_ARB_YES_LOCK_LOCAL_MINUTE
    return local_now.hour * 60 + local_now.minute >= lock_mins


def find_band_arbs(
    markets: list[dict[str, Any]],
    obs_values: dict[str, float],
    noaa_obs_values: dict[str, float] | None = None,
    obs_dates: dict[str, date] | None = None,
) -> list[BandArbSignal]:
    """Scan open KXHIGH and KXLOWT markets for bands definitively passed through by METAR.

    For KXHIGH (daily high) markets:
      A "between" band is definitively NO when observed_max >= strike_hi + 0.5.
      An "under" (bottom-tier) market is definitively NO when observed_max >= strike - 0.5.

    For KXLOWT (daily low) markets:
      A "between" band is definitively NO when observed_min <= strike_lo - 0.5
      (the daily low dropped below the band floor).
      An "over" (top-tier) market is definitively NO when observed_min <= strike - 0.5
      (the daily low dropped below the "over" threshold).

    When noaa_obs_values is provided, the METAR reading is corroborated against
    the NOAA observed station value.  Signals are suppressed when:
      - noaa_obs_values is provided but has no entry for this city AND the market
        NO ask exceeds BAND_ARB_NOAA_NONE_MAX_NO_ASK (market not yet confirming).
      - The METAR/NOAA divergence exceeds BAND_ARB_MAX_SOURCE_DIVERGENCE_F,
        indicating a sensor error (primary safety net after BLOCKER 3 removal).

    Args:
        markets:         All open Kalshi market dicts (normalized with yes_bid).
        obs_values:      METAR observed daily high/low per metric (°F).
                         Keys: both temp_high_* and temp_low_* are accepted.
        noaa_obs_values: noaa_observed values per metric (°F). When provided,
                         used to corroborate METAR and filter station mismatches.
                         Pass None to skip corroboration (METAR-only mode).

    Returns:
        List of BandArbSignal objects, one per definitively-NO market within
        the profitability window.  Empty list when obs_values is empty.
    """
    if not obs_values:
        return []

    signals: list[BandArbSignal] = []

    for mkt in markets:
        ticker = mkt.get("ticker", "")
        is_high_market = "KXHIGH" in ticker
        is_low_market  = "KXLOWT" in ticker
        if not is_high_market and not is_low_market:
            continue

        parsed = parse_market(mkt)
        if parsed is None:
            continue
        if not (parsed.metric.startswith("temp_high") or parsed.metric.startswith("temp_low")):
            continue
        # Sanity: ensure ticker type matches parsed metric
        if is_high_market and not parsed.metric.startswith("temp_high"):
            continue
        if is_low_market and not parsed.metric.startswith("temp_low"):
            continue

        # --- Date-alignment guard -------------------------------------------
        # Ensure the market resolves on the city's local "today", not tomorrow
        # or yesterday.  METAR's rolling daily max resets at local midnight;
        # applying yesterday's peak to a tomorrow market (as happened with
        # KXHIGHDEN-26APR08-T71 at 11:38 PM MDT Apr 7) is a fatal false positive.
        # Ticker format: KXHIGHDEN-26APR08-T71  → date segment "26APR08"
        _ticker_parts = ticker.split("-")
        if len(_ticker_parts) >= 2:
            _date_seg = _ticker_parts[1]  # e.g. "26APR08"
            _date_match = re.fullmatch(r"(\d{2})([A-Z]{3})(\d{2})", _date_seg)
            if _date_match:
                _yr, _mon_str, _day = _date_match.groups()
                _MONTH_MAP = {
                    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
                }
                _mon = _MONTH_MAP.get(_mon_str)
                if _mon is not None:
                    # CITIES only has temp_high_* keys; normalize temp_low_* for lookup
                    _lookup_metric = parsed.metric.replace("temp_low_", "temp_high_")
                    _city_info = CITIES.get(_lookup_metric)
                    if _city_info is None:
                        # City not in registry — cannot verify date alignment.
                        # Fail closed: skip rather than risk applying yesterday's
                        # METAR reading to a tomorrow market.
                        logging.debug(
                            "BandArb skip: %s — metric %s not in CITIES registry"
                            " (cannot verify date alignment)",
                            ticker, parsed.metric,
                        )
                        continue
                    _city_tz = _city_info[3]
                    _local_today = datetime.now(_city_tz).date()
                    try:
                        _mkt_date = datetime(
                            2000 + int(_yr), _mon, int(_day)
                        ).date()
                    except ValueError:
                        _mkt_date = None
                    if _mkt_date is not None and _mkt_date != _local_today:
                        logging.debug(
                            "BandArb skip: %s resolves %s but city local today is %s",
                            ticker, _mkt_date, _local_today,
                        )
                        continue

                    # Stale-carry guard: METAR resets at local midnight but the previous
                    # day's peak may still be in memory the moment the date ticks over.
                    # obs_dates carries the calendar date the METAR value was computed for;
                    # if it doesn't match the market date, the reading is from yesterday.
                    if obs_dates and parsed.metric in obs_dates and _mkt_date is not None:
                        _metar_date = obs_dates[parsed.metric]
                        if _metar_date != _mkt_date:
                            logging.warning(
                                "BandArb skip: %s — METAR data dated %s != market date %s"
                                " (stale carry at midnight)",
                                ticker, _metar_date, _mkt_date,
                            )
                            continue

        observed_max = obs_values.get(parsed.metric)
        if observed_max is None:
            continue

        yes_bid = mkt.get("yes_bid")
        if yes_bid is None:
            continue
        no_ask = 100 - yes_bid

        # Skip if already priced as near-certain NO (no edge left)
        if BAND_ARB_MIN_NO_ASK > 0 and no_ask < BAND_ARB_MIN_NO_ASK:
            continue
        # Skip if market price exceeds cap
        if BAND_ARB_MAX_NO_ASK > 0 and no_ask > BAND_ARB_MAX_NO_ASK:
            continue

        is_definitive_no = False
        band_ceil = 0.0

        if not is_low_market:
            # --- HIGH-temp markets (KXHIGH) ---
            # Kalshi resolves using the NWS CLI official daily high in integer
            # degrees.  A METAR reading of e.g. 66.04°F displays as "66.0°F"
            # but the official high is 66°F — still inside a [65, 66] band.
            # Only fire when METAR >= strike_hi + 0.5°F so the official rounded
            # value is guaranteed to exceed the upper bound (e.g. METAR ≥ 66.5°F
            # → official 67°F → definitively above a 65-66° band).
            if parsed.direction == "between":
                if parsed.strike_hi is not None and observed_max >= parsed.strike_hi + 0.5:
                    is_definitive_no = True
                    band_ceil = parsed.strike_hi

            elif parsed.direction == "under":
                # NWS rounds to nearest integer. "Under 60°F" resolves NO when the
                # official high ≥ 60°F, which requires METAR ≥ 59.5°F (rounds up
                # to 60). The -0.5 buffer mirrors the +0.5 buffer on "between"
                # markets above.
                if parsed.strike is not None and observed_max >= parsed.strike - 0.5:
                    is_definitive_no = True
                    band_ceil = parsed.strike

        else:
            # --- LOW-temp markets (KXLOWT) ---
            # observed_max here holds the METAR daily *minimum* (keyed by temp_low_*).
            # Logic is symmetric but inverted: we fire NO when the daily low has
            # dropped *below* the band floor, proving the market can't resolve YES.
            # NWS rounds to nearest integer — use 0.5°F buffer same as highs.
            if parsed.direction == "between":
                # "Between lo–hi°F": resolves NO when official low < lo.
                # Fire when METAR_min ≤ strike_lo - 0.5 (rounds down to below lo).
                if parsed.strike_lo is not None and observed_max <= parsed.strike_lo - 0.5:
                    is_definitive_no = True
                    band_ceil = parsed.strike_lo  # the floor that was breached

            elif parsed.direction == "over":
                # "Over X°F": resolves NO when official low ≤ X.
                # Fire when METAR_min ≤ strike - 0.5 (official low will be ≤ strike).
                if parsed.strike is not None and observed_max <= parsed.strike - 0.5:
                    is_definitive_no = True
                    band_ceil = parsed.strike

        if not is_definitive_no:
            # --- YES signal: temperature inside band -----------------------
            if not BAND_ARB_YES_ENABLED:
                continue
            # Only "between" KXHIGH markets have a well-defined interior
            if is_low_market or parsed.direction != "between":
                continue
            if parsed.strike_lo is None or parsed.strike_hi is None:
                continue
            # City blacklist (data shows poor P&L in certain cities)
            _city_suffix = parsed.metric.replace("temp_high_", "")
            if BAND_ARB_YES_BLACKLIST_CITIES and _city_suffix in BAND_ARB_YES_BLACKLIST_CITIES:
                logging.debug("BandArb YES skip: %s — city %s blacklisted", ticker, _city_suffix)
                continue

            # Temperature must be strictly inside the band with NWS rounding buffer
            buf = BAND_ARB_YES_BUFFER_F
            in_band = (parsed.strike_lo + buf) <= observed_max <= (parsed.strike_hi - buf)
            if not in_band:
                continue

            # Determine if we're past the 4:30 PM daily-high lock point
            _lookup_metric_yes = parsed.metric.replace("temp_low_", "temp_high_")
            _city_info_yes = CITIES.get(_lookup_metric_yes)
            if _city_info_yes is None:
                continue
            _city_tz_yes = _city_info_yes[3]
            locked = _is_past_lock(_city_tz_yes)

            # Time gate: before lock, only fire within BAND_ARB_YES_MAX_HOURS_PRELOCK of close
            close_time_str = mkt.get("close_time") or mkt.get("expiration_time", "")
            htc = _hours_to_close(close_time_str)
            if htc is None:
                continue
            if not locked and htc > BAND_ARB_YES_MAX_HOURS_PRELOCK:
                continue

            # YES ask pricing gate (both tiers)
            yes_ask_raw = mkt.get("yes_ask")
            if yes_ask_raw is None:
                continue
            yes_ask_int = int(yes_ask_raw)
            if yes_ask_int < BAND_ARB_YES_MIN_YES_ASK or yes_ask_int > BAND_ARB_YES_MAX_YES_ASK:
                continue

            # NOAA corroboration REQUIRED for YES — no market-price fallback
            noaa_val_yes = (noaa_obs_values or {}).get(parsed.metric)
            if noaa_val_yes is None:
                logging.debug("BandArb YES skip: %s — NOAA absent (required for YES)", ticker)
                continue
            divergence_yes = abs(observed_max - noaa_val_yes)
            if BAND_ARB_YES_MAX_DIVERGENCE_F > 0 and divergence_yes > BAND_ARB_YES_MAX_DIVERGENCE_F:
                logging.warning(
                    "BandArb YES skip: sensor mismatch %s"
                    " (METAR=%.1f NOAA=%.1f diff=%.1f°F > %.1f°F limit)",
                    ticker, observed_max, noaa_val_yes,
                    divergence_yes, BAND_ARB_YES_MAX_DIVERGENCE_F,
                )
                continue

            city_yes = mkt.get("subtitle", "") or ticker
            logging.info(
                "BandArb YES signal (%s): %s  obs=%.1f°F in [%.1f–%.1f]"
                "  YES_ask=%d¢  %.1fh to close",
                "LOCKED" if locked else "pre-lock", ticker,
                observed_max, parsed.strike_lo, parsed.strike_hi, yes_ask_int, htc,
            )
            signals.append(BandArbSignal(
                metric=parsed.metric, ticker=ticker, yes_bid=yes_bid, no_ask=no_ask,
                observed_max=observed_max, band_ceil=parsed.strike_hi,
                direction=parsed.direction, city=city_yes,
                side="yes", yes_ask=yes_ask_int, hours_to_close=htc,
                is_locked=locked, strike_lo=parsed.strike_lo,
            ))
            continue

        # --- noaa_observed corroboration -----------------------------------
        # BLOCKER 3 (require NOAA to confirm band crossing) has been removed.
        # METAR leads NOAA by 5-8 min; waiting for NOAA to cross the ceiling
        # meant the market had already repriced by the time the signal fired.
        # The divergence check (BLOCKER 2) is the primary sensor-error guard.
        _corr_label = ""
        # noaa_obs_values=None means the caller had NO NOAA data at all (e.g. at
        # midnight before the first observation of the day arrives).  Treat this
        # identically to "NOAA present but missing this city" — the NOAA_NONE cap
        # must still apply so that a spurious METAR spike can't fire unchecked.
        noaa_val = (noaa_obs_values or {}).get(parsed.metric)
        if noaa_val is None:
            # NOAA has no data for this city yet — METAR is ahead, or NOAA has
            # no data at all (early morning before first observations arrive).
            # Use the market price as soft confirmation: if NO ask is above
            # the cap the market doesn't believe the crossing yet; skip.
            if BAND_ARB_NOAA_NONE_MAX_NO_ASK > 0 and no_ask > BAND_ARB_NOAA_NONE_MAX_NO_ASK:
                logging.debug(
                    "BandArb skip: %s — METAR=%.1f°F, NOAA absent,"
                    " no_ask=%d¢ > cap=%d¢",
                    ticker, observed_max, no_ask, BAND_ARB_NOAA_NONE_MAX_NO_ASK,
                )
                continue
            _corr_label = "  METAR-only (market confirms)"
        else:
            divergence = abs(observed_max - noaa_val)
            if BAND_ARB_MAX_SOURCE_DIVERGENCE_F > 0 and divergence > BAND_ARB_MAX_SOURCE_DIVERGENCE_F:
                logging.warning(
                    "BandArb skip: sensor mismatch on %s "
                    "(METAR=%.1f°F vs NOAA=%.1f°F, diff=%.1f°F > %.1f°F threshold)",
                    ticker, observed_max, noaa_val,
                    divergence, BAND_ARB_MAX_SOURCE_DIVERGENCE_F,
                )
                continue
            # NOAA sensor confirmed alive and within tolerance.
            # Act even if NOAA hasn't crossed the ceiling yet — it's just lagging.
            if noaa_val >= band_ceil:
                _corr_label = "  METAR+NOAA corroborated"
            else:
                _corr_label = (
                    f"  METAR+NOAA-lagging"
                    f" (NOAA={noaa_val:.1f}°F < ceil={band_ceil:.1f}°F)"
                )

        city = mkt.get("subtitle", "") or ticker
        logging.info(
            "BandArb signal: %s  obs=%.1f°F > ceil=%.1f°F  NO_ask=%d¢  (%s%s)",
            ticker, observed_max, band_ceil, no_ask, parsed.direction,
            _corr_label,
        )

        signals.append(BandArbSignal(
            metric=parsed.metric,
            ticker=ticker,
            yes_bid=yes_bid,
            no_ask=no_ask,
            observed_max=observed_max,
            band_ceil=band_ceil,
            direction=parsed.direction,
            city=city,
        ))

    return signals


def find_forecast_nos(
    markets: list[dict[str, Any]],
    data_points: list[Any],  # list[DataPoint] — avoid circular import
) -> list[ForecastNoSignal]:
    """Scan open KXHIGH/KXLOWT markets for early NO opportunities driven by
    multi-source forecast consensus.

    Unlike find_band_arbs() (which requires METAR temperature confirmation),
    this fires when multiple forecast models agree the temperature will be
    well outside a band, but the market NO price is still cheap.  Backtests
    show NO-settling bands are priced at ~55-65¢ on the trading day, an average
    of 5.4h before METAR confirms — vs. band_arb entry at 85-96¢.

    Entry gates (all must pass):
      1. FORECAST_NO_ENABLED is True
      2. Market is an open KXHIGH or KXLOWT "between"/"under"/"over" band
      3. At least FORECAST_NO_MIN_SOURCES independent sources each show
         forecast_value outside the strike by >= FORECAST_NO_MIN_EDGE_F °F
         (noaa_observed counts double as it is a hard observed lower bound)
      4. no_ask <= FORECAST_NO_MAX_ASK (market hasn't priced it in yet)
      5. City not in FORECAST_NO_BLACKLIST_CITIES

    Args:
        markets:     Open Kalshi market dicts (normalised with yes_bid field).
        data_points: All DataPoints from this poll cycle (mixed sources).

    Returns:
        List of ForecastNoSignal — one per qualifying market.
    """
    if not FORECAST_NO_ENABLED:
        return []

    # Build per-metric lookup: metric → list of (source, value)
    # Only include qualifying forecast/observed sources for temp metrics.
    from collections import defaultdict
    src_values: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for dp in data_points:
        if dp.source not in _FORECAST_NO_SOURCES:
            continue
        if not dp.metric.startswith(("temp_high", "temp_low")):
            continue
        src_values[dp.metric].append((dp.source, dp.value))

    # Build observed-minimum lookup from metar + noaa_observed DataPoints.
    # Used to veto LOW-market signals where the overnight running minimum has
    # already ruled out the NO outcome (e.g. observed_min > strike for a T-tier
    # market after 6AM means the low is locked above the strike → YES confirmed).
    # We keep the lower of metar vs noaa_observed (more conservative veto).
    obs_low_values: dict[str, float] = {}
    for dp in data_points:
        if dp.source not in ("metar", "noaa_observed"):
            continue
        if not dp.metric.startswith("temp_low"):
            continue
        existing = obs_low_values.get(dp.metric)
        if existing is None or dp.value < existing:
            obs_low_values[dp.metric] = dp.value

    # Build observed-maximum lookup from metar DataPoints (temp_high_*).
    # Used to veto HIGH-market signals where the running daily maximum has
    # already crossed the strike — at that point band_arb handles the signal
    # and forecast_no would be a duplicate at a worse price.
    # We use metar only (noaa_observed for temp_high queries the full day,
    # not just overnight, so it reliably represents the running daytime max).
    obs_high_values: dict[str, float] = {}
    for dp in data_points:
        if dp.source != "metar":
            continue
        if not dp.metric.startswith("temp_high"):
            continue
        existing = obs_high_values.get(dp.metric)
        if existing is None or dp.value > existing:
            obs_high_values[dp.metric] = dp.value

    signals: list[ForecastNoSignal] = []

    for mkt in markets:
        ticker = mkt.get("ticker", "")
        is_high_market = "KXHIGH" in ticker
        is_low_market  = "KXLOWT" in ticker
        if not is_high_market and not is_low_market:
            continue

        parsed = parse_market(mkt)
        if parsed is None:
            continue
        if not (parsed.metric.startswith("temp_high") or parsed.metric.startswith("temp_low")):
            continue
        if is_high_market and not parsed.metric.startswith("temp_high"):
            continue
        if is_low_market and not parsed.metric.startswith("temp_low"):
            continue

        # City blacklist
        city = parsed.metric.replace("temp_high_", "").replace("temp_low_", "")
        if FORECAST_NO_BLACKLIST_CITIES and city in FORECAST_NO_BLACKLIST_CITIES:
            continue

        # Date alignment: only fire for today's market (same guard as find_band_arbs)
        _ticker_parts = ticker.split("-")
        if len(_ticker_parts) >= 2:
            _date_seg = _ticker_parts[1]
            _date_match = re.fullmatch(r"(\d{2})([A-Z]{3})(\d{2})", _date_seg)
            if _date_match:
                _yr, _mon_str, _day = _date_match.groups()
                _MONTH_MAP = {
                    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
                }
                _mon = _MONTH_MAP.get(_mon_str)
                if _mon is not None:
                    _lookup_metric = parsed.metric.replace("temp_low_", "temp_high_")
                    _city_info = CITIES.get(_lookup_metric)
                    if _city_info is None:
                        continue
                    _city_tz = _city_info[3]
                    _local_today = datetime.now(_city_tz).date()
                    try:
                        _mkt_date = datetime(2000 + int(_yr), _mon, int(_day)).date()
                    except ValueError:
                        continue
                    if _mkt_date != _local_today:
                        continue

        yes_bid = mkt.get("yes_bid")
        if yes_bid is None:
            continue
        no_ask = 100 - yes_bid

        # Price gate: only enter if market hasn't priced it in yet, and the
        # NO price still implies genuine uncertainty (not near-certain YES).
        if no_ask <= 0 or no_ask < FORECAST_NO_MIN_ASK or no_ask > FORECAST_NO_MAX_ASK:
            continue

        # Compute city-local hour for time-gated checks below.
        # LOW markets use temp_low_* metrics; CITIES keys use temp_high_*, so translate.
        _lh_metric = parsed.metric.replace("temp_low_", "temp_high_")
        _lh_tz_info = CITIES.get(_lh_metric)
        _local_hour = datetime.now(_lh_tz_info[3]).hour if _lh_tz_info is not None else 12

        # LOW "under" afternoon gate: NO wins if overnight low stays above strike.
        # After FORECAST_NO_LOW_UNDER_MAX_HOUR local time, overnight cooling has
        # started and model accuracy for overnight minima drops sharply.
        if is_low_market and parsed.direction == "under" and parsed.strike is not None:
            if _local_hour >= FORECAST_NO_LOW_UNDER_MAX_HOUR:
                logging.debug(
                    "ForecastNO skip: %s — LOW under blocked after %02d:00 local"
                    " (local hour %d, overnight uncertainty too high)",
                    ticker, FORECAST_NO_LOW_UNDER_MAX_HOUR, _local_hour,
                )
                continue

        # Observed-data veto for LOW markets: if metar/noaa_observed has already
        # established a running minimum that makes NO impossible (or duplicates
        # band_arb), skip regardless of what forecast models say.
        if is_low_market:
            _obs_min = obs_low_values.get(parsed.metric)
            if _obs_min is not None:
                if parsed.direction == "over" and parsed.strike is not None:
                    # NO wins if daily_low ≤ strike.
                    # If observed min is already above strike after overnight window
                    # closes (local hour ≥ lock hour), YES is confirmed — block.
                    if _obs_min > parsed.strike and _local_hour >= FORECAST_NO_OVERNIGHT_LOCK_HOUR:
                        logging.debug(
                            "ForecastNO veto: %s — observed_min=%.1f°F > strike=%.1f°F"
                            " at local hour %d (overnight low locked above strike, YES confirmed)",
                            ticker, _obs_min, parsed.strike, _local_hour,
                        )
                        continue
                    # Pre-dawn proximity veto: after OBS_MIN_HOUR but before lock hour
                    # the temperature has nearly bottomed out.  If the running minimum
                    # is still more than MARGIN above the strike, the required drop is
                    # implausible in the remaining pre-dawn window.
                    if (
                        FORECAST_NO_LOW_OVER_OBS_MIN_HOUR <= _local_hour < FORECAST_NO_OVERNIGHT_LOCK_HOUR
                        and _obs_min > parsed.strike + FORECAST_NO_LOW_OVER_OBS_MARGIN_F
                    ):
                        logging.debug(
                            "ForecastNO veto (over pre-dawn): %s — local_hour=%d"
                            " obs_min=%.1f°F > strike+margin=%.1f°F"
                            " (%.1f°F drop required before dawn is implausible)",
                            ticker, _local_hour, _obs_min,
                            parsed.strike + FORECAST_NO_LOW_OVER_OBS_MARGIN_F,
                            _obs_min - parsed.strike,
                        )
                        continue

                elif parsed.direction == "under" and parsed.strike is not None:
                    # NO wins if daily_low ≥ strike.
                    # Hard veto: observed min already below strike → band_arb territory.
                    if _obs_min < parsed.strike:
                        logging.debug(
                            "ForecastNO veto: %s — observed_min=%.1f°F < strike=%.1f°F"
                            " (daily low already below strike, band_arb territory)",
                            ticker, _obs_min, parsed.strike,
                        )
                        continue
                    # Proximity veto: observed min within FORECAST_NO_LOW_OBS_MARGIN_F of
                    # the strike.  Any further overnight cooling triggers YES resolution.
                    if _obs_min < parsed.strike + FORECAST_NO_LOW_OBS_MARGIN_F:
                        logging.debug(
                            "ForecastNO veto (low proximity): %s — observed_min=%.1f°F"
                            " within %.1f°F of strike=%.1f°F (overnight cooling risk)",
                            ticker, _obs_min, FORECAST_NO_LOW_OBS_MARGIN_F, parsed.strike,
                        )
                        continue

                elif parsed.direction == "between" and parsed.strike_hi is not None and parsed.strike_lo is not None:
                    # "Too warm" NO (edge = obs - strike_hi): if observed min already
                    # above strike_hi after overnight window closes → locked above band.
                    if _obs_min > parsed.strike_hi and _local_hour >= FORECAST_NO_OVERNIGHT_LOCK_HOUR:
                        logging.debug(
                            "ForecastNO veto: %s — observed_min=%.1f°F > strike_hi=%.1f°F"
                            " at local hour %d (overnight low locked above band)",
                            ticker, _obs_min, parsed.strike_hi, _local_hour,
                        )
                        continue
                    # "Too cold" NO: if observed min already below strike_lo → band_arb.
                    if _obs_min < parsed.strike_lo:
                        logging.debug(
                            "ForecastNO veto: %s — observed_min=%.1f°F < strike_lo=%.1f°F"
                            " (band_arb handles cold-breach signals)",
                            ticker, _obs_min, parsed.strike_lo,
                        )
                        continue

        # Observed-data veto for HIGH markets: if METAR running maximum has already
        # crossed the strike, band_arb handles the confirmation — forecast_no would
        # be a duplicate entry at a worse price.
        if is_high_market:
            _obs_max = obs_high_values.get(parsed.metric)
            if _obs_max is not None:
                if parsed.direction == "under" and parsed.strike is not None:
                    # NO wins if daily_high < strike.
                    # If observed max already ≥ strike → YES confirmed, band_arb handles.
                    if _obs_max >= parsed.strike:
                        logging.debug(
                            "ForecastNO veto: %s — observed_max=%.1f°F >= strike=%.1f°F"
                            " (daily high already at/above strike, band_arb territory)",
                            ticker, _obs_max, parsed.strike,
                        )
                        continue
                    # Late-day gap veto: past FORECAST_NO_HIGH_LATE_HOUR local and
                    # running max is more than MARGIN below the strike.  Temperature
                    # has peaked; it won't bridge the gap in the cooling afternoon.
                    if (
                        _local_hour >= FORECAST_NO_HIGH_LATE_HOUR
                        and parsed.strike - _obs_max > FORECAST_NO_HIGH_LATE_MARGIN_F
                    ):
                        logging.debug(
                            "ForecastNO veto (late-day gap): %s — local_hour=%d"
                            " obs_max=%.1f°F strike=%.1f°F gap=%.1f°F > margin=%.1f°F",
                            ticker, _local_hour, _obs_max, parsed.strike,
                            parsed.strike - _obs_max, FORECAST_NO_HIGH_LATE_MARGIN_F,
                        )
                        continue

                elif parsed.direction == "between" and parsed.strike_hi is not None and parsed.strike_lo is not None:
                    # NO wins if daily_high > strike_hi OR daily_high < strike_lo.
                    # "Too hot" branch: if observed max already > strike_hi → band_arb.
                    if _obs_max > parsed.strike_hi:
                        logging.debug(
                            "ForecastNO veto: %s — observed_max=%.1f°F > strike_hi=%.1f°F"
                            " (daily high already above band ceiling, band_arb territory)",
                            ticker, _obs_max, parsed.strike_hi,
                        )
                        continue

        # Compute per-source edge for this market's band direction
        sources_map = src_values.get(parsed.metric, [])
        if not sources_map:
            continue

        # Hoist HRRR values so both veto blocks below can share them.
        hrrr_vals = [v for s, v in sources_map if s == "hrrr"]
        hrrr_val: float | None = max(hrrr_vals) if hrrr_vals else None

        # --- HRRR dissent veto --------------------------------------------------
        # If HRRR is present and forecasts the outcome on the WRONG side of the
        # strike, block regardless of what other models say.  HRRR is the most
        # accurate short-range terrain-aware model; its disagreement with global
        # models (nws_hourly, open_meteo) indicates local dynamics are being missed.
        if FORECAST_NO_HRRR_VETO and hrrr_val is not None:
            if not is_low_market:
                # HIGH "under" market: NO wins if daily_high < strike.
                # If HRRR forecasts the high will NOT reach the strike → veto.
                if parsed.direction == "under" and parsed.strike is not None:
                    if hrrr_val < parsed.strike:
                        logging.debug(
                            "ForecastNO veto (HRRR dissent): %s —"
                            " hrrr=%.1f°F < strike=%.1f°F (HRRR predicts YES)",
                            ticker, hrrr_val, parsed.strike,
                        )
                        continue
            else:
                # LOW "under" market: NO wins if daily_low ≥ strike.
                # If HRRR forecasts the low will be below the strike → veto.
                if parsed.direction == "under" and parsed.strike is not None:
                    if hrrr_val < parsed.strike:
                        logging.debug(
                            "ForecastNO veto (HRRR dissent LOW): %s —"
                            " hrrr=%.1f°F < strike=%.1f°F (HRRR predicts YES)",
                            ticker, hrrr_val, parsed.strike,
                        )
                        continue

        # --- HRRR terrain-city / model-spread veto ------------------------------
        # Two conditions, either alone triggers the veto:
        #   1. Terrain city: HRRR present but edge < TERRAIN_MIN_EDGE_F
        #      → require stronger HRRR confirmation for front-range cities.
        #   2. Global spread: HRRR present, spread vs other models > SPREAD_F,
        #      AND HRRR edge < TERRAIN_MIN_EDGE_F → model uncertainty too high.
        if hrrr_val is not None and parsed.direction == "under" and parsed.strike is not None:
            hrrr_edge = hrrr_val - parsed.strike  # positive = HRRR says NO
            city = (
                parsed.metric.replace("temp_high_", "")
                             .replace("temp_low_", "")
            )
            # Condition 1 — terrain city gate
            if (
                city in FORECAST_NO_HRRR_TERRAIN_CITIES
                and hrrr_edge < FORECAST_NO_HRRR_TERRAIN_MIN_EDGE_F
            ):
                logging.debug(
                    "ForecastNO veto (HRRR terrain): %s — city=%s"
                    " hrrr_edge=%.1f°F < min_edge=%.1f°F",
                    ticker, city, hrrr_edge, FORECAST_NO_HRRR_TERRAIN_MIN_EDGE_F,
                )
                continue

            # Condition 2 — global model-spread gate
            other_vals = [v for s, v in sources_map if s in ("nws_hourly", "open_meteo")]
            if other_vals:
                spread = max(other_vals) - hrrr_val
                if (
                    spread > FORECAST_NO_HRRR_SPREAD_F
                    and hrrr_edge < FORECAST_NO_HRRR_TERRAIN_MIN_EDGE_F
                ):
                    logging.debug(
                        "ForecastNO veto (HRRR spread): %s —"
                        " spread=%.1f°F > %.1f°F, hrrr_edge=%.1f°F < %.1f°F",
                        ticker, spread, FORECAST_NO_HRRR_SPREAD_F,
                        hrrr_edge, FORECAST_NO_HRRR_TERRAIN_MIN_EDGE_F,
                    )
                    continue

        qualifying: list[tuple[str, float, float]] = []  # (source, value, edge_F)
        for source, value in sources_map:
            if not is_low_market:
                # HIGH market: NO settles when temp > band ceiling
                if parsed.direction == "between" and parsed.strike_hi is not None:
                    edge = value - parsed.strike_hi
                elif parsed.direction == "under" and parsed.strike is not None:
                    edge = value - parsed.strike
                else:
                    continue
            else:
                # LOW market: NO settles when the daily low stays ABOVE the strike
                # (i.e., it doesn't get as cold as the market fears).
                #   "between"  – NO when daily_low > strike_hi (stayed above band)
                #                OR daily_low < strike_lo (fell below band)
                #                For forecast_no we only care about the "too warm"
                #                case: edge = value - strike_hi
                #   "under"    – YES if daily_low < strike; NO if daily_low >= strike.
                #                e.g. KXLOWT-B63.5: YES if low<63.5, NO if low≥63.5.
                #                Forecast signal: model says low will be well above
                #                strike → edge = value - strike.
                #   "over"     – top-tier, YES if daily_low > strike; NO if low ≤ strike.
                #                Forecast: model says low will be well below strike
                #                → edge = strike - value.
                if parsed.direction == "between" and parsed.strike_hi is not None:
                    edge = value - parsed.strike_hi  # forecast too warm for YES
                elif parsed.direction == "under" and parsed.strike is not None:
                    edge = value - parsed.strike  # forecast above strike → NO wins
                elif parsed.direction == "over" and parsed.strike is not None:
                    edge = parsed.strike - value  # forecast below strike → NO wins
                else:
                    continue

            if edge >= FORECAST_NO_MIN_EDGE_F:
                qualifying.append((source, value, edge))

        if not qualifying:
            continue

        # Count unique qualifying sources.  Deduplicate by source name so that
        # sources emitting multiple hourly forecasts per poll (open_meteo,
        # nws_hourly) cannot satisfy MIN_SOURCES on their own.
        source_score = len({src for src, _, _ in qualifying})
        if source_score < FORECAST_NO_MIN_SOURCES:
            continue

        # Near-term model anchor: require at least one hrrr or nws_hourly source
        # in the qualifying set.  Day-ahead global models (open_meteo, noaa) have
        # MAE ~4-5°F vs 2-3°F for HRRR/NWS.  Without near-term corroboration the
        # signal confidence is too low to justify the position.
        if FORECAST_NO_REQUIRE_NEAR_TERM:
            _near_term = {"hrrr", "nws_hourly"}
            if not any(s in _near_term for s, _, _e in qualifying):
                logging.debug(
                    "ForecastNO skip: %s — no near-term model in qualifying sources %s"
                    " (day-ahead models only, insufficient confidence)",
                    ticker, [s for s, _, _e in qualifying],
                )
                continue

        min_edge = min(e for _, _v, e in qualifying)
        source_names = [s for s, _, _e in qualifying]

        # Score: blend of source count and edge magnitude, capped at 0.95
        # More sources + larger edge = higher confidence
        raw_score = min(0.95, 0.60 + 0.05 * source_score + 0.01 * min_edge)

        # p_estimate: starts at FORECAST_NO_MIN_EDGE_F-derived base
        # 5°F edge with NOAA day-1 MAE ~3.5°F → ~92% P(correct direction)
        p_estimate = min(0.95, 0.75 + 0.02 * min_edge)

        logging.info(
            "ForecastNO signal: %s  no_ask=%d¢  edge=%.1f°F  sources=%s  score=%.2f",
            ticker, no_ask, min_edge, source_names, raw_score,
        )

        signals.append(ForecastNoSignal(
            ticker=ticker,
            metric=parsed.metric,
            city=city,
            direction=parsed.direction,
            no_ask=no_ask,
            yes_bid=yes_bid,
            min_edge_f=min_edge,
            source_count=source_score,
            sources=source_names,
            score=raw_score,
            p_estimate=p_estimate,
            source_details=qualifying,
        ))

    return signals
