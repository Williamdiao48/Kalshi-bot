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
  BAND_ARB_MIN_NO_ASK (default 3¢): below this the market has already priced
    the band as near-certain NO — no edge remaining.
  BAND_ARB_MAX_NO_ASK (default 25¢): above this the market still believes the
    band is live.  Could indicate a stale METAR, station mismatch, or late-day
    reading before official ASOS update.  Skip; let the normal pipeline handle.
    Set to 0 to disable the cap.

Environment variables
---------------------
  BAND_ARB_EXECUTION_ENABLED        'true'/'false'. Default: true.
  BAND_ARB_MIN_NO_ASK               Minimum NO ask in cents. Default: 3.
  BAND_ARB_MAX_NO_ASK               Maximum NO ask in cents. Default: 95.
                                    Raised from 25 — corroboration with
                                    noaa_observed makes high NO-ask entries safe.
  BAND_ARB_MAX_SOURCE_DIVERGENCE_F  Max METAR vs noaa_observed divergence (°F)
                                    before suppressing the signal. Default: 4.0.
                                    Catches station mismatches (e.g. DEN 27.5°F gap).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .market_parser import parse_market
from .news.noaa import CITIES  # city timezone lookup for date-alignment guard

BAND_ARB_EXECUTION_ENABLED: bool = (
    os.environ.get("BAND_ARB_EXECUTION_ENABLED", "true").lower() == "true"
)
BAND_ARB_MIN_NO_ASK: int = int(os.environ.get("BAND_ARB_MIN_NO_ASK", "3"))
BAND_ARB_MAX_NO_ASK: int = int(os.environ.get("BAND_ARB_MAX_NO_ASK", "95"))
# Maximum divergence between METAR and noaa_observed before suppressing a band
# arb signal.  A 27.5°F gap (DEN, APR06) indicates station mismatch; 4°F is
# a conservative threshold that catches gross mismatches while tolerating the
# typical 1–2°F normal inter-sensor variation.  Set to 0 to disable check.
BAND_ARB_MAX_SOURCE_DIVERGENCE_F: float = float(
    os.environ.get("BAND_ARB_MAX_SOURCE_DIVERGENCE_F", "4.0")
)


@dataclass
class BandArbSignal:
    """A definitively-NO band/bottom-tier market identified by METAR observed data.

    The YES side of this market will settle to 0¢ because the observed daily
    maximum temperature has already passed through (or exceeded) this band.
    Buying NO at the current NO ask captures near-certain profit.

    Fields
    ------
    metric       Canonical metric key, e.g. "temp_high_ny".
    ticker       Kalshi market ticker.
    yes_bid      Current YES bid in cents (NO ask = 100 − yes_bid).
    no_ask       Cost to buy one NO contract = 100 − yes_bid cents.
    observed_max METAR observed daily maximum (°F) that triggers this signal.
    band_ceil    Upper bound breached: strike_hi for "between", strike for "under".
    direction    Market type: "between" | "under".
    city         Human-readable label for logging (from market subtitle or ticker).
    """

    metric: str
    ticker: str
    yes_bid: int
    no_ask: int
    observed_max: float
    band_ceil: float
    direction: str
    city: str


def find_band_arbs(
    markets: list[dict[str, Any]],
    obs_values: dict[str, float],
    noaa_obs_values: dict[str, float] | None = None,
) -> list[BandArbSignal]:
    """Scan open KXHIGH markets for bands definitively passed through by METAR.

    Checks every KXHIGH market against the METAR observed daily high for its
    city.  A "between" band is definitively NO when observed_max > strike_hi.
    An "under" (bottom-tier) market is definitively NO when observed_max >=
    strike (the high has already reached or exceeded the upper threshold).

    When noaa_obs_values is provided, the METAR reading is corroborated against
    the NOAA observed station max.  Signals are suppressed when:
      - noaa_obs_values is provided but has no entry for this city (NOAA hasn't
        confirmed yet — METAR's 5-8 min edge may not have propagated).
      - The METAR/NOAA divergence exceeds BAND_ARB_MAX_SOURCE_DIVERGENCE_F,
        indicating a station mismatch (e.g. METAR KDEN vs NOAA ASOS station).
      - NOAA has not yet confirmed the band was crossed (noaa_val < band_ceil).

    Args:
        markets:         All open Kalshi market dicts (normalized with yes_bid).
        obs_values:      METAR observed daily max per metric (°F).
        noaa_obs_values: noaa_observed daily max per metric (°F). When provided,
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
        if "KXHIGH" not in ticker:
            continue

        parsed = parse_market(mkt)
        if parsed is None:
            continue
        if not parsed.metric.startswith("temp_high"):
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
                    _city_info = CITIES.get(parsed.metric)
                    if _city_info is not None:
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

        if parsed.direction == "between":
            if parsed.strike_hi is not None and observed_max > parsed.strike_hi:
                is_definitive_no = True
                band_ceil = parsed.strike_hi

        elif parsed.direction == "under":
            if parsed.strike is not None and observed_max >= parsed.strike:
                is_definitive_no = True
                band_ceil = parsed.strike

        if not is_definitive_no:
            continue

        # --- noaa_observed corroboration -----------------------------------
        # With BAND_ARB_MAX_NO_ASK raised to 95¢, we may enter at prices where
        # a station mismatch would be catastrophic.  Require noaa_observed to
        # independently confirm the band was crossed and agree within
        # BAND_ARB_MAX_SOURCE_DIVERGENCE_F before executing at high NO prices.
        if noaa_obs_values is not None:
            noaa_val = noaa_obs_values.get(parsed.metric)
            if noaa_val is None:
                # NOAA hasn't confirmed yet — METAR may be 5-8 min ahead.
                # Skip until NOAA catches up; false positives are too costly
                # at high NO ask prices.
                logging.debug(
                    "BandArb skip: %s — METAR=%.1f°F but NOAA not yet updated",
                    ticker, observed_max,
                )
                continue
            divergence = abs(observed_max - noaa_val)
            if BAND_ARB_MAX_SOURCE_DIVERGENCE_F > 0 and divergence > BAND_ARB_MAX_SOURCE_DIVERGENCE_F:
                logging.warning(
                    "BandArb skip: station mismatch on %s "
                    "(METAR=%.1f°F vs NOAA=%.1f°F, diff=%.1f°F > %.1f°F threshold)",
                    ticker, observed_max, noaa_val,
                    divergence, BAND_ARB_MAX_SOURCE_DIVERGENCE_F,
                )
                continue
            if noaa_val < band_ceil:
                # NOAA hasn't crossed the band yet — wait for confirmation.
                logging.debug(
                    "BandArb skip: %s — METAR=%.1f°F crossed ceil=%.1f°F "
                    "but NOAA=%.1f°F has not yet",
                    ticker, observed_max, band_ceil, noaa_val,
                )
                continue

        city = mkt.get("subtitle", "") or ticker
        logging.info(
            "BandArb signal: %s  obs=%.1f°F > ceil=%.1f°F  NO_ask=%d¢  (%s%s)",
            ticker, observed_max, band_ceil, no_ask, parsed.direction,
            "  METAR+NOAA corroborated" if noaa_obs_values is not None else "",
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
