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
  BAND_ARB_EXECUTION_ENABLED   'true'/'false'. Default: false (detect-only).
  BAND_ARB_MIN_NO_ASK          Minimum NO ask in cents. Default: 3.
  BAND_ARB_MAX_NO_ASK          Maximum NO ask in cents. Default: 25 (0 = no cap).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from .market_parser import parse_market

BAND_ARB_EXECUTION_ENABLED: bool = (
    os.environ.get("BAND_ARB_EXECUTION_ENABLED", "false").lower() == "true"
)
BAND_ARB_MIN_NO_ASK: int = int(os.environ.get("BAND_ARB_MIN_NO_ASK", "3"))
BAND_ARB_MAX_NO_ASK: int = int(os.environ.get("BAND_ARB_MAX_NO_ASK", "25"))


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
) -> list[BandArbSignal]:
    """Scan open KXHIGH markets for bands definitively passed through by METAR.

    Checks every KXHIGH market against the METAR observed daily high for its
    city.  A "between" band is definitively NO when observed_max > strike_hi.
    An "under" (bottom-tier) market is definitively NO when observed_max >=
    strike (the high has already reached or exceeded the upper threshold).

    The market's current YES bid is used to compute the NO ask cost.  Only
    markets priced within [BAND_ARB_MIN_NO_ASK, BAND_ARB_MAX_NO_ASK] are
    returned — too cheap means the market already knows; too expensive means
    the market disagrees (possible station mismatch, stale METAR, or METAR
    reading that hasn't propagated to the ASOS settlement station yet).

    Args:
        markets:    All open Kalshi market dicts (normalized with yes_bid).
        obs_values: Mapping of metric key → METAR observed daily max (°F).
                    Typically built from metar.fetch_city_forecasts() results.

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
        # Skip if market strongly disagrees (potential station mismatch)
        if BAND_ARB_MAX_NO_ASK > 0 and no_ask > BAND_ARB_MAX_NO_ASK:
            continue

        is_definitive_no = False
        band_ceil = 0.0

        if parsed.direction == "between":
            # YES wins if strike_lo ≤ high ≤ strike_hi
            # Definitively NO when observed_max has already exceeded strike_hi
            if parsed.strike_hi is not None and observed_max > parsed.strike_hi:
                is_definitive_no = True
                band_ceil = parsed.strike_hi

        elif parsed.direction == "under":
            # YES wins if high < strike (bottom-tier: "will high be below X°F?")
            # Definitively NO when observed_max has reached or exceeded strike
            if parsed.strike is not None and observed_max >= parsed.strike:
                is_definitive_no = True
                band_ceil = parsed.strike

        if not is_definitive_no:
            continue

        city = mkt.get("subtitle", "") or ticker
        logging.debug(
            "BandArb signal: %s  obs=%.1f°F > ceil=%.1f°F  NO_ask=%d¢  (%s)",
            ticker, observed_max, band_ceil, no_ask, parsed.direction,
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
