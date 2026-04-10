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
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .market_parser import parse_market
from .news.noaa import CITIES  # city timezone lookup for date-alignment guard

BAND_ARB_EXECUTION_ENABLED: bool = (
    os.environ.get("BAND_ARB_EXECUTION_ENABLED", "true").lower() == "true"
)
BAND_ARB_MIN_NO_ASK: int = int(os.environ.get("BAND_ARB_MIN_NO_ASK", "1"))
BAND_ARB_MAX_NO_ASK: int = int(os.environ.get("BAND_ARB_MAX_NO_ASK", "99"))
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
      - noaa_obs_values is provided but has no entry for this city AND the market
        NO ask exceeds BAND_ARB_NOAA_NONE_MAX_NO_ASK (market not yet confirming).
      - The METAR/NOAA divergence exceeds BAND_ARB_MAX_SOURCE_DIVERGENCE_F,
        indicating a sensor error (primary safety net after BLOCKER 3 removal).

    Note: NOAA band-crossing confirmation (BLOCKER 3) has been removed.  When
    NOAA is present and within the divergence threshold the signal fires even if
    NOAA hasn't yet crossed the ceiling — NOAA lags METAR by 5-8 min and the
    divergence check already confirms the sensor is trustworthy.

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
            # Kalshi resolves using the NWS CLI official daily high in integer
            # degrees.  A METAR reading of e.g. 66.04°F displays as "66.0°F"
            # but the official high is 66°F — still inside a [65, 66] band.
            # Only fire when METAR >= strike_hi + 0.5°F so the official rounded
            # value is guaranteed to exceed the upper bound (e.g. METAR ≥ 66.5°F
            # → official 67°F → definitively above a 65-66° band).
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

        if not is_definitive_no:
            continue

        # --- noaa_observed corroboration -----------------------------------
        # BLOCKER 3 (require NOAA to confirm band crossing) has been removed.
        # METAR leads NOAA by 5-8 min; waiting for NOAA to cross the ceiling
        # meant the market had already repriced by the time the signal fired.
        # The divergence check (BLOCKER 2) is the primary sensor-error guard.
        _corr_label = ""
        if noaa_obs_values is not None:
            noaa_val = noaa_obs_values.get(parsed.metric)
            if noaa_val is None:
                # NOAA has no data for this city yet — METAR is ahead.
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
