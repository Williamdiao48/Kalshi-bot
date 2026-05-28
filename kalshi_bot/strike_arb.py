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
import math
import os
from .utils import env_bool, env_float, env_int, parse_iso_dt
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any

import importlib.util as _ilu
from pathlib import Path as _Path

from .market_parser import parse_market
from .cities import CITIES  # city timezone lookup for date-alignment guard
from .calibration import forecast_no_win_prob as _cal_win_prob

# Per-city/month p75/p90 peak-time thresholds (minutes since local midnight).
# Loaded from data/peak_hour_p90.py; p75 is used by band_arb is_locked,
# p90 is exposed for solo numeric YES observational lock checks.
_BAND_ARB_P75_MINUTES: dict[str, dict[int, int]] = {}
_BAND_ARB_P90_MINUTES: dict[str, dict[int, int]] = {}
# Per-city/month p75 trough times for overnight LOWS (minutes since local midnight).
# Used by warm-NO entry gate: only enter after the p75 trough time for this city/month.
# Only values < 720 (noon) are used; winter/evening values fall back to the
# BAND_ARB_LOW_CEIL_MIN_HOUR default (6 AM).
_BAND_ARB_LOW_P75_MINUTES: dict[str, dict[int, int]] = {}
_p75_path = _Path(__file__).parent.parent / "data" / "peak_hour_p90.py"
if _p75_path.exists():
    _spec = _ilu.spec_from_file_location("peak_hour_p90", _p75_path)
    if _spec and _spec.loader:
        _p75_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_p75_mod)  # type: ignore[union-attr]
        _BAND_ARB_P75_MINUTES = getattr(_p75_mod, "P75_MINUTES", {})
        _BAND_ARB_P90_MINUTES = getattr(_p75_mod, "P90_MINUTES", {})
        _BAND_ARB_LOW_P75_MINUTES = getattr(_p75_mod, "LOW_P75_MINUTES", {})

# Per-source/city/month forecast bias corrections from 10-year Open-Meteo backtest.
# Key: (source, city_suffix, month)  Value: mean(forecast − actual) °F
# Apply: corrected_value = raw_value − bias  (cold model → shift up; warm → shift down)
_OM_BIAS: dict[tuple[str, str, int], float] = {}
_bias_path = _Path(__file__).parent / "openmeteo_bias_table.py"
if _bias_path.exists():
    _bias_spec = _ilu.spec_from_file_location("openmeteo_bias_table", _bias_path)
    if _bias_spec and _bias_spec.loader:
        _bias_mod = _ilu.module_from_spec(_bias_spec)
        _bias_spec.loader.exec_module(_bias_mod)  # type: ignore[union-attr]
        _OM_BIAS = getattr(_bias_mod, "BIAS_F", {})

BAND_ARB_EXECUTION_ENABLED: bool = (
    env_bool("BAND_ARB_EXECUTION_ENABLED", True)
)
BAND_ARB_MIN_NO_ASK: int = env_int("BAND_ARB_MIN_NO_ASK", 20)
BAND_ARB_MAX_NO_ASK: int = env_int("BAND_ARB_MAX_NO_ASK", 95)
# Maximum divergence between METAR and noaa_observed before suppressing a band
# arb signal.  A 27.5°F gap (DEN, APR06) indicates sensor failure; 4°F is
# a conservative threshold that catches gross errors while tolerating the
# typical 1–2°F normal inter-sensor variation.  Set to 0 to disable check.
# This is now the primary safety net — BLOCKER 3 (band-crossing confirmation)
# has been removed; the divergence check is the main guard against sensor spikes.
BAND_ARB_MAX_SOURCE_DIVERGENCE_F: float = float(
    os.environ.get("BAND_ARB_MAX_SOURCE_DIVERGENCE_F", "4.0")
)
# HRRR contradiction veto: if the HRRR forecast for today's high is more than
# BAND_ARB_HRRR_CONTRADICT_F below the METAR observed max AND HRRR is also
# below the band ceiling (HRRR agrees with YES winning), block the band_arb
# NO signal.  Catches stale METAR carry: at local midnight the previous day's
# peak may still be in the METAR running-max while HRRR correctly forecasts a
# cold next day.  Example: Denver Apr 17 — METAR=53.96°F (Apr 16 carry),
# HRRR=39°F (correct), market=90¢ YES (also correct); gap=14.96°F >> 8°F.
# Set to 0 to disable.  Default 8.0°F.
BAND_ARB_HRRR_CONTRADICT_F: float = float(
    os.environ.get("BAND_ARB_HRRR_CONTRADICT_F", "8.0")
)
# When NOAA has no data for this city yet (METAR is 5-8 min ahead), only act
# if the market price provides soft confirmation (NO ask ≤ this cap).  At 40¢
# NO ask the market is ~40% certain the band was crossed.  Set to 0 to block
# all signals when NOAA is absent (restores old BLOCKER 1 behaviour).
BAND_ARB_NOAA_NONE_MAX_NO_ASK: int = env_int("BAND_ARB_NOAA_NONE_MAX_NO_ASK", 40)
# NWS rounding buffer for LOW-temp band_arb (KXLOWT markets) — applied always.
# 0.5°F mirrors the +0.5 used for HIGH markets: ensures the METAR reading is
# far enough below the floor that NWS integer rounding can't put the official
# low back inside the band (e.g. METAR 24.4°F rounds to 24°F, but 24.6°F
# rounds to 25°F which is still inside [25–26]).
BAND_ARB_LOW_BUFFER_F: float = env_float("BAND_ARB_LOW_BUFFER_F", 0.5)
# Warm-side NO: minimum clearance (°F) between the METAR running daily min and the
# band ceiling before firing a daytime NO signal on KXLOWT "between" markets.
# Backtest (2026-03-15 to 2026-05-21, P75+0h anchor): WR=57% at margin<1.5°F,
# WR=87% at margin>=1.5°F. 1.5°F is the margin gate threshold.
BAND_ARB_LOW_CEIL_BUFFER_F: float = env_float("BAND_ARB_LOW_CEIL_BUFFER_F", 1.5)
# Fallback earliest local hour for warm-side NO entries (used when per-city/month
# P75 trough data is unavailable or indicates a winter evening trough ≥ noon).
# The primary gate is per-city/month P75 from data/peak_hour_p90.py (LOW_P75_MINUTES);
# for summer months P75 ≈ 6–9 AM local, which is the real entry gate.
# Backtest (2022–2026, n=2,642 margin≥1.5): WR at P75+0h ≈ 85–93% depending on
# margin; P75 entries earn 3–15x more EV than noon entries due to lower NO ask prices.
BAND_ARB_LOW_CEIL_MIN_HOUR: int = env_int("BAND_ARB_LOW_CEIL_MIN_HOUR", 6)
# Latest local hour for warm-side NO entries. After mid-afternoon the next
# night's cooling window opens and late-day signals degrade (16+: 50% WR).
BAND_ARB_LOW_CEIL_MAX_HOUR: int = env_int("BAND_ARB_LOW_CEIL_MAX_HOUR", 15)
# Maximum NO ask for warm-side KXLOWT NO entries.
# Backtest (May 2026, 69 shadow trades): ≤88¢ = 100% WR across all non-DEN cities
# in noon–3pm window; ≥89¢ introduces losses (negative-EV Kelly).
BAND_ARB_LOW_CEIL_MAX_NO_ASK: int = env_int("BAND_ARB_LOW_CEIL_MAX_NO_ASK", 88)
# Blocklist of city suffixes that may NOT generate warm-side NO signals.
# DEN is structurally unreliable (33% WR in backtest — alpine morning cold air
# pools persist past noon; METAR at airport lags valley lows).
# All other cities are allowed when the hour + ask gates pass.
BAND_ARB_LOW_WARM_NO_BLOCK_CITIES: frozenset[str] = frozenset(
    c.strip().lower()
    for c in os.environ.get("BAND_ARB_LOW_WARM_NO_BLOCK_CITIES", "den").split(",")
    if c.strip()
)
# Extra buffer applied for KXLOWT when noaa_observed has no data yet (api.weather.gov
# data gap, or the midnight-to-1 AM window before the first QC obs arrives).
# During this window only METAR and the market-price cap protect against
# QC-rejected METAR anomalies; 2.0°F keeps the combined effective buffer at
# floor − 2.5 until NOAA data is available to provide real confirmation.
BAND_ARB_LOW_NOAA_ABSENT_BUFFER_F: float = float(
    os.environ.get("BAND_ARB_LOW_NOAA_ABSENT_BUFFER_F", "2.0")
)
# NOAA day-1 forecast veto: when NOAA observed data is absent or stale, block
# the band_arb NO signal if the NWS day-1 forecast places the official high
# on the YES side of the strike (day1 < band_ceil for HIGH markets; > for LOW).
# This catches overnight/early-morning windows where only METAR is available
# but the official NWS forecast contradicts the observed crossing — the most
# common cause being an inter-station gap (e.g. KATL vs KFFC) or a QC-failed
# outlier.  Ignored when NOAA observed is fresh and in-range (trusted over
# a day-ahead forecast).  Set to "false" to disable.  Default: true.
BAND_ARB_NOAA_DAY1_VETO: bool = (
    os.environ.get("BAND_ARB_NOAA_DAY1_VETO", "true").lower() not in ("0", "false", "no")
)
# NWS CLI (nws_climo) hard veto: if the NWS Climatological Report — the exact
# settlement source Kalshi uses — has published today's preliminary maximum/minimum
# and it contradicts a NO resolution, block the signal.
#   HIGH markets: nws_climo_val < band_ceil → CLI says max is below ceiling → YES wins.
#   LOW markets:  nws_climo_val > band_ceil → CLI says min is above floor   → YES wins.
# The CLI preliminary report is typically published 5–8 PM local time, so this veto
# is most relevant for late-afternoon band crossings where METAR and the settlement
# station may disagree (inter-station gap, QC adjustment, etc.).
# Set to "false" to disable.  Default: true.
BAND_ARB_NWS_CLIMO_VETO: bool = (
    os.environ.get("BAND_ARB_NWS_CLIMO_VETO", "true").lower() not in ("0", "false", "no")
)

# --- Band-arb YES signal configuration ------------------------------------
BAND_ARB_YES_ENABLED: bool = env_bool("BAND_ARB_YES_ENABLED", True)
# Comma-separated city suffixes to skip for YES signals (e.g. "aus,bos"). Default: "aus"
BAND_ARB_YES_BLACKLIST_CITIES: frozenset[str] = frozenset(
    c.strip().lower() for c in os.environ.get("BAND_ARB_YES_BLACKLIST_CITIES", "aus").split(",")
    if c.strip()
)
# Pre-lock: only fire within this many hours of close
BAND_ARB_YES_MAX_HOURS_PRELOCK: float = env_float("BAND_ARB_YES_MAX_HOURS_PRELOCK", 6.0)
# Early entry: fire up to this many minutes before the P75 lock point when
# the running max is already OVERSHOT (past GFS morning forecast) AND still rising.
# Backtest (Feb–May 2026): P75-1h overshot WR=74.5% avg=+24.4¢ vs P75 all WR=70.3%.
# Best sub-filter (overshot+rising+price≥40): 95.8% WR, +25.7¢.
BAND_ARB_YES_EARLY_ENTRY_ENABLED: bool = env_bool("BAND_ARB_YES_EARLY_ENTRY_ENABLED", True)
BAND_ARB_YES_EARLY_ENTRY_MINUTES: int = env_int("BAND_ARB_YES_EARLY_ENTRY_MINUTES", 60)
# Min YES ask for early entries (backtest: 40¢+ overshot profitable; keep below normal 50¢ min)
BAND_ARB_YES_EARLY_MIN_YES_ASK: int = env_int("BAND_ARB_YES_EARLY_MIN_YES_ASK", 40)
# Max YES ask to enter (market already priced in above this). Default: 85¢
BAND_ARB_YES_MAX_YES_ASK: int = env_int("BAND_ARB_YES_MAX_YES_ASK", 85)
# Min YES ask (no edge if market is already near-certain YES). Default: 10¢
BAND_ARB_YES_MIN_YES_ASK: int = env_int("BAND_ARB_YES_MIN_YES_ASK", 50)
# Buffer (°F) inside band edges before firing.
# Kalshi KXHIGHT bands are 1°F wide (e.g. B56.5 = [56, 57]°F).  A symmetric
# 1.0°F buffer makes in_band always False for these bands.  0.0 = fire whenever
# observed_max is inside [strike_lo, strike_hi]; NWS rounding safety is provided
# by the lock-time gate (past 4:30 PM) and NOAA corroboration requirement.
BAND_ARB_YES_BUFFER_F: float = env_float("BAND_ARB_YES_BUFFER_F", 0.0)
# Max METAR vs NOAA divergence. NOAA is required for YES (no market-price fallback).
BAND_ARB_YES_MAX_DIVERGENCE_F: float = env_float("BAND_ARB_YES_MAX_DIVERGENCE_F", 3.0)
# Local hour + minute at which daily high is considered locked (matches NOAA_OBS_PEAK_PAST)
BAND_ARB_YES_LOCK_LOCAL_HOUR: int = env_int("BAND_ARB_YES_LOCK_LOCAL_HOUR", 16)
BAND_ARB_YES_LOCK_LOCAL_MINUTE: int = env_int("BAND_ARB_YES_LOCK_LOCAL_MINUTE", 30)
# Synoptic Celsius band arb (5-minute NWS updates via integer-°C range math)
SYNOPTIC_BAND_ARB_NO_ENABLED:  bool = env_bool("SYNOPTIC_BAND_ARB_NO_ENABLED", True)

# --- Band-arb YES for KXLOWT (symmetric to KXHIGHT YES) --------------------
# Fires when the METAR running daily min + NOAA observed both confirm the low
# is inside the band after the morning lock (BAND_ARB_LOW_CEIL_MIN_HOUR).
# Min YES ask lower than KXHIGHT (10¢ vs 50¢) because overnight lows are
# often mispriced — market assigns low probability even when observation is clear.
BAND_ARB_LOW_YES_ENABLED: bool = env_bool("BAND_ARB_LOW_YES_ENABLED", True)
BAND_ARB_LOW_YES_MIN_YES_ASK: int = env_int("BAND_ARB_LOW_YES_MIN_YES_ASK", 30)
BAND_ARB_LOW_YES_MAX_YES_ASK: int = env_int("BAND_ARB_LOW_YES_MAX_YES_ASK", 85)
# Max hours-to-close for KXLOWT YES entries. Unlike KXHIGHT (running max only rises),
# the overnight low can still drop with a cold front after the morning minimum is set.
# Only enter when the overnight risk window has substantially closed (~6 PM local).
BAND_ARB_LOW_YES_MAX_HTC: float = env_float("BAND_ARB_LOW_YES_MAX_HTC", 6.0)
# GFS daily min clearance gate for warm-side NO.
# Backtest (2022–2026, Section 11): for pos=2/3 (affordable NO trades the bot
# actually enters), a ≥+2°F GFS threshold adds ~1% WR lift while cutting 55–67%
# Buffer (°F) around the band used in the GFS in-band veto for KXLOWT warm-NO.
# If GFS daily-low forecast falls within ±BAND_ARB_LOW_GFS_IN_BAND_BUFFER_F of the
# band [floor, ceil], the trade is blocked.  1°F ≈ half a NWS rounding step.
# Set to 0.0 to disable. (Legacy clearance constant below kept for env-var compat.)
BAND_ARB_LOW_GFS_IN_BAND_BUFFER_F: float = env_float("BAND_ARB_LOW_GFS_IN_BAND_BUFFER_F", 0.5)
BAND_ARB_LOW_GFS_MIN_CLEARANCE_F: float = env_float("BAND_ARB_LOW_GFS_MIN_CLEARANCE_F", 0.0)

# --- Forecast-driven NO signal configuration --------------------------------
FORECAST_NO_ENABLED: bool = env_bool("FORECAST_NO_ENABLED", True)
# Minimum forecast-to-strike edge (°F) for a source's value to count toward
# corroboration.  With NOAA day-1 MAE ~3-4°F, 5°F means P(correct) > 85%.
FORECAST_NO_MIN_EDGE_F: float = env_float("FORECAST_NO_MIN_EDGE_F", 2.0)
# Direction-specific edge overrides for "between" band NO signals.
# Backtest (May 2026, 19 days, n=35 best config): NO_LOW WR=94.4%, NO_HIGH WR=82.4%
# at min_edge=2.0 — both directions are profitable at 2.0, so unified threshold.
FORECAST_NO_NO_HIGH_MIN_EDGE_F: float = float(
    os.environ.get("FORECAST_NO_NO_HIGH_MIN_EDGE_F", "2.0")
)
FORECAST_NO_NO_LOW_MIN_EDGE_F: float = float(
    os.environ.get("FORECAST_NO_NO_LOW_MIN_EDGE_F", "2.0")
)
# Number of independent sources required (noaa_observed counts as 2 if edge >= 2°F)
# Backtest: src=4 → 88.6% WR (+16¢ avg); src=2/3 → ~70% WR (+2¢ avg). Clear cliff.
FORECAST_NO_MIN_SOURCES: int = env_int("FORECAST_NO_MIN_SOURCES", 4)
# Higher source minimum for KXLOWT forecast_no signals — overnight lows are harder
# to forecast than daytime highs (cold-front timing uncertainty, boundary-layer
# decoupling).  Requires 3 sources (hrrr + nws_hourly + open_meteo or noaa) all
# crossing the edge threshold.  Set to 0 to use FORECAST_NO_MIN_SOURCES.
FORECAST_NO_LOWT_MIN_SOURCES: int = int(
    os.environ.get("FORECAST_NO_LOWT_MIN_SOURCES", "3")
)
# Maximum bid-ask spread (¢) for KXLOWT forecast_no entries.  Tighter than the
# global spread limit because forecast_no has no observed confirmation — pure
# model signal, higher uncertainty.  0 = disable (use global spread limit).
FORECAST_NO_LOWT_MAX_SPREAD_CENTS: int = int(
    os.environ.get("FORECAST_NO_LOWT_MAX_SPREAD_CENTS", "15")
)
# Maximum NO ask to enter — market hasn't yet priced in the outcome.
# Raised from 70 to 80: at 75-80¢ NO ask, Kelly still recommends 1-2 contracts
# for high-confidence signals (p≥0.90) and the market hasn't fully repriced.
FORECAST_NO_MAX_ASK: int = env_int("FORECAST_NO_MAX_ASK", 80)
# Minimum NO ask — skip markets where YES is near-certain (market priced it in).
# A 2¢ NO ask means the market is 98% confident YES wins; the forecast edge
# would need to be enormous to justify buying NO.  15¢ floor (85¢ YES bid cap)
# keeps entries in the zone where we still have meaningful information edge.
FORECAST_NO_MIN_ASK: int = env_int("FORECAST_NO_MIN_ASK", 45)
# Block forecast_no on threshold markets where direction=over (will temp EXCEED T?).
# Backtest (18 T-market trades): direction=over has 0 wins / 5 losses / -$4.13.
# Summer heat overshoots are routine; a model saying 81°F only needs to miss by
# 7°F for a T88 YES to resolve.  direction=under is less exposed (temp must drop).
FORECAST_NO_BLOCK_THRESHOLD_OVER: bool = env_bool("FORECAST_NO_BLOCK_THRESHOLD_OVER", True)
# Restrict forecast_no to "between" (B-market) entries only.
# Backtest (May 2026, 19 days): between_only → 88.6% WR, +16¢ avg (n=35);
# T-markets (direction=under/over) drag current policy to -3.2¢ avg at 66.5% WR.
FORECAST_NO_BETWEEN_ONLY: bool = env_bool("FORECAST_NO_BETWEEN_ONLY", True)
# Maximum hours until market close at entry time.
# Backtest (191 live trades): ≤20h → +573¢ / 75% WR; ≥24h → -1639¢ / 54% WR.
# Overnight entries (00-10h UTC) produce long-horizon forecasts with high uncertainty;
# the 20h cap keeps us in the same-day window where models are most reliable.
FORECAST_NO_MAX_HOURS: float = float(os.environ.get("FORECAST_NO_MAX_HOURS", "20.0"))
# City suffixes to skip (same default as band_arb YES — AUS has low hit rate)
FORECAST_NO_BLACKLIST_CITIES: frozenset[str] = frozenset(
    c.strip().lower() for c in
    os.environ.get("FORECAST_NO_BLACKLIST_CITIES", "aus").split(",") if c.strip()
)
# Per-source minimum edge overrides for the inner qualifying loop.
# hrrr/nws_hourly backtest at 84-92% win rate with 2-3°F edge; full global
# 5°F gate was silently killing all their signals before they reached this loop.
# Sources not listed here fall back to the direction-based threshold below.
_FORECAST_NO_SOURCE_MIN_EDGE: dict[str, float] = {
    "hrrr":              2.5,
    "nws_hourly":        3.0,
    "open_meteo_ecmwf":  3.5,
}

# Qualifying forecast sources.
# noaa_observed is intentionally excluded: for HIGH markets the running daily
# max above the strike triggers band_arb (duplicating the signal); for LOW
# markets the running daily min data has proven unreliable (station returns
# current temperature rather than the true overnight low in some cities).
_FORECAST_NO_SOURCES: frozenset[str] = frozenset({
    "hrrr", "nws_hourly", "open_meteo", "noaa",
    # Model-specific Open-Meteo sources (fetched by open_meteo.fetch_model_forecasts).
    # open_meteo_gfs is intentionally excluded: the blended "open_meteo" source IS the
    # GFS model — 10-year backtest confirmed identical values to 0.1°F.  Counting both
    # would double-count the same signal and inflate corroboration scores.
    "open_meteo_ecmwf", "open_meteo_icon", "open_meteo_gem",
})
# Local hour (city-local) at or after which the overnight low window is considered
# closed and the observed running minimum is treated as the definitive daily low.
# noaa_observed for temp_low_* only queries midnight→5AM to avoid afternoon highs
# contaminating the reading, so by 6AM that window is complete.
FORECAST_NO_OVERNIGHT_LOCK_HOUR: int = int(
    os.environ.get("FORECAST_NO_OVERNIGHT_LOCK_HOUR", "6")
)
# Require at least one Open-Meteo ensemble model in the qualifying sources.
# Live P&L analysis (134 trades): open_meteo family wins 65-68% vs 53-57% for
# NOAA/HRRR/NWS.  Root cause: NWS and HRRR are the most widely consumed US
# forecasts — Kalshi market-makers already price them in.  Open-Meteo's
# international ensembles (ECMWF, GEM, ICON) are less followed and still carry
# genuine information alpha.  "bad_only" signals (no open_meteo) win 14% daytime
# and 0% overnight — reliably wrong, block entirely.
FORECAST_NO_REQUIRE_OPEN_METEO: bool = os.environ.get(
    "FORECAST_NO_REQUIRE_OPEN_METEO", "true"
).lower() != "false"

# Earliest local hour at which a pure open_meteo-only signal may fire.
# When no NOAA/HRRR/NWS model corroborates, overnight signals (before 7 AM local)
# win at 50% — the temperature hasn't developed yet and all models are uncertain.
# Mixed signals (open_meteo + NOAA/HRRR both agree) are exempt: cross-family
# consensus at any hour is a genuinely stronger signal (71% win rate overnight).
# Set to 0 to disable.
FORECAST_NO_OPEN_METEO_DAYTIME_HOUR: int = int(
    os.environ.get("FORECAST_NO_OPEN_METEO_DAYTIME_HOUR", "7")
)
# For LOW "under" markets (NO wins if overnight low stays above strike), block
# entries after this local hour.  Afternoon cooling begins ~15:00 and forecast
# model accuracy for overnight minimums degrades sharply past this point.
FORECAST_NO_LOW_UNDER_MAX_HOUR: int = int(
    os.environ.get("FORECAST_NO_LOW_UNDER_MAX_HOUR", "15")
)
# Midnight METAR gap veto: block LOW "under" forecast_no signals during the
# overnight window (local hour < FORECAST_NO_OVERNIGHT_LOCK_HOUR) when no
# METAR observation has been collected yet for this metric.  Without an
# observed minimum, the proximity veto cannot run — and the daily METAR
# tracking resets at midnight, leaving a ~15-minute window where obs_min
# is None even though the overnight low is already near the strike.
FORECAST_NO_LOW_UNDER_REQUIRE_METAR: bool = (
    os.environ.get("FORECAST_NO_LOW_UNDER_REQUIRE_METAR", "true").lower()
    not in ("false", "0", "no")
)
# HRRR dissent veto: if HRRR is present and forecasts the daily high/low on the
# WRONG side of the strike (i.e., HRRR thinks YES is likely), block the signal
# regardless of how many other sources qualify.  HRRR is the most accurate
# short-range terrain-aware model; its disagreement indicates the other models
# are missing local dynamics (e.g. Rocky Mountain front-range cooling in Denver).
FORECAST_NO_HRRR_VETO: bool = (
    env_bool("FORECAST_NO_HRRR_VETO", True)
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
# Fires when the running observed minimum is still more than a time-scaled margin
# above the strike — meaning the required drop before dawn is implausible.
#
# The required margin grows linearly with hours remaining until dawn (lock hour),
# so the veto is tight near dawn (small drop needed to be implausible) and lenient
# near midnight (large drop needed before it's considered impossible):
#
#   required_margin = FORECAST_NO_LOW_OVER_OBS_MARGIN_F
#                     + FORECAST_NO_LOW_OVER_OBS_MARGIN_PER_HOUR
#                       × (FORECAST_NO_OVERNIGHT_LOCK_HOUR − local_hour − 1)
#
# Example with defaults (base=2.0°F, per_hour=0.75°F, lock=6):
#   5 AM  → 2.0 + 0.75×0 = 2.00°F   (nearly dawn — tight)
#   4 AM  → 2.0 + 0.75×1 = 2.75°F
#   3 AM  → 2.0 + 0.75×2 = 3.50°F
#   2 AM  → 2.0 + 0.75×3 = 4.25°F
#   1 AM  → 2.0 + 0.75×4 = 5.00°F
#   midnight → 2.0 + 0.75×5 = 5.75°F  (most of the night ahead — lenient)
#
# Window starts at FORECAST_NO_LOW_OVER_OBS_MIN_HOUR (default: midnight = 0).
FORECAST_NO_LOW_OVER_OBS_MIN_HOUR: int = int(
    os.environ.get("FORECAST_NO_LOW_OVER_OBS_MIN_HOUR", "0")
)
# Base margin at the hour just before the lock (°F).  Smaller = tighter veto near dawn.
FORECAST_NO_LOW_OVER_OBS_MARGIN_F: float = float(
    os.environ.get("FORECAST_NO_LOW_OVER_OBS_MARGIN_F", "2.0")
)
# Additional margin added per hour of distance from the lock hour (°F/hr).
# Higher = more lenient earlier in the night.
FORECAST_NO_LOW_OVER_OBS_MARGIN_PER_HOUR: float = float(
    os.environ.get("FORECAST_NO_LOW_OVER_OBS_MARGIN_PER_HOUR", "1.0")
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

# Maximum allowed spread (°F) between the highest and lowest weather-model
# forecast for a given metric before the signal is blocked.  When models
# disagree by more than this amount, forecast uncertainty is too high to
# justify a trade.  Applies to all directions (including band/"between" markets).
# Calibration: hist backtest (2026-05-20, n=150) shows 4–8°F spread bucket
# is profitable (WR=71–72%, avg=+4–7¢); 8°F+ breaks down (WR=50%, avg=-15¢).
# 4°F default was too conservative — blocked profitable signals in that range.
# Set to 0 to disable.
FORECAST_NO_MODEL_SPREAD_F: float = float(
    os.environ.get("FORECAST_NO_MODEL_SPREAD_F", "8.0")
)
# Minimum model spread (°F) required to fire a forecast_no signal.
# Backtest (May 2026, n=285, best config):
#   0–2°F bucket: WR=70.1%, avg=+2.3¢  (below average)
#   2–4°F bucket: WR=76.9%, avg=+6.5¢  (best bucket)
# A floor of 1°F cuts only degenerate full-consensus cases (all models within
# 1°F of each other) where the market has likely already priced in the forecast.
FORECAST_NO_MODEL_SPREAD_MIN_F: float = float(
    os.environ.get("FORECAST_NO_MODEL_SPREAD_MIN_F", "1.0")
)


def _synoptic_celsius_band(
    celsius_c: int,
    strike_lo: float,
    strike_hi: float,
) -> str | None:
    """Check if integer Celsius reading definitively places the temp in or above a Kalshi band.

    For a 5-minute synoptic reading of N°C, the actual temperature is guaranteed
    in [F_low, F_high] = [(N−0.5)×1.8+32, (N+0.499)×1.8+32] (width 1.8°F).

    For a Kalshi "between" band [strike_lo, strike_hi] the settlement window is
    [strike_lo−0.5, strike_hi+0.5) (NWS rounds to nearest integer °F, 2°F wide).

    Returns:
      'YES'  if the entire 1.8°F interval fits inside the settlement window
             → temp is definitively inside the band regardless of rounding
      'NO'   if the lower bound already exceeds the ceiling (strike_hi+0.5)
             → temp is definitively above the band; market resolves NO
      None   if the result is ambiguous (interval straddles a boundary)
    """
    f_low  = (celsius_c - 0.5)    * 1.8 + 32.0
    f_high = (celsius_c + 0.4999) * 1.8 + 32.0
    eff_lo = strike_lo - 0.5
    eff_hi = strike_hi + 0.5
    if f_low >= eff_hi:
        return "NO"
    if f_low >= eff_lo and f_high < eff_hi:
        return "YES"
    return None


def _noaa_obs_bounds(temp_f: float) -> tuple[float, float]:
    """Return (lower_f, upper_f) bounding the true temperature given NWS API °C rounding.

    NWS observations are stored in °C and converted to °F by the API.
    Synoptic readings (5-min grid, :00/:05/…/:55) round to integer °C → ±0.5°C = ±0.9°F.
    METAR readings (:53) use 0.1°C precision → ±0.05°C = ±0.09°F (negligible).
    Detection: if the °C value is within 0.01°C of an integer → synoptic precision.
    """
    t_c = (temp_f - 32.0) * 5.0 / 9.0
    half_step = 0.5 if abs(t_c - round(t_c)) < 0.01 else 0.05
    return (t_c - half_step) * 9.0 / 5.0 + 32.0, (t_c + half_step) * 9.0 / 5.0 + 32.0


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
    # Corroboration snapshot — logged to trades.note for future analysis
    noaa_val: float | None = None       # NOAA observed at signal time (None = absent/stale)
    hrrr_val: float | None = None       # HRRR forecast at signal time
    nws_climo_val: float | None = None  # NWS CLI value if available (settlement source)
    corr_status: str = ""               # "metar_only" | "metar_noaa_corroborated" | "metar_noaa_lagging"
    yes_ask_entry: int = 0              # YES ask at entry (enables spread + market_p logging)
    shadow: bool = False                # True = blocked by whitelist; log only, do not execute
    # GFS morning forecast gate (band_arb YES only)
    # gfs_morning_f: frozen morning GFS daily-max forecast for this city/date
    # gfs_lagging: True when METAR running max < morning forecast at signal time
    #   → 1-contract cap (data logging); False/None → full Kelly sizing
    gfs_morning_f: float | None = None
    gfs_lagging: bool | None = None
    is_rising: bool | None = None  # True = spot still at running max; False = plateaued


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
    # Additional context logged to trades for future analysis
    yes_ask: int = 0                    # YES ask at signal time (for spread + market_p)
    hours_to_close: float | None = None # Hours until market close at signal time
    model_spread_f: float | None = None # Max − min across forecast models (°F)
    # For "between" band markets: which side of the band the qualifying sources agree on.
    # "NO_HIGH" = forecast above strike_hi (too hot); "NO_LOW" = below strike_lo (too cold).
    # None for "under"/"over" markets where direction is already unambiguous.
    no_direction: str | None = None
    # Strike values from ParsedMarket — stored here so trade_executor can persist
    # them in the note for later use by exit_manager signal invalidation checks.
    strike:    float | None = None    # "under" / "over" markets
    strike_lo: float | None = None    # "between" lower bound
    strike_hi: float | None = None    # "between" upper bound
    # METAR observed temperature at signal time — running max for HIGH markets,
    # running min for LOW markets.  obs_gap_f = obs_temp_f − effective_strike
    # (positive = safe margin away from the strike; negative = already crossed).
    # Not used for trade logic; stored in note for post-hoc analysis.
    obs_temp_f: float | None = None
    obs_gap_f:  float | None = None


def _hours_to_close(close_time_str: str) -> float | None:
    """Return hours until market close, or None if unparseable."""
    from datetime import timezone
    if not close_time_str:
        return None
    try:
        close_dt = parse_iso_dt(close_time_str)
        return max(0.0, (close_dt - datetime.now(timezone.utc)).total_seconds() / 3600)
    except (ValueError, TypeError):
        return None


def _is_past_lock(city_tz, metric: str | None = None) -> bool:
    """Return True if local city time has passed the p75 daily-high lock point for this city/month.

    Uses per-city/month p75 thresholds from data/peak_hour_p90.py when available;
    falls back to the global BAND_ARB_YES_LOCK_LOCAL_HOUR/MINUTE env-var gate.
    """
    local_now = datetime.now(city_tz)
    if metric is not None:
        p75 = _BAND_ARB_P75_MINUTES.get(metric, {}).get(local_now.month)
        if p75 is not None:
            return local_now.hour * 60 + local_now.minute >= p75
    lock_mins = BAND_ARB_YES_LOCK_LOCAL_HOUR * 60 + BAND_ARB_YES_LOCK_LOCAL_MINUTE
    return local_now.hour * 60 + local_now.minute >= lock_mins


def _minutes_until_lock(city_tz, metric: str | None = None) -> int | None:
    """Minutes remaining until the P75 lock point, or None if already past."""
    local_now = datetime.now(city_tz)
    current_mins = local_now.hour * 60 + local_now.minute
    if metric is not None:
        p75 = _BAND_ARB_P75_MINUTES.get(metric, {}).get(local_now.month)
        if p75 is not None:
            delta = p75 - current_mins
            return delta if delta > 0 else None
    lock_mins = BAND_ARB_YES_LOCK_LOCAL_HOUR * 60 + BAND_ARB_YES_LOCK_LOCAL_MINUTE
    delta = lock_mins - current_mins
    return delta if delta > 0 else None


def is_past_p90(metric: str, city_tz) -> bool:
    """Return True if local city time has passed the p90 daily-high threshold.

    Used by solo numeric YES observational trades (noaa_observed/metar) to gate
    locked sizing. Band_arb uses p75 (_is_past_lock) since it requires
    corroboration; solo numeric only gets locked sizing after p90 because without
    corroboration it needs a tighter time gate (90% of days have already peaked).
    Falls back to p75 if p90 data is missing for this city/month.
    """
    local_now = datetime.now(city_tz)
    lookup = _BAND_ARB_P90_MINUTES.get(metric, {}).get(local_now.month)
    if lookup is None:
        lookup = _BAND_ARB_P75_MINUTES.get(metric, {}).get(local_now.month)
    if lookup is not None:
        return local_now.hour * 60 + local_now.minute >= lookup
    lock_mins = BAND_ARB_YES_LOCK_LOCAL_HOUR * 60 + BAND_ARB_YES_LOCK_LOCAL_MINUTE
    return local_now.hour * 60 + local_now.minute >= lock_mins


def find_band_arbs(
    markets: list[dict[str, Any]],
    obs_values: dict[str, float],
    noaa_obs_values: dict[str, float] | None = None,
    obs_dates: dict[str, date] | None = None,
    noaa_obs_dates: dict[str, date] | None = None,
    noaa_day1_values: dict[str, float] | None = None,
    hrrr_values: dict[str, float] | None = None,
    nws_climo_values: dict[str, float] | None = None,
    synoptic_celsius: dict[str, int | None] | None = None,
    gfs_morning_values: dict[str, float] | None = None,
    metar_is_rising: dict[str, bool | None] | None = None,
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
        noaa_obs_dates:  Local calendar date of each noaa_observed reading
                         (keyed by metric).  When provided, a NOAA reading
                         whose local_date != the market date is treated as
                         absent (stale carry from a prior day).
        noaa_day1_values: NWS day-1 forecast highs/lows per metric (°F).
                         Used as a veto only when noaa_obs_values is absent
                         or stale: if the day-1 forecast places the official
                         value on the YES side of the strike, the NO signal
                         is suppressed (see BAND_ARB_NOAA_DAY1_VETO).
        nws_climo_values: NWS CLI (Climatological Report) observed daily
                         max/min temperatures per metric (°F).  This is the
                         exact settlement source Kalshi uses.  When present
                         and contradicting the NO signal (CLI value on the
                         YES side of the strike), the signal is hard-vetoed
                         regardless of METAR (see BAND_ARB_NWS_CLIMO_VETO).
                         Typically only available after ~5 PM local time.

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
            logging.debug(
                "BandArb NO skip: %s — no_ask=%d¢ below min=%d¢ (market strongly YES)",
                ticker, no_ask, BAND_ARB_MIN_NO_ASK,
            )
            continue
        # Skip if market price exceeds cap
        if BAND_ARB_MAX_NO_ASK > 0 and no_ask > BAND_ARB_MAX_NO_ASK:
            logging.debug(
                "BandArb NO skip: %s — no_ask=%d¢ above max=%d¢ (already priced in)",
                ticker, no_ask, BAND_ARB_MAX_NO_ASK,
            )
            continue

        is_definitive_no = False
        is_warm_no = False
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
                elif (
                    SYNOPTIC_BAND_ARB_NO_ENABLED
                    and synoptic_celsius is not None
                    and parsed.strike_lo is not None
                    and parsed.strike_hi is not None
                ):
                    syn_c = synoptic_celsius.get(parsed.metric)
                    if syn_c is not None and _synoptic_celsius_band(syn_c, parsed.strike_lo, parsed.strike_hi) == "NO":
                        is_definitive_no = True
                        band_ceil = parsed.strike_hi
                        logging.info(
                            "BandArb synoptic-NO: %s  syn=%d°C → F_low=%.1f°F > ceiling=%.1f°F",
                            ticker, syn_c,
                            (syn_c - 0.5) * 1.8 + 32.0, parsed.strike_hi + 0.5,
                        )

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
                # 0.5°F rounding buffer: METAR must be clearly below floor so
                # NWS integer rounding can't put the official low back in-band.
                # When NOAA is absent an extra buffer is applied below.
                if parsed.strike_lo is not None and observed_max <= parsed.strike_lo - BAND_ARB_LOW_BUFFER_F:
                    is_definitive_no = True
                    band_ceil = parsed.strike_lo  # the floor that was breached
                elif (
                    BAND_ARB_LOW_CEIL_BUFFER_F > 0
                    and parsed.strike_hi is not None
                    and observed_max >= parsed.strike_hi + BAND_ARB_LOW_CEIL_BUFFER_F
                ):
                    # Warm-side NO: running daily min is already above the band ceiling.
                    # Enter after the per-city/month P75 trough time (typically 6–9 AM
                    # in summer), which is when 75% of overnight lows have been set.
                    # P75 from LOW_P75_MINUTES is used when available and < noon (720 min);
                    # otherwise falls back to BAND_ARB_LOW_CEIL_MIN_HOUR (default 6 AM).
                    # Upper bound: BAND_ARB_LOW_CEIL_MAX_HOUR (default 3 PM) prevents
                    # next-night cooling risk. Same-day market only (hours_to_close < 20).
                    _low_tz_key = parsed.metric.replace("temp_low_", "temp_high_")
                    _low_tz_info = CITIES.get(_low_tz_key)
                    _low_local_now = (
                        datetime.now(_low_tz_info[3])
                        if _low_tz_info is not None
                        else datetime.now()
                    )
                    _low_local_hour = _low_local_now.hour
                    _low_local_minutes = _low_local_hour * 60 + _low_local_now.minute
                    # Resolve minimum entry time: P75 if available and a morning value,
                    # else fall back to BAND_ARB_LOW_CEIL_MIN_HOUR * 60.
                    _p75_low = _BAND_ARB_LOW_P75_MINUTES.get(parsed.metric, {}).get(_low_local_now.month)
                    _min_entry_minutes = (
                        _p75_low
                        if _p75_low is not None and _p75_low < 720
                        else BAND_ARB_LOW_CEIL_MIN_HOUR * 60
                    )
                    _close_time_str_warm = mkt.get("close_time") or mkt.get("expiration_time", "")
                    _warm_htc = _hours_to_close(_close_time_str_warm)
                    if (
                        _warm_htc is not None
                        and _warm_htc < 20
                        and _low_local_minutes >= _min_entry_minutes
                        and _low_local_hour <= BAND_ARB_LOW_CEIL_MAX_HOUR
                    ):
                        is_warm_no = True
                        band_ceil = parsed.strike_hi

            elif parsed.direction == "over":
                # "Over X°F": resolves NO when official low ≤ X.
                # Same 0.5°F rounding buffer as "between".
                if parsed.strike is not None and observed_max <= parsed.strike - BAND_ARB_LOW_BUFFER_F:
                    is_definitive_no = True
                    band_ceil = parsed.strike

        # ASOS shadow: record every KXHIGH ASOS crossing before gates filter them.
        # Mirrors the warm-NO shadow pattern for KXLOWT blocked cities — lets us
        # measure ungated ASOS signal performance vs. gated real trades.
        if is_definitive_no and not is_low_market:
            signals.append(BandArbSignal(
                metric=parsed.metric,
                ticker=ticker,
                yes_bid=yes_bid,
                no_ask=no_ask,
                observed_max=observed_max,
                band_ceil=band_ceil,
                direction=parsed.direction,
                city=mkt.get("subtitle", "") or ticker,
                side="no",
                hrrr_val=(hrrr_values or {}).get(parsed.metric),
                corr_status="asos_shadow",
                shadow=True,
            ))

        # --- LOW-market NOAA confirmation guard ----------------------------
        # For KXLOWT NO signals: if noaa_observed has fresh data and shows
        # the running-min is within NWS rounding distance of the floor, the
        # official rounded value could still be ≥ the floor (YES territory).
        # Threshold is band_ceil - 0.5: values ≥ 24.5 round to ≥ 25 under
        # NWS integer rounding, so a 24.6°F NOAA reading on a 25°F floor
        # means the official low may resolve 25°F → inside band → YES wins.
        # This mirrors the +0.5 buffer used on the METAR side for HIGH markets.
        if is_definitive_no and is_low_market and noaa_obs_values is not None:
            _noaa_low_confirm = noaa_obs_values.get(parsed.metric)
            if _noaa_low_confirm is not None and _noaa_obs_bounds(_noaa_low_confirm)[1] >= band_ceil - 0.5:
                logging.warning(
                    "BandArb skip (LOW NOAA contradict): %s —"
                    " METAR_min=%.1f°F < floor=%.1f°F"
                    " but NOAA_min=%.1f°F >= floor-0.5=%.1f°F"
                    " (NWS rounds to ≥ floor; NOAA overrides METAR)",
                    ticker, observed_max, band_ceil,
                    _noaa_low_confirm, band_ceil - 0.5,
                )
                is_definitive_no = False

        if not is_definitive_no:
            # --- Warm-side NO: running daily min confirmed above band ceiling ---
            if is_warm_no:
                _warm_suffix = parsed.metric.replace("temp_low_", "")
                if _warm_suffix in BAND_ARB_LOW_WARM_NO_BLOCK_CITIES:
                    logging.debug(
                        "BandArb warm-NO shadow: %s — city '%s' in blocklist, tracking only",
                        ticker, _warm_suffix,
                    )
                    _warm_city_shadow = mkt.get("subtitle", "") or ticker
                    signals.append(BandArbSignal(
                        metric=parsed.metric,
                        ticker=ticker,
                        yes_bid=yes_bid,
                        no_ask=no_ask,
                        observed_max=observed_max,
                        band_ceil=parsed.strike_hi,
                        direction=parsed.direction,
                        city=_warm_city_shadow,
                        side="no",
                        hours_to_close=_warm_htc or 0.0,
                        strike_lo=parsed.strike_hi,
                        shadow=True,
                    ))
                    continue

                _warm_ceil_nws = band_ceil + 0.5  # NWS rounds to nearest integer; ceil+0.5 is the safe threshold

                # NOAA hard gate (required, same as band_arb YES):
                # The METAR station and NWS settlement station can differ.  NOAA
                # observed must also show the running min above the NWS-adjusted
                # ceiling so we know the settlement value can't round back into band.
                _noaa_warm = (noaa_obs_values or {}).get(parsed.metric)
                if _noaa_warm is None or _noaa_warm < _warm_ceil_nws:
                    logging.warning(
                        "BandArb warm-NO skip: %s — NOAA obs %s < ceil+0.5=%.1f°F"
                        " (NOAA corroboration required; METAR station may differ from settlement)",
                        ticker,
                        f"{_noaa_warm:.1f}°F" if _noaa_warm is not None else "absent",
                        _warm_ceil_nws,
                    )
                    continue

                # HRRR veto: if HRRR forecasts the low dropping to/below the
                # NWS-adjusted ceiling, the temperature hasn't locked yet — block.
                if hrrr_values is not None:
                    _hrrr_warm = hrrr_values.get(parsed.metric)
                    if _hrrr_warm is not None and _hrrr_warm < _warm_ceil_nws:
                        logging.warning(
                            "BandArb warm-NO skip (HRRR veto): %s —"
                            " HRRR=%.1f°F < ceil+0.5=%.1f°F"
                            " (HRRR predicts low will drop into band ceiling)",
                            ticker, _hrrr_warm, _warm_ceil_nws,
                        )
                        continue

                # NWS CLI veto: if the settlement source already published today's
                # preliminary low and it's at/below the ceiling, block.
                if BAND_ARB_NWS_CLIMO_VETO and nws_climo_values is not None:
                    _climo_warm = nws_climo_values.get(parsed.metric)
                    if _climo_warm is not None and _climo_warm <= band_ceil:
                        logging.warning(
                            "BandArb warm-NO skip (nws_climo veto): %s —"
                            " CLI=%.1f°F <= ceil=%.1f°F (settlement source in band)",
                            ticker, _climo_warm, band_ceil,
                        )
                        continue

                # GFS in-band veto: if the GFS daily-low forecast lands inside the
                # band (or within BAND_ARB_LOW_GFS_IN_BAND_BUFFER_F of either
                # boundary), block.  GFS predicting in-band means YES is likely;
                # we should not bet warm-side NO when GFS disagrees.
                # Note: this uses temp_low_* from gfs_morning_values (populated
                # only after the first open_meteo_gfs reading of the day, typically
                # ~18 UTC), so it only fires on late-afternoon/evening entries.
                _warm_gfs_min: float | None = None  # carried into signal for sizer
                if BAND_ARB_LOW_GFS_IN_BAND_BUFFER_F >= 0 and gfs_morning_values is not None:
                    _gfs_low_min = gfs_morning_values.get(parsed.metric)
                    if _gfs_low_min is not None:
                        _warm_gfs_min = _gfs_low_min
                        _band_lo = parsed.strike_lo if parsed.strike_lo is not None else (band_ceil - 1.0)
                        _buf = BAND_ARB_LOW_GFS_IN_BAND_BUFFER_F
                        if _band_lo - _buf <= _gfs_low_min <= band_ceil + _buf:
                            logging.warning(
                                "BandArb warm-NO skip (GFS in-band): %s —"
                                " GFS_low=%.1f°F in band [%.1f,%.1f]±%.1f°F",
                                ticker, _gfs_low_min, _band_lo, band_ceil, _buf,
                            )
                            continue

                # Cap: above 85¢ only 5–15¢ upside remains — skip.
                if no_ask > BAND_ARB_LOW_CEIL_MAX_NO_ASK:
                    logging.debug(
                        "BandArb warm-NO skip %s: no_ask=%d¢ > max=%d¢",
                        ticker, no_ask, BAND_ARB_LOW_CEIL_MAX_NO_ASK,
                    )
                    continue

                # All gates passed — emit warm-NO signal
                _warm_corr = (
                    "metar_warm_noaa_corroborated"
                    if _noaa_warm >= _warm_ceil_nws + 1.0
                    else "metar_warm_confirmed"
                )
                _warm_city = mkt.get("subtitle", "") or ticker
                _hrrr_warm_val = (hrrr_values or {}).get(parsed.metric)
                signals.append(BandArbSignal(
                    metric=parsed.metric,
                    ticker=ticker,
                    yes_bid=yes_bid,
                    no_ask=no_ask,
                    observed_max=observed_max,
                    band_ceil=band_ceil,
                    direction=parsed.direction,
                    city=_warm_city,
                    side="no",
                    hours_to_close=_warm_htc or 0.0,
                    strike_lo=band_ceil,
                    corr_status=_warm_corr,
                    hrrr_val=_hrrr_warm_val,
                    gfs_morning_f=_warm_gfs_min,
                ))
                logging.info(
                    "BandArb warm-NO %s: running_min=%.1f°F >= ceil=%.1f°F+%.1f°F"
                    ", NOAA=%.1f°F HRRR=%s (local_hour=%d, htc=%.1fh) — daytime KXLOWT NO",
                    ticker, observed_max, band_ceil,
                    BAND_ARB_LOW_CEIL_BUFFER_F, _noaa_warm,
                    f"{_hrrr_warm_val:.1f}°F" if _hrrr_warm_val is not None else "absent",
                    _low_local_hour, _warm_htc or 0.0,
                )
                continue

            # --- KXLOWT YES: running daily min inside the band --------------
            # Buy YES when METAR + NOAA confirm the running min is inside the
            # band. NOT symmetric to KXHIGHT: unlike the daily high (which can
            # only rise), the overnight low can still drop with a cold front.
            # Entry only allowed within BAND_ARB_LOW_YES_MAX_HTC hours of close
            # (overnight risk window substantially passed) and is_locked=False.
            if (
                BAND_ARB_LOW_YES_ENABLED
                and is_low_market
                and parsed.direction == "between"
                and parsed.strike_lo is not None
                and parsed.strike_hi is not None
            ):
                in_band_low = parsed.strike_lo <= observed_max <= parsed.strike_hi
                if in_band_low:
                    _low_yes_tz_key = parsed.metric.replace("temp_low_", "temp_high_")
                    _low_yes_tz_info = CITIES.get(_low_yes_tz_key)
                    _low_yes_local_hour = (
                        datetime.now(_low_yes_tz_info[3]).hour
                        if _low_yes_tz_info is not None else datetime.now().hour
                    )
                    _low_yes_locked = _low_yes_local_hour >= BAND_ARB_LOW_CEIL_MIN_HOUR
                    _low_yes_close_str = mkt.get("close_time") or mkt.get("expiration_time", "")
                    _low_yes_htc = _hours_to_close(_low_yes_close_str)
                    if _low_yes_htc is not None and _low_yes_htc < BAND_ARB_LOW_YES_MAX_HTC and _low_yes_locked:
                        _low_yes_ask_raw = mkt.get("yes_ask")
                        if _low_yes_ask_raw is not None:
                            _low_yes_ask = int(_low_yes_ask_raw)
                            if BAND_ARB_LOW_YES_MIN_YES_ASK <= _low_yes_ask <= BAND_ARB_LOW_YES_MAX_YES_ASK:
                                noaa_val_low = (noaa_obs_values or {}).get(parsed.metric)
                                if noaa_val_low is None:
                                    logging.debug(
                                        "BandArb LOW-YES skip: %s — NOAA absent (required)", ticker
                                    )
                                elif abs(observed_max - noaa_val_low) > BAND_ARB_YES_MAX_DIVERGENCE_F:
                                    logging.warning(
                                        "BandArb LOW-YES skip: %s sensor mismatch"
                                        " METAR_min=%.1f NOAA=%.1f diff=%.1f°F > %.1f°F",
                                        ticker, observed_max, noaa_val_low,
                                        abs(observed_max - noaa_val_low),
                                        BAND_ARB_YES_MAX_DIVERGENCE_F,
                                    )
                                elif not (parsed.strike_lo - 0.5 <= noaa_val_low <= parsed.strike_hi + 0.5):
                                    logging.debug(
                                        "BandArb LOW-YES skip: %s — NOAA=%.1f°F outside band"
                                        " [%.1f–%.1f] with rounding buffer",
                                        ticker, noaa_val_low, parsed.strike_lo, parsed.strike_hi,
                                    )
                                else:
                                    _ly_lb, _ = _noaa_obs_bounds(noaa_val_low)
                                    _ly_corr = (
                                        "metar_noaa_corroborated"
                                        if _ly_lb >= parsed.strike_lo
                                        else "metar_noaa_lagging"
                                    )
                                    logging.info(
                                        "BandArb LOW-YES: %s  obs_min=%.1f°F"
                                        " in [%.1f–%.1f]  NOAA=%.1f°F"
                                        "  YES_ask=%d¢  %.1fh to close",
                                        ticker, observed_max,
                                        parsed.strike_lo, parsed.strike_hi,
                                        noaa_val_low, _low_yes_ask, _low_yes_htc,
                                    )
                                    signals.append(BandArbSignal(
                                        metric=parsed.metric,
                                        ticker=ticker,
                                        yes_bid=yes_bid,
                                        no_ask=no_ask,
                                        observed_max=observed_max,
                                        band_ceil=parsed.strike_hi,
                                        direction=parsed.direction,
                                        city=mkt.get("subtitle", "") or ticker,
                                        side="yes",
                                        yes_ask=_low_yes_ask,
                                        hours_to_close=_low_yes_htc,
                                        is_locked=False,
                                        strike_lo=parsed.strike_lo,
                                        noaa_val=noaa_val_low,
                                        hrrr_val=(hrrr_values or {}).get(parsed.metric),
                                        corr_status=_ly_corr,
                                        yes_ask_entry=_low_yes_ask,
                                    ))
                                    continue

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
                logging.debug(
                    "BandArb YES skip: %s — obs=%.1f°F outside band [%.1f–%.1f] (buf=%.1f)",
                    ticker, observed_max, parsed.strike_lo + buf, parsed.strike_hi - buf, buf,
                )
                continue

            # Determine if we're past the P75 daily-high lock point
            _lookup_metric_yes = parsed.metric.replace("temp_low_", "temp_high_")
            _city_info_yes = CITIES.get(_lookup_metric_yes)
            if _city_info_yes is None:
                continue
            _city_tz_yes = _city_info_yes[3]
            locked = _is_past_lock(_city_tz_yes, metric=parsed.metric)

            # GFS morning gate (needed for both time gate and sizer)
            _gfs_f = (gfs_morning_values or {}).get(parsed.metric)
            _gfs_lagging: bool | None = None
            if _gfs_f is not None:
                _gfs_lagging = observed_max < _gfs_f

            # Rising flag for early-entry gate
            _is_rising_now = (metar_is_rising or {}).get(parsed.metric)

            # Time gate: three paths allowed —
            #   (a) Locked (past P75): always fire.
            #   (b) Pre-lock, within BAND_ARB_YES_MAX_HOURS_PRELOCK of close: fire.
            #   (c) Early entry: within BAND_ARB_YES_EARLY_ENTRY_MINUTES of P75 lock,
            #       AND overshot (running max ≥ GFS morning forecast) AND still rising.
            #       Backtest: P75-1h overshot+rising+price≥40¢ → 95.8% WR, +25.7¢.
            close_time_str = mkt.get("close_time") or mkt.get("expiration_time", "")
            htc = _hours_to_close(close_time_str)
            if htc is None:
                continue
            if not locked:
                _mins_to_lock = _minutes_until_lock(_city_tz_yes, metric=parsed.metric)
                _early_entry = (
                    BAND_ARB_YES_EARLY_ENTRY_ENABLED
                    and _mins_to_lock is not None
                    and _mins_to_lock <= BAND_ARB_YES_EARLY_ENTRY_MINUTES
                    and _gfs_lagging is False   # overshot: running max ≥ GFS forecast
                    and _is_rising_now is True  # spot still at/near running max
                )
                if not _early_entry and htc > BAND_ARB_YES_MAX_HOURS_PRELOCK:
                    continue
            else:
                _early_entry = False

            # YES ask pricing gate — lower minimum for early entries (market not yet fully priced)
            yes_ask_raw = mkt.get("yes_ask")
            if yes_ask_raw is None:
                continue
            yes_ask_int = int(yes_ask_raw)
            _min_ask = BAND_ARB_YES_EARLY_MIN_YES_ASK if _early_entry else BAND_ARB_YES_MIN_YES_ASK
            if yes_ask_int < _min_ask or yes_ask_int > BAND_ARB_YES_MAX_YES_ASK:
                logging.debug(
                    "BandArb YES skip: %s — yes_ask=%d outside [%d, %d]%s",
                    ticker, yes_ask_int, _min_ask, BAND_ARB_YES_MAX_YES_ASK,
                    " (early-entry)" if _early_entry else "",
                )
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
                "  YES_ask=%d¢  %.1fh to close%s",
                "LOCKED" if locked else ("early-entry" if _early_entry else "pre-lock"),
                ticker, observed_max, parsed.strike_lo, parsed.strike_hi, yes_ask_int, htc,
                f"  [overshot+rising, {_minutes_until_lock(_city_tz_yes, metric=parsed.metric)}min to lock]"
                if _early_entry else "",
            )
            # YES always requires NOAA corroboration (checked above).
            # Use lower bound of noaa_val_yes: "corroborated" only if even the
            # lowest plausible NOAA reading has cleared the band ceiling.
            _yes_lb, _ = _noaa_obs_bounds(noaa_val_yes)
            if _yes_lb >= parsed.strike_hi:  # type: ignore[operator]
                _yes_corr_status = "metar_noaa_corroborated"
            else:
                _yes_corr_status = "metar_noaa_lagging"
            if _gfs_f is not None:
                logging.debug(
                    "BandArb YES GFS gate %s: obs=%.1f°F  gfs_morning=%.1f°F  %s",
                    ticker, observed_max, _gfs_f,
                    "LAGGING (1-contract cap)" if _gfs_lagging else "OVERSHOT (full Kelly)",
                )
            signals.append(BandArbSignal(
                metric=parsed.metric, ticker=ticker, yes_bid=yes_bid, no_ask=no_ask,
                observed_max=observed_max, band_ceil=parsed.strike_hi,
                direction=parsed.direction, city=city_yes,
                side="yes", yes_ask=yes_ask_int, hours_to_close=htc,
                is_locked=locked, strike_lo=parsed.strike_lo,
                noaa_val=noaa_val_yes,
                hrrr_val=(hrrr_values or {}).get(parsed.metric),
                nws_climo_val=(nws_climo_values or {}).get(parsed.metric),
                corr_status=_yes_corr_status,
                yes_ask_entry=yes_ask_int,
                gfs_morning_f=_gfs_f,
                gfs_lagging=_gfs_lagging,
                is_rising=(metar_is_rising or {}).get(parsed.metric),
            ))
            continue

        # --- NWS CLI (nws_climo) hard veto ------------------------------------
        # The NWS Climatological Report is the exact settlement source Kalshi
        # uses.  If today's preliminary CLI value is available and contradicts
        # the NO signal (CLI places the temperature on the YES side of the
        # strike), block the signal — the settlement data overrides METAR.
        #   HIGH: nws_climo_val < band_ceil → CLI max is below ceiling → YES wins.
        #   LOW:  nws_climo_val > band_ceil → CLI min is above floor   → YES wins.
        # No action when nws_climo confirms NO (value on correct side) — the
        # existing corroboration path handles that case.
        if BAND_ARB_NWS_CLIMO_VETO and nws_climo_values is not None:
            _climo_val = nws_climo_values.get(parsed.metric)
            if _climo_val is not None:
                _climo_contradicts = (
                    _climo_val < band_ceil if not is_low_market
                    else _climo_val > band_ceil
                )
                if _climo_contradicts:
                    logging.warning(
                        "BandArb skip (nws_climo veto): %s —"
                        " METAR=%.1f°F, CLI=%.1f°F %s band_ceil=%.1f°F"
                        " (settlement source contradicts signal)",
                        ticker, observed_max, _climo_val,
                        "<" if not is_low_market else ">", band_ceil,
                    )
                    continue

        # --- HRRR contradiction veto ---------------------------------------
        # If HRRR forecast is more than BAND_ARB_HRRR_CONTRADICT_F below the
        # METAR observed max AND HRRR is also below the band ceiling (HRRR
        # predicts YES will win), the two sources directly contradict each
        # other.  The most common cause is stale METAR carry at midnight: the
        # previous day's peak persists in the running-max until the new day's
        # first observation overwrites it.  Block rather than risk trading on
        # yesterday's temperature against a market that is correctly priced for
        # today's cold forecast.
        if BAND_ARB_HRRR_CONTRADICT_F > 0 and hrrr_values is not None:
            _hrrr_val = hrrr_values.get(parsed.metric)
            if (
                _hrrr_val is not None
                and _hrrr_val < band_ceil
                and (observed_max - _hrrr_val) > BAND_ARB_HRRR_CONTRADICT_F
            ):
                logging.warning(
                    "BandArb skip (HRRR contradiction): %s —"
                    " METAR=%.1f°F vs HRRR=%.1f°F"
                    " (gap=%.1f°F > %.1f°F threshold),"
                    " HRRR below ceil=%.1f°F — METAR likely stale carry",
                    ticker, observed_max, _hrrr_val,
                    observed_max - _hrrr_val, BAND_ARB_HRRR_CONTRADICT_F,
                    band_ceil,
                )
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
        # Date-alignment check: if the NOAA observation is dated for a different
        # local calendar day than the market (stale carry at midnight), treat it
        # as absent so it can't corroborate a false signal on today's market.
        if noaa_val is not None and noaa_obs_dates and _mkt_date is not None:
            _noaa_date = noaa_obs_dates.get(parsed.metric)
            if _noaa_date is not None and _noaa_date != _mkt_date:
                logging.warning(
                    "BandArb: %s — NOAA data dated %s != market date %s"
                    " (stale carry); treating NOAA as absent",
                    ticker, _noaa_date, _mkt_date,
                )
                noaa_val = None
        if noaa_val is None:
            # NOAA has no data for this city yet — METAR is ahead, or NOAA has
            # no data at all (early morning before first observations arrive).
            # For LOW markets apply an extra buffer: without NOAA confirmation
            # a QC-rejected METAR spike can't be ruled out.  The combined
            # effective floor is band_ceil - (BUFFER_F + NOAA_ABSENT_BUFFER_F).
            if is_low_market and BAND_ARB_LOW_NOAA_ABSENT_BUFFER_F > 0:
                # Stack on top of the initial BAND_ARB_LOW_BUFFER_F already
                # applied: effective floor = band_ceil - BUFFER_F - ABSENT_BUFFER_F.
                # e.g. floor=25, BUFFER=0.5, ABSENT=2.0 → must be ≤ 22.5°F.
                _absent_floor = band_ceil - BAND_ARB_LOW_BUFFER_F - BAND_ARB_LOW_NOAA_ABSENT_BUFFER_F
                if observed_max > _absent_floor:
                    logging.debug(
                        "BandArb skip: %s — NOAA absent, METAR_min=%.1f°F"
                        " > absent-buffer floor=%.1f°F"
                        " (floor=%.1f - %.1f - %.1f°F);"
                        " waiting for NOAA confirmation",
                        ticker, observed_max, _absent_floor,
                        band_ceil, BAND_ARB_LOW_BUFFER_F,
                        BAND_ARB_LOW_NOAA_ABSENT_BUFFER_F,
                    )
                    continue
            # NOAA day-1 forecast veto: NOAA observed is absent/stale, so the
            # only observational source is METAR.  If the NWS day-1 forecast
            # says the official value will land on the YES side of the strike
            # (day1 < band_ceil for HIGH; day1 > band_ceil for LOW), the
            # forecast directly contradicts the observed crossing.  This is
            # the most reliable signal that METAR is reading a different
            # station from Kalshi's resolution station (inter-station gap) or
            # has a QC-failed outlier reading.  Block rather than trade.
            if BAND_ARB_NOAA_DAY1_VETO and noaa_day1_values is not None:
                _day1_val = noaa_day1_values.get(parsed.metric)
                if _day1_val is not None:
                    _day1_contradicts = (
                        _day1_val < band_ceil if not is_low_market
                        else _day1_val > band_ceil
                    )
                    if _day1_contradicts:
                        logging.warning(
                            "BandArb skip (day1 veto): %s —"
                            " METAR=%.1f°F, NOAA observed absent/stale,"
                            " day1=%.1f°F %s band_ceil=%.1f°F"
                            " (forecast says YES wins)",
                            ticker, observed_max, _day1_val,
                            "<" if not is_low_market else ">", band_ceil,
                        )
                        continue
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
            # Require the conservative lower bound to clear the trigger threshold:
            # if even the lowest plausible NOAA reading is below band_ceil - 0.5°F,
            # NOAA cannot confirm the temperature is genuinely above the trigger.
            _noaa_lb, _ = _noaa_obs_bounds(noaa_val)
            if _noaa_lb < band_ceil - 0.5:
                logging.warning(
                    "BandArb: %s — NOAA=%.2f°F (lb=%.2f°F) below trigger"
                    " threshold %.1f°F; treating NOAA as absent",
                    ticker, noaa_val, _noaa_lb, band_ceil - 0.5,
                )
                noaa_val = None
                _corr_label = "  METAR-only (NOAA lb below trigger)"
            elif _noaa_lb >= band_ceil:
                _corr_label = "  METAR+NOAA corroborated"
            else:
                _corr_label = (
                    f"  METAR+NOAA-lagging"
                    f" (NOAA_lb={_noaa_lb:.2f}°F < ceil={band_ceil:.1f}°F)"
                )

        city = mkt.get("subtitle", "") or ticker
        logging.info(
            "BandArb signal: %s  obs=%.1f°F > ceil=%.1f°F  NO_ask=%d¢  (%s%s)",
            ticker, observed_max, band_ceil, no_ask, parsed.direction,
            _corr_label,
        )

        # Derive corroboration status for logging
        if noaa_val is None:
            _corr_status = "metar_only"
        elif noaa_val >= band_ceil:
            _corr_status = "metar_noaa_corroborated"
        else:
            _corr_status = "metar_noaa_lagging"

        # NWS CLI confirms NO if it's on the correct side of the ceiling
        _climo_val_for_signal = (nws_climo_values or {}).get(parsed.metric)

        # YES ask from market (needed for spread/market_p logging)
        _yes_ask_no = int(mkt.get("yes_ask") or 0)

        # HRRR value at signal time
        _hrrr_val_for_signal = (hrrr_values or {}).get(parsed.metric)

        signals.append(BandArbSignal(
            metric=parsed.metric,
            ticker=ticker,
            yes_bid=yes_bid,
            no_ask=no_ask,
            observed_max=observed_max,
            band_ceil=band_ceil,
            direction=parsed.direction,
            city=city,
            noaa_val=noaa_val,
            hrrr_val=_hrrr_val_for_signal,
            nws_climo_val=_climo_val_for_signal,
            corr_status=_corr_status,
            yes_ask_entry=_yes_ask_no,
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
    # CRITICAL: only include today's forecast (forecast_offset == 0).
    # Multi-day DataPoints (day+1, day+2 …) target different dates and have
    # progressively warmer/cooler values as the season changes.  Including them
    # in the spread computation produces artificially large spreads (e.g. today's
    # NWS forecast for BOS=55°F vs day+5=70°F → 15°F "model spread") that have
    # nothing to do with inter-model disagreement and block all ForecastNO signals.
    from collections import defaultdict
    src_values: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for dp in data_points:
        if dp.source not in _FORECAST_NO_SOURCES:
            continue
        if not dp.metric.startswith(("temp_high", "temp_low")):
            continue
        if (dp.metadata or {}).get("forecast_offset", 0) != 0:
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

    # Build observed-maximum lookup from metar/nws_asos DataPoints (temp_high_*).
    # Used to veto HIGH-market signals where the running daily maximum has
    # already crossed the strike — at that point band_arb handles the signal
    # and forecast_no would be a duplicate at a worse price.
    # nws_asos (5-min cadence) supplements METAR (20–60 min cadence).
    obs_high_values: dict[str, float] = {}
    for dp in data_points:
        if dp.source not in ("metar", "nws_asos"):
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
        yes_ask_fno: int = int(mkt.get("yes_ask") or 0)

        # Tighter spread gate for KXLOWT forecast_no (pure model signal, no
        # observed confirmation — illiquid spreads inflate round-trip cost).
        if is_low_market and FORECAST_NO_LOWT_MAX_SPREAD_CENTS > 0:
            _yes_ask = mkt.get("yes_ask")
            if _yes_ask is not None:
                _fno_spread = _yes_ask - yes_bid
                if _fno_spread > FORECAST_NO_LOWT_MAX_SPREAD_CENTS:
                    logging.debug(
                        "ForecastNO skip: %s — KXLOWT spread %d¢ > %d¢",
                        ticker, _fno_spread, FORECAST_NO_LOWT_MAX_SPREAD_CENTS,
                    )
                    continue

        # Price gate: only enter if market hasn't priced it in yet, and the
        # NO price still implies genuine uncertainty (not near-certain YES).
        if no_ask <= 0 or no_ask < FORECAST_NO_MIN_ASK or no_ask > FORECAST_NO_MAX_ASK:
            continue

        # Compute city-local hour for time-gated checks below.
        # LOW markets use temp_low_* metrics; CITIES keys use temp_high_*, so translate.
        _lh_metric = parsed.metric.replace("temp_low_", "temp_high_")
        _lh_tz_info = CITIES.get(_lh_metric)
        _local_now = datetime.now(_lh_tz_info[3]) if _lh_tz_info is not None else datetime.now()
        _local_hour = _local_now.hour
        _local_month = _local_now.month

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
                    # Pre-dawn proximity veto: from OBS_MIN_HOUR through lock hour,
                    # the required margin scales with hours remaining before dawn —
                    # lenient at midnight, tight near 5 AM (see constant comments).
                    if FORECAST_NO_LOW_OVER_OBS_MIN_HOUR <= _local_hour < FORECAST_NO_OVERNIGHT_LOCK_HOUR:
                        _hours_to_lock = FORECAST_NO_OVERNIGHT_LOCK_HOUR - _local_hour - 1
                        _scaled_margin = (
                            FORECAST_NO_LOW_OVER_OBS_MARGIN_F
                            + FORECAST_NO_LOW_OVER_OBS_MARGIN_PER_HOUR * _hours_to_lock
                        )
                        if _obs_min > parsed.strike + _scaled_margin:
                            logging.debug(
                                "ForecastNO veto (over pre-dawn): %s — local_hour=%d"
                                " obs_min=%.1f°F > strike+margin=%.1f°F"
                                " (margin=%.1f + %.1f×%dh; %.1f°F drop implausible before dawn)",
                                ticker, _local_hour, _obs_min,
                                parsed.strike + _scaled_margin,
                                FORECAST_NO_LOW_OVER_OBS_MARGIN_F,
                                FORECAST_NO_LOW_OVER_OBS_MARGIN_PER_HOUR,
                                _hours_to_lock,
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
                    # Proximity veto: observed min within FORECAST_NO_LOW_OBS_MARGIN_F
                    # of the band ceiling.  Any further overnight cooling could push the
                    # daily low into the band → YES resolution.  Mirrors the same gate
                    # that exists for direction=="under".
                    # Trade #145: obs_min=46°F, strike_hi=45°F — only 1°F above ceiling
                    # at 3:32 AM EDT with 2+ hours until dawn; temperature fell into the
                    # band by sunrise despite HRRR/NOAA forecasting 48-53°F.
                    # NWS CLI rounding: effective band ceiling = strike_hi + BAND_ARB_LOW_BUFFER_F
                    # (e.g. strike_hi=45°F → ceiling=45.5°F, since temps in [44.5,45.5) round to 45).
                    _effective_ceiling = parsed.strike_hi + BAND_ARB_LOW_BUFFER_F
                    if _obs_min < _effective_ceiling + FORECAST_NO_LOW_OBS_MARGIN_F:
                        logging.info(
                            "ForecastNO veto (between proximity): %s — observed_min=%.1f°F"
                            " within %.1f°F of effective_ceiling=%.1f°F"
                            " (strike_hi=%.1f + rounding=%.1f; overnight cooling risk)",
                            ticker, _obs_min, FORECAST_NO_LOW_OBS_MARGIN_F,
                            _effective_ceiling, parsed.strike_hi, BAND_ARB_LOW_BUFFER_F,
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

        # Midnight METAR gap veto: no METAR observation yet for this LOW "under"
        # market during the overnight window.  The daily running-minimum resets at
        # midnight, leaving a brief window (typically ~15 min) where obs_min is None
        # but the temperature may already be near the strike — the proximity veto
        # cannot run without data.  Block until the first METAR reading arrives.
        if (
            is_low_market
            and parsed.direction == "under"
            and parsed.strike is not None
            and obs_low_values.get(parsed.metric) is None
            and FORECAST_NO_LOW_UNDER_REQUIRE_METAR
            and _local_hour < FORECAST_NO_OVERNIGHT_LOCK_HOUR
        ):
            logging.debug(
                "ForecastNO veto (no METAR overnight): %s — no observed min for %s"
                " at local hour %d (midnight gap, proximity veto cannot run)",
                ticker, parsed.metric, _local_hour,
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
                    # "Too cold" branch: if observed max already at/above the band floor,
                    # the daily high is proved >= strike_lo — "too cold" NO is impossible.
                    # The daily maximum never decreases, so any METAR reading >= strike_lo
                    # guarantees the final official high will be in or above the band.
                    if _obs_max >= parsed.strike_lo:
                        logging.debug(
                            "ForecastNO veto (too-cold disproven): %s —"
                            " observed_max=%.1f°F >= strike_lo=%.1f°F"
                            " (daily max proved ≥ band floor, too-cold NO impossible)",
                            ticker, _obs_max, parsed.strike_lo,
                        )
                        continue

        # Compute per-source edge for this market's band direction
        sources_map = src_values.get(parsed.metric, [])
        if not sources_map:
            continue

        # Apply per-source/city/month bias correction (10-year Open-Meteo backtest).
        # Sources not in _OM_BIAS (hrrr, nws_hourly, noaa) pass through unchanged.
        _city_key = parsed.metric.replace("temp_high_", "").replace("temp_low_", "")
        corr_sources_map = [
            (s, v - _OM_BIAS.get((s, _city_key, _local_month), 0.0))
            for s, v in sources_map
        ]

        # Hoist HRRR values so both veto blocks below can share them.
        hrrr_vals = [v for s, v in corr_sources_map if s == "hrrr"]
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
            other_vals = [v for s, v in corr_sources_map if s in ("nws_hourly", "open_meteo")]
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

        # --- Global model-spread gate (all directions) --------------------------
        # If the spread between the highest and lowest model forecast exceeds
        # FORECAST_NO_MODEL_SPREAD_F, models are too far apart to trust any
        # single signal — block regardless of which direction the market faces.
        # This catches band/"between" markets that the HRRR-specific spread veto
        # (which only runs for direction=="under") would otherwise miss.
        if FORECAST_NO_MODEL_SPREAD_F > 0 or FORECAST_NO_MODEL_SPREAD_MIN_F > 0:
            _spread_sources = _FORECAST_NO_SOURCES
            _spread_vals = [v for s, v in corr_sources_map if s in _spread_sources]
            if len(_spread_vals) >= 2:
                _model_spread = max(_spread_vals) - min(_spread_vals)
                if FORECAST_NO_MODEL_SPREAD_F > 0 and _model_spread > FORECAST_NO_MODEL_SPREAD_F:
                    logging.warning(
                        "ForecastNO skip (model spread too wide): %s —"
                        " spread=%.1f°F > %.1f°F across %d models"
                        " (forecast uncertainty too high)",
                        ticker, _model_spread, FORECAST_NO_MODEL_SPREAD_F,
                        len(_spread_vals),
                    )
                    continue
                if FORECAST_NO_MODEL_SPREAD_MIN_F > 0 and _model_spread < FORECAST_NO_MODEL_SPREAD_MIN_F:
                    logging.debug(
                        "ForecastNO skip (model spread too tight): %s —"
                        " spread=%.1f°F < %.1f°F (consensus already priced in)",
                        ticker, _model_spread, FORECAST_NO_MODEL_SPREAD_MIN_F,
                    )
                    continue

        # qualifying: (source, value, edge_F, no_dir)
        # no_dir is "NO_HIGH" / "NO_LOW" for "between" bands; None for under/over.
        qualifying: list[tuple[str, float, float, str | None]] = []
        for source, value in corr_sources_map:
            no_dir: str | None = None
            if not is_low_market:
                # HIGH market: NO settles when forecast misses the strike.
                if parsed.direction == "between":
                    # Track WHICH side the source is on — critical for consensus.
                    # A source above strike_hi and one below strike_lo disagree on
                    # the actual temperature and must NOT both count toward consensus.
                    edge_warm = (value - parsed.strike_hi) if parsed.strike_hi is not None else float("-inf")
                    edge_cold = (parsed.strike_lo - value) if parsed.strike_lo is not None else float("-inf")
                    if edge_warm >= edge_cold:
                        edge   = edge_warm
                        no_dir = "NO_HIGH"
                    else:
                        edge   = edge_cold
                        no_dir = "NO_LOW"
                elif parsed.direction == "under" and parsed.strike is not None:
                    edge = value - parsed.strike
                elif parsed.direction == "over" and parsed.strike is not None:
                    edge = parsed.strike - value
                else:
                    continue
            else:
                # LOW market: NO settles when the daily low stays ABOVE the strike.
                if parsed.direction == "between" and parsed.strike_hi is not None:
                    edge = value - parsed.strike_hi  # forecast too warm for YES
                    no_dir = "NO_HIGH"  # qualifying sources are always warm-side for KXLOWT
                elif parsed.direction == "under" and parsed.strike is not None:
                    edge = value - parsed.strike
                elif parsed.direction == "over" and parsed.strike is not None:
                    edge = parsed.strike - value
                else:
                    continue

            # Direction-specific edge threshold for "between" NO signals.
            # Per-source overrides apply first (hrrr/nws_hourly have lower MAE).
            if source in _FORECAST_NO_SOURCE_MIN_EDGE:
                _edge_required = _FORECAST_NO_SOURCE_MIN_EDGE[source]
            elif no_dir == "NO_HIGH":
                _edge_required = FORECAST_NO_NO_HIGH_MIN_EDGE_F
            elif no_dir == "NO_LOW":
                _edge_required = FORECAST_NO_NO_LOW_MIN_EDGE_F
            else:
                _edge_required = FORECAST_NO_MIN_EDGE_F

            if edge >= _edge_required:
                qualifying.append((source, value, edge, no_dir))

        if not qualifying:
            continue

        # Directional consistency check for "between" bands.
        # If any qualifying sources say NO_HIGH and others say NO_LOW, the models
        # disagree on which side of the band the temperature will land — suppress.
        if parsed.direction == "between":
            _dirs = {d for _, _, _, d in qualifying if d is not None}
            if "NO_HIGH" in _dirs and "NO_LOW" in _dirs:
                logging.debug(
                    "ForecastNO skip (directional conflict): %s — "
                    "sources split NO_HIGH/NO_LOW (models disagree on temperature)",
                    ticker,
                )
                continue
            # All qualifying sources agree; record which direction
            _signal_no_direction: str | None = next(
                (d for _, _, _, d in qualifying if d is not None), None
            )
        else:
            _signal_no_direction = None

        # Count unique qualifying sources.  Deduplicate by source name so that
        # sources emitting multiple hourly forecasts per poll cannot inflate count.
        source_score = len({src for src, _, _, _ in qualifying})
        # KXLOWT uses a higher threshold — overnight lows harder to forecast.
        _fno_min_sources = (
            FORECAST_NO_LOWT_MIN_SOURCES
            if is_low_market and FORECAST_NO_LOWT_MIN_SOURCES > 0
            else FORECAST_NO_MIN_SOURCES
        )
        if source_score < _fno_min_sources:
            logging.debug(
                "ForecastNO skip: %s — need %d source(s)%s, have %d",
                ticker, _fno_min_sources,
                " (KXLOWT)" if is_low_market else "",
                source_score,
            )
            continue

        # Require at least one Open-Meteo ensemble model.
        # NOAA/HRRR/NWS forecasts are already priced in by Kalshi market-makers;
        # signals with none of the international ensemble models carry no alpha.
        _open_meteo_family = {
            "open_meteo", "open_meteo_gfs", "open_meteo_ecmwf",
            "open_meteo_icon", "open_meteo_gem",
        }
        _qualifying_sources = {s for s, _, _, _ in qualifying}
        if FORECAST_NO_REQUIRE_OPEN_METEO:
            if not (_qualifying_sources & _open_meteo_family):
                logging.debug(
                    "ForecastNO skip: %s — no open_meteo model in qualifying"
                    " sources %s (NOAA/HRRR/NWS already priced in by market)",
                    ticker, sorted(_qualifying_sources),
                )
                continue

        # Daytime gate for pure open_meteo-only signals.
        # Before 7 AM local, open_meteo-only signals win 50% (coin flip) — the
        # temperature hasn't developed and all models are equally uncertain.
        # Mixed signals (open_meteo + NOAA/HRRR both agree) are exempt: 71% overnight.
        if FORECAST_NO_OPEN_METEO_DAYTIME_HOUR > 0:
            _trad_sources = {"hrrr", "nws_hourly", "noaa"}
            _is_open_meteo_only = bool(_qualifying_sources & _open_meteo_family) and not bool(_qualifying_sources & _trad_sources)
            if _is_open_meteo_only:
                _fno_lookup = parsed.metric.replace("temp_low_", "temp_high_")
                _fno_city_info = CITIES.get(_fno_lookup)
                _fno_local_hour = (
                    datetime.now(_fno_city_info[3]).hour
                    if _fno_city_info is not None else datetime.now().hour
                )
                if _fno_local_hour < FORECAST_NO_OPEN_METEO_DAYTIME_HOUR:
                    logging.debug(
                        "ForecastNO skip: %s — open_meteo-only at local hour %d"
                        " < %d (daytime gate; mixed signals exempt)",
                        ticker, _fno_local_hour, FORECAST_NO_OPEN_METEO_DAYTIME_HOUR,
                    )
                    continue

        # Block all T-markets (threshold under/over) — only B-markets (between) qualify.
        # Backtest: between_only → 88.6% WR; T-markets drag overall WR to 66.5% at -3¢/trade.
        if FORECAST_NO_BETWEEN_ONLY and parsed.direction != "between":
            logging.debug(
                "ForecastNO skip: %s — direction=%s T-market blocked (between_only=True)",
                ticker, parsed.direction,
            )
            continue

        # Block threshold-over signals: "will high EXCEED T?" is structurally weak
        # in summer — models only need to undershoot by a few degrees for YES to win.
        if FORECAST_NO_BLOCK_THRESHOLD_OVER and parsed.direction == "over":
            logging.debug(
                "ForecastNO skip: %s — direction=over threshold market blocked"
                " (0/5 win rate historically; high temp overshoot risk)",
                ticker,
            )
            continue

        min_edge = min(e for _, _, e, _ in qualifying)
        max_edge = max(e for _, _, e, _ in qualifying)
        source_names = [s for s, _, _, _ in qualifying]

        # Score: blend of source count and edge magnitude, capped at 0.95.
        raw_score = min(0.95, 0.60 + 0.05 * source_score + 0.01 * min_edge)

        # p_estimate: direction-aware base (fallback if calibration model not loaded).
        # NO_HIGH is more reliable than NO_LOW per backtest (95%+ vs 80%+ at 2°F edge).
        # Source count bonus: each additional independent source raises confidence.
        _p_base = 0.80 if _signal_no_direction == "NO_HIGH" else 0.72
        p_estimate = min(0.95, _p_base + 0.015 * min_edge + 0.02 * source_score)

        # Compute actual model spread across all forecast sources for logging.
        _spread_sources_fno = _FORECAST_NO_SOURCES
        _spread_vals_fno = [v for s, v in corr_sources_map if s in _spread_sources_fno]
        _model_spread_fno: float | None = (
            round(max(_spread_vals_fno) - min(_spread_vals_fno), 1)
            if len(_spread_vals_fno) >= 2 else None
        )

        # Hours to close at signal time
        _close_time_str_fno = mkt.get("close_time") or mkt.get("expiration_time", "")
        _htc_fno = _hours_to_close(_close_time_str_fno)

        # Reject signals too far from close.  Backtest: ≥24h trades are -1639¢ at 54% WR;
        # ≤20h trades are +573¢ at 75% WR.  Skip if hours unknown (can't verify window).
        if _htc_fno is None or _htc_fno > FORECAST_NO_MAX_HOURS:
            logging.debug(
                "ForecastNO skip %s: hours_to_close=%.1f > max=%.1f",
                ticker, _htc_fno or -1, FORECAST_NO_MAX_HOURS,
            )
            continue

        # Calibration model: compute per-model edge vs actual Kalshi band ceiling.
        # Currently calibrated to filtered historical (35.7% WR) + live (67.7% WR).
        # Outputs ~0.50–0.65 for typical signals — conservative Kelly sizing vs hardcoded.
        _src_vals = dict(corr_sources_map)
        _cal_strike = parsed.strike_hi if _signal_no_direction == "NO_HIGH" else parsed.strike_lo
        _cal_sign   = 1.0             if _signal_no_direction == "NO_HIGH" else -1.0

        def _model_edge_f(src: str, _st=_cal_strike, _sg=_cal_sign) -> float:
            v = _src_vals.get(src)
            return round((v - _st) * _sg, 2) if v is not None else float("nan")

        _cal_n_sup = sum(
            1 for src in ("open_meteo_ecmwf", "open_meteo_icon", "open_meteo_gem", "hrrr")
            if not math.isnan(_model_edge_f(src)) and _model_edge_f(src) > 0
        )
        _cal_p = _cal_win_prob(
            edge_ecmwf=_model_edge_f("open_meteo_ecmwf"), edge_icon=_model_edge_f("open_meteo_icon"),
            edge_gem=_model_edge_f("open_meteo_gem"),      edge_hrrr=_model_edge_f("hrrr"),
            model_spread=_model_spread_fno or 0.0,         n_supporting=_cal_n_sup,
            is_no_high=_signal_no_direction == "NO_HIGH",  is_high_market=parsed.metric.startswith("temp_high_"),
            month_sin=math.sin(2*math.pi*_local_month/12), month_cos=math.cos(2*math.pi*_local_month/12),
        )
        if _cal_p is not None:
            p_estimate = _cal_p

        logging.info(
            "ForecastNO signal: %s  no_ask=%d¢  edge=%.1f°F  dir=%s  sources=%s  score=%.2f",
            ticker, no_ask, min_edge,
            _signal_no_direction or parsed.direction,
            source_names, raw_score,
        )

        # METAR observed temp and gap from effective strike at signal time.
        _obs_temp_fno: float | None = (
            obs_high_values.get(parsed.metric) if is_high_market
            else obs_low_values.get(parsed.metric)
        )
        _effective_strike_fno: float | None = (
            parsed.strike if parsed.direction in ("under", "over")
            else (parsed.strike_hi if _signal_no_direction == "NO_HIGH" else parsed.strike_lo)
        )
        _obs_gap_fno: float | None = (
            round(_obs_temp_fno - _effective_strike_fno, 1)
            if _obs_temp_fno is not None and _effective_strike_fno is not None
            else None
        )

        # source_details stores (source, value, edge) for backward compat with executor
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
            source_details=[(s, v, e) for s, v, e, _ in qualifying],
            yes_ask=yes_ask_fno,
            hours_to_close=_htc_fno,
            model_spread_f=_model_spread_fno,
            no_direction=_signal_no_direction,
            strike=parsed.strike,
            strike_lo=parsed.strike_lo,
            strike_hi=parsed.strike_hi,
            obs_temp_f=_obs_temp_fno,
            obs_gap_f=_obs_gap_fno,
        ))

    return signals
