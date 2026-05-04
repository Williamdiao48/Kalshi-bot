"""Backtest forecast-based YES/NO band-arb trading using GFS, ECMWF, and ICON.

For each resolved KXHIGH 'between' band, looks up the Open-Meteo model forecast
for the measurement day, generates YES or NO signals based on how the forecast
compares to the band bounds, then simulates trading with real Kalshi candlestick
prices and a range of profit-take / stop-loss thresholds.

Signal logic (edge_min = minimum °F clearance OUTSIDE band required for NO signal):
  YES  strike_lo <= forecast <= strike_hi  (forecast inside band, any position)
  NO   forecast > strike_hi + edge_min  OR  forecast < strike_lo - edge_min  (clearly outside)
  skip forecast outside but within edge_min of nearest edge  (too close to call)

Entry at a fixed local hour (default 9 AM) using Kalshi candlestick ask/bid prices.
YES entry:  buy YES at yes_ask_cents
NO entry:   buy NO at no_ask_cents ≈ 100 − yes_bid_cents

Consensus modes (in addition to individual models):
  majority_vote  ≥2 of 3 models agree on direction (no YES/NO conflict)
  all_agree      all 3 models agree on same direction

Key data files (generate first if missing):
  venv/bin/python scripts/fetch_openmeteo_forecast_history.py --days 90
  venv/bin/python scripts/fetch_kxhigh_history.py --days 90

Usage:
  venv/bin/python scripts/backtest_forecast_accuracy.py
  venv/bin/python scripts/backtest_forecast_accuracy.py --model ecmwf_ifs
  venv/bin/python scripts/backtest_forecast_accuracy.py --entry-hour 9
  venv/bin/python scripts/backtest_forecast_accuracy.py --edge-min 1.5
  venv/bin/python scripts/backtest_forecast_accuracy.py --pt 0.40 --sl 0.40
  venv/bin/python scripts/backtest_forecast_accuracy.py --months 3 4 5
  venv/bin/python scripts/backtest_forecast_accuracy.py --cities dca chi bos
  venv/bin/python scripts/backtest_forecast_accuracy.py --no-cache
  venv/bin/python scripts/backtest_forecast_accuracy.py --out data/forecast_accuracy.txt
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.auth import generate_headers   # noqa: E402
from kalshi_bot.markets import KALSHI_API_BASE # noqa: E402
from kalshi_bot.news.noaa import CITIES        # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "band_arb_candle_cache.json"
_SEM       = asyncio.Semaphore(5)

_INDIVIDUAL_MODELS = ["gfs_seamless", "ecmwf_ifs", "icon_seamless", "gfs_hrrr", "gem_seamless"]

# Core global models used for majority/all-agree consensus voting.
# HRRR and GEM are individual-only (not included in consensus vote counts).
_CONSENSUS_MODELS  = ["gfs_seamless", "ecmwf_ifs", "icon_seamless"]

# any_model: any of the 5 individual models signals, no YES/NO conflict
# majority_vote: ≥2 of 3 core models agree (no conflict)
# all_agree: all 3 core models agree
_CONSENSUS_MODES   = ["any_model", "majority_vote", "all_agree"]

# PT/SL grid — None means hold to settlement
_PT_VALUES        = [0.25, 0.40, 0.60, 0.80, None]
_SL_VALUES        = [0.25, 0.40, 0.60, None]

# Fixed-cent SL grid — absolute loss in cents before stopping out.
# More meaningful for high-entry NO positions where percentage thresholds
# (e.g. 25% SL on 82¢ entry = exit at 61.5¢) represent huge dollar losses,
# and for low-entry YES positions where percentage SL is too tight.
_SL_CENTS_VALUES  = [5, 10, 15, 20, 25]

# Edge sweep — minimum °F clearance applied symmetrically to YES and NO.
# YES: forecast must be at least edge inside each boundary.
# NO:  forecast must be at least edge outside the band.
# Larger edge = fewer signals, higher conviction.
_EDGE_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Trades with entry_price >= NO_EDGE_THRESHOLD are filtered (market already
# fully priced-in, no arb available). Applies to both YES (yes_ask >= 95¢)
# and NO (100 − yes_bid >= 95¢, i.e., yes_bid ≤ 5¢) entries.
NO_EDGE_THRESHOLD = 95  # cents


# ── Data loading ──────────────────────────────────────────────────────────────

def load_forecasts() -> dict[tuple[str, str, str], float]:
    """Load openmeteo_forecasts.csv → {(metric, date, model): forecast_high_f}.

    The 'date' key is the LOCAL measurement date (kxhigh_bands date − 1 day).
    """
    path = DATA_DIR / "openmeteo_forecasts.csv"
    if not path.exists():
        log.error("openmeteo_forecasts.csv not found — run fetch_openmeteo_forecast_history.py")
        sys.exit(1)
    data: dict[tuple[str, str, str], float] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            data[(r["city_metric"], r["date"], r["model"])] = float(r["forecast_high_f"])
    log.info("Loaded %d forecast rows", len(data))
    return data


def load_bands() -> list[dict]:
    path = DATA_DIR / "kxhigh_bands.csv"
    if not path.exists():
        log.error("kxhigh_bands.csv not found — run fetch_kxhigh_history.py")
        sys.exit(1)
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            r["strike_lo"] = float(r["strike_lo"])
            r["strike_hi"] = float(r["strike_hi"])
            rows.append(r)
    return rows


def load_cache() -> dict[str, list[dict]]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict[str, list[dict]]) -> None:
    CACHE_FILE.write_text(json.dumps(cache))


# ── Candlestick fetch ─────────────────────────────────────────────────────────

async def fetch_candles(
    session: aiohttp.ClientSession,
    ticker: str,
    band_date_str: str,
) -> list[dict]:
    """Fetch hourly candlesticks.

    band_date_str is kxhigh_bands["date"] (= measurement_date + 1 day in UTC).
    The window prev_day 12:00 UTC → band_date 07:00 UTC covers the full
    measurement day in all US timezones.
    """
    band_date = datetime.strptime(band_date_str, "%Y-%m-%d").date()
    prev_day  = band_date - timedelta(days=1)
    start_ts  = int(datetime(prev_day.year, prev_day.month, prev_day.day,
                             12, 0, tzinfo=timezone.utc).timestamp())
    end_ts    = int(datetime(band_date.year, band_date.month, band_date.day,
                             7, 0, tzinfo=timezone.utc).timestamp())

    series  = ticker.rsplit("-", 2)[0]
    path    = f"/trade-api/v2/series/{series}/markets/{ticker}/candlesticks"
    headers = generate_headers("GET", path)
    params  = {"period_interval": 60, "start_ts": start_ts, "end_ts": end_ts}

    async with _SEM:
        try:
            async with session.get(
                f"{KALSHI_API_BASE}/series/{series}/markets/{ticker}/candlesticks",
                params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as r:
                if r.status == 429:
                    log.warning("Rate-limited %s — sleeping 3s", ticker)
                    await asyncio.sleep(3.0)
                    return []
                if r.status != 200:
                    return []
                data = await r.json()
        except Exception as exc:
            log.debug("Candle fetch error %s: %s", ticker, exc)
            return []
        await asyncio.sleep(0.2)

    return data.get("candlesticks", [])


def candles_to_hourly_ask(candles: list[dict], city_tz) -> dict[int, float]:
    """Return {local_hour: yes_ask_cents} from candlestick close prices."""
    result: dict[int, float] = {}
    for c in candles:
        v = c.get("yes_ask", {}).get("close_dollars")
        if v is None:
            continue
        try:
            cents = round(float(v) * 100)
        except (ValueError, TypeError):
            continue
        ts = datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc)
        result[ts.astimezone(city_tz).hour] = cents
    return result


def candles_to_hourly_bid(candles: list[dict], city_tz) -> dict[int, float]:
    """Return {local_hour: yes_bid_cents} from candlestick close prices."""
    result: dict[int, float] = {}
    for c in candles:
        v = c.get("yes_bid", {}).get("close_dollars")
        if v is None:
            continue
        try:
            cents = round(float(v) * 100)
        except (ValueError, TypeError):
            continue
        ts = datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc)
        result[ts.astimezone(city_tz).hour] = cents
    return result


def get_price_at_hour(price_map: dict[int, float], hour: int) -> float | None:
    """Look up price at target hour, falling back to adjacent hours (±2)."""
    val = price_map.get(hour)
    if val is not None:
        return val
    for delta in [1, -1, 2, -2]:
        val = price_map.get(hour + delta)
        if val is not None:
            return val
    return None


# ── Signal classification ─────────────────────────────────────────────────────

def classify_signal(
    forecast_high: float,
    strike_lo: float,
    strike_hi: float,
    edge_min: float,
) -> str | None:
    """Return 'YES', 'NO_HIGH', 'NO_LOW', or None.

    YES:     forecast inside band.
    NO_HIGH: forecast above band by ≥ edge_min (too hot → band won't resolve YES).
    NO_LOW:  forecast below band by ≥ edge_min (too cold).
    None:    outside but within edge_min — too close to call.

    Tracking direction separately matters for consensus: a model saying NO_HIGH
    and another saying NO_LOW are predicting opposite temperatures and must NOT
    be counted as agreement.
    """
    if strike_lo <= forecast_high <= strike_hi:
        return "YES"
    if forecast_high > strike_hi + edge_min:
        return "NO_HIGH"
    if forecast_high < strike_lo - edge_min:
        return "NO_LOW"
    return None


# ── Trade simulation ──────────────────────────────────────────────────────────

def simulate_trade(
    band: dict,
    signal: str,
    entry_hour: int,
    hourly_ask: dict[int, float],
    hourly_bid: dict[int, float],
    pt_threshold: float | None,
    sl_threshold: float | None,
    sl_cents: float | None = None,
) -> dict | None:
    """Simulate one forecast trade.

    YES position:  buy YES at yes_ask, track yes_ask for exits.
    NO position:   buy NO at (100 - yes_bid), track (100 - yes_bid) for exits.

    Returns None if entry price is unavailable.
    """
    win_result = "yes" if signal == "YES" else "no"

    if signal == "YES":
        entry_price = get_price_at_hour(hourly_ask, entry_hour)
        if entry_price is None or entry_price <= 0:
            return None
        # Hourly price series for exit scanning (yes_ask)
        price_series = hourly_ask
    else:
        # NO ask = 100 - YES bid
        entry_bid = get_price_at_hour(hourly_bid, entry_hour)
        if entry_bid is None or entry_bid <= 0:
            return None
        entry_price = 100 - entry_bid
        # Build NO ask series from yes_bid (invert)
        price_series = {h: 100 - b for h, b in hourly_bid.items() if b > 0}

    if entry_price <= 0 or entry_price >= NO_EDGE_THRESHOLD:
        return {
            "ticker":      band["ticker"],
            "metric":      band["metric"],
            "date":        band["date"],
            "result":      band["result"],
            "strike_lo":   band["strike_lo"],
            "strike_hi":   band["strike_hi"],
            "signal":      signal,
            "entry_hour":  entry_hour,
            "entry_price": entry_price,
            "exit_hour":   None,
            "exit_price":  None,
            "exit_reason": "no_edge",
            "pnl_cents":   None,
            "win":         None,
        }

    pt_price = entry_price * (1 + pt_threshold) if pt_threshold is not None else None
    if sl_cents is not None:
        sl_price = entry_price - sl_cents
        if sl_price <= 0:
            sl_price = None  # SL below 0 is unreachable
    elif sl_threshold is not None:
        sl_price = entry_price * (1 - sl_threshold)
    else:
        sl_price = None

    exit_hour   = None
    exit_price  = None
    exit_reason = None

    for hour in sorted(h for h in price_series if h > entry_hour):
        price = price_series[hour]
        if pt_price is not None and price >= pt_price:
            exit_hour   = hour
            exit_price  = price
            exit_reason = "profit_take"
            break
        if sl_price is not None and price <= sl_price:
            exit_hour   = hour
            exit_price  = price
            exit_reason = "stop_loss"
            break

    if exit_reason is None:
        exit_reason = "settlement"
        exit_price  = 100 if band["result"] == win_result else 0
        exit_hour   = 99

    pnl = exit_price - entry_price

    return {
        "ticker":      band["ticker"],
        "metric":      band["metric"],
        "date":        band["date"],
        "result":      band["result"],
        "strike_lo":   band["strike_lo"],
        "strike_hi":   band["strike_hi"],
        "signal":      signal,
        "entry_hour":  entry_hour,
        "entry_price": entry_price,
        "exit_hour":   exit_hour,
        "exit_price":  exit_price,
        "exit_reason": exit_reason,
        "pnl_cents":   pnl,
        "win":         band["result"] == win_result,
    }


# ── Consensus helper ─────────────────────────────────────────────────────────

def _consensus_signal(signals: dict[str, str | None], mode: str) -> str | None:
    """Derive a consensus signal requiring direction agreement.

    For NO signals, models must agree on the SAME direction (NO_HIGH or NO_LOW).
    A mix of NO_HIGH and NO_LOW is a conflict — the models disagree on what
    temperature will actually occur and must not be combined.

    any_model:     any of 5 individual models, no directional conflict
    majority_vote: ≥2 of 3 core models agree on same direction
    all_agree:     all 3 core models agree on same direction
    """
    if mode == "any_model":
        pool = [s for s in signals.values() if s is not None]
    else:
        pool = [signals.get(m) for m in _CONSENSUS_MODELS]
        pool = [s for s in pool if s is not None]

    if not pool:
        return None

    yes_n  = pool.count("YES")
    nh_n   = pool.count("NO_HIGH")
    nl_n   = pool.count("NO_LOW")
    any_no = nh_n + nl_n

    threshold = 2 if mode in ("any_model", "majority_vote") else len(_CONSENSUS_MODELS)

    # YES consensus: enough YES votes, no NO votes of any kind
    if yes_n >= threshold and any_no == 0:
        return "YES"
    # NO_HIGH consensus: enough same-direction NO votes, no YES or opposite-NO
    if nh_n >= threshold and yes_n == 0 and nl_n == 0:
        return "NO_HIGH"
    # NO_LOW consensus
    if nl_n >= threshold and yes_n == 0 and nh_n == 0:
        return "NO_LOW"
    return None


# ── Main simulation ───────────────────────────────────────────────────────────

async def run_simulation(
    bands: list[dict],
    forecasts: dict[tuple[str, str, str], float],
    entry_hour: int,
    edge_min: float,
    city_filter: set[str] | None,
    month_filter: set[int] | None,
    use_cache: bool,
    pt: float | None,
    sl: float | None,
) -> dict[str, list[dict]]:
    """Return trades_by_model dict for all individual models + consensus modes."""

    # Filter bands
    active_bands = []
    for b in bands:
        if city_filter:
            suffix = b["metric"].replace("temp_high_", "")
            if suffix not in city_filter:
                continue
        if month_filter:
            month = int(b["date"].split("-")[1])
            if month not in month_filter:
                continue
        active_bands.append(b)

    log.info("Active bands: %d", len(active_bands))

    # Fetch missing candlesticks
    cache = load_cache() if use_cache else {}
    to_fetch = list(dict.fromkeys(
        b["ticker"] for b in active_bands if b["ticker"] not in cache
    ))
    if to_fetch:
        ticker_to_date = {b["ticker"]: b["date"] for b in active_bands}
        log.info("Fetching candlesticks: %d new, %d cached",
                 len(to_fetch), len(active_bands) - len(to_fetch))
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_candles(session, t, ticker_to_date[t]) for t in to_fetch]
            results = await asyncio.gather(*tasks)
        for ticker, candles in zip(to_fetch, results):
            cache[ticker] = candles
        save_cache(cache)

    # Build candidate list: (band, signal_by_model, hourly_ask, hourly_bid)
    # For each band, pre-compute signal for each individual model.
    candidates: list[dict] = []
    missing_forecast = 0

    for b in active_bands:
        metric    = b["metric"]
        band_date = date.fromisoformat(b["date"])
        meas_date = (band_date - timedelta(days=1)).isoformat()

        city_entry = CITIES.get(metric)
        if city_entry is None:
            continue
        city_tz = city_entry[3]

        candles    = cache.get(b["ticker"], [])
        hourly_ask = candles_to_hourly_ask(candles, city_tz)
        hourly_bid = candles_to_hourly_bid(candles, city_tz)

        # Signals per model
        signals: dict[str, str | None] = {}
        for model in _INDIVIDUAL_MODELS:
            fc = forecasts.get((metric, meas_date, model))
            if fc is None:
                signals[model] = None
            else:
                signals[model] = classify_signal(fc, b["strike_lo"], b["strike_hi"], edge_min)

        any_known = any(s is not None for s in signals.values())
        if not any_known:
            missing_forecast += 1

        candidates.append({
            "band":        b,
            "signals":     signals,
            "hourly_ask":  hourly_ask,
            "hourly_bid":  hourly_bid,
        })

    log.info("Candidates: %d  (missing all forecasts: %d)", len(candidates), missing_forecast)

    # Build trades for each individual model
    trades_by_model: dict[str, list[dict]] = {}

    for model in _INDIVIDUAL_MODELS:
        trades: list[dict] = []
        for cand in candidates:
            signal = cand["signals"].get(model)
            if signal is None:
                continue
            trade = simulate_trade(
                cand["band"], signal, entry_hour,
                cand["hourly_ask"], cand["hourly_bid"],
                pt, sl,
            )
            if trade is not None:
                trades.append(trade)
        trades_by_model[model] = trades

    # Build consensus trades
    for mode in _CONSENSUS_MODES:
        trades: list[dict] = []
        for cand in candidates:
            signal = _consensus_signal(cand["signals"], mode)
            if signal is None:
                continue

            trade = simulate_trade(
                cand["band"], signal, entry_hour,
                cand["hourly_ask"], cand["hourly_bid"],
                pt, sl,
            )
            if trade is not None:
                trades.append(trade)
        trades_by_model[mode] = trades

    return trades_by_model


# ── PT/SL sweep ───────────────────────────────────────────────────────────────

def sweep_pt_sl(candidates: list[dict], entry_hour: int) -> dict[tuple, dict[str, float]]:
    """For each individual model, sweep all (pt, sl) combos.

    Returns {(model, pt, sl): {total_pnl, n_trades, win_rate}} dict.
    """
    results: dict[tuple, dict[str, float]] = {}

    all_keys = list(_INDIVIDUAL_MODELS) + list(_CONSENSUS_MODES)

    for key in all_keys:
        is_consensus = key in _CONSENSUS_MODES
        for pt in _PT_VALUES:
            for sl in _SL_VALUES:
                pnl_list: list[float] = []
                wins = 0
                for cand in candidates:
                    if is_consensus:
                        signal = _consensus_signal(cand["signals"], key)
                    else:
                        signal = cand["signals"].get(key)
                    if signal is None:
                        continue
                    trade = simulate_trade(
                        cand["band"], signal, entry_hour,
                        cand["hourly_ask"], cand["hourly_bid"],
                        pt, sl,
                    )
                    if trade is None or trade["exit_reason"] == "no_edge" or trade["pnl_cents"] is None:
                        continue
                    pnl_list.append(trade["pnl_cents"])
                    if trade["win"]:
                        wins += 1

                n = len(pnl_list)
                results[(key, pt, sl)] = {
                    "n":          n,
                    "total_pnl":  sum(pnl_list),
                    "win_rate":   wins / n * 100 if n else float("nan"),
                }

    return results


def sweep_pt_sl_cents(candidates: list[dict], entry_hour: int) -> dict[tuple, dict[str, float]]:
    """Sweep PT% × fixed-cent SL combinations for all models and consensus modes.

    Returns {(model, pt, sl_cents): {total_pnl, n_trades, win_rate}}.
    Fixed-cent SL avoids the problem where percentage SL thresholds are
    unreachable for high-entry NO positions (e.g. 25% SL on 82¢ = exit at
    61.5¢, a 20¢ loss, which may never trigger before settlement).
    """
    results: dict[tuple, dict[str, float]] = {}

    all_keys = list(_INDIVIDUAL_MODELS) + list(_CONSENSUS_MODES)

    for key in all_keys:
        is_consensus = key in _CONSENSUS_MODES
        for pt in _PT_VALUES:
            for sl_c in _SL_CENTS_VALUES:
                pnl_list: list[float] = []
                wins = 0
                for cand in candidates:
                    if is_consensus:
                        signal = _consensus_signal(cand["signals"], key)
                    else:
                        signal = cand["signals"].get(key)
                    if signal is None:
                        continue
                    trade = simulate_trade(
                        cand["band"], signal, entry_hour,
                        cand["hourly_ask"], cand["hourly_bid"],
                        pt, sl_threshold=None, sl_cents=sl_c,
                    )
                    if trade is None or trade["exit_reason"] == "no_edge" or trade["pnl_cents"] is None:
                        continue
                    pnl_list.append(trade["pnl_cents"])
                    if trade["win"]:
                        wins += 1

                n = len(pnl_list)
                results[(key, pt, sl_c)] = {
                    "n":         n,
                    "total_pnl": sum(pnl_list),
                    "win_rate":  wins / n * 100 if n else float("nan"),
                }

    return results


def sweep_edge(candidates: list[dict], entry_hour: int) -> dict[tuple, dict]:
    """Sweep edge_min applied symmetrically to YES and NO signals.

    Uses hold-to-settlement (no PT/SL) to isolate pure signal quality.
    YES signals require the forecast to sit at least edge_min inside each
    band boundary — tighter than the default (which requires no clearance
    for YES). This shows the tradeoff between conviction and trade count.

    Returns {(model_or_mode, edge): {yes_n, yes_wins, noh_n, noh_wins,
                                      nol_n, nol_wins, n_total, total_pnl}}.
    """
    results: dict[tuple, dict] = {}
    all_keys = list(_INDIVIDUAL_MODELS) + list(_CONSENSUS_MODES)

    for edge in _EDGE_VALUES:
        # Reclassify signals with symmetric edge applied to both YES and NO
        reclassified: list[dict] = []
        for cand in candidates:
            b = cand["band"]
            lo, hi = b["strike_lo"], b["strike_hi"]
            new_signals: dict[str, str | None] = {}
            for model, fc in cand.get("raw_forecasts", {}).items():
                if fc is None:
                    new_signals[model] = None
                elif lo + edge <= fc <= hi - edge:
                    new_signals[model] = "YES"
                elif fc > hi + edge:
                    new_signals[model] = "NO_HIGH"
                elif fc < lo - edge:
                    new_signals[model] = "NO_LOW"
                else:
                    new_signals[model] = None
            reclassified.append({**cand, "signals": new_signals})

        for key in all_keys:
            is_consensus = key in _CONSENSUS_MODES
            yes_n = yes_wins = 0
            noh_n = noh_wins = 0
            nol_n = nol_wins = 0
            total_pnl = 0.0

            for cand in reclassified:
                signal = (
                    _consensus_signal(cand["signals"], key)
                    if is_consensus
                    else cand["signals"].get(key)
                )
                if signal is None:
                    continue
                trade = simulate_trade(
                    cand["band"], signal, entry_hour,
                    cand["hourly_ask"], cand["hourly_bid"],
                    pt_threshold=None, sl_threshold=None,
                )
                if trade is None or trade["exit_reason"] == "no_edge" or trade["pnl_cents"] is None:
                    continue
                total_pnl += trade["pnl_cents"]
                won = trade["win"]
                if signal == "YES":
                    yes_n += 1
                    if won: yes_wins += 1
                elif signal == "NO_HIGH":
                    noh_n += 1
                    if won: noh_wins += 1
                else:
                    nol_n += 1
                    if won: nol_wins += 1

            results[(key, edge)] = {
                "yes_n": yes_n, "yes_wins": yes_wins,
                "noh_n": noh_n, "noh_wins": noh_wins,
                "nol_n": nol_n, "nol_wins": nol_wins,
                "n_total": yes_n + noh_n + nol_n,
                "total_pnl": total_pnl,
            }

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _trade_stats(trades: list[dict]) -> dict:
    real_trades = [t for t in trades if t["exit_reason"] != "no_edge" and t["pnl_cents"] is not None]
    no_edge     = [t for t in trades if t["exit_reason"] == "no_edge"]
    unavailable = [t for t in trades if t["pnl_cents"] is None and t["exit_reason"] != "no_edge"]
    yes_trades     = [t for t in real_trades if t["signal"] == "YES"]
    no_high_trades = [t for t in real_trades if t["signal"] == "NO_HIGH"]
    no_low_trades  = [t for t in real_trades if t["signal"] == "NO_LOW"]
    no_trades      = no_high_trades + no_low_trades

    n = len(real_trades)
    if n == 0:
        return {
            "total_sigs": len(trades), "n_trades": 0, "no_edge": len(no_edge),
            "unavail": len(unavailable), "yes_trades": 0, "no_trades": 0,
            "no_high_trades": 0, "no_low_trades": 0,
            "wins": 0, "win_rate": float("nan"), "avg_entry": float("nan"),
            "avg_pnl": float("nan"), "total_pnl_dollars": 0.0,
        }

    wins = sum(1 for t in real_trades if t["win"])
    by_reason: dict[str, int] = {}
    for t in real_trades:
        by_reason[t["exit_reason"]] = by_reason.get(t["exit_reason"], 0) + 1

    return {
        "total_sigs":      len(trades),
        "n_trades":        n,
        "no_edge":         len(no_edge),
        "unavail":         len(unavailable),
        "yes_trades":      len(yes_trades),
        "no_trades":       len(no_trades),
        "no_high_trades":  len(no_high_trades),
        "no_low_trades":   len(no_low_trades),
        "wins":            wins,
        "win_rate":        wins / n * 100,
        "avg_entry":       sum(t["entry_price"] for t in real_trades) / n,
        "avg_pnl":         sum(t["pnl_cents"] for t in real_trades) / n,
        "total_pnl_dollars": sum(t["pnl_cents"] for t in real_trades) / 100,
        "by_reason":       by_reason,
    }


def build_report(
    trades_by_model: dict[str, list[dict]],
    sweep_results: dict[tuple, dict],
    sweep_cents_results: dict[tuple, dict],
    sweep_edge_results: dict[tuple, dict],
    entry_hour: int,
    edge_min: float,
    pt_fixed: float | None,
    sl_fixed: float | None,
    do_sweep: bool,
) -> str:
    buf = io.StringIO()

    def w(s: str = "") -> None:
        buf.write(s + "\n")

    pt_label = f"{pt_fixed*100:.0f}%" if pt_fixed is not None else "hold"
    sl_label = f"{sl_fixed*100:.0f}%" if sl_fixed is not None else "hold"

    w()
    w("=" * 68)
    w("  FORECAST MODEL ACCURACY SUMMARY")
    w(f"  entry_hour={entry_hour}  edge_min={edge_min}°F  PT={pt_label}  SL={sl_label}")
    w("=" * 68)
    w(f"  {'Model':<18} {'YES':>5} {'NO_H':>6} {'NO_L':>6} {'NoEdge':>7} {'Trades':>7} {'Win%':>7} {'Avg P&L':>9} {'Total P&L':>10}")
    w("  " + "-" * 79)

    # Blank separator before consensus block
    all_model_keys = list(_INDIVIDUAL_MODELS) + list(_CONSENSUS_MODES)
    for i, model in enumerate(all_model_keys):
        if i == len(_INDIVIDUAL_MODELS):
            w()  # visual break between individual and consensus
        trades = trades_by_model.get(model, [])
        st = _trade_stats(trades)
        if st["n_trades"] == 0:
            w(f"  {model:<18} {'—':>5} {'—':>6} {'—':>6} {'—':>7} {'0':>7} {'—':>7} {'—':>9} {'$0.00':>10}")
            continue
        win_pct   = f"{st['win_rate']:.1f}%" if not _isnan(st["win_rate"]) else "—"
        avg_pnl   = f"{st['avg_pnl']:+.1f}¢"  if not _isnan(st["avg_pnl"])  else "—"
        total_pnl = f"${st['total_pnl_dollars']:+.2f}"
        no_edge_n = st["no_edge"]
        w(f"  {model:<18} {st['yes_trades']:>5} {st['no_high_trades']:>6} {st['no_low_trades']:>6} {no_edge_n:>7} {st['n_trades']:>7} "
          f"{win_pct:>7} {avg_pnl:>9} {total_pnl:>10}")

    # Exit reason breakdown for each model
    w()
    w("=" * 68)
    w("  EXIT REASON BREAKDOWN")
    w("=" * 68)
    w(f"  {'Model':<18} {'PT':>7} {'SL':>7} {'Settle_W':>10} {'Settle_L':>10}")
    w("  " + "-" * 56)
    for model in all_model_keys:
        trades = trades_by_model.get(model, [])
        real_trades = [t for t in trades if t["exit_reason"] != "no_edge" and t["pnl_cents"] is not None]
        if not real_trades:
            continue
        n = len(real_trades)
        by_r = {}
        for t in real_trades:
            by_r[t["exit_reason"]] = by_r.get(t["exit_reason"], 0) + 1
        pt_n  = by_r.get("profit_take", 0)
        sl_n  = by_r.get("stop_loss", 0)
        sw_n  = sum(1 for t in real_trades if t["exit_reason"] == "settlement" and t["win"])
        sl_n2 = sum(1 for t in real_trades if t["exit_reason"] == "settlement" and not t["win"])
        w(f"  {model:<18} {pt_n:>7} {sl_n:>7} {sw_n:>10} {sl_n2:>10}")

    # PT/SL sweep grid (individual models + consensus modes)
    if do_sweep and sweep_results:
        for model in list(_INDIVIDUAL_MODELS) + list(_CONSENSUS_MODES):
            w()
            w("=" * 68)
            w(f"  PT/SL GRID — {model}  (total P&L in $, $1/trade, YES+NO combined)")
            w("=" * 68)
            pt_headers = [f"{p*100:.0f}%" if p is not None else "hold" for p in _PT_VALUES]
            header = f"  {'SL↓/PT→':<9}" + "".join(f"{h:>8}" for h in pt_headers)
            w(header)
            w("  " + "-" * (9 + 8 * len(_PT_VALUES)))

            best_pnl  = float("-inf")
            best_combo: tuple | None = None

            for sl in _SL_VALUES:
                sl_label_g = f"{sl*100:.0f}%" if sl is not None else "hold"
                row = f"  {sl_label_g:<9}"
                for pt in _PT_VALUES:
                    key = (model, pt, sl)
                    r   = sweep_results.get(key, {})
                    n   = r.get("n", 0)
                    pnl = r.get("total_pnl", 0.0)
                    pnl_d = pnl / 100
                    if n > 0 and pnl_d > best_pnl:
                        best_pnl   = pnl_d
                        best_combo = (pt, sl)
                    row += f" {pnl_d:>+7.2f}" if n > 0 else "        —"
                w(row)

            if best_combo:
                pt_b, sl_b = best_combo
                pt_s = f"{pt_b*100:.0f}%" if pt_b is not None else "hold"
                sl_s = f"{sl_b*100:.0f}%" if sl_b is not None else "hold"
                w(f"\n  Best combo: PT={pt_s}  SL={sl_s}  →  ${best_pnl:+.2f}")

    # Fixed-cent SL grids
    if do_sweep and sweep_cents_results:
        for model in list(_INDIVIDUAL_MODELS) + list(_CONSENSUS_MODES):
            w()
            w("=" * 68)
            w(f"  PT% / FIXED-CENT SL GRID — {model}  (total P&L in $, $1/trade)")
            w("=" * 68)
            pt_headers = [f"{p*100:.0f}%" if p is not None else "hold" for p in _PT_VALUES]
            header = f"  {'SL↓/PT→':<10}" + "".join(f"{h:>8}" for h in pt_headers)
            w(header)
            w("  " + "-" * (10 + 8 * len(_PT_VALUES)))

            best_pnl   = float("-inf")
            best_combo = None

            for sl_c in _SL_CENTS_VALUES:
                row = f"  {sl_c}¢{'':<7}"
                for pt in _PT_VALUES:
                    key = (model, pt, sl_c)
                    r   = sweep_cents_results.get(key, {})
                    n   = r.get("n", 0)
                    pnl = r.get("total_pnl", 0.0)
                    pnl_d = pnl / 100
                    if n > 0 and pnl_d > best_pnl:
                        best_pnl   = pnl_d
                        best_combo = (pt, sl_c)
                    row += f" {pnl_d:>+7.2f}" if n > 0 else "        —"
                w(row)

            if best_combo:
                pt_b, sl_b = best_combo
                pt_s = f"{pt_b*100:.0f}%" if pt_b is not None else "hold"
                w(f"\n  Best combo: PT={pt_s}  SL={sl_b}¢  →  ${best_pnl:+.2f}")

    # Edge sweep
    if do_sweep and sweep_edge_results:
        w()
        w("=" * 68)
        w("  EDGE SWEEP  (hold to settlement, $1/trade)")
        w("  Edge applied symmetrically: YES requires forecast ≥ edge inside band,")
        w("  NO requires forecast ≥ edge outside band.")
        w("=" * 68)
        for i, model in enumerate(all_model_keys):
            if i == len(_INDIVIDUAL_MODELS):
                w()
            w(f"  {model}")
            w(f"  {'Edge':>6}  {'YES N':>6}  {'YES win%':>9}  {'NO_H N':>7}  {'NO_H win%':>10}  {'NO_L N':>7}  {'NO_L win%':>10}  {'Total P&L':>10}")
            w("  " + "-" * 77)
            for edge in _EDGE_VALUES:
                r = sweep_edge_results.get((model, edge), {})
                n_total = r.get("n_total", 0)
                if n_total == 0:
                    continue
                yn  = r["yes_n"];  yw = r["yes_wins"]
                hn  = r["noh_n"];  hw = r["noh_wins"]
                ln  = r["nol_n"];  lw = r["nol_wins"]
                y_pct = f"{yw/yn*100:.1f}%" if yn else "—"
                h_pct = f"{hw/hn*100:.1f}%" if hn else "—"
                l_pct = f"{lw/ln*100:.1f}%" if ln else "—"
                pnl_d = r["total_pnl"] / 100
                w(f"  {edge:.1f}°F  {yn:>6}  {y_pct:>9}  {hn:>7}  {h_pct:>10}  {ln:>7}  {l_pct:>10}  ${pnl_d:>+9.2f}")

    # By signal direction (YES vs NO_HIGH vs NO_LOW)
    w()
    w("=" * 68)
    w("  BY SIGNAL DIRECTION")
    w("=" * 68)
    w(f"  {'Model':<18} {'Direction':<10} {'N':>6} {'Win%':>7} {'AvgEntry':>9} {'Avg P&L':>9} {'Total P&L':>10}")
    w("  " + "-" * 73)
    for i, model in enumerate(all_model_keys):
        if i == len(_INDIVIDUAL_MODELS):
            w()
        trades = trades_by_model.get(model, [])
        real_trades = [t for t in trades if t["exit_reason"] != "no_edge" and t["pnl_cents"] is not None]
        first_row = True
        for direction in ["YES", "NO_HIGH", "NO_LOW"]:
            dt = [t for t in real_trades if t["signal"] == direction]
            n = len(dt)
            if n == 0:
                continue
            wins = sum(1 for t in dt if t["win"])
            win_pct_d = f"{wins/n*100:.1f}%"
            avg_entry_d = sum(t["entry_price"] for t in dt) / n
            avg_pnl_d = sum(t["pnl_cents"] for t in dt) / n
            total_pnl_d = sum(t["pnl_cents"] for t in dt) / 100
            model_col = model if first_row else ""
            dir_label = direction.replace("_", " ")
            w(f"  {model_col:<18} {dir_label:<10} {n:>6} {win_pct_d:>7} {avg_entry_d:>8.1f}¢ {avg_pnl_d:>+8.1f}¢ ${total_pnl_d:>+9.2f}")
            first_row = False

    # By city
    w()
    w("=" * 68)
    w("  BY CITY (individual models, fixed PT/SL)")
    w("=" * 68)
    city_header = f"  {'City':<22}" + "".join(
        f" {m.split('_')[0][:7]:>8}" for m in _INDIVIDUAL_MODELS
    )
    w(city_header + "  (total P&L $)")
    w("  " + "-" * 68)

    all_cities = sorted({t["metric"].replace("temp_high_", "")
                         for tlist in trades_by_model.values() for t in tlist})
    for city in all_cities:
        city_name = next(
            (v[0] for k, v in CITIES.items() if k.endswith(city)),
            city.upper(),
        )
        row = f"  {city_name:<22}"
        for model in _INDIVIDUAL_MODELS:
            trades = [t for t in trades_by_model.get(model, [])
                      if t["metric"].replace("temp_high_", "") == city
                      and t["exit_reason"] != "no_edge" and t["pnl_cents"] is not None]
            pnl = sum(t["pnl_cents"] for t in trades) / 100 if trades else 0.0
            row += f" {pnl:>+8.2f}"
        w(row)

    # By month
    w()
    w("=" * 68)
    w("  BY MONTH")
    w("=" * 68)
    month_header = f"  {'Month':<10}" + "".join(
        f" {m.split('_')[0][:7]:>9}" for m in _INDIVIDUAL_MODELS
    )
    w(month_header + "  (N trades / win%)")
    w("  " + "-" * 68)

    all_months = sorted({int(t["date"].split("-")[1])
                         for tlist in trades_by_model.values() for t in tlist})
    month_names = {2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",
                   8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec",1:"Jan"}
    for month in all_months:
        row = f"  {month_names.get(month, str(month)):<10}"
        for model in _INDIVIDUAL_MODELS:
            trades = [t for t in trades_by_model.get(model, [])
                      if int(t["date"].split("-")[1]) == month
                      and t["exit_reason"] != "no_edge" and t["pnl_cents"] is not None]
            n = len(trades)
            if n == 0:
                row += f"       {'—':>9}"
                continue
            wins = sum(1 for t in trades if t["win"])
            row += f" {n:>4}/{wins/n*100:.0f}%"
        w(row)

    # By entry price bucket (YES signals only)
    w()
    w("=" * 68)
    w("  BY ENTRY PRICE BUCKET — YES signals")
    w("=" * 68)
    buckets = [(0, 20), (20, 35), (35, 50), (50, 70), (70, 95)]
    bk_header = f"  {'Bucket':<12}" + "".join(
        f" {m.split('_')[0][:7]:>9}" for m in _INDIVIDUAL_MODELS
    )
    w(bk_header + "  (N / win%)")
    w("  " + "-" * 68)
    for lo, hi in buckets:
        row = f"  {lo}–{hi}¢{'':<7}"
        for model in _INDIVIDUAL_MODELS:
            trades = [t for t in trades_by_model.get(model, [])
                      if t["signal"] == "YES"
                      and t["exit_reason"] != "no_edge"
                      and t["pnl_cents"] is not None
                      and lo <= t["entry_price"] < hi]
            n = len(trades)
            if n == 0:
                row += f"       {'—':>9}"
                continue
            wins = sum(1 for t in trades if t["win"])
            row += f" {n:>4}/{wins/n*100:.0f}%"
        w(row)

    w()
    return buf.getvalue()


def _isnan(x) -> bool:
    try:
        return x != x
    except Exception:
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    forecasts = load_forecasts()
    bands     = load_bands()

    city_filter  = set(args.cities)  if args.cities  else None
    month_filter = set(args.months)  if args.months  else None

    # For PT/SL sweep, run with no thresholds first to collect candidates,
    # then sweep. For fixed PT/SL, run once directly.
    do_sweep = (args.pt is None and args.sl is None)

    if do_sweep:
        # First pass: fetch and cache all candlesticks (no PT/SL needed yet)
        log.info("Running simulation (no PT/SL, for sweep)…")
        await run_simulation(
            bands, forecasts, args.entry_hour, args.edge_min,
            city_filter, month_filter, not args.no_cache,
            None, None,
        )
        # Rebuild candidates from cache for PT/SL sweep
        active_bands = []
        for b in bands:
            if city_filter:
                suffix = b["metric"].replace("temp_high_", "")
                if suffix not in city_filter:
                    continue
            if month_filter:
                if int(b["date"].split("-")[1]) not in month_filter:
                    continue
            active_bands.append(b)

        cache = load_cache()
        candidates_for_sweep: list[dict] = []
        for b in active_bands:
            metric    = b["metric"]
            band_date = date.fromisoformat(b["date"])
            meas_date = (band_date - timedelta(days=1)).isoformat()
            city_entry = CITIES.get(metric)
            if city_entry is None:
                continue
            city_tz = city_entry[3]
            candles    = cache.get(b["ticker"], [])
            hourly_ask = candles_to_hourly_ask(candles, city_tz)
            hourly_bid = candles_to_hourly_bid(candles, city_tz)
            raw_fcs: dict[str, float | None] = {}
            signals: dict[str, str | None] = {}
            for model in _INDIVIDUAL_MODELS:
                fc = forecasts.get((metric, meas_date, model))
                raw_fcs[model] = fc
                signals[model] = (
                    classify_signal(fc, b["strike_lo"], b["strike_hi"], args.edge_min)
                    if fc is not None else None
                )
            candidates_for_sweep.append({
                "band": b, "signals": signals, "raw_forecasts": raw_fcs,
                "hourly_ask": hourly_ask, "hourly_bid": hourly_bid,
            })

        log.info("Running PT% × SL% sweep…")
        sweep_results = sweep_pt_sl(candidates_for_sweep, args.entry_hour)

        log.info("Running PT% × fixed-cent SL sweep…")
        sweep_cents_results = sweep_pt_sl_cents(candidates_for_sweep, args.entry_hour)

        log.info("Running edge sweep…")
        sweep_edge_results = sweep_edge(candidates_for_sweep, args.entry_hour)

        # Re-run with best (pt, sl) per model for the main summary
        # Use ecmwf_ifs best combo for the main display
        best_pt, best_sl = None, None
        best_total = float("-inf")
        for (model, pt, sl), r in sweep_results.items():
            if model == "ecmwf_ifs" and r["n"] > 0:
                if r["total_pnl"] / 100 > best_total:
                    best_total = r["total_pnl"] / 100
                    best_pt, best_sl = pt, sl

        log.info("Best ECMWF combo: PT=%s SL=%s  →  $%+.2f",
                 f"{best_pt*100:.0f}%" if best_pt else "hold",
                 f"{best_sl*100:.0f}%" if best_sl else "hold",
                 best_total)

        trades_best = await run_simulation(
            bands, forecasts, args.entry_hour, args.edge_min,
            city_filter, month_filter, True,  # use cache
            best_pt, best_sl,
        )
        trades_by_model = trades_best

    else:
        log.info("Running simulation with PT=%s SL=%s…",
                 f"{args.pt*100:.0f}%" if args.pt else "hold",
                 f"{args.sl*100:.0f}%" if args.sl else "hold")
        trades_by_model = await run_simulation(
            bands, forecasts, args.entry_hour, args.edge_min,
            city_filter, month_filter, not args.no_cache,
            args.pt, args.sl,
        )
        sweep_results = {}
        sweep_cents_results = {}
        sweep_edge_results = {}
        best_pt, best_sl = args.pt, args.sl

    report = build_report(
        trades_by_model, sweep_results, sweep_cents_results, sweep_edge_results,
        args.entry_hour, args.edge_min,
        best_pt if do_sweep else args.pt,
        best_sl if do_sweep else args.sl,
        do_sweep,
    )

    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        log.info("Report written to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest forecast-based YES/NO band-arb trading."
    )
    parser.add_argument("--model", choices=_INDIVIDUAL_MODELS, default=None,
                        help="Single model to run (default: all)")
    parser.add_argument("--entry-hour", type=int, default=9,
                        help="Local entry hour (default: 9)")
    parser.add_argument("--edge-min", type=float, default=1.0,
                        help="Minimum °F clearance from band edge (default: 1.0)")
    parser.add_argument("--pt", type=float, default=None,
                        help="Profit-take threshold, e.g. 0.40 (default: sweep)")
    parser.add_argument("--sl", type=float, default=None,
                        help="Stop-loss threshold, e.g. 0.40 (default: sweep)")
    parser.add_argument("--months", nargs="+", type=int, default=None,
                        help="Month numbers to include, e.g. --months 3 4 5")
    parser.add_argument("--cities", nargs="+", default=None,
                        help="City suffixes, e.g. --cities dca chi bos")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-fetch all candlesticks (ignore cache)")
    parser.add_argument("--out", default=None,
                        help="Save report to file (default: stdout only)")

    args = parser.parse_args()
    asyncio.run(main(args))
