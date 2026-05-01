"""Backtest: forecast-driven momentum strategy.

Hypothesis
----------
When a public weather forecast (Open-Meteo, NWS) says the daily high will
be well above a YES strike, casual Kalshi users will discover the same
forecast and bid YES prices up over the next few hours.  We enter YES early
(before casual buyers react) and profit-take before the next major forecast
update pulls the advantage away.

Two operating modes
-------------------
1. live  (default) — uses raw_forecasts rows with yes_bid != NULL, collected
   by the live bot.  Needs ≥1 day of data after the yes_bid column was added.
   For each ticker, simulates buying at the first bullish reading of the day
   and selling at subsequent readings (1 h, 2 h, 4 h later) or at settlement.

2. synthetic — fetches Open-Meteo historical archive (last N days) and
   backtests signal accuracy WITHOUT needing Kalshi price history.
   For each city/day/hour, estimates P(high > strike) from a normal model
   calibrated to seasonal NWS MAE.  Simulates buying at the implied fair-value
   price and checks whether the actual high confirmed the signal.
   Good for testing directional quality (signal accuracy) before live data
   accumulates.

Usage
-----
  # Live mode (real market price data — needs accumulated raw_forecasts)
  venv/bin/python scripts/backtest_forecast_momentum.py --mode live

  # Synthetic mode (available today, uses Open-Meteo history)
  venv/bin/python scripts/backtest_forecast_momentum.py --mode synthetic --days 14

  # Synthetic mode for specific cities
  venv/bin/python scripts/backtest_forecast_momentum.py --mode synthetic \\
      --cities bos nyc chi --days 30

  # Live mode with custom DB path
  venv/bin/python scripts/backtest_forecast_momentum.py --mode live \\
      --db opportunity_log.db
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.news.noaa import CITIES

# Seasonal NWS MAE (°F) by city — 12 values (Jan-Dec).
# Larger = more forecast uncertainty = wider probability distributions.
CITY_SIGMA: dict[str, list[float]] = {
    "temp_high_bos": [4.5, 4.2, 4.0, 3.5, 3.0, 2.8, 2.5, 2.5, 2.8, 3.2, 3.8, 4.5],
    "temp_high_ny":  [4.0, 3.8, 3.7, 3.3, 2.8, 2.6, 2.4, 2.4, 2.7, 3.0, 3.5, 4.2],
    "temp_high_chi": [5.0, 4.8, 4.5, 4.0, 3.5, 3.2, 3.0, 3.0, 3.3, 3.8, 4.5, 5.2],
    "temp_high_lax": [2.5, 2.5, 2.8, 3.0, 3.0, 3.0, 2.8, 2.8, 2.8, 2.8, 2.5, 2.5],
    "temp_high_sfo": [3.0, 3.0, 3.2, 3.5, 3.8, 4.0, 4.0, 4.0, 3.8, 3.5, 3.0, 3.0],
    "temp_high_sea": [3.5, 3.5, 3.5, 3.2, 3.0, 3.0, 2.8, 2.8, 3.0, 3.5, 3.8, 4.0],
    "temp_high_den": [5.0, 5.0, 5.0, 4.5, 4.0, 3.5, 3.0, 3.0, 3.5, 4.5, 5.0, 5.5],
    "temp_high_phx": [3.5, 3.5, 3.5, 3.5, 4.0, 4.0, 4.0, 4.0, 3.8, 3.5, 3.5, 3.5],
    "temp_high_mia": [3.0, 3.0, 3.0, 2.8, 2.8, 2.5, 2.5, 2.5, 2.8, 3.0, 3.0, 3.0],
    "temp_high_dfw": [4.0, 4.0, 4.5, 4.5, 3.8, 3.5, 3.0, 3.0, 3.5, 4.0, 4.5, 4.5],
    "temp_high_hou": [4.0, 4.0, 4.0, 4.0, 3.5, 3.0, 2.8, 2.8, 3.2, 3.8, 4.2, 4.2],
    "temp_high_atl": [4.0, 4.0, 4.0, 3.5, 3.2, 3.0, 2.8, 2.8, 3.0, 3.5, 4.0, 4.0],
    "temp_high_msp": [5.5, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 3.0, 3.5, 4.5, 5.5, 6.0],
    "temp_high_dca": [4.0, 3.8, 3.8, 3.5, 3.0, 2.8, 2.5, 2.5, 2.8, 3.2, 3.8, 4.2],
    "temp_high_las": [4.0, 4.0, 4.5, 4.5, 4.5, 4.0, 3.5, 3.5, 4.0, 4.5, 4.5, 4.5],
    "temp_high_okc": [4.5, 4.5, 5.0, 5.0, 4.5, 4.0, 3.5, 3.5, 4.0, 4.5, 5.0, 5.0],
    "temp_high_sat": [3.8, 3.8, 4.0, 4.0, 3.8, 3.5, 3.2, 3.2, 3.5, 4.0, 4.0, 4.0],
    "temp_high_phl": [4.2, 4.0, 3.8, 3.5, 3.0, 2.8, 2.5, 2.5, 2.8, 3.2, 3.8, 4.5],
    "temp_high_aus": [3.8, 3.8, 4.0, 4.0, 3.8, 3.5, 3.2, 3.2, 3.5, 4.0, 4.0, 4.0],
    "temp_high_msy": [3.5, 3.5, 3.8, 3.5, 3.2, 3.0, 2.8, 2.8, 3.0, 3.5, 3.8, 3.8],
}
# Fallback sigma if city not in table
_DEFAULT_SIGMA = 4.0


# ── probability helpers ───────────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _p_above(forecast: float, strike: float, sigma: float) -> float:
    """P(actual_high > strike) given current forecast with forecast error sigma."""
    if sigma <= 0:
        return 1.0 if forecast > strike else 0.0
    z = (forecast - strike) / sigma
    return _normal_cdf(z)


def _sigma_at_hour(metric: str, local_hour: int, month: int) -> float:
    """
    Forecast error sigma shrinks as the day progresses (more observations,
    less remaining warming).  We scale the day-ahead MAE by a factor that
    goes from 1.0 at 6am down to 0.5 at the expected peak hour (~3pm).
    """
    base = CITY_SIGMA.get(metric, [_DEFAULT_SIGMA] * 12)[month - 1]
    # Linear taper: full sigma at 6am local, half sigma at 15:00 local
    peak_hour = 15
    dawn_hour = 6
    if local_hour <= dawn_hour:
        scale = 1.0
    elif local_hour >= peak_hour:
        scale = 0.5
    else:
        frac = (local_hour - dawn_hour) / (peak_hour - dawn_hour)
        scale = 1.0 - 0.5 * frac
    return base * scale


# ── Open-Meteo archive fetcher ────────────────────────────────────────────────

def _fetch_openmeteo_hourly(lat: float, lon: float,
                             start: date, end: date) -> list[dict]:
    """
    Fetch hourly temperature (°F) from Open-Meteo historical archive.
    Returns list of {"time": datetime, "temp_f": float}.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        f"&hourly=temperature_2m&temperature_unit=fahrenheit&timezone=UTC"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  [WARN] Open-Meteo fetch failed: {e}", file=sys.stderr)
        return []
    times = data.get("hourly", {}).get("time", [])
    temps = data.get("hourly", {}).get("temperature_2m", [])
    result = []
    for t_str, t_val in zip(times, temps):
        if t_val is None:
            continue
        dt = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)
        result.append({"time": dt, "temp_f": t_val})
    return result


# ── LIVE MODE ─────────────────────────────────────────────────────────────────

# Sources that casual Kalshi buyers are likely to check ("public" sources).
# These are the entry-signal sources for the momentum hypothesis.
PUBLIC_SOURCES = {"open_meteo", "nws_hourly", "noaa", "noaa_day2"}

# Sources we consider "late" — already well-known by settlement time.
# (Internal / high-frequency — advantage gone by the time market reacts.)
# We use ALL sources for entry in live mode since the hypothesis is about timing.
ALL_SOURCES = PUBLIC_SOURCES | {"hrrr", "metar", "noaa_observed", "forecast_no"}


def _live_mode(args) -> None:
    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row

    rows = db.execute(
        """
        SELECT
            DATE(logged_at)     AS day,
            strftime('%H', logged_at) AS utc_hour,
            ticker,
            source,
            metric,
            data_value,
            strike,
            direction,
            edge,
            yes_bid,
            logged_at
        FROM raw_forecasts
        WHERE yes_bid IS NOT NULL
          AND direction IN ('above', 'below', 'between')
          AND metric LIKE 'temp_high_%'
        ORDER BY ticker, logged_at
        """
    ).fetchall()

    if not rows:
        print(
            "\nNo raw_forecasts rows with yes_bid found.\n"
            "The yes_bid column was added recently — the bot needs to run\n"
            "for at least one poll cycle before live mode has data.\n"
            "Try --mode synthetic for a historical signal-quality test."
        )
        return

    # Group by (ticker, day) → list of readings sorted by time
    by_ticker_day: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r["ticker"], r["day"])
        by_ticker_day[key].append(dict(r))

    # Strategy parameters
    min_edge_entry   = args.min_edge     # ¢ below implied prob to consider "underpriced"
    hold_hours       = [1, 2, 4]        # exit windows to test

    results_by_hold: dict[int, list[float]] = {h: [] for h in hold_hours}
    source_pnl: dict[str, dict[int, list[float]]] = defaultdict(lambda: {h: [] for h in hold_hours})
    hour_pnl: dict[int, dict[int, list[float]]] = defaultdict(lambda: {h: [] for h in hold_hours})

    n_signals = 0

    for (ticker, day), readings in sorted(by_ticker_day.items()):
        # Only consider public sources as entry triggers
        bullish = [
            r for r in readings
            if r["source"] in PUBLIC_SOURCES
            and r["direction"] == "above"
            and r["edge"] is not None and r["edge"] >= args.min_edge_f
        ]
        if not bullish:
            continue

        # Entry = first bullish reading of the day
        entry = bullish[0]
        entry_yes_bid = entry["yes_bid"]
        if entry_yes_bid is None or entry_yes_bid <= 0:
            continue

        entry_dt = datetime.fromisoformat(entry["logged_at"])
        entry_hour = int(entry["utc_hour"])
        n_signals += 1

        for hold_h in hold_hours:
            # Find the yes_bid reading closest to entry_dt + hold_h hours
            target_dt = entry_dt + timedelta(hours=hold_h)
            # Any source, not just public — we're measuring market price
            future = [
                r for r in readings
                if datetime.fromisoformat(r["logged_at"]) >= target_dt
                and r["yes_bid"] is not None
            ]
            if not future:
                continue
            exit_bid = future[0]["yes_bid"]
            pnl = exit_bid - entry_yes_bid  # ¢ per contract
            results_by_hold[hold_h].append(pnl)
            source_pnl[entry["source"]][hold_h].append(pnl)
            hour_pnl[entry_hour][hold_h].append(pnl)

    print(f"\n{'═'*70}")
    print("  LIVE MODE — Forecast Momentum Backtest")
    print(f"{'═'*70}")
    print(f"  DB: {args.db}")
    print(f"  Public-source bullish signals found: {n_signals}")
    print(f"  Min forecast edge: {args.min_edge_f}°F above strike")
    print(f"  Entry: first public bullish reading per ticker per day")
    print(f"  Exit: yes_bid at entry_time + N hours\n")

    if n_signals == 0:
        print("  No signals matched. Relax --min-edge-f or wait for more data.")
        return

    print(f"  {'Hold time':>12}  {'N':>5}  {'AvgP&L':>8}  {'Win%':>7}  {'TotalP&L':>10}")
    print(f"  {'-'*50}")
    for h in hold_hours:
        series = results_by_hold[h]
        if not series:
            print(f"  {h}h{'':>9}  {0:>5}  {'—':>8}  {'—':>7}  {'—':>10}")
            continue
        avg = sum(series) / len(series)
        wins = sum(1 for p in series if p > 0)
        tot  = sum(series)
        print(f"  {h}h{'':>9}  {len(series):>5}  {avg:>+7.1f}¢  {wins/len(series):>6.1%}  {tot:>+9.0f}¢")

    print(f"\n  By entry source:")
    print(f"  {'Source':<18}  {'N':>4}  {'AvgP&L@2h':>10}")
    print(f"  {'-'*36}")
    for src, by_hold in sorted(source_pnl.items()):
        s2 = by_hold.get(2, [])
        if not s2:
            continue
        print(f"  {src:<18}  {len(s2):>4}  {sum(s2)/len(s2):>+9.1f}¢")

    print(f"\n  By UTC entry hour:")
    print(f"  {'Hour':>6}  {'N':>4}  {'AvgP&L@2h':>10}")
    print(f"  {'-'*25}")
    for hr in sorted(hour_pnl):
        s2 = hour_pnl[hr].get(2, [])
        if not s2:
            continue
        print(f"  {hr:>5}h  {len(s2):>4}  {sum(s2)/len(s2):>+9.1f}¢")

    print(f"\n{'═'*70}")
    print("  Interpretation:")
    print("  +ve avg P&L at 2h = market price rising after forecast signal.")
    print("  +ve at 1h but not 4h = fast momentum, profit-take early.")
    print("  -ve = no exploitable lag; market prices in forecasts immediately.")
    print(f"{'═'*70}\n")


# ── SYNTHETIC MODE ────────────────────────────────────────────────────────────

def _synth_mode(args) -> None:
    end_date   = date.today() - timedelta(days=1)   # yesterday (fully settled)
    start_date = end_date - timedelta(days=args.days - 1)

    # Filter cities
    if args.cities:
        abbrevs = set(args.cities)
        city_keys = [k for k, v in CITIES.items() if v[4] in abbrevs]
        if not city_keys:
            print(f"ERROR: no city matched {args.cities}. Valid abbrevs: "
                  f"{sorted(v[4] for v in CITIES.values())}")
            sys.exit(1)
    else:
        city_keys = list(CITIES.keys())

    # Strike offsets to test — these are multiples of 0.5°F centred around
    # the forecast.  We simulate a market where strike = floor(forecast) + offset.
    strike_offsets_f = [-4, -2, -1, 0, 1, 2, 4]

    # Hours of day (local) to simulate entry
    entry_hours_local = [7, 8, 9, 10, 11, 12, 13]  # 7am–1pm local

    # Hold windows after entry before exiting (hours)
    hold_hours = [1, 2, 4]

    all_results: dict[tuple[int, int], list[float]] = {}
    for entry_h in entry_hours_local:
        for hold_h in hold_hours:
            all_results[(entry_h, hold_h)] = []

    source_counts: dict[str, int] = defaultdict(int)
    wins_by_edge: dict[str, list[bool]] = defaultdict(list)

    print(f"\n{'═'*70}")
    print("  SYNTHETIC MODE — Forecast Momentum Signal Quality")
    print(f"{'═'*70}")
    print(f"  Period : {start_date} → {end_date}  ({args.days} days)")
    print(f"  Cities : {len(city_keys)}")
    print(f"  Fetching hourly temperature data from Open-Meteo archive...\n")

    for metric in city_keys:
        city_name, lat, lon, city_tz = CITIES[metric]
        print(f"  {city_name:<22} ...", end="", flush=True)

        hourly = _fetch_openmeteo_hourly(lat, lon, start_date, end_date)
        if not hourly:
            print(" SKIP (no data)")
            continue

        # Group by local calendar date
        by_date: dict[date, list[dict]] = defaultdict(list)
        for obs in hourly:
            local_dt = obs["time"].astimezone(city_tz)
            by_date[local_dt.date()].append({
                "local_hour": local_dt.hour,
                "temp_f":     obs["temp_f"],
                "local_dt":   local_dt,
            })

        n_signals = 0
        for d, obs_list in sorted(by_date.items()):
            obs_list.sort(key=lambda x: x["local_hour"])
            # Actual daily high = max observed temp for the day
            actual_high = max(o["temp_f"] for o in obs_list)

            for entry_h in entry_hours_local:
                # Find the observation closest to entry_h local
                entry_obs = min(obs_list, key=lambda o: abs(o["local_hour"] - entry_h),
                                default=None)
                if entry_obs is None:
                    continue

                forecast_at_entry = entry_obs["temp_f"]
                sigma = _sigma_at_hour(metric, entry_h, d.month)

                # Simulate several strike levels around the forecast
                for offset in strike_offsets_f:
                    strike = round(forecast_at_entry) + offset

                    p_yes = _p_above(forecast_at_entry, strike, sigma)
                    cost_cents = round(p_yes * 100)
                    if cost_cents <= 0 or cost_cents >= 100:
                        continue

                    # YES wins if actual high > strike
                    yes_won = actual_high > strike

                    # Implied P at exit hour (more data → sigma shrinks)
                    for hold_h in hold_hours:
                        exit_h = min(entry_h + hold_h, 17)  # cap at 5pm
                        sigma_exit = _sigma_at_hour(metric, exit_h, d.month)
                        # Forecast at exit ≈ same (it's the observation-driven
                        # rolling estimate; we use actual temp at exit_h as the
                        # "updated forecast" available at that hour)
                        exit_obs = min(obs_list, key=lambda o: abs(o["local_hour"] - exit_h),
                                       default=None)
                        if exit_obs is None:
                            continue
                        forecast_at_exit = exit_obs["temp_f"]
                        p_yes_exit = _p_above(forecast_at_exit, strike, sigma_exit)
                        exit_cents = round(p_yes_exit * 100)

                        # P&L: we sell at exit implied price
                        pnl = exit_cents - cost_cents
                        all_results[(entry_h, hold_h)].append(pnl)

                # Track signal accuracy (edge > threshold) — for directional quality
                edge_f = abs(forecast_at_entry - round(forecast_at_entry))
                strong = forecast_at_entry > round(forecast_at_entry) + 1  # 1°F above strike
                if strong:
                    wins_by_edge["edge≥1°F"].append(yes_won)
                    n_signals += 1
                if forecast_at_entry > round(forecast_at_entry) + 3:
                    wins_by_edge["edge≥3°F"].append(yes_won)

        print(f" {n_signals} strong signals")

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  IMPLIED-PRICE P&L (buy YES at forecast-implied prob, sell N hours later)")
    print(f"  (Positive = market correctly re-prices as day progresses)\n")
    print(f"  {'Entry h':<9}  {'Hold':>5}  {'N':>5}  {'AvgP&L':>8}  {'Win%':>7}  {'TotalP&L':>10}")
    print(f"  {'-'*55}")

    for entry_h in entry_hours_local:
        for hold_h in hold_hours:
            series = all_results.get((entry_h, hold_h), [])
            if not series:
                continue
            avg  = sum(series) / len(series)
            wins = sum(1 for p in series if p > 0)
            tot  = sum(series)
            print(
                f"  {entry_h:02d}:00 local  {hold_h:>4}h  "
                f"{len(series):>5}  {avg:>+7.1f}¢  {wins/len(series):>6.1%}  {tot:>+9.0f}¢"
            )
        print()

    print(f"{'─'*70}")
    print(f"  DIRECTIONAL ACCURACY (does forecast edge predict YES settlement?)\n")
    print(f"  {'Edge category':<14}  {'N':>5}  {'YES win%':>9}  {'note'}")
    print(f"  {'-'*50}")
    for label in sorted(wins_by_edge):
        ws = wins_by_edge[label]
        if not ws:
            continue
        wr = sum(ws) / len(ws)
        note = "strong edge" if wr > 0.70 else ("weak" if wr < 0.55 else "ok")
        print(f"  {label:<14}  {len(ws):>5}  {wr:>8.1%}  {note}")

    print(f"\n{'═'*70}")
    print("  Interpretation guide:")
    print("  • avg_pnl of implied-price simulation measures whether forecasts")
    print("    systematically re-price upward during the day (momentum effect).")
    print("  • Win% of directional signal tests if 'forecast above strike'")
    print("    actually predicts YES settlement (signal accuracy).")
    print("  • Best entry hours: rows with highest avg_pnl and fewest hours hold.")
    print("  • In live mode (--mode live), avg_pnl is in ACTUAL Kalshi cents,")
    print("    not implied price — a direct measure of market lag.")
    print(f"{'═'*70}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest forecast-driven momentum strategy."
    )
    parser.add_argument(
        "--mode", choices=["live", "synthetic"], default="synthetic",
        help="live = uses raw_forecasts DB; synthetic = Open-Meteo history (default: synthetic)"
    )
    parser.add_argument("--db", default="opportunity_log.db",
                        help="Path to opportunity_log.db (live mode only)")
    parser.add_argument("--days", type=int, default=14,
                        help="Days of history for synthetic mode (default: 14)")
    parser.add_argument("--cities", nargs="+", default=None, metavar="ABBREV",
                        help="City abbreviations to include: bos nyc chi lax sfo ... (default: all)")
    parser.add_argument("--min-edge-f", type=float, default=2.0, dest="min_edge_f",
                        help="Min forecast edge in °F above strike to trigger entry (default: 2.0)")
    parser.add_argument("--min-edge", type=float, default=5.0,
                        help="Min yes_bid gap vs implied prob to enter (live mode, cents, default: 5)")
    args = parser.parse_args()

    if args.mode == "live":
        _live_mode(args)
    else:
        _synth_mode(args)


if __name__ == "__main__":
    main()
