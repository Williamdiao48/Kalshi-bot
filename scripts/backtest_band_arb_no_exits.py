"""Exit-based P&L backtest for band-arb NO signals.

For each KXHIGH 'between' band where the METAR running daily max cleared the
band (running_max >= strike_hi), simulates entering a NO position at the NO_ask
price (= 100 - yes_bid) and tracks how P&L evolves toward settlement.

Key mechanics:
- ALL NO-settling bands reach 100¢ by settlement (97% p_win is accurate).
- Hold-to-settlement is the baseline; question is whether early PT beats it.
- At avg 84.5¢ entry, max gain = 15.5¢. A 20% PT requires 101.4¢ — impossible.
- Only entries ≤83¢ can benefit from any % PT; sweep from 3-15% to find optimum.

Data sources (all cached, no API calls):
  data/kxhigh_bands.csv           — 570 NO-settling bands + metadata
  data/band_arb_candle_cache.json — hourly YES_ask/YES_bid candles (143 NO bands)
  data/mesonet_hourly.csv         — METAR running_max by city/date/local_hour

Usage:
  venv/bin/python scripts/backtest_band_arb_no_exits.py
  venv/bin/python scripts/backtest_band_arb_no_exits.py --max-entry 85
  venv/bin/python scripts/backtest_band_arb_no_exits.py --city dca sea phx
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "band_arb_candle_cache.json"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mesonet() -> dict[tuple[str, str, int], float]:
    path = DATA_DIR / "mesonet_hourly.csv"
    data: dict[tuple[str, str, int], float] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            data[(r["city_metric"], r["date"], int(r["local_hour"]))] = float(r["running_max_f"])
    return data


def load_bands() -> list[dict]:
    path = DATA_DIR / "kxhigh_bands.csv"
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            r["strike_lo"]  = float(r["strike_lo"])
            r["strike_hi"]  = float(r["strike_hi"])
            r["band_width"] = float(r["band_width"])
            rows.append(r)
    return rows


def load_cache() -> dict[str, list[dict]]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _city_tz(metric: str):
    from kalshi_bot.news.noaa import CITIES
    info = CITIES.get(metric.replace("temp_low_", "temp_high_"))
    return info[3] if info else timezone.utc


def candles_to_no_hourly(candles: list[dict], city_tz) -> dict[int, dict]:
    """Convert candlesticks to {local_hour: {no_ask_close, no_ask_low (=NO high), no_ask_high}}.

    NO_ask = 100 - YES_bid.
    NO_ask rises as YES_bid falls (market moves against YES / toward NO settlement).
    For PT simulation: NO_ask peak within hour = 100 - YES_bid.low_dollars (YES at its lowest).
    """
    result: dict[int, dict] = {}
    for c in candles:
        yes_bid = c.get("yes_bid", {})
        close_str = yes_bid.get("close_dollars")
        low_str   = yes_bid.get("low_dollars")    # YES low → NO high
        high_str  = yes_bid.get("high_dollars")   # YES high → NO low
        if not close_str:
            continue
        try:
            no_close = max(1, 100 - round(float(close_str) * 100))
            no_high  = max(1, 100 - round(float(low_str)   * 100)) if low_str  else no_close
            no_low   = max(1, 100 - round(float(high_str)  * 100)) if high_str else no_close
        except (ValueError, TypeError):
            continue
        ts = datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc)
        local_hour = ts.astimezone(city_tz).hour
        result[local_hour] = {"close": no_close, "high": no_high, "low": no_low}
    return result


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate_no(
    band: dict,
    entry_hour: int,
    hourly: dict[int, dict],
    pt: float,
    sl: float,
    use_intraday: bool = True,
    trailing: float | None = None,
) -> dict | None:
    """Simulate one NO trade. Entry at NO_ask when temp first clears band."""
    # Find entry price
    entry_data = None
    for delta in [0, 1, -1, 2, -2]:
        entry_data = hourly.get(entry_hour + delta)
        if entry_data:
            break
    if not entry_data or entry_data["close"] <= 0:
        return None

    entry_ask = entry_data["close"]
    # PT target: NO_ask must reach entry * (1 + pt). Cap at 100¢ (max NO value).
    pt_price  = min(100, entry_ask * (1 + pt))
    sl_price  = entry_ask * (1 - sl)

    exit_hour   = None
    exit_price  = None
    exit_reason = None
    peak_ask    = entry_ask

    future_hours = sorted(h for h in hourly if h > entry_hour)
    for hour in future_hours:
        d = hourly[hour]
        hour_high  = d["high"]  if use_intraday else d["close"]
        hour_low   = d["low"]   if use_intraday else d["close"]
        hour_close = d["close"]

        if hour_high > peak_ask:
            peak_ask = hour_high

        # Trailing stop: if peak exceeded entry and close pulls back by trail%
        if trailing is not None and peak_ask > entry_ask:
            trail_floor = peak_ask * (1 - trailing)
            if hour_close <= trail_floor:
                exit_hour   = hour
                exit_price  = hour_close
                exit_reason = "trailing_stop"
                break

        # Fixed PT: intraday high reached target
        if hour_high >= pt_price:
            exit_hour   = hour
            exit_price  = min(100, hour_high)
            exit_reason = "profit_take"
            break

        # Stop-loss: intraday low fell to SL level
        if hour_low <= sl_price:
            exit_hour   = hour
            exit_price  = max(0, hour_low)
            exit_reason = "stop_loss"
            break

    if exit_reason is None:
        # Held to settlement — NO always settles to 100¢ (band resolved NO)
        exit_reason = "settlement"
        exit_price  = 100
        exit_hour   = 99

    pnl = exit_price - entry_ask
    return {
        "ticker":      band["ticker"],
        "metric":      band["metric"],
        "date":        band["date"],
        "entry_hour":  entry_hour,
        "entry_ask":   entry_ask,
        "peak_ask":    peak_ask,
        "exit_hour":   exit_hour,
        "exit_price":  exit_price,
        "exit_reason": exit_reason,
        "pnl_cents":   pnl,
        "hours_held":  (exit_hour - entry_hour) if exit_hour != 99 else None,
    }


def run_sim(
    triggered: list[tuple[dict, int]],
    cache: dict,
    pt: float,
    sl: float,
    use_intraday: bool = True,
    trailing: float | None = None,
    max_entry: int = 100,
) -> list[dict]:
    results = []
    for band, entry_hour in triggered:
        candles = cache.get(band["ticker"], [])
        if not candles:
            continue
        tz = _city_tz(band["metric"])
        hourly = candles_to_no_hourly(candles, tz)
        res = simulate_no(band, entry_hour, hourly, pt, sl,
                          use_intraday=use_intraday, trailing=trailing)
        if res is None:
            continue
        if res["entry_ask"] > max_entry:
            continue
        results.append(res)
    return results


def avg_pnl(sims: list[dict]) -> float:
    return sum(s["pnl_cents"] for s in sims) / len(sims) if sims else 0.0


# ── Reports ───────────────────────────────────────────────────────────────────

def print_reports(triggered: list[tuple[dict, int]], cache: dict) -> None:

    # ── 1. Hold-to-settlement baseline ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("1. HOLD-TO-SETTLEMENT BASELINE  (all NO-settling bands with candle data)")
    print(f"{'='*70}")
    all_triggered = []
    for band, entry_hour in triggered:
        candles = cache.get(band["ticker"], [])
        if not candles:
            continue
        tz = _city_tz(band["metric"])
        hourly = candles_to_no_hourly(candles, tz)
        entry_data = None
        for delta in [0, 1, -1, 2, -2]:
            entry_data = hourly.get(entry_hour + delta)
            if entry_data:
                break
        if not entry_data or entry_data["close"] <= 0:
            continue
        entry_ask = entry_data["close"]
        future_hours = sorted(h for h in hourly if h > entry_hour)
        hours_to_95 = next((h - entry_hour for h in future_hours
                            if hourly[h]["close"] >= 95), None)
        all_triggered.append({
            "metric":       band["metric"],
            "entry_hour":   entry_hour,
            "entry_ask":    entry_ask,
            "gain_at_settle": 100 - entry_ask,
            "gain_pct":     (100 - entry_ask) / entry_ask * 100,
            "hours_to_95":  hours_to_95,
        })

    n = len(all_triggered)
    print(f"  Bands simulated:        {n}")
    print(f"  Avg entry NO_ask:       {sum(r['entry_ask'] for r in all_triggered)/n:.1f}¢")
    print(f"  Avg gain (hold):        {sum(r['gain_at_settle'] for r in all_triggered)/n:.1f}¢")
    print(f"  Avg gain%:              {sum(r['gain_pct'] for r in all_triggered)/n:.0f}%")
    settled_95 = [r for r in all_triggered if r["hours_to_95"] is not None]
    print(f"  Reach 95¢+:             {len(settled_95)}/{n} "
          f"(avg {sum(r['hours_to_95'] for r in settled_95)/max(len(settled_95),1):.1f}h)")

    print(f"\n  {'Range':>10}  {'N':>4}  {'AvgEntry':>9}  {'AvgGain':>8}  {'Gain%':>6}  {'Hrs→95¢':>8}")
    buckets = [(0,40,"≤40¢"),(40,60,"41-60¢"),(60,80,"61-80¢"),(80,101,"≥81¢")]
    for lo, hi, label in buckets:
        g = [r for r in all_triggered if lo < r["entry_ask"] <= hi]
        if not g: continue
        h95 = [r["hours_to_95"] for r in g if r["hours_to_95"] is not None]
        print(f"  {label:>10}  {len(g):>4}  {sum(r['entry_ask'] for r in g)/len(g):>8.1f}¢"
              f"  {sum(r['gain_at_settle'] for r in g)/len(g):>7.1f}¢"
              f"  {sum(r['gain_pct'] for r in g)/len(g):>5.0f}%"
              f"  {sum(h95)/max(len(h95),1):>7.1f}h")

    # ── 2. PT threshold sweep ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("2. PT THRESHOLD SWEEP vs HOLD-TO-SETTLEMENT  (intraday high, SL=95%)")
    print(f"{'='*70}")
    hold_pnl = avg_pnl(run_sim(triggered, cache, pt=0.0001, sl=0.95))  # near-zero PT = always hold
    print(f"  Hold-to-settle baseline: {hold_pnl:+.2f}¢/trade  (N={len(run_sim(triggered, cache, pt=0.0001, sl=0.95))})")
    print(f"\n  {'PT':>6}  {'N':>4}  {'PT%':>5}  {'SL%':>5}  {'Set%':>5}  {'AvgP&L':>8}  {'vs_Hold':>8}")
    best_pnl = hold_pnl; best_pt = None
    for pt in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
        sims = run_sim(triggered, cache, pt=pt, sl=0.95)
        if not sims: continue
        n_s = len(sims)
        pt_r  = sum(1 for s in sims if s["exit_reason"] == "profit_take") / n_s
        sl_r  = sum(1 for s in sims if s["exit_reason"] == "stop_loss") / n_s
        set_r = sum(1 for s in sims if s["exit_reason"] == "settlement") / n_s
        pnl   = avg_pnl(sims)
        vs_hold = pnl - hold_pnl
        marker = " ←best" if pnl > best_pnl else ""
        if pnl > best_pnl:
            best_pnl = pnl; best_pt = pt
        print(f"  PT={pt*100:>4.0f}%  {n_s:>4}  {pt_r*100:>5.0f}%  {sl_r*100:>5.0f}%"
              f"  {set_r*100:>5.0f}%  {pnl:>+7.2f}¢  {vs_hold:>+7.2f}¢{marker}")

    best_pt = best_pt or 0.05  # fallback

    # ── 3. MAX_NO_ASK entry filter sweep ───────────────────────────────────
    print(f"\n{'='*70}")
    print(f"3. MAX_NO_ASK ENTRY FILTER  (PT={best_pt*100:.0f}%  SL=95%  intraday high)")
    print(f"{'='*70}")
    print(f"  Restricting max entry price — excludes high-entry bands with tiny potential gain.")
    print(f"  {'MaxEntry':>9}  {'N':>4}  {'AvgEntry':>9}  {'PT%':>5}  {'Set%':>5}  {'AvgP&L':>8}  {'Avg_hold':>9}")
    for max_e in [99, 95, 90, 85, 80, 75, 70]:
        sims = run_sim(triggered, cache, pt=best_pt, sl=0.95, max_entry=max_e)
        hold = run_sim(triggered, cache, pt=0.0001, sl=0.95, max_entry=max_e)
        if not sims: continue
        n_s = len(sims)
        pt_r  = sum(1 for s in sims if s["exit_reason"] == "profit_take") / n_s
        set_r = sum(1 for s in sims if s["exit_reason"] == "settlement") / n_s
        pnl   = avg_pnl(sims)
        hold_pnl_f = avg_pnl(hold) if hold else 0
        print(f"  ≤{max_e:>7}¢  {n_s:>4}  {sum(s['entry_ask'] for s in sims)/n_s:>8.1f}¢"
              f"  {pt_r*100:>5.0f}%  {set_r*100:>5.0f}%  {pnl:>+7.2f}¢  {hold_pnl_f:>+8.2f}¢")

    # ── 4. By city ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"4. BY CITY  (PT={best_pt*100:.0f}%  SL=95%  all entries)")
    print(f"{'='*70}")
    sims_all = run_sim(triggered, cache, pt=best_pt, sl=0.95)
    by_city: dict[str, list] = {}
    for s in sims_all:
        city = s["metric"].replace("temp_high_", "")
        by_city.setdefault(city, []).append(s)
    print(f"  {'City':>6}  {'N':>4}  {'PT%':>5}  {'AvgEntry':>9}  {'AvgP&L':>8}  {'Hold_PnL':>9}")
    for city in sorted(by_city, key=lambda c: -avg_pnl(by_city[c])):
        g = by_city[city]
        n_g = len(g)
        pt_r = sum(1 for s in g if s["exit_reason"] == "profit_take") / n_g
        ae   = sum(s["entry_ask"] for s in g) / n_g
        hold = 100 - ae  # avg hold gain
        print(f"  {city:>6}  {n_g:>4}  {pt_r*100:>5.0f}%  {ae:>8.1f}¢  "
              f"{avg_pnl(g):>+7.2f}¢  {hold:>+8.2f}¢")

    # ── 5. Trailing stop for NO ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("5. TRAILING STOP FOR NO SIGNALS  (protects against false-signal reversal)")
    print(f"{'='*70}")
    print(f"  {'Strategy':>28}  {'N':>4}  {'PT%':>5}  {'TS%':>5}  {'Set%':>5}  {'AvgP&L':>8}")
    print(f"  {'Hold to settlement':>28}  {len(run_sim(triggered,cache,pt=0.0001,sl=0.95)):>4}"
          f"  {'0%':>5}  {'0%':>5}  {'~100%':>5}  {avg_pnl(run_sim(triggered,cache,pt=0.0001,sl=0.95)):>+7.2f}¢")
    for pt_t, trail in [(0.0001, None), (best_pt, None),
                        (999, 0.05), (999, 0.10), (999, 0.15), (999, 0.20),
                        (best_pt, 0.05), (best_pt, 0.10)]:
        label = (f"Fixed PT={pt_t*100:.0f}%" if trail is None and pt_t < 100
                 else f"Trail {trail*100:.0f}%" if pt_t >= 100
                 else f"PT={pt_t*100:.0f}%+Trail={trail*100:.0f}%")
        if label == "Fixed PT=0%": continue
        sims = run_sim(triggered, cache, pt=pt_t, sl=0.95, trailing=trail)
        if not sims: continue
        n_s = len(sims)
        pt_r  = sum(1 for s in sims if s["exit_reason"] == "profit_take") / n_s
        ts_r  = sum(1 for s in sims if s["exit_reason"] == "trailing_stop") / n_s
        set_r = sum(1 for s in sims if s["exit_reason"] == "settlement") / n_s
        print(f"  {label:>28}  {n_s:>4}  {pt_r*100:>5.0f}%  {ts_r*100:>5.0f}%"
              f"  {set_r*100:>5.0f}%  {avg_pnl(sims):>+7.2f}¢")

    # ── 6. Peak gain analysis ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("6. PEAK NO_ASK ANALYSIS  (max intraday NO_ask after entry)")
    print(f"{'='*70}")
    peak_data = []
    for band, entry_hour in triggered:
        candles = cache.get(band["ticker"], [])
        if not candles: continue
        tz = _city_tz(band["metric"])
        hourly = candles_to_no_hourly(candles, tz)
        entry_data = None
        for delta in [0, 1, -1, 2, -2]:
            entry_data = hourly.get(entry_hour + delta)
            if entry_data: break
        if not entry_data: continue
        entry_ask = entry_data["close"]
        future_highs = [hourly[h]["high"] for h in sorted(hourly) if h > entry_hour]
        peak = max(future_highs) if future_highs else entry_ask
        peak_data.append({"entry_ask": entry_ask, "peak": peak,
                          "peak_gain_pct": (peak - entry_ask) / entry_ask * 100})

    if peak_data:
        for pct in [3, 5, 7, 10, 15]:
            n_reach = sum(1 for r in peak_data if r["peak_gain_pct"] >= pct)
            print(f"  Peak gain ≥{pct:>2}%:  {n_reach:>3}/{len(peak_data)}  "
                  f"({n_reach/len(peak_data)*100:.0f}%)")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    hold_base = avg_pnl(run_sim(triggered, cache, pt=0.0001, sl=0.95))
    best_sims = run_sim(triggered, cache, pt=best_pt, sl=0.95)
    improvement = avg_pnl(best_sims) - hold_base if best_sims else 0
    print(f"  Hold-to-settlement avg P&L:  {hold_base:+.2f}¢/trade")
    print(f"  Best PT ({best_pt*100:.0f}%) avg P&L:       {avg_pnl(best_sims):+.2f}¢/trade")
    print(f"  Improvement from PT:         {improvement:+.2f}¢/trade")
    if improvement > 0.5:
        print(f"\n  → ADD \"band_arb:no\": {best_pt} to EXIT_SOURCE_PROFIT_TAKE")
    else:
        print(f"\n  → Hold-to-settlement is optimal. No PT threshold needed for NO signals.")
        print(f"  → Consider tightening BAND_ARB_MAX_NO_ASK to improve avg gain per trade.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    log.info("Loading data...")
    mesonet = load_mesonet()
    bands   = load_bands()
    cache   = load_cache()

    city_filter = set(args.city) if args.city else None

    # Find NO-settling bands where temp cleared the band
    triggered: list[tuple[dict, int]] = []
    no_bands = [b for b in bands if b["result"] == "no"]
    for b in no_bands:
        metric = b["metric"]
        if city_filter:
            suffix = metric.replace("temp_high_", "")
            if suffix not in city_filter:
                continue
        s_hi = b["strike_hi"]
        for hour in range(6, 22):
            v = mesonet.get((metric, b["date"], hour))
            if v is not None and v >= s_hi:
                triggered.append((b, hour))
                break

    log.info("NO bands with temp trigger: %d / %d", len(triggered), len(no_bands))
    with_candles = sum(1 for b, _ in triggered if cache.get(b["ticker"]))
    log.info("Bands with candle data:     %d", with_candles)

    print_reports(triggered, cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exit-based P&L backtest for band-arb NO signals.")
    parser.add_argument("--city", nargs="+", default=None,
                        help="Filter to city suffixes e.g. --city dca sea")
    parser.add_argument("--max-entry", type=int, default=100,
                        help="Only include bands with entry NO_ask ≤ this (default: 100)")
    args = parser.parse_args()
    main(args)
