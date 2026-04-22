"""Exit-based P&L backtest for band-arb YES signal.

For each KXHIGH 'between' band where the METAR running daily max entered the
band intraday, simulates entering a YES position and tracks whether the exit
manager's profit-take or stop-loss would have triggered before settlement.

This answers: "Even if the temperature later escapes the band (resolves NO),
does the YES price spike enough intraday to capture a profit-take exit?"

Methodology
-----------
1. Entry trigger: first local hour where running_max ∈ [strike_lo, strike_hi]
2. Entry price: YES_ask at that hour (from Kalshi candlestick API)
3. Each subsequent hour: check if YES_ask crossed profit-take or stop-loss
4. Exit at first trigger, or hold to settlement (100¢ YES or 0¢ NO)

Profit-take: YES_ask >= entry * (1 + PT_THRESHOLD)  (default 50% gain)
Stop-loss:   YES_ask <= entry * (1 - SL_THRESHOLD)  (default 50% loss)

Output: console report + data/band_arb_yes_exit_sim.csv

Usage:
  venv/bin/python scripts/backtest_band_arb_yes_exits.py
  venv/bin/python scripts/backtest_band_arb_yes_exits.py --pt 0.30 --sl 0.40
  venv/bin/python scripts/backtest_band_arb_yes_exits.py --city aus lax --no-cache
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.auth import generate_headers          # noqa: E402
from kalshi_bot.markets import KALSHI_API_BASE        # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "band_arb_candle_cache.json"
_SEM       = asyncio.Semaphore(6)

# Default exit thresholds
DEFAULT_PT = 0.50   # profit-take: 50% gain
DEFAULT_SL = 0.50   # stop-loss:   50% loss


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mesonet() -> dict[tuple[str, str, int], float]:
    path = DATA_DIR / "mesonet_hourly.csv"
    if not path.exists():
        log.error("mesonet_hourly.csv not found — run fetch_mesonet_history.py first")
        sys.exit(1)
    data: dict[tuple[str, str, int], float] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            data[(r["city_metric"], r["date"], int(r["local_hour"]))] = float(r["running_max_f"])
    return data


def load_bands() -> list[dict]:
    path = DATA_DIR / "kxhigh_bands.csv"
    if not path.exists():
        log.error("kxhigh_bands.csv not found — run fetch_kxhigh_history.py first")
        sys.exit(1)
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


def save_cache(cache: dict[str, list[dict]]) -> None:
    CACHE_FILE.write_text(json.dumps(cache))


# ── Kalshi candlestick fetch ──────────────────────────────────────────────────

async def fetch_candles(
    session: aiohttp.ClientSession,
    ticker: str,
    market_date_str: str,
) -> list[dict]:
    """Fetch hourly candlesticks for the trading day preceding settlement."""
    mkt_date = datetime.strptime(market_date_str, "%Y-%m-%d").date()
    prev_day = mkt_date - timedelta(days=1)
    start_ts = int(datetime(prev_day.year, prev_day.month, prev_day.day,
                            12, 0, tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime(mkt_date.year, mkt_date.month, mkt_date.day,
                            7, 0, tzinfo=timezone.utc).timestamp())

    # Series is everything before the last two dash-segments
    series = ticker.rsplit("-", 2)[0]
    path   = f"/trade-api/v2/series/{series}/markets/{ticker}/candlesticks"
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
                    log.warning("Rate-limited on %s — sleeping 3s", ticker)
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


def candles_to_hourly_ask(
    candles: list[dict],
    city_tz,
) -> dict[int, float]:
    """Convert candlestick list to {local_hour: yes_ask_cents} dict."""
    result: dict[int, float] = {}
    for c in candles:
        ask = c.get("yes_ask", {})
        close_str = ask.get("close_dollars")
        if close_str is None:
            continue
        try:
            ask_cents = round(float(close_str) * 100)
        except (ValueError, TypeError):
            continue
        ts = datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc)
        local_hour = ts.astimezone(city_tz).hour
        result[local_hour] = ask_cents
    return result


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_exits(
    band: dict,
    entry_hour: int,
    entry_temp: float,
    hourly_ask: dict[int, float],
    pt_threshold: float,
    sl_threshold: float,
) -> dict | None:
    """Simulate entry and exit for one band.

    Returns a result dict or None if entry price is unavailable.
    """
    entry_ask = hourly_ask.get(entry_hour)
    if entry_ask is None:
        # Try adjacent hours (candlestick may be in next/prev hour)
        for delta in [1, -1, 2, -2]:
            entry_ask = hourly_ask.get(entry_hour + delta)
            if entry_ask is not None:
                break
    if entry_ask is None or entry_ask <= 0:
        return None

    pt_price = entry_ask * (1 + pt_threshold)
    sl_price = entry_ask * (1 - sl_threshold)

    exit_hour  = None
    exit_price = None
    exit_reason = None

    # Scan hours after entry
    for hour in sorted(h for h in hourly_ask if h > entry_hour):
        ask = hourly_ask[hour]
        if ask >= pt_price:
            exit_hour   = hour
            exit_price  = ask
            exit_reason = "profit_take"
            break
        if ask <= sl_price:
            exit_hour   = hour
            exit_price  = ask
            exit_reason = "stop_loss"
            break

    if exit_reason is None:
        # Held to settlement
        exit_reason = "settlement"
        exit_price  = 100 if band["result"] == "yes" else 0
        exit_hour   = 99  # sentinel

    pnl_cents = exit_price - entry_ask

    return {
        "ticker":      band["ticker"],
        "metric":      band["metric"],
        "date":        band["date"],
        "result":      band["result"],
        "strike_lo":   band["strike_lo"],
        "strike_hi":   band["strike_hi"],
        "entry_hour":  entry_hour,
        "entry_temp":  entry_temp,
        "entry_ask":   entry_ask,
        "exit_hour":   exit_hour,
        "exit_price":  exit_price,
        "exit_reason": exit_reason,
        "pnl_cents":   pnl_cents,
        "pnl_pct":     round(pnl_cents / entry_ask * 100, 1),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_reports(sims: list[dict], pt: float, sl: float) -> None:
    if not sims:
        print("No simulation results.")
        return

    n = len(sims)
    total_pnl = sum(s["pnl_cents"] for s in sims)
    avg_entry = sum(s["entry_ask"] for s in sims) / n

    by_reason = {}
    for s in sims:
        by_reason.setdefault(s["exit_reason"], []).append(s)

    pt_sims = by_reason.get("profit_take", [])
    sl_sims = by_reason.get("stop_loss", [])
    set_sims = by_reason.get("settlement", [])
    set_yes  = [s for s in set_sims if s["result"] == "yes"]
    set_no   = [s for s in set_sims if s["result"] == "no"]

    print(f"\n{'='*62}")
    print(f"BAND-ARB YES EXIT BACKTEST  (PT={pt*100:.0f}%  SL={sl*100:.0f}%)")
    print(f"{'='*62}")
    print(f"  Bands simulated      : {n}")
    print(f"  Avg entry YES_ask    : {avg_entry:.0f}¢")
    print(f"  Net P&L (all trades) : {total_pnl/n:+.1f}¢/trade  "
          f"(${total_pnl/n/100:+.2f}/contract)")
    print()
    print(f"  Exit breakdown:")
    print(f"    Profit-take  : {len(pt_sims):3d} ({len(pt_sims)/n*100:.0f}%)"
          f"  avg_gain={sum(s['pnl_cents'] for s in pt_sims)/max(len(pt_sims),1):+.1f}¢")
    print(f"    Stop-loss    : {len(sl_sims):3d} ({len(sl_sims)/n*100:.0f}%)"
          f"  avg_loss={sum(s['pnl_cents'] for s in sl_sims)/max(len(sl_sims),1):+.1f}¢")
    print(f"    Settlement YES: {len(set_yes):3d} ({len(set_yes)/n*100:.0f}%)"
          f"  avg_gain={sum(s['pnl_cents'] for s in set_yes)/max(len(set_yes),1):+.1f}¢")
    print(f"    Settlement NO : {len(set_no):3d} ({len(set_no)/n*100:.0f}%)"
          f"  avg_loss={sum(s['pnl_cents'] for s in set_no)/max(len(set_no),1):+.1f}¢")

    # ── By true outcome ──────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("BY TRUE OUTCOME")
    print(f"{'='*62}")
    for outcome in ("yes", "no"):
        subset = [s for s in sims if s["result"] == outcome]
        if not subset:
            continue
        pnl = sum(s["pnl_cents"] for s in subset)
        pt_n  = sum(1 for s in subset if s["exit_reason"] == "profit_take")
        sl_n  = sum(1 for s in subset if s["exit_reason"] == "stop_loss")
        set_n = sum(1 for s in subset if s["exit_reason"] == "settlement")
        print(f"\n  True outcome = {outcome.upper()}  (N={len(subset)})")
        print(f"    Net P&L     : {pnl/len(subset):+.1f}¢/trade")
        print(f"    profit_take : {pt_n} ({pt_n/len(subset)*100:.0f}%)")
        print(f"    stop_loss   : {sl_n} ({sl_n/len(subset)*100:.0f}%)")
        print(f"    settlement  : {set_n} ({set_n/len(subset)*100:.0f}%)")

    # ── By entry hour ────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("BY ENTRY HOUR  (local time)")
    print(f"{'='*62}")
    by_hour: dict[int, list[dict]] = {}
    for s in sims:
        by_hour.setdefault(s["entry_hour"], []).append(s)
    print(f"  {'Hour':>4}  {'N':>4}  {'PT%':>5}  {'SL%':>5}  {'AvgP&L':>8}  {'YES%':>6}")
    for hour in sorted(by_hour):
        grp = by_hour[hour]
        pt_r  = sum(1 for s in grp if s["exit_reason"] == "profit_take") / len(grp)
        sl_r  = sum(1 for s in grp if s["exit_reason"] == "stop_loss")   / len(grp)
        avg_p = sum(s["pnl_cents"] for s in grp) / len(grp)
        yes_r = sum(1 for s in grp if s["result"] == "yes") / len(grp)
        print(f"  {hour:>4}h  {len(grp):>4}  {pt_r*100:>5.0f}%  {sl_r*100:>5.0f}%"
              f"  {avg_p:>+7.1f}¢  {yes_r*100:>5.0f}%")

    # ── By entry price bucket ─────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("BY ENTRY YES_ASK PRICE")
    print(f"{'='*62}")
    buckets = [(0,20,"<20¢"),(20,35,"20-35¢"),(35,50,"35-50¢"),
               (50,65,"50-65¢"),(65,85,"65-85¢")]
    print(f"  {'Price':>8}  {'N':>4}  {'PT%':>5}  {'SL%':>5}  {'AvgP&L':>8}  {'Net$/contract':>14}")
    for lo, hi, label in buckets:
        grp = [s for s in sims if lo <= s["entry_ask"] < hi]
        if not grp: continue
        pt_r  = sum(1 for s in grp if s["exit_reason"] == "profit_take") / len(grp)
        sl_r  = sum(1 for s in grp if s["exit_reason"] == "stop_loss")   / len(grp)
        avg_p = sum(s["pnl_cents"] for s in grp) / len(grp)
        print(f"  {label:>8}  {len(grp):>4}  {pt_r*100:>5.0f}%  {sl_r*100:>5.0f}%"
              f"  {avg_p:>+7.1f}¢  ${avg_p/100:>+.2f}")

    # ── Sweep: what if we only enter at certain price ranges? ────────────
    print(f"\n{'='*62}")
    print("P&L SWEEP — ENTRY PRICE FILTER")
    print(f"{'='*62}")
    print(f"  {'Max_entry':>10}  {'N':>4}  {'PT%':>5}  {'SL%':>5}  {'AvgP&L':>8}")
    for max_ask in [30, 40, 50, 60, 70, 85]:
        grp = [s for s in sims if s["entry_ask"] <= max_ask]
        if not grp: continue
        pt_r  = sum(1 for s in grp if s["exit_reason"] == "profit_take") / len(grp)
        sl_r  = sum(1 for s in grp if s["exit_reason"] == "stop_loss")   / len(grp)
        avg_p = sum(s["pnl_cents"] for s in grp) / len(grp)
        print(f"  ≤{max_ask:>8}¢  {len(grp):>4}  {pt_r*100:>5.0f}%  {sl_r*100:>5.0f}%"
              f"  {avg_p:>+7.1f}¢")

    # ── By city ──────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("BY CITY")
    print(f"{'='*62}")
    by_city: dict[str, list[dict]] = {}
    for s in sims:
        city = s["metric"].replace("temp_high_", "")
        by_city.setdefault(city, []).append(s)
    print(f"  {'City':>6}  {'N':>4}  {'YES%':>5}  {'PT%':>5}  {'SL%':>5}  {'AvgP&L':>8}  {'AvgEntry':>9}")
    for city in sorted(by_city, key=lambda c: -sum(s["pnl_cents"] for s in by_city[c])/len(by_city[c])):
        grp = by_city[city]
        yes_r = sum(1 for s in grp if s["result"] == "yes") / len(grp)
        pt_r  = sum(1 for s in grp if s["exit_reason"] == "profit_take") / len(grp)
        sl_r  = sum(1 for s in grp if s["exit_reason"] == "stop_loss")   / len(grp)
        avg_p = sum(s["pnl_cents"] for s in grp) / len(grp)
        avg_e = sum(s["entry_ask"] for s in grp) / len(grp)
        print(f"  {city:>6}  {len(grp):>4}  {yes_r*100:>5.0f}%  {pt_r*100:>5.0f}%"
              f"  {sl_r*100:>5.0f}%  {avg_p:>+7.1f}¢  {avg_e:>8.0f}¢")

    # ── Min entry price sweep ─────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("MIN ENTRY PRICE SWEEP  (only enter if YES_ask >= threshold)")
    print(f"{'='*62}")
    print(f"  {'Min_entry':>10}  {'N':>4}  {'YES%':>5}  {'PT%':>5}  {'SL%':>5}  {'AvgP&L':>8}")
    for min_ask in [0, 15, 20, 30, 40, 50]:
        grp = [s for s in sims if s["entry_ask"] >= min_ask]
        if not grp: continue
        yes_r = sum(1 for s in grp if s["result"] == "yes") / len(grp)
        pt_r  = sum(1 for s in grp if s["exit_reason"] == "profit_take") / len(grp)
        sl_r  = sum(1 for s in grp if s["exit_reason"] == "stop_loss")   / len(grp)
        avg_p = sum(s["pnl_cents"] for s in grp) / len(grp)
        print(f"  ≥{min_ask:>8}¢  {len(grp):>4}  {yes_r*100:>5.0f}%  {pt_r*100:>5.0f}%"
              f"  {sl_r*100:>5.0f}%  {avg_p:>+7.1f}¢")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    from kalshi_bot.news.noaa import CITIES

    mesonet = load_mesonet()
    bands   = load_bands()
    cache   = {} if args.no_cache else load_cache()

    # City filter
    city_filter = set(args.city) if args.city else None

    # Find bands where running_max entered the band intraday
    triggered: list[tuple[dict, int, float]] = []  # (band, entry_hour, entry_temp)
    for b in bands:
        metric = b["metric"]
        if city_filter:
            suffix = metric.replace("temp_high_", "")
            if suffix not in city_filter:
                continue
        s_lo = b["strike_lo"]
        s_hi = b["strike_hi"]
        for hour in range(args.hour_start, args.hour_end + 1):
            v = mesonet.get((metric, b["date"], hour))
            if v is not None and s_lo <= v <= s_hi:
                triggered.append((b, hour, v))
                break

    log.info("Bands with intraday temperature entry: %d / %d", len(triggered), len(bands))

    # Fetch candlesticks (with cache)
    to_fetch = [(b, h, t) for b, h, t in triggered if b["ticker"] not in cache]
    log.info("Fetching candlesticks: %d new, %d cached",
             len(to_fetch), len(triggered) - len(to_fetch))

    if to_fetch:
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_candles(session, b["ticker"], b["date"]) for b, _, _ in to_fetch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for (b, _, _), result in zip(to_fetch, results):
            if isinstance(result, Exception):
                log.warning("Candle fetch failed for %s: %s", b["ticker"], result)
                cache[b["ticker"]] = []
            else:
                cache[b["ticker"]] = result

        save_cache(cache)
        log.info("Saved candle cache (%d tickers)", len(cache))

    # Simulate exits
    sims: list[dict] = []
    no_price_count = 0
    for b, entry_hour, entry_temp in triggered:
        candles = cache.get(b["ticker"], [])
        if not candles:
            no_price_count += 1
            continue

        # Get city timezone for local hour conversion
        city_entry = CITIES.get(b["metric"])
        if city_entry is None:
            continue
        _, _, _, city_tz = city_entry

        hourly_ask = candles_to_hourly_ask(candles, city_tz)
        result = simulate_exits(
            b, entry_hour, entry_temp, hourly_ask,
            args.pt, args.sl,
        )
        if result is None:
            no_price_count += 1
        else:
            sims.append(result)

    log.info("Simulated: %d trades  (%d skipped — no price data)", len(sims), no_price_count)

    # Save CSV
    out_path = DATA_DIR / "band_arb_yes_exit_sim.csv"
    if sims:
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sims[0].keys()))
            writer.writeheader()
            writer.writerows(sims)
        log.info("Saved %s", out_path)

    print_reports(sims, args.pt, args.sl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exit-based P&L backtest for band-arb YES signal."
    )
    parser.add_argument("--pt",         type=float, default=DEFAULT_PT,
                        help=f"Profit-take threshold (default: {DEFAULT_PT})")
    parser.add_argument("--sl",         type=float, default=DEFAULT_SL,
                        help=f"Stop-loss threshold (default: {DEFAULT_SL})")
    parser.add_argument("--hour-start", type=int,   default=8,
                        help="Earliest entry hour (default: 8)")
    parser.add_argument("--hour-end",   type=int,   default=16,
                        help="Latest entry hour (default: 16)")
    parser.add_argument("--city",       nargs="+",  default=None,
                        help="Filter to city suffixes e.g. --city aus lax")
    parser.add_argument("--no-cache",   action="store_true",
                        help="Ignore cached candlestick data and re-fetch")
    args = parser.parse_args()
    asyncio.run(main(args))
