"""Compare p75 vs p90 entry gate for band-arb YES signals.

For each resolved KXHIGH 'between' band, checks whether the METAR running daily
max was inside the band at the city's p75 and p90 peak-time thresholds.  When
in-band, looks up the Kalshi YES ask price at that hour from the candlestick API
and simulates entering a YES position held to settlement.

Key outputs:
  1. Summary table: p75 vs p90 — trade count, win rate, avg entry, total P&L
  2. Signals gained/lost by waiting from p75 → p90
  3. Breakdowns by city, month, entry price bucket

No-edge filter: if the YES ask is already ≥ 95¢ at the gate time, the market
has fully priced in the outcome — no arbitrage available, marked separately.

Data dependencies (run first if stale):
  venv/bin/python scripts/fetch_mesonet_history.py --days 90
  venv/bin/python scripts/fetch_kxhigh_history.py --days 90

Usage:
  venv/bin/python scripts/backtest_band_arb_yes_gate.py
  venv/bin/python scripts/backtest_band_arb_yes_gate.py --cities dca chi bos
  venv/bin/python scripts/backtest_band_arb_yes_gate.py --months 4 5
  venv/bin/python scripts/backtest_band_arb_yes_gate.py --no-cache
  venv/bin/python scripts/backtest_band_arb_yes_gate.py --out data/gate_comparison.txt
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.auth import generate_headers          # noqa: E402
from kalshi_bot.markets import KALSHI_API_BASE        # noqa: E402
from kalshi_bot.news.noaa import CITIES               # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "band_arb_candle_cache.json"
_SEM       = asyncio.Semaphore(4)

NO_EDGE_THRESHOLD = 95   # cents — market already agrees, no arb


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
            r["strike_lo"] = float(r["strike_lo"])
            r["strike_hi"] = float(r["strike_hi"])
            rows.append(r)
    return rows


def load_gate_minutes() -> tuple[dict, dict]:
    """Load P75_MINUTES and P90_MINUTES from data/peak_hour_p90.py."""
    p90_path = DATA_DIR / "peak_hour_p90.py"
    if not p90_path.exists():
        log.error("peak_hour_p90.py not found — run backtest_peak_hour.py --dict first")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("peak_hour_p90", p90_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, "P75_MINUTES", {}), getattr(mod, "P90_MINUTES", {})


def load_cache() -> dict[str, list[dict]]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict[str, list[dict]]) -> None:
    CACHE_FILE.write_text(json.dumps(cache))


# ── Kalshi candlestick fetch (reused from backtest_band_arb_yes_exits.py) ─────

async def fetch_candles(
    session: aiohttp.ClientSession,
    ticker: str,
    market_date_str: str,
) -> list[dict]:
    mkt_date = datetime.strptime(market_date_str, "%Y-%m-%d").date()
    prev_day = mkt_date - timedelta(days=1)
    start_ts = int(datetime(prev_day.year, prev_day.month, prev_day.day,
                            12, 0, tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime(mkt_date.year, mkt_date.month, mkt_date.day,
                            7, 0, tzinfo=timezone.utc).timestamp())
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


def candles_to_hourly_ask(candles: list[dict], city_tz) -> dict[int, float]:
    """Convert candlestick list to {local_hour: yes_ask_cents} using close price."""
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


def get_ask_at_hour(hourly_ask: dict[int, float], hour: int) -> float | None:
    """Look up ask at gate hour, falling back to adjacent hours."""
    ask = hourly_ask.get(hour)
    if ask is not None:
        return ask
    for delta in [1, -1, 2, -2]:
        ask = hourly_ask.get(hour + delta)
        if ask is not None:
            return ask
    return None


# ── Simulation ────────────────────────────────────────────────────────────────

async def run_simulation(
    bands: list[dict],
    mesonet: dict[tuple[str, str, int], float],
    p75_minutes: dict,
    p90_minutes: dict,
    city_filter: set[str] | None,
    month_filter: set[int] | None,
    use_cache: bool,
) -> list[dict]:
    """Return one record per (band, gate) combination."""

    # Pre-filter bands
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

    log.info("Bands after filter: %d", len(active_bands))

    # Determine which tickers need candlestick fetches
    cache = load_cache() if use_cache else {}
    to_fetch = [b["ticker"] for b in active_bands if b["ticker"] not in cache]
    to_fetch = list(dict.fromkeys(to_fetch))  # deduplicate, preserve order

    if to_fetch:
        log.info("Fetching candlesticks: %d new, %d cached",
                 len(to_fetch), len(active_bands) - len(to_fetch))
        ticker_to_date = {b["ticker"]: b["date"] for b in active_bands}
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_candles(session, t, ticker_to_date[t]) for t in to_fetch]
            results = await asyncio.gather(*tasks)
        for ticker, candles in zip(to_fetch, results):
            cache[ticker] = candles
        save_cache(cache)

    # Build records
    records: list[dict] = []
    gates = [("p75", p75_minutes), ("p90", p90_minutes)]

    for b in active_bands:
        metric    = b["metric"]
        date_str  = b["date"]
        month     = int(date_str.split("-")[1])
        strike_lo = b["strike_lo"]
        strike_hi = b["strike_hi"]
        result    = b["result"]   # "yes" or "no"

        city_entry = CITIES.get(metric)
        if city_entry is None:
            continue
        city_tz = city_entry[3]

        candles    = cache.get(b["ticker"], [])
        hourly_ask = candles_to_hourly_ask(candles, city_tz)

        for gate_label, gate_minutes_dict in gates:
            monthly = gate_minutes_dict.get(metric, {})
            gate_min = monthly.get(month)
            if gate_min is None:
                continue
            gate_hour = gate_min // 60

            running_max = mesonet.get((metric, date_str, gate_hour))
            if running_max is None:
                records.append({
                    "gate": gate_label, "ticker": b["ticker"],
                    "metric": metric, "date": date_str, "month": month,
                    "strike_lo": strike_lo, "strike_hi": strike_hi,
                    "result": result, "gate_hour": gate_hour,
                    "running_max": None, "in_band": False,
                    "yes_ask": None, "outcome": "no_mesonet",
                    "pnl": None,
                })
                continue

            in_band = strike_lo <= running_max <= strike_hi

            if not in_band:
                records.append({
                    "gate": gate_label, "ticker": b["ticker"],
                    "metric": metric, "date": date_str, "month": month,
                    "strike_lo": strike_lo, "strike_hi": strike_hi,
                    "result": result, "gate_hour": gate_hour,
                    "running_max": running_max, "in_band": False,
                    "yes_ask": None, "outcome": "not_in_band",
                    "pnl": None,
                })
                continue

            yes_ask = get_ask_at_hour(hourly_ask, gate_hour)

            if yes_ask is None:
                outcome = "data_unavailable"
                pnl     = None
            elif yes_ask >= NO_EDGE_THRESHOLD:
                outcome = "no_edge"
                pnl     = None
            elif yes_ask <= 0:
                outcome = "data_unavailable"
                pnl     = None
            else:
                pnl     = (100 - yes_ask) if result == "yes" else (-yes_ask)
                outcome = "win" if result == "yes" else "loss"

            records.append({
                "gate": gate_label, "ticker": b["ticker"],
                "metric": metric, "date": date_str, "month": month,
                "strike_lo": strike_lo, "strike_hi": strike_hi,
                "result": result, "gate_hour": gate_hour,
                "running_max": running_max, "in_band": True,
                "yes_ask": yes_ask, "outcome": outcome,
                "pnl": pnl,
            })

    return records


# ── Reporting ─────────────────────────────────────────────────────────────────

def _gate_stats(recs: list[dict]) -> dict:
    bands_checked   = len(recs)
    in_band         = [r for r in recs if r["in_band"]]
    no_edge         = [r for r in in_band if r["outcome"] == "no_edge"]
    unavailable     = [r for r in in_band if r["outcome"] == "data_unavailable"]
    trades          = [r for r in in_band if r["outcome"] in ("win", "loss")]
    wins            = [r for r in trades if r["outcome"] == "win"]
    losses          = [r for r in trades if r["outcome"] == "loss"]
    n_trades        = len(trades)
    win_rate        = len(wins) / n_trades * 100 if n_trades else float("nan")
    avg_entry       = sum(r["yes_ask"] for r in trades) / n_trades if n_trades else float("nan")
    total_pnl_cents = sum(r["pnl"] for r in trades) if trades else 0.0
    avg_pnl         = total_pnl_cents / n_trades if n_trades else float("nan")
    return {
        "bands_checked": bands_checked,
        "in_band": len(in_band),
        "no_edge": len(no_edge),
        "unavailable": len(unavailable),
        "trades": n_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_entry": avg_entry,
        "avg_pnl": avg_pnl,
        "total_pnl_cents": total_pnl_cents,
        "total_pnl_dollars": total_pnl_cents / 100,
    }


def build_report(records: list[dict]) -> str:
    p75_recs = [r for r in records if r["gate"] == "p75"]
    p90_recs = [r for r in records if r["gate"] == "p90"]
    p75s = _gate_stats(p75_recs)
    p90s = _gate_stats(p90_recs)

    lines: list[str] = []

    def row(label: str, p75_val: str, p90_val: str, indent: int = 0) -> None:
        pad = "  " * indent
        lines.append(f"  {pad}{label:<36}  {p75_val:>12}  {p90_val:>12}")

    lines.append("")
    lines.append("=" * 68)
    lines.append("  BAND-ARB YES GATE COMPARISON: p75 vs p90")
    lines.append("=" * 68)
    lines.append(f"  {'':36}  {'p75':>12}  {'p90':>12}")
    lines.append("  " + "-" * 64)

    def fmt_n(n: int | float) -> str:
        return f"{int(n):,}" if isinstance(n, (int, float)) and not (n != n) else "—"

    def fmt_pct(v: float, denom: int) -> str:
        if denom == 0 or v != v:
            return "—"
        return f"{v:.1f}%  ({int(v/100*denom)})"

    row("Bands checked", fmt_n(p75s["bands_checked"]), fmt_n(p90s["bands_checked"]))
    row("In-band signals",
        f"{p75s['in_band']:,}  ({p75s['in_band']/max(p75s['bands_checked'],1)*100:.1f}%)",
        f"{p90s['in_band']:,}  ({p90s['in_band']/max(p90s['bands_checked'],1)*100:.1f}%)")
    row("  No edge (ask ≥95¢)", fmt_n(p75s["no_edge"]), fmt_n(p90s["no_edge"]), indent=1)
    row("  Data unavailable", fmt_n(p75s["unavailable"]), fmt_n(p90s["unavailable"]), indent=1)
    row("  Trades simulated", fmt_n(p75s["trades"]), fmt_n(p90s["trades"]), indent=1)
    row("    Wins", fmt_n(p75s["wins"]), fmt_n(p90s["wins"]), indent=2)
    row("    Losses", fmt_n(p75s["losses"]), fmt_n(p90s["losses"]), indent=2)

    def fmt_rate(s: dict) -> str:
        if s["trades"] == 0:
            return "—"
        return f"{s['win_rate']:.1f}%"

    def fmt_cents(v: float) -> str:
        return "—" if v != v else f"{v:.1f}¢"

    def fmt_dollars(v: float) -> str:
        return "—" if v != v else f"${v:+.2f}"

    lines.append("  " + "-" * 64)
    row("Win rate", fmt_rate(p75s), fmt_rate(p90s))
    row("Avg entry price", fmt_cents(p75s["avg_entry"]), fmt_cents(p90s["avg_entry"]))
    row("Avg P&L / trade", fmt_cents(p75s["avg_pnl"]), fmt_cents(p90s["avg_pnl"]))
    row("Total P&L ($1/trade)",
        fmt_dollars(p75s["total_pnl_dollars"]),
        fmt_dollars(p90s["total_pnl_dollars"]))

    # ── Signals gained / lost ────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 68)
    lines.append("  MOVING FROM p75 → p90")
    lines.append("=" * 68)

    # Group by ticker to compare gate outcomes
    by_ticker_p75: dict[str, dict] = {r["ticker"]: r for r in p75_recs}
    by_ticker_p90: dict[str, dict] = {r["ticker"]: r for r in p90_recs}
    all_tickers = set(by_ticker_p75) | set(by_ticker_p90)

    gained: list[dict] = []   # not in-band at p75, is trade at p90
    lost:   list[dict] = []   # is trade at p75, not in-band at p90

    for t in all_tickers:
        r75 = by_ticker_p75.get(t)
        r90 = by_ticker_p90.get(t)
        p75_is_trade = r75 is not None and r75["outcome"] in ("win", "loss")
        p90_is_trade = r90 is not None and r90["outcome"] in ("win", "loss")
        p75_in_band  = r75 is not None and r75["in_band"]

        if p90_is_trade and not p75_is_trade:
            gained.append(r90)
        if p75_is_trade and not p90_is_trade:
            lost.append(r75)

    def _mini_stats(recs: list[dict]) -> str:
        n = len(recs)
        if n == 0:
            return "none"
        wins = sum(1 for r in recs if r["outcome"] == "win")
        avg_ask = sum(r["yes_ask"] for r in recs) / n
        avg_pnl = sum(r["pnl"] for r in recs) / n
        return (f"{n} trades  win={wins/n*100:.0f}%  "
                f"avg_entry={avg_ask:.0f}¢  avg_pnl={avg_pnl:+.1f}¢")

    lines.append(f"  Signals GAINED by waiting (in-band at p90, not p75):")
    lines.append(f"    {_mini_stats(gained)}")
    lines.append(f"  Signals LOST by waiting (in-band at p75, not p90):")
    lines.append(f"    {_mini_stats(lost)}")

    # ── By city ──────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 68)
    lines.append("  BY CITY")
    lines.append("=" * 68)
    lines.append(f"  {'City':<20}  {'p75 trades':>10}  {'p75 win%':>8}  "
                 f"{'p75 P&L':>8}  {'p90 trades':>10}  {'p90 win%':>8}  {'p90 P&L':>8}")
    lines.append("  " + "-" * 82)

    all_metrics = sorted({r["metric"] for r in records})
    for metric in all_metrics:
        city_name = CITIES[metric][0] if metric in CITIES else metric
        s75 = _gate_stats([r for r in p75_recs if r["metric"] == metric])
        s90 = _gate_stats([r for r in p90_recs if r["metric"] == metric])
        lines.append(
            f"  {city_name:<20}  {s75['trades']:>10}  "
            f"{fmt_rate(s75):>8}  {fmt_dollars(s75['total_pnl_dollars']):>8}  "
            f"{s90['trades']:>10}  "
            f"{fmt_rate(s90):>8}  {fmt_dollars(s90['total_pnl_dollars']):>8}"
        )

    # ── By entry price bucket ─────────────────────────────────────────────────
    price_buckets = [(0, 20), (20, 35), (35, 50), (50, 70), (70, 95)]

    lines.append("")
    lines.append("=" * 68)
    lines.append("  BY ENTRY PRICE BUCKET")
    lines.append("=" * 68)
    lines.append(f"  {'Bucket':>10}  {'p75 N':>7}  {'p75 win%':>8}  {'p75 P&L':>8}"
                 f"  {'p90 N':>7}  {'p90 win%':>8}  {'p90 P&L':>8}")
    lines.append("  " + "-" * 68)

    for lo, hi in price_buckets:
        label = f"{lo}–{hi}¢"
        b75 = [r for r in p75_recs if r["outcome"] in ("win","loss")
               and r["yes_ask"] is not None and lo <= r["yes_ask"] < hi]
        b90 = [r for r in p90_recs if r["outcome"] in ("win","loss")
               and r["yes_ask"] is not None and lo <= r["yes_ask"] < hi]
        s75 = _gate_stats(b75 + [{"in_band": False}] * 0) if b75 else _gate_stats([])
        s90 = _gate_stats(b90 + [{"in_band": False}] * 0) if b90 else _gate_stats([])

        def _bucket_row(recs: list[dict]) -> tuple[str, str, str]:
            n = len(recs)
            if n == 0:
                return "—", "—", "—"
            wins = sum(1 for r in recs if r["outcome"] == "win")
            pnl  = sum(r["pnl"] for r in recs) / 100
            return str(n), f"{wins/n*100:.0f}%", f"${pnl:+.2f}"

        n75, wr75, pnl75 = _bucket_row(b75)
        n90, wr90, pnl90 = _bucket_row(b90)
        lines.append(f"  {label:>10}  {n75:>7}  {wr75:>8}  {pnl75:>8}"
                     f"  {n90:>7}  {wr90:>8}  {pnl90:>8}")

    # ── By month ─────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 68)
    lines.append("  BY MONTH")
    lines.append("=" * 68)
    lines.append(f"  {'Month':<8}  {'p75 trades':>10}  {'p75 win%':>8}  "
                 f"{'p75 P&L':>8}  {'p90 trades':>10}  {'p90 win%':>8}  {'p90 P&L':>8}")
    lines.append("  " + "-" * 74)

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    all_months = sorted({r["month"] for r in records})
    for m in all_months:
        s75 = _gate_stats([r for r in p75_recs if r["month"] == m])
        s90 = _gate_stats([r for r in p90_recs if r["month"] == m])
        lines.append(
            f"  {month_names.get(m, str(m)):<8}  {s75['trades']:>10}  "
            f"{fmt_rate(s75):>8}  {fmt_dollars(s75['total_pnl_dollars']):>8}  "
            f"{s90['trades']:>10}  "
            f"{fmt_rate(s90):>8}  {fmt_dollars(s90['total_pnl_dollars']):>8}"
        )

    lines.append("")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    mesonet     = load_mesonet()
    bands       = load_bands()
    p75_minutes, p90_minutes = load_gate_minutes()

    log.info("Mesonet rows: %d  |  Bands: %d", len(mesonet), len(bands))

    city_filter  = set(args.cities)  if args.cities  else None
    month_filter = set(args.months)  if args.months  else None

    records = await run_simulation(
        bands, mesonet, p75_minutes, p90_minutes,
        city_filter, month_filter, not args.no_cache,
    )

    log.info("Records generated: %d", len(records))

    report = build_report(records)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        log.info("Report written to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare p75 vs p90 entry gate for band-arb YES signals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cities", nargs="+", default=None,
                        help="City suffixes, e.g. dca chi bos (default: all)")
    parser.add_argument("--months", nargs="+", type=int, default=None,
                        help="Months to include, e.g. 4 5 (default: all)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore candlestick cache and re-fetch all tickers")
    parser.add_argument("--out", default=None,
                        help="Write report to this file in addition to stdout")
    args = parser.parse_args()
    asyncio.run(main(args))
