"""
Audit trades where weather models disagreed significantly at entry.

Compares HRRR, nws_hourly, open_meteo, noaa, metar, noaa_observed forecasts
logged in the opportunities table near each trade entry.  Reports the
spread between the min and max model forecast, and whether the trade won.

Usage:
    venv/bin/python scripts/audit_model_disagreement.py
    venv/bin/python scripts/audit_model_disagreement.py --min-spread 5
    venv/bin/python scripts/audit_model_disagreement.py --min-spread 3 --kinds forecast_no numeric
    venv/bin/python scripts/audit_model_disagreement.py --summary-only
"""

import argparse
import sqlite3
from collections import defaultdict

DB_PATH = "opportunity_log.db"

WEATHER_MODELS = {"hrrr", "nws_hourly", "open_meteo", "noaa", "owm", "metar", "noaa_observed"}

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--min-spread", type=float, default=5.0,
                    help="Minimum model spread (°F) to report (default: 5)")
parser.add_argument("--kinds", nargs="+", default=["forecast_no", "numeric"],
                    help="opportunity_kind values to include (default: forecast_no numeric)")
parser.add_argument("--summary-only", action="store_true",
                    help="Skip per-trade detail, show aggregate tables only")
parser.add_argument("--db", default=DB_PATH, help=f"Path to SQLite DB (default: {DB_PATH})")
args = parser.parse_args()

conn = sqlite3.connect(args.db)

# ── Load resolved trades ─────────────────────────────────────────────────────

placeholders = ",".join("?" * len(args.kinds))
trades = conn.execute(f"""
    SELECT id, ticker, side, opportunity_kind, exit_pnl_cents, logged_at,
           source, corroborating_sources
    FROM trades
    WHERE exited_at IS NOT NULL
      AND exit_pnl_cents IS NOT NULL
      AND opportunity_kind IN ({placeholders})
    ORDER BY id
""", args.kinds).fetchall()

# ── For each trade, find model forecasts from opportunities table ─────────────

# Get best (highest-edge) data_value per source per ticker
opp_rows = conn.execute("""
    SELECT ticker, source, data_value, edge, direction
    FROM opportunities
    WHERE source IN ({})
""".format(",".join(f'"{s}"' for s in WEATHER_MODELS))).fetchall()

# Build: ticker → {source → (data_value, edge)}
from collections import defaultdict
ticker_models: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
for ticker, src, val, edge, direction in opp_rows:
    if val is None:
        continue
    existing = ticker_models[ticker].get(src)
    if existing is None or abs(edge or 0) > abs(existing[1]):
        ticker_models[ticker][src] = (float(val), float(edge or 0))

conn.close()

# ── Analyse ──────────────────────────────────────────────────────────────────

results = []

for row in trades:
    tid, ticker, side, kind, pnl, entry_time, src, corr = row
    won = pnl > 0

    model_vals = ticker_models.get(ticker, {})
    if len(model_vals) < 2:
        continue  # need at least 2 models to compute spread

    vals = {m: v for m, (v, e) in model_vals.items()}
    lo_src = min(vals, key=vals.get)
    hi_src = max(vals, key=vals.get)
    spread = vals[hi_src] - vals[lo_src]

    if spread < args.min_spread:
        continue

    results.append({
        "id": tid,
        "ticker": ticker,
        "kind": kind,
        "pnl": pnl,
        "won": won,
        "spread": spread,
        "lo_src": lo_src,
        "lo_val": vals[lo_src],
        "hi_src": hi_src,
        "hi_val": vals[hi_src],
        "models": vals,
    })

# ── Per-trade detail ──────────────────────────────────────────────────────────

if not args.summary_only:
    print(f"=== TRADES WITH MODEL SPREAD >= {args.min_spread}°F ===\n")
    print(f"{'Result':<6} {'ID':>4}  {'Ticker':<42} {'Spread':>7}  {'Low model → High model'}")
    print("-" * 95)
    for r in sorted(results, key=lambda x: -x["spread"]):
        flag = "WIN " if r["won"] else "LOSS"
        model_str = "  ".join(
            f"{m}={v:.1f}F" for m, v in sorted(r["models"].items(), key=lambda x: x[1])
        )
        print(f"{flag}  #{r['id']:>3}  {r['ticker']:<42} {r['spread']:>5.1f}°F  {model_str}")
    print()

# ── Summary tables ────────────────────────────────────────────────────────────

total = len(results)
wins = [r for r in results if r["won"]]
losses = [r for r in results if not r["won"]]
all_trades_in_kinds = [t for t in trades]
all_wins = [t for t in all_trades_in_kinds if t[4] > 0]

print(f"=== SUMMARY (min spread = {args.min_spread}°F) ===")
print(f"  Trades with spread >= {args.min_spread}°F : {total} of {len(trades)} total")
print(f"  Among high-disagreement trades  : {len(wins)} wins / {len(losses)} losses  "
      f"({len(wins)/total*100:.0f}% win rate)" if total else "  No trades meet criteria")
print(f"  Overall win rate (all kinds)    : {len(all_wins)/len(all_trades_in_kinds)*100:.0f}%"
      f" ({len(all_wins)}/{len(all_trades_in_kinds)})")
print()

if total:
    avg_spread_win  = sum(r["spread"] for r in wins)  / len(wins)  if wins   else 0
    avg_spread_loss = sum(r["spread"] for r in losses) / len(losses) if losses else 0
    print(f"  Avg spread on WINS  : {avg_spread_win:.1f}°F")
    print(f"  Avg spread on LOSSES: {avg_spread_loss:.1f}°F")
    print()

    # Win rate by spread bucket
    buckets = [(5, 10), (10, 15), (15, 25), (25, 999)]
    print(f"  {'Spread bucket':<18} {'Wins':>5} {'Losses':>7} {'Win rate':>9}")
    print(f"  {'-'*45}")
    for lo, hi in buckets:
        bw = [r for r in results if lo <= r["spread"] < hi and r["won"]]
        bl = [r for r in results if lo <= r["spread"] < hi and not r["won"]]
        bt = len(bw) + len(bl)
        if bt == 0:
            continue
        label = f"{lo}-{hi if hi < 999 else '∞'}°F"
        print(f"  {label:<18} {len(bw):>5} {len(bl):>7} {len(bw)/bt*100:>8.0f}%")
    print()

    # Which model pairs disagree most on losses vs wins
    from itertools import combinations
    pair_wins  = defaultdict(int)
    pair_losses = defaultdict(int)
    pair_spread_wins  = defaultdict(list)
    pair_spread_losses = defaultdict(list)

    for r in results:
        models = list(r["models"].keys())
        for a, b in combinations(sorted(models), 2):
            diff = abs(r["models"][a] - r["models"][b])
            if diff < args.min_spread:
                continue
            key = f"{a} vs {b}"
            if r["won"]:
                pair_wins[key] += 1
                pair_spread_wins[key].append(diff)
            else:
                pair_losses[key] += 1
                pair_spread_losses[key].append(diff)

    all_pairs = set(list(pair_wins.keys()) + list(pair_losses.keys()))
    if all_pairs:
        print(f"  {'Model pair':<30} {'Wins':>5} {'Losses':>7} {'Win rate':>9}")
        print(f"  {'-'*55}")
        for p in sorted(all_pairs, key=lambda x: -(pair_wins[x]+pair_losses[x])):
            w = pair_wins[p]
            l = pair_losses[p]
            t = w + l
            print(f"  {p:<30} {w:>5} {l:>7} {w/t*100:>8.0f}%")
