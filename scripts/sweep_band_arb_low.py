"""
Parameter sweep for band_arb_low (warm-NO) trades.

Sweeps three gates against settled shadow + real KXLOWT history:

  min_ask       Minimum NO ask (¢) — below this, market disagrees too strongly
  max_ask       Maximum NO ask (¢) — above this, profit margin too thin
  max_local_hr  Latest local hour allowed — blocks evening signals before
                overnight low is established

For each combination, reports: n, WR, total PnL, avg PnL/trade.
Also shows per-city breakdown for the best config.

Data sources:
  shadow_band_arb  — paper trades for non-whitelisted cities (pnl_cents = settlement PnL)
  trades           — real/dry-run trades for all KXLOWT markets (uses exit_pnl_cents
                     where available; falls back to settlement PnL from outcome+limit_price)
"""

import sqlite3
import re
from datetime import datetime, timedelta
from itertools import product

DB = "data/db/opportunity_log.db"

# City → UTC offset (standard; CDT/PDT etc. in May = standard - 1h... actually May = DST)
# May is DST for US: EDT=UTC-4, CDT=UTC-5, MDT=UTC-6, PDT=UTC-7
CITY_UTC_OFFSET = {
    "LAX": -7, "LAS": -7, "LV": -7,
    "SFO": -7, "SEA": -7,
    "PHX": -7,  # Arizona no DST
    "DEN": -6,
    "CHI": -5, "MIN": -5, "HOU": -5, "DAL": -5, "DFW": -5,
    "OKC": -5, "SATX": -5, "SAT": -5,
    "ATL": -4, "MIA": -4, "NYC": -4, "NY": -4,
    "BOS": -4, "PHIL": -4, "PHL": -4, "DC": -4, "DCA": -4,
    "NOLA": -5, "MSY": -5,
    "AUS": -5,
}

def extract_city(ticker: str) -> str:
    # KXLOWT<CITY>-26MAY...  or  KXLOWTLAX-...
    m = re.match(r"KXLOWT([A-Z]+)-", ticker)
    return m.group(1) if m else ""

def local_hour(logged_at: str, city: str) -> int | None:
    try:
        dt = datetime.fromisoformat(logged_at.replace("Z", "+00:00"))
        offset = CITY_UTC_OFFSET.get(city)
        if offset is None:
            return None
        local_dt = dt + timedelta(hours=offset)
        return local_dt.hour
    except Exception:
        return None

def load_shadow(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT ticker, limit_price, margin_f, outcome, pnl_cents, logged_at, contracts
        FROM shadow_band_arb WHERE outcome IS NOT NULL
    """).fetchall()
    trades = []
    for ticker, ask, margin, outcome, pnl, logged_at, contracts in rows:
        city = extract_city(ticker)
        lh = local_hour(logged_at, city)
        trades.append({
            "source": "shadow",
            "ticker": ticker,
            "city": city,
            "ask": ask,
            "margin": margin,
            "outcome": outcome,
            "pnl": pnl,
            "contracts": contracts or 10,
            "local_hour": lh,
        })
    return trades

def load_real(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT ticker, limit_price, outcome, exit_pnl_cents, logged_at, count
        FROM trades
        WHERE source='band_arb' AND outcome IS NOT NULL AND side='no' AND ticker LIKE 'KXLOWT%'
    """).fetchall()
    trades = []
    for ticker, ask, outcome, exit_pnl, logged_at, contracts in rows:
        city = extract_city(ticker)
        lh = local_hour(logged_at, city)
        # Use exit_pnl if available; else compute settlement PnL
        if exit_pnl is not None:
            pnl = exit_pnl
        elif outcome == "won":
            pnl = (100 - ask) * (contracts or 1)
        else:
            pnl = -ask * (contracts or 1)
        trades.append({
            "source": "real",
            "ticker": ticker,
            "city": city,
            "ask": ask,
            "margin": None,
            "outcome": outcome,
            "pnl": pnl,
            "contracts": contracts or 1,
            "local_hour": lh,
        })
    return trades

def evaluate(trades, min_ask, max_ask, max_local_hr, city_set=None):
    selected = []
    for t in trades:
        if t["ask"] < min_ask or t["ask"] > max_ask:
            continue
        if t["local_hour"] is not None and t["local_hour"] > max_local_hr:
            continue
        if city_set is not None and t["city"] not in city_set:
            continue
        selected.append(t)
    if not selected:
        return None
    n = len(selected)
    wins = sum(1 for t in selected if t["outcome"] == "won")
    total_pnl = sum(t["pnl"] for t in selected)
    return {
        "n": n,
        "wins": wins,
        "wr": 100 * wins / n,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / n,
    }

def fmt(r):
    if r is None:
        return "n/a"
    return (f"n={r['n']:3d}  WR={r['wr']:5.1f}%  "
            f"total={r['total_pnl']:+8.0f}¢  avg={r['avg_pnl']:+7.1f}¢")

def evaluate2(trades, min_ask, max_ask, min_local_hr, max_local_hr, city_set=None):
    selected = []
    for t in trades:
        if t["ask"] < min_ask or t["ask"] > max_ask:
            continue
        lh = t["local_hour"]
        if lh is not None:
            if lh < min_local_hr or lh > max_local_hr:
                continue
        if city_set is not None and t["city"] not in city_set:
            continue
        selected.append(t)
    if not selected:
        return None
    n = len(selected)
    wins = sum(1 for t in selected if t["outcome"] == "won")
    total_pnl = sum(t["pnl"] for t in selected)
    return {
        "n": n,
        "wins": wins,
        "wr": 100 * wins / n,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / n,
    }

def main():
    conn = sqlite3.connect(DB)
    shadow = load_shadow(conn)
    real   = load_real(conn)
    all_trades = shadow + real

    # attach local hour to each for display
    shadow_with_hr = [t for t in shadow if t["local_hour"] is not None]
    real_with_hr   = [t for t in real   if t["local_hour"] is not None]
    all_with_hr    = shadow_with_hr + real_with_hr

    print(f"Loaded: {len(shadow)} shadow, {len(real)} real ({len(all_trades)} total)")
    print(f"  With resolved local hour: {len(all_with_hr)}\n")

    # ── 1. Raw hour distribution ─────────────────────────────────────────
    print("── 1. Trade distribution by local entry hour (all ask ranges) ───────────────")
    print(f"  {'Hr':>4} │ {'n':>4}  {'WR':>6}  {'avg PnL':>9}  losses")
    print("  " + "─" * 65)
    from collections import defaultdict
    by_hour: dict[int, list] = defaultdict(list)
    for t in all_with_hr:
        by_hour[t["local_hour"]].append(t)
    for hr in sorted(by_hour):
        trades_hr = by_hour[hr]
        n = len(trades_hr)
        wins = sum(1 for t in trades_hr if t["outcome"] == "won")
        wr = 100 * wins / n
        avg = sum(t["pnl"] for t in trades_hr) / n
        losses = [t["ticker"] for t in trades_hr if t["outcome"] == "lost"]
        loss_str = ", ".join(t.split("-")[0][6:] + t.split("-")[2] for t in losses) if losses else ""
        print(f"  {hr:>4} │ {n:>4}  {wr:>5.1f}%  {avg:>+9.1f}¢  {loss_str}")

    # ── 2. Cumulative effect of min_local_hr gate (all asks) ─────────────
    print("\n── 2. Min local hour gate (no ask filter, maxHr=23) ─────────────────────────")
    print(f"  {'minHr':>5} │ {'n':>4}  {'WR':>6}  {'total':>9}  {'avg':>8}  trades_dropped")
    print("  " + "─" * 65)
    base_n = len(all_with_hr)
    for min_hr in [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        r = evaluate2(all_with_hr, 0, 999, min_hr, 23)
        if r:
            dropped = base_n - r["n"]
            print(f"  {min_hr:>5} │ {r['n']:>4}  {r['wr']:>5.1f}%  "
                  f"{r['total_pnl']:>+9.0f}¢  {r['avg_pnl']:>+8.1f}¢  -{dropped}")

    # ── 3. Cumulative effect of max_local_hr gate (all asks) ─────────────
    print("\n── 3. Max local hour gate (no ask filter, minHr=0) ──────────────────────────")
    print(f"  {'maxHr':>5} │ {'n':>4}  {'WR':>6}  {'total':>9}  {'avg':>8}  trades_dropped")
    print("  " + "─" * 65)
    for max_hr in [23, 22, 21, 20, 19, 18, 17, 16, 15, 14]:
        r = evaluate2(all_with_hr, 0, 999, 0, max_hr)
        if r:
            dropped = base_n - r["n"]
            print(f"  {max_hr:>5} │ {r['n']:>4}  {r['wr']:>5.1f}%  "
                  f"{r['total_pnl']:>+9.0f}¢  {r['avg_pnl']:>+8.1f}¢  -{dropped}")

    # ── 4. Hour window (min+max) sweep — all asks ─────────────────────────
    print("\n── 4. Hour window sweep (no ask filter, min n=5) ────────────────────────────")
    print(f"  {'minHr':>5} {'maxHr':>5} │ {'n':>4}  {'WR':>6}  {'total':>9}  {'avg':>8}")
    print("  " + "─" * 55)
    hr_results = []
    for min_hr in range(0, 17):
        for max_hr in range(min_hr + 3, 24):
            r = evaluate2(all_with_hr, 0, 999, min_hr, max_hr)
            if r and r["n"] >= 5:
                hr_results.append((min_hr, max_hr, r))
    hr_results.sort(key=lambda x: x[2]["avg_pnl"], reverse=True)
    for min_hr, max_hr, r in hr_results[:15]:
        print(f"  {min_hr:>5}  {max_hr:>5} │ {r['n']:>4}  {r['wr']:>5.1f}%  "
              f"{r['total_pnl']:>+9.0f}¢  {r['avg_pnl']:>+8.1f}¢")

    # ── 5. Full 3-way sweep: min_ask × max_ask × hour window ─────────────
    print("\n── 5. Full sweep: ask range + hour window (min n=5) ─────────────────────────")
    print(f"  {'minA':>4} {'maxA':>4} {'minH':>4} {'maxH':>4} │ {'n':>4}  {'WR':>6}  {'total':>9}  {'avg':>8}")
    print("  " + "─" * 65)
    full_results = []
    for min_a, max_a, min_hr, max_hr in product(
        [0, 40, 50, 60],
        [70, 75, 80, 85, 90],
        [0, 6, 8, 9, 10, 12, 14],
        [16, 17, 18, 19, 20, 21, 23],
    ):
        if min_a >= max_a or min_hr >= max_hr:
            continue
        r = evaluate2(all_with_hr, min_a, max_a, min_hr, max_hr)
        if r and r["n"] >= 5:
            full_results.append((min_a, max_a, min_hr, max_hr, r))
    full_results.sort(key=lambda x: x[4]["avg_pnl"], reverse=True)
    for min_a, max_a, min_hr, max_hr, r in full_results[:20]:
        print(f"  {min_a:>4}  {max_a:>4}  {min_hr:>4}  {max_hr:>4} │ "
              f"{r['n']:>4}  {r['wr']:>5.1f}%  {r['total_pnl']:>+9.0f}¢  {r['avg_pnl']:>+8.1f}¢")

    # ── 6. Within best ask range (60-80), show hour distribution ─────────
    print("\n── 6. Hour distribution within best ask range (60–80¢) ─────────────────────")
    print(f"  {'Hr':>4} │ {'n':>4}  {'WR':>6}  {'avg PnL':>9}  tickers")
    print("  " + "─" * 70)
    by_hour_filtered: dict[int, list] = defaultdict(list)
    for t in all_with_hr:
        if 60 <= t["ask"] <= 80:
            by_hour_filtered[t["local_hour"]].append(t)
    for hr in sorted(by_hour_filtered):
        trades_hr = by_hour_filtered[hr]
        n = len(trades_hr)
        wins = sum(1 for t in trades_hr if t["outcome"] == "won")
        wr = 100 * wins / n
        avg = sum(t["pnl"] for t in trades_hr) / n
        tickers = [t["ticker"].split("-")[0][6:] + " " + t["ticker"].split("-")[2]
                   for t in trades_hr]
        print(f"  {hr:>4} │ {n:>4}  {wr:>5.1f}%  {avg:>+9.1f}¢  {', '.join(tickers)}")

    # ── 7. Source / corroboration ─────────────────────────────────────────
    print("\n── 7. Source / corroboration ────────────────────────────────────────────────")
    shadow_corr = defaultdict(list)
    for t in shadow:
        shadow_corr[t.get("corr_status") or "(empty)"].append(t)
    real_corr = defaultdict(list)
    for t in real:
        real_corr[t.get("corr_status") or "(empty)"].append(t)
    print("  Shadow corr_status:")
    for k, trades_c in sorted(shadow_corr.items()):
        n = len(trades_c); wins = sum(1 for t in trades_c if t["outcome"]=="won")
        print(f"    {k!r:35s} n={n}  WR={100*wins/n:.0f}%")
    print("  Real corroborating_sources: all 'metar' (no variation — source not a useful gate)")

    # ── 8. Edge (margin_f) analysis ───────────────────────────────────────
    # margin_f = observed_running_min - band_ceil (shadow only; real trades lack this field)
    shadow_with_margin = [t for t in shadow if t.get("margin") is not None]
    print(f"\n── 8. Edge (margin_f) — shadow only, n={len(shadow_with_margin)} ──────────────────────")

    # 8a. Raw margin buckets
    print("\n  8a. Raw margin buckets (no other filters):")
    print(f"  {'Margin':>12} │ {'n':>4}  {'WR':>6}  {'avg PnL':>9}  losses")
    print("  " + "─" * 65)
    buckets = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,99)]
    for lo, hi in buckets:
        grp = [t for t in shadow_with_margin if lo <= t["margin"] < hi]
        if not grp: continue
        n = len(grp); wins = sum(1 for t in grp if t["outcome"]=="won")
        avg = sum(t["pnl"] for t in grp) / n
        losses = [t["ticker"].split("-")[0][6:]+t["ticker"].split("-")[2]
                  for t in grp if t["outcome"]=="lost"]
        label = f"{lo}–{hi}°F" if hi < 99 else f">={lo}°F"
        print(f"  {label:>12} │ {n:>4}  {100*wins/n:>5.1f}%  {avg:>+9.1f}¢  {', '.join(losses)}")

    # 8b. Min margin gate sweep
    print("\n  8b. Min margin gate (no other filters):")
    print(f"  {'minEdge':>7} │ {'n':>4}  {'WR':>6}  {'total':>9}  {'avg':>8}  dropped")
    print("  " + "─" * 55)
    base_m = len(shadow_with_margin)
    for min_m in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        grp = [t for t in shadow_with_margin if t["margin"] >= min_m]
        if not grp: continue
        n = len(grp); wins = sum(1 for t in grp if t["outcome"]=="won")
        total = sum(t["pnl"] for t in grp)
        print(f"  {min_m:>7.1f} │ {n:>4}  {100*wins/n:>5.1f}%  "
              f"{total:>+9.0f}¢  {total/n:>+8.1f}¢  -{base_m-n}")

    # 8c. Margin + hour window
    print(f"\n  8c. Margin + best hour window (minHr=12, maxHr=15):")
    print(f"  {'minEdge':>7} │ {'n':>4}  {'WR':>6}  {'total':>9}  {'avg':>8}")
    print("  " + "─" * 50)
    for min_m in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        grp = [t for t in shadow_with_margin
               if t["margin"] >= min_m
               and t["local_hour"] is not None
               and 12 <= t["local_hour"] <= 15]
        if not grp: continue
        n = len(grp); wins = sum(1 for t in grp if t["outcome"]=="won")
        total = sum(t["pnl"] for t in grp)
        print(f"  {min_m:>7.1f} │ {n:>4}  {100*wins/n:>5.1f}%  "
              f"{total:>+9.0f}¢  {total/n:>+8.1f}¢")

    # 8d. Margin × hour heatmap: WR% — isolates causation
    # Rows = margin bucket, Cols = hour bucket.
    # If edge matters independently, WR should improve going down each column.
    # If hour matters independently, WR should improve going right across each row.
    hr_bins   = [(9,9),(10,11),(12,13),(14,15),(16,23)]
    mg_bins   = [(0,2),(2,4),(4,6),(6,99)]
    hr_labels = ["hr=9 ","10-11","12-13","14-15","16+  "]
    mg_labels = ["0-2°F","2-4°F","4-6°F","6°F+ "]

    def cell(trades_in, mlo, mhi, hlo, hhi):
        grp = [t for t in trades_in
               if mlo <= t["margin"] < mhi
               and t["local_hour"] is not None
               and hlo <= t["local_hour"] <= hhi]
        if not grp: return "    -    "
        n = len(grp); wins = sum(1 for t in grp if t["outcome"]=="won")
        avg = sum(t["pnl"] for t in grp)/n
        return f"{100*wins/n:3.0f}%({n}) {avg:+5.0f}¢"

    print(f"\n  8d. Margin × Hour heatmap  [WR%(n) avg¢]  (shadow only)")
    print(f"  {'':7} " + "  ".join(f"{l:>17}" for l in hr_labels))
    for (mlo,mhi), ml in zip(mg_bins, mg_labels):
        row = [cell(shadow_with_margin, mlo, mhi, hlo, hhi)
               for hlo, hhi in hr_bins]
        print(f"  {ml}  " + "  ".join(f"{c:>17}" for c in row))

    # 8e. Controlling for hour: does edge matter within each hour bucket?
    print(f"\n  8e. Within hour=9 only — edge sweep (causation check: is edge real or just proxy for time?)")
    print(f"  {'minEdge':>7} │ {'n':>4}  {'WR':>6}  {'avg':>8}")
    print("  " + "─" * 40)
    hr9 = [t for t in shadow_with_margin if t["local_hour"] == 9]
    for min_m in [0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        grp = [t for t in hr9 if t["margin"] >= min_m]
        if not grp: continue
        n = len(grp); wins = sum(1 for t in grp if t["outcome"]=="won")
        print(f"  {min_m:>7.1f} │ {n:>4}  {100*wins/n:>5.1f}%  "
              f"{sum(t['pnl'] for t in grp)/n:>+8.1f}¢")

    print(f"\n  8e. Within hour=12–15 only — edge sweep")
    print(f"  {'minEdge':>7} │ {'n':>4}  {'WR':>6}  {'avg':>8}")
    print("  " + "─" * 40)
    hr_noon = [t for t in shadow_with_margin
               if t["local_hour"] is not None and 12 <= t["local_hour"] <= 15]
    for min_m in [0.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        grp = [t for t in hr_noon if t["margin"] >= min_m]
        if not grp: continue
        n = len(grp); wins = sum(1 for t in grp if t["outcome"]=="won")
        print(f"  {min_m:>7.1f} │ {n:>4}  {100*wins/n:>5.1f}%  "
              f"{sum(t['pnl'] for t in grp)/n:>+8.1f}¢")

    # 8f. Controlling for edge: does hour matter within each edge bucket?
    print(f"\n  8f. Within edge 0-2°F only — hour sweep (controlling for edge)")
    print(f"  {'minHr':>5} │ {'n':>4}  {'WR':>6}  {'avg':>8}")
    print("  " + "─" * 40)
    low_edge = [t for t in shadow_with_margin if t["margin"] < 2.0]
    for min_hr in [0, 9, 10, 11, 12, 13, 14]:
        grp = [t for t in low_edge
               if t["local_hour"] is not None and t["local_hour"] >= min_hr]
        if not grp: continue
        n = len(grp); wins = sum(1 for t in grp if t["outcome"]=="won")
        print(f"  {min_hr:>5} │ {n:>4}  {100*wins/n:>5.1f}%  "
              f"{sum(t['pnl'] for t in grp)/n:>+8.1f}¢")

    print(f"\n  8f. Within edge >=3°F only — hour sweep")
    print(f"  {'minHr':>5} │ {'n':>4}  {'WR':>6}  {'avg':>8}")
    print("  " + "─" * 40)
    high_edge = [t for t in shadow_with_margin if t["margin"] >= 3.0]
    for min_hr in [0, 9, 10, 11, 12, 13, 14]:
        grp = [t for t in high_edge
               if t["local_hour"] is not None and t["local_hour"] >= min_hr]
        if not grp: continue
        n = len(grp); wins = sum(1 for t in grp if t["outcome"]=="won")
        print(f"  {min_hr:>5} │ {n:>4}  {100*wins/n:>5.1f}%  "
              f"{sum(t['pnl'] for t in grp)/n:>+8.1f}¢")

    conn.close()

if __name__ == "__main__":
    main()
