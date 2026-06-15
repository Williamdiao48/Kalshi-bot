"""
Compare IEM ASOS daily max (what we use for entry decisions) vs Kalshi settlement.

Loads the continued_heat_temp_cache.json records that have a matching band,
fetches IEM daily summaries (max_tmpf) for the same stations, and checks:
  1. Does IEM daily max agree with IEM hourly running max? (same station, should be ≈)
  2. Does IEM daily max predict Kalshi settlement correctly?
  3. For losses: was IEM daily max INSIDE the band while Kalshi settled "no"?
     → If yes, suggests IEM ≠ Kalshi settlement source (measurement divergence)

Run:
  venv/bin/python scripts/check_iem_vs_settlement.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
TEMP_CACHE = DATA_DIR / "continued_heat_temp_cache.json"
BANDS_CSV  = DATA_DIR / "kxhigh_bands.csv"

# IEM stations (metric → (station_id, network)) — from build_forecast_calibration.py
IEM_STATIONS: dict[str, tuple[str, str]] = {
    "temp_high_lax": ("LAX", "CA_ASOS"),
    "temp_high_den": ("DEN", "CO_ASOS"),
    "temp_high_chi": ("MDW", "IL_ASOS"),
    "temp_high_ny":  ("NYC", "NY_ASOS"),
    "temp_high_mia": ("MIA", "FL_ASOS"),
    "temp_high_aus": ("AUS", "TX_ASOS"),
    "temp_high_dal": ("DAL", "TX_ASOS"),
    "temp_high_bos": ("BOS", "MA_ASOS"),
    "temp_high_hou": ("HOU", "TX_ASOS"),
    "temp_high_dfw": ("DFW", "TX_ASOS"),
    "temp_high_sfo": ("SFO", "CA_ASOS"),
    "temp_high_sea": ("SEA", "WA_ASOS"),
    "temp_high_phx": ("PHX", "AZ_ASOS"),
    "temp_high_phl": ("PHL", "PA_ASOS"),
    "temp_high_atl": ("ATL", "GA_ASOS"),
    "temp_high_msp": ("MSP", "MN_ASOS"),
    "temp_high_dca": ("DCA", "DC_ASOS"),
    "temp_high_las": ("LAS", "NV_ASOS"),
    "temp_high_okc": ("OKC", "OK_ASOS"),
    "temp_high_sat": ("SAT", "TX_ASOS"),
    "temp_high_msy": ("MSY", "LA_ASOS"),
}


async def fetch_iem_daily(session: aiohttp.ClientSession, station: str, network: str, year: int) -> dict[str, float]:
    """Fetch IEM daily max_tmpf for a station/year. Returns {date_str: max_tmpf}."""
    url = f"https://mesonet.agron.iastate.edu/api/1/daily.json?station={station}&network={network}&year={year}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            result = {}
            for row in data.get("data", []):
                d = row.get("date")
                v = row.get("max_tmpf")
                if d and v is not None:
                    try:
                        result[d] = float(v)
                    except (TypeError, ValueError):
                        pass
            return result
    except Exception as e:
        print(f"  [IEM daily] {station}/{year}: {e}", file=sys.stderr)
        return {}


def load_bands() -> dict[tuple[str, str], list[dict]]:
    """Load kxhigh_bands.csv. Band date column is UTC close date; local_date = utc_close - 1 day."""
    from datetime import date, timedelta
    result: dict[tuple[str, str], list[dict]] = defaultdict(list)
    if not BANDS_CSV.exists():
        print("ERROR: kxhigh_bands.csv not found")
        return result
    with BANDS_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["direction"] != "between":
                continue
            utc_date = row["date"]
            try:
                local_date = str(date.fromisoformat(utc_date) - timedelta(days=1))
            except ValueError:
                continue
            key = (row["metric"], local_date)
            result[key].append({
                "ticker":    row["ticker"],
                "strike_lo": float(row["strike_lo"]),
                "strike_hi": float(row["strike_hi"]),
                "result":    row["result"],
                "utc_date":  utc_date,
            })
    return result


async def main() -> None:
    print("Loading temp cache…")
    with TEMP_CACHE.open() as f:
        records: list[dict] = json.load(f)
    print(f"  {len(records):,} temperature records loaded")

    bands_by_key = load_bands()
    total_bands = sum(len(v) for v in bands_by_key.values())
    print(f"  {total_bands:,} band rows loaded from kxhigh_bands.csv\n")

    # Find banded records: (metric, date) with a band where strike_lo <= actual_high <= strike_hi
    # We'll use actual_high as a proxy for what the final IEM daily max would be
    banded: list[dict] = []
    for rec in records:
        metric = rec["metric"]
        date_str = rec["date"]
        actual_high = rec.get("actual_high")
        if actual_high is None:
            continue
        bands = bands_by_key.get((metric, date_str), [])
        for b in bands:
            if b["strike_lo"] <= actual_high <= b["strike_hi"]:
                banded.append({
                    "metric":      metric,
                    "date":        date_str,
                    "actual_high": actual_high,          # IEM hourly running max
                    "asos_at_p75": rec.get("asos_at_p75"),
                    "overshoot":   rec.get("overshoot"),
                    "strike_lo":   b["strike_lo"],
                    "strike_hi":   b["strike_hi"],
                    "result":      b["result"],
                    "ticker":      b["ticker"],
                })
                break  # at most one matching band per record

    print(f"Banded records (actual_high inside band): {len(banded):,}")

    # Also find banded records where actual_high is NOT inside but band exists (for comparison)
    banded_above: list[dict] = []
    banded_below: list[dict] = []
    for rec in records:
        metric = rec["metric"]
        date_str = rec["date"]
        actual_high = rec.get("actual_high")
        asos_at_p75 = rec.get("asos_at_p75")
        if actual_high is None or asos_at_p75 is None:
            continue
        bands = bands_by_key.get((metric, date_str), [])
        # Find band that asos_at_p75 fell in (entry band)
        for b in bands:
            if b["strike_lo"] <= asos_at_p75 <= b["strike_hi"]:
                entry = {
                    "metric":      metric,
                    "date":        date_str,
                    "actual_high": actual_high,
                    "asos_at_p75": asos_at_p75,
                    "strike_lo":   b["strike_lo"],
                    "strike_hi":   b["strike_hi"],
                    "result":      b["result"],
                }
                if actual_high > b["strike_hi"]:
                    banded_above.append(entry)
                elif actual_high < b["strike_lo"]:
                    banded_below.append(entry)
                break

    print(f"  (banded entry but actual_high ABOVE band ceiling): {len(banded_above):,}")
    print(f"  (banded entry but actual_high BELOW band floor):   {len(banded_below):,}\n")

    # Collect unique (metric, year) pairs to fetch IEM daily summaries
    needed: dict[str, set[int]] = defaultdict(set)
    for rec in banded + banded_above:
        year = int(rec["date"][:4])
        needed[rec["metric"]].add(year)

    print("Fetching IEM daily summaries (max_tmpf)…")
    iem_daily: dict[tuple[str, str], float] = {}  # (metric, date_str) → max_tmpf

    async with aiohttp.ClientSession() as session:
        for metric, years in sorted(needed.items()):
            station, network = IEM_STATIONS.get(metric, (None, None))
            if station is None:
                continue
            for year in sorted(years):
                data = await fetch_iem_daily(session, station, network, year)
                for d, v in data.items():
                    iem_daily[(metric, d)] = v
            print(f"  {metric:<20} {station}  {sorted(years)}")

    print(f"\n  Total IEM daily records fetched: {len(iem_daily):,}\n")

    # ──────────────────────────────────────────────
    # Analysis 1: IEM hourly max vs IEM daily max
    # ──────────────────────────────────────────────
    print("=" * 70)
    print("  1. IEM HOURLY RUNNING MAX vs IEM DAILY MAX  (same station, banded records)")
    print("     Checks: is our actual_high a reliable proxy for the full-day max?")
    print("=" * 70)

    diffs_all: list[float] = []
    for rec in banded + banded_above + banded_below:
        key = (rec["metric"], rec["date"])
        iem_max = iem_daily.get(key)
        if iem_max is None:
            continue
        diffs_all.append(iem_max - rec["actual_high"])

    if diffs_all:
        n = len(diffs_all)
        mean_d = sum(diffs_all) / n
        max_d  = max(diffs_all)
        min_d  = min(diffs_all)
        buckets = {0: 0, 0.5: 0, 1.0: 0, 1.5: 0, 2.0: 0}
        for d in diffs_all:
            for thresh in [0, 0.5, 1.0, 1.5, 2.0]:
                if abs(d) > thresh:
                    buckets[thresh] += 1
        print(f"  n={n}  mean_diff={mean_d:+.3f}°F  range=[{min_d:+.1f}, {max_d:+.1f}]°F")
        print(f"  |diff| > 0.0°F:  {buckets[0]:>4} ({100*buckets[0]/n:.1f}%)")
        print(f"  |diff| > 0.5°F:  {buckets[0.5]:>4} ({100*buckets[0.5]/n:.1f}%)")
        print(f"  |diff| > 1.0°F:  {buckets[1.0]:>4} ({100*buckets[1.0]/n:.1f}%)")
        print(f"  |diff| > 1.5°F:  {buckets[1.5]:>4} ({100*buckets[1.5]/n:.1f}%)")
        print(f"  |diff| > 2.0°F:  {buckets[2.0]:>4} ({100*buckets[2.0]/n:.1f}%)")

    # ──────────────────────────────────────────────
    # Analysis 2: Does IEM daily max predict Kalshi settlement?
    # ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  2. IEM DAILY MAX vs KALSHI SETTLEMENT  (banded records only)")
    print("     Checks: does IEM max_tmpf == what Kalshi settles on?")
    print("=" * 70)

    n_agree = n_disagree = n_no_iem = 0
    disagree_examples: list[dict] = []

    for rec in banded:
        key = (rec["metric"], rec["date"])
        iem_max = iem_daily.get(key)
        if iem_max is None:
            n_no_iem += 1
            continue
        iem_in_band = rec["strike_lo"] <= iem_max <= rec["strike_hi"]
        kalshi_yes  = rec["result"] == "yes"
        if iem_in_band == kalshi_yes:
            n_agree += 1
        else:
            n_disagree += 1
            disagree_examples.append({
                **rec,
                "iem_daily_max": iem_max,
                "iem_in_band":   iem_in_band,
                "kalshi_yes":    kalshi_yes,
            })

    n_total = n_agree + n_disagree
    if n_total > 0:
        print(f"  (records where actual_high in band at P75 time → entered YES)")
        print(f"  IEM says in-band → Kalshi agrees:    {n_agree:>4} / {n_total} ({100*n_agree/n_total:.1f}%)")
        print(f"  IEM says in-band → Kalshi disagrees: {n_disagree:>4} / {n_total} ({100*n_disagree/n_total:.1f}%)")
        print(f"  No IEM daily data:                   {n_no_iem:>4}")
        if disagree_examples:
            print(f"\n  Disagreement examples (IEM says in-band but Kalshi settled differently):")
            for ex in disagree_examples[:15]:
                ov_flag = "OV" if (ex.get("overshoot") or 0) >= 0 else "LA"
                print(f"    {ex['date']} {ex['metric']:<22} "
                      f"band=[{ex['strike_lo']:.0f},{ex['strike_hi']:.0f}] "
                      f"actual_high={ex['actual_high']:.0f} "
                      f"iem_daily={ex['iem_daily_max']:.1f} "
                      f"kalshi={ex['result']} {ov_flag}")

    # ──────────────────────────────────────────────
    # Analysis 3: For banded entry records, how does IEM daily max vs actual settlement compare?
    # (all records where we ENTERED based on asos_at_p75, regardless of actual_high)
    # ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  3. FULL PICTURE: all entry records (asos_at_p75 inside band)")
    print("     Tracks temperature path from entry to settlement")
    print("=" * 70)

    all_entry: list[dict] = []
    for rec in records:
        metric = rec["metric"]
        date_str = rec["date"]
        asos_at_p75 = rec.get("asos_at_p75")
        if asos_at_p75 is None:
            continue
        bands = bands_by_key.get((metric, date_str), [])
        for b in bands:
            if b["strike_lo"] <= asos_at_p75 <= b["strike_hi"]:
                iem_max = iem_daily.get((metric, date_str))
                overshoot = rec.get("overshoot", 0) or 0
                all_entry.append({
                    "metric":        metric,
                    "date":          date_str,
                    "asos_at_p75":   asos_at_p75,
                    "actual_high":   rec.get("actual_high"),
                    "iem_daily_max": iem_max,
                    "strike_lo":     b["strike_lo"],
                    "strike_hi":     b["strike_hi"],
                    "result":        b["result"],
                    "overshoot":     overshoot,
                    "is_overshot":   overshoot >= 0,
                })
                break

    n_entry = len(all_entry)
    if n_entry == 0:
        print("  No entry records found.")
    else:
        groups = {
            "ALL":      all_entry,
            "OVERSHOT": [r for r in all_entry if r["is_overshot"]],
            "LAGGING":  [r for r in all_entry if not r["is_overshot"]],
        }
        for grp_name, grp in groups.items():
            n = len(grp)
            if n == 0:
                continue
            wr_kalshi = sum(1 for r in grp if r["result"] == "yes") / n
            # IEM daily prediction (where available)
            grp_iem = [r for r in grp if r["iem_daily_max"] is not None]
            wr_iem   = sum(1 for r in grp_iem if r["strike_lo"] <= r["iem_daily_max"] <= r["strike_hi"]) / len(grp_iem) if grp_iem else None
            # Discordance: IEM says win but Kalshi settled no, or vice versa
            discord  = sum(1 for r in grp_iem
                           if (r["strike_lo"] <= r["iem_daily_max"] <= r["strike_hi"]) != (r["result"] == "yes"))
            print(f"\n  {grp_name} (n={n})")
            print(f"    Kalshi settlement WR:    {100*wr_kalshi:.1f}%")
            if wr_iem is not None:
                print(f"    IEM daily prediction WR: {100*wr_iem:.1f}%  (n={len(grp_iem)})")
                print(f"    IEM/Kalshi discordance:  {discord} / {len(grp_iem)} ({100*discord/len(grp_iem):.1f}%)")

        # Break discordance by direction
        print()
        print("  Discordance breakdown (IEM daily != Kalshi settlement):")
        for grp_name, grp in groups.items():
            grp_iem = [r for r in grp if r["iem_daily_max"] is not None]
            if not grp_iem:
                continue
            iem_yes_kalshi_no = [r for r in grp_iem
                                  if r["strike_lo"] <= r["iem_daily_max"] <= r["strike_hi"]
                                  and r["result"] == "no"]
            iem_no_kalshi_yes = [r for r in grp_iem
                                  if not (r["strike_lo"] <= r["iem_daily_max"] <= r["strike_hi"])
                                  and r["result"] == "yes"]
            print(f"    {grp_name:<10} IEM-in-band but Kalshi=no: {len(iem_yes_kalshi_no):>3} ({100*len(iem_yes_kalshi_no)/len(grp_iem):.1f}%)"
                  f"   IEM-outside but Kalshi=yes: {len(iem_no_kalshi_yes):>3} ({100*len(iem_no_kalshi_yes)/len(grp_iem):.1f}%)")

        # Show IEM-in-band but Kalshi=no examples (our "false losses")
        iem_yes_kalshi_no = [r for r in all_entry
                             if r["iem_daily_max"] is not None
                             and r["strike_lo"] <= r["iem_daily_max"] <= r["strike_hi"]
                             and r["result"] == "no"]
        if iem_yes_kalshi_no:
            print(f"\n  IEM says in-band but Kalshi=no ({len(iem_yes_kalshi_no)} records):")
            print(f"    {'Date':<12} {'Metric':<22} {'Band':<12} {'P75_max':>7} {'IEM_daily':>10} {'Kalshi':>8}")
            for r in iem_yes_kalshi_no[:20]:
                print(f"    {r['date']:<12} {r['metric']:<22} "
                      f"[{r['strike_lo']:.0f},{r['strike_hi']:.0f}]     "
                      f"{r['asos_at_p75']:>7.1f}   {r['iem_daily_max']:>10.1f}   {r['result']:>8}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
