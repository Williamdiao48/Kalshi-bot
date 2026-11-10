#!/usr/bin/env python3
"""Settled Kalshi temperature B-band labels, fetched per EVENT (city-day).

The `status=settled` markets listing ages out old markets, so a series-wide
re-fetch silently returns only a recent window (the old classifier's label cache
has multi-month gaps per series). Fetching by event ticker
(e.g. KXHIGHCHI-26SEP15) returns every market of that city-day regardless of
age, so this is the label source for the classifier v3 rebuild.

Kalshi moves older markets off /markets to /historical/markets (around two to
three months back, unevenly by series), so an event that comes back empty from
/markets is retried there before being recorded as "not listed that day".

Events already in the old series-wide cache are reused; the rest are fetched
one at a time (paced, since the live bot shares the API read quota) and cached
under data/classifier_v3/events/ once final, so reruns only fetch new days.

    venv/bin/python scripts/fetch_temp_band_labels.py --start 2026-05-27
Output: data/classifier_v3/labels.parquet (one row per settled B-band).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kalshi_bot.auth import generate_headers  # noqa: E402
from kalshi_bot.classifier_v3 import SERIES_CITY, parse_band_ticker  # noqa: E402
from kalshi_bot.markets import KALSHI_API_BASE  # noqa: E402

OUT_DIR = Path("data/classifier_v3")
EVENT_DIR = OUT_DIR / "events"
KALSHI_CACHE = Path("data/backtest/kalshi_markets_cache.json")   # old series-wide cache
FINAL = {"settled", "finalized"}
HISTORICAL = "/historical/markets"


def event_ticker(series: str, d: date) -> str:
    return f"{series}-{d.strftime('%y%b%d').upper()}"


def seed_from_cache(events: set[str]) -> dict[str, list[dict]]:
    """Events already present in the old series-wide label cache (settled only)."""
    if not KALSHI_CACHE.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for mkts in json.loads(KALSHI_CACHE.read_text()).values():
        for m in mkts:
            ev = m.get("event_ticker") or m.get("ticker", "").rsplit("-", 1)[0]
            if ev in events:
                out.setdefault(ev, []).append(m)
    return out


async def _get_markets(session: aiohttp.ClientSession, path: str, ev: str) -> list[dict] | None:
    """One event's markets from `path`, retrying 429s / transient errors. None = failed."""
    for attempt in range(6):
        try:
            async with session.get(
                f"{KALSHI_API_BASE}{path}",
                headers=generate_headers("GET", f"/trade-api/v2{path}"),
                params={"event_ticker": ev, "limit": 100},
                timeout=aiohttp.ClientTimeout(total=20),
            ) as r:
                if r.status == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return (await r.json()).get("markets", [])
        except Exception as exc:
            if attempt == 5:
                print(f"  {ev} {path}: giving up ({exc})", flush=True)
            await asyncio.sleep(2 ** attempt)
    return None


async def fetch_all(events: list[str], rate: float) -> dict[str, list[dict]]:
    """Paced, resumable: one request at a time at <= `rate`/s (the live bot shares
    the API read quota), each final event written to disk as soon as it lands."""
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    out = seed_from_cache(set(events))
    todo = []
    for ev in events:
        p = EVENT_DIR / f"{ev}.json"
        if p.exists():
            out[ev] = json.loads(p.read_text())
        elif ev not in out:
            todo.append(ev)
    print(f"{len(events):,} events: {len(events) - len(todo):,} known, fetching {len(todo):,}",
          flush=True)
    async with aiohttp.ClientSession() as session:
        for i, ev in enumerate(todo, 1):
            mkts = await _get_markets(session, "/markets", ev)
            await asyncio.sleep(1.0 / rate)
            if mkts == []:                       # aged out of /markets?
                mkts = await _get_markets(session, HISTORICAL, ev)
                await asyncio.sleep(1.0 / rate)
            if mkts is None:
                continue
            out[ev] = mkts
            # Freeze to disk only once every market in the event is final
            # (an empty list = listed on neither endpoint, i.e. no market that day).
            if all(m.get("status") in FINAL for m in mkts):
                (EVENT_DIR / f"{ev}.json").write_text(json.dumps(mkts))
            if i % 250 == 0:
                print(f"  {i:,}/{len(todo):,}", flush=True)
    return out


def to_frame(by_event: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for mkts in by_event.values():
        for m in mkts:
            b = parse_band_ticker(m.get("ticker", ""))
            if b is None or m.get("status") not in FINAL or m.get("result") not in ("yes", "no"):
                continue
            series, d, _band_lo, band_ceil = b
            city, is_high = SERIES_CITY[series]
            rows.append(dict(
                ticker=m["ticker"], series=series, city=city, is_high=int(is_high),
                date_target=d.isoformat(),
                floor_strike=m.get("floor_strike"), cap_strike=m.get("cap_strike"),
                band_ceil=band_ceil, close_time=m.get("close_time"),
                result=m["result"], won_no=int(m["result"] == "no"),
            ))
    df = pd.DataFrame(rows)
    if not df.empty:
        bad = df["cap_strike"].notna() & (df["cap_strike"] != df["band_ceil"])
        if bad.any():
            raise SystemExit(f"ticker-derived band_ceil != cap_strike on {bad.sum()} rows")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-05-27")
    ap.add_argument("--end", default=None, help="inclusive; default = 2 days ago (UTC)")
    ap.add_argument("--rate", type=float, default=4.0, help="max requests per second")
    a = ap.parse_args()
    start = date.fromisoformat(a.start)
    end = (date.fromisoformat(a.end) if a.end
           else datetime.now(timezone.utc).date() - timedelta(days=2))
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    events = [event_ticker(s, d) for s in sorted(SERIES_CITY) for d in days]
    by_event = asyncio.run(fetch_all(events, a.rate))
    df = to_frame(by_event)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "labels.parquet", index=False)
    print(f"{len(df):,} settled B-bands, {df.groupby(['series', 'date_target']).ngroups:,} "
          f"series-days, {df['date_target'].min()} -> {df['date_target'].max()}, "
          f"NO win rate {df['won_no'].mean():.3f}")
    print(df.groupby("series")["date_target"].nunique().to_string())


if __name__ == "__main__":
    main()
