#!/usr/bin/env python3
"""One-off historical fill of cli_daily (official NWS CLI highs/lows).

The poll loop keeps cli_daily current once per day; this script fills past years
so the lookahead model can be trained on settlement-grade labels.

    venv/bin/python scripts/backfill_cli_labels.py --years 2026
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kalshi_bot.db import open_db, run_migrations  # noqa: E402
from kalshi_bot.shadow_forecast import backfill_cli_actuals  # noqa: E402


async def _run(db: str, years: list[int]) -> None:
    conn = open_db(db)
    try:
        run_migrations(conn)
        async with aiohttp.ClientSession() as session:
            n = await backfill_cli_actuals(conn, session, years=years)
        span = conn.execute("SELECT MIN(date), MAX(date), COUNT(*), COUNT(DISTINCT city) "
                            "FROM cli_daily").fetchone()
        print(f"upserted {n:,} rows; cli_daily now {span[2]:,} rows, "
              f"{span[3]} cities, {span[0]} -> {span[1]}")
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/db/opportunity_log.db")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    a = ap.parse_args()
    asyncio.run(_run(a.db, a.years))


if __name__ == "__main__":
    main()
