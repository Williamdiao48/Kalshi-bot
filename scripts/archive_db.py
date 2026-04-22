#!/usr/bin/env python3
"""Purge bulk data from opportunity_log.db, archive it, then delete for a fresh dry run.

Run once with the bot stopped:
  venv/bin/python scripts/archive_db.py

What this does:
  1. Deletes raw_forecasts, price_snapshots, and opportunities from the live DB
     (these are the high-volume tables that balloon to millions of rows)
  2. VACUUMs the DB to reclaim disk space
  3. Copies the pruned DB as a datestamped archive (keeps trades + circuit_breakers)
  4. Exports the trades table to a CSV for human-readable permanent reference
  5. Deletes the live DB so the bot auto-creates a clean one on next start

The archive DB and CSV are written to the project root alongside opportunity_log.db.
"""

import csv
import os
import shutil
import sqlite3
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "opportunity_log.db"

if not DB_PATH.exists():
    print(f"ERROR: {DB_PATH} not found — nothing to archive.")
    raise SystemExit(1)

stamp = date.today().strftime("%Y%m%d")
archive_db_path  = PROJECT_ROOT / f"opportunity_log_archive_{stamp}.db"
archive_csv_path = PROJECT_ROOT / f"trades_archive_{stamp}.csv"

print(f"Source DB:  {DB_PATH}")
print(f"Archive DB: {archive_db_path}")
print(f"Trades CSV: {archive_csv_path}")
print()

conn = sqlite3.connect(DB_PATH)

# ---- 1. Report current sizes ------------------------------------------------
for table in ("raw_forecasts", "price_snapshots", "opportunities", "trades", "circuit_breakers"):
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n:,} rows")
    except sqlite3.OperationalError:
        print(f"  {table}: (table not found)")
print()

# ---- 2. Purge bulk tables ---------------------------------------------------
bulk_tables = [
    ("raw_forecasts",   "bulk forecast data — ~100 rows/poll cycle"),
    ("price_snapshots", "price trajectory snapshots — ~5 rows/poll cycle"),
    ("opportunities",   "surfaced opportunities — ~100 rows/poll cycle"),
]

for table, desc in bulk_tables:
    try:
        n = conn.execute(f"DELETE FROM {table}").rowcount
        print(f"Purged {table}: {n:,} rows deleted  ({desc})")
    except sqlite3.OperationalError as e:
        print(f"Skipped {table}: {e}")

conn.commit()

# ---- 3. VACUUM to reclaim disk space ----------------------------------------
print("\nVACUUMing database (reclaiming disk space)...")
conn.execute("VACUUM")
conn.close()
print("VACUUM complete.")

# ---- 4. Copy pruned DB as archive -------------------------------------------
shutil.copy(DB_PATH, archive_db_path)
size_kb = archive_db_path.stat().st_size // 1024
print(f"\nArchived pruned DB → {archive_db_path.name}  ({size_kb} KB)")

# ---- 5. Export trades to CSV ------------------------------------------------
conn2 = sqlite3.connect(DB_PATH)
cur = conn2.execute("SELECT * FROM trades ORDER BY id")
cols = [d[0] for d in cur.description]
rows = cur.fetchall()
conn2.close()

with open(archive_csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    w.writerows(rows)
print(f"Exported {len(rows)} trades → {archive_csv_path.name}")

# ---- 6. Delete live DB -------------------------------------------------------
os.remove(DB_PATH)
print(f"\nDeleted {DB_PATH.name}")
print("\nDone. The bot will create a fresh opportunity_log.db on next start.")
print("Remember to remove DRAWDOWN_IGNORE_BEFORE_ID from .env if set.")
