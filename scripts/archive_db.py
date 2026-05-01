#!/usr/bin/env python3
"""Archive the current dry run and prepare for a fresh start.

Run once with the bot stopped:
  venv/bin/python scripts/archive_db.py

What this does:
  1. Archives dry_run_overview.txt → dry_run_overview_archive_YYYYMMDD.txt
  2. Purges bulk tables (raw_forecasts, price_snapshots, opportunities) from the live DB
  3. VACUUMs the live DB to reclaim disk space
  4. Copies the pruned DB → opportunity_log_archive_YYYYMMDD.db  (trades + circuit_breakers)
  5. Exports trades table → trades_archive_YYYYMMDD.csv
  6. Deletes the live DB so the bot auto-creates a clean one on next start
  7. Deletes known stale/redundant files (zero-byte legacy DBs, oversized old .bak files,
     stale analysis txt files)

Files NOT touched:
  state.db            — RSS/EDGAR seen-documents tracker; keep to avoid reprocessing history
  dry_run_overview_archive_old.txt — already archived
  opportunity_log_archive_20260412.db / trades_archive_20260412.csv — prior archive

After running:
  - Remove DRAWDOWN_IGNORE_BEFORE_ID from .env if set
  - Start the bot normally: venv/bin/python run.py
"""

import csv
import os
import shutil
import sqlite3
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ARCHIVES_DIR = PROJECT_ROOT / "archives"
DB_PATH = PROJECT_ROOT / "opportunity_log.db"

if not DB_PATH.exists():
    print(f"ERROR: {DB_PATH} not found — nothing to archive.")
    raise SystemExit(1)

ARCHIVES_DIR.mkdir(exist_ok=True)

stamp = date.today().strftime("%Y%m%d")
archive_db_path       = ARCHIVES_DIR / f"opportunity_log_archive_{stamp}.db"
archive_csv_path      = ARCHIVES_DIR / f"trades_archive_{stamp}.csv"
overview_src          = PROJECT_ROOT / "dry_run_overview.txt"
archive_overview_path = ARCHIVES_DIR / f"dry_run_overview_archive_{stamp}.txt"

print(f"Source DB:       {DB_PATH}")
print(f"Archive DB:      {archive_db_path}")
print(f"Trades CSV:      {archive_csv_path}")
print(f"Overview archive:{archive_overview_path}")
print()

# ---- 1. Archive dry_run_overview.txt -----------------------------------------
if overview_src.exists():
    shutil.copy(overview_src, archive_overview_path)
    print(f"Archived overview → {archive_overview_path.name}")
else:
    print("dry_run_overview.txt not found — skipping overview archive")

# ---- 2. Report current table sizes -------------------------------------------
conn = sqlite3.connect(DB_PATH)

for table in ("raw_forecasts", "price_snapshots", "opportunities", "trades", "circuit_breakers"):
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n:,} rows")
    except sqlite3.OperationalError:
        print(f"  {table}: (table not found)")
print()

# ---- 3. Purge bulk tables ----------------------------------------------------
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

# ---- 4. VACUUM to reclaim disk space -----------------------------------------
print("\nVACUUMing database (reclaiming disk space)...")
conn.execute("VACUUM")
conn.close()
print("VACUUM complete.")

# ---- 5. Copy pruned DB as archive --------------------------------------------
shutil.copy(DB_PATH, archive_db_path)
size_kb = archive_db_path.stat().st_size // 1024
print(f"\nArchived pruned DB → {archive_db_path.name}  ({size_kb} KB)")

# ---- 6. Export trades to CSV -------------------------------------------------
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

# ---- 7. Delete live DB -------------------------------------------------------
# Also remove WAL/SHM side-files if present
for ext in ("", "-shm", "-wal"):
    p = DB_PATH.parent / (DB_PATH.name + ext)
    if p.exists():
        os.remove(p)
        print(f"Deleted {p.name}")

# ---- 8. Delete stale/redundant files ----------------------------------------
STALE_FILES = [
    # Zero-byte legacy DBs from old bot versions
    "bot_state.db",
    "dry_run.db",
    "dry_run_ledger.db",
    "kalshi_bot.db",
    "opportunities.db",
    "trades.db",
    # Huge raw .bak files superseded by the structured Apr 12 archive
    "opportunity_log.db.bak_20260414_181851",
    "opportunity_log.db.bak_20260415_bugfix2",
    # Stale one-off analysis files (regenerate as needed)
    "exit_analysis.txt",
    "exit_analysis_58.txt",
    "momentum_test.txt",
    # Current overview — now archived above; will be rewritten on first bot cycle
    "dry_run_overview.txt",
]

print("\nCleaning up stale files:")
for name in STALE_FILES:
    p = PROJECT_ROOT / name
    if p.exists():
        size_mb = p.stat().st_size / 1_048_576
        os.remove(p)
        print(f"  Deleted {name}  ({size_mb:.1f} MB)")
    else:
        print(f"  Skipped {name}  (not found)")

# ---- Done -------------------------------------------------------------------
print("\n" + "=" * 60)
print("Archive complete. Next steps:")
print("  1. Remove DRAWDOWN_IGNORE_BEFORE_ID from .env if set")
print("  2. Start the bot: venv/bin/python run.py")
print("     (fresh opportunity_log.db will be created automatically)")
print("=" * 60)
