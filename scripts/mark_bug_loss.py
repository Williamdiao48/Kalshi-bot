"""Mark or unmark trades as bug-losses excluded from drawdown calculation.

Bug-loss trades were driven by code bugs or data anomalies — not bad signal
quality — and should not penalise the drawdown-based position-sizing of
unrelated strategies.

Usage
-----
  # Mark trades 141 and 142 as bug-losses
  venv/bin/python scripts/mark_bug_loss.py --ids 141 142

  # Unmark (restore to normal)
  venv/bin/python scripts/mark_bug_loss.py --ids 141 142 --unmark

  # Show all currently marked trades
  venv/bin/python scripts/mark_bug_loss.py --list
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark trades as bug-losses.")
    parser.add_argument("--ids", nargs="*", type=int, default=[], metavar="ID")
    parser.add_argument("--unmark", action="store_true", help="Remove bug-loss flag")
    parser.add_argument("--list", action="store_true", help="List all bug-loss trades")
    parser.add_argument("--db", default="opportunity_log.db")
    args = parser.parse_args()

    db = sqlite3.connect(args.db)

    # Ensure column exists (in case DB predates the migration)
    try:
        db.execute("ALTER TABLE trades ADD COLUMN bug_loss INTEGER NOT NULL DEFAULT 0")
        db.commit()
    except Exception:
        pass  # column already exists

    if args.list:
        rows = db.execute(
            """
            SELECT id, logged_at, ticker, side, count, exit_pnl_cents, exit_reason
            FROM trades WHERE bug_loss = 1 ORDER BY id
            """
        ).fetchall()
        if not rows:
            print("No bug-loss trades marked.")
        else:
            print(f"{'ID':>4}  {'Entered':19}  {'Ticker':35}  {'Side':3}  {'Count':5}  {'PnL':>8}  Reason")
            for r in rows:
                pnl = f"{r[5]:+.0f}¢" if r[5] is not None else "open"
                print(f"{r[0]:>4}  {r[1][:19]}  {r[2]:35}  {r[3]:3}  {r[4]:5}  {pnl:>8}  {r[6]}")
        return

    if not args.ids:
        parser.print_help()
        sys.exit(0)

    val = 0 if args.unmark else 1
    db.executemany("UPDATE trades SET bug_loss=? WHERE id=?", [(val, i) for i in args.ids])
    db.commit()

    action = "Unmarked" if args.unmark else "Marked as bug-loss"
    for trade_id in args.ids:
        row = db.execute(
            "SELECT ticker, exit_pnl_cents FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
        if row:
            pnl = f"{row[1]:+.0f}¢" if row[1] is not None else "open"
            print(f"  {action}: trade {trade_id} — {row[0]} ({pnl})")
        else:
            print(f"  WARNING: trade {trade_id} not found")


if __name__ == "__main__":
    main()
