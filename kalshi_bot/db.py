"""SQLite connection factory and schema migration manager."""

import logging
import sqlite3
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

# Canonical paths for all live databases.  Import these instead of
# constructing the path inline so a single edit moves all databases at once.
DB_DIR = _PROJECT_ROOT / "data" / "db"
OPPORTUNITY_LOG_DB = DB_DIR / "opportunity_log.db"
STATE_DB           = DB_DIR / "state.db"


def open_db(path: Path | str) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection in autocommit mode."""
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply all pending schema migrations in version order.

    Safe to call multiple times — already-applied migrations are skipped.
    Must be called AFTER all CREATE TABLE IF NOT EXISTS statements have run
    (i.e., after class constructors in main.py).

    To add a new migration: append an ``if current < N`` block and bump N.
    Never edit existing blocks — they are already recorded in schema_version.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version    INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
        )
    """)
    current: int = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0

    def _add_col(table: str, col: str, typedef: str) -> None:
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")

    if current < 1:
        # V1 — all trades-table columns added since the initial schema.
        # Consolidated from TradeExecutor._migrate_schema, ExitManager._EXIT_COLUMNS,
        # and OpportunityLog._migrate_trades_exit_columns (now removed).
        for col, typedef in [
            ("kelly_fraction",        "REAL"),
            ("p_estimate",            "REAL"),
            ("source",                "TEXT"),
            ("outcome",               "TEXT"),
            ("fill_price_cents",      "INTEGER"),
            ("spread_id",             "TEXT"),
            ("market_p_entry",        "REAL"),
            ("yes_bid_entry",         "INTEGER"),
            ("yes_ask_entry",         "INTEGER"),
            ("signal_p_yes",          "REAL"),
            ("corroborating_sources", "TEXT"),
            ("exited_at",             "TEXT"),
            ("exit_price_cents",      "INTEGER"),
            ("exit_pnl_cents",        "REAL"),
            ("exit_reason",           "TEXT"),
            ("exit_order_id",         "TEXT"),
            ("peak_past",             "INTEGER"),
            ("exit_reason_detail",    "TEXT"),
            ("peak_pct_gain",         "REAL"),
            ("peak_at",               "TEXT"),
            ("exit_yes_bid",          "INTEGER"),
            ("exit_yes_ask",          "INTEGER"),
            ("bug_loss",              "INTEGER"),
        ]:
            _add_col("trades", col, typedef)
        conn.execute("INSERT INTO schema_version(version) VALUES(1)")
        logging.info("DB schema migration V1 applied.")

    # Add future migrations here:
    # if current < 2:
    #     ...
    #     conn.execute("INSERT INTO schema_version(version) VALUES(2)")
