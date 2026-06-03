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

    if current < 2:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nba_snapshots (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at           TEXT    NOT NULL,
                game_date           TEXT    NOT NULL,
                matchup_id          INTEGER NOT NULL,
                home_team           TEXT    NOT NULL,
                away_team           TEXT    NOT NULL,
                pinnacle_home       REAL,
                pinnacle_away       REAL,
                kalshi_ticker_home  TEXT,
                kalshi_ticker_away  TEXT,
                kalshi_home_bid     INTEGER,
                kalshi_home_ask     INTEGER,
                kalshi_away_bid     INTEGER,
                kalshi_away_ask     INTEGER
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nba_snapshots_game
                ON nba_snapshots (matchup_id, logged_at)
        """)
        conn.execute("INSERT INTO schema_version(version) VALUES(2)")
        logging.info("DB schema migration V2 applied (nba_snapshots).")

    if current < 3:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_band_arb (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at    TEXT    NOT NULL,
                ticker       TEXT    NOT NULL,
                side         TEXT    NOT NULL,
                limit_price  INTEGER NOT NULL,
                contracts    INTEGER NOT NULL,
                city         TEXT,
                observed_f   REAL,
                band_ceil_f  REAL,
                margin_f     REAL,
                corr_status  TEXT,
                hrrr_val_f   REAL,
                outcome      TEXT,
                pnl_cents    REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_band_arb_ticker
                ON shadow_band_arb (ticker, logged_at)
        """)
        conn.execute("INSERT INTO schema_version(version) VALUES(3)")
        logging.info("DB schema migration V3 applied (shadow_band_arb).")

    if current < 4:
        _add_col("trades", "settled_result",    "TEXT")
        _add_col("trades", "settled_pnl_cents", "REAL")
        conn.execute("INSERT INTO schema_version(version) VALUES(4)")
        logging.info("DB schema migration V4 applied (settled_result, settled_pnl_cents).")

    if current < 5:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forecast_shadow_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at         TEXT    NOT NULL,
                city              TEXT    NOT NULL,
                date_target       TEXT    NOT NULL,
                is_high           INTEGER NOT NULL,
                source            TEXT    NOT NULL,
                forecast_f        REAL    NOT NULL,
                actual_f          REAL,
                actual_fetched_at TEXT
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_fsl_dedup
                ON forecast_shadow_log (city, date_target, is_high, source,
                                        strftime('%Y-%m-%dT%H', logged_at))
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fsl_lookup
                ON forecast_shadow_log (city, date_target, is_high, actual_f)
        """)
        conn.execute("INSERT INTO schema_version(version) VALUES(5)")
        logging.info("DB schema migration V5 applied (forecast_shadow_log).")

    if current < 6:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_yes_momentum (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at      TEXT    NOT NULL,
                ticker         TEXT    NOT NULL,
                series         TEXT,
                entry_bid      INTEGER NOT NULL,
                prior_high_bid INTEGER,
                contracts      INTEGER NOT NULL DEFAULT 1,
                pt_target      INTEGER NOT NULL DEFAULT 40,
                exit_reason    TEXT,
                exited_at      TEXT,
                outcome        TEXT,
                pnl_cents      REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sym_open
                ON shadow_yes_momentum (ticker, exit_reason)
        """)
        conn.execute("INSERT INTO schema_version(version) VALUES(6)")
        logging.info("DB schema migration V6 applied (shadow_yes_momentum).")

    if current < 7:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_model_no (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at      TEXT    NOT NULL,
                ticker         TEXT    NOT NULL,
                series         TEXT,
                is_high        INTEGER NOT NULL DEFAULT 0,
                model_p        REAL    NOT NULL,
                market_p_no    REAL    NOT NULL,
                edge           REAL    NOT NULL,
                margin_f       REAL,
                hvc            REAL,
                clim_prob      REAL,
                hour_utc       INTEGER,
                exit_reason    TEXT,
                exited_at      TEXT,
                outcome        TEXT,
                pnl_cents      REAL
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_shadow_model_no_open
                ON shadow_model_no (ticker) WHERE exit_reason IS NULL
        """)
        conn.execute("INSERT INTO schema_version(version) VALUES(7)")
        logging.info("DB schema migration V7 applied (shadow_model_no).")
