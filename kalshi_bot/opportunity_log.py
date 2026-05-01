"""Persistent opportunity log backed by SQLite.

Every opportunity that survives the full pipeline (liquidity filter + scoring +
cross-cycle cooldown) and is displayed to the user is recorded here. The log
serves two purposes:

  1. CROSS-CYCLE DEDUPLICATION (display suppression)
     The same (ticker, signal_key) pair is not re-displayed within a configurable
     cooldown window (OPPORTUNITY_COOLDOWN_MINUTES, default 60). This prevents
     the bot from spamming the same market opportunity every 60-second cycle
     while a news story is still fresh in the RSS feeds.

     signal_key is the matched_term for text opportunities and the metric name
     for numeric opportunities. Keying on (ticker, signal_key) rather than
     ticker alone allows the same market to fire again if a *different* signal
     matches it (e.g., a separate article on a different topic).

  2. BACKTESTING AUDIT TRAIL
     Each logged row captures the score, bid/ask spread, days-to-close, and the
     relevant signal data at the moment of surfacing. After the fact, you can
     join this table against historical Kalshi price data to measure whether the
     market moved in the implied direction — quantifying actual alpha.

Schema
------
The `opportunities` table has a shared header section (all opportunities) and
two nullable sections for type-specific fields:

  Shared fields:
    id, logged_at, kind, ticker, signal_key, market_title,
    score, yes_bid, yes_ask, volume, days_to_close, source

  Text-only fields (kind = 'text'):
    topic, doc_title, doc_url

  Numeric-only fields (kind = 'numeric'):
    data_value, unit, direction, strike, strike_lo, strike_hi,
    implied_outcome, edge

Database location
-----------------
Written to `opportunity_log.db` in the project root (same directory as
`state.db`). The path is overridable via the constructor for testing.
"""

import logging
import os
import sqlite3

from .db import open_db
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .matcher import Opportunity
from .numeric_matcher import NumericOpportunity
from .polymarket_matcher import PolyOpportunity

_DEFAULT_DB_PATH = Path(__file__).parent.parent / "opportunity_log.db"

# ---------------------------------------------------------------------------
# Retention policy — rows older than these thresholds are deleted on startup.
# Set to 0 to disable purging for a given table.
# trades and circuit_breakers are never purged (tiny tables, irreplaceable).
# ---------------------------------------------------------------------------
RETENTION_OPPORTUNITIES_DAYS: int = int(os.environ.get("RETENTION_OPPORTUNITIES_DAYS", "30"))
RETENTION_RAW_FORECASTS_DAYS: int = int(os.environ.get("RETENTION_RAW_FORECASTS_DAYS", "30"))
RETENTION_PRICE_SNAPSHOTS_DAYS: int = int(os.environ.get("RETENTION_PRICE_SNAPSHOTS_DAYS", "30"))

_CREATE_RAW_FORECASTS_SQL = """
CREATE TABLE IF NOT EXISTS raw_forecasts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at   TEXT    NOT NULL,
    source      TEXT    NOT NULL,   -- e.g. "weatherapi", "noaa", "open_meteo"
    metric      TEXT    NOT NULL,   -- e.g. "temp_high_lax"
    ticker      TEXT    NOT NULL,   -- e.g. "KXHIGHLAX-26APR06-T75"
    data_value  REAL    NOT NULL,   -- forecast temperature (°F)
    strike      REAL,               -- market strike threshold
    direction   TEXT,               -- "over" | "under" | "between"
    edge        REAL,               -- |forecast - strike| (°F)
    yes_bid     INTEGER             -- Kalshi YES bid price in cents at time of logging
)
"""

_CREATE_SIGNAL_SUPPRESSION_SQL = """
CREATE TABLE IF NOT EXISTS signal_suppression_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at  TEXT    NOT NULL,
    ticker     TEXT    NOT NULL,
    source     TEXT    NOT NULL,
    metric     TEXT,
    gate       TEXT    NOT NULL,
    edge_f     REAL,
    value      REAL,
    strike     REAL,
    yes_bid    INTEGER,
    note       TEXT
)
"""

_CREATE_SUPPRESSION_IDX_SQL = """
CREATE INDEX IF NOT EXISTS idx_suppression_ticker
    ON signal_suppression_log (ticker, logged_at)
"""

_CREATE_RAW_IDX_SQL = """
CREATE INDEX IF NOT EXISTS idx_raw_forecasts_ticker_source
    ON raw_forecasts (ticker, source, logged_at)
"""

# View: de-duplicate raw_forecasts by collapsing all intra-day cycles into one
# row per (date, ticker, source, direction).  avg_forecast and avg_edge are
# averaged across cycles; n_cycles shows how many poll cycles contributed.
_CREATE_RAW_DAILY_VIEW_SQL = """
CREATE VIEW IF NOT EXISTS raw_forecasts_daily AS
SELECT
    DATE(logged_at)          AS date,
    ticker,
    source,
    direction,
    ROUND(AVG(data_value), 1) AS avg_forecast,
    ROUND(AVG(edge), 1)       AS avg_edge,
    COUNT(*)                  AS n_cycles
FROM raw_forecasts
GROUP BY DATE(logged_at), ticker, source, direction
"""

# View: join trades to raw_forecasts_daily so every trade row shows ALL
# forecast sources that were present on that ticker that day — not just the
# primary source stored in trades.source.
#
# Usage examples:
#   -- All sources that corroborated trade #220:
#   SELECT signal_source, direction, avg_forecast, avg_edge
#   FROM trade_context WHERE trade_id = 220;
#
#   -- Summary: sources grouped per trade:
#   SELECT trade_id, ticker, side, GROUP_CONCAT(signal_source) AS sources
#   FROM trade_context GROUP BY trade_id;
_CREATE_TRADE_CONTEXT_VIEW_SQL = """
CREATE VIEW IF NOT EXISTS trade_context AS
SELECT
    t.id                     AS trade_id,
    DATE(t.logged_at)        AS trade_date,
    t.ticker,
    t.side,
    t.limit_price,
    t.status,
    t.outcome,
    t.exit_pnl_cents,
    t.source                 AS primary_source,
    rf.source                AS signal_source,
    rf.direction,
    rf.avg_forecast,
    rf.avg_edge
FROM trades t
JOIN raw_forecasts_daily rf
    ON  rf.ticker = t.ticker
    AND rf.date   = DATE(t.logged_at)
WHERE t.opportunity_kind = 'numeric'
"""

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS opportunities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at       TEXT    NOT NULL,
    kind            TEXT    NOT NULL CHECK(kind IN ('text', 'numeric')),
    ticker          TEXT    NOT NULL,
    signal_key      TEXT    NOT NULL,   -- matched_term (text) | metric (numeric)
    market_title    TEXT    NOT NULL,
    score           REAL    NOT NULL,
    yes_bid         INTEGER,            -- cents, NULL if unavailable
    yes_ask         INTEGER,            -- cents, NULL if unavailable
    volume          INTEGER,            -- NULL if unavailable
    days_to_close   REAL,               -- NULL if unavailable
    source          TEXT,               -- feed name or data source

    -- text-only columns (NULL for numeric rows)
    topic           TEXT,
    doc_title       TEXT,
    doc_url         TEXT,

    -- numeric-only columns (NULL for text rows)
    data_value      REAL,
    unit            TEXT,
    direction       TEXT,
    strike          REAL,
    strike_lo       REAL,
    strike_hi       REAL,
    implied_outcome TEXT,
    edge            REAL
)
"""

_CREATE_IDX_TICKER_SQL = """
CREATE INDEX IF NOT EXISTS idx_opp_ticker_signal_logged
    ON opportunities (ticker, signal_key, logged_at)
"""

_CREATE_IDX_LOGGED_SQL = """
CREATE INDEX IF NOT EXISTS idx_opp_logged_at
    ON opportunities (logged_at)
"""


class OpportunityLog:
    """SQLite-backed log of surfaced trading opportunities.

    Thread-safety note: SQLite in WAL mode allows one writer at a time.
    This class is designed for single-threaded / single-async-event-loop use
    (same as the rest of the bot). Do not share across threads.

    Usage:
        log = OpportunityLog()
        suppressed = log.recently_surfaced_pairs(cooldown_minutes=60)
        # ... filter opportunities against suppressed ...
        log.log_text(opp, detail, score, days_to_close)
        log.log_numeric(opp, detail, score, days_to_close)
        log.close()
    """

    def __init__(self, db_path: Path | str = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._conn = open_db(self._db_path)
        self._init_schema()
        self._purge_old_rows()
        logging.debug("OpportunityLog opened at %s", self._db_path)

    def _purge_old_rows(self) -> None:
        """Delete rows older than configured retention windows.

        Called once on startup. Only purges high-volume tables (opportunities,
        raw_forecasts, price_snapshots). trades and circuit_breakers are never
        purged — they are tiny and serve as a permanent audit trail.
        Set any retention env var to 0 to disable purging for that table.
        """
        thresholds = [
            ("opportunities",   "logged_at",   RETENTION_OPPORTUNITIES_DAYS),
            ("raw_forecasts",   "logged_at",   RETENTION_RAW_FORECASTS_DAYS),
            ("price_snapshots", "snapshot_at", RETENTION_PRICE_SNAPSHOTS_DAYS),
        ]
        total = 0
        for table, ts_col, days in thresholds:
            if days <= 0:
                continue
            exists = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not exists:
                continue
            cur = self._conn.execute(
                f"DELETE FROM {table} WHERE {ts_col} < datetime('now', ?)",
                (f"-{days} days",),
            )
            if cur.rowcount:
                logging.info(
                    "Retention purge: deleted %d rows from %s (older than %d days)",
                    cur.rowcount, table, days,
                )
                total += cur.rowcount
        if total:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _init_schema(self) -> None:
        # Connection is in autocommit mode (isolation_level=None); each execute
        # is its own implicit transaction. No context manager needed.
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_IDX_TICKER_SQL)
        self._conn.execute(_CREATE_IDX_LOGGED_SQL)
        self._conn.execute(_CREATE_RAW_FORECASTS_SQL)
        self._conn.execute(_CREATE_RAW_IDX_SQL)
        self._conn.execute(_CREATE_RAW_DAILY_VIEW_SQL)
        self._conn.execute(_CREATE_TRADE_CONTEXT_VIEW_SQL)
        self._conn.execute(_CREATE_SIGNAL_SUPPRESSION_SQL)
        self._conn.execute(_CREATE_SUPPRESSION_IDX_SQL)
        self._migrate_trades_exit_columns()

    def _migrate_trades_exit_columns(self) -> None:
        """Add analysis columns to trades that didn't exist in older schema versions."""
        table_exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trades'"
        ).fetchone()
        if not table_exists:
            return
        existing = {r[1] for r in self._conn.execute("PRAGMA table_info(trades)")}
        for col, typedef in [
            ("exit_reason_detail", "TEXT"),
            ("peak_pct_gain",      "REAL"),
            ("peak_at",            "TEXT"),
            ("exit_yes_bid",       "INTEGER"),
            ("exit_yes_ask",       "INTEGER"),
        ]:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typedef}")

    # -----------------------------------------------------------------------
    # Cross-cycle deduplication
    # -----------------------------------------------------------------------

    def recently_surfaced_pairs(self, cooldown_minutes: int) -> set[tuple[str, str]]:
        """Return (ticker, signal_key) pairs logged within the cooldown window.

        Call once per cycle and check membership with `in` for O(1) lookups.
        Any opportunity whose (ticker, signal_key) appears in the returned set
        was already shown to the user recently and should be suppressed from
        display (but a new signal type on the same ticker is still allowed).

        Args:
            cooldown_minutes: How far back to look. 0 = no suppression (return
                              empty set), which surfaces all opportunities every
                              cycle.

        Returns:
            Set of (ticker, signal_key) tuples.
        """
        if cooldown_minutes <= 0:
            return set()

        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=cooldown_minutes)
        ).isoformat()

        rows = self._conn.execute(
            "SELECT DISTINCT ticker, signal_key FROM opportunities WHERE logged_at >= ?",
            (cutoff,),
        ).fetchall()

        return {(row[0], row[1]) for row in rows}

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def log_raw_forecasts(self, opps: "list[NumericOpportunity]") -> None:
        """Log ALL numeric weather opportunities before edge/spread gating.

        Called once per poll cycle with the full pre-gate opportunity list so
        every source's forecast is captured regardless of whether it passes the
        edge threshold.  This builds the training dataset needed for per-source
        accuracy calibration in scripts/backtest_source_accuracy.py.

        Duplicate (ticker, source) pairs within the same calendar day are
        intentionally kept — they show how forecasts evolve intraday.
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for opp in opps:
            if not opp.metric.startswith(("temp_high_", "temp_low_")):
                continue
            # current_market_price is yes_bid in cents (int) or "N/A"
            yes_bid = opp.current_market_price if isinstance(opp.current_market_price, int) else None
            rows.append((
                now,
                opp.source,
                opp.metric,
                opp.market_ticker,
                opp.data_value,
                opp.strike,
                opp.direction,
                opp.edge,
                yes_bid,
            ))
        if rows:
            self._conn.executemany(
                """
                INSERT INTO raw_forecasts
                    (logged_at, source, metric, ticker, data_value, strike, direction, edge, yes_bid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def log_forecast_no_sources(self, signals: "list") -> None:
        """Log per-source qualifying details for forecast_no signals to raw_forecasts.

        Each ForecastNoSignal carries source_details: list of (source, value, edge_F).
        Writing these to raw_forecasts makes open_meteo and noaa contributions visible
        alongside the HRRR/METAR entries already logged by log_raw_forecasts().
        """
        from .strike_arb import ForecastNoSignal
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for sig in signals:
            if not isinstance(sig, ForecastNoSignal):
                continue
            # Derive strike from direction for the raw_forecasts schema
            strike: float | None = None
            if hasattr(sig, "direction"):
                # We don't have parsed.strike here; use None — logged for visibility only
                pass
            for source, value, edge in sig.source_details:
                rows.append((
                    now,
                    source,
                    sig.metric,
                    sig.ticker,
                    value,
                    None,       # strike — not available on ForecastNoSignal
                    sig.direction,
                    edge,
                    None,       # yes_bid — not available here
                ))
        if rows:
            self._conn.executemany(
                """
                INSERT INTO raw_forecasts
                    (logged_at, source, metric, ticker, data_value, strike, direction, edge, yes_bid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def log_text(
        self,
        opp: Opportunity,
        detail: dict | None,
        score: float,
        days_to_close: float,
    ) -> None:
        """Persist a text/keyword opportunity to the log.

        The dedup key written to ``signal_key`` is the first matched term
        (lowercased), or ``opp.topic`` as fallback.  This means the same
        topic can re-fire on the same market once the cooldown window expires,
        but a different term on the same market fires immediately.

        Args:
            opp:           The matched text opportunity.
            detail:        Live market detail dict from ``fetch_market_detail``
                           (may be None if the fetch failed).
            score:         Composite score in [0, 1] from ``score_text_opportunity``.
            days_to_close: Fractional days until market close_time.
        """
        bid = detail.get("yes_bid") if detail else None
        ask = detail.get("yes_ask") if detail else None
        vol = detail.get("volume") if detail else None
        signal_key = (opp.matched_terms[0] if opp.matched_terms else opp.topic).lower()

        self._conn.execute(
            """
            INSERT INTO opportunities (
                logged_at, kind, ticker, signal_key, market_title, score,
                yes_bid, yes_ask, volume, days_to_close, source,
                topic, doc_title, doc_url
            ) VALUES (?, 'text', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                opp.market_ticker,
                signal_key,
                opp.market_title,
                score,
                bid,
                ask,
                vol,
                None if days_to_close == float("inf") else days_to_close,
                opp.source,
                opp.topic,
                opp.doc_title,
                opp.doc_url,
            ),
        )

    def log_numeric(
        self,
        opp: NumericOpportunity,
        detail: dict | None,
        score: float,
        days_to_close: float,
    ) -> None:
        """Persist a numeric/data opportunity to the log.

        The dedup key written to ``signal_key`` is ``opp.metric`` (e.g.
        ``"temp_high_lax"``).  This means one NOAA DataPoint and one METAR
        DataPoint for the same metric on the same market share a dedup key —
        the second source is suppressed within the cooldown window.  This is
        intentional: both point to the same underlying signal.

        Args:
            opp:           The matched numeric opportunity.
            detail:        Live market detail dict from ``fetch_market_detail``
                           (may be None if the fetch failed).
            score:         Composite score in [0, 1] from ``score_numeric_opportunity``.
            days_to_close: Fractional days until market close_time.
        """
        bid = detail.get("yes_bid") if detail else None
        ask = detail.get("yes_ask") if detail else None
        vol = detail.get("volume") if detail else None

        self._conn.execute(
            """
            INSERT INTO opportunities (
                logged_at, kind, ticker, signal_key, market_title, score,
                yes_bid, yes_ask, volume, days_to_close, source,
                data_value, unit, direction, strike, strike_lo, strike_hi,
                implied_outcome, edge
            ) VALUES (?, 'numeric', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                opp.market_ticker,
                opp.metric,         # signal_key for numeric = metric name
                opp.market_title,
                score,
                bid,
                ask,
                vol,
                None if days_to_close == float("inf") else days_to_close,
                opp.source,
                opp.data_value,
                opp.unit,
                opp.direction,
                opp.strike,
                opp.strike_lo,
                opp.strike_hi,
                opp.implied_outcome,
                opp.edge,
            ),
        )

    def log_poly(
        self,
        opp: PolyOpportunity,
        detail: dict | None,
        score: float,
        days_to_close: float = float("inf"),
    ) -> None:
        """Persist an external-forecast divergence opportunity to the log.

        Works for Polymarket, Metaculus, and Manifold opportunities — all
        represented as PolyOpportunity with different ``source`` values.

        Stored as kind='numeric' so it participates in the same cooldown logic.
        signal_key = "{source}:{poly_market_id}" for dedup.
        data_value = poly_p_yes × 100 (probability as a 0–100 scale value).
        strike     = kalshi_mid (the price the external source disagrees with).
        edge       = divergence (|ext_p − kalshi_mid/100|).
        """
        bid = detail.get("yes_bid") if detail else None
        ask = detail.get("yes_ask") if detail else None
        vol = detail.get("volume") if detail else None

        self._conn.execute(
            """
            INSERT INTO opportunities (
                logged_at, kind, ticker, signal_key, market_title, score,
                yes_bid, yes_ask, volume, days_to_close, source,
                doc_title,
                data_value, unit, direction, strike,
                implied_outcome, edge
            ) VALUES (?, 'numeric', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                opp.kalshi_ticker,
                f"{opp.source}:{opp.poly_market_id}",
                opp.kalshi_title,
                score,
                bid,
                ask,
                vol,
                None if days_to_close == float("inf") else days_to_close,
                opp.source,
                opp.poly_question,
                round(opp.poly_p_yes * 100, 2),
                "%",
                "over" if opp.implied_side == "yes" else "under",
                opp.kalshi_mid,
                opp.implied_side.upper(),
                opp.divergence,
            ),
        )

    def log_suppression(
        self,
        ticker: str,
        source: str,
        gate: str,
        *,
        metric: str | None = None,
        edge_f: float | None = None,
        value: float | None = None,
        strike: float | None = None,
        yes_bid: int | None = None,
        note: str | None = None,
    ) -> None:
        """Record a signal that was suppressed by a quality gate.

        Builds an audit trail of filtered-out signals so future analysis can
        ask "what did the gates block and was it correct to block it?"

        Args:
            ticker:  Kalshi market ticker.
            source:  Data source (e.g. "hrrr", "noaa", "nws_hourly").
            gate:    Short label identifying which gate suppressed the signal.
                     Vocabulary: "hrrr_spread", "corr_lone_rt_no",
                     "corr_lone_daily_no", "corr_solo_yes", "corr_future_lone",
                     "band_arb_metar_only", "kxlowt_fc_contra",
                     "obs_afternoon_gate", "edge_min", "ttc_window".
            metric:  Metric name (e.g. "temp_high_lax").
            edge_f:  Edge in °F (or other units) at suppression time.
            value:   Source forecast value.
            strike:  Market strike threshold.
            yes_bid: YES bid price in cents at suppression time.
            note:    Optional free-text (e.g. the threshold that fired).
        """
        self._conn.execute(
            """
            INSERT INTO signal_suppression_log
                (logged_at, ticker, source, metric, gate, edge_f, value, strike, yes_bid, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                ticker, source, metric, gate,
                edge_f, value, strike, yes_bid, note,
            ),
        )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
