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
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .matcher import Opportunity
from .numeric_matcher import NumericOpportunity
from .polymarket_matcher import PolyOpportunity

_DEFAULT_DB_PATH = Path(__file__).parent.parent / "opportunity_log.db"

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
    edge        REAL                -- |forecast - strike| (°F)
)
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
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,   # autocommit; we manage transactions manually
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        logging.debug("OpportunityLog opened at %s", self._db_path)

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
        rows = [
            (
                now,
                opp.source,
                opp.metric,
                opp.market_ticker,
                opp.data_value,
                opp.strike,
                opp.direction,
                opp.edge,
            )
            for opp in opps
            if opp.metric.startswith("temp_high_")
        ]
        if rows:
            self._conn.executemany(
                """
                INSERT INTO raw_forecasts
                    (logged_at, source, metric, ticker, data_value, strike, direction, edge)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

        Args:
            opp:          The matched text opportunity.
            detail:       Live market detail dict (may be None).
            score:        Composite score in [0, 1].
            days_to_close: Fractional days until market closes.
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

        Args:
            opp:          The matched numeric opportunity.
            detail:       Live market detail dict (may be None).
            score:        Composite score in [0, 1].
            days_to_close: Fractional days until market closes.
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

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
