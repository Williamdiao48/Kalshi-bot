"""Per-source win-rate tracking for the Kalshi bot.

After a Kalshi market resolves, we compare the trade direction (``side``)
against the actual outcome (``result`` field from the Kalshi API) to record
whether each trade won or lost.  Aggregating across many trades gives a
per-source win rate that drives calibration.

How it works
------------
1. A *settlement pass* runs periodically.  It queries the ``trades`` table
   for rows where ``outcome IS NULL``.
2. For each unique ticker in those rows, we call the Kalshi market-detail
   endpoint and check the ``result`` field.
3. If ``result`` is "yes" or "no" the market has resolved:
       trade.side == result  →  outcome = 'won'
       trade.side != result  →  outcome = 'lost'
   If ``result`` is "void" we record outcome = 'void' (the bet was cancelled).
   Markets still open (``result`` is None) are left with outcome = NULL.
4. After settling, a *summary pass* groups trades by ``source`` and prints
   the win rate for each source that has at least one resolved trade.

Calibration use
---------------
After accumulating 20+ resolved trades per source, copy the observed win
rates into ``KELLY_METRIC_PRIORS`` (env var JSON) to replace the initial
guesses with data-driven priors.  Example:

    KELLY_METRIC_PRIORS='{"noaa_observed": 0.78, "polymarket": 0.64}'

A source with a win rate below ~0.52 has no positive edge and its trades
should be disabled by raising its divergence / edge threshold.

WIN_RATE_REPORT_INTERVAL
------------------------
How many poll cycles between settlement+summary passes.  Default 60
(= every 60 minutes at the default 60-second poll interval).
Set to 1 to report every cycle (useful during initial tuning).
"""

import json
import logging
import os
from .utils import env_float, env_int
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from .db import open_db, OPPORTUNITY_LOG_DB
from .market_parser import TICKER_TO_METRIC
from .markets import fetch_market_detail

WIN_RATE_REPORT_INTERVAL: int = env_int("WIN_RATE_REPORT_INTERVAL", 60)

_DEFAULT_DB_PATH = OPPORTUNITY_LOG_DB

# Minimum resolved trades for a source to appear in the summary.
_MIN_SAMPLE = env_int("WIN_RATE_MIN_SAMPLE", 3)

# Minimum settled trades per source before the auto-calibrator replaces the
# static KELLY_DEFAULT_P.  Below this threshold the sample is too small to
# trust and the static prior is returned unchanged.
KELLY_CALIBRATION_MIN_SAMPLES: int = int(
    os.environ.get("KELLY_CALIBRATION_MIN_SAMPLES", "10")
)

# Default fallback P(win) used in blending when the source has no configured prior.
# Imported by TradeExecutor to stay in sync.
KELLY_DEFAULT_P_FALLBACK: float = env_float("KELLY_DEFAULT_P", 0.6)


class WinRateTracker:
    """Settle resolved trades and compute per-source win rates.

    Usage::

        tracker = WinRateTracker()
        # call every WIN_RATE_REPORT_INTERVAL cycles:
        await tracker.settle_and_report(session)
        tracker.close()
    """

    def __init__(
        self,
        db_path: Path | str = _DEFAULT_DB_PATH,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._conn = conn if conn is not None else open_db(self._db_path)

    # -----------------------------------------------------------------------
    # Settlement pass
    # -----------------------------------------------------------------------

    async def settle_and_report(self, session: aiohttp.ClientSession) -> None:
        """Fetch outcomes for unresolved trades, then print the win-rate summary."""
        settled = await self._settle_resolved_trades(session)
        if settled:
            logging.info("Win-rate tracker: settled %d trade(s).", settled)
        self._log_summary()

    async def _settle_resolved_trades(self, session: aiohttp.ClientSession) -> int:
        """Check Kalshi API for outcomes on unsettled trades.

        Returns the number of trades newly settled this pass.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT ticker FROM trades WHERE outcome IS NULL"
        ).fetchall()

        if not rows:
            return 0

        tickers = [r[0] for r in rows]
        settled = 0

        for ticker in tickers:
            try:
                detail = await fetch_market_detail(session, ticker)
            except Exception as exc:
                logging.debug("Win-rate: market detail fetch failed for %s: %s", ticker, exc)
                continue

            if not detail:
                continue

            result = (detail.get("result") or "").lower()
            if result not in ("yes", "no", "void"):
                continue  # still open

            # Update all unresolved trades for this ticker
            rows_for_ticker = self._conn.execute(
                "SELECT id, side, source, logged_at, note FROM trades WHERE ticker = ? AND outcome IS NULL",
                (ticker,),
            ).fetchall()

            for trade_id, side, trade_source, logged_at, trade_note in rows_for_ticker:
                if result == "void":
                    outcome = "void"
                elif side == result:
                    outcome = "won"
                else:
                    outcome = "lost"

                self._conn.execute(
                    "UPDATE trades SET outcome = ? WHERE id = ?",
                    (outcome, trade_id),
                )
                settled += 1
                logging.debug(
                    "Settled trade #%d  ticker=%s  side=%s  result=%s  outcome=%s",
                    trade_id, ticker, side, result, outcome,
                )
                self._record_forecast_accuracy(
                    trade_id, ticker, trade_source, logged_at, trade_note, outcome,
                )

        return settled

    # -----------------------------------------------------------------------
    # Forecast accuracy recording
    # -----------------------------------------------------------------------

    def _record_forecast_accuracy(
        self,
        trade_id: int,
        ticker: str,
        trade_source: str | None,
        logged_at: str | None,
        trade_note: str | None,
        outcome: str,
    ) -> None:
        """Insert forecast_accuracy rows for a newly settled temperature trade.

        One row per forecast source (uses sources_detail from trade note for
        forecast_no multi-source trades, or the primary source otherwise).
        Ground truth is the METAR 6-hour synoptic max (metar_6hr source in
        raw_forecasts) with fallback to METAR running max.
        """
        prefix = ticker.split("-")[0]
        metric = TICKER_TO_METRIC.get(prefix)
        if metric is None or not metric.startswith(("temp_high_", "temp_low_")):
            return
        city = metric.split("_")[-1]

        trade_date = (logged_at or "")[:10]
        if not trade_date:
            return

        # Prefer 6-hour synoptic max; fall back to hourly running max
        row = self._conn.execute(
            "SELECT MAX(data_value) FROM raw_forecasts "
            "WHERE source = 'metar_6hr' AND metric = ? AND date(logged_at) = ?",
            (metric, trade_date),
        ).fetchone()
        actual_f = row[0] if row else None
        actual_src = "metar_6hr"

        if actual_f is None:
            row = self._conn.execute(
                "SELECT MAX(data_value) FROM raw_forecasts "
                "WHERE source = 'metar' AND metric = ? AND date(logged_at) = ?",
                (metric, trade_date),
            ).fetchone()
            actual_f = row[0] if row else None
            actual_src = "metar_running_max"

        if actual_f is None:
            logging.debug(
                "forecast_accuracy: no METAR ground truth for %s on %s — skipping",
                metric, trade_date,
            )
            return

        note = json.loads(trade_note or "{}")
        sources_detail = note.get("sources_detail", [])
        hours_to_close = note.get("hours_to_close")

        if sources_detail:
            rows_to_insert = [(src, float(val)) for src, val, *_ in sources_detail]
        else:
            forecast_f = note.get("data_value")
            rows_to_insert = [(trade_source or "unknown", forecast_f)] if forecast_f is not None else []

        if not rows_to_insert:
            return

        now_str = datetime.now(timezone.utc).isoformat()
        for src, forecast_f in rows_to_insert:
            bias_f = round(forecast_f - actual_f, 3) if forecast_f is not None else None
            self._conn.execute(
                """INSERT INTO forecast_accuracy
                   (settled_at, trade_id, source, metric, city,
                    forecast_f, actual_f, actual_src, bias_f, hours_to_close, outcome)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now_str, trade_id, src, metric, city,
                 forecast_f, actual_f, actual_src, bias_f, hours_to_close, outcome),
            )
            logging.debug(
                "forecast_accuracy: trade #%d  source=%s  forecast=%.1f  actual=%.1f  bias=%.1f",
                trade_id, src, forecast_f or 0, actual_f, bias_f or 0,
            )

    # -----------------------------------------------------------------------
    # Win-rate summary
    # -----------------------------------------------------------------------

    def _log_summary(self) -> None:
        """Query and log per-source win rates for all sources with enough data."""
        rows = self._conn.execute(
            """
            SELECT
                COALESCE(source, 'unknown') AS src,
                COUNT(*) AS total,
                SUM(CASE
                    WHEN exit_pnl_cents IS NOT NULL AND exit_pnl_cents > 0 THEN 1
                    WHEN exit_pnl_cents IS NULL AND outcome = 'won'        THEN 1
                    ELSE 0 END) AS wins,
                SUM(CASE
                    WHEN exit_pnl_cents IS NOT NULL AND exit_pnl_cents < 0 THEN 1
                    WHEN exit_pnl_cents IS NULL AND outcome = 'lost'       THEN 1
                    ELSE 0 END) AS losses,
                SUM(CASE WHEN outcome = 'void' THEN 1 ELSE 0 END) AS voids,
                SUM(CASE
                    WHEN exit_pnl_cents IS NULL AND outcome IS NULL THEN 1
                    ELSE 0 END) AS pending
            FROM trades
            GROUP BY src
            ORDER BY total DESC
            """
        ).fetchall()

        if not rows:
            logging.info("Win-rate summary: no trades recorded yet.")
            return

        lines = ["WIN-RATE SUMMARY (per source):"]

        for src, total, wins, losses, voids, pending in rows:
            resolved = wins + losses
            if resolved < _MIN_SAMPLE:
                lines.append(
                    f"  {src:<20}  {resolved}/{total} resolved  "
                    f"(need {_MIN_SAMPLE - resolved} more for stats)"
                )
                continue
            win_rate = wins / resolved
            edge_label = _edge_label(win_rate)
            void_str = f"  {voids}v" if voids else ""
            lines.append(
                f"  {src:<20}  {wins}W/{losses}L{void_str}  "
                f"({win_rate:.1%})  {edge_label}  [{pending} pending]"
            )

        logging.info("\n".join(lines))

    def get_calibrated_priors(
        self, min_samples: int = KELLY_CALIBRATION_MIN_SAMPLES
    ) -> dict[str, float]:
        """Return source → blended P(win) for sources with enough settled data.

        Blending formula (confidence-weighted):
            confidence = clamp((n - min_samples) / (100 - min_samples), 0, 1)
            calibrated  = confidence × win_rate + (1 - confidence) × fallback_p

        This means:
          - At exactly ``min_samples`` resolved trades → 0% data weight (pure prior).
          - At 100 resolved trades → 100% data weight (pure observed win rate).
          - In between → linear interpolation.

        Only sources with at least ``min_samples`` resolved (won+lost) trades are
        included.  Sources below the threshold are omitted so the caller can fall
        back to the static prior unchanged.

        Returns:
            Dict mapping source name (e.g. ``"noaa_observed"``) to blended P(win).
        """
        # Use exit_pnl_cents as primary P&L ground truth (consistent with
        # circuit-breaker logic): positive pnl → won, negative → lost.
        # For settled-only trades with no early exit, fall back to outcome.
        rows = self._conn.execute(
            """
            SELECT
                COALESCE(source, 'unknown') AS src,
                SUM(CASE
                    WHEN exit_pnl_cents IS NOT NULL AND exit_pnl_cents > 0 THEN 1
                    WHEN exit_pnl_cents IS NULL AND outcome = 'won'        THEN 1
                    ELSE 0 END) AS wins,
                SUM(CASE
                    WHEN exit_pnl_cents IS NOT NULL AND exit_pnl_cents < 0 THEN 1
                    WHEN exit_pnl_cents IS NULL AND outcome = 'lost'       THEN 1
                    ELSE 0 END) AS losses
            FROM trades
            WHERE (exit_pnl_cents IS NOT NULL OR outcome IN ('won', 'lost'))
            GROUP BY src
            """
        ).fetchall()

        result: dict[str, float] = {}
        for src, wins, losses in rows:
            n = wins + losses
            if n < min_samples:
                continue
            win_rate = wins / n
            # confidence grows linearly from 0 at min_samples to 1 at 100 samples
            confidence = min(1.0, (n - min_samples) / max(1, 100 - min_samples))
            blended = confidence * win_rate + (1.0 - confidence) * KELLY_DEFAULT_P_FALLBACK
            result[src] = round(blended, 4)
            logging.debug(
                "Calibrated prior: source=%s  n=%d  win_rate=%.1%%  "
                "confidence=%.2f  blended_p=%.4f",
                src, n, win_rate * 100, confidence, blended,
            )

        return result

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


def _edge_label(win_rate: float) -> str:
    """Return a human-readable edge strength label."""
    if win_rate >= 0.70:
        return "STRONG EDGE"
    if win_rate >= 0.60:
        return "good edge"
    if win_rate >= 0.52:
        return "marginal edge"
    if win_rate >= 0.48:
        return "breakeven (~no edge)"
    return "NEGATIVE EDGE — review"
