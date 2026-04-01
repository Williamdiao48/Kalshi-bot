"""P&L Attribution Dashboard.

Reads from the trades SQLite database and writes a human-readable breakdown
of realized P&L and win rates across four dimensions:

  - By metric category  (temp_high, price_btc, bls_nfp, ...)
  - By signal source    (noaa_observed, binance, open_meteo, ...)
  - By direction        (YES vs NO contract)
  - By days-to-close at entry  (0–1, 1–7, 7–30, >30 day buckets)

Only resolved trades are included (settled won/lost + early exits).
Open and pending positions are excluded.

Usage
-----
  # From the project root, run manually:
  venv/bin/python -m kalshi_bot.analytics

  # Or call from main.py after settlement:
  from .analytics import run_attribution
  run_attribution(db_path, output_path, lookback_days=30)

Output file
-----------
  pnl_attribution.txt  in the project root (default).
  Override with PNL_ATTRIBUTION_PATH env var.

Environment variables
---------------------
  ANALYTICS_LOOKBACK_DAYS   Integer days of history to include.
                            Default: 30.  Set to 0 for all-time.
  ANALYTICS_MODE            'dry_run' | 'live' | 'both'.  Default: dry_run.
  PNL_ATTRIBUTION_PATH      Output file path.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .market_parser import TICKER_TO_METRIC

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH  = Path(__file__).parent.parent / "opportunity_log.db"
_DEFAULT_OUT_PATH = Path(__file__).parent.parent / "pnl_attribution.txt"

ANALYTICS_LOOKBACK_DAYS: int = int(os.environ.get("ANALYTICS_LOOKBACK_DAYS", "30"))
ANALYTICS_MODE: str = os.environ.get("ANALYTICS_MODE", "dry_run")
PNL_ATTRIBUTION_PATH: Path = Path(
    os.environ.get("PNL_ATTRIBUTION_PATH", str(_DEFAULT_OUT_PATH))
)


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

@dataclass
class _ResolvedTrade:
    trade_id:    int
    logged_at:   str
    ticker:      str
    side:        str       # 'yes' | 'no'
    count:       int
    limit_price: int       # YES price in cents
    source:      str
    outcome:     str | None   # 'won' | 'lost' | 'void' | None
    exit_pnl:    float | None # signed cents if exited early
    exit_reason: str | None   # 'profit_take' | 'stop_loss' | 'trailing_stop'
    initial_dtc: float | None # days-to-close at first snapshot (proxy for entry DTC)

    @property
    def pnl_cents(self) -> float:
        """Signed realized P&L in cents."""
        if self.exit_pnl is not None:
            return self.exit_pnl
        if self.outcome == "won":
            if self.side == "yes":
                return (100 - self.limit_price) * self.count
            else:
                return self.limit_price * self.count
        if self.outcome == "lost":
            if self.side == "yes":
                return -self.limit_price * self.count
            else:
                return -(100 - self.limit_price) * self.count
        return 0.0

    @property
    def is_win(self) -> bool:
        if self.exit_pnl is not None:
            return self.exit_pnl > 0
        return self.outcome == "won"

    @property
    def exit_type(self) -> str:
        """Human-readable resolution type."""
        if self.exit_reason == "profit_take":
            return "profit_take"
        if self.exit_reason in ("stop_loss", "trailing_stop"):
            return self.exit_reason
        if self.outcome == "won":
            return "settled_win"
        if self.outcome == "lost":
            return "settled_loss"
        return "other"


@dataclass
class _Bucket:
    pnl_cents: float = 0.0
    wins:      int   = 0
    losses:    int   = 0

    def add(self, trade: _ResolvedTrade) -> None:
        self.pnl_cents += trade.pnl_cents
        if trade.is_win:
            self.wins  += 1
        else:
            self.losses += 1

    @property
    def total(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.total if self.total > 0 else None


# ---------------------------------------------------------------------------
# Metric grouping
# ---------------------------------------------------------------------------

def _metric_for_ticker(ticker: str) -> str:
    """Return the canonical metric key for a ticker, or 'unknown'."""
    for prefix, metric in TICKER_TO_METRIC.items():
        if ticker.startswith(prefix):
            return metric
    return "unknown"


def _metric_group(metric: str) -> str:
    """Collapse city/coin/currency suffixes to a coarser group label.

    Examples:
      temp_high_lax  → temp_high
      price_btc_usd  → price_btc
      rate_eur_usd   → rate_eur
      bls_nfp        → bls_nfp   (unchanged)
    """
    parts = metric.split("_")
    # Three-segment metrics where first token is a broad category:
    if len(parts) >= 3 and parts[0] in ("temp", "price", "rate"):
        return "_".join(parts[:2])
    return metric


def _dtc_bucket(dtc: float | None) -> str:
    """Map initial days-to-close to a display bucket label."""
    if dtc is None:
        return "unknown"
    if dtc <= 1.0:
        return "0–1 days"
    if dtc <= 7.0:
        return "1–7 days"
    if dtc <= 30.0:
        return "7–30 days"
    return ">30 days"


_DTC_ORDER = ["0–1 days", "1–7 days", "7–30 days", ">30 days", "unknown"]


# ---------------------------------------------------------------------------
# Database query
# ---------------------------------------------------------------------------

def _load_trades(
    conn: sqlite3.Connection,
    mode: str,
    cutoff: str | None,
) -> list[_ResolvedTrade]:
    """Load all resolved trades from the DB, with their initial DTC."""
    mode_clause = (
        "t.mode IN ('dry_run', 'live')"
        if mode == "both"
        else f"t.mode = '{mode}'"
    )
    cutoff_clause = f"AND t.logged_at >= '{cutoff}'" if cutoff else ""

    sql = f"""
    SELECT
        t.id,
        t.logged_at,
        t.ticker,
        t.side,
        t.count,
        t.limit_price,
        COALESCE(t.source, 'unknown')  AS source,
        t.outcome,
        t.exit_pnl_cents,
        t.exit_reason,
        MAX(ps.days_to_close)          AS initial_dtc
    FROM trades t
    LEFT JOIN price_snapshots ps ON ps.trade_id = t.id
    WHERE {mode_clause}
      AND (
            t.outcome IN ('won', 'lost')
            OR t.exited_at IS NOT NULL
      )
      AND t.outcome IS NOT 'void'
      {cutoff_clause}
    GROUP BY t.id
    ORDER BY t.logged_at ASC
    """
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:
        logging.warning("analytics: DB query failed: %s", exc)
        return []

    trades = []
    for row in rows:
        (trade_id, logged_at, ticker, side, count, limit_price,
         source, outcome, exit_pnl, exit_reason, initial_dtc) = row
        trades.append(_ResolvedTrade(
            trade_id    = trade_id,
            logged_at   = logged_at,
            ticker      = ticker,
            side        = side,
            count       = count,
            limit_price = limit_price,
            source      = source,
            outcome     = outcome,
            exit_pnl    = exit_pnl,
            exit_reason = exit_reason,
            initial_dtc = initial_dtc,
        ))
    return trades


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(trades: list[_ResolvedTrade]) -> dict[str, dict[str, _Bucket]]:
    """Return nested dict: dimension → key → Bucket."""
    by_metric: dict[str, _Bucket] = defaultdict(_Bucket)
    by_source:  dict[str, _Bucket] = defaultdict(_Bucket)
    by_side:    dict[str, _Bucket] = defaultdict(_Bucket)
    by_dtc:     dict[str, _Bucket] = defaultdict(_Bucket)
    by_exit:    dict[str, _Bucket] = defaultdict(_Bucket)

    for t in trades:
        metric_key = _metric_group(_metric_for_ticker(t.ticker))
        by_metric[metric_key].add(t)
        by_source[t.source].add(t)
        by_side[t.side.upper()].add(t)
        by_dtc[_dtc_bucket(t.initial_dtc)].add(t)
        by_exit[t.exit_type].add(t)

    return {
        "metric": dict(by_metric),
        "source": dict(by_source),
        "side":   dict(by_side),
        "dtc":    dict(by_dtc),
        "exit":   dict(by_exit),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pnl(cents: float) -> str:
    sign = "+" if cents >= 0 else ""
    return f"{sign}${cents / 100:.2f}"


def _fmt_row(label: str, bucket: _Bucket, *, suffix: str = "", label_width: int = 22) -> str:
    wr = bucket.win_rate
    wr_str = f"{wr * 100:.0f}% win rate" if wr is not None else "no data"
    pnl_str = _fmt_pnl(bucket.pnl_cents)
    note = f"  ← {suffix}" if suffix else ""
    return (
        f"  {label:<{label_width}}  {pnl_str:>9}  "
        f"({bucket.total} trades, {wr_str}){note}"
    )


def _section(
    title: str,
    buckets: dict[str, _Bucket],
    *,
    order: list[str] | None = None,
    top_n: int = 0,
    suffix_fn=None,
) -> list[str]:
    """Render one attribution section."""
    lines = [f"\nBy {title}:"]
    if not buckets:
        lines.append("  (no data)")
        return lines

    # Determine ordering
    if order:
        keys = [k for k in order if k in buckets]
        keys += sorted(k for k in buckets if k not in order)
    else:
        # Sort by P&L descending
        keys = sorted(buckets, key=lambda k: -buckets[k].pnl_cents)

    if top_n:
        keys = keys[:top_n]

    for k in keys:
        b = buckets[k]
        sfx = suffix_fn(k, b) if suffix_fn else ""
        lines.append(_fmt_row(k, b, suffix=sfx))

    return lines


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_attribution(
    db_path: Path | str = _DEFAULT_DB_PATH,
    output_path: Path | str = PNL_ATTRIBUTION_PATH,
    lookback_days: int = ANALYTICS_LOOKBACK_DAYS,
    mode: str = ANALYTICS_MODE,
) -> str:
    """Compute P&L attribution and write it to ``output_path``.

    Returns the report text.  Also writes it to ``output_path`` unless
    ``output_path`` is ``None``.

    Args:
        db_path:       Path to the SQLite DB (default: ``opportunity_log.db``).
        output_path:   Where to write the report.  ``None`` → skip file write.
        lookback_days: How many calendar days of history to include.  0 = all-time.
        mode:          'dry_run' | 'live' | 'both'.
    """
    db_path     = Path(db_path)
    output_path = Path(output_path) if output_path is not None else None

    if not db_path.exists():
        logging.warning("analytics: DB not found at %s", db_path)
        return ""

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        cutoff: str | None = None
        if lookback_days > 0:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=lookback_days)
            ).strftime("%Y-%m-%dT%H:%M:%S")

        trades = _load_trades(conn, mode, cutoff)
    finally:
        conn.close()

    if not trades:
        report = _empty_report(lookback_days, mode)
    else:
        report = _build_report(trades, lookback_days, mode)

    if output_path is not None:
        try:
            output_path.write_text(report + "\n", encoding="utf-8")
            logging.info("analytics: wrote attribution to %s", output_path)
        except OSError as exc:
            logging.error("analytics: could not write attribution file: %s", exc)

    return report


def _empty_report(lookback_days: int, mode: str) -> str:
    window = f"last {lookback_days} days" if lookback_days > 0 else "all time"
    return (
        f"=== P&L Attribution ({window}, mode={mode}) ===\n"
        "  No resolved trades found.\n"
    )


def _build_report(
    trades: list[_ResolvedTrade],
    lookback_days: int,
    mode: str,
) -> str:
    aggs = _aggregate(trades)

    window = f"last {lookback_days} days" if lookback_days > 0 else "all time"
    total_pnl  = sum(t.pnl_cents for t in trades)
    total_wins = sum(1 for t in trades if t.is_win)
    overall_wr = total_wins / len(trades) if trades else 0.0

    lines: list[str] = [
        f"=== P&L Attribution ({window}, mode={mode}) ===",
        f"  Resolved trades : {len(trades)}",
        f"  Overall win rate: {overall_wr * 100:.1f}%",
        f"  Net P&L         : {_fmt_pnl(total_pnl)}",
    ]

    # --- By metric category ---
    lines += _section(
        "metric category",
        aggs["metric"],
        top_n=0,
    )

    # --- By signal source ---
    lines += _section(
        "signal source",
        aggs["source"],
    )

    # --- By direction ---
    lines += _section(
        "direction",
        aggs["side"],
        order=["YES", "NO"],
        suffix_fn=lambda k, b: (
            "most profitable" if b.pnl_cents == max(
                v.pnl_cents for v in aggs["side"].values()
            ) else ""
        ),
    )

    # --- By days-to-close at entry ---
    lines += _section(
        "days-to-close at entry",
        aggs["dtc"],
        order=_DTC_ORDER,
        suffix_fn=lambda k, b: (
            "most profitable" if b.pnl_cents == max(
                v.pnl_cents for v in aggs["dtc"].values()
            ) else ""
        ),
    )

    # --- By exit type ---
    lines += _section(
        "exit type",
        aggs["exit"],
        order=["settled_win", "settled_loss", "profit_take", "stop_loss",
               "trailing_stop", "other"],
    )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    lookback = int(sys.argv[1]) if len(sys.argv) > 1 else ANALYTICS_LOOKBACK_DAYS
    out      = sys.argv[2]    if len(sys.argv) > 2 else str(PNL_ATTRIBUTION_PATH)
    mode_arg = sys.argv[3]    if len(sys.argv) > 3 else ANALYTICS_MODE

    report = run_attribution(
        db_path=_DEFAULT_DB_PATH,
        output_path=out,
        lookback_days=lookback,
        mode=mode_arg,
    )
    print(report)
