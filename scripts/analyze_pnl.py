#!/usr/bin/env python3
"""Dry-run P&L analyzer for the Kalshi information-alpha bot.

Reads every dry_run trade from opportunity_log.db, fetches the current
market state from the Kalshi API, computes realized and unrealized P&L,
then writes a human-readable report and a CSV for further analysis.

P&L accounting
--------------
All Kalshi contracts pay 100¢ to the winning side and 0¢ to the loser.
The bot stores `yes_price` in `limit_price` for both sides, so the cost
basis and payoff are:

  YES buy  (limit_price = yes_ask):
    cost       =  limit_price cents per contract
    gain (YES) =  100 − limit_price   cents per contract
    loss (NO)  = −limit_price         cents per contract

  NO buy  (limit_price = yes_bid, actual NO cost = 100 − yes_bid):
    cost       =  100 − limit_price   cents per contract
    gain (NO)  =  limit_price         cents per contract
    loss (YES) =  limit_price − 100   cents per contract

Unrealized (mark-to-market, mid = (bid+ask)/2):
    YES trade:   current_mid − limit_price    cents per contract
    NO trade:    limit_price − current_mid    cents per contract

Usage
-----
    venv/bin/python analyze_pnl.py
    venv/bin/python analyze_pnl.py --output report.txt --csv trades.csv
    venv/bin/python analyze_pnl.py --since 2026-03-01
    venv/bin/python analyze_pnl.py --min-score 0.6 --stdout
"""

import argparse
import asyncio
import csv
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

# Allow importing kalshi_bot from the project root
sys.path.insert(0, str(Path(__file__).parent))
from kalshi_bot.auth import generate_headers  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DB_PATH        = Path(__file__).parent / "opportunity_log.db"
_DEFAULT_REPORT = Path(__file__).parent / "pnl_report.txt"
_DEFAULT_CSV    = Path(__file__).parent / "pnl_trades.csv"

_MARKETS_BASE_PATH = "/trade-api/v2/markets"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    trade_id:         int
    logged_at:        str
    ticker:           str
    side:             str    # "yes" | "no"
    count:            int
    limit_price:      int    # yes_price in cents
    score:            float
    opportunity_kind: str
    kelly_fraction:   float | None = None
    p_estimate:       float | None = None

    # Populated after API fetch
    settled:     bool         = False
    result:      str | None   = None   # "yes" | "no" | None (still open)
    current_bid: int | None   = None
    current_ask: int | None   = None

    @property
    def current_mid(self) -> float | None:
        if self.current_bid is not None and self.current_ask is not None:
            return (self.current_bid + self.current_ask) / 2.0
        return None

    @property
    def cost_per_contract(self) -> int:
        """What the bot would have paid per contract in cents."""
        return self.limit_price if self.side == "yes" else (100 - self.limit_price)

    @property
    def pnl_cents(self) -> float | None:
        """Total P&L for this trade in cents, or None if unknown."""
        if self.settled and self.result is not None:
            return self._realized_cents()
        if self.current_mid is not None:
            return self._unrealized_cents()
        return None

    @property
    def pnl_dollars(self) -> float | None:
        c = self.pnl_cents
        return c / 100.0 if c is not None else None

    @property
    def pnl_status(self) -> str:
        if self.settled and self.result is not None:
            if self._realized_cents() > 0:
                return "win"
            if self._realized_cents() < 0:
                return "loss"
            return "breakeven"
        if self.current_mid is not None:
            return "open"
        return "unknown"

    def _realized_cents(self) -> float:
        if self.side == "yes":
            per = (100 - self.limit_price) if self.result == "yes" else (-self.limit_price)
        else:  # "no"
            per = self.limit_price if self.result == "no" else (self.limit_price - 100)
        return per * self.count

    def _unrealized_cents(self) -> float:
        mid = self.current_mid  # already checked not None
        if self.side == "yes":
            per = mid - self.limit_price
        else:
            per = self.limit_price - mid
        return per * self.count


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def load_trades(db_path: Path, since: str | None, min_score: float) -> list[TradeRecord]:
    """Load dry_run trades from the trades table."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    try:
        query = """
            SELECT id, logged_at, ticker, side, count, limit_price, score, opportunity_kind,
                   kelly_fraction, p_estimate
            FROM trades
            WHERE mode = 'dry_run'
        """
        params: list[Any] = []
        if since:
            query += " AND logged_at >= ?"
            params.append(since)
        if min_score > 0:
            query += " AND score >= ?"
            params.append(min_score)
        query += " ORDER BY logged_at ASC"

        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    return [
        TradeRecord(
            trade_id         = row[0],
            logged_at        = row[1],
            ticker           = row[2],
            side             = row[3],
            count            = row[4],
            limit_price      = row[5],
            score            = row[6],
            opportunity_kind = row[7],
            kelly_fraction   = row[8],
            p_estimate       = row[9],
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Kalshi API fetch
# ---------------------------------------------------------------------------

def _api_base() -> str:
    return (
        "https://trading-api.kalshi.com"
        if os.environ.get("KALSHI_ENVIRONMENT", "demo") == "production"
        else "https://demo-api.kalshi.co"
    )


async def _fetch_one(
    session: aiohttp.ClientSession, ticker: str
) -> dict[str, Any] | None:
    path = f"{_MARKETS_BASE_PATH}/{ticker}"
    headers = generate_headers("GET", path)
    try:
        async with session.get(
            f"{_api_base()}{path}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("market")
    except Exception:
        return None


async def enrich_trades(trades: list[TradeRecord]) -> None:
    """Fetch market state for all unique tickers and attach to trades in-place."""
    unique_tickers = list({t.ticker for t in trades})

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = await asyncio.gather(
            *[_fetch_one(session, ticker) for ticker in unique_tickers],
            return_exceptions=True,
        )

    market_by_ticker: dict[str, dict] = {}
    for ticker, result in zip(unique_tickers, results):
        if isinstance(result, dict):
            market_by_ticker[ticker] = result

    for trade in trades:
        mkt = market_by_ticker.get(trade.ticker)
        if mkt is None:
            continue

        status = mkt.get("status", "")
        result_field = mkt.get("result", "")  # "yes", "no", or "" / None

        if status in ("settled", "finalized") and result_field in ("yes", "no"):
            trade.settled = True
            trade.result  = result_field
        else:
            trade.current_bid = mkt.get("yes_bid")
            trade.current_ask = mkt.get("yes_ask")


# ---------------------------------------------------------------------------
# P&L aggregation helpers
# ---------------------------------------------------------------------------

def _fmt_dollars(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}${value:.2f}"


def _pnl_by_metric(trades: list[TradeRecord]) -> dict[str, dict]:
    """Group P&L totals by metric prefix (first component of ticker or opportunity_kind)."""
    groups: dict[str, dict] = {}

    for t in trades:
        # Use ticker prefix up to first '-' as the grouping key
        key = t.ticker.split("-")[0]
        if key not in groups:
            groups[key] = {"trades": 0, "wins": 0, "losses": 0,
                           "realized_cents": 0.0, "unrealized_cents": 0.0}
        g = groups[key]
        g["trades"] += 1

        if t.settled and t.result is not None:
            r = t._realized_cents()
            g["realized_cents"] += r
            if r > 0:
                g["wins"] += 1
            elif r < 0:
                g["losses"] += 1
        elif t.current_mid is not None:
            g["unrealized_cents"] += t._unrealized_cents()

    return groups


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(trades: list[TradeRecord], generated_at: str) -> str:
    buf: list[str] = []
    W = "=" * 68
    S = "-" * 68

    settled  = [t for t in trades if t.settled and t.result is not None]
    open_    = [t for t in trades if not t.settled and t.current_mid is not None]
    unknown  = [t for t in trades if t.pnl_cents is None]

    wins   = [t for t in settled if t._realized_cents() > 0]
    losses = [t for t in settled if t._realized_cents() < 0]

    realized_cents   = sum(t._realized_cents()   for t in settled)
    unrealized_cents = sum(t._unrealized_cents()  for t in open_)
    total_cents      = realized_cents + unrealized_cents

    win_rate = (len(wins) / len(settled) * 100) if settled else 0.0

    buf.append(W)
    buf.append("  DRY-RUN P&L REPORT")
    buf.append(f"  Generated: {generated_at}")
    buf.append(W)
    buf.append(f"  Total trades analyzed : {len(trades)}")
    buf.append(f"  Settled (realized)    : {len(settled)}")
    if settled:
        buf.append(f"    Win rate            : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    buf.append(f"  Open (unrealized)     : {len(open_)}")
    buf.append(f"  Unknown / no data     : {len(unknown)}")
    buf.append(S)
    buf.append(f"  Realized P&L          : {_fmt_dollars(realized_cents / 100)}")
    buf.append(f"  Unrealized P&L (mid)  : {_fmt_dollars(unrealized_cents / 100)}")
    buf.append(f"  Total P&L             : {_fmt_dollars(total_cents / 100)}")

    # ---- per-ticker-prefix breakdown ----------------------------------------
    groups = _pnl_by_metric(trades)
    if groups:
        buf.append(S)
        buf.append("  BY TICKER PREFIX")
        buf.append(S)
        buf.append(f"  {'Prefix':<22}  {'Trades':>6}  {'W/L':>7}  {'Realized':>10}  {'Unreal.':>10}")
        buf.append(f"  {'-'*22}  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*10}")
        for key in sorted(groups, key=lambda k: -groups[k]["realized_cents"]):
            g = groups[key]
            wl = f"{g['wins']}W/{g['losses']}L"
            buf.append(
                f"  {key:<22}  {g['trades']:>6}  {wl:>7}"
                f"  {_fmt_dollars(g['realized_cents'] / 100):>10}"
                f"  {_fmt_dollars(g['unrealized_cents'] / 100):>10}"
            )

    # ---- settled trades ------------------------------------------------------
    if settled:
        buf.append(S)
        buf.append("  SETTLED TRADES (REALIZED)")
        buf.append(S)
        buf.append(
            f"  {'Date':<12}  {'Ticker':<30}  {'Side':<4}  "
            f"{'Cost':>5}  {'Res':>4}  {'P&L':>8}  {'Score':>5}"
        )
        buf.append(
            f"  {'-'*12}  {'-'*30}  {'-'*4}  "
            f"{'-'*5}  {'-'*4}  {'-'*8}  {'-'*5}"
        )
        for t in sorted(settled, key=lambda x: x.logged_at):
            date   = t.logged_at[:10]
            side   = t.side.upper()
            cost   = f"{t.cost_per_contract}¢"
            result = (t.result or "").upper()
            pnl    = _fmt_dollars(t._realized_cents() / 100)
            buf.append(
                f"  {date:<12}  {t.ticker:<30}  {side:<4}  "
                f"{cost:>5}  {result:>4}  {pnl:>8}  {t.score:.2f}"
            )

    # ---- open trades --------------------------------------------------------
    if open_:
        buf.append(S)
        buf.append("  OPEN TRADES (MARK-TO-MARKET)")
        buf.append(S)
        buf.append(
            f"  {'Date':<12}  {'Ticker':<30}  {'Side':<4}  "
            f"{'Entry':>6}  {'Mid':>5}  {'P&L':>8}  {'Score':>5}"
        )
        buf.append(
            f"  {'-'*12}  {'-'*30}  {'-'*4}  "
            f"{'-'*6}  {'-'*5}  {'-'*8}  {'-'*5}"
        )
        for t in sorted(open_, key=lambda x: x.logged_at):
            date  = t.logged_at[:10]
            side  = t.side.upper()
            entry = f"{t.limit_price}¢"
            mid   = f"{t.current_mid:.0f}¢" if t.current_mid is not None else "N/A"
            pnl   = _fmt_dollars(t._unrealized_cents() / 100)
            buf.append(
                f"  {date:<12}  {t.ticker:<30}  {side:<4}  "
                f"{entry:>6}  {mid:>5}  {pnl:>8}  {t.score:.2f}"
            )

    # ---- unknown ------------------------------------------------------------
    if unknown:
        buf.append(S)
        buf.append(f"  {len(unknown)} trade(s) could not be priced (market data unavailable)")

    buf.append(W)
    return "\n".join(buf)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(trades: list[TradeRecord], csv_path: Path) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trade_id", "logged_at", "ticker", "side", "count",
            "limit_price_cents", "cost_per_contract_cents", "score",
            "kelly_fraction", "p_estimate",
            "opportunity_kind", "settled", "result",
            "current_mid_cents", "pnl_cents", "pnl_dollars", "pnl_status",
        ])
        for t in trades:
            writer.writerow([
                t.trade_id,
                t.logged_at,
                t.ticker,
                t.side,
                t.count,
                t.limit_price,
                t.cost_per_contract,
                round(t.score, 4),
                round(t.kelly_fraction, 4) if t.kelly_fraction is not None else "",
                round(t.p_estimate, 4) if t.p_estimate is not None else "",
                t.opportunity_kind,
                t.settled,
                t.result or "",
                round(t.current_mid, 1) if t.current_mid is not None else "",
                round(t.pnl_cents, 2) if t.pnl_cents is not None else "",
                round(t.pnl_dollars, 4) if t.pnl_dollars is not None else "",
                t.pnl_status,
            ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze P&L for Kalshi bot dry-run trades."
    )
    p.add_argument(
        "--output", "-o",
        default=str(_DEFAULT_REPORT),
        help=f"Text report output path (default: {_DEFAULT_REPORT})",
    )
    p.add_argument(
        "--csv", "-c",
        default=str(_DEFAULT_CSV),
        help=f"CSV output path (default: {_DEFAULT_CSV})",
    )
    p.add_argument(
        "--since",
        default=None,
        metavar="DATE",
        help="Only analyze trades on or after this date (ISO format, e.g. 2026-03-01)",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        metavar="SCORE",
        help="Only analyze trades with score >= this value (default: 0.0 = all)",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the report to stdout",
    )
    return p.parse_args()


async def _main() -> None:
    args = parse_args()

    if not _DB_PATH.exists():
        print(f"Database not found: {_DB_PATH}")
        print("Run the bot in dry-run mode first to accumulate trade data.")
        sys.exit(1)

    print(f"Loading trades from {_DB_PATH} …")
    trades = load_trades(_DB_PATH, since=args.since, min_score=args.min_score)

    if not trades:
        print("No dry_run trades found matching the given filters.")
        sys.exit(0)

    print(f"Found {len(trades)} trade(s) — fetching market states from Kalshi API …")
    await enrich_trades(trades)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report = build_report(trades, generated_at)

    # Write text report
    report_path = Path(args.output)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"Report written to: {report_path}")

    # Write CSV
    csv_path = Path(args.csv)
    write_csv(trades, csv_path)
    print(f"CSV written to:    {csv_path}")

    # Optionally print to stdout
    if args.stdout:
        print()
        print(report)


if __name__ == "__main__":
    asyncio.run(_main())
