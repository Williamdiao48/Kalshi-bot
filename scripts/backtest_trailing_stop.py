"""Backtest trailing stop thresholds on resolved band_arb YES trades.

Uses actual price snapshot data from opportunity_log.db to simulate
how different EXIT_TRAILING_DRAWDOWN thresholds would affect P&L.

Trailing stop logic (mirrors exit_manager.py):
  Phase 1: peak pct_gain must exceed threshold  → stop armed
  Phase 2: current pct_gain drops >= threshold below peak → fire

Spread gate: stop suppressed if yes_ask − yes_bid > MAX_SPREAD_CENTS.
Floor gate:  stop suppressed if yes_bid <= FLOOR_PRICE_CENTS.

Usage:
  venv/bin/python scripts/backtest_trailing_stop.py
  venv/bin/python scripts/backtest_trailing_stop.py --max-spread 20 --floor 2
  venv/bin/python scripts/backtest_trailing_stop.py --thresholds 0.10 0.15 0.20 0.25 0.30
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path("data/db/opportunity_log.db")

DEFAULT_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
DEFAULT_MAX_SPREAD = 15
DEFAULT_FLOOR = 2


@dataclass
class Snap:
    snapshot_at: str
    yes_bid: int | None
    yes_ask: int | None
    post_exit: int


@dataclass
class Trade:
    trade_id: int
    ticker: str
    limit_price: int
    count: int
    outcome: str | None          # 'won' | 'lost' | 'void'
    actual_pnl: float | None     # cents, as recorded
    snaps: list[Snap] = field(default_factory=list)

    @property
    def settlement_pnl(self) -> float:
        """P&L at settlement (100¢ win, 0¢ loss)."""
        if self.outcome == "won":
            return (100 - self.limit_price) * self.count
        elif self.outcome == "lost":
            return -self.limit_price * self.count
        return 0.0  # void


def simulate(
    trade: Trade,
    threshold: float,
    max_spread: int,
    floor: int,
) -> tuple[float, str, int | None]:
    """Simulate trailing stop for one trade.

    Returns (pnl_cents, exit_type, exit_bid).
    exit_type: 'trailing_stop' | 'settlement_won' | 'settlement_lost' | 'settlement_void'
    """
    if threshold <= 0.0:
        label = f"settlement_{trade.outcome or 'void'}"
        return trade.settlement_pnl, label, None

    entry = trade.limit_price
    count = trade.count
    peak_pct: float = -999.0
    armed = False

    for snap in trade.snaps:
        bid = snap.yes_bid
        ask = snap.yes_ask
        if bid is None:
            continue

        pct = (bid - entry) / entry

        if pct > peak_pct:
            peak_pct = pct

        if not armed and peak_pct >= threshold:
            armed = True

        if armed:
            if bid <= floor:
                continue
            spread = (ask or 100) - bid
            if spread > max_spread:
                continue
            if (peak_pct - pct) >= threshold:
                return (bid - entry) * count, "trailing_stop", bid

    label = f"settlement_{trade.outcome or 'void'}"
    return trade.settlement_pnl, label, None


@dataclass
class Stats:
    n: int = 0
    stops_fired: int = 0
    total_pnl: float = 0.0
    stop_pnl: float = 0.0
    sett_pnl: float = 0.0

    def add(self, pnl: float, exit_type: str) -> None:
        self.n += 1
        self.total_pnl += pnl
        if exit_type == "trailing_stop":
            self.stops_fired += 1
            self.stop_pnl += pnl
        else:
            self.sett_pnl += pnl

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.n if self.n else 0.0


def load_trades(db_path: Path) -> list[Trade]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT id, ticker, limit_price, count, outcome, exit_pnl_cents
        FROM trades
        WHERE opportunity_kind = 'band_arb'
          AND side = 'yes'
          AND outcome IS NOT NULL
          AND outcome != 'void'
        ORDER BY id
    """).fetchall()

    trades: list[Trade] = []
    for row in rows:
        trade_id, ticker, limit_price, count, outcome, actual_pnl = row
        snaps_raw = conn.execute("""
            SELECT snapshot_at, yes_bid, yes_ask, post_exit
            FROM price_snapshots
            WHERE trade_id = ?
            ORDER BY snapshot_at
        """, (trade_id,)).fetchall()
        snaps = [Snap(r[0], r[1], r[2], r[3]) for r in snaps_raw]
        trades.append(Trade(trade_id, ticker, limit_price, count, outcome, actual_pnl, snaps))

    conn.close()
    return trades


def per_trade_detail(
    trades: list[Trade],
    threshold: float,
    baseline_pnls: dict[int, float],
    max_spread: int,
    floor: int,
) -> None:
    """Print per-trade detail for a given threshold — only trades where outcome differs."""
    changed: list[tuple[Trade, float, str, int | None]] = []
    for t in trades:
        pnl, exit_type, exit_bid = simulate(t, threshold, max_spread, floor)
        base = baseline_pnls[t.trade_id]
        if abs(pnl - base) > 1.0:  # changed by more than 1¢
            changed.append((t, pnl, exit_type, exit_bid))

    if not changed:
        print("  No trades changed vs no-stop baseline.")
        return

    for t, pnl, exit_type, exit_bid in changed:
        base = baseline_pnls[t.trade_id]
        delta = pnl - base
        sign = "+" if delta >= 0 else ""
        stop_at = f" (exit bid={exit_bid}¢)" if exit_bid else ""
        print(
            f"  #{t.trade_id:3d} {t.ticker:<32s} outcome={t.outcome:4s}"
            f"  entry={t.limit_price}¢×{t.count}"
            f"  base=${base/100:+.2f}  sim=${pnl/100:+.2f}"
            f"  delta={sign}${delta/100:.2f}"
            f"  [{exit_type}{stop_at}]"
        )


def main(
    thresholds: list[float],
    max_spread: int,
    floor: int,
    detail_threshold: float | None,
    db_path: Path,
) -> None:
    trades = load_trades(db_path)
    if not trades:
        print("No resolved band_arb YES trades found.")
        return

    won = sum(1 for t in trades if t.outcome == "won")
    lost = sum(1 for t in trades if t.outcome == "lost")
    with_snaps = sum(1 for t in trades if t.snaps)
    print(f"\nBand-Arb YES Trailing Stop Backtest")
    print(f"{'='*70}")
    print(f"Trades: {len(trades)} resolved  ({won} won, {lost} lost)  |  {with_snaps} have price snapshots")
    print(f"Spread gate: >{max_spread}¢  |  Floor gate: ≤{floor}¢\n")

    # Baseline: no stop
    base_stats = Stats()
    baseline_pnls: dict[int, float] = {}
    for t in trades:
        pnl = t.settlement_pnl
        baseline_pnls[t.trade_id] = pnl
        base_stats.add(pnl, "settlement")

    # Header
    col = "{:<8s}  {:>6s}  {:>6s}  {:>6s}  {:>9s}  {:>8s}  {:>8s}"
    print(col.format("Thresh", "Stops", "Avg¢", "Avg$", "Total$", "vs Base$", "vs Base%"))
    print("-" * 70)

    def row(label: str, s: Stats, base_total: float) -> None:
        delta = s.total_pnl - base_total
        base_pct = (delta / abs(base_total) * 100) if base_total else 0.0
        sign = "+" if delta >= 0 else ""
        print(
            f"{label:<8s}  {s.stops_fired:>6d}  "
            f"{s.avg_pnl:>+6.0f}¢  "
            f"${s.avg_pnl/100:>+6.2f}  "
            f"${s.total_pnl/100:>+8.2f}  "
            f"{sign}${delta/100:>7.2f}  "
            f"{sign}{base_pct:>6.1f}%"
        )

    row("no stop", base_stats, base_stats.total_pnl)

    results: dict[float, Stats] = {}
    for thresh in thresholds:
        s = Stats()
        for t in trades:
            pnl, exit_type, _ = simulate(t, thresh, max_spread, floor)
            s.add(pnl, exit_type)
        results[thresh] = s
        row(f"{thresh:.2f}", s, base_stats.total_pnl)

    # Per-trade detail for requested threshold
    if detail_threshold is not None and detail_threshold > 0:
        print(f"\nPer-trade changes at threshold={detail_threshold:.2f}:")
        per_trade_detail(trades, detail_threshold, baseline_pnls, max_spread, floor)

    # Show the worst drawdowns seen in winning trades (to illustrate false-stop risk)
    print(f"\nWinning trades — worst drawdown from peak seen in snapshots:")
    print(f"  {'ID':>4s}  {'Ticker':<32s}  {'Entry':>5s}  {'PeakBid':>7s}  {'MinBid':>6s}  {'MaxDrop%':>8s}")
    for t in trades:
        if t.outcome != "won" or not t.snaps:
            continue
        entry = t.limit_price
        peak_bid = max((s.yes_bid for s in t.snaps if s.yes_bid), default=None)
        min_bid = min((s.yes_bid for s in t.snaps if s.yes_bid), default=None)
        if peak_bid is None or min_bid is None:
            continue
        peak_pct = (peak_bid - entry) / entry
        min_pct_from_peak = (min_bid - peak_bid) / peak_bid * 100
        print(
            f"  #{t.trade_id:>3d}  {t.ticker:<32s}  {entry:>5d}¢"
            f"  {peak_bid:>7d}¢  {min_bid:>6d}¢  {min_pct_from_peak:>+7.1f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest trailing stop thresholds on band_arb YES trades.")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to opportunity_log.db")
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS,
        help="Trailing stop thresholds to test (pct gain from entry cost)",
    )
    parser.add_argument("--max-spread", type=int, default=DEFAULT_MAX_SPREAD, help="Max bid-ask spread for stop (¢)")
    parser.add_argument("--floor", type=int, default=DEFAULT_FLOOR, help="Floor price — suppress stop at or below (¢)")
    parser.add_argument(
        "--detail", type=float, default=None,
        help="Print per-trade breakdown for this threshold (e.g. --detail 0.20)",
    )
    args = parser.parse_args()

    main(
        thresholds=sorted(set(args.thresholds)),
        max_spread=args.max_spread,
        floor=args.floor,
        detail_threshold=args.detail,
        db_path=Path(args.db),
    )
