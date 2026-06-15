"""Backtest "pending stabilization" entry logic for band_arb YES trades.

Instead of entering immediately on signal, the bot monitors the market bid
for up to MAX_WAIT minutes before committing.  Entry rules tested:

  immediate   — enter at the original ask (baseline)
  wait_N      — enter at whatever ask is available N minutes after signal
  stabilize   — enter when bid recovers RECOVERY¢ above its local minimum
                after first dipping at least MIN_DIP¢ below the initial bid

Abort condition (applied in all non-immediate strategies):
  If bid drops to ≤ ABORT_THRESHOLD¢ at any point during the wait window,
  skip the trade entirely.  This is the primary ceiling-breach filter.

Usage:
  venv/bin/python scripts/backtest_pending_entry.py
  venv/bin/python scripts/backtest_pending_entry.py --abort 10 --recovery 3
  venv/bin/python scripts/backtest_pending_entry.py --max-wait 20 --detail
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path("data/db/opportunity_log.db")

# Default parameters
DEFAULT_ABORT     = 15   # abort if bid falls to ≤ this during wait (¢)
DEFAULT_MIN_DIP   = 3    # minimum dip below bid0 before stabilization entry armed (¢)
DEFAULT_RECOVERY  = 2    # cents above local min before entering (stabilization)
DEFAULT_MAX_WAIT  = 30   # max minutes to wait before entering at market anyway


@dataclass
class Snap:
    elapsed_min: float   # minutes since first snapshot
    yes_bid:     int
    yes_ask:     int | None


@dataclass
class Trade:
    trade_id:   int
    ticker:     str
    entry_ask:  int      # actual ask paid (limit_price for YES)
    count:      int
    outcome:    str      # 'won' | 'lost'
    snaps:      list[Snap] = field(default_factory=list)

    @property
    def settlement_pnl(self) -> float:
        return (100 - self.entry_ask) * self.count if self.outcome == "won" else -self.entry_ask * self.count


@dataclass
class SimResult:
    entered:    bool
    entry_ask:  int | None   # ask paid (None = aborted)
    entry_min:  float | None # minutes elapsed at entry
    aborted_at: int | None   # bid level that triggered abort
    pnl:        float        # cents (0 if aborted)
    note:       str          # short description of what happened


def _ask_at(snaps: list[Snap], elapsed: float) -> int | None:
    """Best ask estimate at a given elapsed minute mark."""
    for s in reversed(snaps):
        if s.elapsed_min <= elapsed:
            return s.yes_ask
    return snaps[0].yes_ask if snaps else None


def simulate_immediate(trade: Trade) -> SimResult:
    """Baseline: enter immediately at the original ask."""
    pnl = (100 - trade.entry_ask) * trade.count if trade.outcome == "won" else -trade.entry_ask * trade.count
    return SimResult(
        entered=True, entry_ask=trade.entry_ask, entry_min=0.0,
        aborted_at=None, pnl=pnl, note="immediate",
    )


def simulate_wait(
    trade: Trade,
    wait_min: float,
    abort_threshold: int,
    max_wait: float,
) -> SimResult:
    """Enter at the first snapshot ≥ wait_min, abort if bid drops ≤ abort_threshold."""
    if not trade.snaps:
        return simulate_immediate(trade)

    best_ask = None
    best_min = None

    for snap in trade.snaps:
        if abort_threshold > 0 and snap.yes_bid <= abort_threshold:
            return SimResult(
                entered=False, entry_ask=None, entry_min=snap.elapsed_min,
                aborted_at=snap.yes_bid, pnl=0.0,
                note=f"aborted bid={snap.yes_bid}¢ ≤ {abort_threshold}¢ at {snap.elapsed_min:.0f}m",
            )
        if snap.elapsed_min >= wait_min and best_ask is None:
            best_ask = snap.yes_ask or snap.yes_bid + 2  # estimate if ask missing
            best_min = snap.elapsed_min
        if snap.elapsed_min >= max_wait:
            break

    if best_ask is None:
        # Wait longer than available snapshots — use last available ask
        last = trade.snaps[-1]
        best_ask = last.yes_ask or last.yes_bid + 2
        best_min = last.elapsed_min

    pnl = (100 - best_ask) * trade.count if trade.outcome == "won" else -best_ask * trade.count
    return SimResult(
        entered=True, entry_ask=best_ask, entry_min=best_min,
        aborted_at=None, pnl=pnl, note=f"wait {wait_min:.0f}m (entered @{best_min:.0f}m)",
    )


def simulate_stabilize(
    trade: Trade,
    min_dip:         int,
    recovery_cents:  int,
    abort_threshold: int,
    max_wait:        float,
) -> SimResult:
    """Enter when bid recovers RECOVERY¢ above local min after dipping MIN_DIP¢.

    Also aborts if bid ≤ abort_threshold at any point.
    If the price never dips enough, enters at max_wait anyway.
    """
    if not trade.snaps:
        return simulate_immediate(trade)

    bid0      = trade.snaps[0].yes_bid
    local_min = bid0
    armed     = False  # True once bid has dipped min_dip below bid0

    for snap in trade.snaps:
        # Abort gate: real ceiling breach
        if abort_threshold > 0 and snap.yes_bid <= abort_threshold:
            return SimResult(
                entered=False, entry_ask=None, entry_min=snap.elapsed_min,
                aborted_at=snap.yes_bid, pnl=0.0,
                note=f"aborted bid={snap.yes_bid}¢ ≤ {abort_threshold}¢ at {snap.elapsed_min:.0f}m",
            )

        if snap.yes_bid < local_min:
            local_min = snap.yes_bid

        # Arm when dip is deep enough
        if not armed and (bid0 - local_min) >= min_dip:
            armed = True

        # Enter on recovery
        if armed and (snap.yes_bid - local_min) >= recovery_cents:
            ask = snap.yes_ask or snap.yes_bid + 2
            pnl = (100 - ask) * trade.count if trade.outcome == "won" else -ask * trade.count
            return SimResult(
                entered=True, entry_ask=ask, entry_min=snap.elapsed_min,
                aborted_at=None, pnl=pnl,
                note=f"stabilized @{snap.elapsed_min:.0f}m (dip→{local_min}¢, +{recovery_cents}¢ recovery)",
            )

        if snap.elapsed_min >= max_wait:
            break

    # Never stabilized in time — enter at market at max_wait
    last_snap = None
    for snap in trade.snaps:
        if snap.elapsed_min <= max_wait:
            last_snap = snap
    if last_snap is None:
        last_snap = trade.snaps[-1]
    ask = last_snap.yes_ask or last_snap.yes_bid + 2
    pnl = (100 - ask) * trade.count if trade.outcome == "won" else -ask * trade.count
    return SimResult(
        entered=True, entry_ask=ask, entry_min=last_snap.elapsed_min,
        aborted_at=None, pnl=pnl,
        note=f"timeout @{last_snap.elapsed_min:.0f}m (no stabilization)",
    )


@dataclass
class Stats:
    n_total:   int = 0
    n_entered: int = 0
    n_aborted: int = 0
    total_pnl: float = 0.0
    entry_asks: list[int] = field(default_factory=list)
    entry_mins: list[float] = field(default_factory=list)

    def add(self, result: SimResult) -> None:
        self.n_total += 1
        if result.entered:
            self.n_entered += 1
            self.total_pnl += result.pnl
            if result.entry_ask:
                self.entry_asks.append(result.entry_ask)
            if result.entry_min is not None:
                self.entry_mins.append(result.entry_min)
        else:
            self.n_aborted += 1
            # aborted trades: pnl = 0 (no position), but count them in total
            # so we don't inflate win rate by only counting entered trades

    @property
    def avg_ask(self) -> float:
        return sum(self.entry_asks) / len(self.entry_asks) if self.entry_asks else 0.0

    @property
    def avg_entry_min(self) -> float:
        return sum(self.entry_mins) / len(self.entry_mins) if self.entry_mins else 0.0


def load_trades(db_path: Path) -> list[Trade]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT id, ticker, limit_price, count, outcome, logged_at
        FROM trades
        WHERE opportunity_kind = 'band_arb' AND side = 'yes'
          AND outcome IS NOT NULL AND outcome != 'void'
        ORDER BY id
    """).fetchall()

    trades = []
    for tid, ticker, limit_price, count, outcome, logged_at in rows:
        snaps_raw = conn.execute("""
            SELECT snapshot_at, yes_bid, yes_ask
            FROM price_snapshots
            WHERE trade_id = ? AND yes_bid IS NOT NULL
            ORDER BY snapshot_at
        """, (tid,)).fetchall()

        if not snaps_raw:
            continue

        from datetime import datetime
        t0 = datetime.fromisoformat(snaps_raw[0][0])
        snaps = []
        for sat, bid, ask in snaps_raw:
            st = datetime.fromisoformat(sat)
            elapsed = (st - t0).total_seconds() / 60.0
            snaps.append(Snap(elapsed, bid, ask))

        trades.append(Trade(tid, ticker, limit_price, count, outcome, snaps))

    conn.close()
    return trades


def run_all(
    trades: list[Trade],
    abort: int,
    min_dip: int,
    recovery: int,
    max_wait: float,
    detail: bool,
) -> None:
    strategies: list[tuple[str, callable]] = [
        ("immediate",        lambda t: simulate_immediate(t)),
        ("wait 5m",          lambda t: simulate_wait(t, 5,  abort, max_wait)),
        ("wait 10m",         lambda t: simulate_wait(t, 10, abort, max_wait)),
        ("wait 15m",         lambda t: simulate_wait(t, 15, abort, max_wait)),
        ("wait 30m",         lambda t: simulate_wait(t, 30, abort, max_wait)),
        (f"stabilize({recovery}¢)", lambda t: simulate_stabilize(t, min_dip, recovery, abort, max_wait)),
        (f"stabilize({recovery*2}¢)", lambda t: simulate_stabilize(t, min_dip, recovery * 2, abort, max_wait)),
    ]

    won  = sum(1 for t in trades if t.outcome == "won")
    lost = sum(1 for t in trades if t.outcome == "lost")
    print(f"\nPending-Entry Backtest — Band-Arb YES")
    print(f"{'='*78}")
    print(f"Trades: {len(trades)} ({won} won, {lost} lost)")
    print(f"Abort threshold: ≤{abort}¢  |  Min dip: {min_dip}¢  |  "
          f"Recovery: {recovery}¢  |  Max wait: {max_wait:.0f}m\n")

    hdr = f"{'Strategy':<22}  {'Ent':>4}  {'Abt':>4}  {'AvgAsk':>7}  {'AvgWait':>8}  "
    hdr += f"{'TotalP&L':>9}  {'vs Base':>8}"
    print(hdr)
    print("-" * 78)

    base_pnl = None
    for name, fn in strategies:
        stats = Stats()
        results: list[tuple[Trade, SimResult]] = []
        for t in trades:
            r = fn(t)
            stats.add(r)
            results.append((t, r))

        if base_pnl is None:
            base_pnl = stats.total_pnl

        delta    = stats.total_pnl - base_pnl
        sign     = "+" if delta >= 0 else ""
        avg_ask  = f"{stats.avg_ask:.1f}¢" if stats.entry_asks else "—"
        avg_wait = f"{stats.avg_entry_min:.1f}m" if stats.entry_mins else "—"

        print(
            f"{name:<22}  {stats.n_entered:>4}  {stats.n_aborted:>4}  "
            f"{avg_ask:>7}  {avg_wait:>8}  "
            f"${stats.total_pnl/100:>+8.2f}  {sign}${delta/100:>7.2f}"
        )

        if detail:
            for t, r in results:
                base_r = simulate_immediate(t)
                if abs(r.pnl - base_r.pnl) > 10 or not r.entered:
                    delta_t = r.pnl - base_r.pnl
                    sign_t  = "+" if delta_t >= 0 else ""
                    print(
                        f"    #{t.trade_id:>3d} {t.ticker:<32s} {t.outcome:>4s}  "
                        f"base=${base_r.pnl/100:+.2f}  "
                        f"sim=${r.pnl/100:+.2f}  delta={sign_t}${delta_t/100:.2f}"
                        f"  [{r.note}]"
                    )

    # Show abort sensitivity — what thresholds would catch the losers?
    print(f"\n{'—'*78}")
    print("Abort threshold sensitivity (stabilize strategy):")
    print(f"  {'Threshold':>10}  {'Aborted':>8}  {'TotalP&L':>10}  {'vs immed':>10}")
    base = sum(t.settlement_pnl for t in trades)
    for thr in [5, 10, 15, 20, 25, 30, 35, 40]:
        stats = Stats()
        for t in trades:
            r = simulate_stabilize(t, min_dip, recovery, thr, max_wait)
            stats.add(r)
        delta = stats.total_pnl - base
        sign  = "+" if delta >= 0 else ""
        print(
            f"  {thr:>10}¢  {stats.n_aborted:>8}  "
            f"${stats.total_pnl/100:>+9.2f}  {sign}${delta/100:>9.2f}"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest pending/stabilization entry for band_arb YES."
    )
    parser.add_argument("--db",         default=str(DB_PATH))
    parser.add_argument("--abort",      type=int,   default=DEFAULT_ABORT,
                        help="Abort if bid ≤ this during wait (¢)")
    parser.add_argument("--min-dip",    type=int,   default=DEFAULT_MIN_DIP,
                        help="Min dip below initial bid before stabilization armed (¢)")
    parser.add_argument("--recovery",   type=int,   default=DEFAULT_RECOVERY,
                        help="Cents above local min to trigger entry (¢)")
    parser.add_argument("--max-wait",   type=float, default=DEFAULT_MAX_WAIT,
                        help="Max minutes to wait before entering at market")
    parser.add_argument("--detail",     action="store_true",
                        help="Print per-trade breakdown for strategies that differ from baseline")
    args = parser.parse_args()

    trades = load_trades(Path(args.db))
    if not trades:
        print("No resolved band_arb YES trades with snapshots found.")
    else:
        run_all(
            trades,
            abort=args.abort,
            min_dip=args.min_dip,
            recovery=args.recovery,
            max_wait=args.max_wait,
            detail=args.detail,
        )
