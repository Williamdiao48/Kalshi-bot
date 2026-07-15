"""Shadow-model overview report generation (V1 + V2 NO models).

Reads from the ``shadow_model_no`` and ``shadow_model_no_v2`` tables and writes
human-readable ``*_overview.txt`` reports.  Each shadow entry is a 1-contract
NO bet; P&L per entry is ``+(100 - entry_no_price)`` on a win, ``-entry_no_price``
on a loss.

This module holds the pure report-building logic so it can be called both from
the CLI (``scripts/shadow_model_overview.py``) and from the running bot's daily
refresh hook.  ``regenerate_overviews()`` is the single entry point.
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .db import OPPORTUNITY_LOG_DB

_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = OPPORTUNITY_LOG_DB
OUT_V1  = _ROOT / "shadow_model_no_overview.txt"
OUT_V2  = _ROOT / "shadow_model_no_v2_overview.txt"
STARTING_CAPITAL_CENTS = 40_000  # $400.00

# model key -> (table, display label, output path)
_MODELS: dict[str, tuple[str, str, Path]] = {
    "v1": ("shadow_model_no",    "V1 — shadow_model_no",    OUT_V1),
    "v2": ("shadow_model_no_v2", "V2 — shadow_model_no_v2", OUT_V2),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load(conn: sqlite3.Connection, table: str) -> list[dict]:
    cols = [
        "id", "logged_at", "ticker", "series", "is_high",
        "model_p", "market_p_no", "edge", "margin_f", "hvc",
        "hour_utc", "exit_reason", "exited_at", "outcome", "pnl_cents",
    ]
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM {table} ORDER BY logged_at ASC"
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def _no_price_cents(r: dict) -> int:
    """NO price in cents = market_p_no * 100."""
    return round((r["market_p_no"] or 0.5) * 100)


def _yes_ask_implied(r: dict) -> int:
    """Implied YES ask = 100 - no_price."""
    return 100 - _no_price_cents(r)


def _compute_risk_metrics(rows: list[dict], starting_capital: int) -> dict:
    resolved = [r for r in rows if r["outcome"] in ("won", "lost")]
    if len(resolved) < 5:
        return {}

    pnl_list: list[float] = []
    returns:  list[float] = []
    for r in resolved:
        pnl  = float(r["pnl_cents"])
        cost = _no_price_cents(r)
        if cost <= 0:
            continue
        pnl_list.append(pnl)
        returns.append(pnl / cost)

    if len(returns) < 5:
        return {}

    # Equity curve and drawdown
    equity = float(starting_capital)
    peak   = equity
    max_dd = 0.0
    for pnl in pnl_list:
        equity += pnl
        peak    = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    current_dd = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0

    # Annualisation
    n = len(returns)
    try:
        t0 = datetime.fromisoformat(resolved[0]["logged_at"])
        t1 = datetime.fromisoformat(resolved[-1]["logged_at"])
        years = max((t1 - t0).total_seconds() / 31_557_600, 1 / 365)
        tpy   = n / years
    except (ValueError, TypeError):
        tpy = 365.0

    mean_r = statistics.mean(returns)
    std_r  = statistics.pstdev(returns)
    sharpe = (mean_r / std_r * math.sqrt(tpy)) if std_r > 0 else None

    neg     = [r for r in returns if r < 0]
    dn_std  = statistics.pstdev(neg) if len(neg) >= 2 else 0.0
    sortino = (mean_r / dn_std * math.sqrt(tpy)) if dn_std > 0 else None

    wins    = [p for p in pnl_list if p > 0]
    losses  = [p for p in pnl_list if p <= 0]
    gain_sum = sum(wins)
    loss_sum = abs(sum(losses))
    pf = gain_sum / loss_sum if loss_sum > 0 else None

    return {
        "n":             n,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "max_dd":        max_dd,
        "current_dd":    current_dd,
        "win_rate":      len(wins) / n,
        "wins":          len(wins),
        "losses":        len(losses),
        "avg_gain":      statistics.mean(wins)   if wins   else 0.0,
        "avg_loss":      statistics.mean(losses) if losses else 0.0,
        "profit_factor": pf,
    }


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def _build_overview(rows: list[dict], model_name: str) -> str:
    W = "=" * 68
    S = "-" * 68

    def _d(cents: float) -> str:
        sign = "+" if cents >= 0 else ""
        return f"{sign}${cents / 100:.2f}"

    settled = [r for r in rows if r["outcome"] in ("won", "lost") and r["exit_reason"] != "profit_take"]
    exited  = [r for r in rows if r["exit_reason"] == "profit_take"]
    open_   = [r for r in rows if r["outcome"] is None]

    # Flag pre-gate entries: those with NO price outside the gated range (15–85¢)
    # These existed before the EV / min-payout / yes_ask gates were added (Jun 15).
    pre_gate = [r for r in rows if not (15 <= _no_price_cents(r) <= 85)]
    pre_gate_pnl = sum(float(r["pnl_cents"]) for r in pre_gate if r["pnl_cents"] is not None)

    settled_gains  = sum(max(0.0, float(r["pnl_cents"])) for r in settled if r["pnl_cents"] is not None)
    settled_losses = sum(min(0.0, float(r["pnl_cents"])) for r in settled if r["pnl_cents"] is not None)
    exit_gains     = sum(max(0.0, float(r["pnl_cents"])) for r in exited  if r["pnl_cents"] is not None)
    exit_losses    = sum(min(0.0, float(r["pnl_cents"])) for r in exited  if r["pnl_cents"] is not None)

    realized_gains_cents  = settled_gains  + exit_gains
    realized_losses_cents = settled_losses + exit_losses

    # Open entries have no live price; show as zero unrealized
    unrealized_cents = 0.0

    current_balance_cents = (
        STARTING_CAPITAL_CENTS
        + realized_gains_cents
        + realized_losses_cents
        + unrealized_cents
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    buf: list[str] = [
        W,
        f"  SHADOW MODEL OVERVIEW  [{model_name}]",
        f"  Last updated: {now}",
        W,
        f"  Starting capital :  ${STARTING_CAPITAL_CENTS / 100:>8.2f}  (simulated, 1-contract per entry)",
        f"  Realized gains   :  {_d(realized_gains_cents):>9}  "
        f"(settled wins: {_d(settled_gains)}  exits: {_d(exit_gains)})",
        f"  Realized losses  :  {_d(realized_losses_cents):>9}  "
        f"(settled losses: {_d(settled_losses)}  exits: {_d(exit_losses)})",
        f"  Unrealized P&L   :  {_d(unrealized_cents):>9}  (open entries — no live price)",
        S,
        f"  Current balance  :  ${current_balance_cents / 100:>8.2f}",
        S,
        f"  Total entries    :  {len(rows)}",
        f"  Settled (normal) :  {len(settled)}",
        f"  Exited (early)   :  {len(exited)}",
        f"  Open / pending   :  {len(open_)}",
        f"  Pre-gate entries :  {len(pre_gate)}  (NO price outside 15–85¢; added before Jun 15 gates)",
        W,
    ]

    if pre_gate:
        gated_rows   = [r for r in rows if 15 <= _no_price_cents(r) <= 85]
        gated_settled = [r for r in gated_rows if r["outcome"] in ("won", "lost")]
        gated_wins   = sum(1 for r in gated_settled if r["pnl_cents"] is not None and float(r["pnl_cents"]) > 0)
        gated_net    = sum(float(r["pnl_cents"]) for r in gated_settled if r["pnl_cents"] is not None)
        gated_wr     = gated_wins / len(gated_settled) * 100 if gated_settled else 0
        buf.extend([
            "  NOTE: PRE-GATE DATA DISTORTS HEADLINE NUMBERS",
            S,
            f"  {len(pre_gate)} entries have NO price <15¢ or >85¢ — these predate the Jun 15",
            f"  entry gates (EV gate, yes_ask cap, min-payout gate). They include",
            f"  1¢ NO bets (market said YES was certain) and 99¢ bets (1¢ payout).",
            f"  Pre-gate entries contributed {_d(pre_gate_pnl)} to total P&L.",
            S,
            f"  GATED-RANGE ONLY (15–85¢ NO, {len(gated_settled)} settled):",
            f"  Win rate : {gated_wr:.1f}%  ({gated_wins}W / {len(gated_settled)-gated_wins}L)",
            f"  Net P&L  : {_d(gated_net)}",
            W,
        ])

    rm = _compute_risk_metrics(rows, STARTING_CAPITAL_CENTS)
    if rm:
        def _pct(v: float) -> str: return f"{v * 100:.1f}%"
        def _opt(v: object, fmt: str) -> str: return fmt % v if v is not None else "n/a"
        buf.extend([
            "  RISK METRICS",
            S,
            f"  Resolved entries :  {rm['n']}",
            f"  Sharpe ratio     :  {_opt(rm['sharpe'],  '%.2f')}  (annualized per-entry)",
            f"  Sortino ratio    :  {_opt(rm['sortino'], '%.2f')}",
            f"  Max drawdown     :  -{_pct(rm['max_dd'])}  (from equity peak)",
            f"  Current drawdown :  -{_pct(rm['current_dd'])}",
            f"  Win rate         :  {_pct(rm['win_rate'])}  ({rm['wins']}W / {rm['losses']}L)",
            f"  Avg gain         :  {_d(rm['avg_gain'])}",
            f"  Avg loss         :  {_d(rm['avg_loss'])}",
            f"  Profit factor    :  {_opt(rm['profit_factor'], '%.2f')}  (total gains / total losses)",
            W,
        ])

    # Per-type breakdown (KXHIGH vs KXLOWT)
    for is_high, label in [(1, "KXHIGH"), (0, "KXLOWT")]:
        sub = [r for r in rows if r["outcome"] in ("won", "lost") and r["is_high"] == is_high]
        if not sub:
            continue
        wins_sub   = sum(1 for r in sub if r["pnl_cents"] is not None and float(r["pnl_cents"]) > 0)
        losses_sub = len(sub) - wins_sub
        net_sub    = sum(float(r["pnl_cents"]) for r in sub if r["pnl_cents"] is not None)
        wr_sub     = wins_sub / len(sub) * 100
        avg_no     = sum((1 - r["market_p_no"]) * 100 for r in sub) / len(sub)
        avg_pnl    = net_sub / len(sub)
        buf.extend([
            f"  {label} breakdown ({len(sub)} settled)",
            S,
            f"  Win rate   :  {wr_sub:.1f}%  ({wins_sub}W / {losses_sub}L)",
            f"  Avg NO price at entry :  {avg_no:.1f}¢",
            f"  Avg P&L per entry     :  {_d(avg_pnl)}",
            f"  Net P&L               :  {_d(net_sub)}",
            W,
        ])

    buf.extend([
        "  ENTRY HISTORY  (oldest → newest)",
        W,
    ])

    running_balance = STARTING_CAPITAL_CENTS
    for r in rows:
        date_str  = r["logged_at"][:16].replace("T", " ")
        is_high   = r["is_high"]
        no_price  = _no_price_cents(r)
        win_price = 100 - no_price
        model_pct = round((r["model_p"] or 0) * 100)
        gate_flag = "  [PRE-GATE]" if not (15 <= no_price <= 85) else ""

        pnl = float(r["pnl_cents"]) if r["pnl_cents"] is not None else None

        if r["outcome"] == "won":
            status = "SETTLED WIN "
        elif r["outcome"] == "lost":
            status = "SETTLED LOSS"
        elif r["exit_reason"] == "profit_take":
            status = "EXITED PROFIT-TAKE"
        else:
            status = "OPEN        "

        if pnl is not None:
            pnl_tag = f"  P&L {_d(pnl)}"
            running_balance += pnl
        else:
            pnl_tag = "  P&L pending"

        bal_str = f"  balance ${running_balance / 100:.2f}"

        buf.append(
            f"  {date_str}  NO  {r['ticker']:<34}"
            f"  paid {no_price}¢  (+{win_price}¢ profit/ea)"
        )
        buf.append(
            f"         model={model_pct}%  mkt_no={no_price}¢"
            f"  [{status}]{pnl_tag}{bal_str}{gate_flag}"
        )
        buf.append("")

    buf.extend([
        S,
        f"  Total realized gains  : {_d(realized_gains_cents)}",
        f"  Total realized losses : {_d(realized_losses_cents)}",
        f"  Net realized P&L      : {_d(realized_gains_cents + realized_losses_cents)}",
        W,
    ])

    return "\n".join(buf)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def regenerate_overviews(
    models: tuple[str, ...] = ("v1", "v2"),
    db_path: Path | str | None = None,
) -> dict[str, int]:
    """Rebuild the requested shadow-model overview files from the live DB.

    Opens its own short-lived read-only connection so it is safe to call from a
    worker thread (via ``asyncio.to_thread``) without touching the bot's shared
    connection.  Returns ``{model_key: entry_count}`` for the files written.
    """
    counts: dict[str, int] = {}
    conn = sqlite3.connect(str(db_path or DB_PATH))
    try:
        for key in models:
            table, label, out = _MODELS[key]
            rows = _load(conn, table)
            out.write_text(_build_overview(rows, label) + "\n", encoding="utf-8")
            counts[key] = len(rows)
    finally:
        conn.close()
    return counts
