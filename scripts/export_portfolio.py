"""Portfolio P&L export and visualization.

Reads the live dry_run trades database and outputs:
  - CSV exports (data/portfolio/)
      trades.csv            — all resolved trades with full fields
      equity_curve.csv      — cumulative P&L over time (one row per exit)
      pnl_by_source.csv     — win rate, avg P&L, total P&L per source
      price_history.csv     — per-trade price snapshots (long format)
  - PNG charts (data/portfolio/)
      equity_curve.png      — cumulative P&L + per-trade bar overlay
      pnl_by_source.png     — P&L and win rate per source (grouped bar)
      drawdown.png          — rolling drawdown from equity peak
      trade_timeline.png    — scatter of each trade's P&L over time (colored by source)

Usage:
    venv/bin/python scripts/export_portfolio.py
    venv/bin/python scripts/export_portfolio.py --db data/db/opportunity_log.db --out data/portfolio
"""

import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_DB  = Path(__file__).parent.parent / "data" / "db" / "opportunity_log.db"
_DEFAULT_OUT = Path(__file__).parent.parent / "data" / "portfolio"

SOURCE_COLORS = {
    "band_arb":       "#2196F3",
    "forecast_no":    "#FF9800",
    "noaa_observed":  "#9C27B0",
    "metar":          "#00BCD4",
    "hrrr":           "#4CAF50",
    "noaa":           "#F44336",
    "open_meteo":     "#795548",
    "nws_hourly":     "#607D8B",
    "yahoo_wti_futures": "#CDDC39",
}

def _color(source: str) -> str:
    return SOURCE_COLORS.get(source, "#9E9E9E")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(db: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(str(db))

    trades = pd.read_sql_query(
        """
        SELECT
            id, logged_at, exited_at, ticker, side, source, opportunity_kind,
            fill_price_cents, exit_price_cents, exit_pnl_cents / 100.0 AS pnl_dollars,
            count AS contracts, exit_reason, p_estimate, score,
            yes_bid_entry, yes_ask_entry, corroborating_sources, note, bug_loss
        FROM trades
        WHERE mode = 'dry_run' AND exit_pnl_cents IS NOT NULL
          AND exited_at IS NOT NULL
        ORDER BY exited_at
        """,
        conn,
        parse_dates=["logged_at", "exited_at"],
    )

    snapshots = pd.read_sql_query(
        """
        SELECT
            ps.id, ps.trade_id, ps.snapshot_at,
            ps.yes_bid, ps.yes_ask, ps.exit_price,
            ps.unrealized_cents / 100.0 AS unrealized_dollars,
            ps.pct_gain, ps.days_to_close, ps.post_exit
        FROM price_snapshots ps
        JOIN trades t ON t.id = ps.trade_id
        WHERE t.mode = 'dry_run' AND t.exit_pnl_cents IS NOT NULL
        ORDER BY ps.snapshot_at
        """,
        conn,
        parse_dates=["snapshot_at"],
    )

    conn.close()
    return trades, snapshots


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

def _export_csvs(out: Path, trades: pd.DataFrame, snapshots: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # 1. Full trades table
    trades.to_csv(out / "trades.csv", index=False)
    print(f"  trades.csv           ({len(trades)} rows)")

    # 2. Equity curve
    equity = trades[["exited_at", "source", "ticker", "side", "pnl_dollars"]].copy()
    equity["cumulative_pnl"] = equity["pnl_dollars"].cumsum()
    equity["trade_num"] = range(1, len(equity) + 1)
    equity.to_csv(out / "equity_curve.csv", index=False)
    print(f"  equity_curve.csv     ({len(equity)} rows)")

    # 3. P&L by source
    by_source = (
        trades.groupby("source")
        .agg(
            trades=("pnl_dollars", "count"),
            wins=("pnl_dollars", lambda x: (x > 0).sum()),
            total_pnl=("pnl_dollars", "sum"),
            avg_pnl=("pnl_dollars", "mean"),
            median_pnl=("pnl_dollars", "median"),
        )
        .reset_index()
    )
    by_source["win_rate"] = by_source["wins"] / by_source["trades"]
    by_source = by_source.sort_values("total_pnl", ascending=False)
    by_source.to_csv(out / "pnl_by_source.csv", index=False)
    print(f"  pnl_by_source.csv    ({len(by_source)} rows)")

    # 4. Price history (long format)
    snapshots.to_csv(out / "price_history.csv", index=False)
    print(f"  price_history.csv    ({len(snapshots)} rows)")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

_FONT = {"family": "monospace"}


def _to_naive_utc(series: pd.Series) -> pd.Series:
    """Convert tz-aware datetime series to naive UTC (matplotlib-safe)."""
    s = pd.to_datetime(series, utc=True)
    return s.dt.tz_localize(None)


def _equity_curve_chart(out: Path, trades: pd.DataFrame) -> None:
    fig, (ax_main, ax_bar) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )

    times = _to_naive_utc(trades["exited_at"])
    cum   = trades["pnl_dollars"].cumsum()
    pnl   = trades["pnl_dollars"]

    # Main: cumulative P&L line
    ax_main.plot(times, cum, color="#2196F3", linewidth=2, zorder=3)
    ax_main.fill_between(times, 0, cum, where=cum >= 0, alpha=0.12, color="#2196F3")
    ax_main.fill_between(times, 0, cum, where=cum < 0,  alpha=0.12, color="#F44336")
    ax_main.axhline(0, color="#555", linewidth=0.8, linestyle="--")

    # Scatter per trade colored by source
    for src in trades["source"].unique():
        mask = trades["source"] == src
        ax_main.scatter(
            times[mask], cum[mask],
            color=_color(src), s=40, zorder=4, label=src, edgecolors="white", linewidth=0.4,
        )

    ax_main.set_title("Cumulative P&L — Dry Run", fontdict=_FONT, fontsize=14, pad=10)
    ax_main.set_ylabel("Cumulative P&L ($)", fontdict=_FONT)
    ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:+.2f}"))
    ax_main.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax_main.grid(axis="y", alpha=0.3)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=30)

    # Bar: per-trade P&L
    colors = [("#2196F3" if v >= 0 else "#F44336") for v in pnl]
    ax_bar.bar(range(len(pnl)), pnl.values, color=colors, width=0.8)
    ax_bar.axhline(0, color="#555", linewidth=0.8)
    ax_bar.set_xlabel("Trade #", fontdict=_FONT)
    ax_bar.set_ylabel("P&L ($)", fontdict=_FONT, fontsize=9)
    ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:+.2f}"))
    ax_bar.grid(axis="y", alpha=0.3)

    # Annotations
    final = cum.iloc[-1]
    n_trades = len(trades)
    n_wins = (pnl > 0).sum()
    ax_main.annotate(
        f"Final P&L: ${final:+.2f}  |  {n_wins}/{n_trades} wins ({100*n_wins/n_trades:.0f}%)",
        xy=(0.01, 0.97), xycoords="axes fraction",
        fontsize=10, fontfamily="monospace",
        va="top", color="#111",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85),
    )

    plt.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  equity_curve.png")


def _source_chart(out: Path, trades: pd.DataFrame) -> None:
    by_src = (
        trades.groupby("source")
        .agg(
            n=("pnl_dollars", "count"),
            wins=("pnl_dollars", lambda x: (x > 0).sum()),
            total=("pnl_dollars", "sum"),
            avg=("pnl_dollars", "mean"),
        )
        .reset_index()
    )
    by_src["win_rate"] = by_src["wins"] / by_src["n"] * 100
    by_src = by_src.sort_values("total", ascending=True)

    fig, (ax_pnl, ax_wr) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total P&L horizontal bar
    colors = [(_color(s) if v >= 0 else "#EF9A9A") for s, v in zip(by_src["source"], by_src["total"])]
    bars = ax_pnl.barh(by_src["source"], by_src["total"], color=colors, height=0.6)
    ax_pnl.axvline(0, color="#555", linewidth=0.8)
    ax_pnl.set_xlabel("Total P&L ($)", fontdict=_FONT)
    ax_pnl.set_title("Total P&L by Source", fontdict=_FONT, fontsize=12)
    ax_pnl.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:+.2f}"))
    for bar, row in zip(bars, by_src.itertuples()):
        label = f"  n={row.n}  avg=${row.avg:+.2f}"
        ax_pnl.text(
            bar.get_width() + (0.1 if bar.get_width() >= 0 else -0.1),
            bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left" if bar.get_width() >= 0 else "right",
            fontsize=8, fontfamily="monospace",
        )
    ax_pnl.grid(axis="x", alpha=0.3)

    # Right: win rate bar
    colors_wr = [_color(s) for s in by_src["source"]]
    ax_wr.barh(by_src["source"], by_src["win_rate"], color=colors_wr, height=0.6, alpha=0.85)
    ax_wr.axvline(50, color="#555", linewidth=0.8, linestyle="--")
    ax_wr.set_xlabel("Win Rate (%)", fontdict=_FONT)
    ax_wr.set_title("Win Rate by Source", fontdict=_FONT, fontsize=12)
    ax_wr.set_xlim(0, 105)
    for i, row in enumerate(by_src.itertuples()):
        ax_wr.text(row.win_rate + 1, i, f"{row.win_rate:.0f}%", va="center", fontsize=9, fontfamily="monospace")
    ax_wr.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "pnl_by_source.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  pnl_by_source.png")


def _drawdown_chart(out: Path, trades: pd.DataFrame) -> None:
    cum   = trades["pnl_dollars"].cumsum()
    peak  = cum.cummax()
    dd    = cum - peak
    times = _to_naive_utc(trades["exited_at"])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(times, dd, 0, color="#F44336", alpha=0.4)
    ax.plot(times, dd, color="#F44336", linewidth=1.2)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_title("Drawdown from Equity Peak", fontdict=_FONT, fontsize=12)
    ax.set_ylabel("Drawdown ($)", fontdict=_FONT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate(rotation=30)

    max_dd = dd.min()
    ax.annotate(
        f"Max drawdown: ${max_dd:.2f}",
        xy=(0.01, 0.05), xycoords="axes fraction",
        fontsize=10, fontfamily="monospace",
        va="bottom", color="#B71C1C",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85),
    )

    plt.tight_layout()
    fig.savefig(out / "drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  drawdown.png")


def _trade_timeline_chart(out: Path, trades: pd.DataFrame) -> None:
    times = _to_naive_utc(trades["exited_at"])
    pnl   = trades["pnl_dollars"]

    fig, ax = plt.subplots(figsize=(14, 5))

    for src in trades["source"].unique():
        mask = trades["source"] == src
        ax.scatter(
            times[mask], pnl[mask],
            label=src, color=_color(src),
            s=50, edgecolors="white", linewidth=0.4, zorder=3, alpha=0.85,
        )

    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.set_title("Per-Trade P&L over Time (by Source)", fontdict=_FONT, fontsize=12)
    ax.set_ylabel("P&L ($)", fontdict=_FONT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:+.2f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    fig.savefig(out / "trade_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  trade_timeline.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Export portfolio P&L to CSV and PNG charts")
    p.add_argument("--db",  default=str(_DEFAULT_DB),  help="Path to opportunity_log.db")
    p.add_argument("--out", default=str(_DEFAULT_OUT), help="Output directory")
    args = p.parse_args()

    db  = Path(args.db)
    out = Path(args.out)

    print(f"Loading trades from {db} ...")
    trades, snapshots = _load(db)
    print(f"  {len(trades)} resolved trades, {len(snapshots)} price snapshots\n")

    print("Exporting CSVs ...")
    _export_csvs(out, trades, snapshots)

    print("\nRendering charts ...")
    _equity_curve_chart(out, trades)
    _source_chart(out, trades)
    _drawdown_chart(out, trades)
    _trade_timeline_chart(out, trades)

    total_pnl = trades["pnl_dollars"].sum()
    n_wins    = (trades["pnl_dollars"] > 0).sum()
    n         = len(trades)
    print(f"\nDone.  Output → {out}/")
    print(f"  Resolved trades : {n}")
    print(f"  Win rate        : {n_wins}/{n} ({100*n_wins/n:.0f}%)")
    print(f"  Total P&L       : ${total_pnl:+.2f}")


if __name__ == "__main__":
    main()
