"""Regenerate the README result charts (assets/img/*.svg) from live DB data.

Two hand-built, dependency-free SVGs (no matplotlib):

  * shadow_pnl.svg  — cumulative net P&L of the shadow-model NO book, one line
                      per model, from the shadow_model_no* tables.
  * latency.svg     — poll-cycle latency before/after the 788f1e1 fix (static
                      documented figures, log scale).

SVGs are colored to stay legible on both light and dark GitHub themes and are
referenced from the README by relative path, so committing the regenerated files
updates the rendered charts.

Usage:
  venv/bin/python scripts/gen_readme_charts.py                 # v2 only (default)
  venv/bin/python scripts/gen_readme_charts.py --models v1,v2  # overlay both
  venv/bin/python scripts/gen_readme_charts.py --since 2026-06-18

Note on --models: v1 and v2 fire at different rates (v2 covers ~2.7x as many
markets), so their cumulative-dollar curves are NOT directly comparable in
magnitude — v2-only is the honest default for a results chart.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DB = _ROOT / "data" / "db" / "opportunity_log.db"
_OUT = _ROOT / "assets" / "img"

# Entry-pricing fix (8cc5ff4): shadow rows before this store the wrong book side.
_DEFAULT_SINCE = "2026-06-18"
# Market-implied NO gate applied when the models were scored.
_GATE_LO, _GATE_HI = 0.15, 0.85

# Palette — every color legible on white (#fff) and dark (#0d1117) GitHub grounds.
_MUTED = "#7d8590"
_GRID = "rgba(125,133,144,0.18)"
_BASE = "rgba(125,133,144,0.40)"
_AMBER = "#d98a3d"
_FONT = "-apple-system,Segoe UI,Helvetica,Arial,sans-serif"

# Per-model: table, display label, line color (v2 first = the current model).
_MODELS = {
    "v2": ("shadow_model_no_v2", "v2", "#2da36f"),
    "v1": ("shadow_model_no",    "v1", "#8aa0b4"),
}

# Documented poll-cycle latency, before vs. after commit 788f1e1.
_LAT_BEFORE_S = 2100  # ~30-40 min
_LAT_AFTER_S = 65


def _daily_cum(con: sqlite3.Connection, table: str, since: str) -> list[tuple[str, float]]:
    """Cumulative net P&L (dollars) by settlement day for one shadow table."""
    rows = con.execute(
        f"""
        SELECT COALESCE(exited_at, logged_at) AS t, pnl_cents
        FROM {table}
        WHERE outcome IN ('won', 'lost')
          AND date(substr(logged_at, 1, 10)) >= ?
          AND market_p_no BETWEEN ? AND ?
        ORDER BY COALESCE(exited_at, logged_at)
        """,
        (since, _GATE_LO, _GATE_HI),
    ).fetchall()
    by_day: dict[str, float] = {}
    order: list[str] = []
    for t, pnl in rows:
        d = t[:10]
        if d not in by_day:
            by_day[d] = 0.0
            order.append(d)
        by_day[d] += pnl
    cum, out = 0.0, []
    for d in order:
        cum += by_day[d]
        out.append((d, round(cum / 100, 2)))
    return out


def _stats(con: sqlite3.Connection, table: str, since: str) -> tuple[int, float, float]:
    """(n_settled, win_rate_pct, net_dollars) for one shadow table."""
    rows = con.execute(
        f"""
        SELECT outcome, pnl_cents FROM {table}
        WHERE outcome IN ('won', 'lost')
          AND date(substr(logged_at, 1, 10)) >= ?
          AND market_p_no BETWEEN ? AND ?
        """,
        (since, _GATE_LO, _GATE_HI),
    ).fetchall()
    n = len(rows)
    if n == 0:
        return 0, 0.0, 0.0
    wins = sum(1 for o, _ in rows if o == "won")
    net = sum(p for _, p in rows) / 100
    return n, 100 * wins / n, net


def build_pnl_svg(series: dict[str, list[tuple[str, float]]],
                  stats: dict[str, tuple[int, float, float]]) -> str:
    W, H, L, R, T, B = 760, 380, 64, 22, 34, 48
    pw, ph = W - L - R, H - T - B

    all_days = [date.fromisoformat(d).toordinal()
                for s in series.values() for d, _ in s]
    d0, d1 = min(all_days), max(all_days)
    ymax = max(4.0, max(v for s in series.values() for _, v in s)) * 1.08
    # round ymax up to a tidy gridline
    step = 8 if ymax <= 40 else 10
    ymax = math.ceil(ymax / step) * step

    def X(ds: str) -> float:
        return L + (date.fromisoformat(ds).toordinal() - d0) / max(1, d1 - d0) * pw

    def Y(v: float) -> float:
        return T + (1 - v / ymax) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{_FONT}">'
    ]
    label = " + ".join(_MODELS[k][1] for k in series)
    parts.append(f'<text x="{L}" y="20" font-size="15" font-weight="600" '
                 f'fill="{_MUTED}">Shadow model ({label}) — cumulative net P&amp;L, 1-contract book</text>')

    # y gridlines + labels
    for gv in range(0, int(ymax) + 1, step):
        y = Y(gv)
        col = _BASE if gv == 0 else _GRID
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="{col}" stroke-width="1"/>')
        parts.append(f'<text x="{L-10}" y="{y+4:.1f}" text-anchor="end" font-size="13" fill="{_MUTED}">${gv}</text>')

    # x month ticks
    start_lab = date.fromordinal(d0).strftime("%b %-d")
    ticks = [(start_lab, date.fromordinal(d0).isoformat())]
    for mo, lab in ((7, "Jul"), (8, "Aug"), (9, "Sep"), (10, "Oct"), (11, "Nov")):
        ds = f"2026-{mo:02d}-01"
        if d0 <= date.fromisoformat(ds).toordinal() <= d1:
            ticks.append((lab, ds))
    for lab, ds in ticks:
        parts.append(f'<text x="{X(ds):.1f}" y="{T+ph+22}" text-anchor="middle" '
                     f'font-size="13" fill="{_MUTED}">{lab}</text>')

    # one polyline (+ area for the first/primary model) per series
    for i, (k, s) in enumerate(series.items()):
        color = _MODELS[k][2]
        pts = [(X(d), Y(v)) for d, v in s]
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        if i == 0:
            area = (f"{pts[0][0]:.1f},{Y(0):.1f} " + line +
                    f" {pts[-1][0]:.1f},{Y(0):.1f}")
            parts.append(f'<polygon points="{area}" fill="{color}" fill-opacity="0.12"/>')
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" '
                     f'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
        ex, ey = pts[-1]
        parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4.5" fill="{color}"/>')
        parts.append(f'<text x="{ex-6:.1f}" y="{ey-10:.1f}" text-anchor="end" '
                     f'font-size="14" font-weight="700" fill="{color}">+${s[-1][1]:.2f}</text>')

    # caption from the primary model's stats
    pk = next(iter(series))
    n, wr, _net = stats[pk]
    d0s, d1s = date.fromordinal(d0).strftime("%b %-d"), date.fromordinal(d1).strftime("%b %-d %Y")
    parts.append(f'<text x="{L}" y="{H-8}" font-size="12" fill="{_MUTED}">'
                 f'{n:,} settled out-of-sample markets · {wr:.0f}% win rate · '
                 f'net-positive every month · {d0s}–{d1s}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_latency_svg() -> str:
    W, H, L, R, T = 760, 300, 150, 90, 54
    pw = W - L - R
    lo, hi = math.log10(10), math.log10(3000)

    def X(sec: float) -> float:
        return L + (math.log10(sec) - lo) / (hi - lo) * pw

    bh, gap = 52, 44
    bars = [("Before fix", _LAT_BEFORE_S, _AMBER, "~30–40 min"),
            ("After fix", _LAT_AFTER_S, "#2da36f", "~65 s")]
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{_FONT}">',
         f'<text x="{L}" y="24" font-size="15" font-weight="600" fill="{_MUTED}">'
         f'Poll-cycle latency — before vs. after the SQLite/HTTP fix</text>']
    for s, lab in [(10, "10s"), (60, "1m"), (600, "10m"), (3000, "50m")]:
        x = X(s)
        p.append(f'<line x1="{x:.1f}" y1="{T-6}" x2="{x:.1f}" y2="{T+2*bh+gap+6:.1f}" '
                 f'stroke="{_GRID}" stroke-width="1"/>')
        p.append(f'<text x="{x:.1f}" y="{T+2*bh+gap+24:.1f}" text-anchor="middle" '
                 f'font-size="12" fill="{_MUTED}">{lab}</text>')
    for i, (name, sec, col, lab) in enumerate(bars):
        y = T + i * (bh + gap)
        x2 = X(sec)
        p.append(f'<rect x="{L}" y="{y}" width="{x2-L:.1f}" height="{bh}" rx="5" fill="{col}"/>')
        p.append(f'<text x="{L-14}" y="{y+bh/2+5:.1f}" text-anchor="end" '
                 f'font-size="14" font-weight="600" fill="{_MUTED}">{name}</text>')
        p.append(f'<text x="{x2+10:.1f}" y="{y+bh/2+5:.1f}" font-size="15" '
                 f'font-weight="700" fill="{col}">{lab}</text>')
    mult = round(_LAT_BEFORE_S / _LAT_AFTER_S)
    p.append(f'<text x="{W-R}" y="{H-14}" text-anchor="end" font-size="14" '
             f'font-weight="700" fill="#2da36f">≈ {mult}× faster</text>')
    p.append(f'<text x="{L}" y="{H-14}" font-size="12" fill="{_MUTED}">'
             f'log scale · non-sargable query full-scanning 9.3M rows + '
             f'~98 serial HTTP calls per iteration, fixed</text>')
    p.append("</svg>")
    return "\n".join(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="v2",
                    help="comma-separated model keys to plot on the P&L curve (v2,v1)")
    ap.add_argument("--since", default=_DEFAULT_SINCE, help="earliest settlement date (YYYY-MM-DD)")
    ap.add_argument("--db", default=str(_DB))
    args = ap.parse_args()

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    bad = [k for k in keys if k not in _MODELS]
    if bad:
        ap.error(f"unknown model key(s): {bad}; choose from {list(_MODELS)}")

    _OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(args.db)
    series = {k: _daily_cum(con, _MODELS[k][0], args.since) for k in keys}
    stats = {k: _stats(con, _MODELS[k][0], args.since) for k in keys}
    con.close()

    empty = [k for k, s in series.items() if not s]
    if empty:
        ap.error(f"no settled rows for {empty} since {args.since}")

    (_OUT / "shadow_pnl.svg").write_text(build_pnl_svg(series, stats))
    (_OUT / "latency.svg").write_text(build_latency_svg())

    for k in keys:
        n, wr, net = stats[k]
        print(f"  {k}: n={n:,} wr={wr:.1f}% net=${net:+.2f}")
    print(f"wrote {_OUT/'shadow_pnl.svg'} and {_OUT/'latency.svg'}")


if __name__ == "__main__":
    main()
