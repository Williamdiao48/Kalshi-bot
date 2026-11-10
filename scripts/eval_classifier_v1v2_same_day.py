#!/usr/bin/env python3
"""Re-score the retired v1/v2 band-NO classifier shadow logs on same-day rows only.

The v1/v2 live loggers never checked a ticker's date, so most of their rows
scored TOMORROW's bands against today's running max/min (audit 2026-11-07).
This splits every settled row by whether it was logged on the contract's own
LST date (the NWS settlement day) and scores model vs market on each slice, so
the old classifier's record can be quoted on the subset where it was used as
designed.

Rows before 2026-06-18 are excluded: until commit 8cc5ff4 the shadow loggers
did not store the NO ask, so their market_p_no is a phantom.

    venv/bin/python scripts/eval_classifier_v1v2_same_day.py
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kalshi_bot import classifier_v3 as C  # noqa: E402
from scripts.eval_lookahead_vs_market import brier, clustered_bootstrap, logloss  # noqa: E402

_DATE_RE = re.compile(r"^([A-Z]+)-(\d{2}[A-Z]{3}\d{2})-")
FIRST_CLEAN = "2026-06-18"


def load(db: str, table: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    d = pd.read_sql_query(
        f"SELECT ticker, logged_at, is_high, outcome, model_p, market_p_no FROM {table} "
        f"WHERE outcome IN ('won','lost') AND logged_at >= ?", con, params=(FIRST_CLEAN,))
    con.close()
    rows = []
    for r in d.itertuples(index=False):
        m = _DATE_RE.match(r.ticker)
        if not m or m.group(1) not in C.SERIES_CITY:
            continue
        city, _ = C.SERIES_CITY[m.group(1)]
        tdate = datetime.strptime(m.group(2), "%y%b%d").date()
        ldate = C.lst_date(city, C.parse_ts(r.logged_at))
        rows.append(dict(
            city=city, date_target=tdate.isoformat(), side="HIGH" if r.is_high else "LOW",
            match="same-day" if tdate == ldate else ("NEXT-day" if tdate > ldate else "prev-day"),
            y=float(r.outcome == "won"), model_p=r.model_p, market_p_no=r.market_p_no,
        ))
    return pd.DataFrame(rows)


def score(s: pd.DataFrame, label: str, n_boot: int) -> None:
    s = s.reset_index(drop=True).copy()
    for c, k in (("model_p", "m"), ("market_p_no", "k")):
        p = s[c].clip(1e-6, 1 - 1e-6)
        s[f"br_{k}"], s[f"ll_{k}"] = brier(p, s["y"]), logloss(p, s["y"])
    br = clustered_bootstrap(s, "br_m", "br_k", n_boot)
    ll = clustered_bootstrap(s, "ll_m", "ll_k", n_boot)
    print(f"  {label:<16} n={len(s):6,} city-days={s.groupby(['city', 'date_target']).ngroups:5,} "
          f"NO rate {s['y'].mean():.3f}  model {s['model_p'].mean():.3f}  "
          f"mkt {s['market_p_no'].mean():.3f} | Brier {s['br_m'].mean():.4f} vs "
          f"{s['br_k'].mean():.4f}  diff {br[0]:+.4f} [{br[1]:+.4f},{br[2]:+.4f}]  "
          f"LL diff {ll[0]:+.4f} [{ll[1]:+.4f},{ll[2]:+.4f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/db/opportunity_log.db")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    for table in ("shadow_model_no", "shadow_model_no_v2"):
        d = load(a.db, table)
        if d.empty:
            continue
        print(f"\n=== {table}: {len(d):,} settled rows since {FIRST_CLEAN}")
        print(pd.crosstab(d["match"], d["side"]).to_string())
        g = d[d["market_p_no"].between(0.15, 0.85)]
        print(f"model vs market (NO ask in [15,85]c, diff <0 => model better):")
        for (side, match), s in g.groupby(["side", "match"]):
            if len(s) >= 30:
                score(s, f"{side} {match}", a.boot)
        score(g[g["match"] == "same-day"], "ALL same-day", a.boot)
    print("\n(the NO ask sits half a spread above the mid, so it slightly overstates P(NO))")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
