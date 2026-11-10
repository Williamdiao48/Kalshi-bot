#!/usr/bin/env python3
"""Training rows for the band-NO classifier v3, strictly point-in-time.

For every settled B-band (data/classifier_v3/labels.parquet, from
fetch_temp_band_labels.py) and every hourly METAR decision instant on the
contract's own LST date before the market closes, emit the feature row that
kalshi_bot.classifier_v3 would have produced live at that instant. Only rows
inside the model domain (classifier_v3.GATE) are kept.

The feature inputs are the bot's own forecast_shadow_log rows logged at or
before the instant. Nothing comes from a historical-forecast archive, which is
the look-ahead that sank v1/v2's training data.

    venv/bin/python scripts/build_classifier_v3_data.py
    venv/bin/python scripts/build_classifier_v3_data.py --selftest   # parity + leak canary
"""
from __future__ import annotations

import argparse
import math
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kalshi_bot import classifier_v3 as C  # noqa: E402

DATA = Path("data/classifier_v3")
LABELS = DATA / "labels.parquet"
OUT = DATA / "train_rows.parquet"


def load_all_rows(db: str, start: str) -> dict[tuple[str, str, int], list[C.Row]]:
    """Bulk version of C.load_rows: same columns, same parse, same sort."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    q = (f"SELECT city, date_target, is_high, logged_at, source, forecast_f "
         f"FROM forecast_shadow_log WHERE date_target >= ? "
         f"AND source IN ({','.join('?' * len(C.SOURCES))})")
    groups: dict[tuple[str, str, int], list[C.Row]] = defaultdict(list)
    for city, d, hi, t, s, v in con.execute(q, (start, *C.SOURCES)):
        if v is not None:
            groups[(city, d, int(hi))].append((C.parse_ts(t), s, float(v)))
    con.close()
    return {k: C.sort_rows(v) for k, v in groups.items()}


def build(labels: pd.DataFrame, groups: dict) -> pd.DataFrame:
    out = []
    for (city, d, hi), bands in labels.groupby(["city", "date_target", "is_high"]):
        rows = groups.get((city, d, int(hi)))
        if not rows:
            continue
        day = date.fromisoformat(d)
        close = pd.Timestamp(bands["close_time"].min()).to_pydatetime()
        band_list = list(zip(bands["ticker"], bands["band_ceil"].astype(int), bands["won_no"]))
        for asof in C.obs_times(rows):
            if asof >= close or C.lst_date(city, asof) != day:
                continue
            st = C.day_state(rows, asof, city, day)
            if st is None:
                continue
            for ticker, ceil, won in band_list:
                f = C.band_features(st, ceil, city)
                if C.in_domain(f["margin_f"], bool(hi)):
                    out.append({"ticker": ticker, "city": city, "date_target": d,
                                "is_high": int(hi), "asof": asof.isoformat(),
                                "band_ceil": ceil, "won_no": int(won), **f})
    return pd.DataFrame(out)


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def selftest(db: str, n: int) -> None:
    """(1) Parity: rebuild sampled rows through the LIVE path (per-city-day SQL +
    features_at with now = asof) and require exact equality.
    (2) Leak canary: recompute with every row logged after asof deleted; the
    features must not move."""
    df = pd.read_parquet(OUT)
    sample = df.sample(min(n, len(df)), random_state=0)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    parity_bad = leak_bad = missing = 0
    for r in sample.itertuples(index=False):
        asof = C.parse_ts(r.asof)
        d = date.fromisoformat(r.date_target)
        # now = asof exactly: asof + epsilon can cross LST midnight for a row logged
        # in the day's last minute, which the live path (correctly) refuses.
        got = C.features_at(con, r.city, d, bool(r.is_high), [(r.ticker, int(r.band_ceil))], asof)
        if got is None or got[0] != asof or not got[1]:
            missing += 1
            continue
        live = got[1][0]["feats"]
        if not all(_same(getattr(r, f), live[f]) for f in C.FEATURES):
            parity_bad += 1
        rows = C.load_rows(con, r.city, d, bool(r.is_high))
        past = [x for x in rows if x[0] <= asof]
        trunc = C.band_features(C.day_state(past, asof, r.city, d), int(r.band_ceil), r.city)
        if not all(_same(trunc[f], live[f]) for f in C.FEATURES):
            leak_bad += 1
    con.close()
    print(f"selftest on {len(sample)} rows: parity mismatches {parity_bad}, "
          f"leak-canary mismatches {leak_bad}, unreachable live {missing}")
    if parity_bad or leak_bad or missing:
        raise SystemExit("SELFTEST FAILED")
    print("SELFTEST PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/db/opportunity_log.db")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    if a.selftest:
        selftest(a.db, a.n)
        return
    labels = pd.read_parquet(LABELS)
    groups = load_all_rows(a.db, labels["date_target"].min())
    df = build(labels, groups)
    df.to_parquet(OUT, index=False)
    for hi, g in df.groupby("is_high"):
        print(f"{'HIGH' if hi else 'LOW '}: {len(g):,} rows, {g['ticker'].nunique():,} bands, "
              f"{g.groupby(['city', 'date_target']).ngroups:,} city-days, "
              f"{g['date_target'].min()} -> {g['date_target'].max()}, "
              f"NO rate {g['won_no'].mean():.3f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    random.seed(0)
    main()
