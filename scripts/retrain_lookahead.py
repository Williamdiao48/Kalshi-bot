#!/usr/bin/env python3
"""Weekly retrain of the lookahead HIGH model, with a promotion gate.

Run by launchd (scripts/install_retrain_job.sh). Steps:
  1. refresh CLI labels (the bot does this daily; repeated here in case it was down)
  2. rebuild point-in-time features
  3. walk-forward + fit a CANDIDATE model
  4. promote it over the live pkl only if the recent out-of-sample checks pass;
     otherwise keep the current model and exit non-zero

The running bot notices the new pkl's mtime and reloads it — no restart needed.
If this job keeps failing, the bot pauses lookahead logging once the live model
is STALE_SKIP_DAYS old (see scripts/shadow_lookahead.py).

    venv/bin/python scripts/retrain_lookahead.py            # what launchd runs
    venv/bin/python scripts/retrain_lookahead.py --dry-run  # train + gate, don't promote
"""
from __future__ import annotations

import argparse
import datetime as dt
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LIVE = ROOT / "data/models/lookahead_high.pkl"
CANDIDATE = ROOT / "data/models/lookahead_high.candidate.pkl"
OOF = ROOT / "data/lookahead/oof_predictions_high.parquet"

GATE_DAYS = 30           # judge the candidate on its most recent OOS month
MAX_CITY_BIAS = 1.5      # deg F, per-city mean error over GATE_DAYS
RECAL_RANGE = (0.8, 1.4)


def _run(*args: str) -> None:
    print(f"$ {' '.join(args)}", flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def gate(oof: pd.DataFrame, recal: float) -> list[str]:
    """Reasons to reject the candidate (empty = promote)."""
    d = pd.to_datetime(oof["date_target"])
    r = oof[d >= d.max() - pd.Timedelta(days=GATE_DAYS)]
    fails = []
    mae_m = float(np.mean(np.abs(r["mu"] - r["actual_f"])))
    mae_c = float(np.mean(np.abs(r["consensus_median"] - r["actual_f"])))
    print(f"gate: last {GATE_DAYS}d MAE model {mae_m:.3f} vs consensus {mae_c:.3f}")
    if not mae_m <= mae_c:
        fails.append(f"model MAE {mae_m:.3f} > consensus {mae_c:.3f}")
    bias = (r["mu"] - r["actual_f"]).groupby(r["city"]).mean()
    worst = bias.abs().idxmax()
    print(f"gate: worst city bias {bias[worst]:+.2f}F ({worst})")
    if abs(bias[worst]) > MAX_CITY_BIAS:
        fails.append(f"{worst} bias {bias[worst]:+.2f}F beyond +/-{MAX_CITY_BIAS}")
    print(f"gate: recal factor {recal:.3f}")
    if not RECAL_RANGE[0] <= recal <= RECAL_RANGE[1]:
        fails.append(f"recal {recal:.3f} outside {RECAL_RANGE}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(f"=== lookahead retrain {dt.datetime.now().isoformat(timespec='seconds')} ===")

    year = dt.date.today().year
    _run("scripts/backfill_cli_labels.py", "--years", str(year))
    _run("scripts/build_lookahead_features.py")
    _run("scripts/train_lookahead_high_model.py", "--save-model", str(CANDIDATE))

    with open(CANDIDATE, "rb") as f:
        cand = pickle.load(f)
    fails = gate(pd.read_parquet(OOF), cand["recal"])
    if fails:
        print("REJECTED — keeping live model:\n  " + "\n  ".join(fails))
        CANDIDATE.unlink(missing_ok=True)
        return 1
    if a.dry_run:
        print(f"PASS (dry run) — candidate {cand['version']} not promoted")
        CANDIDATE.unlink(missing_ok=True)
        return 0
    CANDIDATE.replace(LIVE)                 # atomic; the bot hot-reloads on mtime
    print(f"PROMOTED {cand['version']} -> {LIVE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
