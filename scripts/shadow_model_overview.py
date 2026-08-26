"""
Generate dry_run_overview.txt-style reports for the V1 and V2 shadow models.

Reads from shadow_model_no and shadow_model_no_v2 tables and writes:
  shadow_model_no_overview.txt
  shadow_model_no_v2_overview.txt

The report-building logic lives in ``kalshi_bot.shadow_report`` so the running
bot can regenerate these files on its daily hook; this script is a thin CLI
wrapper around it.

Usage:
  venv/bin/python scripts/shadow_model_overview.py
  venv/bin/python scripts/shadow_model_overview.py --model v1
  venv/bin/python scripts/shadow_model_overview.py --model v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kalshi_bot.shadow_report import OUT_V1, OUT_V2, regenerate_overviews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["v1", "v2", "both"], default="both")
    args = parser.parse_args()

    models = ("v1", "v2") if args.model == "both" else (args.model,)
    counts = regenerate_overviews(models)

    for key, out in (("v1", OUT_V1), ("v2", OUT_V2)):
        if key in counts:
            print(f"Wrote {out}  ({counts[key]} entries)")


if __name__ == "__main__":
    main()
