"""NLI-based directional re-filtering of heuristically labeled pairs.

Reads the kept=True records from labeled_pairs.jsonl (produced by label_heuristic.py)
and re-filters them using a local NLI cross-encoder model to verify that each article's
*content* directionally agrees with the market outcome.

The heuristic labeler assigns labels based on market outcome alone (relevant article +
market resolved YES → label=1). The problem: an article can be topically relevant but
give no directional signal. This script discards those ambiguous pairs, keeping only
examples where the NLI model confirms the article entails the market direction.

Model: cross-encoder/nli-deberta-v3-small (~85MB, CPU/MPS, no API needed)

Usage:
    venv/bin/python scripts/label_with_nli.py            # full run
    venv/bin/python scripts/label_with_nli.py --dry-run  # stats only

Output:
    data/labeled_pairs_nli.jsonl  — same schema as labeled_pairs.jsonl

Then rebuild the training set:
    INPUT_FILE=data/labeled_pairs_nli.jsonl venv/bin/python scripts/build_training_set.py

Environment variables:
    NLI_MODEL           HuggingFace model ID (default: cross-encoder/nli-deberta-v3-small)
    NLI_BATCH_SIZE      Inference batch size (default: 32)
    NLI_AMBIGUITY_GAP   Min |entailment-contradiction| to keep a pair (default: 0.15)
    NLI_ARTICLE_CHARS   Article chars passed to NLI (default: 1500)
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_root        = Path(__file__).parent.parent
INPUT_FILE   = _root / "data" / "labeled_pairs.jsonl"
OUTPUT_FILE  = _root / "data" / "labeled_pairs_nli.jsonl"

NLI_MODEL       = os.environ.get("NLI_MODEL",        "cross-encoder/nli-deberta-v3-small")
NLI_BATCH_SIZE  = int(os.environ.get("NLI_BATCH_SIZE",   "16"))
AMBIGUITY_GAP   = float(os.environ.get("NLI_AMBIGUITY_GAP", "0.05"))
ARTICLE_CHARS   = int(os.environ.get("NLI_ARTICLE_CHARS",  "1500"))

# NLI label order returned by cross-encoder/nli-* models
# (contradiction=0, entailment=1, neutral=2) — confirmed for DeBERTa-v3 NLI models
_ENTAILMENT = 1


def _build_hypotheses(market_title: str) -> tuple[str, str]:
    """Return (yes_hypothesis, no_hypothesis) for a market title.

    'Will the Fed raise rates?' →
        yes: 'The Fed will raise rates.'
        no:  'The Fed will not raise rates.'

    Using symmetric hypotheses removes the NLI model's inherent directional
    bias: instead of comparing against an absolute threshold, we compare
    YES-entailment vs NO-entailment for the same article, so the bias cancels.
    """
    title = market_title.strip()
    m = re.match(r"^[Ww]ill\s+(.+?)\??\s*$", title)
    if m:
        core = m.group(1).capitalize()
        return (f"{core} will happen.", f"{core} will not happen.")
    return (title, f"This will not happen: {title}")


def main(dry_run: bool = False) -> None:
    if not INPUT_FILE.exists():
        logging.error("Input not found: %s", INPUT_FILE)
        sys.exit(1)

    # ---- Load kept pairs ---------------------------------------------------
    all_records: list[dict] = []
    kept_records: list[dict] = []

    for line in INPUT_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        all_records.append(rec)
        if rec.get("kept"):
            kept_records.append(rec)

    logging.info(
        "Loaded %d total records; %d with kept=True.",
        len(all_records), len(kept_records),
    )

    if not kept_records:
        logging.error("No kept=True records found. Run label_heuristic.py first.")
        sys.exit(1)

    # ---- Load NLI model ----------------------------------------------------
    logging.info("Loading NLI model: %s (CPU) …", NLI_MODEL)
    from sentence_transformers import CrossEncoder  # type: ignore
    # Force CPU: DeBERTa-v3's disentangled attention causes MPS memory spikes
    # that OOM even on 16 GB+ unified memory. CPU is slower but stable.
    model = CrossEncoder(NLI_MODEL, device="cpu")
    logging.info("Model ready.")

    # ---- Build symmetric inference pairs -----------------------------------
    # For each pair we score TWO hypotheses: YES and NO.
    # Comparing them cancels the NLI model's inherent directional bias.
    yes_pairs: list[tuple[str, str]] = []
    no_pairs:  list[tuple[str, str]] = []

    for rec in kept_records:
        hyp_yes, hyp_no = _build_hypotheses(rec.get("market_title", ""))
        body = (rec.get("article_body") or "")[:ARTICLE_CHARS]
        yes_pairs.append((body, hyp_yes))
        no_pairs.append((body, hyp_no))

    logging.info(
        "Running symmetric NLI inference on %d pairs × 2 (batch_size=%d)…",
        len(kept_records), NLI_BATCH_SIZE,
    )
    yes_scores = model.predict(
        yes_pairs, batch_size=NLI_BATCH_SIZE, show_progress_bar=True, apply_softmax=True,
    )
    logging.info("YES hypotheses scored. Scoring NO hypotheses…")
    no_scores = model.predict(
        no_pairs, batch_size=NLI_BATCH_SIZE, show_progress_bar=True, apply_softmax=True,
    )
    logging.info("Inference complete.")

    # ---- Apply directional filter ------------------------------------------
    # gap > 0  → article leans YES  (yes_entailment > no_entailment)
    # gap < 0  → article leans NO
    kept_nli   = 0
    disc_ambig = 0
    disc_contr = 0

    updated: list[dict] = []

    for rec, yes_row, no_row in zip(kept_records, yes_scores, no_scores):
        yes_ent = float(yes_row[_ENTAILMENT])
        no_ent  = float(no_row[_ENTAILMENT])
        gap     = yes_ent - no_ent   # positive = article leans YES

        market_result = (rec.get("market_result") or "").lower()
        nli_direction = "yes" if gap > 0 else "no"

        if abs(gap) < AMBIGUITY_GAP:
            disc_ambig += 1
            rec = {
                **rec,
                "kept":           False,
                "claude_reasoning": (
                    f"nli_ambiguous: yes_ent={yes_ent:.3f} no_ent={no_ent:.3f} gap={gap:.3f}"
                ),
                "sample_weight":  0.0,
            }
        elif nli_direction != market_result:
            disc_contr += 1
            rec = {
                **rec,
                "kept":           False,
                "claude_reasoning": (
                    f"nli_contradiction: nli={nli_direction} market={market_result} "
                    f"yes_ent={yes_ent:.3f} no_ent={no_ent:.3f}"
                ),
                "sample_weight":  0.0,
            }
        else:
            kept_nli += 1
            # confidence = how strongly the article favours the correct direction
            # Floor at 0.1 so near-zero-gap pairs still contribute to training
            confidence = max(abs(gap), 0.1)
            rec = {
                **rec,
                "kept":           True,
                "sample_weight":  round(min(confidence, 1.0), 4),
                "claude_reasoning": (
                    f"nli_symmetric: yes_ent={yes_ent:.3f} no_ent={no_ent:.3f} "
                    f"gap={gap:+.3f}"
                ),
            }

        updated.append(rec)

    # ---- Summary -----------------------------------------------------------
    total = len(kept_records)
    logging.info("=" * 60)
    logging.info("NLI filtering complete.")
    logging.info("  Input pairs:              %d", total)
    logging.info("  Kept (directional match): %d  (%.1f%%)", kept_nli, 100 * kept_nli / max(total, 1))
    logging.info("  Discarded — ambiguous:    %d  (%.1f%%)", disc_ambig, 100 * disc_ambig / max(total, 1))
    logging.info("  Discarded — contradicts:  %d  (%.1f%%)", disc_contr, 100 * disc_contr / max(total, 1))

    if dry_run:
        logging.info("Dry run — no output written.")
        return

    # ---- Write output ------------------------------------------------------
    # Write updated kept records + pass-through all originally kept=False records unchanged
    kept_false_originals = [r for r in all_records if not r.get("kept")]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for rec in kept_false_originals:
            fh.write(json.dumps(rec) + "\n")
        for rec in updated:
            fh.write(json.dumps(rec) + "\n")

    logging.info(
        "Wrote %d records to %s (%d kept=True).",
        len(kept_false_originals) + len(updated),
        OUTPUT_FILE,
        kept_nli,
    )
    logging.info(
        "Next step: INPUT_FILE=data/labeled_pairs_nli.jsonl "
        "venv/bin/python scripts/build_training_set.py"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLI-based directional label filtering.")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only; do not write.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
