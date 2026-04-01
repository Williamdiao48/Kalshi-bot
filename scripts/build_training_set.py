"""Assemble the final training set from labeled (market, article) pairs.

Reads labeled_pairs.jsonl, keeps only `kept=true` records, formats the
input text, performs a stratified market-level train/val/test split, and
writes the result to data/training_set.jsonl.

Key design decisions:
  - Split is by MARKET, not by ARTICLE.  All articles for a given market go
    into the same split.  This prevents train/test contamination where the
    model sees different articles about the same market in both splits and
    learns market-specific patterns rather than generalisable signal.
  - Stratification is by market_category so every category is proportionally
    represented in each split.
  - input_text uses [MARKET] / [SEP] tokens that DistilBERT's tokenizer
    handles natively.  The article portion is truncated to 1500 chars so the
    combined string stays within DistilBERT's 512-token window.
  - market_text and article_text are saved separately so the training script
    can encode them as a sentence-pair if preferred.

Pipeline position:
    label_with_claude.py
        → THIS SCRIPT
            → train_model.py  (Phase 2)

Usage:
    venv/bin/python scripts/build_training_set.py

Input:
    data/labeled_pairs.jsonl

Output:
    data/training_set.jsonl  — one record per (market, article) pair, fields:
        market_ticker, market_title, market_category, article_url,
        article_headline, article_date, hours_before_resolution,
        input_text, market_text, article_text,
        training_label, sample_weight, split,
        claude_score, claude_direction, claude_reasoning
    data/training_set_stats.json — summary statistics for sanity-checking

Environment variables:
    TRAIN_SEED        Random seed for reproducible splits (default: 42).
    TRAIN_FRAC        Fraction of markets for train split (default: 0.70).
    VAL_FRAC          Fraction for validation split (default: 0.15).
                      Test gets the remainder (1 - TRAIN_FRAC - VAL_FRAC).
    ARTICLE_BODY_CHARS  Characters of body included in input_text (default: 1500).
"""

import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_root        = Path(__file__).parent.parent
INPUT_FILE   = Path(os.environ.get("INPUT_FILE",  _root / "data" / "labeled_pairs.jsonl"))
OUTPUT_FILE  = Path(os.environ.get("OUTPUT_FILE", _root / "data" / "training_set.jsonl"))
STATS_FILE   = Path(__file__).parent.parent / "data" / "training_set_stats.json"

TRAIN_SEED: int        = int(os.environ.get("TRAIN_SEED", "42"))
TRAIN_FRAC: float      = float(os.environ.get("TRAIN_FRAC", "0.70"))
VAL_FRAC: float        = float(os.environ.get("VAL_FRAC", "0.15"))
ARTICLE_BODY_CHARS: int = int(os.environ.get("ARTICLE_BODY_CHARS", "1500"))

# Minimum markets per category before we stop stratifying that category
# and just pool with "other" (avoids empty val/test splits for tiny categories)
MIN_MARKETS_TO_STRATIFY = 5


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def _format_market_text(record: dict[str, Any]) -> str:
    """Produce the market-side text for the model input.

    Includes the title and, if available, a shortened version of the
    resolution criteria to give the model the exact resolution condition.
    """
    title = (record.get("market_title") or "").strip()
    rules = (record.get("rules_primary") or "").strip()
    if rules and rules.lower() != title.lower():
        # Truncate rules to avoid overwhelming the market-side budget (~100 tokens)
        rules_short = rules[:300].rsplit(" ", 1)[0] if len(rules) > 300 else rules
        return f"{title} Resolution: {rules_short}"
    return title


def _format_article_text(record: dict[str, Any]) -> str:
    """Produce the article-side text: headline + body."""
    headline = (record.get("article_headline") or "").strip()
    body     = (record.get("article_body") or "").strip()[:ARTICLE_BODY_CHARS]
    if headline and body:
        # Avoid duplicating if the body starts with the headline
        if body.lower().startswith(headline.lower()[:40]):
            return body
        return f"{headline}. {body}"
    return headline or body


def _build_input_text(market_text: str, article_text: str) -> str:
    """Combine market and article into the DistilBERT [MARKET]...[SEP]... format."""
    return f"[MARKET] {market_text} [SEP] {article_text}"


# ---------------------------------------------------------------------------
# Market-level stratified split
# ---------------------------------------------------------------------------

def _stratified_split(
    market_tickers: list[str],
    category_of: dict[str, str],
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> dict[str, str]:
    """Assign each market_ticker to 'train', 'val', or 'test'.

    Markets are split within each category so every category is proportionally
    represented.  Categories with fewer than MIN_MARKETS_TO_STRATIFY markets
    are pooled into a single '_small_' group and split together.
    """
    # Group markets by category
    by_cat: dict[str, list[str]] = defaultdict(list)
    for ticker in market_tickers:
        cat = category_of.get(ticker, "other")
        by_cat[cat].append(ticker)

    # Pool small categories
    small: list[str] = []
    large: dict[str, list[str]] = {}
    for cat, tickers in by_cat.items():
        if len(tickers) < MIN_MARKETS_TO_STRATIFY:
            small.extend(tickers)
        else:
            large[cat] = tickers
    if small:
        large["_small_"] = small

    assignments: dict[str, str] = {}

    for cat, tickers in large.items():
        shuffled = tickers[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, round(n * train_frac))
        n_val   = max(1, round(n * val_frac))
        # Ensure at least 1 in test if n >= 3
        if n >= 3 and n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)

        for i, ticker in enumerate(shuffled):
            if i < n_train:
                assignments[ticker] = "train"
            elif i < n_train + n_val:
                assignments[ticker] = "val"
            else:
                assignments[ticker] = "test"

    return assignments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_FILE.exists():
        logging.error(
            "Input file not found: %s\nRun label_with_claude.py first.",
            INPUT_FILE,
        )
        sys.exit(1)

    # Load and filter to kept=True records only
    all_records: list[dict[str, Any]] = []
    kept_records: list[dict[str, Any]] = []
    for line in INPUT_FILE.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        all_records.append(rec)
        if rec.get("kept"):
            # Validate: kept records must have a training label
            if rec.get("training_label") not in (0, 1):
                logging.warning(
                    "kept=True but training_label=%s — skipping %s / %s",
                    rec.get("training_label"),
                    rec.get("market_ticker"), rec.get("article_url", "")[:60],
                )
                continue
            kept_records.append(rec)

    logging.info(
        "Loaded %d total labeled pairs; %d kept (%.0f%%).",
        len(all_records), len(kept_records),
        100 * len(kept_records) / max(len(all_records), 1),
    )

    if not kept_records:
        logging.error("No kept records found. Check label_with_claude.py output.")
        sys.exit(1)

    # Deduplicate within (market_ticker, article_url) — keep highest-scoring
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in kept_records:
        key = (rec.get("market_ticker", ""), rec.get("article_url", ""))
        existing = dedup.get(key)
        if existing is None or rec.get("claude_score", 0) > existing.get("claude_score", 0):
            dedup[key] = rec
    kept_records = list(dedup.values())
    logging.info("After deduplication: %d pairs.", len(kept_records))

    # Build category and market maps
    market_to_category: dict[str, str] = {}
    market_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in kept_records:
        ticker = rec.get("market_ticker", "UNKNOWN")
        market_to_category[ticker] = rec.get("market_category", "other")
        market_to_records[ticker].append(rec)

    all_market_tickers = list(market_to_records.keys())
    logging.info("Unique markets: %d", len(all_market_tickers))

    # Stratified market-level split
    rng = random.Random(TRAIN_SEED)
    split_of = _stratified_split(
        all_market_tickers,
        market_to_category,
        TRAIN_FRAC,
        VAL_FRAC,
        rng,
    )

    # Build output records
    output_records: list[dict[str, Any]] = []
    for ticker, records in market_to_records.items():
        split = split_of.get(ticker, "train")
        for rec in records:
            market_text  = _format_market_text(rec)
            article_text = _format_article_text(rec)
            input_text   = _build_input_text(market_text, article_text)

            output_records.append({
                "market_ticker":           ticker,
                "market_title":            rec.get("market_title", ""),
                "market_category":         rec.get("market_category", "other"),
                "article_url":             rec.get("article_url", ""),
                "article_headline":        rec.get("article_headline", ""),
                "article_date":            rec.get("article_date", ""),
                "hours_before_resolution": rec.get("hours_before_resolution"),
                "input_text":              input_text,
                "market_text":             market_text,
                "article_text":            article_text,
                "training_label":          rec["training_label"],
                "sample_weight":           rec.get("sample_weight", rec.get("claude_score", 1.0)),
                "split":                   split,
                "claude_score":            rec.get("claude_score"),
                "claude_direction":        rec.get("claude_direction"),
                "claude_reasoning":        rec.get("claude_reasoning"),
            })

    # Sort: train first, then val, then test (cosmetic, not required)
    _order = {"train": 0, "val": 1, "test": 2}
    output_records.sort(key=lambda r: _order.get(r["split"], 3))

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for rec in output_records:
            fh.write(json.dumps(rec) + "\n")

    logging.info("Wrote %d records to %s.", len(output_records), OUTPUT_FILE)

    # ---------------------------------------------------------------------------
    # Stats
    # ---------------------------------------------------------------------------
    split_counts  = Counter(r["split"]    for r in output_records)
    label_counts  = Counter(r["training_label"] for r in output_records)
    cat_counts    = Counter(r["market_category"] for r in output_records)
    score_values  = [r["claude_score"] for r in output_records if r["claude_score"] is not None]

    # Per-split label balance
    split_label: dict[str, Counter] = defaultdict(Counter)
    for r in output_records:
        split_label[r["split"]][r["training_label"]] += 1

    logging.info("=" * 60)
    logging.info("Training set summary")
    logging.info("  Total pairs:     %d", len(output_records))
    logging.info("  Label balance:   YES(1)=%d  NO(0)=%d  ratio=%.2f",
                 label_counts[1], label_counts[0],
                 label_counts[1] / max(label_counts[0], 1))
    logging.info("  Split sizes:")
    for sp in ("train", "val", "test"):
        n   = split_counts[sp]
        yes = split_label[sp][1]
        no  = split_label[sp][0]
        logging.info("    %-6s  %4d  (YES=%d NO=%d)", sp, n, yes, no)
    logging.info("  Categories:")
    for cat, n in cat_counts.most_common():
        logging.info("    %-25s %d", cat, n)
    if score_values:
        avg_w = sum(score_values) / len(score_values)
        logging.info("  Avg sample weight: %.3f", avg_w)

    # Warn if label balance is very skewed
    total = len(output_records)
    yes_frac = label_counts[1] / max(total, 1)
    if yes_frac > 0.75 or yes_frac < 0.25:
        logging.warning(
            "Label imbalance detected: %.0f%% YES, %.0f%% NO. "
            "Consider class-weighted loss in train_model.py.",
            100 * yes_frac, 100 * (1 - yes_frac),
        )

    # Sample — show one example per split
    logging.info("Sample records:")
    shown: set[str] = set()
    for r in output_records:
        if r["split"] not in shown:
            shown.add(r["split"])
            logging.info(
                "  [%s] label=%d weight=%.2f  %s",
                r["split"], r["training_label"], r["sample_weight"],
                r["input_text"][:100],
            )

    # Write stats JSON
    stats = {
        "total":          len(output_records),
        "label_1_yes":    label_counts[1],
        "label_0_no":     label_counts[0],
        "yes_fraction":   round(yes_frac, 4),
        "split_counts":   dict(split_counts),
        "category_counts": dict(cat_counts),
        "avg_sample_weight": round(avg_w, 4) if score_values else None,
        "train_seed":     TRAIN_SEED,
        "train_frac":     TRAIN_FRAC,
        "val_frac":       VAL_FRAC,
        "unique_markets": len(all_market_tickers),
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logging.info("Stats written to %s.", STATS_FILE)


if __name__ == "__main__":
    main()
