"""Heuristic weak-supervision labeling for (market, article) pairs.

Replaces the Gemini/Claude API labeler with a free, local, deterministic pipeline
that can process all 66,000+ pairs in ~15 minutes on CPU.

Four signals per pair:
  1. Semantic similarity  — sentence-transformers all-MiniLM-L6-v2 cosine similarity
                            between article body and market title+rules.  Captures
                            synonyms ("step down" ≈ "resign") that TF-IDF misses.
  2. Keyword density      — fraction of market-title content words found in article.
  3. Temporal proximity   — exp(-hours_before_resolution / 48) decay.
  4. VADER sentiment gate — discards pairs where article sentiment strongly contradicts
                            the market outcome (direction-blindness mitigation).

Hard gates (applied before scoring):
  - Article body < 200 chars         → skip
  - hours_before_resolution > 168    → kept=False  (> 1 week — too early)
  - hours_before_resolution < 12
    AND category in political/
    regulatory/corporate             → kept=False  (post-outcome confirmation articles)
  - Per-market cap of 10 kept pairs  → kept=False  (prevent dominant markets)

Combined score:
  heuristic_score = 0.6 * semantic_sim + 0.2 * keyword_density + 0.2 * temporal_score
  Keep threshold: heuristic_score > 0.35

Output format is schema-compatible with labeled_pairs.jsonl so build_training_set.py
requires zero changes.  Field names claude_score / claude_direction / claude_reasoning
are reused for pipeline compatibility.

Usage:
    venv/bin/python scripts/label_heuristic.py            # full run
    venv/bin/python scripts/label_heuristic.py --dry-run  # stats only, no write

Environment variables:
    HEURISTIC_SCORE_THRESHOLD   Keep threshold (default: 0.35)
    HEURISTIC_MARKET_CAP        Max kept pairs per market (default: 25)
    HEURISTIC_MAX_LOOKBACK_H    Max hours before resolution to include (default: 720)
    HEURISTIC_EMBARGO_H         Min hours before resolution for political/regulatory/
                                corporate markets (default: 12)
    HEURISTIC_VADER_THRESHOLD   VADER compound magnitude for contradiction gate
                                (default: 0.7 — only very strong polarity discarded)
    HEURISTIC_ARTICLE_CHARS     Article chars sent to MiniLM (default: 3000)
    HEURISTIC_ENCODE_BATCH      MiniLM encoding batch size (default: 512)
"""

import argparse
import json
import logging
import math
import os
import re
import string
import sys
from collections import defaultdict
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
INPUT_FILE  = Path(__file__).parent.parent / "data" / "market_article_pairs.jsonl"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "labeled_pairs.jsonl"

SCORE_THRESHOLD:  float = float(os.environ.get("HEURISTIC_SCORE_THRESHOLD",  "0.35"))
MARKET_CAP:       int   = int(os.environ.get("HEURISTIC_MARKET_CAP",         "25"))
MAX_LOOKBACK_H:   float = float(os.environ.get("HEURISTIC_MAX_LOOKBACK_H",   "720"))
EMBARGO_H:        float = float(os.environ.get("HEURISTIC_EMBARGO_H",        "12"))
VADER_THRESHOLD:  float = float(os.environ.get("HEURISTIC_VADER_THRESHOLD",  "0.7"))
ARTICLE_CHARS:    int   = int(os.environ.get("HEURISTIC_ARTICLE_CHARS",      "3000"))
ENCODE_BATCH:     int   = int(os.environ.get("HEURISTIC_ENCODE_BATCH",       "512"))

# Market categories where the temporal embargo applies (original Kalshi/Manifold labels)
_EMBARGO_CATEGORIES: frozenset[str] = frozenset({
    "political_action", "regulatory", "corporate",
})
# kalshi_category values (from filter_kalshi_markets.py) that also get the embargo
_KALSHI_EMBARGO_CATEGORIES: frozenset[str] = frozenset({
    "politics",
})

# Minimal English stopwords (no NLTK dependency)
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "up", "down", "out", "off",
    "and", "or", "but", "not", "no", "nor", "so", "yet", "both", "either",
    "neither", "each", "more", "most", "other", "some", "such", "than",
    "that", "this", "these", "those", "if", "when", "where", "which",
    "who", "whom", "whose", "what", "how", "why", "all", "any", "it",
    "its", "he", "she", "they", "we", "i", "you", "their", "our", "your",
    "his", "her", "my", "about", "between", "during", "while",
})

_PUNCT_RE = re.compile(r"[^\w\s]")


def _market_keywords(title: str, rules: str) -> list[str]:
    """Extract meaningful content words from market title and rules."""
    text = (title + " " + rules).lower()
    text = _PUNCT_RE.sub(" ", text)
    words = [w for w in text.split() if w and w not in _STOPWORDS and not w.isdigit()]
    return words


def _keyword_density(keywords: list[str], body: str) -> float:
    """Fraction of market keywords that appear in the article body."""
    if not keywords:
        return 0.0
    body_lower = body.lower()
    hits = sum(1 for kw in keywords if kw in body_lower)
    return hits / len(keywords)


def _temporal_score(hours: float) -> float:
    """Exponential decay: peaks near resolution, low for distant articles."""
    return math.exp(-max(hours, 1.0) / 48.0)


def _best_vader_sentence(body: str, keywords: list[str]) -> str:
    """Find the sentence in body with the highest keyword density for VADER."""
    sentences = re.split(r"(?<=[.!?])\s+", body)
    if not sentences:
        return body[:500]
    best_sentence = sentences[0]
    best_count    = 0
    for sent in sentences:
        sent_lower = sent.lower()
        count = sum(1 for kw in keywords if kw in sent_lower)
        if count > best_count:
            best_count    = count
            best_sentence = sent
    return best_sentence


def _load_done_pairs(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            done.add((rec["market_ticker"], rec["article_url"]))
        except (json.JSONDecodeError, KeyError):
            pass
    return done


def main(dry_run: bool = False) -> None:
    # ---- Load inputs -------------------------------------------------------
    if not INPUT_FILE.exists():
        logging.error("Input not found: %s", INPUT_FILE)
        sys.exit(1)

    pairs: list[dict] = [
        json.loads(line)
        for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logging.info("Loaded %d pairs from %s.", len(pairs), INPUT_FILE)

    done = _load_done_pairs(OUTPUT_FILE)
    remaining = [
        p for p in pairs
        if (p.get("market_ticker", ""), p.get("article_url", "")) not in done
    ]
    logging.info("Already labeled: %d.  Remaining: %d.", len(done), len(remaining))

    if not remaining:
        logging.info("Nothing to do.")
        return

    # ---- Load models -------------------------------------------------------
    logging.info("Loading sentence-transformers (all-MiniLM-L6-v2)…")
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vader    = SentimentIntensityAnalyzer()
    logging.info("Models ready.")

    # ---- Build text corpora ------------------------------------------------
    logging.info("Preparing text corpora…")
    doc_texts:   list[str] = []
    query_texts: list[str] = []
    keywords_list: list[list[str]] = []

    for p in remaining:
        body    = (p.get("article_body") or "")[:ARTICLE_CHARS]
        title   = p.get("market_title", "")
        rules   = p.get("rules_primary") or ""
        query   = title + ". " + rules
        kws     = _market_keywords(title, rules)

        doc_texts.append(body)
        query_texts.append(query)
        keywords_list.append(kws)

    # ---- Encode with MiniLM -----------------------------------------------
    logging.info("Encoding %d article bodies… (this takes ~10–15 min on CPU)", len(remaining))
    doc_embeddings = embedder.encode(
        doc_texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    logging.info("Encoding %d market queries…", len(remaining))
    query_embeddings = embedder.encode(
        query_texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    logging.info("Encoding complete.")

    # ---- Compute cosine similarities (element-wise, not full matrix) -------
    # util.cos_sim builds an N×N matrix (~17 GB for 66k pairs) — we only need
    # the diagonal.  torch.nn.functional.cosine_similarity computes pair[i]
    # vs pair[i] directly with no intermediate matrix.
    logging.info("Computing cosine similarities…")
    import torch
    import torch.nn.functional as F
    cos_scores = F.cosine_similarity(doc_embeddings, query_embeddings, dim=1).tolist()

    # ---- Label each pair ---------------------------------------------------
    kept_per_market: dict[str, int] = defaultdict(int)
    counters: dict[str, int] = defaultdict(int)
    records: list[dict] = []

    for i, pair in enumerate(remaining):
        hours          = float(pair.get("hours_before_resolution") or 0.0)
        body           = (pair.get("article_body") or "")
        category       = pair.get("market_category", "other")
        kalshi_cat     = pair.get("kalshi_category", "")
        ticker         = pair.get("market_ticker", "")
        outcome        = (pair.get("market_result") or "").lower()
        kws            = keywords_list[i]

        _embargo_applies = (
            category in _EMBARGO_CATEGORIES
            or kalshi_cat in _KALSHI_EMBARGO_CATEGORIES
        )

        # Hard gates
        if len(body) < 200:
            counters["gate_short_body"] += 1
            kept = False
            reason = "short_body"
        elif hours > MAX_LOOKBACK_H:
            counters["gate_too_early"] += 1
            kept = False
            reason = "too_early"
        elif hours < EMBARGO_H and _embargo_applies:
            counters["gate_embargo"] += 1
            kept = False
            reason = "embargo"
        else:
            semantic_sim   = float(cos_scores[i])
            kd             = _keyword_density(kws, body)
            temporal       = _temporal_score(hours)
            heuristic_score = 0.6 * semantic_sim + 0.2 * kd + 0.2 * temporal

            if heuristic_score <= SCORE_THRESHOLD:
                counters["gate_low_score"] += 1
                kept = False
                reason = "low_score"
            else:
                # Assign tentative label
                training_label = 1 if outcome == "yes" else 0

                # VADER contradiction gate
                best_sent = _best_vader_sentence(body, kws)
                compound  = vader.polarity_scores(best_sent)["compound"]
                if training_label == 1 and compound < -VADER_THRESHOLD:
                    counters["gate_vader_neg"] += 1
                    kept = False
                    reason = "vader_contradiction"
                elif training_label == 0 and compound > VADER_THRESHOLD:
                    counters["gate_vader_pos"] += 1
                    kept = False
                    reason = "vader_contradiction"
                elif kept_per_market[ticker] >= MARKET_CAP:
                    counters["gate_market_cap"] += 1
                    kept = False
                    reason = "market_cap"
                else:
                    kept = True
                    reason = "kept"
                    kept_per_market[ticker] += 1
                    counters[f"kept_{outcome}"] += 1

        # Build output record
        if kept:
            label_val = 1 if outcome == "yes" else 0
            sem_s     = float(cos_scores[i])
            kd_val    = _keyword_density(kws, body)
            t_val     = _temporal_score(hours)
            score_val = 0.6 * sem_s + 0.2 * kd_val + 0.2 * t_val
            record = {
                **pair,
                "claude_score":     round(score_val, 4),
                "claude_direction": "YES" if label_val == 1 else "NO",
                "claude_reasoning": (
                    f"heuristic: sem={sem_s:.2f} kd={kd_val:.2f}"
                    f" temporal={_temporal_score(hours):.2f}"
                ),
                "training_label": label_val,
                "sample_weight":  round(score_val, 4),
                "kept":           True,
            }
        else:
            record = {
                **pair,
                "claude_score":     0.0,
                "claude_direction": "NEUTRAL",
                "claude_reasoning": f"heuristic_gate: {reason}",
                "training_label":   None,
                "sample_weight":    0.0,
                "kept":             False,
            }

        records.append(record)
        counters["total"] += 1

    # ---- Summary -----------------------------------------------------------
    total_kept = counters["kept_yes"] + counters["kept_no"]
    pct_kept   = 100 * total_kept / max(counters["total"], 1)

    logging.info("=" * 60)
    logging.info("Labeling complete.")
    logging.info("  Total processed:         %d", counters["total"])
    logging.info("  Kept (training):         %d  (%.1f%%)", total_kept, pct_kept)
    logging.info("    Kept YES:              %d", counters["kept_yes"])
    logging.info("    Kept NO:               %d", counters["kept_no"])
    logging.info("  Discarded — low score:   %d", counters["gate_low_score"])
    logging.info("  Discarded — too early:   %d", counters["gate_too_early"])
    logging.info("  Discarded — embargo:     %d", counters["gate_embargo"])
    logging.info("  Discarded — short body:  %d", counters["gate_short_body"])
    logging.info("  Discarded — VADER neg:   %d", counters["gate_vader_neg"])
    logging.info("  Discarded — VADER pos:   %d", counters["gate_vader_pos"])
    logging.info("  Discarded — market cap:  %d", counters["gate_market_cap"])

    if dry_run:
        logging.info("Dry run — no output written.")
        return

    # ---- Write output ------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    logging.info("Output appended to %s  (%d new records).", OUTPUT_FILE, len(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heuristic labeler for market-article pairs.")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only; do not write output.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
