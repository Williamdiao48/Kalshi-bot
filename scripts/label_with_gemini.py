"""Label (market, article) pairs using the Gemini API (free tier).

Uses knowledge distillation to generate training labels for the semantic
signal model.  Gemini scores each pair on a 0.0–1.0 scale based on how much
incremental predictive information the article contains, given the actual
market outcome.

Free tier limits (Gemini 2.0 Flash Lite as of 2026):
    30 requests / minute
    1,500 requests / day
    1M tokens / day
  At ~1000 tokens/pair → 1,500 pairs/day → 5,000 pairs in ~4 days.
  Each day's run takes ~100 minutes (1400 ÷ 14 RPM).
  Resume support means you just re-run each day — it skips done pairs.

Pair selection logic (only KEEP agreement pairs):
  - Gemini direction YES  + actual outcome yes  → label=1  KEEP
  - Gemini direction NO   + actual outcome no   → label=0  KEEP
  - Gemini direction YES  + actual outcome no   → DISCARD  (misleading article)
  - Gemini direction NO   + actual outcome yes  → DISCARD  (misleading article)
  - Gemini direction NEUTRAL                    → DISCARD  (noise)

Score filter (applied after direction check):
  - score > 0.5   → keep (clear signal)
  - score < 0.2   → keep (clear noise — teaches model what non-signal looks like)
  - 0.2–0.5       → discard (ambiguous — degrades training)

Usage:
    venv/bin/python scripts/label_with_claude.py

Input:
    data/market_article_pairs.jsonl

Output:
    data/labeled_pairs.jsonl — ALL labeled pairs with a `kept` flag.
    Only `kept=true` records are used by build_training_set.py, but saving
    everything avoids re-calling the API if thresholds change later.

Environment variables:
    GEMINI_API_KEY          Required. Get from aistudio.google.com/app/apikey
    LABEL_MODEL             Gemini model (default: gemini-2.0-flash-lite).
    LABEL_RPM               Max requests per minute (default: 14 — just under
                            the free-tier limit of 15 to avoid bursting over).
    LABEL_MAX_PAIRS         Stop after N pairs (default: 1400, safely under the
                            1500/day free limit). Re-run tomorrow to continue.
    LABEL_ARTICLE_CHARS     Characters of body sent to Gemini (default: 2000).
"""

import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("google.genai.client").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE  = Path(__file__).parent.parent / "data" / "market_article_pairs.jsonl"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "labeled_pairs.jsonl"

LABEL_MODEL: str    = os.environ.get("LABEL_MODEL", "gemini-2.0-flash-lite")
LABEL_RPM: int      = int(os.environ.get("LABEL_RPM", "14"))
LABEL_MAX_PAIRS: int = int(os.environ.get("LABEL_MAX_PAIRS", "1400"))
ARTICLE_CHARS: int  = int(os.environ.get("LABEL_ARTICLE_CHARS", "2000"))

MAX_RETRIES  = 4
SCORE_HIGH   = 0.5
SCORE_LOW    = 0.2

_JSON_RE = re.compile(r'\{[^{}]*"score"[^{}]*\}', re.DOTALL)

# ---------------------------------------------------------------------------
# Rate limiter — enforces LABEL_RPM globally across all concurrent tasks
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Token-bucket rate limiter for async coroutines.

    Serialises requests so no more than `rpm` are sent per 60-second window.
    Each acquire() call waits until the next available slot.
    """
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / rpm   # minimum seconds between requests
        self._lock     = asyncio.Lock()
        self._last_ts  = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now  = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = asyncio.get_event_loop().time()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_LABEL_PROMPT = """\
You are generating training data for a prediction market signal model.

Market title: "{market_title}"
Resolution criteria: "{market_rules}"
Actual outcome (historical ground truth): {outcome}

Article headline: "{headline}"
Hours before market resolved: {hours_before:.0f}h
Article text:
{body}

---

A trader reads this article {hours_before:.0f} hours before the market resolves.

Task: Score how much *incremental* predictive signal this article provides about
the market outcome — meaning what specific information it adds that a neutral,
reasonably informed observer would NOT already know from general background knowledge.

An article that simply confirms widespread prior expectations scores near 0.
An article that contains a specific, non-obvious new development that directly
implies the outcome scores near 1.

Score 0.0–1.0:
  0.0   = completely irrelevant — no bearing on this market
  0.1   = vaguely topical but zero new information
  0.25  = weakly relevant — existing knowledge, mild confirmation at most
  0.5   = moderately relevant — some new information, mild update warranted
  0.75  = clearly predictive — specific new development justifies meaningful update
  1.0   = strongly predictive — article directly implies the actual outcome

Also output direction: which outcome does this article push toward?
If the article has no directional signal, output "NEUTRAL".

Respond with JSON only — no text before or after:
{{"score": <float>, "direction": "YES" | "NO" | "NEUTRAL", "reasoning": "<one sentence>"}}"""


def _build_prompt(pair: dict[str, Any]) -> str:
    outcome = (pair.get("market_result") or "").upper()
    body    = (pair.get("article_body") or "")[:ARTICLE_CHARS]
    rules   = pair.get("rules_primary") or pair.get("market_title", "")
    hours   = pair.get("hours_before_resolution") or 0.0
    return _LABEL_PROMPT.format(
        market_title=pair.get("market_title", ""),
        market_rules=rules,
        outcome=outcome,
        headline=pair.get("article_headline", ""),
        hours_before=hours,
        body=body,
    )


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------

async def _call_gemini(
    client: Any,
    prompt: str,
    rate_limiter: "_RateLimiter",
) -> dict[str, Any] | None:
    """Call Gemini and parse the JSON response.

    Waits for a rate-limiter slot before each call.
    Retries on quota errors and server errors with exponential backoff.
    Returns None if the response cannot be parsed after retries.
    """
    try:
        from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
    except ImportError:
        ResourceExhausted = ServiceUnavailable = Exception

    from google.genai import types

    config = types.GenerateContentConfig(
        max_output_tokens=1024,
        temperature=0.1,        # low temperature → consistent structured output
        thinking_config=types.ThinkingConfig(thinking_budget=0),  # disable thinking — saves tokens for JSON output
    )

    text = ""
    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()
        try:
            response = await client.aio.models.generate_content(
                model=LABEL_MODEL,
                contents=prompt,
                config=config,
            )
            text = response.text.strip()
            break
        except ResourceExhausted:
            wait = 30 * (attempt + 1)
            logging.warning("Quota exceeded — waiting %ds (attempt %d).", wait, attempt + 1)
            await asyncio.sleep(wait)
        except ServiceUnavailable:
            wait = 5 * (2 ** attempt)
            logging.warning("Service unavailable — waiting %ds (attempt %d).", wait, attempt + 1)
            await asyncio.sleep(wait)
        except Exception as exc:
            # google-genai SDK raises its own exception types; fall back to
            # string matching so 429/503 errors still get retried.
            exc_str = str(exc)
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                wait = 30 * (attempt + 1)
                logging.warning("Quota exceeded — waiting %ds (attempt %d).", wait, attempt + 1)
                await asyncio.sleep(wait)
            elif "503" in exc_str or "UNAVAILABLE" in exc_str:
                wait = 5 * (2 ** attempt)
                logging.warning("Service unavailable — waiting %ds (attempt %d).", wait, attempt + 1)
                await asyncio.sleep(wait)
            else:
                logging.error("Gemini API error (non-retryable): %s", exc)
                return None
    else:
        logging.error("Max retries exhausted.")
        return None

    # Parse JSON — Gemini occasionally wraps output in markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = _JSON_RE.search(text)
    raw = m.group(0) if m else text
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            start = text.index("{")
            end   = text.rindex("}") + 1
            parsed = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            logging.warning("Could not parse Gemini response: %s", text[:120])
            return None

    return {
        "score":     float(parsed.get("score", 0.0)),
        "direction": str(parsed.get("direction", "NEUTRAL")).upper(),
        "reasoning": str(parsed.get("reasoning", "")),
    }


# ---------------------------------------------------------------------------
# Label decision  (identical logic to the original Claude version)
# ---------------------------------------------------------------------------

def _make_label(pair: dict[str, Any], gemini: dict[str, Any]) -> dict[str, Any]:
    score     = gemini["score"]
    direction = gemini["direction"]
    outcome   = (pair.get("market_result") or "").lower()

    direction_matches = (
        (direction == "YES" and outcome == "yes") or
        (direction == "NO"  and outcome == "no")
    )
    score_ok = (score > SCORE_HIGH) or (score < SCORE_LOW)
    kept     = direction_matches and direction != "NEUTRAL" and score_ok

    return {
        **pair,
        "claude_score":     score,          # field kept as claude_* for pipeline compat
        "claude_direction": direction,
        "claude_reasoning": gemini["reasoning"],
        "training_label":   (1 if outcome == "yes" else 0) if kept else None,
        "sample_weight":    score,
        "kept":             kept,
    }


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_done_pairs(output_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not output_path.exists():
        return done
    for line in output_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            done.add((rec["market_ticker"], rec["article_url"]))
        except (json.JSONDecodeError, KeyError):
            pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    try:
        from google import genai as google_genai
    except ImportError:
        logging.error("google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logging.error(
            "GEMINI_API_KEY not set.\n"
            "Get a free key at: aistudio.google.com/app/apikey\n"
            "Then add to .env: GEMINI_API_KEY=\"your-key-here\""
        )
        sys.exit(1)

    if not INPUT_FILE.exists():
        logging.error(
            "Input file not found: %s\nRun fetch_article_bodies.py first.",
            INPUT_FILE,
        )
        sys.exit(1)

    pairs: list[dict[str, Any]] = [
        json.loads(line)
        for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    logging.info("Loaded %d pairs from %s.", len(pairs), INPUT_FILE)

    # Resume: skip already-labeled pairs
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_pairs(OUTPUT_FILE)
    remaining = [
        p for p in pairs
        if (p.get("market_ticker", ""), p.get("article_url", "")) not in done
    ]
    logging.info("Already labeled: %d.  Remaining: %d.", len(done), len(remaining))

    # Enforce daily cap
    if len(remaining) > LABEL_MAX_PAIRS:
        logging.info(
            "Capping at %d pairs (LABEL_MAX_PAIRS) to stay within free-tier daily limit. "
            "Re-run tomorrow to continue.",
            LABEL_MAX_PAIRS,
        )
        remaining = remaining[:LABEL_MAX_PAIRS]

    if not remaining:
        logging.info("Nothing to do.")
        return

    est_minutes = len(remaining) / LABEL_RPM
    logging.info(
        "Labeling %d pairs at %d RPM → estimated %.0f minutes.",
        len(remaining), LABEL_RPM, est_minutes,
    )

    client       = google_genai.Client(api_key=api_key)
    rate_limiter = _RateLimiter(LABEL_RPM)
    counters: dict[str, int] = defaultdict(int)

    async def _process(pair: dict[str, Any], out_fh: Any) -> None:
        prompt = _build_prompt(pair)
        result = await _call_gemini(client, prompt, rate_limiter)
        if result is None:
            counters["api_failed"] += 1
            return

        record = _make_label(pair, result)
        out_fh.write(json.dumps(record) + "\n")
        out_fh.flush()

        counters["labeled"] += 1
        if record["kept"]:
            counters["kept"] += 1
            counters[f"kept_{record['claude_direction'].lower()}"] += 1
        else:
            reason = (
                "neutral"          if record["claude_direction"] == "NEUTRAL"
                else "ambiguous"   if 0.2 <= record["claude_score"] <= 0.5
                else "mismatch"
            )
            counters[f"discard_{reason}"] += 1

    with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
        # Run sequentially — rate limiter already serialises; no benefit to
        # concurrent tasks since we're capped at 14 RPM anyway.
        for i, pair in enumerate(remaining, 1):
            await _process(pair, fh)
            if i % 50 == 0 or i == len(remaining):
                pct_kept = 100 * counters["kept"] / max(counters["labeled"], 1)
                logging.info(
                    "[%d/%d] labeled=%d  kept=%d (%.0f%%)  api_fail=%d",
                    i, len(remaining),
                    counters["labeled"], counters["kept"], pct_kept,
                    counters["api_failed"],
                )

    logging.info("=" * 60)
    logging.info("Session complete.")
    logging.info("  Pairs labeled:       %d", counters["labeled"])
    logging.info("  Kept (training):     %d (%.0f%%)",
                 counters["kept"], 100 * counters["kept"] / max(counters["labeled"], 1))
    logging.info("  Discard - mismatch:  %d", counters["discard_mismatch"])
    logging.info("  Discard - ambiguous: %d", counters["discard_ambiguous"])
    logging.info("  Discard - neutral:   %d", counters["discard_neutral"])
    logging.info("  API failures:        %d", counters["api_failed"])
    logging.info("  Kept YES signal:     %d", counters["kept_yes"])
    logging.info("  Kept NO signal:      %d", counters["kept_no"])
    logging.info("Output: %s", OUTPUT_FILE)

    remaining_total = len(pairs) - len(done) - counters["labeled"]
    if remaining_total > 0:
        logging.info(
            "  %d pairs still unlabeled — re-run tomorrow to continue "
            "(free tier daily limit).",
            remaining_total,
        )


if __name__ == "__main__":
    asyncio.run(main())
