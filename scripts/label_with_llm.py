"""LLM-based directional labeling for (market, article) pairs.

Uses a locally-run LLM (via HuggingFace transformers or Ollama) to score each pair
with a direction (YES/NO/NEUTRAL) and confidence score (0.0–1.0), mirroring the
original Gemini labeler but at zero cost.

Designed to run on Google Colab (T4 GPU + HuggingFace backend) or locally (Ollama).

Backend selection (LLM_BACKEND env var):
  hf      HuggingFace transformers pipeline with 4-bit quantization (default, Colab)
  ollama  Ollama HTTP API at localhost:11434 (local Mac)

Usage (Colab):
    !pip install transformers bitsandbytes accelerate -q
    # Set HF_TOKEN before running (userdata can't be called from a subprocess):
    import os; from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    !LLM_BACKEND=hf LLM_MODEL=Qwen/Qwen2.5-7B-Instruct python label_with_llm.py

Usage (local with Ollama):
    ollama serve &
    ollama pull llama3.2:3b
    LLM_BACKEND=ollama LLM_MODEL=llama3.2:3b venv/bin/python scripts/label_with_llm.py

Output format is schema-compatible with labeled_pairs.jsonl so build_training_set.py
requires zero changes.

Environment variables:
    LLM_BACKEND         "hf" or "ollama" (default: hf)
    LLM_MODEL           HF model ID or Ollama model name
                        (default: Qwen/Qwen2.5-7B-Instruct)
    LLM_MAX_PAIRS       Max pairs to process, sorted by heuristic score (default: 4000)
    LLM_ARTICLE_CHARS   Article body chars in prompt (default: 600)
    LLM_SCORE_THRESHOLD Min LLM score to keep a pair (default: 0.4)
    LLM_TIMEOUT         Per-request timeout seconds, Ollama only (default: 60)
    INPUT_FILE          Path to labeled_pairs.jsonl (default: data/labeled_pairs.jsonl)
    OUTPUT_FILE         Path to output file (default: data/labeled_pairs_llm.jsonl)
"""

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

# Pull HF_TOKEN from Google Colab secrets if not already set in the environment.
if not os.environ.get("HF_TOKEN"):
    try:
        from google.colab import userdata  # type: ignore
        _hf_token = userdata.get("HF_TOKEN")
        if _hf_token:
            os.environ["HF_TOKEN"] = _hf_token
            logging.info("HF_TOKEN loaded from Colab secrets.")
        else:
            logging.warning("Colab userdata.get('HF_TOKEN') returned empty.")
    except Exception as _e:
        logging.warning(
            "Could not load HF_TOKEN from Colab secrets: %s\n"
            "  Fix: set it in the notebook cell before running the script:\n"
            "    import os; from google.colab import userdata\n"
            "    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')",
            _e,
        )

# Explicitly log in so huggingface_hub uses the token even if it cached auth
# state before the env var was set.
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login as _hf_login  # type: ignore
        _hf_login(token=_hf_token, add_to_git_credential=False)
        logging.info("huggingface_hub login successful.")
    except Exception as _e:
        logging.warning("huggingface_hub login failed: %s", _e)
else:
    logging.warning("HF_TOKEN not set — downloads will be rate-limited.")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_root = Path(__file__).parent.parent

BACKEND         = os.environ.get("LLM_BACKEND",   "hf")
MODEL           = os.environ.get("LLM_MODEL",     "Qwen/Qwen2.5-7B-Instruct")
MAX_PAIRS       = int(os.environ.get("LLM_MAX_PAIRS",     "4000"))
ARTICLE_CHARS   = int(os.environ.get("LLM_ARTICLE_CHARS", "600"))
SCORE_THRESHOLD = float(os.environ.get("LLM_SCORE_THRESHOLD", "0.4"))
OLLAMA_TIMEOUT  = int(os.environ.get("LLM_TIMEOUT",       "60"))

INPUT_FILE  = Path(os.environ.get("INPUT_FILE",  _root / "data" / "labeled_pairs.jsonl"))
OUTPUT_FILE = Path(os.environ.get("OUTPUT_FILE", _root / "data" / "labeled_pairs_llm.jsonl"))

_PROMPT_TEMPLATE = """\
You are generating training data for a prediction market signal model.

Market: {market_title}
Rules: {rules}
Actual outcome (historical ground truth): {outcome}

Article ({hours_before:.0f}h before resolution):
{article_body}

A trader reads this article before the market resolves.

Score how much *predictive signal* this article provides (0.0–1.0):
  0.0  = completely irrelevant
  0.25 = weakly relevant, mild confirmation at most
  0.5  = some new information, mild update warranted
  0.75 = specific new development justifies meaningful update
  1.0  = article directly implies the actual outcome

Also output direction: which outcome does this article push toward?
Output NEUTRAL if the article has no directional signal.

Respond with JSON only — no text before or after:
{{"score": <float>, "direction": "YES" | "NO" | "NEUTRAL", "reasoning": "<one sentence>"}}\
"""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_prompt(pair: dict) -> str:
    return _PROMPT_TEMPLATE.format(
        market_title=pair.get("market_title", ""),
        rules=(pair.get("rules_primary") or "")[:200],
        outcome=(pair.get("market_result") or "").upper(),
        hours_before=float(pair.get("hours_before_resolution") or 0),
        article_body=(pair.get("article_body") or "")[:ARTICLE_CHARS],
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> dict:
    """Extract score/direction/reasoning from LLM response.
    Falls back to keyword scan if JSON parse fails.
    """
    # Try to find JSON object in response
    text = text.strip()
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            score     = float(obj.get("score", 0.5))
            direction = str(obj.get("direction", "NEUTRAL")).upper()
            reasoning = str(obj.get("reasoning", ""))
            if direction not in ("YES", "NO", "NEUTRAL"):
                direction = "NEUTRAL"
            return {"score": round(score, 4), "direction": direction, "reasoning": reasoning}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: keyword scan
    upper = text.upper()
    if "YES" in upper and "NO" not in upper:
        direction = "YES"
    elif "NO" in upper and "YES" not in upper:
        direction = "NO"
    else:
        direction = "NEUTRAL"
    return {"score": 0.5, "direction": direction, "reasoning": "fallback_keyword_parse"}


# ---------------------------------------------------------------------------
# Label decision (matches label_with_gemini.py logic)
# ---------------------------------------------------------------------------

def _make_label(pair: dict, llm: dict) -> dict:
    score     = llm["score"]
    direction = llm["direction"]
    outcome   = (pair.get("market_result") or "").lower()

    direction_matches = (
        (direction == "YES" and outcome == "yes") or
        (direction == "NO"  and outcome == "no")
    )
    kept = direction_matches and direction != "NEUTRAL" and score >= SCORE_THRESHOLD

    return {
        **pair,
        "claude_score":     score,
        "claude_direction": direction,
        "claude_reasoning": llm["reasoning"],
        "training_label":   (1 if outcome == "yes" else 0) if kept else None,
        "sample_weight":    score if kept else 0.0,
        "kept":             kept,
    }


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

def _load_hf_pipeline():
    logging.info("Loading HuggingFace model: %s (4-bit quantized)…", MODEL)
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=quantization_config,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    logging.info("HF model ready.")
    return pipe


def _call_hf(pipe, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You output JSON only. No explanations."},
        {"role": "user",   "content": prompt},
    ]
    result = pipe(messages, max_new_tokens=80, do_sample=False)
    # Extract generated text — last message content
    generated = result[0]["generated_text"]
    if isinstance(generated, list):
        # Chat format: last dict is assistant reply
        for msg in reversed(generated):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
    return str(generated)


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> str:
    import urllib.request

    payload = json.dumps({
        "model":  MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 80},
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        data = json.loads(resp.read())
    return data.get("response", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_FILE.exists():
        logging.error("Input not found: %s", INPUT_FILE)
        sys.exit(1)

    # Load kept pairs sorted by heuristic score
    all_pairs = [
        json.loads(line)
        for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept = [p for p in all_pairs if p.get("kept")]
    kept.sort(key=lambda p: float(p.get("claude_score") or 0), reverse=True)
    candidates = kept[:MAX_PAIRS]
    logging.info(
        "Total kept pairs: %d. Processing top %d by heuristic score.",
        len(kept), len(candidates),
    )

    done = _load_done_pairs(OUTPUT_FILE)
    remaining = [
        p for p in candidates
        if (p.get("market_ticker", ""), p.get("article_url", "")) not in done
    ]
    logging.info("Already labeled: %d. Remaining: %d.", len(done), len(remaining))

    if not remaining:
        logging.info("Nothing to do.")
        return

    # Load backend
    hf_pipe = None
    if BACKEND == "hf":
        hf_pipe = _load_hf_pipeline()
    else:
        logging.info("Using Ollama backend (model: %s).", MODEL)

    # Process
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    counters = {"kept": 0, "neutral": 0, "contradiction": 0, "low_score": 0, "error": 0}

    with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
        for i, pair in enumerate(remaining, 1):
            logging.info(
                "[%d/%d] %s — %s",
                i, len(remaining),
                pair.get("market_ticker", "?"),
                (pair.get("article_headline") or pair.get("article_url", ""))[:80],
            )
            prompt = _build_prompt(pair)
            try:
                if BACKEND == "hf":
                    raw = _call_hf(hf_pipe, prompt)
                else:
                    raw = _call_ollama(prompt)
                llm = _parse_response(raw)
            except Exception as exc:
                logging.warning("Pair %d error: %s", i, exc)
                llm = {"score": 0.0, "direction": "NEUTRAL", "reasoning": f"error: {exc}"}
                counters["error"] += 1

            record = _make_label(pair, llm)

            if record["kept"]:
                counters["kept"] += 1
            elif llm["direction"] == "NEUTRAL":
                counters["neutral"] += 1
            elif llm["score"] < SCORE_THRESHOLD:
                counters["low_score"] += 1
            else:
                counters["contradiction"] += 1

            fh.write(json.dumps(record) + "\n")
            fh.flush()

            if i % 50 == 0 or i == len(remaining):
                pct = 100 * i / len(remaining)
                logging.info(
                    "[%d/%d %.0f%%] kept=%d neutral=%d contradiction=%d low_score=%d error=%d",
                    i, len(remaining), pct,
                    counters["kept"], counters["neutral"],
                    counters["contradiction"], counters["low_score"], counters["error"],
                )

    total = sum(counters.values())
    logging.info("=" * 60)
    logging.info("Done. Processed: %d", total)
    logging.info("  Kept:             %d  (%.1f%%)", counters["kept"], 100 * counters["kept"] / max(total, 1))
    logging.info("  Neutral:          %d", counters["neutral"])
    logging.info("  Contradiction:    %d", counters["contradiction"])
    logging.info("  Low score (<%.1f): %d", SCORE_THRESHOLD, counters["low_score"])
    logging.info("  Errors:           %d", counters["error"])
    logging.info("Output: %s", OUTPUT_FILE)
    logging.info(
        "Next: INPUT_FILE=%s venv/bin/python scripts/build_training_set.py",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
