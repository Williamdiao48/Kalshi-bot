"""DistilBERT inference engine for text-opportunity direction scoring.

Loads the fine-tuned Kalshi classifier once at startup and exposes a single
predict() method used by maybe_trade_text() to determine YES/NO direction and
confidence from (market_title, article_text) pairs.

The model was trained on (market, article) pairs where label=1 means the article
predicted YES and the market resolved YES.  Output P(YES) > 0.5 → buy YES;
P(YES) < 0.5 → buy NO.

Environment variables:
    CLASSIFIER_MODEL_PATH  Path to model directory (default: models/kalshi_classifier/)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid heavy imports at module level

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "kalshi_classifier"
CLASSIFIER_MODEL_PATH: Path = Path(
    os.environ.get("CLASSIFIER_MODEL_PATH", str(_DEFAULT_MODEL_PATH))
)

# Input format must match training: [MARKET] title [SEP] article (truncated)
_INPUT_TEMPLATE = "[MARKET] {market} [SEP] {article}"
_ARTICLE_CHARS   = 600   # matches LLM_ARTICLE_CHARS used during labeling


class _Classifier:
    """Singleton wrapper around the fine-tuned DistilBERT model."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._pipeline = None
        self._loaded    = False

    def _load(self) -> bool:
        """Lazy-load model on first predict() call.  Returns True on success."""
        if self._loaded:
            return self._pipeline is not None

        self._loaded = True

        if not self._model_path.exists():
            logging.warning(
                "classifier: model not found at %s — text trading disabled.",
                self._model_path,
            )
            return False

        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
            self._pipeline = hf_pipeline(
                "text-classification",
                model=str(self._model_path),
                tokenizer=str(self._model_path),
                device=-1,          # CPU only
                truncation=True,
                max_length=512,
                top_k=None,         # return scores for all labels
            )
            logging.info("classifier: loaded from %s", self._model_path)
            return True
        except Exception as exc:
            logging.warning("classifier: failed to load: %s — text trading disabled.", exc)
            return False

    def predict(self, market_title: str, article_text: str) -> tuple[str, float]:
        """Return (direction, p_yes) for a (market, article) pair.

        direction: 'YES' | 'NO' | 'NEUTRAL'
        p_yes:     probability in [0.0, 1.0] that the article implies YES

        Returns ('NEUTRAL', 0.5) if the model is unavailable or inference fails.
        """
        if not self._load() or self._pipeline is None:
            return "NEUTRAL", 0.5

        text = _INPUT_TEMPLATE.format(
            market=market_title.strip(),
            article=article_text.strip()[:_ARTICLE_CHARS],
        )

        try:
            results = self._pipeline(text)[0]  # list of {label, score} dicts
            # Labels are LABEL_0 (NO) and LABEL_1 (YES)
            score_map = {r["label"]: r["score"] for r in results}
            p_yes = float(score_map.get("LABEL_1", score_map.get("1", 0.5)))
        except Exception as exc:
            logging.warning("classifier: inference error: %s", exc)
            return "NEUTRAL", 0.5

        if p_yes > 0.5:
            return "YES", p_yes
        elif p_yes < 0.5:
            return "NO", p_yes
        else:
            return "NEUTRAL", 0.5


_instance: _Classifier | None = None


def get_classifier() -> _Classifier:
    """Return the global classifier singleton, creating it on first call."""
    global _instance
    if _instance is None:
        _instance = _Classifier(CLASSIFIER_MODEL_PATH)
    return _instance
