"""
YAKE (Yet Another Keyword Extractor) Algorithm

YAKE is a pure statistical keyword extraction method that requires no external
corpus, no training, and no model. It uses text statistical features (casing,
word frequency, word relatedness, word position) to compute a score for each
candidate keyphrase.

Key characteristics:
- Unsupervised: no training data or corpus needed
- Language-independent: works on any language
- Domain-independent: no domain-specific resources required
- Fast: pure text statistics, no model loading

YAKE scores are INVERTED: lower score = more important keyword.
We convert to confidence via: confidence = 1.0 / (1.0 + yake_score)
This maps 0.0 → 1.0 (best) and large scores → near 0.0 (worst).

Reference:
Campos, R. et al. (2020). "YAKE! Keyword extraction from single documents
using multiple local features"
https://doi.org/10.1016/j.ins.2019.09.013
"""

import logging
import time
from typing import Any

from src.config import VOCAB_ALGORITHM_WEIGHTS
from src.core.vocabulary.algorithms import register_algorithm
from src.core.vocabulary.algorithms.base import (
    AlgorithmResult,
    BaseExtractionAlgorithm,
    CandidateTerm,
)

logger = logging.getLogger(__name__)

# Maximum input text size (1 MB) to prevent memory issues
_MAX_TEXT_BYTES = 1_000_000


@register_algorithm("YAKE")
class YAKEAlgorithm(BaseExtractionAlgorithm):
    """
    YAKE keyword extraction algorithm.

    Extracts keyphrases using pure text statistics — no model, no corpus.
    Complements NER (entities) and RAKE (co-occurrence) with a third
    statistical signal based on casing, position, and frequency features.
    """

    name = "YAKE"
    weight = VOCAB_ALGORITHM_WEIGHTS.get("YAKE", 0.55)

    def __init__(
        self,
        max_ngram_size: int = 3,
        dedup_threshold: float = 0.9,
        window_size: int = 1,
        max_candidates: int = 150,
    ):
        """
        Initialize YAKE algorithm.

        Args:
            max_ngram_size: Maximum n-gram size for keyphrases (default: 3)
            dedup_threshold: Deduplication threshold (default: 0.9)
            window_size: Co-occurrence window size (default: 1)
            max_candidates: Maximum candidates to return (default: 150)
        """
        self.max_ngram_size = max_ngram_size
        self.dedup_threshold = dedup_threshold
        self.window_size = window_size
        self.max_candidates = max_candidates
        self._extractor = None

    @property
    def extractor(self):
        """Lazy-load YAKE extractor instance."""
        if self._extractor is None:
            import yake

            self._extractor = yake.KeywordExtractor(
                lan="en",
                n=self.max_ngram_size,
                dedupLim=self.dedup_threshold,
                windowsSize=self.window_size,
                top=self.max_candidates,
            )
        return self._extractor

    def extract(self, text: str, **kwargs) -> AlgorithmResult:
        """
        Extract keyphrases from text using YAKE.

        Args:
            text: Document text to analyze
            **kwargs: Not used by this algorithm

        Returns:
            AlgorithmResult with candidate keyphrases
        """
        start_time = time.time()

        if not text or not text.strip():
            return AlgorithmResult(
                candidates=[],
                processing_time_ms=0.0,
                metadata={"skipped": True, "reason": "empty text"},
            )

        # Truncate very long texts for performance
        truncated = len(text) > _MAX_TEXT_BYTES
        process_text = text[:_MAX_TEXT_BYTES] if truncated else text

        # Extract keywords (returns list of (keyphrase, score) tuples)
        raw_keywords = self.extractor.extract_keywords(process_text)

        candidates = []
        seen_phrases: set[str] = set()

        for keyphrase, yake_score in raw_keywords:
            if len(candidates) >= self.max_candidates:
                break

            cleaned = keyphrase.strip()

            # Skip single characters
            if len(cleaned) < 2:
                continue

            # Skip pure numbers/punctuation
            if not any(c.isalpha() for c in cleaned):
                continue

            # Dedup (case-insensitive)
            lower = cleaned.lower()
            if lower in seen_phrases:
                continue
            seen_phrases.add(lower)

            # Invert YAKE score: lower YAKE = more important
            # confidence = 1 / (1 + score) maps 0→1.0, large→~0.0
            confidence = 1.0 / (1.0 + yake_score)

            candidates.append(
                CandidateTerm(
                    term=cleaned,
                    source_algorithm=self.name,
                    confidence=confidence,
                    suggested_type="Technical",
                    frequency=1,  # YAKE doesn't provide frequency
                    metadata={
                        "yake_score": round(yake_score, 6),
                        "word_count": len(cleaned.split()),
                    },
                )
            )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            "Extracted %d keyphrases from %d raw in %.1fms (truncated: %s)",
            len(candidates),
            len(raw_keywords),
            processing_time_ms,
            truncated,
        )

        return AlgorithmResult(
            candidates=candidates,
            processing_time_ms=processing_time_ms,
            metadata={
                "raw_keywords_found": len(raw_keywords),
                "filtered_candidates": len(candidates),
                "text_truncated": truncated,
            },
        )

    def get_config(self) -> dict[str, Any]:
        """Return algorithm configuration."""
        return {
            **super().get_config(),
            "max_ngram_size": self.max_ngram_size,
            "dedup_threshold": self.dedup_threshold,
            "window_size": self.window_size,
            "max_candidates": self.max_candidates,
        }
