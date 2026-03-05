"""
DEPRECATED: Too slow for large documents without GPU. Generates all 1-3 gram
candidates, encodes each through a transformer, then runs O(n²) MMR — hangs or
takes 20+ minutes on 177-page (48K word) documents. Overlaps with faster
algorithms (RAKE, YAKE, TopicRank). Preserved here for reference.

KeyBERT Keyword Extraction Algorithm

KeyBERT uses document and candidate embeddings (via sentence-transformers)
to find keyphrases most similar to the document as a whole. It computes
cosine similarity between document embedding and candidate n-gram embeddings.

Key characteristics:
- Semantic: uses transformer embeddings, not just statistics
- MMR diversity: Maximal Marginal Relevance reduces redundant keywords
- Reuses bundled model: all-MiniLM-L6-v2 already ships with the app

The confidence score IS the cosine similarity (already 0-1 normalized).

Reference:
Grootendorst, M. (2020). "KeyBERT: Minimal keyword extraction with BERT"
https://github.com/MaartenGr/KeyBERT
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


@register_algorithm("KeyBERT")
class KeyBERTAlgorithm(BaseExtractionAlgorithm):
    """
    KeyBERT keyword extraction algorithm.

    Uses cosine similarity between document and candidate embeddings
    to find the most representative keyphrases. Reuses the bundled
    all-MiniLM-L6-v2 model (also used by the semantic chunker).
    """

    name = "KeyBERT"
    weight = VOCAB_ALGORITHM_WEIGHTS.get("KeyBERT", 0.65)

    def __init__(
        self,
        top_n: int = 150,
        ngram_range: tuple[int, int] = (1, 3),
        diversity: float = 0.5,
    ):
        """
        Initialize KeyBERT algorithm.

        Args:
            top_n: Number of top keywords to extract (default: 150)
            ngram_range: Min and max n-gram size (default: (1, 3))
            diversity: MMR diversity parameter 0-1 (default: 0.5)
        """
        self.top_n = top_n
        self.ngram_range = ngram_range
        self.diversity = diversity
        self._model = None

    def _load_model(self):
        """Lazy-load KeyBERT with bundled SentenceTransformer model."""
        from keybert import KeyBERT

        from src.config import SEMANTIC_CHUNKER_MODEL_LOCAL_PATH

        if SEMANTIC_CHUNKER_MODEL_LOCAL_PATH.exists():
            model_path = str(SEMANTIC_CHUNKER_MODEL_LOCAL_PATH)
            logger.debug("KeyBERT using bundled model: %s", model_path)
        else:
            model_path = "all-MiniLM-L6-v2"
            logger.debug("KeyBERT using HuggingFace model: %s", model_path)

        self._model = KeyBERT(model=model_path)

    def extract(self, text: str, **kwargs) -> AlgorithmResult:
        """
        Extract keyphrases from text using KeyBERT.

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

        # Lazy-load model
        if self._model is None:
            try:
                self._load_model()
            except Exception as e:
                logger.warning("KeyBERT unavailable: %s", e)
                return AlgorithmResult(
                    candidates=[],
                    processing_time_ms=0.0,
                    metadata={"skipped": True, "reason": str(e)},
                )

        # Truncate very long texts for performance
        truncated = len(text) > _MAX_TEXT_BYTES
        process_text = text[:_MAX_TEXT_BYTES] if truncated else text

        # Extract keywords with MMR diversity
        raw_keywords = self._model.extract_keywords(
            process_text,
            keyphrase_ngram_range=self.ngram_range,
            top_n=self.top_n,
            use_mmr=True,
            diversity=self.diversity,
        )

        candidates = []
        seen_phrases: set[str] = set()

        for keyphrase, score in raw_keywords:
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

            # KeyBERT score is cosine similarity, already 0-1
            confidence = max(0.0, min(float(score), 1.0))

            candidates.append(
                CandidateTerm(
                    term=cleaned,
                    source_algorithm=self.name,
                    confidence=confidence,
                    suggested_type="Technical",
                    frequency=1,  # KeyBERT doesn't provide frequency
                    metadata={
                        "keybert_score": round(float(score), 6),
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
            "top_n": self.top_n,
            "ngram_range": self.ngram_range,
            "diversity": self.diversity,
        }
