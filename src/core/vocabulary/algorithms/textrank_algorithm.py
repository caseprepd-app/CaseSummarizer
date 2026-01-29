"""
TextRank Keyword Extraction Algorithm

Uses pytextrank (a spaCy pipeline component) to extract keyphrases via
graph-based ranking. TextRank builds a word co-occurrence graph and uses
PageRank to identify the most important phrases.

TextRank complements NER and RAKE:
- NER finds named entities (people, organizations, locations)
- RAKE uses co-occurrence statistics within stopword-delimited phrases
- TextRank uses graph centrality across the full document

This algorithm loads a SEPARATE en_core_web_lg instance to avoid mutating
the shared NER pipeline. Reuses the same model already bundled for NER
so the installer doesn't need to ship a second spaCy model.

Reference:
Mihalcea & Tarau (2004), "TextRank: Bringing Order into Text"
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


@register_algorithm("TextRank")
class TextRankAlgorithm(BaseExtractionAlgorithm):
    """
    TextRank-based keyword extraction using pytextrank + spaCy.

    Extracts keyphrases by building a word graph and applying PageRank.
    Effective for finding domain-specific multi-word phrases and
    important single terms that are central to the document's content.

    Reuses en_core_web_lg (already bundled for NER) in a separate
    instance to avoid mutating the shared NER pipeline.
    """

    name = "TextRank"
    weight = VOCAB_ALGORITHM_WEIGHTS.get("TextRank", 0.6)

    def __init__(self, max_candidates: int = 150):
        """
        Initialize TextRank algorithm.

        Args:
            max_candidates: Maximum number of phrases to return
        """
        self.max_candidates = max_candidates
        self._nlp = None

    def _load_nlp(self):
        """
        Load spaCy model with pytextrank pipeline component.

        Reuses en_core_web_lg (already bundled for NER) so the installer
        doesn't need to ship a second spaCy model. Loads a separate
        instance to avoid mutating the shared NER pipeline.
        """
        import pytextrank  # noqa: F401 — registers the pipeline component
        import spacy

        self._nlp = spacy.load("en_core_web_lg")
        self._nlp.add_pipe("textrank")
        logger.debug("Loaded en_core_web_lg with pytextrank pipeline")

    def extract(self, text: str, **kwargs) -> AlgorithmResult:
        """
        Extract keyphrases from text using TextRank.

        Args:
            text: Document text to analyze
            **kwargs: Not used by this algorithm

        Returns:
            AlgorithmResult with candidate keyphrases
        """
        start_time = time.time()

        # Lazy-load model
        if self._nlp is None:
            try:
                self._load_nlp()
            except Exception as e:
                logger.warning("TextRank unavailable: %s", e)
                return AlgorithmResult(
                    candidates=[],
                    processing_time_ms=0.0,
                    metadata={"skipped": True, "reason": str(e)},
                )

        # Truncate very long texts (TextRank is O(n^2) on vocabulary)
        max_chars = 200_000
        truncated = len(text) > max_chars
        process_text = text[:max_chars] if truncated else text

        doc = self._nlp(process_text)

        candidates = []
        seen_phrases: set[str] = set()

        for phrase in doc._.phrases:
            if len(candidates) >= self.max_candidates:
                break

            phrase_text = phrase.text.strip()

            # Skip single-char terms
            if len(phrase_text) < 2:
                continue

            # Skip pure numbers
            if phrase_text.replace(" ", "").isdigit():
                continue

            # Skip pure stopwords (single-word case)
            lower_text = phrase_text.lower()
            if " " not in phrase_text:
                from src.core.utils.tokenizer import STOPWORDS

                if lower_text in STOPWORDS:
                    continue

            # Dedup
            if lower_text in seen_phrases:
                continue
            seen_phrases.add(lower_text)

            # pytextrank rank score is already 0-1 normalized
            confidence = min(phrase.rank, 1.0)

            # Count occurrences from pytextrank's chunk count
            frequency = phrase.count

            candidates.append(
                CandidateTerm(
                    term=phrase_text.title(),
                    source_algorithm=self.name,
                    confidence=confidence,
                    suggested_type="Technical",
                    frequency=max(frequency, 1),
                    metadata={
                        "textrank_score": round(phrase.rank, 4),
                        "chunk_count": phrase.count,
                        "word_count": len(phrase_text.split()),
                    },
                )
            )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            "Extracted %d keyphrases in %.1fms (truncated: %s)",
            len(candidates),
            processing_time_ms,
            truncated,
        )

        return AlgorithmResult(
            candidates=candidates,
            processing_time_ms=processing_time_ms,
            metadata={
                "total_phrases_found": len(doc._.phrases) if doc else 0,
                "filtered_candidates": len(candidates),
                "text_truncated": truncated,
            },
        )

    def get_config(self) -> dict[str, Any]:
        """Return algorithm configuration."""
        return {
            **super().get_config(),
            "max_candidates": self.max_candidates,
        }
