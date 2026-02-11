"""
TextRank Keyword Extraction Algorithm

Uses pytextrank (a spaCy pipeline component) to extract keyphrases via
graph-based ranking. TextRank builds a word co-occurrence graph and uses
PageRank to identify the most important phrases.

TextRank complements NER and RAKE:
- NER finds named entities (people, organizations, locations)
- RAKE uses co-occurrence statistics within stopword-delimited phrases
- TextRank uses graph centrality across the full document

This algorithm can share the NER pipeline's en_core_web_lg instance since
textrank is a read-only analysis pipe. Falls back to loading its own
instance if no shared model is provided.

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

    Can share the NER pipeline's spaCy model to save memory,
    or loads its own instance if none is provided.
    """

    name = "TextRank"
    weight = VOCAB_ALGORITHM_WEIGHTS.get("TextRank", 0.6)

    def __init__(self, max_candidates: int = 150, nlp=None):
        """
        Initialize TextRank algorithm.

        Args:
            max_candidates: Maximum number of phrases to return
            nlp: Optional shared spaCy model (e.g. from NER). If provided,
                 the textrank pipe is added to it. If None, loads its own.
        """
        self.max_candidates = max_candidates
        self._nlp = None

        if nlp is not None:
            import pytextrank  # noqa: F401 — registers the pipeline component

            self._nlp = nlp
            if "textrank" not in self._nlp.pipe_names:
                self._nlp.add_pipe("textrank")
                logger.debug("Added textrank pipe to shared spaCy model")
            else:
                logger.debug("Shared spaCy model already has textrank pipe")

    def _load_nlp(self):
        """
        Load spaCy model with pytextrank pipeline component.

        Fallback: loads en_core_web_lg with textrank when no shared
        model was provided at init time.
        """
        import pytextrank  # noqa: F401 — registers the pipeline component
        import spacy

        from src.config import SPACY_EN_CORE_WEB_LG_PATH

        if SPACY_EN_CORE_WEB_LG_PATH.exists():
            self._nlp = spacy.load(str(SPACY_EN_CORE_WEB_LG_PATH))
            logger.debug("Loaded bundled spaCy model: %s", SPACY_EN_CORE_WEB_LG_PATH)
        else:
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
        from src.config import TEXTRANK_MAX_TEXT_KB

        max_chars = TEXTRANK_MAX_TEXT_KB * 1024
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
