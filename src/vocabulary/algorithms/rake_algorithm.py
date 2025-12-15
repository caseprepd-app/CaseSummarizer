"""
RAKE (Rapid Automatic Keyword Extraction) Algorithm

RAKE is a domain-independent keyword extraction algorithm that uses word frequency
and word co-occurrence to identify key phrases in text. It's particularly good at
finding multi-word technical phrases that NER might miss.

RAKE works by:
1. Splitting text at stopwords and punctuation (phrase delimiters)
2. Calculating word scores based on frequency and degree (co-occurrence)
3. Summing word scores to get phrase scores
4. Ranking phrases by score

This complements NER by finding key phrases that aren't named entities.

Reference:
Rose, S., et al. (2010). "Automatic Keyword Extraction from Individual Documents"
"""

import re
import time
from typing import Any

from rake_nltk import Rake

from src.config import VOCAB_ALGORITHM_WEIGHTS
from src.logging_config import debug_log
from src.vocabulary.algorithms import register_algorithm
from src.vocabulary.algorithms.base import (
    AlgorithmResult,
    BaseExtractionAlgorithm,
    CandidateTerm,
)


@register_algorithm("RAKE")
class RAKEAlgorithm(BaseExtractionAlgorithm):
    """
    RAKE keyword extraction algorithm.

    Extracts key phrases from text using word co-occurrence statistics.
    Particularly effective for finding:
    - Technical terminology
    - Multi-word concepts
    - Domain-specific phrases

    Lower weight than NER because RAKE can produce noise (common phrases
    that happen to have high co-occurrence scores).
    """

    name = "RAKE"
    weight = VOCAB_ALGORITHM_WEIGHTS.get("RAKE", 0.7)  # Secondary algorithm

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 3,
        include_stopwords: bool = False,
        min_frequency: int = 1,
        max_candidates: int = 150,
        min_score: float = 2.0,
    ):
        """
        Initialize RAKE algorithm.

        Args:
            min_length: Minimum number of words in a phrase (default: 1)
            max_length: Maximum number of words in a phrase (default: 3)
            include_stopwords: Whether to include common stopwords (default: False)
            min_frequency: Minimum times a phrase must appear (default: 1)
            max_candidates: Maximum candidates to return (default: 150)
            min_score: Minimum RAKE score to consider (default: 2.0)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.include_stopwords = include_stopwords
        self.min_frequency = min_frequency
        self.max_candidates = max_candidates
        self.min_score = min_score

        # Initialize RAKE with our settings
        self._rake = None

    @property
    def rake(self) -> Rake:
        """Lazy-load RAKE instance."""
        if self._rake is None:
            self._rake = Rake(
                min_length=self.min_length,
                max_length=self.max_length,
                include_repeated_phrases=True,
            )
        return self._rake

    def extract(self, text: str, **kwargs) -> AlgorithmResult:
        """
        Extract key phrases from text using RAKE.

        Args:
            text: Document text to analyze
            **kwargs: Not used by this algorithm

        Returns:
            AlgorithmResult with candidate phrases
        """
        start_time = time.time()

        # Clean text for RAKE processing
        cleaned_text = self._preprocess_text(text)

        # Extract keywords
        self.rake.extract_keywords_from_text(cleaned_text)

        # Get ranked phrases with scores
        ranked_phrases = self.rake.get_ranked_phrases_with_scores()

        candidates = []
        phrase_counts: dict[str, int] = {}

        for score, phrase in ranked_phrases:
            # Skip low-scoring phrases
            if score < self.min_score:
                continue

            # Skip if we've hit our limit
            if len(candidates) >= self.max_candidates:
                break

            # Clean and validate the phrase
            cleaned_phrase = self._clean_phrase(phrase)
            if not cleaned_phrase:
                continue

            # Track frequency (RAKE may find same phrase multiple times)
            lower_phrase = cleaned_phrase.lower()
            phrase_counts[lower_phrase] = phrase_counts.get(lower_phrase, 0) + 1

            # Skip if below minimum frequency
            if phrase_counts[lower_phrase] < self.min_frequency:
                continue

            # Skip if already added (dedup)
            if any(c.term.lower() == lower_phrase for c in candidates):
                continue

            # Calculate confidence from RAKE score (normalize to 0-1)
            # RAKE scores typically range from 1-50, with most good phrases 3-15
            confidence = min(score / 15.0, 1.0)

            candidates.append(CandidateTerm(
                term=cleaned_phrase,
                source_algorithm=self.name,
                confidence=confidence,
                suggested_type="Technical",  # RAKE primarily finds technical phrases
                frequency=phrase_counts[lower_phrase],
                metadata={
                    "rake_score": score,
                    "word_count": len(cleaned_phrase.split()),
                }
            ))

        processing_time_ms = (time.time() - start_time) * 1000

        debug_log(
            f"[RAKE] Extracted {len(candidates)} phrases from "
            f"{len(ranked_phrases)} raw candidates in {processing_time_ms:.1f}ms"
        )

        return AlgorithmResult(
            candidates=candidates,
            processing_time_ms=processing_time_ms,
            metadata={
                "raw_phrases_found": len(ranked_phrases),
                "filtered_candidates": len(candidates),
                "min_score_threshold": self.min_score,
            }
        )

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for RAKE extraction.

        Removes elements that confuse RAKE:
        - Page numbers
        - Line numbers (common in legal transcripts)
        - Excessive whitespace

        Args:
            text: Raw document text

        Returns:
            Cleaned text suitable for RAKE processing
        """
        cleaned = text

        # Remove line numbers at start of lines (common in transcripts)
        cleaned = re.sub(r'^\s*\d{1,2}\s+', '', cleaned, flags=re.MULTILINE)

        # Remove standalone numbers (page numbers, etc.)
        cleaned = re.sub(r'\b\d+\b', '', cleaned)

        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned.strip()

    def _clean_phrase(self, phrase: str) -> str:
        """
        Clean and validate a RAKE phrase.

        Args:
            phrase: Raw phrase from RAKE

        Returns:
            Cleaned phrase, or empty string if invalid
        """
        # Basic cleanup
        cleaned = phrase.strip()

        # Skip single characters
        if len(cleaned) < 2:
            return ""

        # Skip phrases that are just numbers or punctuation
        if not re.search(r'[a-zA-Z]', cleaned):
            return ""

        # Skip very long phrases (likely noise)
        if len(cleaned) > 50:
            return ""

        # Skip phrases starting/ending with common junk
        junk_starts = ('the', 'a', 'an', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 'for')
        lower_cleaned = cleaned.lower()
        if any(lower_cleaned.startswith(j + ' ') for j in junk_starts):
            # Strip the junk prefix
            for j in junk_starts:
                if lower_cleaned.startswith(j + ' '):
                    cleaned = cleaned[len(j) + 1:]
                    break

        # Capitalize first letter for consistency
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned.strip()

    def get_config(self) -> dict[str, Any]:
        """Return algorithm configuration."""
        return {
            **super().get_config(),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "min_frequency": self.min_frequency,
            "max_candidates": self.max_candidates,
            "min_score": self.min_score,
        }
