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

FILTERING SCOPE:
This algorithm handles basic validation only:
- Score threshold (min RAKE score)
- Phrase length (2-50 characters)
- Stopword removal for single-word results
- Invalid phrase filtering (numbers only, etc.)

Multi-word PHRASE filtering (based on component word commonality) is done
CENTRALLY by rarity_filter.py after all algorithms contribute their candidates.
This is important because RAKE may extract phrases like "the same" or "left side"
that score well algorithmically but contain only common words.

Reference:
Rose, S., et al. (2010). "Automatic Keyword Extraction from Individual Documents"
"""

import re
import time
from typing import Any

from rake_nltk import Rake

from src.config import VOCAB_ALGORITHM_WEIGHTS
from src.logging_config import debug_log
from src.core.utils.tokenizer import STOPWORDS
from src.core.vocabulary.algorithms import register_algorithm
from src.core.vocabulary.algorithms.base import (
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

        # Keep original text for frequency counting
        original_text_lower = text.lower()

        # Clean text for RAKE processing
        cleaned_text = self._preprocess_text(text)

        # Extract keywords
        self.rake.extract_keywords_from_text(cleaned_text)

        # Get ranked phrases with scores
        ranked_phrases = self.rake.get_ranked_phrases_with_scores()

        candidates = []
        seen_phrases: set[str] = set()

        # Track filtering stats for debugging
        filtered_by_score = 0
        filtered_by_invalid = 0
        filtered_by_frequency = 0

        for score, phrase in ranked_phrases:
            # Skip low-scoring phrases
            if score < self.min_score:
                filtered_by_score += 1
                continue

            # Skip if we've hit our limit
            if len(candidates) >= self.max_candidates:
                break

            # Clean and validate the phrase
            cleaned_phrase = self._clean_phrase(phrase)
            if not cleaned_phrase:
                filtered_by_invalid += 1
                continue

            # Skip if already added (dedup)
            lower_phrase = cleaned_phrase.lower()
            if lower_phrase in seen_phrases:
                continue
            seen_phrases.add(lower_phrase)

            # Count actual occurrences in original text (case-insensitive)
            actual_frequency = self._count_phrase_occurrences(lower_phrase, original_text_lower)

            # Skip if below minimum frequency
            if actual_frequency < self.min_frequency:
                filtered_by_frequency += 1
                continue

            # Calculate confidence from RAKE score (normalize to 0-1)
            # RAKE scores typically range from 1-50, with most good phrases 3-15
            confidence = min(score / 15.0, 1.0)

            candidates.append(CandidateTerm(
                term=cleaned_phrase,
                source_algorithm=self.name,
                confidence=confidence,
                suggested_type="Technical",  # RAKE primarily finds technical phrases
                frequency=actual_frequency,
                metadata={
                    "rake_score": score,
                    "word_count": len(cleaned_phrase.split()),
                }
            ))

        processing_time_ms = (time.time() - start_time) * 1000

        debug_log(
            f"[RAKE] Extracted {len(candidates)} phrases from "
            f"{len(ranked_phrases)} raw (filtered: {filtered_by_score} low-score, "
            f"{filtered_by_frequency} low-freq, {filtered_by_invalid} invalid) "
            f"in {processing_time_ms:.1f}ms"
        )

        return AlgorithmResult(
            candidates=candidates,
            processing_time_ms=processing_time_ms,
            metadata={
                "raw_phrases_found": len(ranked_phrases),
                "filtered_candidates": len(candidates),
                "filtered_by_score": filtered_by_score,
                "filtered_by_frequency": filtered_by_frequency,
                "filtered_by_invalid": filtered_by_invalid,
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

    def _count_phrase_occurrences(self, phrase_lower: str, text_lower: str) -> int:
        """
        Count actual occurrences of a phrase in text.

        Uses word boundary matching to avoid partial matches.
        For example, "spinal" shouldn't match inside "transpedicular spinal fusion".

        Args:
            phrase_lower: Lowercase phrase to search for
            text_lower: Lowercase text to search in

        Returns:
            Number of times phrase appears in text
        """
        # Use word boundaries for accurate counting
        # \b matches word boundaries (start/end of word)
        pattern = r'\b' + re.escape(phrase_lower) + r'\b'
        matches = re.findall(pattern, text_lower)
        return len(matches)

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
                    lower_cleaned = cleaned.lower()
                    break

        # Session 53: Filter single-word results that are common stopwords
        # RAKE sometimes returns single common words like "same", "left", "also"
        if ' ' not in cleaned and lower_cleaned in STOPWORDS:
            return ""

        # Session 78: Use title case for proper noun consistency
        # Previously only capitalized first letter, causing "Luigi napolitano"
        # instead of "Luigi Napolitano"
        if cleaned:
            cleaned = cleaned.title()

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
