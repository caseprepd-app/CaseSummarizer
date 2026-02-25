"""
Gibberish Filter Utility

Detects nonsense/random character sequences that pass pattern-based filters.
Uses spell checking combined with edit distance and sequence similarity
to distinguish true gibberish from typos.

Uses dual metrics:
- Edit distance ratio: How many characters changed relative to word length
- Sequence similarity: How similar the word is to its best correction
- BOTH must pass thresholds, otherwise it's gibberish (strict when they disagree)

This catches PDF garbage like "modmess" (sim=0.77) while allowing real typos
like "Jenidns" (sim=0.86) and "Smitb" (sim=0.80).

NOTE: This filter should NOT be applied to PERSON entities, as foreign
names may incorrectly trigger gibberish detection.

Usage:
    from src.core.utils.gibberish_filter import is_gibberish

    if is_gibberish("xkjwqr"):
        print("Detected gibberish")
"""

import logging

from spellchecker import SpellChecker

from src.config import EDIT_DISTANCE_RATIO_THRESHOLD, GIBBERISH_SIMILARITY_THRESHOLD
from src.core.vocabulary.string_utils import edit_distance, fuzzy_match

logger = logging.getLogger(__name__)

# Alias for backward compatibility (used by is_gibberish function)
SIMILARITY_THRESHOLD = GIBBERISH_SIMILARITY_THRESHOLD


class GibberishFilter:
    """
    Wrapper for gibberish detection using spell checking with dual metrics.

    Uses BOTH edit distance ratio AND sequence similarity to distinguish
    gibberish from typos.

    A word is considered gibberish if:
    1. It's not in the dictionary, AND
    2. It has no correction, OR its best correction fails EITHER metric:
       - Edit distance ratio > 35% (too many character changes)
       - Sequence similarity < 80% (not similar enough)

    This is stricter than just checking for corrections, catching PDF garbage
    like "modmess" while still allowing real typos like "Jenidns" and "Smitb".
    """

    _instance = None
    _spell = None

    def __init__(self):
        """Initialize the spell checker."""
        self._spell = SpellChecker()

    @classmethod
    def get_instance(cls) -> "GibberishFilter":
        """
        Get singleton instance (lazy load).

        Returns:
            The shared GibberishFilter instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_gibberish(self, text: str) -> bool:
        """
        Check if text appears to be gibberish/nonsense.

        Multi-word phrases are gibberish if ANY word is gibberish.

        Args:
            text: Text to check (single word or short phrase)

        Returns:
            True if text appears to be gibberish
        """
        if not text or len(text) < 4:
            return False

        words = text.split()
        for word in words:
            if len(word) < 4:
                continue
            clean_word = self._clean_for_check(word)
            if len(clean_word) < 4:
                continue
            if self._is_word_gibberish(clean_word):
                return True

        return False

    def _is_word_gibberish(self, word: str) -> bool:
        """
        Check if a single word is gibberish using dual metrics.

        Uses edit distance ratio AND sequence similarity.
        Both must pass for a word to be considered a valid typo.

        Args:
            word: Cleaned lowercase word to check

        Returns:
            True if word is gibberish
        """
        # If word is in dictionary, not gibberish
        if word in self._spell:
            return False

        # Short words (<=4 chars): metrics break down, be lenient
        if len(word) <= 4:
            best = self._spell.correction(word)
            if best and best != word:
                return False  # Has a correction = probably typo
            candidates = self._spell.candidates(word)
            # Has candidates = probably typo; otherwise gibberish
            return not candidates

        # Get best correction
        best = self._spell.correction(word)
        if best is None or best == word:
            return True  # No correction = gibberish

        # Calculate both metrics
        edit_dist = edit_distance(word, best)
        edit_ratio = edit_dist / len(word)
        _, similarity = fuzzy_match(word, best)

        # BOTH must pass - strict when they disagree
        ratio_pass = edit_ratio <= EDIT_DISTANCE_RATIO_THRESHOLD
        sim_pass = similarity >= SIMILARITY_THRESHOLD

        # Valid typo (both pass) = not gibberish; otherwise gibberish
        return not (ratio_pass and sim_pass)

    def _clean_for_check(self, word: str) -> str:
        """
        Clean word for spell checking.

        Removes punctuation and lowercases for consistent checking.

        Args:
            word: Word to clean

        Returns:
            Cleaned word
        """
        cleaned = "".join(c for c in word if c.isalpha())
        return cleaned.lower()


def is_gibberish(text: str) -> bool:
    """
    Convenience function to check if text is gibberish.

    Args:
        text: Text to check

    Returns:
        True if text appears to be gibberish
    """
    return GibberishFilter.get_instance().is_gibberish(text)
