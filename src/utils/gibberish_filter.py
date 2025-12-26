"""
Gibberish Filter Utility

Detects nonsense/random character sequences that pass pattern-based filters.
Uses spell checking to identify words that are:
1. Not in the dictionary
2. Have no reasonable corrections (true gibberish, not typos)

NOTE: This filter should NOT be applied to PERSON entities, as foreign
names may incorrectly trigger gibberish detection.

Usage:
    from src.utils.gibberish_filter import is_gibberish

    if is_gibberish("xkjwqr"):
        print("Detected gibberish")
"""

import logging

from spellchecker import SpellChecker

logger = logging.getLogger(__name__)


class GibberishFilter:
    """
    Wrapper for gibberish detection using spell checking.

    A word is considered gibberish if:
    1. It's not in the dictionary (unknown)
    2. It has no reasonable spelling corrections

    This distinguishes true garbage (random characters) from typos
    (which have corrections) and valid rare words (which are in dictionary).
    """

    _instance = None
    _spell = None

    def __init__(self):
        """Initialize the spell checker."""
        self._spell = SpellChecker()

    @classmethod
    def get_instance(cls) -> 'GibberishFilter':
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

        A word is gibberish if it's unknown AND has no corrections.
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
        Check if a single word is gibberish.

        Args:
            word: Cleaned lowercase word to check

        Returns:
            True if word is gibberish (unknown with no corrections)
        """
        # If word is in dictionary, not gibberish
        if word in self._spell:
            return False

        # If word has corrections, it's a typo, not gibberish
        corrections = self._spell.candidates(word)
        if corrections and len(corrections) > 0:
            return False

        # Unknown word with no corrections = gibberish
        return True

    def _clean_for_check(self, word: str) -> str:
        """
        Clean word for spell checking.

        Removes punctuation and lowercases for consistent checking.

        Args:
            word: Word to clean

        Returns:
            Cleaned word
        """
        cleaned = ''.join(c for c in word if c.isalpha())
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
