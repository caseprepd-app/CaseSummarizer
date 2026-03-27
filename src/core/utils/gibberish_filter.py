"""
Gibberish Filter Utility

Detects nonsense/random character sequences that pass pattern-based filters.
Two-layer approach:

Layer 1 — Markov chain (gibberish-detector): Scores character bigram
    probabilities against English. Catches obvious garbage like "xkjwqr"
    or "qqqqq" cheaply — no spell-checker lookup needed.

Layer 2 — Spell-checker with dual metrics: For words that *look* plausibly
    English (pass Markov check), uses edit distance ratio + sequence
    similarity to distinguish real typos from subtler garbage like "modmess".

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
import string
import threading

from spellchecker import SpellChecker

from src.config import EDIT_DISTANCE_RATIO_THRESHOLD, GIBBERISH_SIMILARITY_THRESHOLD
from src.core.vocabulary.string_utils import edit_distance, fuzzy_match

logger = logging.getLogger(__name__)

# Alias for backward compatibility (used by is_gibberish function)
SIMILARITY_THRESHOLD = GIBBERISH_SIMILARITY_THRESHOLD


def _build_markov_detector():
    """
    Build and cache a Markov chain gibberish detector trained on NLTK words.

    Returns:
        gibberish_detector.detector.Detector instance, or None if unavailable
    """
    try:
        from gibberish_detector.detector import Detector
        from gibberish_detector.trainer import train_on_content
        from nltk.corpus import words as nltk_words

        corpus = "\n".join(nltk_words.words())
        model = train_on_content(corpus, string.ascii_lowercase)
        return Detector(model, threshold=5.5)
    except ImportError:
        logger.debug("gibberish-detector not installed; Markov layer disabled")
        return None
    except Exception as e:
        logger.warning("Failed to build Markov gibberish model: %s", e)
        return None


class GibberishFilter:
    """
    Two-layer gibberish detection: Markov chain + spell-checker.

    Layer 1 (fast): Markov chain scores character bigram probabilities.
        Words with scores above 4.0 are obvious gibberish (e.g. "xkjwqr").

    Layer 2 (thorough): For words that pass the Markov check, uses spell
        checking with dual metrics (edit distance ratio AND similarity).
        Both must pass for a word to be considered a valid typo.

    A word is considered gibberish if:
    - Markov check flags it as gibberish, OR
    - It's not in the dictionary AND has no close correction
    """

    _instance = None
    _instance_lock = threading.Lock()
    _spell = None
    _markov = None

    def __init__(self):
        """Initialize the spell checker and Markov detector."""
        self._spell = SpellChecker()
        self._markov = _build_markov_detector()
        if self._markov:
            logger.debug("Markov gibberish detector loaded")

    @classmethod
    def get_instance(cls) -> "GibberishFilter":
        """
        Get singleton instance (lazy load, thread-safe).

        Returns:
            The shared GibberishFilter instance
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._instance_lock:
            cls._instance = None

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
        Check if a single word is gibberish using two layers.

        Layer 1: Markov chain — catches obvious garbage cheaply.
        Layer 2: Spell-checker with dual metrics — catches subtler noise.

        Args:
            word: Cleaned lowercase word to check

        Returns:
            True if word is gibberish
        """
        # If word is in dictionary, not gibberish
        if word in self._spell:
            return False

        # Layer 1: Markov chain (fast pre-check)
        if self._markov and self._markov.is_gibberish(word):
            logger.debug("Markov flagged gibberish: '%s'", word)
            return True

        # Layer 2: Spell-checker with dual metrics
        return self._spellcheck_gibberish(word)

    def _spellcheck_gibberish(self, word: str) -> bool:
        """
        Check if a word is gibberish using spell-checker dual metrics.

        Uses edit distance ratio AND sequence similarity.
        Both must pass for a word to be considered a valid typo.

        Args:
            word: Cleaned lowercase word to check

        Returns:
            True if word is gibberish
        """
        # Short words (<=4 chars): metrics break down, be lenient
        if len(word) <= 4:
            best = self._spell.correction(word)
            if best and best != word:
                return False  # Has a correction = probably typo
            candidates = self._spell.candidates(word)
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
