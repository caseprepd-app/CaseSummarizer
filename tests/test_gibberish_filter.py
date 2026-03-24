"""
Tests for gibberish_filter.py — gibberish detection via spell checking.

Uses dual metrics (edit distance ratio + sequence similarity) to distinguish
real typos from PDF garbage.
"""

import pytest

from src.core.utils.gibberish_filter import GibberishFilter, is_gibberish


@pytest.fixture()
def gf():
    """Fresh GibberishFilter instance (not singleton)."""
    return GibberishFilter()


# =========================================================================
# Core gibberish detection
# =========================================================================


class TestGibberishDetection:
    """Tests for is_gibberish behavior."""

    def test_real_english_words_not_gibberish(self, gf):
        """Common English words pass."""
        assert not gf.is_gibberish("plaintiff")
        assert not gf.is_gibberish("hospital")
        assert not gf.is_gibberish("motion")

    def test_obvious_gibberish_detected(self, gf):
        """Random character sequences are gibberish."""
        assert gf.is_gibberish("xkjwqrzm")
        assert gf.is_gibberish("qzxvbnml")

    def test_short_text_not_gibberish(self, gf):
        """Strings under 4 chars are never flagged."""
        assert not gf.is_gibberish("the")
        assert not gf.is_gibberish("a")
        assert not gf.is_gibberish("")

    def test_empty_string(self, gf):
        """Empty string is not gibberish."""
        assert not gf.is_gibberish("")

    def test_real_typo_not_gibberish(self, gf):
        """Close typos of real words should pass (e.g. Smitb -> Smith)."""
        # These are close enough to real words that dual metrics should pass
        assert not gf.is_gibberish("hopsital")

    def test_multi_word_any_gibberish_fails(self, gf):
        """Multi-word phrase fails if ANY word is gibberish."""
        assert gf.is_gibberish("good xkjwqrzm text")

    def test_multi_word_all_real_passes(self, gf):
        """Multi-word phrase of real words passes."""
        assert not gf.is_gibberish("the quick brown fox")

    def test_pdf_garbage_detected(self, gf):
        """Common PDF extraction garbage is caught."""
        assert gf.is_gibberish("modmessxyz")
        assert gf.is_gibberish("zqxwvp")


# =========================================================================
# Internal helpers
# =========================================================================


class TestCleanForCheck:
    """Tests for _clean_for_check."""

    def test_strips_punctuation(self, gf):
        """Removes non-alpha characters."""
        assert gf._clean_for_check("hello!") == "hello"
        assert gf._clean_for_check("(test)") == "test"

    def test_lowercases(self, gf):
        """Result is lowercased."""
        assert gf._clean_for_check("HELLO") == "hello"

    def test_empty_after_cleaning(self, gf):
        """All-punctuation returns empty string."""
        assert gf._clean_for_check("!!!") == ""


# =========================================================================
# Singleton
# =========================================================================


class TestSingleton:
    """Tests for get_instance singleton pattern."""

    def test_returns_instance(self):
        """get_instance returns a GibberishFilter."""
        GibberishFilter._instance = None
        inst = GibberishFilter.get_instance()
        assert isinstance(inst, GibberishFilter)

    def test_same_instance_returned(self):
        """Subsequent calls return the same instance."""
        GibberishFilter._instance = None
        a = GibberishFilter.get_instance()
        b = GibberishFilter.get_instance()
        assert a is b


# =========================================================================
# Convenience function
# =========================================================================


class TestConvenienceFunction:
    """Tests for the module-level is_gibberish() function."""

    def test_delegates_to_singleton(self):
        """Module-level function uses singleton."""
        GibberishFilter._instance = None
        result = is_gibberish("xkjwqrzm")
        assert result is True
        assert GibberishFilter._instance is not None

    def test_real_word_passes(self):
        """Module-level function passes real words."""
        assert not is_gibberish("hospital")
