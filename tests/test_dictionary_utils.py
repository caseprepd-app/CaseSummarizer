"""
Tests for dictionary_utils.py — DictionaryTextValidator.

Requires NLTK 'words' corpus. Tests skip if corpus not available.
"""

import pytest

try:
    from src.core.extraction.dictionary_utils import DictionaryTextValidator

    _NLTK_AVAILABLE = True
except (RuntimeError, LookupError):
    _NLTK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _NLTK_AVAILABLE, reason="NLTK words corpus not installed")


@pytest.fixture()
def validator():
    """Fresh DictionaryTextValidator."""
    return DictionaryTextValidator()


# =========================================================================
# is_valid_word
# =========================================================================


class TestIsValidWord:
    """Tests for is_valid_word()."""

    def test_common_english_word(self, validator):
        """Common words are valid."""
        assert validator.is_valid_word("plaintiff")
        assert validator.is_valid_word("hospital")
        assert validator.is_valid_word("motion")

    def test_case_insensitive(self, validator):
        """Lookup is case-insensitive."""
        assert validator.is_valid_word("Plaintiff")
        assert validator.is_valid_word("HOSPITAL")

    def test_strips_punctuation(self, validator):
        """Strips trailing punctuation before lookup."""
        assert validator.is_valid_word("plaintiff,")
        assert validator.is_valid_word("(hospital)")
        assert validator.is_valid_word("motion.")

    def test_gibberish_not_valid(self, validator):
        """Random strings are not valid words."""
        assert not validator.is_valid_word("xkjwqr")
        assert not validator.is_valid_word("zqxwvp")


# =========================================================================
# calculate_confidence
# =========================================================================


class TestCalculateConfidence:
    """Tests for calculate_confidence()."""

    def test_all_english_high_confidence(self, validator):
        """All-English text gets high confidence."""
        score = validator.calculate_confidence("the quick brown fox")
        assert score >= 90.0

    def test_empty_text_zero(self, validator):
        """Empty text returns 0.0."""
        assert validator.calculate_confidence("") == 0.0

    def test_no_alpha_tokens_zero(self, validator):
        """Text with no alphabetic tokens returns 0.0."""
        assert validator.calculate_confidence("123 456 789") == 0.0

    def test_mixed_text_moderate_confidence(self, validator):
        """Mix of real and garbage words gets moderate score."""
        score = validator.calculate_confidence("the xkjwqr brown zqxwvp")
        assert 20.0 <= score <= 80.0

    def test_pure_garbage_low_confidence(self, validator):
        """Pure garbage text gets low confidence."""
        score = validator.calculate_confidence("xkjwqr zqxwvp qwerty")
        assert score < 50.0


# =========================================================================
# tokenize_for_voting
# =========================================================================


class TestTokenizeForVoting:
    """Tests for tokenize_for_voting()."""

    def test_basic_split(self, validator):
        """Splits on whitespace preserving punctuation."""
        tokens = validator.tokenize_for_voting("Hello, world!")
        assert tokens == ["Hello,", "world!"]

    def test_empty_string(self, validator):
        """Empty string returns empty list."""
        assert validator.tokenize_for_voting("") == []

    def test_preserves_attached_punctuation(self, validator):
        """Punctuation stays attached to words."""
        tokens = validator.tokenize_for_voting("Dr. Smith's case")
        assert tokens == ["Dr.", "Smith's", "case"]


# =========================================================================
# Legal keywords
# =========================================================================


class TestLegalKeywords:
    """Tests for legal_keywords set."""

    def test_contains_core_legal_terms(self, validator):
        """Legal keywords include core court terms."""
        assert "COURT" in validator.legal_keywords
        assert "PLAINTIFF" in validator.legal_keywords
        assert "DEFENDANT" in validator.legal_keywords

    def test_all_uppercase(self, validator):
        """All legal keywords are uppercase."""
        for kw in validator.legal_keywords:
            assert kw == kw.upper()
