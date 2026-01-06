"""
Tests for the Name Regularizer module.

Tests the two filtering strategies:
1. Fragment filter - removes word fragments of canonical terms
2. Typo filter - removes 1-character edit distance variants
"""

import pytest

from src.core.vocabulary.name_regularizer import (
    _is_fragment_of,
    filter_name_fragments,
    filter_typo_variants,
    regularize_names,
)
from src.core.vocabulary.string_utils import edit_distance as _edit_distance


class TestEditDistance:
    """Tests for the edit distance calculation."""

    def test_identical_strings(self):
        """Edit distance of identical strings is 0."""
        assert _edit_distance("hello", "hello") == 0
        assert _edit_distance("Barbra Jenkins", "Barbra Jenkins") == 0

    def test_single_insertion(self):
        """One character insertion = distance 1."""
        assert _edit_distance("test", "tests") == 1
        assert _edit_distance("Jenkins", "Jenkinss") == 1

    def test_single_deletion(self):
        """One character deletion = distance 1."""
        assert _edit_distance("tests", "test") == 1
        assert _edit_distance("Barbra", "Barbr") == 1

    def test_single_substitution(self):
        """One character substitution = distance 1."""
        assert _edit_distance("cat", "bat") == 1
        assert _edit_distance("Jenkins", "Jenkinb") == 1

    def test_multiple_edits(self):
        """Multiple edits accumulate."""
        assert _edit_distance("kitten", "sitting") == 3
        assert _edit_distance("abc", "xyz") == 3

    def test_empty_string(self):
        """Empty string has distance = length of other string."""
        assert _edit_distance("", "hello") == 5
        assert _edit_distance("hello", "") == 5
        assert _edit_distance("", "") == 0

    def test_case_sensitive(self):
        """Edit distance is case-sensitive."""
        assert _edit_distance("Hello", "hello") == 1
        assert _edit_distance("ABC", "abc") == 3


class TestIsFragmentOf:
    """Tests for fragment detection."""

    def test_single_word_fragment_of_two_word(self):
        """Single word from two-word name is a fragment."""
        assert _is_fragment_of("Di", "Di Leo") is True
        assert _is_fragment_of("Leo", "Di Leo") is True

    def test_single_word_fragment_of_three_word(self):
        """Single word from three-word name is a fragment."""
        assert _is_fragment_of("Memorial", "Memorial General Hospital") is True
        assert _is_fragment_of("Hospital", "Memorial General Hospital") is True

    def test_two_word_fragment_of_three_word(self):
        """Two words from three-word name is a fragment."""
        assert _is_fragment_of("Memorial Hospital", "Memorial General Hospital") is True

    def test_exact_match_not_fragment(self):
        """Exact match is not a fragment (not a PROPER subset)."""
        assert _is_fragment_of("Di Leo", "Di Leo") is False
        assert _is_fragment_of("Jenkins", "Jenkins") is False

    def test_longer_not_fragment(self):
        """Longer string cannot be a fragment of shorter."""
        assert _is_fragment_of("Di Leo Smith", "Di Leo") is False
        assert _is_fragment_of("Memorial Hospital", "Hospital") is False

    def test_partial_word_not_fragment(self):
        """Partial word match is NOT a fragment."""
        # "Hospital" is not a fragment of "Memorial Hospital" if we're checking
        # "Hosp" against "Memorial Hospital" - but "Hospital" as a whole word IS
        assert _is_fragment_of("Memor", "Memorial Hospital") is False
        assert _is_fragment_of("Hospit", "Memorial Hospital") is False

    def test_unrelated_words_not_fragment(self):
        """Unrelated words are not fragments."""
        assert _is_fragment_of("Smith", "Di Leo") is False
        assert _is_fragment_of("John", "Jane Doe") is False

    def test_case_insensitive(self):
        """Fragment check is case-insensitive."""
        assert _is_fragment_of("di", "Di Leo") is True
        assert _is_fragment_of("DI", "di leo") is True
        assert _is_fragment_of("LEO", "Di Leo") is True


class TestFilterNameFragments:
    """Tests for the fragment filter function."""

    def _make_vocab(self, terms_and_counts: list[tuple[str, int]]) -> list[dict]:
        """Helper to create vocabulary list from (term, count) tuples."""
        return [
            {"Term": term, "In-Case Freq": count}
            for term, count in terms_and_counts
        ]

    def test_removes_fragments_from_bottom(self):
        """Fragments in bottom 3/4 are removed if canonical is in top quartile."""
        vocab = self._make_vocab([
            ("Di Leo", 50),      # Top quartile - canonical
            ("Smith", 40),       # Top quartile
            ("Johnson", 30),     # Top quartile
            ("Brown", 25),       # Top quartile (4 items = quartile of 1)
            ("Di", 5),           # Bottom - fragment of "Di Leo"
            ("Leo", 3),          # Bottom - fragment of "Di Leo"
            ("Jones", 10),       # Bottom - not a fragment
        ])

        result = filter_name_fragments(vocab, top_fraction=0.25)

        terms = [r["Term"] for r in result]
        assert "Di Leo" in terms
        assert "Di" not in terms
        assert "Leo" not in terms
        assert "Jones" in terms  # Should remain

    def test_preserves_canonical_terms(self):
        """Canonical terms are never removed, even if they're fragments of each other."""
        vocab = self._make_vocab([
            ("Memorial Hospital", 100),
            ("Memorial", 90),  # Would be fragment, but it's in top fraction
            ("Hospital", 80),  # Would be fragment, but it's in top fraction
            ("Other", 10),
        ])

        # With 75% top fraction, first 3 of 4 items are in top
        result = filter_name_fragments(vocab, top_fraction=0.75)

        terms = [r["Term"] for r in result]
        assert "Memorial Hospital" in terms
        assert "Memorial" in terms  # In top fraction, preserved
        assert "Hospital" in terms  # In top fraction, preserved

    def test_empty_vocabulary(self):
        """Empty vocabulary returns empty."""
        assert filter_name_fragments([]) == []

    def test_small_vocabulary(self):
        """Small vocabulary (< 4 items) returns unchanged."""
        vocab = self._make_vocab([("A", 10), ("B", 5), ("C", 3)])
        result = filter_name_fragments(vocab)
        assert len(result) == 3

    def test_no_multiword_canonicals(self):
        """If no multi-word terms in top quartile, nothing is filtered."""
        vocab = self._make_vocab([
            ("Smith", 50),
            ("Jones", 40),
            ("Brown", 30),
            ("Lee", 20),
            ("Di", 5),  # Not filtered - no multi-word in top
        ])

        result = filter_name_fragments(vocab)
        assert len(result) == 5


class TestFilterTypoVariants:
    """Tests for the typo filter function."""

    def _make_vocab(self, terms_and_counts: list[tuple[str, int]]) -> list[dict]:
        """Helper to create vocabulary list from (term, count) tuples."""
        return [
            {"Term": term, "In-Case Freq": count}
            for term, count in terms_and_counts
        ]

    def test_removes_one_char_typos(self):
        """Typos with 1-character difference are removed."""
        vocab = self._make_vocab([
            ("Barbra Jenkins", 50),     # Canonical
            ("Memorial Hospital", 40),  # Canonical
            ("Another Name", 30),       # Canonical
            ("Fourth Term", 25),        # Canonical
            ("Barbr Jenkins", 5),       # Typo - missing 'a'
            ("Barbra Jenkinss", 3),     # Typo - extra 's'
            ("Something Else", 10),     # Not a typo
        ])

        result = filter_typo_variants(vocab, top_fraction=0.25)

        terms = [r["Term"] for r in result]
        assert "Barbra Jenkins" in terms
        assert "Barbr Jenkins" not in terms
        assert "Barbra Jenkinss" not in terms
        assert "Something Else" in terms

    def test_preserves_two_char_difference(self):
        """Terms with 2+ character difference are preserved (not typos)."""
        vocab = self._make_vocab([
            ("Barbra Jenkins", 50),
            ("Second Term", 40),
            ("Third Term", 30),
            ("Fourth Term", 25),
            ("Barb Jenkins", 5),  # 2-char diff (ra -> _), preserved
        ])

        result = filter_typo_variants(vocab, top_fraction=0.25)

        terms = [r["Term"] for r in result]
        assert "Barb Jenkins" in terms  # 2-char diff, not filtered

    def test_skips_short_terms(self):
        """Short terms (< min_length) are not typo-checked."""
        vocab = self._make_vocab([
            ("Long Canonical Term", 50),
            ("Another Term", 40),
            ("Third Term", 30),
            ("Fourth", 25),
            ("Di", 5),   # Too short for typo check
            ("Do", 3),   # 1-char diff from "Di" but both too short
        ])

        result = filter_typo_variants(vocab, top_fraction=0.25, min_term_length=5)

        terms = [r["Term"] for r in result]
        assert "Di" in terms  # Too short, preserved
        assert "Do" in terms  # Too short, preserved

    def test_empty_vocabulary(self):
        """Empty vocabulary returns empty."""
        assert filter_typo_variants([]) == []

    def test_small_vocabulary(self):
        """Small vocabulary (< 4 items) returns unchanged."""
        vocab = self._make_vocab([("ABC", 10), ("DEF", 5), ("GHI", 3)])
        result = filter_typo_variants(vocab)
        assert len(result) == 3


class TestRegularizeNames:
    """Tests for the combined regularization function."""

    def _make_vocab(self, terms_and_counts: list[tuple[str, int]]) -> list[dict]:
        """Helper to create vocabulary list from (term, count) tuples."""
        return [
            {"Term": term, "In-Case Freq": count}
            for term, count in terms_and_counts
        ]

    def test_applies_both_filters(self):
        """Both fragment and typo filters are applied."""
        # With 8 items and top_fraction=0.25, top 2 are canonical
        vocab = self._make_vocab([
            ("Di Leo", 100),             # Top - canonical multi-word
            ("Barbra Jenkins", 90),      # Top - canonical for typo check
            ("Third Term", 80),          # Bottom
            ("Fourth Term", 70),         # Bottom
            ("Di", 10),                  # Bottom - Fragment of "Di Leo"
            ("Leo", 8),                  # Bottom - Fragment of "Di Leo"
            ("Barbr Jenkins", 5),        # Bottom - Typo of "Barbra Jenkins"
            ("Valid Term", 3),           # Bottom - Should remain
        ])

        result = regularize_names(vocab, top_fraction=0.25)

        terms = [r["Term"] for r in result]
        assert "Di Leo" in terms
        assert "Barbra Jenkins" in terms
        assert "Di" not in terms           # Fragment removed
        assert "Leo" not in terms          # Fragment removed
        assert "Barbr Jenkins" not in terms  # Typo removed
        assert "Valid Term" in terms       # Preserved

    def test_empty_vocabulary(self):
        """Empty vocabulary returns empty."""
        assert regularize_names([]) == []

    def test_preserves_order_after_filtering(self):
        """Top terms remain at top after filtering."""
        vocab = self._make_vocab([
            ("High Count", 100),
            ("Second", 50),
            ("Third", 30),
            ("Fourth", 20),
            ("Low Count", 5),
        ])

        result = regularize_names(vocab)

        # First term should still be highest count
        assert result[0]["Term"] == "High Count"
        assert result[0]["In-Case Freq"] == 100


class TestRealWorldScenarios:
    """Tests based on real-world use cases from the issue description."""

    def _make_vocab(self, terms_and_counts: list[tuple[str, int]]) -> list[dict]:
        """Helper to create vocabulary list from (term, count) tuples."""
        return [
            {"Term": term, "In-Case Freq": count}
            for term, count in terms_and_counts
        ]

    def test_ms_di_leo_scenario(self):
        """
        The original issue: "Ms. Di Leo" should not produce "Di" and "Leo" separately.
        """
        vocab = self._make_vocab([
            ("Ms. Di Leo", 50),   # Canonical
            ("Dr. Smith", 45),
            ("Hospital Name", 40),
            ("John Doe", 35),
            ("Jane Doe", 30),
            ("Di", 3),           # Should be removed
            ("Leo", 2),          # Should be removed
            ("Other Person", 10),
        ])

        result = regularize_names(vocab, top_fraction=0.25)

        terms = [r["Term"] for r in result]
        assert "Ms. Di Leo" in terms
        assert "Di" not in terms
        assert "Leo" not in terms
        assert "Other Person" in terms

    def test_barbra_jenkins_typo_scenario(self):
        """
        The typo scenario: "Barbra Jenkins" variants with OCR errors.
        """
        vocab = self._make_vocab([
            ("Barbra Jenkins", 75),    # Canonical
            ("Memorial Hospital", 60),
            ("Dr. Williams", 50),
            ("North Shore", 40),
            ("Barbr Jenkins", 5),      # OCR error - missing 'a'
            ("Barbra Jenkibs", 3),     # OCR error - 'n' → 'b'
            ("Barbra Jenkinss", 2),    # OCR error - extra 's'
            ("Completely Different", 20),
        ])

        result = regularize_names(vocab, top_fraction=0.25)

        terms = [r["Term"] for r in result]
        assert "Barbra Jenkins" in terms
        assert "Barbr Jenkins" not in terms
        assert "Barbra Jenkibs" not in terms
        assert "Barbra Jenkinss" not in terms
        assert "Completely Different" in terms

    def test_mixed_scenario(self):
        """Combined fragments and typos in one vocabulary."""
        # With 12 items and top_fraction=0.25, top 3 are canonical
        vocab = self._make_vocab([
            ("Wagner Doman Leto", 100),  # Top - canonical multi-word
            ("Robert Wighton", 95),      # Top - canonical for typo check
            ("Di Leo", 90),              # Top - canonical multi-word
            ("Memorial Hospital", 70),   # Bottom - but not a typo target
            ("Valid Entry", 50),         # Bottom - should remain
            ("Wagner", 15),              # Bottom - Fragment
            ("Doman", 12),               # Bottom - Fragment
            ("Leto", 10),                # Bottom - Fragment
            ("Di", 8),                   # Bottom - Fragment
            ("Leo", 6),                  # Bottom - Fragment
            ("Robrt Wighton", 4),        # Bottom - Typo
            ("Robert Wightn", 3),        # Bottom - Typo
        ])

        # Use min_canonical_count=3 to match top_fraction=0.25 (12 * 0.25 = 3)
        result = regularize_names(vocab, top_fraction=0.25, min_canonical_count=3)

        terms = [r["Term"] for r in result]

        # Top canonical terms preserved
        assert "Wagner Doman Leto" in terms
        assert "Robert Wighton" in terms
        assert "Di Leo" in terms

        # Non-canonical but valid terms preserved
        assert "Memorial Hospital" in terms
        assert "Valid Entry" in terms

        # Fragments removed (single words that are subsets of multi-word canonicals)
        assert "Di" not in terms
        assert "Leo" not in terms

        # Session 78: Typos are now handled by CanonicalScorer with weighted scoring
        # When no variants are known, the higher-weighted score wins
        # "Robert Wighton" (count=100) wins over lower-frequency typos
        assert "Robrt Wighton" not in terms  # Removed - lower weighted score
        assert "Robert Wightn" not in terms  # Removed - lower weighted score
