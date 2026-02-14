"""
Tests for safer name deduplication improvements.

Covers three absorption gates (ambiguity, common-word, shared-last-name)
and shared-last-name guards in title synthesis.
"""

from unittest.mock import patch

from src.core.vocabulary.name_deduplicator import (
    _absorb_single_word_names,
    _find_shared_last_names,
    _synthesize_titled_names,
    deduplicate_names,
)


def _person(name, occurrences=1):
    """Helper to create a Person term dict."""
    return {"Term": name, "Is Person": "Yes", "Occurrences": occurrences}


def _other(name, occurrences=1):
    """Helper to create a non-Person term dict."""
    return {"Term": name, "Is Person": "No", "Occurrences": occurrences}


def _term_names(terms):
    """Extract sorted Term names from a list of term dicts."""
    return sorted(t["Term"] for t in terms)


# ---------------------------------------------------------------------------
# _find_shared_last_names
# ---------------------------------------------------------------------------


class TestFindSharedLastNames:
    """Tests for _find_shared_last_names helper."""

    def test_detects_shared_last_name(self):
        """Two people with different first names sharing 'jones'."""
        terms = [_person("James Jones"), _person("Patricia Jones")]
        assert "jones" in _find_shared_last_names(terms)

    def test_no_shared_when_unique(self):
        """Different last names are not shared."""
        terms = [_person("James Jones"), _person("Patricia Smith")]
        assert _find_shared_last_names(terms) == set()

    def test_sees_through_titles(self):
        """Titled entries like 'Mr. Jones' count toward sharing detection."""
        terms = [_person("James Jones"), _person("Mr. Jones")]
        # "Mr. Jones" strips to "Jones" (single word, no first name part)
        # so there's "james" and "" -- only one distinct first, not shared.
        # But "Mrs. Jones" + "James Jones" has two people.
        terms2 = [_person("James Jones"), _person("Mrs. Patricia Jones")]
        assert "jones" in _find_shared_last_names(terms2)

    def test_single_word_names_ignored(self):
        """Single-word names have no first+last split, so don't count."""
        terms = [_person("Jones"), _person("James Jones")]
        # "Jones" is single-word after stripping, no first name
        assert _find_shared_last_names(terms) == set()


# ---------------------------------------------------------------------------
# Gate 1: Ambiguity (word appears in 2+ multi-word names)
# ---------------------------------------------------------------------------


class TestAmbiguityGate:
    """Single-word names matching 2+ multi-word names are NOT absorbed."""

    def test_ambiguous_word_not_absorbed(self):
        """'Hiraldo' matches both 'Emmanuel Hiraldo' and 'Giuseppe Hiraldo'."""
        terms = [
            _person("Hiraldo", 3),
            _person("Emmanuel Hiraldo", 5),
            _person("Giuseppe Hiraldo", 4),
        ]
        with patch("src.core.vocabulary.name_deduplicator.is_common_word", return_value=False):
            result = _absorb_single_word_names(terms)
        names = _term_names(result)
        assert "Hiraldo" in names, "Ambiguous single-word should be kept"

    def test_unambiguous_word_absorbed(self):
        """'Hiraldo' matches only one multi-word name -- safe to absorb."""
        terms = [
            _person("Hiraldo", 3),
            _person("Emmanuel Hiraldo", 5),
        ]
        with patch("src.core.vocabulary.name_deduplicator.is_common_word", return_value=False):
            result = _absorb_single_word_names(terms)
        names = _term_names(result)
        assert "Hiraldo" not in names, "Unambiguous single-word should be absorbed"
        target = next(t for t in result if t["Term"] == "Emmanuel Hiraldo")
        assert target["Occurrences"] == 8  # 5 + 3


# ---------------------------------------------------------------------------
# Gate 2: Common word
# ---------------------------------------------------------------------------


class TestCommonWordGate:
    """Common English words are NOT absorbed into multi-word Person names."""

    def test_common_word_not_absorbed(self):
        """'Park' is a common word and should not be absorbed."""
        terms = [
            _person("Park", 2),
            _person("North Central Park", 5),
        ]
        with patch("src.core.vocabulary.name_deduplicator.is_common_word", return_value=True):
            result = _absorb_single_word_names(terms)
        names = _term_names(result)
        assert "Park" in names, "Common word should not be absorbed"

    def test_uncommon_word_absorbed(self):
        """'Hiraldo' is not a common word and should be absorbed."""
        terms = [
            _person("Hiraldo", 3),
            _person("Emmanuel Hiraldo", 5),
        ]
        with patch("src.core.vocabulary.name_deduplicator.is_common_word", return_value=False):
            result = _absorb_single_word_names(terms)
        names = _term_names(result)
        assert "Hiraldo" not in names, "Uncommon word should be absorbed"


# ---------------------------------------------------------------------------
# Gate 3: Shared last name
# ---------------------------------------------------------------------------


class TestSharedLastNameGate:
    """Single-word names that are shared last names are NOT absorbed."""

    def test_shared_last_name_not_absorbed(self):
        """'Jones' is shared by multiple people -- keep it separate."""
        terms = [
            _person("Jones", 4),
            _person("James Jones", 10),
        ]
        shared = {"jones"}
        with patch("src.core.vocabulary.name_deduplicator.is_common_word", return_value=False):
            result = _absorb_single_word_names(terms, shared_last_names=shared)
        names = _term_names(result)
        assert "Jones" in names, "Shared last name should not be absorbed"

    def test_non_shared_last_name_absorbed(self):
        """'Hiraldo' is not shared -- safe to absorb."""
        terms = [
            _person("Hiraldo", 3),
            _person("Emmanuel Hiraldo", 5),
        ]
        with patch("src.core.vocabulary.name_deduplicator.is_common_word", return_value=False):
            result = _absorb_single_word_names(terms, shared_last_names=set())
        names = _term_names(result)
        assert "Hiraldo" not in names


# ---------------------------------------------------------------------------
# Title synthesis with shared last names
# ---------------------------------------------------------------------------


class TestTitleSynthesisSharedLastName:
    """Title synthesis respects shared last names."""

    def test_mr_kept_separate_when_shared(self):
        """'Mr. Jones' stays separate when 'James Jones' + 'Patricia Jones' exist."""
        terms = [
            _person("James Jones", 10),
            _person("Patricia Jones", 8),
            _person("Mr. Jones", 3),
        ]
        shared = {"jones"}
        result = _synthesize_titled_names(terms, shared_last_names=shared)
        names = _term_names(result)
        assert "Mr. Jones" in names, "Generic title should stay when last name is shared"

    def test_dr_merges_when_unambiguous(self):
        """'Dr. Smith' merges into 'James Smith' when Smith is not shared."""
        terms = [
            _person("James Smith", 10),
            _person("Dr. Smith", 3),
        ]
        result = _synthesize_titled_names(terms, shared_last_names=set())
        names = _term_names(result)
        assert "Dr. Smith" not in names, "Role title should merge when unambiguous"
        # Should have annotated the target
        target = next(
            t
            for t in result
            if ("Smith" in t["Term"] and "Dr." not in t["Term"]) or "(Dr.)" in t["Term"]
        )
        assert target["Occurrences"] == 13  # 10 + 3

    def test_mr_merges_when_not_shared(self):
        """'Mr. Smith' merges when only one Smith exists and not shared."""
        terms = [
            _person("James Smith", 10),
            _person("Mr. Smith", 3),
        ]
        result = _synthesize_titled_names(terms, shared_last_names=set())
        names = _term_names(result)
        assert "Mr. Smith" not in names, "Generic title should merge when unambiguous"

    def test_generic_title_ambiguous_shared(self):
        """'Mr. Jones' stays separate when shared, even with multiple candidates."""
        terms = [
            _person("James Jones", 10),
            _person("Bob Jones", 8),
            _person("Mr. Jones", 3),
        ]
        shared = {"jones"}
        result = _synthesize_titled_names(terms, shared_last_names=shared)
        names = _term_names(result)
        assert "Mr. Jones" in names


# ---------------------------------------------------------------------------
# Integration: deduplicate_names end-to-end
# ---------------------------------------------------------------------------


class TestDeduplicateNamesIntegration:
    """End-to-end tests through deduplicate_names."""

    def test_shared_last_name_prevents_absorption(self):
        """Full pipeline: 'Jones' kept when James + Patricia Jones exist."""
        terms = [
            _person("James Jones", 10),
            _person("Patricia Jones", 8),
            _person("Jones", 4),
        ]
        with patch("src.core.vocabulary.name_deduplicator.is_common_word", return_value=False):
            result = deduplicate_names(terms)
        names = _term_names(result)
        assert "Jones" in names

    def test_common_word_preserved_end_to_end(self):
        """Full pipeline: 'Park' not absorbed into 'North Central Park'."""
        terms = [
            _person("Park", 2),
            _person("North Central Park", 5),
        ]
        with patch("src.core.vocabulary.name_deduplicator.is_common_word", return_value=True):
            result = deduplicate_names(terms)
        names = _term_names(result)
        assert "Park" in names
