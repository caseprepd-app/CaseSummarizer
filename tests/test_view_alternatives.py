"""
Tests for the "View Alternatives" feature.

Tests cover:
1. alternative_reasons.py — reason-building helper functions
2. canonical_scorer.py — _selection_branch and _scored_variants metadata
3. name_deduplicator.py — _alternatives attachment at all 4 selection points
4. name_regularizer.py — _alternatives for fragment and typo removals
"""

import pytest

from src.core.vocabulary.alternative_reasons import (
    build_alternatives_from_legacy,
    build_alternatives_from_scorer,
    build_fragment_alternative,
    build_single_word_alternative,
    build_titled_alternative,
    build_typo_alternative,
)
from src.core.vocabulary.canonical_scorer import CanonicalScorer
from src.core.vocabulary.term_sources import TermSources

# =========================================================================
# alternative_reasons.py — Unit Tests
# =========================================================================


class TestBuildAlternativesFromScorer:
    """Test build_alternatives_from_scorer for all three branches."""

    def test_single_known_branch(self):
        """Single-known branch: winner is 'only variant in dictionary'."""
        variants = [
            {"Term": "Jenkins", "Occurrences": 40},
            {"Term": "Jenidns", "Occurrences": 3},
        ]
        alts, reason = build_alternatives_from_scorer(
            variants, "Jenkins", "single_known", scored_list=None
        )
        assert reason == "Only variant found in names dictionary"
        assert len(alts) == 1
        assert alts[0]["variant"] == "Jenidns"
        assert alts[0]["frequency"] == 3
        assert "Not found in names dictionary" in alts[0]["reason"]

    def test_none_known_branch_with_scores(self):
        """None-known branch: winner has highest confidence-weighted score."""
        variants = [
            {"Term": "Djamel", "Occurrences": 10},
            {"Term": "Djamle", "Occurrences": 2},
        ]
        scored_list = [("Djamel", 45.2), ("Djamle", 12.4)]
        alts, reason = build_alternatives_from_scorer(variants, "Djamel", "none_known", scored_list)
        assert "45.2" in reason
        assert len(alts) == 1
        assert alts[0]["variant"] == "Djamle"
        assert "12.4" in alts[0]["reason"]

    def test_multiple_known_branch(self):
        """Multiple-known branch: winner among known variants."""
        variants = [
            {"Term": "Smith", "Occurrences": 30},
            {"Term": "Smyth", "Occurrences": 5},
        ]
        scored_list = [("Smith", 60.0), ("Smyth", 10.0)]
        alts, reason = build_alternatives_from_scorer(
            variants, "Smith", "multiple_known", scored_list
        )
        assert "among known variants" in reason
        assert len(alts) == 1
        assert alts[0]["variant"] == "Smyth"

    def test_canonical_excluded_from_alternatives(self):
        """The canonical term itself should not appear in the alternatives list."""
        variants = [
            {"Term": "Jones", "Occurrences": 50},
            {"Term": "Jomes", "Occurrences": 2},
        ]
        alts, _ = build_alternatives_from_scorer(variants, "Jones", "single_known")
        variant_names = [a["variant"] for a in alts]
        assert "Jones" not in variant_names

    def test_case_insensitive_canonical_matching(self):
        """Canonical matching should be case-insensitive."""
        variants = [
            {"Term": "JENKINS", "Occurrences": 10},
            {"Term": "jenkins", "Occurrences": 5},
            {"Term": "Jenidns", "Occurrences": 1},
        ]
        # "jenkins" matches canonical "JENKINS" case-insensitively
        alts, _ = build_alternatives_from_scorer(variants, "JENKINS", "single_known")
        variant_names = [a["variant"] for a in alts]
        assert "jenkins" not in variant_names
        assert "Jenidns" in variant_names

    def test_empty_variants(self):
        """Empty variants list should return empty alternatives."""
        alts, reason = build_alternatives_from_scorer([], "Nothing", "single_known")
        assert alts == []
        assert reason == "Only variant found in names dictionary"

    def test_all_caps_rejection_reason(self):
        """ALL CAPS variant should include formatting penalty in reason."""
        variants = [
            {"Term": "Jenkins", "Occurrences": 40},
            {"Term": "JENIDNS", "Occurrences": 3},
        ]
        alts, _ = build_alternatives_from_scorer(variants, "Jenkins", "single_known")
        assert "ALL CAPS formatting penalty" in alts[0]["reason"]

    def test_unknown_branch_fallback(self):
        """Unknown branch string should produce generic reason."""
        variants = [
            {"Term": "A", "Occurrences": 1},
            {"Term": "B", "Occurrences": 1},
        ]
        _, reason = build_alternatives_from_scorer(variants, "A", "unexpected_branch")
        assert reason == "Selected as canonical variant"


class TestBuildAlternativesFromLegacy:
    """Test build_alternatives_from_legacy for the heuristic path."""

    def test_basic_legacy_alternatives(self):
        """Legacy path should list losers with heuristic score reason."""
        sorted_group = [
            {
                "original": {"Term": "Arthur Jenkins", "Occurrences": 45},
                "cleaned": "Arthur Jenkins",
            },
            {
                "original": {"Term": "ARTHUR JENKINS", "Occurrences": 12},
                "cleaned": "ARTHUR JENKINS",
            },
        ]
        alts, reason = build_alternatives_from_legacy(sorted_group, "Occurrences")
        assert "heuristic" in reason.lower()
        assert len(alts) == 1
        assert alts[0]["variant"] == "ARTHUR JENKINS"
        assert alts[0]["frequency"] == 12
        assert "ALL CAPS" in alts[0]["reason"]

    def test_empty_group(self):
        """Empty group should return empty list."""
        alts, reason = build_alternatives_from_legacy([], "Occurrences")
        assert alts == []
        assert reason == "No variants"

    def test_single_entry_group(self):
        """Single entry has no losers, so no alternatives."""
        sorted_group = [
            {"original": {"Term": "Jenkins", "Occurrences": 50}, "cleaned": "Jenkins"},
        ]
        alts, _ = build_alternatives_from_legacy(sorted_group, "Occurrences")
        assert alts == []


class TestBuildHelperFunctions:
    """Test individual helper builder functions."""

    def test_build_single_word_alternative(self):
        """Single-word alternative should reference the target name."""
        entry = {"Term": "Hiraldo", "Occurrences": 5}
        alt = build_single_word_alternative(entry, "Emmanuel Hiraldo")
        assert alt["variant"] == "Hiraldo"
        assert alt["frequency"] == 5
        assert "Single-word fragment" in alt["reason"]
        assert "Emmanuel Hiraldo" in alt["reason"]

    def test_build_titled_alternative(self):
        """Titled alternative should describe the merge."""
        entry = {"Term": "Dr. Jones", "Occurrences": 8}
        alt = build_titled_alternative(entry, "dr.")
        assert alt["variant"] == "Dr. Jones"
        assert alt["frequency"] == 8
        assert "Title-prefixed" in alt["reason"]

    def test_build_fragment_alternative(self):
        """Fragment alternative should reference the canonical term."""
        alt = build_fragment_alternative("Di", 3, "Di Leo")
        assert alt["variant"] == "Di"
        assert alt["frequency"] == 3
        assert "Name fragment" in alt["reason"]
        assert "Di Leo" in alt["reason"]

    def test_build_typo_alternative_single_known(self):
        """Typo alternative with single_known branch."""
        alt = build_typo_alternative("Jenidns", 2, "single_known")
        assert alt["variant"] == "Jenidns"
        assert alt["frequency"] == 2
        assert "Not found in names dictionary" in alt["reason"]

    def test_build_typo_alternative_with_scores(self):
        """Typo alternative with score data should include score."""
        scored = [("Jenkins", 45.0), ("Jenidns", 5.0)]
        alt = build_typo_alternative("Jenidns", 2, "none_known", scored, 45.0)
        assert "5.0" in alt["reason"]

    def test_build_single_word_missing_freq(self):
        """Missing frequency should default to 0."""
        entry = {"Term": "Hiraldo"}
        alt = build_single_word_alternative(entry, "Emmanuel Hiraldo")
        assert alt["frequency"] == 0

    def test_build_titled_missing_freq(self):
        """Missing frequency should default to 0."""
        entry = {"Term": "Dr. Jones"}
        alt = build_titled_alternative(entry, "dr.")
        assert alt["frequency"] == 0


# =========================================================================
# canonical_scorer.py — Metadata Tests
# =========================================================================


class TestScorerMetadata:
    """Test that CanonicalScorer attaches _selection_branch and _scored_variants."""

    @pytest.fixture
    def scorer(self):
        """Create scorer with a small test dictionary."""
        known_words = {"john", "smith", "jones", "jenkins", "mary", "williams"}
        return CanonicalScorer(known_words)

    def _make_variant(self, term, freq=10, confidence=0.9):
        """Helper to create a variant dict with TermSources."""
        return {
            "Term": term,
            "Occurrences": freq,
            "sources": TermSources.create_legacy(freq, confidence),
        }

    def test_single_known_branch_metadata(self, scorer):
        """Single known variant should set _selection_branch = 'single_known'."""
        variants = [
            self._make_variant("Jenkins", 40),
            self._make_variant("Jenidns", 3, 0.5),
        ]
        result = scorer.select_canonical(variants)
        assert result["_selection_branch"] == "single_known"
        # single_known branch doesn't go through _select_by_score
        assert "_scored_variants" not in result

    def test_none_known_branch_metadata(self, scorer):
        """No known variants should set _selection_branch = 'none_known'."""
        variants = [
            self._make_variant("Djamel", 10),
            self._make_variant("Djamle", 2, 0.5),
        ]
        result = scorer.select_canonical(variants)
        assert result["_selection_branch"] == "none_known"
        assert "_scored_variants" in result
        assert len(result["_scored_variants"]) == 2
        # First scored variant should be the winner (highest score)
        assert result["_scored_variants"][0][0] == result["Term"]

    def test_multiple_known_branch_metadata(self, scorer):
        """Multiple known variants should set _selection_branch = 'multiple_known'."""
        variants = [
            self._make_variant("Smith", 30),
            self._make_variant("Jones", 5),
        ]
        result = scorer.select_canonical(variants)
        assert result["_selection_branch"] == "multiple_known"
        assert "_scored_variants" in result

    def test_scored_variants_are_tuples(self, scorer):
        """_scored_variants should be list of (term, score) tuples."""
        variants = [
            self._make_variant("Djamel", 10),
            self._make_variant("Djamle", 2, 0.5),
        ]
        result = scorer.select_canonical(variants)
        for item in result["_scored_variants"]:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_single_variant_no_metadata(self, scorer):
        """Single variant (no competition) shouldn't have branch metadata."""
        variants = [self._make_variant("Jenkins", 40)]
        result = scorer.select_canonical(variants)
        # Single variant returns immediately, no branch needed
        assert "_selection_branch" not in result


# =========================================================================
# name_deduplicator.py — Integration Tests
# =========================================================================


class TestDeduplicatorAlternatives:
    """Test that deduplicate_names attaches _alternatives at each phase."""

    def _make_person(self, term, freq=10, confidence=0.9):
        """Helper to create a Person term dict with TermSources."""
        return {
            "Term": term,
            "Is Person": "Yes",
            "Occurrences": freq,
            "sources": TermSources.create_legacy(freq, confidence),
        }

    def test_scorer_path_attaches_alternatives(self):
        """OCR variant merge via scorer should attach _alternatives."""
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            self._make_person("Arthur Jenkins", 45, 0.95),
            self._make_person("Anhur Jenkins", 3, 0.60),
        ]
        result = deduplicate_names(terms)

        # Should merge into one entry
        assert len(result) == 1
        canonical = result[0]
        assert "_alternatives" in canonical
        assert "_canonical_reason" in canonical
        assert len(canonical["_alternatives"]) >= 1

        # The rejected variant should be listed
        alt_variants = [a["variant"] for a in canonical["_alternatives"]]
        # One of "Anhur Jenkins" or the normalized form should appear
        assert any("Anhur" in v or "nhur" in v for v in alt_variants)

    def test_single_word_absorption_attaches_alternatives(self):
        """Single-word name absorbed into full name should appear in _alternatives."""
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            self._make_person("Emmanuel Hiraldo", 20, 0.95),
            self._make_person("Hiraldo", 5, 0.90),
        ]
        result = deduplicate_names(terms)

        # "Hiraldo" should be absorbed into "Emmanuel Hiraldo"
        full_name = [t for t in result if "Emmanuel" in t.get("Term", "")]
        assert len(full_name) == 1
        canonical = full_name[0]

        assert "_alternatives" in canonical
        alt_variants = [a["variant"] for a in canonical["_alternatives"]]
        assert "Hiraldo" in alt_variants
        alt = next(a for a in canonical["_alternatives"] if a["variant"] == "Hiraldo")
        assert "Single-word fragment" in alt["reason"]

    def test_titled_name_merge_attaches_alternatives(self):
        """Title-prefixed name merged into full name should appear in _alternatives."""
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            self._make_person("James Jones", 30, 0.95),
            self._make_person("Dr. Jones", 8, 0.90),
        ]
        result = deduplicate_names(terms)

        # Find the surviving full name entry (may have "(Dr.)" appended)
        full_name = [t for t in result if "James" in t.get("Term", "")]
        assert len(full_name) == 1
        canonical = full_name[0]

        assert "_alternatives" in canonical
        alt_variants = [a["variant"] for a in canonical["_alternatives"]]
        assert "Dr. Jones" in alt_variants
        alt = next(a for a in canonical["_alternatives"] if a["variant"] == "Dr. Jones")
        assert "Title-prefixed" in alt["reason"]

    def test_no_alternatives_for_unique_names(self):
        """Names with no variants should not have _alternatives key."""
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            self._make_person("Alice Johnson", 20, 0.95),
            self._make_person("Bob Williams", 15, 0.90),
        ]
        result = deduplicate_names(terms)

        # Both should survive, neither should have alternatives
        assert len(result) == 2
        for entry in result:
            alts = entry.get("_alternatives", [])
            # Unique names shouldn't accumulate alternatives
            assert len(alts) == 0

    def test_non_person_terms_unchanged(self):
        """Non-person terms should not have _alternatives."""
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            {"Term": "radiculopathy", "Is Person": "No", "Occurrences": 10},
            self._make_person("John Smith", 20, 0.95),
        ]
        result = deduplicate_names(terms)

        non_person = [t for t in result if t.get("Term") == "radiculopathy"]
        assert len(non_person) == 1
        assert "_alternatives" not in non_person[0]

    def test_alternatives_accumulate_across_phases(self):
        """A name that gets variants from multiple phases should accumulate them."""
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            self._make_person("Emmanuel Hiraldo", 20, 0.95),
            self._make_person("Hiraldo", 5, 0.90),  # single-word → absorbed
            self._make_person("Dr. Hiraldo", 3, 0.85),  # titled → merged
        ]
        result = deduplicate_names(terms)

        full_name = [t for t in result if "Emmanuel" in t.get("Term", "")]
        assert len(full_name) == 1
        canonical = full_name[0]

        # Should have alternatives from both absorption and title merge
        assert "_alternatives" in canonical
        assert len(canonical["_alternatives"]) >= 2

    def test_legacy_path_attaches_alternatives(self):
        """Legacy path (no TermSources) should attach _alternatives."""
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        # Terms without 'sources' key trigger legacy path
        terms = [
            {"Term": "Arthur Jenkins", "Is Person": "Yes", "Occurrences": 45},
            {"Term": "Anhur Jenkins", "Is Person": "Yes", "Occurrences": 3},
        ]
        result = deduplicate_names(terms)
        assert len(result) == 1
        canonical = result[0]
        assert "_alternatives" in canonical
        assert "_canonical_reason" in canonical
        assert "heuristic" in canonical["_canonical_reason"].lower()


# =========================================================================
# name_regularizer.py — Integration Tests
# =========================================================================


class TestRegularizerAlternatives:
    """Test that regularize_names attaches _alternatives for fragments and typos."""

    def test_fragment_removal_attaches_alternatives(self):
        """Fragment removed by regularizer should appear as alternative on canonical."""
        from src.core.vocabulary.name_regularizer import regularize_names

        # "Di Leo" high-frequency, "Di" low-frequency fragment
        vocab = [
            {"Term": "Di Leo", "Occurrences": 50},
            {"Term": "Di", "Occurrences": 3},
            # Need enough terms for the filter to activate (min 4)
            {"Term": "Smith Johnson", "Occurrences": 40},
            {"Term": "Williams Brown", "Occurrences": 35},
            {"Term": "Mary Davis", "Occurrences": 30},
        ]
        result = regularize_names(vocab)

        # "Di" should be removed
        result_terms = [t.get("Term") for t in result]
        assert "Di" not in result_terms

        # "Di Leo" should have "Di" as an alternative
        di_leo = [t for t in result if t.get("Term") == "Di Leo"]
        assert len(di_leo) == 1
        alts = di_leo[0].get("_alternatives", [])
        alt_variants = [a["variant"] for a in alts]
        assert "Di" in alt_variants
        alt = next(a for a in alts if a["variant"] == "Di")
        assert "fragment" in alt["reason"].lower()

    def test_typo_removal_attaches_alternatives(self):
        """Typo removed by regularizer should appear as alternative on canonical."""
        from src.core.vocabulary.name_regularizer import regularize_names

        # "Jenkins" is a real word, "Jenidns" is a typo (1 edit distance)
        vocab = [
            {"Term": "Jenkins", "Occurrences": 40},
            {"Term": "Jenidns", "Occurrences": 2},
            # Need enough terms for the filter to activate
            {"Term": "Smith Johnson", "Occurrences": 35},
            {"Term": "Williams Brown", "Occurrences": 30},
            {"Term": "Mary Davis", "Occurrences": 25},
        ]
        result = regularize_names(vocab)

        # "Jenidns" should be removed
        result_terms = [t.get("Term") for t in result]
        assert "Jenidns" not in result_terms

        # "Jenkins" should have "Jenidns" as an alternative
        jenkins = [t for t in result if t.get("Term") == "Jenkins"]
        assert len(jenkins) == 1
        alts = jenkins[0].get("_alternatives", [])
        alt_variants = [a["variant"] for a in alts]
        assert "Jenidns" in alt_variants

    def test_no_alternatives_for_unique_terms(self):
        """Terms with no typos or fragments should not have _alternatives."""
        from src.core.vocabulary.name_regularizer import regularize_names

        vocab = [
            {"Term": "Completely Unique", "Occurrences": 40},
            {"Term": "Another Different", "Occurrences": 35},
            {"Term": "Third Distinct", "Occurrences": 30},
            {"Term": "Fourth Separate", "Occurrences": 25},
        ]
        result = regularize_names(vocab)

        for entry in result:
            alts = entry.get("_alternatives", [])
            assert len(alts) == 0


# =========================================================================
# End-to-end data flow test
# =========================================================================


class TestEndToEndAlternativesFlow:
    """Test the full data flow from dedup through to dict keys."""

    def test_alternatives_are_plain_dicts(self):
        """All _alternatives entries should be plain dicts with string/int values."""
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            {
                "Term": "Arthur Jenkins",
                "Is Person": "Yes",
                "Occurrences": 45,
                "sources": TermSources.create_legacy(45, 0.95),
            },
            {
                "Term": "Anhur Jenkins",
                "Is Person": "Yes",
                "Occurrences": 3,
                "sources": TermSources.create_legacy(3, 0.60),
            },
        ]
        result = deduplicate_names(terms)
        canonical = result[0]

        for alt in canonical.get("_alternatives", []):
            assert isinstance(alt, dict)
            assert "variant" in alt
            assert "reason" in alt
            assert "frequency" in alt
            assert isinstance(alt["variant"], str)
            assert isinstance(alt["reason"], str)
            assert isinstance(alt["frequency"], (int, float))

    def test_canonical_reason_is_string(self):
        """_canonical_reason should always be a non-empty string."""
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            {
                "Term": "Arthur Jenkins",
                "Is Person": "Yes",
                "Occurrences": 45,
                "sources": TermSources.create_legacy(45, 0.95),
            },
            {
                "Term": "Anhur Jenkins",
                "Is Person": "Yes",
                "Occurrences": 3,
                "sources": TermSources.create_legacy(3, 0.60),
            },
        ]
        result = deduplicate_names(terms)
        canonical = result[0]
        reason = canonical.get("_canonical_reason", "")
        assert isinstance(reason, str)
        assert len(reason) > 0
