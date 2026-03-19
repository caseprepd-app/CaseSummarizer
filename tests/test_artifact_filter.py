"""Tests for artifact_filter.py"""

from unittest.mock import patch


def _make_term(term, is_person=True, freq=5):
    return {"Term": term, "Is Person": is_person, "Occurrences": freq}


class TestFilterSubstringArtifacts:
    @patch("src.core.vocabulary.artifact_filter.is_common_word", return_value=False)
    def test_removes_canonical_substring_with_punctuation(self, mock_common):
        from src.core.vocabulary.artifact_filter import filter_substring_artifacts

        vocab = [
            _make_term("Ms. Di Leo:", is_person=True, freq=2),
            _make_term("Ms. Di Leo", is_person=True, freq=10),
        ]
        canonical_count = 10
        result = filter_substring_artifacts(vocab, canonical_count, "Occurrences", "Term")
        result_terms = [t["Term"] for t in result]
        assert "Ms. Di Leo" in result_terms
        assert "Ms. Di Leo:" not in result_terms

    @patch("src.core.vocabulary.artifact_filter.is_common_word", return_value=False)
    def test_exact_match_canonical_not_removed(self, mock_common):
        from src.core.vocabulary.artifact_filter import filter_substring_artifacts

        vocab = [
            _make_term("John Smith", is_person=True, freq=10),
        ]
        result = filter_substring_artifacts(vocab, 10, "Occurrences", "Term")
        assert len(result) == 1
        assert result[0]["Term"] == "John Smith"

    @patch("src.core.vocabulary.artifact_filter.is_common_word", return_value=False)
    def test_non_person_not_affected_by_person_filters(self, mock_common):
        from src.core.vocabulary.artifact_filter import filter_substring_artifacts

        vocab = [
            _make_term("Cervical Spine", is_person=False, freq=10),
            _make_term("Cervical", is_person=False, freq=5),
        ]
        result = filter_substring_artifacts(vocab, 10, "Occurrences", "Term")
        # Non-person terms have different filtering rules
        assert len(result) >= 1


class TestIsCommonWordVariant:
    def test_canonical_plus_common_word(self):
        """'patient' is a real common word, so canonical + 'Patient' is a variant."""
        from src.core.vocabulary.artifact_filter import _is_common_word_variant

        result = _is_common_word_variant("Luigi Napolitano Patient", "Luigi Napolitano")
        assert result is True

    def test_same_length_not_variant(self):
        """Identical terms cannot be a variant of each other."""
        from src.core.vocabulary.artifact_filter import _is_common_word_variant

        result = _is_common_word_variant("Luigi Napolitano", "Luigi Napolitano")
        assert result is False


class TestRemoveComponentNames:
    def test_single_word_person_removed_when_part_of_multiword(self):
        from src.core.vocabulary.artifact_filter import _remove_component_names

        vocab = [
            _make_term("Arthur", is_person=True, freq=3),
            _make_term("Arthur Jenkins", is_person=True, freq=8),
        ]
        result = _remove_component_names(vocab, "Term")
        result_terms = [t["Term"] for t in result]
        assert "Arthur Jenkins" in result_terms
        assert "Arthur" not in result_terms

    def test_single_word_non_person_not_removed(self):
        from src.core.vocabulary.artifact_filter import _remove_component_names

        vocab = [
            _make_term("Arthur", is_person=False, freq=3),
            _make_term("Arthur Jenkins", is_person=True, freq=8),
        ]
        result = _remove_component_names(vocab, "Term")
        result_terms = [t["Term"] for t in result]
        # Non-person "Arthur" should not be removed by person-component logic
        assert "Arthur" in result_terms


class TestRemoveHeaderArtifacts:
    def test_multi_word_person_treated_as_canonical(self):
        """Multi-word Person entries are treated as canonical and not self-removed.
        Header artifact removal only works when combined with upstream deduplication
        that would have already cleaned these entries."""
        from src.core.vocabulary.artifact_filter import _remove_header_artifacts

        # "Jones Cross" is itself a multi-word Person entry, so it's added to
        # canonical_names and skipped (not removed).
        vocab = [
            _make_term("Jones Cross", is_person=True, freq=2),
            _make_term("Bob Jones", is_person=True, freq=10),
        ]
        result = _remove_header_artifacts(vocab, "Term")
        result_terms = [t["Term"] for t in result]
        # Both are multi-word Person entries, both treated as canonical
        assert "Jones Cross" in result_terms
        assert "Bob Jones" in result_terms

    def test_canonical_name_not_removed(self):
        from src.core.vocabulary.artifact_filter import _remove_header_artifacts

        vocab = [
            _make_term("John Smith", is_person=True, freq=10),
        ]
        result = _remove_header_artifacts(vocab, "Term")
        assert len(result) == 1
        assert result[0]["Term"] == "John Smith"

    def test_non_person_not_affected(self):
        from src.core.vocabulary.artifact_filter import _remove_header_artifacts

        vocab = [
            _make_term("Smith Direct", is_person=False, freq=2),
            _make_term("John Smith", is_person=True, freq=10),
        ]
        result = _remove_header_artifacts(vocab, "Term")
        result_terms = [t["Term"] for t in result]
        # Non-person entries are never removed by this filter
        assert "Smith Direct" in result_terms
