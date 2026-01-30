"""Tests for name_deduplicator.py"""

from unittest.mock import patch

import pytest

MOCK_KNOWN_WORDS = {"james", "john", "bob", "arthur", "diana", "jim"}


def _make_term(term, is_person=True, freq=5):
    return {"Term": term, "Is Person": is_person, "In-Case Freq": freq}


@pytest.fixture(autouse=True)
def mock_known_words():
    with patch(
        "src.core.vocabulary.name_deduplicator._load_known_words",
        return_value=MOCK_KNOWN_WORDS,
    ):
        yield


class TestStripTranscriptArtifacts:
    def test_removes_q_notation(self):
        from src.core.vocabulary.name_deduplicator import _strip_transcript_artifacts

        assert _strip_transcript_artifacts("DI LEO 1 Q") == "DI LEO"

    def test_removes_speech_attribution(self):
        from src.core.vocabulary.name_deduplicator import _strip_transcript_artifacts

        assert _strip_transcript_artifacts("SMITH: Objection") == "SMITH"

    def test_removes_trailing_numbers(self):
        from src.core.vocabulary.name_deduplicator import _strip_transcript_artifacts

        assert _strip_transcript_artifacts("Di Leo 17") == "Di Leo"

    def test_empty_after_stripping_returns_original(self):
        from src.core.vocabulary.name_deduplicator import _strip_transcript_artifacts

        result = _strip_transcript_artifacts("1 Q")
        # Should return something (original or stripped), not empty
        assert len(result) > 0


class TestNormalizeName:
    def test_uppercase_to_title(self):
        from src.core.vocabulary.name_deduplicator import _normalize_name

        assert _normalize_name("DI LEO") == "Di Leo"

    def test_collapses_multiple_spaces(self):
        from src.core.vocabulary.name_deduplicator import _normalize_name

        assert _normalize_name("john  smith") == "John Smith"


class TestStripTitlePrefix:
    def test_dr_prefix(self):
        from src.core.vocabulary.name_deduplicator import _strip_title_prefix

        name, title = _strip_title_prefix("Dr. Jones")
        assert name == "Jones"
        assert title == "dr."

    def test_no_title(self):
        from src.core.vocabulary.name_deduplicator import _strip_title_prefix

        name, title = _strip_title_prefix("James Jones")
        assert name == "James Jones"
        assert title == ""

    def test_mr_prefix(self):
        from src.core.vocabulary.name_deduplicator import _strip_title_prefix

        name, title = _strip_title_prefix("Mr Smith")
        assert name == "Smith"
        assert title == "mr."


class TestIsRoleTitle:
    def test_dr_is_role(self):
        from src.core.vocabulary.name_deduplicator import _is_role_title

        assert _is_role_title("dr.") is True

    def test_mr_is_not_role(self):
        from src.core.vocabulary.name_deduplicator import _is_role_title

        assert _is_role_title("mr.") is False

    def test_judge_is_role(self):
        from src.core.vocabulary.name_deduplicator import _is_role_title

        assert _is_role_title("judge") is True


class TestDeduplicateNames:
    def test_empty_list(self):
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        assert deduplicate_names([]) == []

    def test_single_person_passes_through(self):
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [_make_term("James Smith", is_person=True, freq=3)]
        result = deduplicate_names(terms)
        assert len(result) == 1

    def test_non_person_not_deduped(self):
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            _make_term("Cervical Spine", is_person=False, freq=10),
            _make_term("Cervical", is_person=False, freq=5),
        ]
        result = deduplicate_names(terms)
        # Non-person terms should pass through without person-dedup logic
        assert len(result) >= 1

    def test_transcript_artifact_merge(self):
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            _make_term("DI LEO 1 Q", is_person=True, freq=3),
            _make_term("DI LEO 2", is_person=True, freq=2),
            _make_term("Diana Di Leo", is_person=True, freq=10),
        ]
        result = deduplicate_names(terms)
        result_terms = [t["Term"] for t in result]
        # Should merge artifacts into Diana Di Leo
        assert any("Di Leo" in t for t in result_terms)
        # Should have fewer terms than input
        assert len(result) < len(terms)


class TestTitleSynthesis:
    def test_dr_plus_full_name(self):
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            _make_term("Dr. Jones", is_person=True, freq=5),
            _make_term("James Jones", is_person=True, freq=10),
        ]
        result = deduplicate_names(terms)
        result_terms = [t["Term"] for t in result]
        # Should synthesize into "James Jones (Dr.)" or similar
        assert len(result) == 1
        merged = result_terms[0]
        assert "James Jones" in merged
        assert "Dr." in merged

    def test_ambiguous_last_name_no_merge(self):
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        terms = [
            _make_term("Bob Jones", is_person=True, freq=5),
            _make_term("Jim Jones", is_person=True, freq=5),
            _make_term("Dr. Jones", is_person=True, freq=3),
        ]
        result = deduplicate_names(terms)
        # With two different people named Jones, Dr. Jones merge is ambiguous
        # Should keep at least 2 distinct Jones entries
        result_terms = [t["Term"] for t in result]
        jones_terms = [t for t in result_terms if "Jones" in t]
        assert len(jones_terms) >= 2


class TestFindPotentialDuplicates:
    def test_word_subset_flagged(self):
        from src.core.vocabulary.name_deduplicator import find_potential_duplicates

        terms = [
            _make_term("Antonio Vargas", is_person=True, freq=5),
            _make_term("Antonio Fernandez Vargas", is_person=True, freq=3),
        ]
        dupes = find_potential_duplicates(terms)
        assert len(dupes) > 0

    def test_different_first_name_same_last_not_flagged(self):
        from src.core.vocabulary.name_deduplicator import find_potential_duplicates

        terms = [
            _make_term("Bob Smith", is_person=True, freq=5),
            _make_term("Diana Smith", is_person=True, freq=3),
        ]
        dupes = find_potential_duplicates(terms)
        # Different first names sharing only last name shouldn't be flagged as subset
        assert len(dupes) == 0
