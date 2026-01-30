"""Tests for transcript_header filter."""

from src.core.vocabulary.filters.transcript_header import (
    TranscriptHeaderFilter,
    _build_canonical_names,
    _contains_section_keyword,
    _has_name_plus_keyword,
    _has_two_canonical_names,
    is_transcript_header_artifact,
)


def _make_term(term, is_person=False, freq=5):
    return {"Term": term, "Is Person": is_person, "In-Case Freq": freq}


class TestContainsSectionKeyword:
    def test_direct(self):
        assert _contains_section_keyword("direct") is True

    def test_cross_examination(self):
        assert _contains_section_keyword("cross examination") is True

    def test_redirect(self):
        assert _contains_section_keyword("redirect") is True

    def test_normal_term(self):
        assert _contains_section_keyword("cervical spine") is False

    def test_partial_match_not_triggered(self):
        # "direction" contains "direct" but not as a whole word
        assert _contains_section_keyword("direction") is False


class TestBuildCanonicalNames:
    def test_extracts_multiword_persons(self):
        vocab = [
            _make_term("John Smith", is_person=True),
            _make_term("cervical spine", is_person=False),
            _make_term("Arthur", is_person=True),  # single word, excluded
        ]
        names = _build_canonical_names(vocab)
        assert "john smith" in names
        assert "cervical spine" not in names
        assert "arthur" not in names


class TestHasNamePlusKeyword:
    def test_name_dash_keyword(self):
        names = {"john smith"}
        assert _has_name_plus_keyword("john smith - direct", names) is True

    def test_name_keyword_no_separator(self):
        names = {"jones"}  # won't match — single word not in canonical
        names = {"bob jones"}
        assert _has_name_plus_keyword("bob jones cross", names) is True

    def test_no_keyword(self):
        names = {"john smith"}
        assert _has_name_plus_keyword("john smith attorney", names) is False

    def test_keyword_without_name(self):
        names = {"john smith"}
        assert _has_name_plus_keyword("direct examination", names) is False


class TestHasTwoCanonicalNames:
    def test_two_distinct_names(self):
        names = {"john smith", "bob jones"}
        assert _has_two_canonical_names("john smith bob jones", names) is True

    def test_single_name(self):
        names = {"john smith", "bob jones"}
        assert _has_two_canonical_names("john smith", names) is False

    def test_overlapping_names(self):
        # "Smith Jones" and "Smith Brown" share "Smith" — if both match, they overlap
        names = {"john smith", "john doe"}
        # "john smith john doe" has both, and they share "john"
        # but smith and doe are distinct, so words_i & words_j = {"john"} != empty
        assert _has_two_canonical_names("john smith john doe", names) is False


class TestIsTranscriptHeaderArtifact:
    def test_section_keyword_alone(self):
        assert is_transcript_header_artifact("Direct", set()) is True

    def test_name_plus_keyword(self):
        names = {"bob jones"}
        assert is_transcript_header_artifact("Bob Jones - Direct", names) is True

    def test_normal_term(self):
        assert is_transcript_header_artifact("cervical spine", set()) is False

    def test_person_name_alone(self):
        names = {"bob jones"}
        assert is_transcript_header_artifact("Bob Jones", names) is False


class TestTranscriptHeaderFilter:
    def test_removes_header_artifacts(self):
        vocab = [
            _make_term("Bob Jones", is_person=True, freq=10),
            _make_term("Jane Doe", is_person=True, freq=8),
            _make_term("Jones Direct", is_person=False, freq=2),
            _make_term("cervical spine", is_person=False, freq=5),
            _make_term("Cross Examination", is_person=False, freq=3),
        ]
        f = TranscriptHeaderFilter()
        result = f.filter(vocab)
        result_terms = [t["Term"] for t in result.vocabulary]
        assert "Bob Jones" in result_terms
        assert "Jane Doe" in result_terms
        assert "cervical spine" in result_terms
        assert "Jones Direct" not in result_terms
        assert "Cross Examination" not in result_terms
        assert result.removed_count == 2

    def test_empty_vocabulary(self):
        f = TranscriptHeaderFilter()
        result = f.filter([])
        assert result.vocabulary == []
        assert result.removed_count == 0
