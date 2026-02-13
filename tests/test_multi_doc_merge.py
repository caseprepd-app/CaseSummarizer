"""
Tests for per-document parallel extraction and merge logic (Session 131).

Tests cover:
- _merge_term_across_docs: Boolean OR, frequency sum, casing votes,
  TermSources construction, quality score, algorithm union
- _merge_multi_doc_results: Indexing, dedup, ML re-boost, sorting
- extract_documents: Single-doc passthrough, multi-doc dispatch, progress callback

These tests use mock vocab dicts (the output format of extract/extract_with_llm)
to test merge logic without running spaCy or real extraction.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.vocabulary import VocabularyExtractor  # noqa: E402
from src.core.vocabulary.term_sources import TermSources  # noqa: E402


def _make_term_dict(
    term="Test Term",
    is_person="No",
    found_by="NER",
    quality_score=70.0,
    occurrences=5,
    rarity_rank=0,
    ner="Yes",
    rake="No",
    bm25="No",
    textrank="No",
    role="Vocabulary term",
    textrank_score=0.0,
):
    """Build a minimal vocab term dict matching extract() output schema."""
    algo_count = sum(1 for flag in [ner, rake, bm25, textrank] if flag == "Yes")
    sources_str = ", ".join(
        name
        for name, flag in [("NER", ner), ("RAKE", rake), ("BM25", bm25), ("TextRank", textrank)]
        if flag == "Yes"
    )
    return {
        "Term": term,
        "Is Person": is_person,
        "Found By": sources_str,
        "Role/Relevance": role,
        "Quality Score": quality_score,
        "Occurrences": occurrences,
        "Google Rarity Rank": rarity_rank,
        # "Definition" removed: definitions no longer generated
        "Sources": sources_str,
        "NER": ner,
        "RAKE": rake,
        "BM25": bm25,
        "TextRank": textrank,
        "Algo Count": algo_count,
        "# Docs": 1,
        "OCR Confidence": "95%",
        "sources": TermSources.create_legacy(occurrences),
        "total_docs_in_session": 1,
        "base_quality_score": quality_score,
        "occurrences": occurrences,
        "rarity_rank": rarity_rank,
        "algorithms": sources_str,
        "is_person": 1 if is_person == "Yes" else 0,
        "total_unique_terms": 10,
        "source_doc_confidence": 95.0,
        "textrank_score": textrank_score,
    }


@pytest.fixture
def extractor():
    """Create VocabularyExtractor with no external files."""
    return VocabularyExtractor(
        exclude_list_path=None,
        medical_terms_path=None,
    )


# =========================================================================
# _merge_term_across_docs
# =========================================================================


class TestMergeTermAcrossDocs:
    """Tests for merging a single term from multiple documents."""

    def test_num_docs_tracks_document_count(self, extractor):
        """Term in 2 docs -> # Docs = 2."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="radiculopathy", occurrences=3)),
            ("doc2", 90.0, _make_term_dict(term="radiculopathy", occurrences=2)),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        assert merged["# Docs"] == 2

    def test_single_doc_term_has_one_doc(self, extractor):
        """Term in 1 doc -> # Docs = 1."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="radiculopathy", occurrences=3)),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        assert merged["# Docs"] == 1

    def test_frequency_is_summed(self, extractor):
        """Occurrences should be sum across all docs."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="Herniation", occurrences=10)),
            ("doc2", 90.0, _make_term_dict(term="Herniation", occurrences=7)),
            ("doc3", 85.0, _make_term_dict(term="herniation", occurrences=3)),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=3, total_unique=5)

        assert merged["Occurrences"] == 20

    def test_is_person_true_if_any_doc(self, extractor):
        """Is Person = 'Yes' if ANY doc detected person."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="John Smith", is_person="No")),
            ("doc2", 90.0, _make_term_dict(term="John Smith", is_person="Yes")),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        assert merged["Is Person"] == "Yes"

    def test_is_person_no_if_none(self, extractor):
        """Is Person = 'No' if no doc detected person."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="radiculopathy", is_person="No")),
            ("doc2", 90.0, _make_term_dict(term="radiculopathy", is_person="No")),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        assert merged["Is Person"] == "No"

    def test_casing_votes_picks_most_common(self, extractor):
        """Term display uses the most common casing across docs."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="John Smith")),
            ("doc2", 90.0, _make_term_dict(term="JOHN SMITH")),
            ("doc3", 85.0, _make_term_dict(term="John Smith")),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=3, total_unique=5)

        assert merged["Term"] == "John Smith"

    def test_found_by_is_union(self, extractor):
        """Found By should be union of algorithms across docs."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="radiculopathy", ner="Yes", rake="No")),
            ("doc2", 90.0, _make_term_dict(term="radiculopathy", ner="No", rake="Yes")),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        found_by = merged["Found By"]
        assert "NER" in found_by
        assert "RAKE" in found_by

    def test_algorithm_flags_union(self, extractor):
        """Each algorithm flag should be 'Yes' if ANY doc had it."""
        doc_entries = [
            (
                "doc1",
                95.0,
                _make_term_dict(term="test", ner="Yes", rake="No", bm25="No", textrank="No"),
            ),
            (
                "doc2",
                90.0,
                _make_term_dict(term="test", ner="No", rake="Yes", bm25="No", textrank="Yes"),
            ),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        assert merged["NER"] == "Yes"
        assert merged["RAKE"] == "Yes"
        assert merged["BM25"] == "No"
        assert merged["TextRank"] == "Yes"
        assert merged["Algo Count"] == 3  # NER + RAKE + TextRank

    def test_role_picks_longest_non_default(self, extractor):
        """Role should pick the longest non-default role across docs."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="Dr. Jones", role="Vocabulary term")),
            ("doc2", 90.0, _make_term_dict(term="Dr. Jones", role="Medical professional")),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        assert merged["Role/Relevance"] == "Medical professional"

    def test_role_defaults_when_all_default(self, extractor):
        """Role falls back to 'Vocabulary term' if all docs have default."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="test", role="Vocabulary term")),
            ("doc2", 90.0, _make_term_dict(term="test", role="Vocabulary term")),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        assert merged["Role/Relevance"] == "Vocabulary term"

    # test_definition_picks_first_non_dash removed: definitions no longer generated

    def test_term_sources_object_created(self, extractor):
        """TermSources object should reflect actual per-doc data."""
        doc_entries = [
            ("doc1", 90.0, _make_term_dict(term="test", occurrences=5)),
            ("doc2", 80.0, _make_term_dict(term="test", occurrences=3)),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        sources = merged["sources"]
        assert isinstance(sources, TermSources)
        assert sources.num_documents == 2
        assert sources.total_count == 8
        # Confidences should be normalized to 0-1
        assert sources.confidences == [0.9, 0.8]

    def test_quality_score_in_valid_range(self, extractor):
        """Quality score should be 0-100."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="test", occurrences=5)),
            ("doc2", 90.0, _make_term_dict(term="test", occurrences=3)),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        assert 0.0 <= merged["Quality Score"] <= 100.0
        assert 0.0 <= merged["base_quality_score"] <= 100.0

    def test_textrank_score_takes_max(self, extractor):
        """textrank_score should be max across docs."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="test", textrank_score=0.3)),
            ("doc2", 90.0, _make_term_dict(term="test", textrank_score=0.7)),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        assert merged["textrank_score"] == 0.7

    def test_min_doc_confidence_used(self, extractor):
        """source_doc_confidence should be the minimum across docs."""
        doc_entries = [
            ("doc1", 95.0, _make_term_dict(term="test")),
            ("doc2", 60.0, _make_term_dict(term="test")),
        ]
        merged = extractor._merge_term_across_docs(doc_entries, total_docs=2, total_unique=5)

        assert merged["source_doc_confidence"] == 60.0


# =========================================================================
# _merge_multi_doc_results
# =========================================================================


class TestMergeMultiDocResults:
    """Tests for merging per-document vocab lists."""

    def test_shared_term_merged_into_one(self, extractor):
        """Same term in 2 docs -> 1 entry in output."""
        results = [
            ("doc1", 95.0, [_make_term_dict(term="radiculopathy", occurrences=3)]),
            ("doc2", 90.0, [_make_term_dict(term="radiculopathy", occurrences=2)]),
        ]
        merged = extractor._merge_multi_doc_results(results, total_docs=2)

        lower_terms = [t["Term"].lower() for t in merged]
        assert lower_terms.count("radiculopathy") == 1

    def test_unique_terms_both_present(self, extractor):
        """Different terms from different docs both appear."""
        results = [
            ("doc1", 95.0, [_make_term_dict(term="radiculopathy")]),
            ("doc2", 90.0, [_make_term_dict(term="herniation")]),
        ]
        merged = extractor._merge_multi_doc_results(results, total_docs=2)

        lower_terms = [t["Term"].lower() for t in merged]
        assert "radiculopathy" in lower_terms
        assert "herniation" in lower_terms

    def test_case_insensitive_merge(self, extractor):
        """Terms differing only in case should merge."""
        results = [
            ("doc1", 95.0, [_make_term_dict(term="John Smith")]),
            ("doc2", 90.0, [_make_term_dict(term="JOHN SMITH")]),
        ]
        merged = extractor._merge_multi_doc_results(results, total_docs=2)

        # Should be 1 entry, not 2
        assert len(merged) == 1
        assert merged[0]["# Docs"] == 2

    def test_sorted_by_quality_score(self, extractor):
        """Output should be sorted by Quality Score descending."""
        results = [
            (
                "doc1",
                95.0,
                [
                    _make_term_dict(term="low_score", quality_score=30.0, occurrences=1),
                    _make_term_dict(term="high_score", quality_score=90.0, occurrences=50),
                ],
            ),
        ]
        merged = extractor._merge_multi_doc_results(results, total_docs=1)

        scores = [t["Quality Score"] for t in merged]
        assert scores == sorted(scores, reverse=True)

    def test_empty_results(self, extractor):
        """Empty per-doc results -> empty merged output."""
        merged = extractor._merge_multi_doc_results([], total_docs=0)
        assert merged == []

    def test_empty_vocab_lists(self, extractor):
        """Docs with no terms -> empty merged output."""
        results = [
            ("doc1", 95.0, []),
            ("doc2", 90.0, []),
        ]
        merged = extractor._merge_multi_doc_results(results, total_docs=2)
        assert merged == []


# =========================================================================
# extract_documents
# =========================================================================


class TestExtractDocuments:
    """Tests for the top-level extract_documents orchestrator."""

    def test_empty_documents_returns_empty(self, extractor):
        """No documents -> empty result."""
        result = extractor.extract_documents([])
        assert result == []

    def test_single_doc_calls_extract(self, extractor):
        """Single document should call extract() directly, not parallel merge."""
        fake_vocab = [_make_term_dict(term="test")]

        with patch.object(extractor, "extract", return_value=fake_vocab) as mock_extract:
            result = extractor.extract_documents(
                [{"text": "some text", "doc_id": "doc1", "confidence": 95.0}],
                use_llm=False,
            )

        mock_extract.assert_called_once()
        assert result == fake_vocab

    def test_single_doc_with_llm_calls_extract_with_llm(self, extractor):
        """Single document with use_llm=True should call extract_with_llm()."""
        fake_vocab = [_make_term_dict(term="test")]

        with patch.object(extractor, "extract_with_llm", return_value=fake_vocab) as mock_llm:
            result = extractor.extract_documents(
                [{"text": "some text", "doc_id": "doc1", "confidence": 95.0}],
                use_llm=True,
            )

        mock_llm.assert_called_once()
        assert result == fake_vocab

    def test_multi_doc_calls_merge(self, extractor):
        """Multiple documents should invoke parallel extraction and merge."""
        fake_vocab = [_make_term_dict(term="test", occurrences=3)]

        # Mock extract() to return fake vocab for each doc
        with patch.object(extractor, "extract", return_value=fake_vocab):
            result = extractor.extract_documents(
                [
                    {"text": "text one", "doc_id": "doc1", "confidence": 95.0},
                    {"text": "text two", "doc_id": "doc2", "confidence": 90.0},
                ],
                use_llm=False,
            )

        # Both docs return same term -> should merge into 1 entry with # Docs = 2
        assert len(result) == 1
        assert result[0]["# Docs"] == 2
        assert result[0]["Occurrences"] == 6  # 3 + 3

    def test_progress_callback_fires(self, extractor):
        """Progress callback should fire once per document."""
        fake_vocab = [_make_term_dict(term="test")]
        callback = MagicMock()

        with patch.object(extractor, "extract", return_value=fake_vocab):
            extractor.extract_documents(
                [
                    {"text": "text one", "doc_id": "doc1", "confidence": 95.0},
                    {"text": "text two", "doc_id": "doc2", "confidence": 90.0},
                ],
                use_llm=False,
                progress_callback=callback,
            )

        assert callback.call_count == 2
        # Each call should have (current, total, doc_id)
        for call_args in callback.call_args_list:
            args = call_args[0]
            assert len(args) == 3
            assert args[1] == 2  # total is always 2

    def test_multi_doc_unique_terms_from_different_docs(self, extractor):
        """Different terms from different docs should all appear."""
        vocab1 = [_make_term_dict(term="Alpha")]
        vocab2 = [_make_term_dict(term="Beta")]

        call_count = [0]

        def fake_extract(text, doc_count=1, doc_confidence=100.0):
            call_count[0] += 1
            return vocab1 if call_count[0] == 1 else vocab2

        with patch.object(extractor, "extract", side_effect=fake_extract):
            result = extractor.extract_documents(
                [
                    {"text": "text one", "doc_id": "doc1", "confidence": 95.0},
                    {"text": "text two", "doc_id": "doc2", "confidence": 90.0},
                ],
                use_llm=False,
            )

        lower_terms = [t["Term"].lower() for t in result]
        assert "alpha" in lower_terms
        assert "beta" in lower_terms
        # Each should have # Docs = 1
        for t in result:
            assert t["# Docs"] == 1
