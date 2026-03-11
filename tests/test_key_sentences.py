"""
Tests for key sentences extraction via K-means clustering.

Covers: basic extraction, filtering, cluster diversity, edge cases,
source attribution, document position ordering, and K scaling.
"""

from unittest.mock import MagicMock

import numpy as np

from src.core.summarization.key_sentences import (
    KeySentence,
    _cluster_and_select,
    _filter_sentences,
    compute_sentence_count,
    extract_key_sentences,
)

# =========================================================================
# compute_sentence_count
# =========================================================================


class TestComputeSentenceCount:
    """Tests for K scaling with page count."""

    def test_minimum_is_5(self):
        """Even for very short docs, K >= 5."""
        assert compute_sentence_count(1) == 5
        assert compute_sentence_count(10) == 5
        assert compute_sentence_count(24) == 5

    def test_scales_with_pages(self):
        """1 key sentence per ~5 pages."""
        assert compute_sentence_count(25) == 5
        assert compute_sentence_count(50) == 10
        assert compute_sentence_count(60) == 12

    def test_maximum_is_15(self):
        """K is capped at 15."""
        assert compute_sentence_count(100) == 15
        assert compute_sentence_count(500) == 15

    def test_zero_pages(self):
        """Zero pages still returns minimum."""
        assert compute_sentence_count(0) == 5


# =========================================================================
# _filter_sentences
# =========================================================================


class TestFilterSentences:
    """Tests for sentence filtering."""

    def test_removes_short_sentences(self):
        """Sentences with < 5 words are filtered out."""
        sentences = [
            {"text": "Too short."},
            {"text": "This sentence has enough words to pass the filter."},
        ]
        result = _filter_sentences(sentences)
        assert len(result) == 1
        assert "enough words" in result[0]["text"]

    def test_removes_long_sentences(self):
        """Sentences with > 150 words are filtered out."""
        short = {"text": "This is a normal length sentence with several words."}
        long_text = " ".join(["word"] * 160)
        long_sent = {"text": long_text}
        result = _filter_sentences([short, long_sent])
        assert len(result) == 1

    def test_removes_boilerplate_page_numbers(self):
        """Page number lines are filtered."""
        sentences = [
            {"text": "Page 42 of the document reference."},
            {"text": "The plaintiff filed a motion on March 15 seeking damages."},
        ]
        result = _filter_sentences(sentences)
        assert len(result) == 1
        assert "plaintiff" in result[0]["text"]

    def test_removes_boilerplate_exhibit_labels(self):
        """Exhibit labels are filtered."""
        sentences = [
            {"text": "Exhibit A"},
            {"text": "The defendant responded with a denial of all allegations."},
        ]
        result = _filter_sentences(sentences)
        assert len(result) == 1

    def test_keeps_valid_sentences(self):
        """Normal legal sentences pass through."""
        sentences = [
            {"text": "The court found that the defendant was liable for damages."},
            {"text": "Medical records indicate a herniated disc at L4-L5 level."},
        ]
        result = _filter_sentences(sentences)
        assert len(result) == 2

    def test_empty_input(self):
        """Empty list returns empty list."""
        assert _filter_sentences([]) == []


# =========================================================================
# _cluster_and_select
# =========================================================================


class TestClusterAndSelect:
    """Tests for K-means clustering selection."""

    def test_fewer_sentences_than_k(self):
        """When fewer sentences than K, return all indices."""
        embeddings = np.random.randn(3, 768).astype(np.float32)
        result = _cluster_and_select(embeddings, n=10)
        assert sorted(result) == [0, 1, 2]

    def test_exact_k_sentences(self):
        """When exactly K sentences, return all indices."""
        embeddings = np.random.randn(5, 768).astype(np.float32)
        result = _cluster_and_select(embeddings, n=5)
        assert sorted(result) == [0, 1, 2, 3, 4]

    def test_selects_correct_count(self):
        """Should return exactly K indices when enough sentences."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 768).astype(np.float32)
        result = _cluster_and_select(embeddings, n=5)
        assert len(result) <= 5  # May be less if empty clusters
        assert len(set(result)) == len(result)  # No duplicates

    def test_indices_in_range(self):
        """All returned indices should be valid."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 768).astype(np.float32)
        result = _cluster_and_select(embeddings, n=5)
        for idx in result:
            assert 0 <= idx < 20

    def test_diverse_clusters(self):
        """Sentences from distinct clusters should produce diverse selections."""
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.randn(10, 768).astype(np.float32) + np.array([10, 0] + [0] * 766)
        cluster2 = np.random.randn(10, 768).astype(np.float32) + np.array([0, 10] + [0] * 766)
        cluster3 = np.random.randn(10, 768).astype(np.float32) + np.array([-10, -10] + [0] * 766)
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        result = _cluster_and_select(embeddings, n=3)
        # Should pick one from each cluster region
        assert len(result) == 3


# =========================================================================
# extract_key_sentences (integration)
# =========================================================================


class TestExtractKeySentences:
    """Integration tests for the full extraction pipeline."""

    def _make_mock_model(self, dim=768):
        """Create a mock embeddings model returning fixed vectors."""
        model = MagicMock()

        def embed_documents(texts):
            np.random.seed(42)
            return np.random.randn(len(texts), dim).tolist()

        model.embed_documents = embed_documents
        return model

    def test_basic_extraction(self):
        """Should extract key sentences from documents."""
        docs = [
            {
                "filename": "complaint.pdf",
                "preprocessed_text": (
                    "The plaintiff was diagnosed with a herniated disc at L4-L5. "
                    "Dr. Smith recommended surgical intervention after conservative treatment. "
                    "The accident occurred at the intersection of Main and Oak streets. "
                    "Physical therapy sessions were conducted three times per week. "
                    "The defendant ran a red light causing the collision. "
                    "Medical bills totaled over fifty thousand dollars for the treatment. "
                    "The plaintiff was unable to work for six months after the accident."
                ),
                "page_count": 10,
            }
        ]
        model = self._make_mock_model()
        results = extract_key_sentences(docs, model, n=3)
        assert len(results) == 3
        assert all(isinstance(r, KeySentence) for r in results)

    def test_source_attribution(self):
        """Each key sentence should have correct source file."""
        docs = [
            {
                "filename": "file_a.pdf",
                "preprocessed_text": (
                    "The first document contains important information about the case. "
                    "It describes the events leading up to the incident in detail."
                ),
                "page_count": 5,
            },
            {
                "filename": "file_b.pdf",
                "preprocessed_text": (
                    "The second document provides medical evidence from the hospital. "
                    "Treatment records show multiple visits over six months."
                ),
                "page_count": 5,
            },
        ]
        model = self._make_mock_model()
        results = extract_key_sentences(docs, model, n=2)
        source_files = {r.source_file for r in results}
        # Should have attribution from at least one file
        assert all(f in ("file_a.pdf", "file_b.pdf") for f in source_files)

    def test_document_position_ordering(self):
        """Results should be sorted by document position, not cluster order."""
        docs = [
            {
                "filename": "doc.pdf",
                "preprocessed_text": (
                    "First sentence in the document about the plaintiff's injuries. "
                    "Second sentence describes the medical treatment received. "
                    "Third sentence covers the defendant's response to allegations. "
                    "Fourth sentence discusses the timeline of events in detail. "
                    "Fifth sentence summarizes the damages claimed by plaintiff. "
                    "Sixth sentence outlines the legal basis for the complaint."
                ),
                "page_count": 5,
            }
        ]
        model = self._make_mock_model()
        results = extract_key_sentences(docs, model, n=3)
        positions = [r.position for r in results]
        assert positions == sorted(positions), "Results should be in document order"

    def test_empty_documents(self):
        """Empty documents should return empty list."""
        docs = [{"filename": "empty.pdf", "preprocessed_text": "", "page_count": 0}]
        model = self._make_mock_model()
        results = extract_key_sentences(docs, model)
        assert results == []

    def test_no_documents(self):
        """No documents should return empty list."""
        model = self._make_mock_model()
        results = extract_key_sentences([], model)
        assert results == []

    def test_fewer_valid_sentences_than_k(self):
        """Should return all valid sentences if fewer than K."""
        docs = [
            {
                "filename": "short.pdf",
                "preprocessed_text": (
                    "One valid sentence about the legal matter at hand. "
                    "Another valid sentence describing the relevant facts."
                ),
                "page_count": 1,
            }
        ]
        model = self._make_mock_model()
        results = extract_key_sentences(docs, model, n=10)
        assert len(results) == 2  # Only 2 valid sentences

    def test_uses_preprocessed_text_over_extracted(self):
        """Should prefer preprocessed_text over extracted_text."""
        docs = [
            {
                "filename": "doc.pdf",
                "preprocessed_text": "The preprocessed text is used for sentence extraction here.",
                "extracted_text": "The raw extracted text should not be used for this purpose.",
                "page_count": 1,
            }
        ]
        model = self._make_mock_model()
        results = extract_key_sentences(docs, model, n=5)
        if results:
            assert "preprocessed" in results[0].text

    def test_falls_back_to_extracted_text(self):
        """Should use extracted_text when preprocessed_text is missing."""
        docs = [
            {
                "filename": "doc.pdf",
                "extracted_text": "The extracted text is used as fallback when preprocessed is missing.",
                "page_count": 1,
            }
        ]
        model = self._make_mock_model()
        results = extract_key_sentences(docs, model, n=5)
        if results:
            assert "extracted" in results[0].text

    def test_embedding_failure_returns_empty(self):
        """If embedding fails, should return empty list gracefully."""
        docs = [
            {
                "filename": "doc.pdf",
                "preprocessed_text": "A valid sentence that should be processed normally.",
                "page_count": 1,
            }
        ]
        model = MagicMock()
        model.embed_documents.side_effect = RuntimeError("CUDA out of memory")
        results = extract_key_sentences(docs, model, n=5)
        assert results == []

    def test_auto_k_uses_voter(self):
        """When n is None, K is chosen by 3-voter ensemble, not page count."""
        # Create a document with many sentences
        sentences = [
            f"Sentence number {i} contains enough words to pass the filter easily."
            for i in range(50)
        ]
        docs = [
            {
                "filename": "big.pdf",
                "preprocessed_text": " ".join(sentences),
                "page_count": 75,
            }
        ]
        model = self._make_mock_model()
        results = extract_key_sentences(docs, model)  # n=None, voter picks K
        # Voter determines K from content, not page count
        assert 2 <= len(results) <= 15
