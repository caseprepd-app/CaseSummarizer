"""
Edge-case tests for HybridRetriever.

Covers empty queries, special characters, very long queries,
all-algorithm failures, unrelated queries, single-document indexing,
and duplicate document indexing.
"""

from unittest.mock import MagicMock

import pytest

from src.core.retrieval.base import DocumentChunk
from src.core.retrieval.chunk_merger import MergedRetrievalResult

# Reusable legal document chunks for indexing
MEDICAL_CHUNKS = [
    DocumentChunk(
        text="The patient presented with acute appendicitis requiring surgical intervention.",
        chunk_id="med_0",
        filename="medical.pdf",
        chunk_num=0,
    ),
    DocumentChunk(
        text="Postoperative recovery was uneventful with discharge on hospital day three.",
        chunk_id="med_1",
        filename="medical.pdf",
        chunk_num=1,
    ),
    DocumentChunk(
        text="Prescribed medications included amoxicillin and ibuprofen for pain management.",
        chunk_id="med_2",
        filename="medical.pdf",
        chunk_num=2,
    ),
]


def _bm25_only_retriever():
    """Create a BM25-only HybridRetriever (no FAISS, no embeddings)."""
    from src.core.retrieval import HybridRetriever

    return HybridRetriever(enable_bm25=True, enable_faiss=False)


def _index_medical_docs(retriever):
    """Index MEDICAL_CHUNKS into the retriever."""
    docs = [
        {
            "filename": "medical.pdf",
            "chunks": [{"text": c.text, "chunk_num": c.chunk_num} for c in MEDICAL_CHUNKS],
        }
    ]
    return retriever.index_documents(docs)


class TestEmptyQueryString:
    """Verify empty-string queries do not crash."""

    def test_empty_query_returns_no_crash(self):
        """Pass empty string as query; expect empty results or ValueError."""
        retriever = _bm25_only_retriever()
        _index_medical_docs(retriever)

        result = retriever.retrieve("", k=5)

        assert isinstance(result, MergedRetrievalResult)
        # Empty query should return zero or very low relevance
        assert len(result.chunks) == 0 or all(c.combined_score >= 0 for c in result.chunks)


class TestSpecialCharactersQuery:
    """Verify queries with only special characters are handled."""

    def test_special_chars_graceful(self):
        """Query with only special chars; expect no crash."""
        retriever = _bm25_only_retriever()
        _index_medical_docs(retriever)

        result = retriever.retrieve("!@#$%^&*()", k=5)

        assert isinstance(result, MergedRetrievalResult)
        # No real words to match; should be empty or harmless
        assert len(result.chunks) == 0 or all(c.combined_score >= 0 for c in result.chunks)


class TestVeryLongQuery:
    """Verify a 1000+ word query completes without hang or OOM."""

    @pytest.mark.timeout(30)
    def test_long_query_completes(self):
        """Query with 1000+ words finishes within timeout."""
        retriever = _bm25_only_retriever()
        _index_medical_docs(retriever)

        long_query = " ".join(["medication"] * 1200)
        result = retriever.retrieve(long_query, k=5)

        assert isinstance(result, MergedRetrievalResult)


class TestAllAlgorithmsFailAfterIndex:
    """After successful indexing, both algorithms raise on retrieve."""

    def test_graceful_when_all_fail(self):
        """Mock both BM25 and FAISS to raise; expect empty result."""
        retriever = _bm25_only_retriever()
        _index_medical_docs(retriever)

        # Patch the BM25 algorithm's retrieve to raise
        bm25_algo = retriever._algorithms["BM25+"]
        original_retrieve = bm25_algo.retrieve
        bm25_algo.retrieve = MagicMock(side_effect=RuntimeError("BM25 exploded"))

        result = retriever.retrieve("appendicitis", k=5)

        assert isinstance(result, MergedRetrievalResult)
        assert len(result.chunks) == 0
        assert result.metadata.get("error") is not None

        # Restore for safety
        bm25_algo.retrieve = original_retrieve


class TestQueryMatchingNothing:
    """Query about unrelated topic against medical documents."""

    def test_unrelated_query_low_relevance(self):
        """Query about quantum computing vs medical docs."""
        retriever = _bm25_only_retriever()
        _index_medical_docs(retriever)

        result = retriever.retrieve("quantum computing superposition entanglement", k=5)

        assert isinstance(result, MergedRetrievalResult)
        # BM25 should find zero keyword overlap
        assert len(result.chunks) == 0


class TestSingleDocumentIndex:
    """Index exactly one short document and retrieve."""

    def test_single_doc_retrieval(self):
        """One document indexed; retrieval with k=1 works."""
        retriever = _bm25_only_retriever()
        docs = [
            {
                "filename": "solo.pdf",
                "chunks": [{"text": "Sole exhibit entered into evidence.", "chunk_num": 0}],
            }
        ]
        retriever.index_documents(docs)

        result = retriever.retrieve("exhibit evidence", k=1)

        assert isinstance(result, MergedRetrievalResult)
        assert len(result.chunks) >= 1
        assert "exhibit" in result.chunks[0].text.lower()


class TestDuplicateDocumentsIndexed:
    """Index the same document text twice."""

    def test_duplicates_no_crash(self):
        """Duplicate texts indexed; retrieval returns valid results."""
        retriever = _bm25_only_retriever()
        docs = [
            {
                "filename": "dup.pdf",
                "chunks": [
                    {
                        "text": "Defendant was negligent in maintaining the premises.",
                        "chunk_num": 0,
                    },
                    {
                        "text": "Defendant was negligent in maintaining the premises.",
                        "chunk_num": 1,
                    },
                ],
            }
        ]
        retriever.index_documents(docs)

        result = retriever.retrieve("negligent premises", k=5)

        assert isinstance(result, MergedRetrievalResult)
        assert len(result.chunks) >= 1
        # All returned chunks should have valid scores
        for chunk in result.chunks:
            assert chunk.combined_score > 0
