"""
Tests for Hybrid Retrieval System (Session 31).

Tests the multi-algorithm retrieval architecture:
- BM25+ lexical search
- FAISS semantic search
- ChunkMerger weighted combination
- HybridRetriever coordination
"""

import pytest

from src.core.retrieval import HybridRetriever
from src.core.retrieval.algorithms import BM25PlusRetriever, get_all_algorithms
from src.core.retrieval.base import (
    AlgorithmRetrievalResult,
    BaseRetrievalAlgorithm,
    DocumentChunk,
    RetrievedChunk,
)
from src.core.retrieval.chunk_merger import ChunkMerger

# Test data - simulated legal document chunks
SAMPLE_CHUNKS = [
    DocumentChunk(
        text="The plaintiff John Smith filed a complaint against XYZ Corporation on January 15, 2024.",
        chunk_id="doc1_0",
        filename="complaint.pdf",
        chunk_num=0,
        section_name="Introduction",
    ),
    DocumentChunk(
        text="The defendant XYZ Corporation is a Delaware corporation with its principal place of business in New York.",
        chunk_id="doc1_1",
        filename="complaint.pdf",
        chunk_num=1,
        section_name="Parties",
    ),
    DocumentChunk(
        text="Plaintiff seeks compensatory damages in the amount of $500,000 for personal injuries sustained.",
        chunk_id="doc1_2",
        filename="complaint.pdf",
        chunk_num=2,
        section_name="Relief",
    ),
    DocumentChunk(
        text="The incident occurred on December 1, 2023, when plaintiff was struck by a delivery vehicle.",
        chunk_id="doc1_3",
        filename="complaint.pdf",
        chunk_num=3,
        section_name="Facts",
    ),
]


class TestBM25PlusRetriever:
    """Test BM25+ algorithm implementation."""

    def test_initialization(self):
        """Test BM25+ retriever initializes correctly."""
        retriever = BM25PlusRetriever()
        assert retriever.name == "BM25+"
        assert retriever.weight == 0.2  # Secondary weight - exact term matching
        assert retriever.enabled is True
        assert retriever.is_indexed is False

    def test_index_documents(self):
        """Test document indexing creates valid index."""
        retriever = BM25PlusRetriever()
        retriever.index_documents(SAMPLE_CHUNKS)

        assert retriever.is_indexed is True
        assert len(retriever._chunks) == 4

    def test_index_empty_raises_error(self):
        """Test indexing empty list raises ValueError."""
        retriever = BM25PlusRetriever()
        with pytest.raises(ValueError, match="empty"):
            retriever.index_documents([])

    def test_retrieve_before_index_raises_error(self):
        """Test retrieval before indexing raises RuntimeError."""
        retriever = BM25PlusRetriever()
        with pytest.raises(RuntimeError, match="not built"):
            retriever.retrieve("test query")

    def test_retrieve_returns_relevant_chunks(self):
        """Test retrieval returns relevant chunks for query."""
        retriever = BM25PlusRetriever()
        retriever.index_documents(SAMPLE_CHUNKS)

        result = retriever.retrieve("Who is the plaintiff?", k=3)

        assert len(result.chunks) > 0
        assert result.query == "Who is the plaintiff?"
        # Should find chunks mentioning "plaintiff"
        found_plaintiff = any("plaintiff" in c.text.lower() for c in result.chunks)
        assert found_plaintiff, "Expected to find chunks mentioning plaintiff"

    def test_retrieve_scores_are_normalized(self):
        """Test relevance scores are normalized to 0-1 range."""
        retriever = BM25PlusRetriever()
        retriever.index_documents(SAMPLE_CHUNKS)

        result = retriever.retrieve("damages compensation", k=5)

        for chunk in result.chunks:
            assert 0.0 <= chunk.relevance_score <= 1.0
            assert chunk.raw_score >= 0  # BM25 scores should be non-negative

    def test_retrieve_respects_k_limit(self):
        """Test k parameter limits returned chunks."""
        retriever = BM25PlusRetriever()
        retriever.index_documents(SAMPLE_CHUNKS)

        result = retriever.retrieve("defendant plaintiff", k=2)

        assert len(result.chunks) <= 2


class TestChunkMerger:
    """Test chunk merging functionality."""

    def test_merge_single_algorithm_result(self):
        """Test merging works with single algorithm."""
        chunks = [
            RetrievedChunk(
                chunk_id="doc1_0",
                text="Test text",
                relevance_score=0.8,
                raw_score=5.0,
                source_algorithm="BM25+",
                filename="test.pdf",
            )
        ]
        result = AlgorithmRetrievalResult(chunks=chunks, query="test")

        merger = ChunkMerger({"BM25+": 1.0})
        merged = merger.merge([result])

        assert len(merged.chunks) == 1
        assert merged.chunks[0].combined_score == 0.8

    def test_merge_combines_duplicate_chunks(self):
        """Test same chunk from multiple algorithms is merged."""
        bm25_chunk = RetrievedChunk(
            chunk_id="doc1_0",
            text="Test text",
            relevance_score=0.8,
            raw_score=5.0,
            source_algorithm="BM25+",
            filename="test.pdf",
        )
        faiss_chunk = RetrievedChunk(
            chunk_id="doc1_0",  # Same chunk_id
            text="Test text",
            relevance_score=0.6,
            raw_score=0.6,
            source_algorithm="FAISS",
            filename="test.pdf",
        )

        bm25_result = AlgorithmRetrievalResult(chunks=[bm25_chunk], query="test")
        faiss_result = AlgorithmRetrievalResult(chunks=[faiss_chunk], query="test")

        merger = ChunkMerger({"BM25+": 1.0, "FAISS": 0.5})
        merged = merger.merge([bm25_result, faiss_result])

        assert len(merged.chunks) == 1
        assert "BM25+" in merged.chunks[0].sources
        assert "FAISS" in merged.chunks[0].sources

    def test_merge_applies_multi_algo_bonus(self):
        """Test bonus is applied when multiple algorithms find same chunk."""
        bm25_chunk = RetrievedChunk(
            chunk_id="doc1_0",
            text="Test text",
            relevance_score=0.5,
            raw_score=3.0,
            source_algorithm="BM25+",
            filename="test.pdf",
        )
        faiss_chunk = RetrievedChunk(
            chunk_id="doc1_0",
            text="Test text",
            relevance_score=0.5,
            raw_score=0.5,
            source_algorithm="FAISS",
            filename="test.pdf",
        )

        bm25_result = AlgorithmRetrievalResult(chunks=[bm25_chunk], query="test")
        faiss_result = AlgorithmRetrievalResult(chunks=[faiss_chunk], query="test")

        merger = ChunkMerger({"BM25+": 1.0, "FAISS": 1.0})
        merger.multi_algo_bonus = 0.1
        merged = merger.merge([bm25_result, faiss_result])

        # Base score would be 0.5, bonus adds 0.1 for second algorithm
        assert merged.chunks[0].combined_score > 0.5

    def test_merge_sorts_by_score(self):
        """Test merged results are sorted by combined score."""
        chunks = [
            RetrievedChunk(
                chunk_id="doc1_0",
                text="Low score",
                relevance_score=0.3,
                raw_score=2.0,
                source_algorithm="BM25+",
                filename="test.pdf",
            ),
            RetrievedChunk(
                chunk_id="doc1_1",
                text="High score",
                relevance_score=0.9,
                raw_score=8.0,
                source_algorithm="BM25+",
                filename="test.pdf",
            ),
        ]
        result = AlgorithmRetrievalResult(chunks=chunks, query="test")

        merger = ChunkMerger()
        merged = merger.merge([result])

        assert merged.chunks[0].combined_score > merged.chunks[1].combined_score


class TestHybridRetriever:
    """Test hybrid retrieval coordination."""

    def test_initialization_bm25_only(self):
        """Test initialization with BM25+ only."""
        retriever = HybridRetriever(
            enable_bm25=True,
            enable_faiss=False,
        )
        assert "BM25+" in retriever._algorithms
        assert "FAISS" not in retriever._algorithms

    def test_index_documents_from_dict(self):
        """Test indexing documents from dict format."""
        retriever = HybridRetriever(
            enable_bm25=True,
            enable_faiss=False,  # Skip FAISS for speed
        )

        documents = [
            {
                "filename": "test.pdf",
                "chunks": [
                    {"text": "Test chunk one", "chunk_num": 0},
                    {"text": "Test chunk two", "chunk_num": 1},
                ],
            }
        ]

        count = retriever.index_documents(documents)
        assert count == 2
        assert retriever.is_indexed is True

    def test_retrieve_combines_algorithms(self):
        """Test retrieval uses all enabled algorithms."""
        retriever = HybridRetriever(
            enable_bm25=True,
            enable_faiss=False,  # Skip FAISS for speed
        )

        documents = [
            {
                "filename": "complaint.pdf",
                "extracted_text": "The plaintiff John Smith filed suit against the defendant XYZ Corp for damages.",
            }
        ]

        retriever.index_documents(documents)
        result = retriever.retrieve("Who is the plaintiff?", k=3)

        assert len(result.chunks) > 0
        assert result.total_algorithms >= 1

    def test_get_algorithm_status(self):
        """Test algorithm status reporting."""
        retriever = HybridRetriever(
            enable_bm25=True,
            enable_faiss=False,
        )

        status = retriever.get_algorithm_status()
        assert "BM25+" in status
        assert status["BM25+"]["enabled"] is True


class TestAlgorithmRegistry:
    """Test algorithm registration and discovery."""

    def test_all_algorithms_registered(self):
        """Test expected algorithms are in registry."""
        algorithms = get_all_algorithms()
        assert "BM25+" in algorithms
        assert "FAISS" in algorithms

    def test_algorithms_are_subclasses(self):
        """Test registered algorithms inherit from base class."""
        algorithms = get_all_algorithms()
        for _name, cls in algorithms.items():
            assert issubclass(cls, BaseRetrievalAlgorithm)
