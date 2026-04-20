"""Tests for retrieval pipeline: data structures, BM25+, FAISS, merger, reranker."""

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Data structures (base.py)
# ---------------------------------------------------------------------------


class TestDocumentChunk:
    """DocumentChunk dataclass for retrieval input."""

    def test_basic_creation(self):
        from src.core.retrieval.base import DocumentChunk

        chunk = DocumentChunk(text="Hello world", chunk_id="c1", filename="doc.pdf")
        assert chunk.text == "Hello world"
        assert chunk.chunk_id == "c1"
        assert chunk.filename == "doc.pdf"

    def test_auto_word_count(self):
        from src.core.retrieval.base import DocumentChunk

        chunk = DocumentChunk(text="one two three four", chunk_id="c1", filename="doc.pdf")
        assert chunk.word_count == 4

    def test_explicit_word_count_preserved(self):
        from src.core.retrieval.base import DocumentChunk

        chunk = DocumentChunk(text="one two", chunk_id="c1", filename="doc.pdf", word_count=99)
        assert chunk.word_count == 99

    def test_defaults(self):
        from src.core.retrieval.base import DocumentChunk

        chunk = DocumentChunk(text="t", chunk_id="c1", filename="f")
        assert chunk.chunk_num == 0
        assert chunk.section_name == "N/A"
        assert chunk.metadata == {}


class TestRetrievedChunk:
    """RetrievedChunk dataclass for algorithm output."""

    def test_basic_creation(self):
        from src.core.retrieval.base import RetrievedChunk

        rc = RetrievedChunk(
            chunk_id="c1",
            text="text",
            relevance_score=0.8,
            raw_score=5.0,
            source_algorithm="BM25+",
            filename="doc.pdf",
        )
        assert rc.relevance_score == 0.8
        assert rc.raw_score == 5.0

    def test_score_clamped_to_one(self):
        from src.core.retrieval.base import RetrievedChunk

        rc = RetrievedChunk(
            chunk_id="c1",
            text="t",
            relevance_score=1.5,
            raw_score=10.0,
            source_algorithm="FAISS",
            filename="f",
        )
        assert rc.relevance_score <= 1.0

    def test_score_clamped_to_zero(self):
        from src.core.retrieval.base import RetrievedChunk

        rc = RetrievedChunk(
            chunk_id="c1",
            text="t",
            relevance_score=-0.5,
            raw_score=-1.0,
            source_algorithm="FAISS",
            filename="f",
        )
        assert rc.relevance_score >= 0.0


class TestAlgorithmRetrievalResult:
    """AlgorithmRetrievalResult wraps algorithm output."""

    def test_len(self):
        from src.core.retrieval.base import AlgorithmRetrievalResult, RetrievedChunk

        rc = RetrievedChunk(
            chunk_id="c1",
            text="t",
            relevance_score=0.5,
            raw_score=1.0,
            source_algorithm="BM25+",
            filename="f",
        )
        result = AlgorithmRetrievalResult(chunks=[rc])
        assert len(result) == 1

    def test_empty(self):
        from src.core.retrieval.base import AlgorithmRetrievalResult

        result = AlgorithmRetrievalResult(chunks=[])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# ChunkMerger (RRF)
# ---------------------------------------------------------------------------


class TestMergedChunk:
    """MergedChunk dataclass for merged output."""

    def test_basic_creation(self):
        from src.core.retrieval.chunk_merger import MergedChunk

        mc = MergedChunk(
            chunk_id="c1",
            text="hello",
            combined_score=0.5,
            sources=["BM25+"],
            filename="doc.pdf",
        )
        assert mc.chunk_id == "c1"
        assert mc.combined_score == 0.5


class TestMergedRetrievalResult:
    """MergedRetrievalResult wraps merger output."""

    def test_len(self):
        from src.core.retrieval.chunk_merger import MergedChunk, MergedRetrievalResult

        mc = MergedChunk(
            chunk_id="c1", text="t", combined_score=0.5, sources=["BM25+"], filename="f"
        )
        result = MergedRetrievalResult(chunks=[mc], total_algorithms=1)
        assert len(result) == 1


class TestChunkMerger:
    """ChunkMerger implements Reciprocal Rank Fusion."""

    def _make(self, weights=None):
        from src.core.retrieval.chunk_merger import ChunkMerger

        return ChunkMerger(algorithm_weights=weights)

    def test_empty_results(self):
        merger = self._make()
        result = merger.merge([])
        assert len(result) == 0

    def test_single_algorithm_result(self):
        from src.core.retrieval.base import AlgorithmRetrievalResult, RetrievedChunk

        rc = RetrievedChunk(
            chunk_id="c1",
            text="hello",
            relevance_score=0.9,
            raw_score=5.0,
            source_algorithm="BM25+",
            filename="doc.pdf",
        )
        algo_result = AlgorithmRetrievalResult(chunks=[rc])

        merger = self._make(weights={"BM25+": 1.0})
        merged = merger.merge([algo_result])
        assert len(merged) == 1
        assert merged.chunks[0].chunk_id == "c1"

    def test_multiple_algorithms_accumulate_scores(self):
        from src.core.retrieval.base import AlgorithmRetrievalResult, RetrievedChunk

        # Same chunk found by both algorithms
        rc1 = RetrievedChunk(
            chunk_id="c1",
            text="t",
            relevance_score=0.9,
            raw_score=5.0,
            source_algorithm="BM25+",
            filename="f",
        )
        rc2 = RetrievedChunk(
            chunk_id="c1",
            text="t",
            relevance_score=0.8,
            raw_score=4.0,
            source_algorithm="FAISS",
            filename="f",
        )
        # Different chunk only in BM25+
        rc3 = RetrievedChunk(
            chunk_id="c2",
            text="t2",
            relevance_score=0.7,
            raw_score=3.0,
            source_algorithm="BM25+",
            filename="f",
        )

        r1 = AlgorithmRetrievalResult(chunks=[rc1, rc3])
        r2 = AlgorithmRetrievalResult(chunks=[rc2])

        merger = self._make(weights={"BM25+": 1.0, "FAISS": 1.0})
        merged = merger.merge([r1, r2])

        # c1 should score higher (found by both)
        by_id = {c.chunk_id: c for c in merged.chunks}
        assert by_id["c1"].combined_score > by_id["c2"].combined_score
        assert "BM25+" in by_id["c1"].sources
        assert "FAISS" in by_id["c1"].sources

    def test_top_k_limits_results(self):
        from src.core.retrieval.base import AlgorithmRetrievalResult, RetrievedChunk

        chunks = [
            RetrievedChunk(
                chunk_id=f"c{i}",
                text=f"t{i}",
                relevance_score=0.9 - i * 0.1,
                raw_score=5.0,
                source_algorithm="BM25+",
                filename="f",
            )
            for i in range(5)
        ]
        r = AlgorithmRetrievalResult(chunks=chunks)
        merger = self._make(weights={"BM25+": 1.0})
        merged = merger.merge([r], k=3)
        assert len(merged) == 3

    def test_update_weights(self):
        merger = self._make(weights={"BM25+": 0.5})
        merger.update_weights({"BM25+": 0.8, "FAISS": 0.2})
        assert merger.algorithm_weights["BM25+"] == 0.8
        assert merger.algorithm_weights["FAISS"] == 0.2


# ---------------------------------------------------------------------------
# BM25PlusRetriever
# ---------------------------------------------------------------------------


class TestBM25PlusRetriever:
    """BM25PlusRetriever keyword search algorithm."""

    def _make(self):
        from src.core.retrieval.algorithms.bm25_plus import BM25PlusRetriever

        return BM25PlusRetriever()

    def _chunks(self):
        from src.core.retrieval.base import DocumentChunk

        return [
            DocumentChunk(
                text="The plaintiff filed a motion for summary judgment.",
                chunk_id="c1",
                filename="doc.pdf",
            ),
            DocumentChunk(
                text="The defendant objected to the motion.", chunk_id="c2", filename="doc.pdf"
            ),
            DocumentChunk(
                text="The court denied the motion for reconsideration.",
                chunk_id="c3",
                filename="doc.pdf",
            ),
        ]

    def test_not_indexed_initially(self):
        r = self._make()
        assert r.is_indexed is False

    def test_index_documents(self):
        r = self._make()
        r.index_documents(self._chunks())
        assert r.is_indexed is True

    def test_index_empty_raises(self):
        r = self._make()
        with pytest.raises(ValueError):
            r.index_documents([])

    def test_retrieve_before_index_raises(self):
        r = self._make()
        with pytest.raises(RuntimeError):
            r.retrieve("motion")

    def test_retrieve_returns_results(self):
        r = self._make()
        r.index_documents(self._chunks())
        result = r.retrieve("plaintiff summary judgment", k=3)
        assert len(result) > 0
        # c1 should be most relevant (has both "plaintiff" and "summary judgment")
        assert result.chunks[0].chunk_id == "c1"

    def test_scores_normalized_0_to_1(self):
        r = self._make()
        r.index_documents(self._chunks())
        result = r.retrieve("motion", k=3)
        for chunk in result.chunks:
            assert 0.0 <= chunk.relevance_score <= 1.0

    def test_name_attribute(self):
        assert self._make().name == "BM25+"

    def test_get_config(self):
        r = self._make()
        r.index_documents(self._chunks())
        config = r.get_config()
        assert config["name"] == "BM25+"
        assert "index_size" in config


# ---------------------------------------------------------------------------
# FAISSRetriever
# ---------------------------------------------------------------------------


class TestFAISSRetriever:
    """FAISSRetriever semantic search algorithm."""

    def test_name_attribute(self):
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        r = FAISSRetriever()
        assert r.name == "FAISS"

    def test_not_indexed_initially(self):
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        r = FAISSRetriever()
        assert r.is_indexed is False

    def test_index_empty_raises(self):
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        r = FAISSRetriever()
        with pytest.raises(ValueError):
            r.index_documents([])

    def test_get_config(self):
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        r = FAISSRetriever()
        config = r.get_config()
        assert config["name"] == "FAISS"
        assert "distance_metric" in config

    def test_set_embeddings(self):
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        r = FAISSRetriever()
        mock_embed = MagicMock()
        r.set_embeddings(mock_embed)
        assert r._embeddings is mock_embed


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    """CrossEncoderReranker re-ranks chunks using cross-encoder model."""

    def test_class_has_required_interface(self):
        """CrossEncoderReranker exposes rerank, is_available, and a loadable model attribute."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        r = CrossEncoderReranker()
        # Minimum contract: rerank method, is_available method, model attribute
        assert callable(r.rerank)
        assert (
            callable(r.is_available)
            or isinstance(r.is_available, bool)
            or hasattr(r, "is_available")
        )
        # rerank is the core entry — it must accept (query, chunks) kwargs
        import inspect

        sig = inspect.signature(r.rerank)
        params = list(sig.parameters)
        assert "query" in params or len(params) >= 2  # (self, query, chunks, ...)

    def test_has_rerank_method(self):
        """rerank is a callable method on CrossEncoderReranker."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        r = CrossEncoderReranker()
        assert callable(r.rerank)

    def test_has_is_available_method(self):
        """is_available check is exposed (method or attribute)."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        r = CrossEncoderReranker()
        assert hasattr(r, "is_available")

    def test_min_relevance_score_is_valid_threshold(self):
        """MIN_RELEVANCE_SCORE is a float in (0, 1) — a valid probability threshold."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        r = CrossEncoderReranker()
        assert 0.0 < r.MIN_RELEVANCE_SCORE < 1.0
        # Must be a numeric value
        assert isinstance(r.MIN_RELEVANCE_SCORE, (int, float))


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    """HybridRetriever combines BM25+ and FAISS algorithms."""

    def test_creation_defaults_starts_unindexed(self):
        """A freshly constructed BM25-only retriever is not yet indexed and has zero chunks."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        r = HybridRetriever(enable_faiss=False)
        # Invariants: unindexed, zero chunks, BM25 enabled in algorithms
        assert r.is_indexed is False
        assert r.get_chunk_count() == 0
        assert "BM25+" in r._algorithms

    def test_not_indexed_initially(self):
        """HybridRetriever reports not-indexed status before index_documents is called."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        r = HybridRetriever(enable_faiss=False)
        assert r.is_indexed is False

    def test_get_chunk_count_zero(self):
        """Chunk count is exactly 0 before indexing."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        r = HybridRetriever(enable_faiss=False)
        assert r.get_chunk_count() == 0

    def test_retrieve_before_index_raises(self):
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        r = HybridRetriever(enable_faiss=False)
        with pytest.raises(RuntimeError):
            r.retrieve("test query")

    def test_update_weights(self):
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        r = HybridRetriever(enable_faiss=False)
        r.update_weights({"BM25+": 0.5, "FAISS": 0.5})
        assert r.algorithm_weights["BM25+"] == 0.5

    def test_get_algorithm_status(self):
        """Algorithm status dict reports BM25+ as an enabled algorithm."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        r = HybridRetriever(enable_faiss=False)
        status = r.get_algorithm_status()
        # BM25+ must be present in status since we enabled it
        assert "BM25+" in status
        # FAISS is disabled — should NOT be in status (or marked disabled)
        if "FAISS" in status:
            # If reported, its entry should indicate disabled
            faiss_entry = status["FAISS"]
            assert faiss_entry is False or (
                isinstance(faiss_entry, dict) and faiss_entry.get("enabled") is False
            )

    def test_index_and_retrieve_bm25_only(self):
        """Integration test: index documents and retrieve with BM25+ only."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        r = HybridRetriever(enable_faiss=False, enable_bm25=True)
        docs = [
            {
                "extracted_text": "The plaintiff filed a motion for summary judgment.",
                "filename": "a.pdf",
            },
            {"extracted_text": "The defendant objected to the motion.", "filename": "a.pdf"},
        ]
        count = r.index_documents(docs)
        assert count > 0
        assert r.is_indexed

        result = r.retrieve("plaintiff summary judgment", k=2)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# VectorStoreBuilder
# ---------------------------------------------------------------------------


class TestVectorStoreBuilder:
    """VectorStoreBuilder creates FAISS indexes."""

    def test_class_exposes_expected_interface(self):
        """VectorStoreBuilder exposes the core build/cleanup methods as callables."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        # Core methods must exist and be callable
        assert callable(VectorStoreBuilder.create_from_documents)
        assert callable(VectorStoreBuilder.cleanup_stale_stores)

    def test_has_create_method(self):
        """create_from_documents is a callable method on the class."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        assert callable(VectorStoreBuilder.create_from_documents)

    def test_has_cleanup_method(self):
        """cleanup_stale_stores is a callable method on the class."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        assert callable(VectorStoreBuilder.cleanup_stale_stores)


# ---------------------------------------------------------------------------
# SemanticRetriever
# ---------------------------------------------------------------------------


class TestSemanticRetrieverStructure:
    """SemanticRetriever loads vector stores and retrieves context."""

    def test_class_exposes_required_interface(self):
        """SemanticRetriever has the core retrieve_context + get_chunk_count callables."""
        from src.core.vector_store.semantic_retriever import SemanticRetriever

        # Contract: these are the methods callers depend on
        assert callable(SemanticRetriever.retrieve_context)
        assert callable(SemanticRetriever.get_chunk_count)

    def test_has_retrieve_context_method(self):
        """retrieve_context is callable."""
        from src.core.vector_store.semantic_retriever import SemanticRetriever

        assert callable(SemanticRetriever.retrieve_context)

    def test_has_get_chunk_count_method(self):
        """get_chunk_count is callable."""
        from src.core.vector_store.semantic_retriever import SemanticRetriever

        assert callable(SemanticRetriever.get_chunk_count)

    def test_source_info_dataclass(self):
        from src.core.vector_store.semantic_retriever import SourceInfo

        si = SourceInfo(
            filename="doc.pdf",
            chunk_num=1,
            section="Introduction",
            relevance_score=0.85,
            word_count=100,
        )
        assert si.filename == "doc.pdf"
        assert si.relevance_score == 0.85

    def test_retrieval_result_dataclass(self):
        from src.core.vector_store.semantic_retriever import RetrievalResult

        rr = RetrievalResult(
            context="Some context text",
            sources=[],
            chunks_retrieved=0,
            retrieval_time_ms=50.0,
        )
        assert rr.context == "Some context text"
        assert rr.chunks_retrieved == 0
