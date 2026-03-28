"""
Tests for src/core/retrieval/hybrid_retriever.py.

Covers initialization, weight management, indexing, and retrieval.
Heavy dependencies (BM25+, FAISS, embeddings) are mocked.
"""

from unittest.mock import patch

from src.core.retrieval.base import DocumentChunk


def _make_doc_chunks(n=3):
    """Create sample DocumentChunk list."""
    return [
        DocumentChunk(
            text=f"Chunk {i} text about the case.",
            chunk_id=f"doc_{i}",
            filename="test.pdf",
            chunk_num=i,
        )
        for i in range(n)
    ]


# Patch targets must be where the names are looked up (inside _init_algorithms)
_BM25_PATCH = "src.core.retrieval.algorithms.bm25_plus.BM25PlusRetriever"
_FAISS_PATCH = "src.core.retrieval.algorithms.faiss_semantic.FAISSRetriever"


class TestHybridRetrieverInit:
    """Tests for HybridRetriever initialization."""

    @patch(_FAISS_PATCH)
    @patch(_BM25_PATCH)
    def test_default_init_creates_both_algorithms(self, mock_bm25, mock_faiss):
        """Should initialize both BM25+ and FAISS by default."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()
        assert "BM25+" in retriever._algorithms
        assert "FAISS" in retriever._algorithms

    @patch(_FAISS_PATCH)
    @patch(_BM25_PATCH)
    def test_disable_bm25(self, mock_bm25, mock_faiss):
        """enable_bm25=False should skip BM25+ algorithm."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(enable_bm25=False)
        assert "BM25+" not in retriever._algorithms

    @patch(_FAISS_PATCH)
    @patch(_BM25_PATCH)
    def test_disable_faiss(self, mock_bm25, mock_faiss):
        """enable_faiss=False should skip FAISS algorithm."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(enable_faiss=False)
        assert "FAISS" not in retriever._algorithms

    @patch(_FAISS_PATCH)
    @patch(_BM25_PATCH)
    def test_invalid_weight_gets_default(self, mock_bm25, mock_faiss):
        """Invalid weight values should be replaced with 1.0."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(algorithm_weights={"FAISS": -1.0, "BM25+": 0.5})
        assert retriever.algorithm_weights["FAISS"] == 1.0
        assert retriever.algorithm_weights["BM25+"] == 0.5


class TestHybridRetrieverWeights:
    """Tests for weight management."""

    @patch(_FAISS_PATCH)
    @patch(_BM25_PATCH)
    def test_update_weights(self, mock_bm25, mock_faiss):
        """update_weights should update both internal state and algorithms."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()
        retriever.update_weights({"FAISS": 0.9, "BM25+": 0.1})

        assert retriever.algorithm_weights["FAISS"] == 0.9
        assert retriever.algorithm_weights["BM25+"] == 0.1


class TestHybridRetrieverIndexed:
    """Tests for indexing state."""

    @patch(_FAISS_PATCH)
    @patch(_BM25_PATCH)
    def test_is_indexed_false_initially(self, mock_bm25, mock_faiss):
        """is_indexed should be False before indexing."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        mock_bm25.return_value.is_indexed = False
        mock_faiss.return_value.is_indexed = False
        mock_bm25.return_value.enabled = True
        mock_faiss.return_value.enabled = True
        retriever = HybridRetriever()
        assert retriever.is_indexed is False

    @patch(_FAISS_PATCH)
    @patch(_BM25_PATCH)
    def test_get_chunk_count(self, mock_bm25, mock_faiss):
        """get_chunk_count should return the length of _chunks."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()
        retriever._chunks = _make_doc_chunks(5)
        assert retriever.get_chunk_count() == 5


class TestHybridRetrieverAlgorithmStatus:
    """Tests for algorithm status reporting."""

    @patch(_FAISS_PATCH)
    @patch(_BM25_PATCH)
    def test_get_algorithm_status(self, mock_bm25, mock_faiss):
        """get_algorithm_status should return status for each algorithm."""
        from src.core.retrieval.hybrid_retriever import HybridRetriever

        mock_bm25.return_value.enabled = True
        mock_bm25.return_value.is_indexed = False
        mock_bm25.return_value.weight = 0.2
        mock_bm25.return_value.get_config.return_value = {}

        mock_faiss.return_value.enabled = True
        mock_faiss.return_value.is_indexed = False
        mock_faiss.return_value.weight = 0.8
        mock_faiss.return_value.get_config.return_value = {}

        retriever = HybridRetriever()
        status = retriever.get_algorithm_status()

        assert "BM25+" in status
        assert "FAISS" in status
        assert status["BM25+"]["enabled"] is True
