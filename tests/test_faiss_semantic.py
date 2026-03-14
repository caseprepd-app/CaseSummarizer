"""
Tests for the FAISS semantic retrieval algorithm.

Covers the FAISSRetriever and its supporting classes without loading
the actual 137M-parameter embedding model (all ML calls are mocked).

Tests:
- DocumentChunk auto-computes word_count from text
- RetrievedChunk clamps relevance_score to [0, 1]
- FAISSRetriever class attributes (name, weight, enabled)
- FAISSRetriever.is_indexed property: False before indexing
- FAISSRetriever.index_documents: raises ValueError on empty input
- FAISSRetriever.retrieve: raises RuntimeError when called before indexing
- FAISSRetriever.get_config: returns dict with required keys
- FAISSRetriever.set_embeddings: stores the model
- _get_embedding_device: returns 'cpu' when CUDA is unavailable
"""

from unittest.mock import MagicMock, patch


class TestDocumentChunk:
    """Tests for the DocumentChunk dataclass (shared with retrieval base)."""

    def test_auto_computes_word_count(self):
        """word_count is computed from text when not explicitly provided."""
        from src.core.retrieval.base import DocumentChunk

        chunk = DocumentChunk(
            text="The plaintiff filed a motion.",
            chunk_id="doc_0",
            filename="complaint.pdf",
        )

        assert chunk.word_count == 5  # "The plaintiff filed a motion." = 5 words

    def test_explicit_word_count_preserved(self):
        """Explicitly provided word_count is not overwritten."""
        from src.core.retrieval.base import DocumentChunk

        chunk = DocumentChunk(
            text="Five words right here now.",
            chunk_id="doc_0",
            filename="doc.pdf",
            word_count=99,  # explicitly set
        )

        assert chunk.word_count == 99

    def test_empty_text_word_count_is_zero(self):
        """Empty text produces word_count=0."""
        from src.core.retrieval.base import DocumentChunk

        chunk = DocumentChunk(text="", chunk_id="id", filename="f.pdf")
        assert chunk.word_count == 0


class TestRetrievedChunk:
    """Tests for the RetrievedChunk dataclass (shared with retrieval base)."""

    def test_relevance_score_clamped_to_zero(self):
        """Negative relevance_score is clamped to 0.0."""
        from src.core.retrieval.base import RetrievedChunk

        chunk = RetrievedChunk(
            chunk_id="id",
            text="Text",
            relevance_score=-0.5,
            raw_score=-0.5,
            source_algorithm="FAISS",
            filename="doc.pdf",
        )

        assert chunk.relevance_score == 0.0

    def test_relevance_score_clamped_to_one(self):
        """relevance_score above 1.0 is clamped to 1.0."""
        from src.core.retrieval.base import RetrievedChunk

        chunk = RetrievedChunk(
            chunk_id="id",
            text="Text",
            relevance_score=1.5,
            raw_score=1.5,
            source_algorithm="FAISS",
            filename="doc.pdf",
        )

        assert chunk.relevance_score == 1.0

    def test_valid_score_unchanged(self):
        """relevance_score in [0, 1] is stored as-is."""
        from src.core.retrieval.base import RetrievedChunk

        chunk = RetrievedChunk(
            chunk_id="id",
            text="Text",
            relevance_score=0.75,
            raw_score=0.75,
            source_algorithm="FAISS",
            filename="doc.pdf",
        )

        assert chunk.relevance_score == 0.75


class TestFAISSRetrieverClassAttributes:
    """Tests for FAISSRetriever class-level attributes."""

    def test_name_is_faiss(self):
        """FAISSRetriever.name is 'FAISS'."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        assert FAISSRetriever.name == "FAISS"

    def test_weight_is_primary(self):
        """FAISSRetriever.weight is 0.8 (primary algorithm)."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        assert FAISSRetriever.weight == 0.8

    def test_enabled_by_default(self):
        """FAISSRetriever.enabled is True."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        assert FAISSRetriever.enabled is True


class TestFAISSRetrieverIsIndexed:
    """Tests for FAISSRetriever.is_indexed property."""

    def test_is_indexed_false_before_indexing(self):
        """is_indexed returns False on a freshly created retriever."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever(embeddings=MagicMock())

        assert retriever.is_indexed is False

    def test_is_indexed_requires_both_store_and_chunks(self):
        """is_indexed requires both _vector_store and _chunks to be set."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever(embeddings=MagicMock())
        retriever._vector_store = MagicMock()
        # _chunks is still empty → should still be False
        assert retriever.is_indexed is False


class TestFAISSRetrieverIndexDocuments:
    """Tests for FAISSRetriever.index_documents."""

    def test_raises_on_empty_chunks(self):
        """index_documents raises ValueError when chunks is empty."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever(embeddings=MagicMock())

        import pytest

        with pytest.raises(ValueError, match="Cannot index empty"):
            retriever.index_documents([])

    @patch("langchain_community.vectorstores.FAISS")
    def test_sets_is_indexed_after_successful_indexing(self, mock_faiss_cls):
        """is_indexed is True after index_documents completes."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever
        from src.core.retrieval.base import DocumentChunk

        mock_faiss_cls.from_documents.return_value = MagicMock()

        mock_emb = MagicMock()
        retriever = FAISSRetriever(embeddings=mock_emb)

        chunks = [
            DocumentChunk(
                text="The plaintiff filed the case.",
                chunk_id="doc_0",
                filename="complaint.pdf",
            )
        ]

        retriever.index_documents(chunks)

        assert retriever.is_indexed is True

    @patch("langchain_community.vectorstores.FAISS")
    def test_stores_chunks_for_later_retrieval(self, mock_faiss_cls):
        """index_documents stores the chunk list on the retriever."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever
        from src.core.retrieval.base import DocumentChunk

        mock_faiss_cls.from_documents.return_value = MagicMock()

        chunks = [
            DocumentChunk(text="Chunk A.", chunk_id="id_0", filename="doc.pdf"),
            DocumentChunk(text="Chunk B.", chunk_id="id_1", filename="doc.pdf"),
        ]

        retriever = FAISSRetriever(embeddings=MagicMock())
        retriever.index_documents(chunks)

        assert len(retriever._chunks) == 2

    @patch("langchain_community.vectorstores.FAISS")
    def test_embeddings_kwarg_overrides_instance_embeddings(self, mock_faiss_cls):
        """Passing embeddings= in kwargs overrides self._embeddings."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever
        from src.core.retrieval.base import DocumentChunk

        mock_faiss_cls.from_documents.return_value = MagicMock()

        new_emb = MagicMock()
        retriever = FAISSRetriever(embeddings=MagicMock())
        chunks = [DocumentChunk(text="Text.", chunk_id="id", filename="f.pdf")]

        retriever.index_documents(chunks, embeddings=new_emb)

        assert retriever._embeddings is new_emb


class TestFAISSRetrieverRetrieve:
    """Tests for FAISSRetriever.retrieve."""

    def test_raises_when_not_indexed(self):
        """retrieve raises RuntimeError when index has not been built."""
        import pytest

        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever(embeddings=MagicMock())

        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.retrieve("Who filed?")

    @patch("langchain_community.vectorstores.FAISS")
    def test_retrieve_returns_algorithm_retrieval_result(self, mock_faiss_cls):
        """retrieve returns an AlgorithmRetrievalResult with the expected structure."""
        from langchain_core.documents import Document

        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever
        from src.core.retrieval.base import AlgorithmRetrievalResult, DocumentChunk

        # Set up FAISS mock to return a fake similarity result
        mock_store = MagicMock()
        mock_doc = Document(
            page_content="The defendant denied liability.",
            metadata={
                "chunk_id": "doc_0",
                "filename": "answer.pdf",
                "chunk_num": 0,
                "section_name": "Defenses",
                "word_count": 5,
            },
        )
        mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.85)]
        mock_faiss_cls.from_documents.return_value = mock_store

        retriever = FAISSRetriever(embeddings=MagicMock())
        chunks = [
            DocumentChunk(
                text="The defendant denied liability.", chunk_id="doc_0", filename="answer.pdf"
            )
        ]
        retriever.index_documents(chunks)

        result = retriever.retrieve("Did the defendant deny?", k=5)

        assert isinstance(result, AlgorithmRetrievalResult)
        assert len(result.chunks) == 1
        assert result.chunks[0].filename == "answer.pdf"
        assert result.chunks[0].source_algorithm == "FAISS"

    @patch("langchain_community.vectorstores.FAISS")
    def test_relevance_score_clamped_to_zero_one(self, mock_faiss_cls):
        """Scores from FAISS are clamped to [0, 1] in the result."""
        from langchain_core.documents import Document

        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever
        from src.core.retrieval.base import DocumentChunk

        mock_store = MagicMock()
        mock_doc = Document(
            page_content="Text.",
            metadata={
                "chunk_id": "id",
                "filename": "f.pdf",
                "chunk_num": 0,
                "section_name": "N/A",
                "word_count": 1,
            },
        )
        # Raw FAISS score outside [0,1]
        mock_store.similarity_search_with_score.return_value = [(mock_doc, 1.2)]
        mock_faiss_cls.from_documents.return_value = mock_store

        retriever = FAISSRetriever(embeddings=MagicMock())
        retriever.index_documents([DocumentChunk(text="Text.", chunk_id="id", filename="f.pdf")])

        result = retriever.retrieve("Query?", k=1)

        assert result.chunks[0].relevance_score <= 1.0


class TestFAISSRetrieverGetConfig:
    """Tests for FAISSRetriever.get_config."""

    def test_get_config_contains_required_keys(self):
        """get_config returns a dict with algorithm, index_size, and embedding_model."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever(embeddings=MagicMock())
        config = retriever.get_config()

        assert "index_size" in config
        assert "embedding_model" in config
        assert "distance_metric" in config
        assert config["distance_metric"] == "cosine"

    def test_get_config_index_size_zero_before_indexing(self):
        """index_size is 0 before any documents are indexed."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever(embeddings=MagicMock())
        config = retriever.get_config()

        assert config["index_size"] == 0


class TestFAISSRetrieverSetEmbeddings:
    """Tests for FAISSRetriever.set_embeddings."""

    def test_set_embeddings_stores_model(self):
        """set_embeddings updates self._embeddings to the provided model."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever()
        new_model = MagicMock()
        retriever.set_embeddings(new_model)

        assert retriever._embeddings is new_model


class TestGetEmbeddingDevice:
    """Tests for the _get_embedding_device helper."""

    def test_returns_cpu_when_torch_not_available(self):
        """Returns 'cpu' when torch is not importable."""
        from src.core.retrieval.algorithms.faiss_semantic import _get_embedding_device

        with patch.dict("sys.modules", {"torch": None}):
            device = _get_embedding_device()

        assert device == "cpu"

    def test_returns_cpu_when_cuda_not_available(self):
        """Returns 'cpu' when torch is present but CUDA is not available."""
        from src.core.retrieval.algorithms.faiss_semantic import _get_embedding_device

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            device = _get_embedding_device()

        assert device == "cpu"

    def test_returns_cuda_when_gpu_available(self):
        """Returns 'cuda' when CUDA is available."""
        from src.core.retrieval.algorithms.faiss_semantic import _get_embedding_device

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            device = _get_embedding_device()

        assert device == "cuda"
