"""
Tests for SemanticService.

Covers the service layer that orchestrates semantic search operations.
Wraps SemanticOrchestrator, VectorStoreBuilder, and FAISS retrieval
for use by the UI layer (which cannot import from src.core directly).

Tests:
- Initialization and is_ready property
- build_index: progress callbacks, success and failure paths, embedding reuse
- ask_question / run_default_questions guards when index not built
- Delegation to orchestrator for all query and export methods
- clear() and cleanup() lifecycle management
- Factory helpers: create_orchestrator, get_vector_store_builder,
  get_default_questions_manager, get_semantic_result_class,
  get_placeholder_texts
"""

from unittest.mock import MagicMock, patch


class TestSemanticServiceInit:
    """Tests for SemanticService initialization."""

    def test_default_initialization(self):
        """Service starts with is_ready=False and no orchestrator or embeddings."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()

        assert service.is_ready is False
        assert service._orchestrator is None
        assert service._embeddings is None
        assert service._temp_dir is None

    def test_custom_path_stored(self, tmp_path):
        """Provided vector_store_path is stored on the instance."""
        from src.services.semantic_service import SemanticService

        service = SemanticService(vector_store_path=tmp_path)

        assert service._vector_store_path == tmp_path

    def test_is_ready_property_reflects_internal_flag(self):
        """is_ready is a read-only view of _is_ready."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        assert service.is_ready is False

        service._is_ready = True
        assert service.is_ready is True


class TestSemanticServiceBuildIndex:
    """Tests for build_index() flow (ML components mocked)."""

    def _make_mock_chunk(self, text="Sample legal text for indexing."):
        """Return a minimal mock UnifiedChunk."""
        chunk = MagicMock()
        chunk.text = text
        return chunk

    @patch("src.core.semantic.SemanticOrchestrator")
    @patch("src.core.vector_store.VectorStoreBuilder")
    @patch("src.core.chunking.create_unified_chunker")
    @patch("src.core.retrieval.algorithms.faiss_semantic.get_embeddings_model")
    def test_build_index_returns_true_on_success(
        self, mock_get_emb, mock_create_chunker, mock_builder_cls, mock_orch_cls, tmp_path
    ):
        """build_index returns True and sets is_ready=True on success."""
        from src.services.semantic_service import SemanticService

        mock_get_emb.return_value = MagicMock()
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.return_value = [self._make_mock_chunk()]
        mock_create_chunker.return_value = mock_chunker
        mock_builder_cls.return_value = MagicMock()
        mock_orch_cls.return_value = MagicMock()

        service = SemanticService(vector_store_path=tmp_path)
        result = service.build_index("Document text.")

        assert result is True
        assert service.is_ready is True

    @patch("src.core.semantic.SemanticOrchestrator")
    @patch("src.core.vector_store.VectorStoreBuilder")
    @patch("src.core.chunking.create_unified_chunker")
    @patch("src.core.retrieval.algorithms.faiss_semantic.get_embeddings_model")
    def test_build_index_calls_progress_callback(
        self, mock_get_emb, mock_create_chunker, mock_builder_cls, mock_orch_cls, tmp_path
    ):
        """build_index calls progress_callback at each stage."""
        from src.services.semantic_service import SemanticService

        mock_get_emb.return_value = MagicMock()
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.return_value = [self._make_mock_chunk()]
        mock_create_chunker.return_value = mock_chunker
        mock_builder_cls.return_value = MagicMock()
        mock_orch_cls.return_value = MagicMock()

        status_msgs = []
        service = SemanticService(vector_store_path=tmp_path)
        service.build_index("Text.", progress_callback=status_msgs.append)

        assert len(status_msgs) >= 3
        # At minimum: model loading, chunking, and ready messages
        assert any("embeddings" in m.lower() or "model" in m.lower() for m in status_msgs)
        assert any("chunk" in m.lower() for m in status_msgs)

    @patch("src.core.retrieval.algorithms.faiss_semantic.get_embeddings_model")
    def test_build_index_returns_false_on_exception(self, mock_get_emb, tmp_path):
        """build_index returns False and leaves is_ready=False when an exception occurs."""
        from src.services.semantic_service import SemanticService

        mock_get_emb.side_effect = RuntimeError("Embedding model unavailable")

        service = SemanticService(vector_store_path=tmp_path)
        result = service.build_index("Some text.")

        assert result is False
        assert service.is_ready is False

    @patch("src.core.semantic.SemanticOrchestrator")
    @patch("src.core.vector_store.VectorStoreBuilder")
    @patch("src.core.chunking.create_unified_chunker")
    @patch("src.core.retrieval.algorithms.faiss_semantic.get_embeddings_model")
    def test_build_index_reuses_cached_embeddings(
        self, mock_get_emb, mock_create_chunker, mock_builder_cls, mock_orch_cls, tmp_path
    ):
        """build_index loads embeddings only once — subsequent calls reuse the cache."""
        from src.services.semantic_service import SemanticService

        mock_get_emb.return_value = MagicMock()
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.return_value = [self._make_mock_chunk()]
        mock_create_chunker.return_value = mock_chunker
        mock_builder_cls.return_value = MagicMock()
        mock_orch_cls.return_value = MagicMock()

        service = SemanticService(vector_store_path=tmp_path)
        service.build_index("First document.")
        service.build_index("Second document.")

        # get_embeddings_model is called only once across both builds
        assert mock_get_emb.call_count == 1

    @patch("src.core.semantic.SemanticOrchestrator")
    @patch("src.core.vector_store.VectorStoreBuilder")
    @patch("src.core.chunking.create_unified_chunker")
    @patch("src.core.retrieval.algorithms.faiss_semantic.get_embeddings_model")
    def test_build_index_creates_orchestrator(
        self, mock_get_emb, mock_create_chunker, mock_builder_cls, mock_orch_cls, tmp_path
    ):
        """build_index constructs a SemanticOrchestrator after building the index."""
        from src.services.semantic_service import SemanticService

        mock_get_emb.return_value = MagicMock()
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.return_value = [self._make_mock_chunk()]
        mock_create_chunker.return_value = mock_chunker
        mock_builder_cls.return_value = MagicMock()
        mock_orch_cls.return_value = MagicMock()

        service = SemanticService(vector_store_path=tmp_path)
        service.build_index("Text.")

        mock_orch_cls.assert_called_once()
        assert service._orchestrator is not None


class TestSemanticServiceNotReady:
    """Tests for guard behavior when the index has not been built."""

    def test_ask_question_returns_none_when_not_ready(self):
        """ask_question returns None when index not built."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        assert service.ask_question("Who are the plaintiffs?") is None

    def test_run_default_questions_returns_empty_when_not_ready(self):
        """run_default_questions returns [] when index not built."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        assert service.run_default_questions() == []

    def test_get_results_returns_empty_when_no_orchestrator(self):
        """get_results returns [] when orchestrator not initialized."""
        from src.services.semantic_service import SemanticService

        assert SemanticService().get_results() == []

    def test_get_exportable_results_returns_empty_when_no_orchestrator(self):
        """get_exportable_results returns [] when orchestrator not initialized."""
        from src.services.semantic_service import SemanticService

        assert SemanticService().get_exportable_results() == []

    def test_export_to_text_returns_empty_string_when_no_orchestrator(self):
        """export_to_text returns '' when orchestrator not initialized."""
        from src.services.semantic_service import SemanticService

        assert SemanticService().export_to_text() == ""

    def test_export_to_csv_returns_empty_string_when_no_orchestrator(self):
        """export_to_csv returns '' when orchestrator not initialized."""
        from src.services.semantic_service import SemanticService

        assert SemanticService().export_to_csv() == ""

    def test_toggle_export_returns_false_when_no_orchestrator(self):
        """toggle_export returns False when orchestrator not initialized."""
        from src.services.semantic_service import SemanticService

        assert SemanticService().toggle_export(0) is False


class TestSemanticServiceDelegation:
    """Tests that service methods delegate correctly to the orchestrator."""

    def _service_with_orchestrator(self):
        """Create a SemanticService with a mock orchestrator already attached."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        service._orchestrator = MagicMock()
        service._is_ready = True
        return service

    def test_ask_question_delegates_to_orchestrator_ask_followup(self):
        """ask_question forwards to orchestrator.ask_followup."""
        service = self._service_with_orchestrator()
        service.ask_question("Who is the defendant?")
        service._orchestrator.ask_followup.assert_called_once_with("Who is the defendant?")

    def test_run_default_questions_delegates_to_orchestrator(self):
        """run_default_questions forwards to orchestrator.run_default_questions."""
        service = self._service_with_orchestrator()
        service.run_default_questions()
        service._orchestrator.run_default_questions.assert_called_once()

    def test_run_default_questions_passes_progress_callback(self):
        """run_default_questions passes the callback through to orchestrator."""
        service = self._service_with_orchestrator()
        cb = MagicMock()
        service.run_default_questions(progress_callback=cb)
        service._orchestrator.run_default_questions.assert_called_once_with(cb)

    def test_toggle_export_delegates_with_index(self):
        """toggle_export forwards the index to orchestrator.toggle_export."""
        service = self._service_with_orchestrator()
        service.toggle_export(2)
        service._orchestrator.toggle_export.assert_called_once_with(2)

    def test_get_results_returns_orchestrator_results(self):
        """get_results returns whatever orchestrator.results holds."""
        service = self._service_with_orchestrator()
        expected = [MagicMock(), MagicMock()]
        service._orchestrator.results = expected
        assert service.get_results() == expected

    def test_get_exportable_results_delegates(self):
        """get_exportable_results delegates to orchestrator."""
        service = self._service_with_orchestrator()
        service._orchestrator.get_exportable_results.return_value = []
        service.get_exportable_results()
        service._orchestrator.get_exportable_results.assert_called_once()

    def test_export_to_text_delegates_and_returns_result(self):
        """export_to_text calls orchestrator.export_to_text and returns its result."""
        service = self._service_with_orchestrator()
        service._orchestrator.export_to_text.return_value = "Q1: ...\nA1: ..."
        result = service.export_to_text()
        assert result == "Q1: ...\nA1: ..."

    def test_export_to_csv_delegates_and_returns_result(self):
        """export_to_csv calls orchestrator.export_to_csv and returns its result."""
        service = self._service_with_orchestrator()
        service._orchestrator.export_to_csv.return_value = "question,answer\n"
        result = service.export_to_csv()
        assert result == "question,answer\n"

    def test_get_default_questions_uses_orchestrator_when_ready(self):
        """get_default_questions delegates to orchestrator when available."""
        service = self._service_with_orchestrator()
        service._orchestrator.get_default_questions.return_value = ["Q1", "Q2"]
        result = service.get_default_questions()
        assert result == ["Q1", "Q2"]
        service._orchestrator.get_default_questions.assert_called_once()


class TestSemanticServiceLifecycle:
    """Tests for clear() and cleanup() lifecycle management."""

    def test_clear_sets_is_ready_false(self):
        """clear() sets is_ready back to False."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        service._is_ready = True
        service._orchestrator = MagicMock()
        service.clear()

        assert service.is_ready is False

    def test_clear_calls_orchestrator_clear_results(self):
        """clear() calls clear_results() on the orchestrator."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        service._orchestrator = MagicMock()
        service.clear()

        service._orchestrator.clear_results.assert_called_once()

    def test_clear_is_safe_when_no_orchestrator(self):
        """clear() does not raise when orchestrator is None."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        service.clear()  # Should not raise

    def test_cleanup_removes_temp_directory(self, tmp_path):
        """cleanup() deletes the temp dir that was created during build_index."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        service._temp_dir = tmp_path
        service._vector_store_path = tmp_path / "index"

        assert tmp_path.exists()
        service.cleanup()
        assert not tmp_path.exists()

    def test_cleanup_sets_temp_dir_to_none(self, tmp_path):
        """cleanup() clears the _temp_dir reference after deletion."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        service._temp_dir = tmp_path
        service.cleanup()

        assert service._temp_dir is None

    def test_cleanup_is_noop_when_no_temp_dir(self):
        """cleanup() does nothing when _temp_dir is None."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        service._temp_dir = None
        service.cleanup()  # Should not raise

    def test_cleanup_is_noop_when_dir_already_deleted(self, tmp_path):
        """cleanup() handles a temp_dir that was already removed."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        phantom = tmp_path / "already_gone"
        service._temp_dir = phantom
        service.cleanup()  # Should not raise


class TestSemanticServiceHelpers:
    """Tests for factory and helper methods on SemanticService."""

    def test_get_placeholder_texts_returns_dict_with_required_keys(self):
        """get_placeholder_texts has 'retrieval' and 'generation' keys."""
        from src.services.semantic_service import SemanticService

        texts = SemanticService().get_placeholder_texts()

        assert "retrieval" in texts
        assert "generation" in texts
        assert isinstance(texts["retrieval"], str)
        assert isinstance(texts["generation"], str)

    def test_get_semantic_result_class_returns_correct_class(self):
        """get_semantic_result_class returns the SemanticResult class itself."""
        from src.core.semantic.semantic_orchestrator import SemanticResult
        from src.services.semantic_service import SemanticService

        result_cls = SemanticService().get_semantic_result_class()
        assert result_cls is SemanticResult

    def test_get_vector_store_builder_returns_builder_instance(self):
        """get_vector_store_builder returns a VectorStoreBuilder instance."""
        from src.core.vector_store import VectorStoreBuilder
        from src.services.semantic_service import SemanticService

        builder = SemanticService().get_vector_store_builder()
        assert isinstance(builder, VectorStoreBuilder)

    def test_get_default_questions_manager_returns_manager_instance(self):
        """get_default_questions_manager returns a DefaultQuestionsManager."""
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager
        from src.services.semantic_service import SemanticService

        manager = SemanticService().get_default_questions_manager()
        assert isinstance(manager, DefaultQuestionsManager)

    def test_create_orchestrator_returns_orchestrator(self, tmp_path):
        """create_orchestrator instantiates and returns a SemanticOrchestrator."""
        from src.services.semantic_service import SemanticService

        with patch("src.core.semantic.SemanticOrchestrator") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = SemanticService().create_orchestrator(
                vector_store_path=tmp_path, embeddings=MagicMock()
            )
            mock_cls.assert_called_once()
            assert result is mock_cls.return_value

    def test_create_orchestrator_ignores_legacy_kwargs(self, tmp_path):
        """create_orchestrator absorbs unknown kwargs without raising."""
        from src.services.semantic_service import SemanticService

        with patch("src.core.semantic.SemanticOrchestrator") as mock_cls:
            mock_cls.return_value = MagicMock()
            SemanticService().create_orchestrator(
                vector_store_path=tmp_path,
                embeddings=MagicMock(),
                answer_mode="extraction",  # legacy param, should be ignored
            )
            mock_cls.assert_called_once()

    def test_retrieve_for_followup_delegates(self, tmp_path):
        """retrieve_for_followup calls orchestrator.retrieve_for_question."""
        from src.services.semantic_service import SemanticService

        mock_orch = MagicMock()
        mock_orch.retrieve_for_question.return_value = MagicMock()

        SemanticService().retrieve_for_followup(mock_orch, "Any question?")
        mock_orch.retrieve_for_question.assert_called_once_with("Any question?", is_followup=True)

    def test_generate_answer_for_followup_delegates(self):
        """generate_answer_for_followup calls orchestrator.generate_answer_for_result."""
        from src.services.semantic_service import SemanticService

        mock_orch = MagicMock()
        mock_result = MagicMock()

        SemanticService().generate_answer_for_followup(mock_orch, mock_result)
        mock_orch.generate_answer_for_result.assert_called_once_with(mock_result)
