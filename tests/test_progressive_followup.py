"""
Tests for progressive follow-up display (retrieval before generation).

Session 87: Verifies the split retrieval/generation flow
and backward compatibility.
"""

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestPlaceholderConstants:
    """Test that placeholder text constants exist and are non-empty."""

    def test_pending_retrieval_text(self):
        from src.core.semantic.semantic_constants import PENDING_RETRIEVAL_TEXT

        assert PENDING_RETRIEVAL_TEXT
        assert isinstance(PENDING_RETRIEVAL_TEXT, str)

    def test_pending_generation_text(self):
        from src.core.semantic.semantic_constants import PENDING_GENERATION_TEXT

        assert PENDING_GENERATION_TEXT
        assert isinstance(PENDING_GENERATION_TEXT, str)

    def test_ollama_unavailable_text_removed(self):
        """OLLAMA_UNAVAILABLE_TEXT was removed (Ollama integration removed Mar 2026)."""
        import src.core.semantic.semantic_constants as consts

        assert not hasattr(consts, "OLLAMA_UNAVAILABLE_TEXT")


# ---------------------------------------------------------------------------
# SemanticOrchestrator split methods
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_orchestrator():
    """Create a SemanticOrchestrator with mocked retriever."""
    with patch("src.core.semantic.semantic_orchestrator.SemanticRetriever") as MockRetriever:
        from src.core.semantic.semantic_orchestrator import SemanticOrchestrator

        mock_retriever = MockRetriever.return_value

        # Default: retriever returns good context
        mock_source = MagicMock()
        mock_source.relevance_score = 0.8
        mock_retrieval = MagicMock()
        mock_retrieval.context = "The plaintiff John Smith filed the case."
        mock_retrieval.sources = [mock_source]
        mock_retrieval.chunks_retrieved = 1
        mock_retrieval.retrieval_time_ms = 42.0
        mock_retriever.retrieve_context.return_value = mock_retrieval
        mock_retriever.get_relevant_sources_summary.return_value = "complaint.pdf, page 1"

        # Mock embeddings for citation excerpt
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 384

        orch = SemanticOrchestrator(
            vector_store_path="fake_path",
            embeddings=mock_embeddings,
            answer_mode="extraction",
        )

        yield orch, mock_retriever, None


class TestRetrieveForQuestion:
    """Test SemanticOrchestrator.retrieve_for_question() (phase 1)."""

    def test_returns_result_with_empty_answer(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator

        result = orch.retrieve_for_question("Who is the plaintiff?")

        assert result.quick_answer == ""
        assert result.is_followup is True

    def test_citation_is_populated(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator

        result = orch.retrieve_for_question("Who is the plaintiff?")

        assert result.citation  # Non-empty
        assert len(result.citation) > 0

    def test_relevance_is_populated(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator

        result = orch.retrieve_for_question("Who is the plaintiff?")

        assert result.relevance > 0

    def test_source_summary_is_populated(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator

        result = orch.retrieve_for_question("Who is the plaintiff?")

        assert result.source_summary == "complaint.pdf, page 1"

    def test_low_quality_retrieval_returns_unanswered(self, mock_orchestrator):
        """When retrieval quality is below gate, return unanswered immediately."""
        orch, mock_retriever, _ = mock_orchestrator
        from src.core.semantic.semantic_constants import UNANSWERED_TEXT

        # Set low relevance score
        mock_source = MagicMock()
        mock_source.relevance_score = 0.01
        mock_retrieval = mock_retriever.retrieve_context.return_value
        mock_retrieval.sources = [mock_source]

        result = orch.retrieve_for_question("Something irrelevant?")

        assert result.quick_answer == UNANSWERED_TEXT
        assert result.relevance == 0.0


class TestGenerateAnswerForResult:
    """Test SemanticOrchestrator.generate_answer_for_result() is a no-op."""

    def test_returns_result_unchanged(self, mock_orchestrator):
        """generate_answer_for_result is a no-op (answer generation removed)."""
        orch, _, _ = mock_orchestrator

        partial = orch.retrieve_for_question("Who is the plaintiff?")
        final = orch.generate_answer_for_result(partial)

        assert final is partial  # Same object, unchanged

    def test_unanswered_returns_as_is(self, mock_orchestrator):
        """Unanswered results pass through unchanged."""
        orch, _, _ = mock_orchestrator
        from src.core.semantic.semantic_constants import UNANSWERED_TEXT
        from src.core.semantic.semantic_orchestrator import SemanticResult

        result = SemanticResult(question="test", quick_answer=UNANSWERED_TEXT)

        returned = orch.generate_answer_for_result(result)

        assert returned.quick_answer == UNANSWERED_TEXT


# ---------------------------------------------------------------------------
# Answer generator: Ollama unavailable
# ---------------------------------------------------------------------------


# TestOllamaUnavailable removed — Ollama integration deprecated


# ---------------------------------------------------------------------------
# SemanticService passthrough methods
# ---------------------------------------------------------------------------


class TestSemanticServicePassthrough:
    """Test SemanticService methods that expose the split flow."""

    def test_retrieve_for_followup_calls_orchestrator(self):
        from src.services.semantic_service import SemanticService

        svc = SemanticService()
        mock_orch = MagicMock()
        mock_orch.retrieve_for_question.return_value = "partial_result"

        result = svc.retrieve_for_followup(mock_orch, "test question")

        mock_orch.retrieve_for_question.assert_called_once_with("test question", is_followup=True)
        assert result == "partial_result"

    def test_generate_answer_for_followup_calls_orchestrator(self):
        from src.services.semantic_service import SemanticService

        svc = SemanticService()
        mock_orch = MagicMock()
        mock_orch.generate_answer_for_result.return_value = "final_result"

        result = svc.generate_answer_for_followup(mock_orch, "partial")

        mock_orch.generate_answer_for_result.assert_called_once_with("partial")
        assert result == "final_result"

    def test_get_placeholder_texts_returns_all_keys(self):
        from src.services.semantic_service import SemanticService

        svc = SemanticService()
        texts = svc.get_placeholder_texts()

        assert "retrieval" in texts
        assert "generation" in texts
        assert all(isinstance(v, str) for v in texts.values())


# ---------------------------------------------------------------------------
# Backward compatibility: _ask_single_question unchanged
# ---------------------------------------------------------------------------


class TestAskSingleQuestionUnchanged:
    """Verify _ask_single_question still works (answer generation removed)."""

    def test_ask_single_question_returns_result(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator

        result = orch._ask_single_question("What is the answer?", is_followup=True)

        assert result.quick_answer == ""  # No answer generation
        assert result.citation  # populated
        assert result.is_followup is True

    def test_ask_followup_still_works(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator

        result = orch.ask_followup("Follow-up question?")

        assert result.quick_answer == ""  # No answer generation
        assert result in orch.results


# ---------------------------------------------------------------------------
# Follow-up thread queue message sequence
# ---------------------------------------------------------------------------


class TestFollowupThreadMessages:
    """Test that the follow-up thread sends correct queue messages."""

    def test_successful_flow_sends_retrieval_then_success(self):
        """Thread should send retrieval_done then success."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        partial = SemanticResult(
            question="test",
            quick_answer="",
            citation="Some citation",
            is_followup=True,
        )

        result_queue = queue.Queue()

        with (
            patch("src.services.SemanticService") as MockSvc,
            patch("src.user_preferences.get_user_preferences") as MockPrefs,
        ):
            mock_svc = MockSvc.return_value
            mock_svc.create_orchestrator.return_value = MagicMock()
            mock_svc.retrieve_for_followup.return_value = partial
            mock_svc.generate_answer_for_followup.return_value = partial
            MockPrefs.return_value.get.return_value = "extraction"

            # Simulate the followup thread body from main_window
            def run():
                from src.services import SemanticService
                from src.user_preferences import get_user_preferences

                semantic_service = SemanticService()
                prefs = get_user_preferences()
                orchestrator = semantic_service.create_orchestrator(
                    vector_store_path="fake",
                    embeddings=MagicMock(),
                    answer_mode=prefs.get("semantic_answer_mode", "extraction"),
                )
                p = semantic_service.retrieve_for_followup(orchestrator, "test")
                result_queue.put(("retrieval_done", p))

                f = semantic_service.generate_answer_for_followup(orchestrator, p)
                result_queue.put(("success", f))

            t = threading.Thread(target=run)
            t.start()
            t.join(timeout=5)

            messages = []
            while not result_queue.empty():
                messages.append(result_queue.get_nowait())

            assert len(messages) == 2
            assert messages[0][0] == "retrieval_done"
            assert messages[1][0] == "success"
