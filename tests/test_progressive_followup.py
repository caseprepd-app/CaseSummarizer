"""
Tests for progressive follow-up display (retrieval before generation).

Session 87: Verifies the split retrieval/generation flow, Ollama
unavailable messaging, and backward compatibility.
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
        from src.core.qa.qa_constants import PENDING_RETRIEVAL_TEXT

        assert PENDING_RETRIEVAL_TEXT
        assert isinstance(PENDING_RETRIEVAL_TEXT, str)

    def test_pending_generation_text(self):
        from src.core.qa.qa_constants import PENDING_GENERATION_TEXT

        assert PENDING_GENERATION_TEXT
        assert isinstance(PENDING_GENERATION_TEXT, str)

    def test_ollama_unavailable_text(self):
        from src.core.qa.qa_constants import OLLAMA_UNAVAILABLE_TEXT

        assert OLLAMA_UNAVAILABLE_TEXT
        assert "Ollama" in OLLAMA_UNAVAILABLE_TEXT
        assert "Settings" in OLLAMA_UNAVAILABLE_TEXT


# ---------------------------------------------------------------------------
# QAOrchestrator split methods
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_orchestrator():
    """Create a QAOrchestrator with mocked retriever and answer generator."""
    with (
        patch("src.core.qa.qa_orchestrator.QARetriever") as MockRetriever,
        patch("src.core.qa.answer_generator.AnswerGenerator") as MockGenerator,
    ):
        from src.core.qa.qa_orchestrator import QAOrchestrator

        mock_retriever = MockRetriever.return_value
        mock_generator = MockGenerator.return_value

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

        mock_generator.generate.return_value = "John Smith is the plaintiff."

        # Mock embeddings for citation excerpt
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 384

        orch = QAOrchestrator(
            vector_store_path="fake_path",
            embeddings=mock_embeddings,
            answer_mode="ollama",
        )

        yield orch, mock_retriever, mock_generator


class TestRetrieveForQuestion:
    """Test QAOrchestrator.retrieve_for_question() (phase 1)."""

    def test_returns_partial_result_with_placeholder_answer(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator
        from src.core.qa.qa_constants import PENDING_GENERATION_TEXT

        result = orch.retrieve_for_question("Who is the plaintiff?")

        assert result.quick_answer == PENDING_GENERATION_TEXT
        assert result.is_followup is True

    def test_citation_is_populated(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator

        result = orch.retrieve_for_question("Who is the plaintiff?")

        assert result.citation  # Non-empty
        assert len(result.citation) > 0

    def test_confidence_is_populated(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator

        result = orch.retrieve_for_question("Who is the plaintiff?")

        assert result.confidence > 0

    def test_source_summary_is_populated(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator

        result = orch.retrieve_for_question("Who is the plaintiff?")

        assert result.source_summary == "complaint.pdf, page 1"

    def test_stashes_retrieval_context(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator

        result = orch.retrieve_for_question("Who is the plaintiff?")

        assert hasattr(result, "_retrieval_context")
        assert result._retrieval_context is not None

    def test_low_quality_retrieval_returns_unanswered(self, mock_orchestrator):
        """When retrieval quality is below gate, return unanswered immediately."""
        orch, mock_retriever, _ = mock_orchestrator
        from src.core.qa.qa_constants import UNANSWERED_TEXT

        # Set low relevance score
        mock_source = MagicMock()
        mock_source.relevance_score = 0.01
        mock_retrieval = mock_retriever.retrieve_context.return_value
        mock_retrieval.sources = [mock_source]

        result = orch.retrieve_for_question("Something irrelevant?")

        assert result.quick_answer == UNANSWERED_TEXT
        assert result._retrieval_context is None
        assert result.confidence == 0.0


class TestGenerateAnswerForResult:
    """Test QAOrchestrator.generate_answer_for_result() (phase 2)."""

    def test_fills_in_quick_answer(self, mock_orchestrator):
        orch, _, mock_gen = mock_orchestrator
        mock_gen.generate.return_value = "John Smith is the plaintiff."

        partial = orch.retrieve_for_question("Who is the plaintiff?")
        final = orch.generate_answer_for_result(partial)

        assert final.quick_answer == "John Smith is the plaintiff."

    def test_cleans_up_retrieval_context(self, mock_orchestrator):
        orch, _, _ = mock_orchestrator

        partial = orch.retrieve_for_question("Who is the plaintiff?")
        assert hasattr(partial, "_retrieval_context")

        final = orch.generate_answer_for_result(partial)

        assert not hasattr(final, "_retrieval_context")

    def test_no_context_returns_as_is(self, mock_orchestrator):
        """If _retrieval_context is None, return result unchanged."""
        orch, _, _ = mock_orchestrator
        from src.core.qa.qa_constants import UNANSWERED_TEXT
        from src.core.qa.qa_orchestrator import QAResult

        result = QAResult(question="test", quick_answer=UNANSWERED_TEXT)
        result._retrieval_context = None

        returned = orch.generate_answer_for_result(result)

        assert returned.quick_answer == UNANSWERED_TEXT


# ---------------------------------------------------------------------------
# Answer generator: Ollama unavailable
# ---------------------------------------------------------------------------


class TestOllamaUnavailable:
    """Test that Ollama disconnected returns clear message, not extraction."""

    def test_returns_unavailable_text_not_extraction(self):
        from src.core.qa.answer_generator import AnswerGenerator
        from src.core.qa.qa_constants import OLLAMA_UNAVAILABLE_TEXT

        gen = AnswerGenerator(mode="ollama")
        gen._ollama_manager = MagicMock()
        gen._ollama_manager.is_connected = False

        result = gen.generate("Who is the plaintiff?", "Some context about John Smith.")

        assert result == OLLAMA_UNAVAILABLE_TEXT

    def test_generation_error_still_falls_back_to_extraction(self):
        """When Ollama IS connected but errors out, extraction fallback still works."""
        from src.core.qa.answer_generator import AnswerGenerator
        from src.core.qa.qa_constants import OLLAMA_UNAVAILABLE_TEXT

        gen = AnswerGenerator(mode="ollama")
        gen._ollama_manager = MagicMock()
        gen._ollama_manager.is_connected = True
        gen._ollama_manager.generate_text.side_effect = RuntimeError("connection lost")

        result = gen.generate("Who is the plaintiff?", "John Smith filed the lawsuit.")

        # Should NOT be the unavailable text -- should be extraction fallback
        assert result != OLLAMA_UNAVAILABLE_TEXT
        assert len(result) > 0


# ---------------------------------------------------------------------------
# QAService passthrough methods
# ---------------------------------------------------------------------------


class TestQAServicePassthrough:
    """Test QAService methods that expose the split flow."""

    def test_retrieve_for_followup_calls_orchestrator(self):
        from src.services.qa_service import QAService

        svc = QAService()
        mock_orch = MagicMock()
        mock_orch.retrieve_for_question.return_value = "partial_result"

        result = svc.retrieve_for_followup(mock_orch, "test question")

        mock_orch.retrieve_for_question.assert_called_once_with("test question", is_followup=True)
        assert result == "partial_result"

    def test_generate_answer_for_followup_calls_orchestrator(self):
        from src.services.qa_service import QAService

        svc = QAService()
        mock_orch = MagicMock()
        mock_orch.generate_answer_for_result.return_value = "final_result"

        result = svc.generate_answer_for_followup(mock_orch, "partial")

        mock_orch.generate_answer_for_result.assert_called_once_with("partial")
        assert result == "final_result"

    def test_get_placeholder_texts_returns_all_keys(self):
        from src.services.qa_service import QAService

        svc = QAService()
        texts = svc.get_placeholder_texts()

        assert "retrieval" in texts
        assert "generation" in texts
        assert "ollama_unavailable" in texts
        assert all(isinstance(v, str) for v in texts.values())

    def test_is_ollama_connected_returns_bool(self):
        from src.services.qa_service import QAService

        svc = QAService()
        mock_mgr = MagicMock()
        mock_mgr.is_connected = False
        with patch("src.services.ai_service.AIService.get_ollama_manager", return_value=mock_mgr):
            assert svc.is_ollama_connected() is False


# ---------------------------------------------------------------------------
# Backward compatibility: _ask_single_question unchanged
# ---------------------------------------------------------------------------


class TestAskSingleQuestionUnchanged:
    """Verify _ask_single_question still works as before (not broken)."""

    def test_ask_single_question_returns_full_result(self, mock_orchestrator):
        orch, _, mock_gen = mock_orchestrator
        mock_gen.generate.return_value = "The answer is 42."

        result = orch._ask_single_question("What is the answer?", is_followup=True)

        assert result.quick_answer == "The answer is 42."
        assert result.citation  # populated
        assert result.is_followup is True

    def test_ask_followup_still_works(self, mock_orchestrator):
        orch, _, mock_gen = mock_orchestrator
        mock_gen.generate.return_value = "Follow-up answer."

        result = orch.ask_followup("Follow-up question?")

        assert result.quick_answer == "Follow-up answer."
        assert result in orch.results


# ---------------------------------------------------------------------------
# Follow-up thread queue message sequence
# ---------------------------------------------------------------------------


class TestFollowupThreadMessages:
    """Test that the follow-up thread sends correct queue messages."""

    def test_successful_flow_sends_retrieval_then_success(self):
        """Thread should send retrieval_done then success."""
        from src.core.qa.qa_constants import PENDING_GENERATION_TEXT
        from src.core.qa.qa_orchestrator import QAResult

        partial = QAResult(
            question="test",
            quick_answer=PENDING_GENERATION_TEXT,
            citation="Some citation",
            is_followup=True,
        )
        partial._retrieval_context = "raw context"

        final = QAResult(
            question="test",
            quick_answer="The real answer.",
            citation="Some citation",
            is_followup=True,
        )

        result_queue = queue.Queue()

        with (
            patch("src.services.QAService") as MockSvc,
            patch("src.user_preferences.get_user_preferences") as MockPrefs,
        ):
            mock_svc = MockSvc.return_value
            mock_svc.create_orchestrator.return_value = MagicMock()
            mock_svc.retrieve_for_followup.return_value = partial
            mock_svc.generate_answer_for_followup.return_value = final
            mock_svc.get_placeholder_texts.return_value = {"ollama_unavailable": "unavail"}
            MockPrefs.return_value.get.return_value = "ollama"

            # Simulate the followup thread body from main_window
            def run():
                from src.services import QAService
                from src.user_preferences import get_user_preferences

                qa_service = QAService()
                prefs = get_user_preferences()
                orchestrator = qa_service.create_orchestrator(
                    vector_store_path="fake",
                    embeddings=MagicMock(),
                    answer_mode=prefs.get("qa_answer_mode", "ollama"),
                )
                p = qa_service.retrieve_for_followup(orchestrator, "test")
                result_queue.put(("retrieval_done", p))

                if p._retrieval_context is None:
                    result_queue.put(("success", p))
                    return

                f = qa_service.generate_answer_for_followup(orchestrator, p)
                ollama_text = qa_service.get_placeholder_texts()["ollama_unavailable"]
                if f.quick_answer == ollama_text:
                    result_queue.put(("no_ollama", f))
                else:
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
