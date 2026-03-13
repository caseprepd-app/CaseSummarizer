"""
Tests for Q&A Orchestrator and Answer Generator.

Tests the Q&A system components:
1. SemanticResult - Data model for Q&A pairs
2. AnswerGenerator - Extraction and Ollama modes
3. SemanticOrchestrator - Question loading and answer coordination
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.semantic import SemanticResult


class TestSemanticResult:
    """Tests for SemanticResult dataclass."""

    def test_qaresult_defaults(self):
        """SemanticResult should have sensible defaults."""

        result = SemanticResult(
            question="Who is the plaintiff?",
            quick_answer="John Smith is the plaintiff.",
            citation="The plaintiff John Smith filed...",
        )

        assert result.question == "Who is the plaintiff?"
        assert result.quick_answer == "John Smith is the plaintiff."
        assert result.citation == "The plaintiff John Smith filed..."
        # Backward compatibility: answer property returns quick_answer
        assert result.answer == "John Smith is the plaintiff."
        assert result.include_in_export is True  # Default to included
        assert result.source_summary == ""
        assert result.confidence == 0.0
        assert result.is_followup is False

    def test_qaresult_full_initialization(self):
        """SemanticResult should accept all fields."""

        result = SemanticResult(
            question="What damages are sought?",
            quick_answer="$100,000 in compensatory damages",
            citation="Plaintiff seeks damages of $100,000...",
            include_in_export=False,
            source_summary="complaint.pdf, pages 5-6",
            confidence=0.85,
            retrieval_time_ms=123.4,
            is_followup=True,
        )

        assert result.include_in_export is False
        assert result.source_summary == "complaint.pdf, pages 5-6"
        assert result.confidence == 0.85
        assert result.retrieval_time_ms == 123.4
        assert result.is_followup is True
        assert result.quick_answer == "$100,000 in compensatory damages"
        assert result.citation == "Plaintiff seeks damages of $100,000..."


class TestSemanticOrchestrator:
    """Tests for SemanticOrchestrator class (without actual vector store)."""

    def test_default_questions_path_exists(self):
        """DEFAULT_QUESTIONS_PATH should point to existing file."""
        from src.core.semantic.semantic_orchestrator import DEFAULT_QUESTIONS_PATH

        # The config/qa_questions.yaml should exist in the project
        assert DEFAULT_QUESTIONS_PATH.exists()

    def test_question_loading_from_yaml(self, tmp_path):
        """Orchestrator should load questions from YAML file."""
        # Create a test YAML file
        yaml_content = """
version: "1.0"
questions:
  - id: "test_q1"
    text: "What type of case is this?"
    category: "General"
    type: "extraction"
  - id: "test_q2"
    text: "Who are the parties?"
    category: "Parties"
    type: "extraction"
"""
        yaml_path = tmp_path / "test_questions.yaml"
        yaml_path.write_text(yaml_content)

        # Mock the SemanticRetriever to avoid needing actual vector store
        with patch("src.core.semantic.semantic_orchestrator.SemanticRetriever"):
            from src.core.semantic import SemanticOrchestrator

            orchestrator = SemanticOrchestrator(
                vector_store_path=tmp_path, embeddings=MagicMock(), questions_path=yaml_path
            )

            questions = orchestrator.get_default_questions()

            assert len(questions) == 2
            assert "What type of case is this?" in questions
            assert "Who are the parties?" in questions

    def test_get_exportable_results_filters_by_flag(self):
        """get_exportable_results should only return included items."""

        # Create mock orchestrator results
        with patch("src.core.semantic.semantic_orchestrator.SemanticRetriever"):
            from src.core.semantic import SemanticOrchestrator

            orchestrator = SemanticOrchestrator(vector_store_path=Path("."), embeddings=MagicMock())

            # Add mock results
            orchestrator.results = [
                SemanticResult(
                    question="Q1", quick_answer="A1", citation="C1", include_in_export=True
                ),
                SemanticResult(
                    question="Q2", quick_answer="A2", citation="C2", include_in_export=False
                ),
                SemanticResult(
                    question="Q3", quick_answer="A3", citation="C3", include_in_export=True
                ),
            ]

            exportable = orchestrator.get_exportable_results()

            assert len(exportable) == 2
            assert all(r.include_in_export for r in exportable)

    def test_toggle_export_changes_flag(self):
        """toggle_export should flip the include_in_export flag."""

        with patch("src.core.semantic.semantic_orchestrator.SemanticRetriever"):
            from src.core.semantic import SemanticOrchestrator

            orchestrator = SemanticOrchestrator(vector_store_path=Path("."), embeddings=MagicMock())

            orchestrator.results = [
                SemanticResult(
                    question="Q1", quick_answer="A1", citation="C1", include_in_export=True
                ),
            ]

            # Toggle off
            new_value = orchestrator.toggle_export(0)
            assert new_value is False
            assert orchestrator.results[0].include_in_export is False

            # Toggle back on
            new_value = orchestrator.toggle_export(0)
            assert new_value is True
            assert orchestrator.results[0].include_in_export is True

    def test_export_to_text_format(self):
        """export_to_text should produce properly formatted text."""

        with patch("src.core.semantic.semantic_orchestrator.SemanticRetriever"):
            from src.core.semantic import SemanticOrchestrator

            orchestrator = SemanticOrchestrator(vector_store_path=Path("."), embeddings=MagicMock())

            orchestrator.results = [
                SemanticResult(
                    question="What type of case is this?",
                    quick_answer="This is a civil personal injury case.",
                    citation="Filed in civil court for personal injury...",
                    source_summary="complaint.pdf",
                    include_in_export=True,
                ),
            ]

            text = orchestrator.export_to_text()

            assert "DOCUMENT SEMANTIC SEARCH RESULTS" in text
            assert "Q1: What type of case is this?" in text
            assert "Quick Answer: This is a civil personal injury case." in text
            assert "[Source: complaint.pdf]" in text

    def test_export_to_text_excludes_unchecked(self):
        """export_to_text should only include checked items."""

        with patch("src.core.semantic.semantic_orchestrator.SemanticRetriever"):
            from src.core.semantic import SemanticOrchestrator

            orchestrator = SemanticOrchestrator(vector_store_path=Path("."), embeddings=MagicMock())

            orchestrator.results = [
                SemanticResult(
                    question="Included", quick_answer="Yes", citation="C1", include_in_export=True
                ),
                SemanticResult(
                    question="Excluded", quick_answer="No", citation="C2", include_in_export=False
                ),
            ]

            text = orchestrator.export_to_text()

            assert "Included" in text
            assert "Excluded" not in text

    def test_clear_results(self):
        """clear_results should empty the results list."""

        with patch("src.core.semantic.semantic_orchestrator.SemanticRetriever"):
            from src.core.semantic import SemanticOrchestrator

            orchestrator = SemanticOrchestrator(vector_store_path=Path("."), embeddings=MagicMock())

            orchestrator.results = [SemanticResult(question="Q", quick_answer="A", citation="C")]
            assert len(orchestrator.results) == 1

            orchestrator.clear_results()
            assert len(orchestrator.results) == 0


class TestSemanticWorker:
    """Tests for SemanticWorker thread."""

    def test_worker_initialization(self):
        """SemanticWorker should initialize with required parameters."""
        from queue import Queue

        from src.services.workers import SemanticWorker

        queue = Queue()
        worker = SemanticWorker(
            vector_store_path=Path("."),
            embeddings=MagicMock(),
            ui_queue=queue,
            answer_mode="extraction",
        )

        assert worker.answer_mode == "extraction"
        assert worker.custom_questions is None

    def test_worker_accepts_custom_questions(self):
        """SemanticWorker should accept custom question list."""
        from queue import Queue

        from src.services.workers import SemanticWorker

        queue = Queue()
        custom_qs = ["Question 1?", "Question 2?"]

        worker = SemanticWorker(
            vector_store_path=Path("."), embeddings=MagicMock(), ui_queue=queue, questions=custom_qs
        )

        assert worker.custom_questions == custom_qs

    def test_worker_stop_signal(self):
        """SemanticWorker should respond to stop signal."""
        from queue import Queue

        from src.services.workers import SemanticWorker

        queue = Queue()
        worker = SemanticWorker(vector_store_path=Path("."), embeddings=MagicMock(), ui_queue=queue)

        assert not worker._stop_event.is_set()

        worker.stop()

        assert worker._stop_event.is_set()
