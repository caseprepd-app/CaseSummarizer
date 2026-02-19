"""
Tests for Q&A Orchestrator and Answer Generator.

Tests the Q&A system components:
1. QAResult - Data model for Q&A pairs
2. AnswerGenerator - Extraction and Ollama modes
3. QAOrchestrator - Question loading and answer coordination
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestQAResult:
    """Tests for QAResult dataclass."""

    def test_qaresult_defaults(self):
        """QAResult should have sensible defaults."""
        from src.core.qa import QAResult

        result = QAResult(
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
        """QAResult should accept all fields."""
        from src.core.qa import QAResult

        result = QAResult(
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


class TestAnswerGenerator:
    """Tests for AnswerGenerator class."""

    def test_initialization_with_extraction_mode(self):
        """AnswerGenerator should initialize with extraction mode."""
        from src.core.qa import AnswerGenerator, AnswerMode

        generator = AnswerGenerator(mode="extraction")

        assert generator.mode == AnswerMode.EXTRACTION

    def test_initialization_with_ollama_mode(self):
        """AnswerGenerator should initialize with ollama mode."""
        from src.core.qa import AnswerGenerator, AnswerMode

        generator = AnswerGenerator(mode="ollama")

        assert generator.mode == AnswerMode.OLLAMA

    def test_generate_returns_message_for_empty_context(self):
        """generate() should return appropriate message for empty context."""
        from src.core.qa import AnswerGenerator

        generator = AnswerGenerator(mode="extraction")
        result = generator.generate("Who is the plaintiff?", "")

        assert "No relevant information" in result

    def test_extraction_mode_extracts_from_context(self):
        """Extraction mode should find relevant sentences from context."""
        from src.core.qa import AnswerGenerator

        generator = AnswerGenerator(mode="extraction")

        context = """
        The plaintiff John Smith filed this lawsuit on January 15, 2024.
        He alleges that the defendant ABC Corporation was negligent.
        The incident occurred at the defendant's warehouse in Brooklyn.
        """

        result = generator.generate("Who is the plaintiff?", context)

        # Should extract sentence mentioning plaintiff
        assert "John Smith" in result or "plaintiff" in result.lower()

    def test_extraction_mode_handles_no_matches(self):
        """Extraction mode should handle case with no keyword matches."""
        from src.core.qa import AnswerGenerator

        generator = AnswerGenerator(mode="extraction")

        context = "This document contains no relevant party information."

        result = generator.generate("Who is the defendant?", context)

        # Should return something (first sentence or "no specific answer")
        assert len(result) > 0

    def test_keyword_extraction_filters_stopwords(self):
        """_extract_keywords should filter common stopwords."""
        from src.core.qa import AnswerGenerator

        generator = AnswerGenerator()

        keywords = generator._extract_keywords("What is the name of the plaintiff?")

        assert "what" not in keywords
        assert "is" not in keywords
        assert "the" not in keywords
        assert "of" not in keywords
        assert "name" in keywords
        assert "plaintiff" in keywords

    @pytest.mark.xfail(
        reason="Sentence splitter does not handle abbreviations (Dr., Jan.)", strict=False
    )
    def test_split_sentences_handles_abbreviations(self):
        """_split_sentences should handle common abbreviations."""
        from src.core.qa import AnswerGenerator

        generator = AnswerGenerator()

        text = "Dr. Smith testified on Jan. 15, 2024. He stated the injury was severe."

        sentences = generator._split_sentences(text)

        # Should not split at "Dr." or "Jan."
        assert len(sentences) == 2

    def test_set_mode_changes_mode(self):
        """set_mode should change the generation mode."""
        from src.core.qa import AnswerGenerator, AnswerMode

        generator = AnswerGenerator(mode="extraction")
        assert generator.mode == AnswerMode.EXTRACTION

        generator.set_mode("ollama")
        assert generator.mode == AnswerMode.OLLAMA


class TestQAOrchestrator:
    """Tests for QAOrchestrator class (without actual vector store)."""

    def test_default_questions_path_exists(self):
        """DEFAULT_QUESTIONS_PATH should point to existing file."""
        from src.core.qa.qa_orchestrator import DEFAULT_QUESTIONS_PATH

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

        # Mock the QARetriever to avoid needing actual vector store
        with patch("src.core.qa.qa_orchestrator.QARetriever"):
            from src.core.qa import QAOrchestrator

            orchestrator = QAOrchestrator(
                vector_store_path=tmp_path, embeddings=MagicMock(), questions_path=yaml_path
            )

            questions = orchestrator.get_default_questions()

            assert len(questions) == 2
            assert "What type of case is this?" in questions
            assert "Who are the parties?" in questions

    def test_get_exportable_results_filters_by_flag(self):
        """get_exportable_results should only return included items."""
        from src.core.qa import QAResult

        # Create mock orchestrator results
        with patch("src.core.qa.qa_orchestrator.QARetriever"):
            from src.core.qa import QAOrchestrator

            orchestrator = QAOrchestrator(vector_store_path=Path("."), embeddings=MagicMock())

            # Add mock results
            orchestrator.results = [
                QAResult(question="Q1", quick_answer="A1", citation="C1", include_in_export=True),
                QAResult(question="Q2", quick_answer="A2", citation="C2", include_in_export=False),
                QAResult(question="Q3", quick_answer="A3", citation="C3", include_in_export=True),
            ]

            exportable = orchestrator.get_exportable_results()

            assert len(exportable) == 2
            assert all(r.include_in_export for r in exportable)

    def test_toggle_export_changes_flag(self):
        """toggle_export should flip the include_in_export flag."""
        from src.core.qa import QAResult

        with patch("src.core.qa.qa_orchestrator.QARetriever"):
            from src.core.qa import QAOrchestrator

            orchestrator = QAOrchestrator(vector_store_path=Path("."), embeddings=MagicMock())

            orchestrator.results = [
                QAResult(question="Q1", quick_answer="A1", citation="C1", include_in_export=True),
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
        from src.core.qa import QAResult

        with patch("src.core.qa.qa_orchestrator.QARetriever"):
            from src.core.qa import QAOrchestrator

            orchestrator = QAOrchestrator(vector_store_path=Path("."), embeddings=MagicMock())

            orchestrator.results = [
                QAResult(
                    question="What type of case is this?",
                    quick_answer="This is a civil personal injury case.",
                    citation="Filed in civil court for personal injury...",
                    source_summary="complaint.pdf",
                    include_in_export=True,
                ),
            ]

            text = orchestrator.export_to_text()

            assert "DOCUMENT Q&A SUMMARY" in text
            assert "Q1: What type of case is this?" in text
            assert "Quick Answer: This is a civil personal injury case." in text
            assert "[Source: complaint.pdf]" in text

    def test_export_to_text_excludes_unchecked(self):
        """export_to_text should only include checked items."""
        from src.core.qa import QAResult

        with patch("src.core.qa.qa_orchestrator.QARetriever"):
            from src.core.qa import QAOrchestrator

            orchestrator = QAOrchestrator(vector_store_path=Path("."), embeddings=MagicMock())

            orchestrator.results = [
                QAResult(
                    question="Included", quick_answer="Yes", citation="C1", include_in_export=True
                ),
                QAResult(
                    question="Excluded", quick_answer="No", citation="C2", include_in_export=False
                ),
            ]

            text = orchestrator.export_to_text()

            assert "Included" in text
            assert "Excluded" not in text

    def test_clear_results(self):
        """clear_results should empty the results list."""
        from src.core.qa import QAResult

        with patch("src.core.qa.qa_orchestrator.QARetriever"):
            from src.core.qa import QAOrchestrator

            orchestrator = QAOrchestrator(vector_store_path=Path("."), embeddings=MagicMock())

            orchestrator.results = [QAResult(question="Q", quick_answer="A", citation="C")]
            assert len(orchestrator.results) == 1

            orchestrator.clear_results()
            assert len(orchestrator.results) == 0


class TestQAWorker:
    """Tests for QAWorker thread."""

    def test_worker_initialization(self):
        """QAWorker should initialize with required parameters."""
        from queue import Queue

        from src.services.workers import QAWorker

        queue = Queue()
        worker = QAWorker(
            vector_store_path=Path("."),
            embeddings=MagicMock(),
            ui_queue=queue,
            answer_mode="extraction",
        )

        assert worker.answer_mode == "extraction"
        assert worker.custom_questions is None

    def test_worker_accepts_custom_questions(self):
        """QAWorker should accept custom question list."""
        from queue import Queue

        from src.services.workers import QAWorker

        queue = Queue()
        custom_qs = ["Question 1?", "Question 2?"]

        worker = QAWorker(
            vector_store_path=Path("."), embeddings=MagicMock(), ui_queue=queue, questions=custom_qs
        )

        assert worker.custom_questions == custom_qs

    def test_worker_stop_signal(self):
        """QAWorker should respond to stop signal."""
        from queue import Queue

        from src.services.workers import QAWorker

        queue = Queue()
        worker = QAWorker(vector_store_path=Path("."), embeddings=MagicMock(), ui_queue=queue)

        assert not worker._stop_event.is_set()

        worker.stop()

        assert worker._stop_event.is_set()
