"""
Tests for v1.0.29 bug sweep — semantic service and orchestrator exports.

Covers get_default_questions (DefaultQuestionsManager path), ask_question
type annotation, export_to_csv column headers, and export_to_text guards.
"""

import csv
import inspect
import io
from unittest.mock import MagicMock, patch


class TestGetDefaultQuestionsUsesManager:
    """get_default_questions must load from DefaultQuestionsManager, not raw YAML."""

    def test_no_orchestrator_uses_default_questions_manager(self, tmp_path):
        """When orchestrator is None, get_default_questions routes through the manager."""
        from src.services.semantic_service import SemanticService

        manager = MagicMock()
        manager.get_enabled_questions.return_value = ["Q1?", "Q2?"]

        service = SemanticService()
        assert service._orchestrator is None

        with patch(
            "src.core.semantic.default_questions_manager.get_default_questions_manager",
            return_value=manager,
        ):
            result = service.get_default_questions()

        manager.get_enabled_questions.assert_called_once()
        assert result == ["Q1?", "Q2?"]

    def test_source_code_imports_default_questions_manager(self):
        """get_default_questions source must reference DefaultQuestionsManager, not raw YAML."""
        from src.services.semantic_service import SemanticService

        source = inspect.getsource(SemanticService.get_default_questions)
        assert "get_default_questions_manager" in source
        assert "load_yaml" not in source

    def test_with_orchestrator_delegates_to_orchestrator(self):
        """When orchestrator exists, get_default_questions delegates to it."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        service._orchestrator = MagicMock()
        service._orchestrator.get_default_questions.return_value = ["OQ1?"]

        result = service.get_default_questions()
        assert result == ["OQ1?"]
        service._orchestrator.get_default_questions.assert_called_once()


class TestAskQuestionReturnType:
    """ask_question should have a correct return type annotation."""

    def test_return_annotation_includes_none(self):
        """ask_question return annotation must allow None (when not ready)."""
        from src.services.semantic_service import SemanticService

        hints = SemanticService.ask_question.__annotations__
        assert "return" in hints
        return_hint = str(hints["return"])
        assert "None" in return_hint

    def test_question_parameter_is_str(self):
        """ask_question takes a str parameter named 'question'."""
        from src.services.semantic_service import SemanticService

        sig = inspect.signature(SemanticService.ask_question)
        assert "question" in sig.parameters
        param = sig.parameters["question"]
        assert param.annotation is str


class TestExportToCsvColumns:
    """export_to_csv must have Question, Citation, Source columns (no Quick Answer)."""

    def _make_orchestrator(self):
        """Create orchestrator bypassing __init__."""
        from src.core.semantic.semantic_orchestrator import SemanticOrchestrator

        orch = object.__new__(SemanticOrchestrator)
        orch.results = []
        orch._questions = []
        return orch

    def test_csv_header_contains_required_columns(self):
        """CSV header must include Question, Citation, and Source columns."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [
            SemanticResult(
                question="Who filed?",
                citation="John Smith filed.",
                source_summary="complaint.pdf",
                include_in_export=True,
            )
        ]

        csv_output = orch.export_to_csv()
        header_line = csv_output.split("\n")[0]

        assert "Question" in header_line
        assert "Citation" in header_line
        assert "Source" in header_line

    def test_csv_header_does_not_contain_quick_answer(self):
        """CSV header must NOT include a Quick Answer column."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [
            SemanticResult(
                question="Q?",
                citation="C.",
                include_in_export=True,
            )
        ]

        csv_output = orch.export_to_csv()
        header_line = csv_output.split("\n")[0]

        assert "Quick Answer" not in header_line

    def test_csv_data_row_has_three_columns(self):
        """Each data row must have exactly three fields."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [
            SemanticResult(
                question="What damages?",
                citation="$100k claimed.",
                source_summary="complaint.pdf, page 5",
                include_in_export=True,
            )
        ]

        csv_output = orch.export_to_csv()
        reader = csv.reader(io.StringIO(csv_output))
        rows = list(reader)

        assert len(rows) == 2  # header + 1 data row
        assert len(rows[0]) == 3  # 3 columns
        assert len(rows[1]) == 3
        assert rows[1][0] == "What damages?"
        assert rows[1][1] == "$100k claimed."
        assert rows[1][2] == "complaint.pdf, page 5"

    def test_csv_empty_when_nothing_exportable(self):
        """export_to_csv returns empty string when no results are exportable."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [SemanticResult(question="Q?", include_in_export=False)]

        assert orch.export_to_csv() == ""

    def test_text_empty_quick_answer_omits_line(self):
        """export_to_text omits Quick Answer line when quick_answer is empty."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [
            SemanticResult(
                question="Who filed?",
                quick_answer="",
                citation="John filed.",
                include_in_export=True,
            )
        ]
        assert "Quick Answer:" not in orch.export_to_text()

    def test_text_nonempty_quick_answer_includes_line(self):
        """export_to_text includes Quick Answer when quick_answer has content."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [
            SemanticResult(
                question="Who filed?",
                quick_answer="John Smith.",
                citation="John Smith filed.",
                include_in_export=True,
            )
        ]
        assert "Quick Answer: John Smith." in orch.export_to_text()
