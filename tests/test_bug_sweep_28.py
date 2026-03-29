"""
Regression tests for bug sweep 2026-03-28.

Tests code fixes from Groups A and B of the compiled findings.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# A4: retrieve_for_question sets quick_answer="" instead of UNANSWERED_TEXT
# ---------------------------------------------------------------------------
class TestA4QuickAnswerEmpty:
    """Verify that low-relevance results get empty quick_answer, not placeholder text."""

    def test_low_relevance_quick_answer_is_empty(self):
        """When relevance gate rejects, quick_answer should be empty string."""
        from src.core.semantic.semantic_orchestrator import (
            SemanticOrchestrator,
        )

        orch = SemanticOrchestrator.__new__(SemanticOrchestrator)
        orch.questions = []
        orch.results = []
        orch._questions = []
        orch.vector_store_path = "/fake"
        orch.embeddings = MagicMock()

        # Mock the retriever
        mock_retrieval = MagicMock()
        mock_retrieval.context = ""
        mock_retrieval.source_summary = ""
        mock_retrieval.retrieval_time_ms = 10
        mock_retrieval.reranked_score = 0.0
        mock_retrieval.chunks_retrieved = 0

        mock_retriever = MagicMock()
        mock_retriever.retrieve_context.return_value = mock_retrieval
        orch.retriever = mock_retriever

        result = orch.retrieve_for_question("test question?")

        assert result.quick_answer == "", f"Expected empty string, got: {result.quick_answer!r}"


# ---------------------------------------------------------------------------
# A5: Citation excerpt sentence-snapping has size cap
# ---------------------------------------------------------------------------
class TestA5ExcerptSizeCap:
    """Verify _format_excerpt falls back when snapping overshoots."""

    def test_format_excerpt_falls_back_on_oversized_snap(self):
        """When sentence snap exceeds 2x max_chars, revert to unsnapped."""
        from src.core.semantic.citation_excerpt import _format_excerpt

        # Build text where sentence boundaries are very far apart
        long_sentence = "A" * 600 + ". "
        text = long_sentence + "B" * 50 + ". " + "C" * 600 + "."
        start = 600  # Middle of text
        end = 655

        result = _format_excerpt(text, start, end, max_chars=50)
        # Should not include the entire 1250+ char text
        assert len(result) < 300, f"Excerpt too long ({len(result)} chars): {result[:80]}..."


# ---------------------------------------------------------------------------
# A7: TXT export skips empty Quick Answer line
# ---------------------------------------------------------------------------
class TestA7TxtExportEmptyAnswer:
    """Verify TXT export omits Quick Answer line when answer is empty."""

    def test_txt_export_omits_empty_quick_answer(self):
        """When quick_answer is empty, TXT export should not include 'Quick Answer:' line."""
        from src.core.semantic.semantic_orchestrator import (
            SemanticOrchestrator,
            SemanticResult,
        )

        orch = SemanticOrchestrator.__new__(SemanticOrchestrator)
        orch._questions = []
        orch.results = [
            SemanticResult(
                question="What happened?",
                quick_answer="",
                citation="Some citation",
                source_summary="doc.pdf",
                relevance=0.5,
                retrieval_time_ms=10,
            )
        ]

        txt = orch.export_to_text()
        assert "Quick Answer:" not in txt

    def test_txt_export_includes_nonempty_quick_answer(self):
        """When quick_answer has content, TXT export should include it."""
        from src.core.semantic.semantic_orchestrator import (
            SemanticOrchestrator,
            SemanticResult,
        )

        orch = SemanticOrchestrator.__new__(SemanticOrchestrator)
        orch._questions = []
        orch.results = [
            SemanticResult(
                question="What happened?",
                quick_answer="Something happened.",
                citation="Some citation",
                source_summary="doc.pdf",
                relevance=0.8,
                retrieval_time_ms=10,
            )
        ]

        txt = orch.export_to_text()
        assert "Quick Answer: Something happened." in txt


# ---------------------------------------------------------------------------
# A8: PDF builder uses debug not warning for cell truncation
# ---------------------------------------------------------------------------
class TestA8PdfBuilderLogLevel:
    """Verify PDF builder uses debug (not warning) for expected cell truncation."""

    def test_cell_truncation_logs_debug_not_warning(self):
        """Cell truncation should log at DEBUG level, not WARNING."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder("Test Report")

        with patch("src.core.export.pdf_builder.logger") as mock_logger:
            builder.add_table(
                ["Col1"],
                [["This is a long text that exceeds twenty five characters for sure"]],
            )
            mock_logger.debug.assert_called()
            mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# A6: HTML export renders Answer div only when non-empty
# ---------------------------------------------------------------------------
class TestA6HtmlExportConditionalAnswer:
    """Verify HTML export only renders Answer div when quick_answer is non-empty."""

    def test_empty_answer_no_answer_div(self):
        """Empty quick_answer should not produce an Answer div."""
        from src.core.export.combined_html_builder import _build_search_section

        @dataclass
        class FakeResult:
            question: str = "Test question?"
            quick_answer: str = ""
            citation: str = "Some citation"
            source_summary: str = "doc.pdf"

        html = _build_search_section([FakeResult()], include_verification=False)
        assert ">Answer<" not in html

    def test_nonempty_answer_has_answer_div(self):
        """Non-empty quick_answer should produce an Answer div."""
        from src.core.export.combined_html_builder import _build_search_section

        @dataclass
        class FakeResult:
            question: str = "Test question?"
            quick_answer: str = "The answer is yes."
            citation: str = "Some citation"
            source_summary: str = "doc.pdf"

        html = _build_search_section([FakeResult()], include_verification=False)
        assert "Answer" in html
        assert "The answer is yes." in html


# ---------------------------------------------------------------------------
# A10: Followup timeouts are reasonable (not 15 hours)
# ---------------------------------------------------------------------------
class TestA10FollowupTimeouts:
    """Verify followup timeouts are reasonable (minutes, not hours)."""

    def test_followup_poll_timeout_under_10_minutes(self):
        """_FOLLOWUP_TIMEOUT_POLLS should be under 6000 (10 min at 100ms)."""
        from src.ui.main_window import MainWindow

        assert MainWindow._FOLLOWUP_TIMEOUT_POLLS <= 6000, (
            f"Timeout too large: {MainWindow._FOLLOWUP_TIMEOUT_POLLS} polls"
        )


# ---------------------------------------------------------------------------
# A2: _remove_file guards match _clear_files
# ---------------------------------------------------------------------------
class TestA2RemoveFileGuards:
    """Verify _remove_file blocks during all active processing states."""

    def test_remove_file_blocked_during_semantic_answering(self):
        """_remove_file should block when _semantic_answering_active is True."""
        import inspect

        from src.ui.main_window import MainWindow

        source = inspect.getsource(MainWindow._remove_file)
        assert "_semantic_answering_active" in source
        assert "_key_sentences_pending" in source
        assert "_followup_pending" in source


# ---------------------------------------------------------------------------
# D4: Dead LLM constants removed from semantic_constants
# ---------------------------------------------------------------------------
class TestD4DeadConstants:
    """Verify dead LLM constants are removed from semantic_constants."""

    def test_no_llm_prompts_in_constants(self):
        """COMPACT_SEMANTIC_PROMPT and FULL_SEMANTIC_PROMPT should not exist."""
        from src.core.semantic import semantic_constants

        assert not hasattr(semantic_constants, "COMPACT_SEMANTIC_PROMPT")
        assert not hasattr(semantic_constants, "FULL_SEMANTIC_PROMPT")
        assert not hasattr(semantic_constants, "COMPACT_PROMPT_THRESHOLD")
        assert not hasattr(semantic_constants, "PENDING_GENERATION_TEXT")
        assert not hasattr(semantic_constants, "UNANSWERED_TEXT")


# ---------------------------------------------------------------------------
# D5: Coreference plumbing removed from unified_chunker
# ---------------------------------------------------------------------------
class TestD5CorefRemoved:
    """Verify coreference plumbing is removed from UnifiedChunker."""

    def test_no_apply_coreference_param(self):
        """UnifiedChunker.__init__ should not accept apply_coreference."""
        import inspect

        from src.core.chunking.unified_chunker import UnifiedChunker

        params = inspect.signature(UnifiedChunker.__init__).parameters
        assert "apply_coreference" not in params

    def test_no_coref_resolver_method(self):
        """UnifiedChunker should not have _get_coref_resolver."""
        from src.core.chunking.unified_chunker import UnifiedChunker

        assert not hasattr(UnifiedChunker, "_get_coref_resolver")


# ---------------------------------------------------------------------------
# D1: VF.GLINER and VF.KEYBERT removed
# ---------------------------------------------------------------------------
class TestD1DeadVocabConstants:
    """Verify deprecated algorithm constants are removed."""

    def test_no_gliner_or_keybert_in_vf(self):
        """VF should not have GLINER or KEYBERT attributes."""
        from src.core.vocab_schema import VF

        assert not hasattr(VF, "GLINER")
        assert not hasattr(VF, "KEYBERT")

    def test_active_algorithms_still_present(self):
        """Active algorithm constants should still exist."""
        from src.core.vocab_schema import VF

        assert hasattr(VF, "NER")
        assert hasattr(VF, "RAKE")
        assert hasattr(VF, "BM25")
        assert hasattr(VF, "TOPICRANK")
        assert hasattr(VF, "YAKE")
        assert hasattr(VF, "MEDICALNER")


# ---------------------------------------------------------------------------
# B1/B2: Dependencies in requirements.txt
# ---------------------------------------------------------------------------
class TestB1B2Dependencies:
    """Verify missing dependencies are now listed."""

    def test_pillow_in_requirements(self):
        """Pillow should be in requirements.txt."""
        from pathlib import Path

        reqs = Path("requirements.txt").read_text()
        assert "Pillow" in reqs

    def test_langchain_core_in_requirements(self):
        """langchain-core should be in requirements.txt."""
        from pathlib import Path

        reqs = Path("requirements.txt").read_text()
        assert "langchain-core" in reqs
