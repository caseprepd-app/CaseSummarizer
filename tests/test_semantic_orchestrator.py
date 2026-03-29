"""
Tests for src/core/semantic/semantic_orchestrator.py.

Covers SemanticResult dataclass, SemanticOrchestrator question loading,
result management, and export formatting. Heavy dependencies (FAISS,
embeddings, retriever) are mocked.
"""


class TestSemanticResult:
    """Tests for the SemanticResult dataclass."""

    def test_default_fields(self):
        """Default values should be sensible."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        r = SemanticResult(question="Test?")
        assert r.question == "Test?"
        assert r.quick_answer == ""
        assert r.citation == ""
        assert r.include_in_export is True
        assert r.relevance == 0.0
        assert r.is_followup is False

    def test_answer_property_returns_quick_answer(self):
        """answer property should return quick_answer when set."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        r = SemanticResult(question="Q?", quick_answer="Yes.")
        assert r.answer == "Yes."

    def test_answer_property_falls_back_to_citation(self):
        """answer property should return citation when quick_answer is empty."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        r = SemanticResult(question="Q?", citation="See page 5.")
        assert r.answer == "See page 5."

    def test_is_answered_true_for_valid_result(self):
        """is_answered should be True for results with relevance > 0."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        r = SemanticResult(question="Q?", relevance=0.8)
        assert r.is_answered is True

    def test_is_answered_false_for_unanswered(self):
        """is_answered should be False for unanswered results."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        r = SemanticResult(question="Q?", relevance=0.0, quick_answer="")
        assert r.is_answered is False

    def test_is_exportable_true_above_floor(self):
        """is_exportable should be True when relevance meets the floor."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        r = SemanticResult(question="Q?", relevance=0.9, citation="Found.")
        assert r.is_exportable is True

    def test_is_exportable_false_for_nan(self):
        """is_exportable should be False for NaN relevance."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        r = SemanticResult(question="Q?", relevance=float("nan"))
        assert r.is_exportable is False

    def test_is_exportable_false_below_floor(self):
        """is_exportable should be False when relevance is below floor."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        r = SemanticResult(question="Q?", relevance=0.01)
        assert r.is_exportable is False


class TestSemanticOrchestratorResultManagement:
    """Tests for result storage and export methods."""

    def _make_orchestrator(self):
        """Create orchestrator with mocked dependencies."""
        from src.core.semantic.semantic_orchestrator import SemanticOrchestrator

        orch = object.__new__(SemanticOrchestrator)
        orch.results = []
        orch._questions = []
        return orch

    def test_get_exportable_results_filters_correctly(self):
        """get_exportable_results should only return results with include_in_export=True."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [
            SemanticResult(question="Q1?", include_in_export=True),
            SemanticResult(question="Q2?", include_in_export=False),
            SemanticResult(question="Q3?", include_in_export=True),
        ]

        exportable = orch.get_exportable_results()
        assert len(exportable) == 2
        assert all(r.include_in_export for r in exportable)

    def test_toggle_export(self):
        """toggle_export should flip include_in_export."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [SemanticResult(question="Q?", include_in_export=True)]

        new_val = orch.toggle_export(0)
        assert new_val is False
        assert orch.results[0].include_in_export is False

    def test_toggle_export_invalid_index(self):
        """toggle_export with invalid index should return False."""
        orch = self._make_orchestrator()
        orch.results = []
        assert orch.toggle_export(99) is False

    def test_clear_results(self):
        """clear_results should empty the results list."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [SemanticResult(question="Q?")]
        orch.clear_results()
        assert orch.results == []

    def test_export_to_text_formats_correctly(self):
        """export_to_text should produce formatted text output."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [
            SemanticResult(
                question="Who filed?",
                quick_answer="Plaintiff.",
                citation="See complaint.",
                include_in_export=True,
                source_summary="complaint.pdf",
            )
        ]

        text = orch.export_to_text()
        assert "Who filed?" in text
        assert "Plaintiff." in text
        assert "complaint.pdf" in text

    def test_export_to_text_empty_when_nothing_exportable(self):
        """export_to_text should return empty string when no results exportable."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [SemanticResult(question="Q?", include_in_export=False)]

        assert orch.export_to_text() == ""

    def test_export_to_csv_has_header_row(self):
        """export_to_csv should include a header row."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        orch = self._make_orchestrator()
        orch.results = [SemanticResult(question="Q?", citation="C.", include_in_export=True)]

        csv = orch.export_to_csv()
        assert "Question" in csv
        assert "Citation" in csv

    def test_export_to_csv_empty_when_no_results(self):
        """export_to_csv should return empty string when no exportable results."""
        orch = self._make_orchestrator()
        orch.results = []
        assert orch.export_to_csv() == ""


class TestCalculateRelevance:
    """Tests for _calculate_relevance()."""

    def _make_orchestrator(self):
        """Create orchestrator bypassing __init__."""
        from src.core.semantic.semantic_orchestrator import SemanticOrchestrator

        return object.__new__(SemanticOrchestrator)

    def test_no_sources_returns_zero(self):
        """Empty sources should return 0.0."""
        from src.core.vector_store.semantic_retriever import RetrievalResult

        orch = self._make_orchestrator()
        result = RetrievalResult(context="", sources=[], chunks_retrieved=0, retrieval_time_ms=0)
        assert orch._calculate_relevance(result) == 0.0

    def test_returns_top_source_score(self):
        """Should return the single source's relevance score directly."""
        from src.core.vector_store.semantic_retriever import RetrievalResult, SourceInfo

        orch = self._make_orchestrator()
        sources = [
            SourceInfo(
                filename="a.pdf", chunk_num=0, section="N/A", relevance_score=0.85, word_count=50
            ),
        ]
        result = RetrievalResult(
            context="x", sources=sources, chunks_retrieved=1, retrieval_time_ms=0
        )

        relevance = orch._calculate_relevance(result)
        assert abs(relevance - 0.85) < 0.01


class TestGenerateAnswerForResult:
    """Tests for generate_answer_for_result (no-op)."""

    def test_returns_same_result(self):
        """Should return the same result unchanged (no-op after LLM removal)."""
        from src.core.semantic.semantic_orchestrator import SemanticOrchestrator, SemanticResult

        orch = object.__new__(SemanticOrchestrator)
        r = SemanticResult(question="Q?", citation="C.")
        assert orch.generate_answer_for_result(r) is r
