"""
Tests for src/core/export/semantic_exporter.py.

Covers semantic search result export to DocumentBuilder.
"""

from unittest.mock import MagicMock


def _make_result(question="Who filed?", citation="Plaintiff filed.", quick_answer=""):
    """Create a mock SemanticResult for export tests."""
    r = MagicMock()
    r.question = question
    r.citation = citation
    r.quick_answer = quick_answer
    r.source_summary = "complaint.pdf"
    return r


def _make_builder():
    """Create a mock DocumentBuilder."""
    b = MagicMock()
    b.add_heading = MagicMock()
    b.add_paragraph = MagicMock()
    b.add_separator = MagicMock()
    return b


class TestExportSemanticResults:
    """Tests for export_semantic_results()."""

    def test_adds_title_heading(self):
        """Should add the title as a level-1 heading."""
        from src.core.export.semantic_exporter import export_semantic_results

        builder = _make_builder()
        export_semantic_results([], builder, title="Search Results")

        builder.add_heading.assert_any_call("Search Results", level=1)

    def test_shows_question_count(self):
        """Should show the number of questions answered."""
        from src.core.export.semantic_exporter import export_semantic_results

        builder = _make_builder()
        results = [_make_result(), _make_result("What happened?")]
        export_semantic_results(results, builder)

        para_texts = [str(c[0][0]) for c in builder.add_paragraph.call_args_list]
        assert any("2 questions" in t for t in para_texts)

    def test_empty_results_shows_no_results_message(self):
        """Should display 'No search results' when empty."""
        from src.core.export.semantic_exporter import export_semantic_results

        builder = _make_builder()
        export_semantic_results([], builder)

        para_texts = [str(c[0][0]) for c in builder.add_paragraph.call_args_list]
        assert any("No search results" in t for t in para_texts)

    def test_each_result_gets_heading(self):
        """Each result should have a Q-numbered heading."""
        from src.core.export.semantic_exporter import export_semantic_results

        builder = _make_builder()
        results = [_make_result("First?"), _make_result("Second?")]
        export_semantic_results(results, builder)

        heading_texts = [str(c[0][0]) for c in builder.add_heading.call_args_list]
        assert any("Q1" in t for t in heading_texts)
        assert any("Q2" in t for t in heading_texts)

    def test_citation_included(self):
        """Should include the citation text."""
        from src.core.export.semantic_exporter import export_semantic_results

        builder = _make_builder()
        results = [_make_result(citation="Excerpt from page 5.")]
        export_semantic_results(results, builder)

        para_texts = [str(c[0][0]) for c in builder.add_paragraph.call_args_list]
        assert any("Excerpt from page 5." in t for t in para_texts)

    def test_source_summary_included(self):
        """Should include the source summary when present."""
        from src.core.export.semantic_exporter import export_semantic_results

        builder = _make_builder()
        result = _make_result()
        result.source_summary = "answer.pdf, Section A"
        export_semantic_results([result], builder)

        para_texts = [str(c[0][0]) for c in builder.add_paragraph.call_args_list]
        assert any("answer.pdf" in t for t in para_texts)

    def test_adds_footer(self):
        """Should add CasePrepd footer."""
        from src.core.export.semantic_exporter import export_semantic_results

        builder = _make_builder()
        export_semantic_results([_make_result()], builder)

        para_calls = [str(c) for c in builder.add_paragraph.call_args_list]
        assert any("CasePrepd" in c for c in para_calls)

    def test_quick_answer_included_when_present(self):
        """Should include quick_answer when it exists."""
        from src.core.export.semantic_exporter import export_semantic_results

        builder = _make_builder()
        result = _make_result(quick_answer="Yes, the plaintiff filed.")
        export_semantic_results([result], builder)

        para_texts = [str(c[0][0]) for c in builder.add_paragraph.call_args_list]
        assert any("Yes, the plaintiff filed." in t for t in para_texts)
