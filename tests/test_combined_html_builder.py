"""
Tests for src/core/export/combined_html_builder.py.

Covers tabbed HTML export with Vocabulary, Search, and Summary sections.
"""

from unittest.mock import MagicMock

from src.core.vocab_schema import VF


def _make_vocab(n=2):
    """Create sample vocabulary data."""
    return [
        {
            VF.TERM: f"term_{i}",
            VF.QUALITY_SCORE: 0.9,
            VF.IS_PERSON: VF.YES if i == 0 else VF.NO,
            VF.FOUND_BY: "NER",
            VF.OCCURRENCES: 3,
        }
        for i in range(n)
    ]


def _make_result(question="Who filed?"):
    """Create a mock SemanticResult."""
    r = MagicMock()
    r.question = question
    r.quick_answer = "The plaintiff filed."
    r.citation = "See page 5."
    r.source_summary = "doc.pdf"
    r.verification = None
    return r


class TestBuildCombinedHtml:
    """Tests for build_combined_html()."""

    def test_no_data_returns_empty_page(self):
        """Should return simple 'No data' page when all inputs empty."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html([], [], "")
        assert "No data to export" in html

    def test_vocab_only_shows_vocab_tab(self):
        """Should show Vocabulary tab button when only vocab data provided."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html(_make_vocab(), [], "")
        assert "Vocabulary" in html
        assert "tab-vocab" in html

    def test_search_only_shows_search_tab(self):
        """Should show Search tab when only search results provided."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html([], [_make_result()], "")
        assert "Search" in html
        assert "tab-qa" in html

    def test_summary_only_shows_summary_tab(self):
        """Should show Summary tab when only summary text provided."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html([], [], "Key findings here.")
        assert "Summary" in html
        assert "tab-summary" in html

    def test_all_three_tabs_present(self):
        """Should show all three tabs when all data provided."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html(_make_vocab(), [_make_result()], "Summary text.")
        assert "Vocabulary" in html
        assert "Search" in html
        assert "Summary" in html

    def test_vocab_terms_in_html(self):
        """Vocabulary terms should appear in the HTML output."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html(_make_vocab(2), [], "")
        assert "term_0" in html
        assert "term_1" in html

    def test_question_in_search_tab(self):
        """Question text should appear in search section."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html([], [_make_result("What happened?")], "")
        assert "What happened?" in html

    def test_first_tab_is_active(self):
        """First tab should have 'active' class."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html(_make_vocab(), [_make_result()], "")
        # First tab button should have active class
        assert 'class="tab-btn active"' in html

    def test_html_has_doctype(self):
        """Output should be a complete HTML document."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html(_make_vocab(), [], "")
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_person_row_class(self):
        """Person entries should have 'person' CSS class on their row."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html(_make_vocab(2), [], "")
        assert 'class="person"' in html

    def test_summary_paragraphs_rendered(self):
        """Summary text paragraphs should be wrapped in <p> tags."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html([], [], "First paragraph.\n\nSecond paragraph.")
        assert "<p>" in html
        assert "First paragraph." in html
        assert "Second paragraph." in html

    def test_timestamp_in_output(self):
        """Output should contain a timestamp."""
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html(_make_vocab(), [], "")
        assert "Generated" in html
