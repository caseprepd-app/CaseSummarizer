"""
Tests for src/core/export/combined_exporter.py.

Covers combined export to DocumentBuilder (Word/PDF) and TXT format.
"""

from unittest.mock import MagicMock, patch


def _make_vocab_data(n=2):
    """Create sample vocabulary dicts."""
    from src.core.vocab_schema import VF

    return [
        {VF.TERM: f"term_{i}", VF.QUALITY_SCORE: 0.9 - i * 0.1, VF.IS_PERSON: VF.NO}
        for i in range(n)
    ]


def _make_semantic_result(question="Who filed?", citation="The plaintiff filed."):
    """Create a mock SemanticResult."""
    r = MagicMock()
    r.question = question
    r.citation = citation
    r.source_summary = "complaint.pdf"
    r.relevance = 0.85
    r.quick_answer = ""
    return r


def _make_builder():
    """Create a mock DocumentBuilder."""
    builder = MagicMock()
    builder.add_heading = MagicMock()
    builder.add_paragraph = MagicMock()
    builder.add_table = MagicMock()
    builder.add_separator = MagicMock()
    return builder


class TestExportCombined:
    """Tests for export_combined() with DocumentBuilder."""

    def test_adds_title_heading(self):
        """Should add the document title as level-1 heading."""
        from src.core.export.combined_exporter import export_combined

        builder = _make_builder()
        export_combined([], [], builder, title="My Report")

        builder.add_heading.assert_any_call("My Report", level=1)

    def test_includes_summary_section_when_provided(self):
        """Should add Summary heading when summary_text is non-empty."""
        from src.core.export.combined_exporter import export_combined

        builder = _make_builder()
        export_combined([], [], builder, summary_text="Case summary here.")

        heading_calls = [c[0] for c in builder.add_heading.call_args_list]
        assert any("Summary" in str(h) for h in heading_calls)

    def test_skips_summary_when_empty(self):
        """Should not add Summary heading when summary_text is empty."""
        from src.core.export.combined_exporter import export_combined

        builder = _make_builder()
        export_combined([], [], builder, summary_text="")

        heading_calls = [str(c) for c in builder.add_heading.call_args_list]
        assert not any("Summary" in h for h in heading_calls)

    @patch("src.core.export.combined_exporter.export_vocabulary")
    def test_calls_export_vocabulary_when_data_present(self, mock_vocab):
        """Should delegate to export_vocabulary when vocab_data is non-empty."""
        from src.core.export.combined_exporter import export_combined

        builder = _make_builder()
        vocab = _make_vocab_data(2)
        export_combined(vocab, [], builder)

        mock_vocab.assert_called_once()

    @patch("src.core.export.combined_exporter.export_semantic_results")
    def test_calls_export_semantic_when_results_present(self, mock_sem):
        """Should delegate to export_semantic_results when results exist."""
        from src.core.export.combined_exporter import export_combined

        builder = _make_builder()
        results = [_make_semantic_result()]
        export_combined([], results, builder)

        mock_sem.assert_called_once()

    def test_adds_footer(self):
        """Should add CasePrepd footer paragraph."""
        from src.core.export.combined_exporter import export_combined

        builder = _make_builder()
        export_combined([], [], builder)

        para_texts = [str(c[0][0]) for c in builder.add_paragraph.call_args_list]
        assert any("CasePrepd" in t for t in para_texts)

    def test_skips_vocab_section_when_empty(self):
        """Should not call export_vocabulary when vocab_data is empty."""
        from src.core.export.combined_exporter import export_combined

        with patch("src.core.export.combined_exporter.export_vocabulary") as mock_v:
            builder = _make_builder()
            export_combined([], [_make_semantic_result()], builder)
            mock_v.assert_not_called()


class TestExportCombinedTxt:
    """Tests for export_combined_txt() plain text export."""

    def test_creates_file_with_content(self, tmp_path):
        """Should write a non-empty text file."""
        from src.core.export.combined_exporter import export_combined_txt

        out = tmp_path / "report.txt"
        result = export_combined_txt(_make_vocab_data(), [], "", str(out))

        assert result is True
        assert out.exists()
        assert len(out.read_text(encoding="utf-8")) > 0

    def test_includes_vocab_terms(self, tmp_path):
        """Should include vocabulary term names in TXT output."""
        from src.core.export.combined_exporter import export_combined_txt

        out = tmp_path / "report.txt"
        export_combined_txt(_make_vocab_data(2), [], "", str(out))
        content = out.read_text(encoding="utf-8")

        assert "term_0" in content
        assert "term_1" in content

    def test_includes_search_results(self, tmp_path):
        """Should include question and citation in TXT output."""
        from src.core.export.combined_exporter import export_combined_txt

        result = _make_semantic_result("What happened?", "An accident occurred.")
        out = tmp_path / "report.txt"
        export_combined_txt([], [result], "", str(out))
        content = out.read_text(encoding="utf-8")

        assert "What happened?" in content
        assert "An accident occurred." in content

    def test_includes_summary_text(self, tmp_path):
        """Should include key excerpts section when summary provided."""
        from src.core.export.combined_exporter import export_combined_txt

        out = tmp_path / "report.txt"
        export_combined_txt([], [], "Important findings here.", str(out))
        content = out.read_text(encoding="utf-8")

        assert "Important findings here." in content

    def test_footer_present(self, tmp_path):
        """Should include CasePrepd footer in TXT output."""
        from src.core.export.combined_exporter import export_combined_txt

        out = tmp_path / "report.txt"
        export_combined_txt([], [], "", str(out))
        content = out.read_text(encoding="utf-8")

        assert "CasePrepd" in content

    def test_empty_data_still_produces_file(self, tmp_path):
        """Should create file even with all empty inputs."""
        from src.core.export.combined_exporter import export_combined_txt

        out = tmp_path / "report.txt"
        result = export_combined_txt([], [], "", str(out))

        assert result is True
        assert out.exists()
