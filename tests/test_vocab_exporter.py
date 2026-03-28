"""
Tests for src/core/export/vocab_exporter.py.

Covers vocabulary export to DocumentBuilder and TXT formats.
"""

from unittest.mock import MagicMock

from src.core.vocab_schema import VF


def _make_vocab(n=3):
    """Create sample vocabulary data."""
    return [
        {
            VF.TERM: f"word_{i}",
            VF.QUALITY_SCORE: round(0.95 - i * 0.05, 2),
            VF.IS_PERSON: VF.YES if i == 0 else VF.NO,
            VF.FOUND_BY: "NER",
            VF.OCCURRENCES: 5 - i,
        }
        for i in range(n)
    ]


def _make_builder():
    """Create a mock DocumentBuilder."""
    b = MagicMock()
    b.add_heading = MagicMock()
    b.add_paragraph = MagicMock()
    b.add_table = MagicMock()
    b.add_separator = MagicMock()
    return b


class TestExportVocabulary:
    """Tests for export_vocabulary() with a DocumentBuilder."""

    def test_adds_title_heading(self):
        """Should add the provided title as a heading."""
        from src.core.export.vocab_exporter import export_vocabulary

        builder = _make_builder()
        export_vocabulary(_make_vocab(), builder, title="Custom Title")

        builder.add_heading.assert_any_call("Custom Title", level=1)

    def test_adds_table_with_correct_row_count(self):
        """Should add a table with one row per vocab entry."""
        from src.core.export.vocab_exporter import export_vocabulary

        builder = _make_builder()
        vocab = _make_vocab(4)
        export_vocabulary(vocab, builder)

        builder.add_table.assert_called_once()
        _, rows = builder.add_table.call_args[0]
        assert len(rows) == 4

    def test_basic_headers_without_details(self):
        """Without include_details, headers should be basic columns."""
        from src.core.export.vocab_exporter import export_vocabulary

        builder = _make_builder()
        export_vocabulary(_make_vocab(), builder, include_details=False)

        headers, _ = builder.add_table.call_args[0]
        assert VF.TERM in headers
        assert "Score" in headers
        assert VF.NER not in headers

    def test_detail_headers_with_include_details(self):
        """With include_details=True, algorithm columns should appear."""
        from src.core.export.vocab_exporter import export_vocabulary

        builder = _make_builder()
        export_vocabulary(_make_vocab(), builder, include_details=True)

        headers, _ = builder.add_table.call_args[0]
        assert VF.NER in headers
        assert VF.RAKE in headers

    def test_num_docs_column_excluded_for_single_doc(self):
        """is_single_doc=True should omit the # Docs column."""
        from src.core.export.vocab_exporter import export_vocabulary

        builder = _make_builder()
        export_vocabulary(_make_vocab(), builder, is_single_doc=True)

        headers, _ = builder.add_table.call_args[0]
        assert VF.NUM_DOCS not in headers

    def test_num_docs_column_included_for_multi_doc(self):
        """is_single_doc=False should include the # Docs column."""
        from src.core.export.vocab_exporter import export_vocabulary

        builder = _make_builder()
        export_vocabulary(_make_vocab(), builder, is_single_doc=False)

        headers, _ = builder.add_table.call_args[0]
        assert VF.NUM_DOCS in headers

    def test_empty_data_shows_no_data_message(self):
        """Empty vocab_data should show 'No vocabulary data' message."""
        from src.core.export.vocab_exporter import export_vocabulary

        builder = _make_builder()
        export_vocabulary([], builder)

        builder.add_table.assert_not_called()
        para_texts = [str(c[0][0]) for c in builder.add_paragraph.call_args_list]
        assert any("No vocabulary" in t for t in para_texts)

    def test_adds_footer(self):
        """Should add CasePrepd footer."""
        from src.core.export.vocab_exporter import export_vocabulary

        builder = _make_builder()
        export_vocabulary(_make_vocab(), builder)

        para_calls = [str(c) for c in builder.add_paragraph.call_args_list]
        assert any("CasePrepd" in c for c in para_calls)


class TestExportVocabularyTxt:
    """Tests for export_vocabulary_txt() plain text export."""

    def test_creates_file(self, tmp_path):
        """Should create a text file with terms."""
        from src.core.export.vocab_exporter import export_vocabulary_txt

        out = tmp_path / "vocab.txt"
        result = export_vocabulary_txt(_make_vocab(2), str(out))

        assert result is True
        assert out.exists()

    def test_contains_term_names(self, tmp_path):
        """Output file should contain each term name."""
        from src.core.export.vocab_exporter import export_vocabulary_txt

        out = tmp_path / "vocab.txt"
        export_vocabulary_txt(_make_vocab(3), str(out))
        content = out.read_text(encoding="utf-8")

        assert "word_0" in content
        assert "word_1" in content
        assert "word_2" in content

    def test_empty_data_creates_empty_file(self, tmp_path):
        """Empty vocab_data should create an empty file."""
        from src.core.export.vocab_exporter import export_vocabulary_txt

        out = tmp_path / "vocab.txt"
        result = export_vocabulary_txt([], str(out))

        assert result is True
        assert out.read_text(encoding="utf-8") == ""

    def test_returns_false_on_write_error(self, tmp_path):
        """Should return False when file writing fails."""
        from src.core.export.vocab_exporter import export_vocabulary_txt

        # Use an invalid path
        result = export_vocabulary_txt(_make_vocab(), "/nonexistent/dir/file.txt")

        assert result is False
