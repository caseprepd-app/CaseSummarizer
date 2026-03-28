"""
Tests for src/core/export/pdf_builder.py.

Covers PDF document builder including sanitization and table generation.
"""

from unittest.mock import patch


class TestSanitizeForLatin1:
    """Tests for _sanitize_for_latin1() text cleaning."""

    def test_ascii_text_unchanged(self):
        """ASCII text should pass through unchanged."""
        from src.core.export.pdf_builder import _sanitize_for_latin1

        assert _sanitize_for_latin1("Hello world") == "Hello world"

    def test_accented_chars_stripped(self):
        """Accented characters should be decomposed to ASCII base."""
        from src.core.export.pdf_builder import _sanitize_for_latin1

        result = _sanitize_for_latin1("café résumé")
        assert "cafe" in result
        assert "resume" in result

    def test_non_latin1_replaced_with_question_mark(self):
        """Characters outside Latin-1 should become '?'."""
        from src.core.export.pdf_builder import _sanitize_for_latin1

        result = _sanitize_for_latin1("Hello \u4e16\u754c")  # Chinese chars
        assert "?" in result

    def test_empty_string(self):
        """Empty string should return empty string."""
        from src.core.export.pdf_builder import _sanitize_for_latin1

        assert _sanitize_for_latin1("") == ""


class TestPdfDocumentBuilder:
    """Tests for PdfDocumentBuilder methods."""

    def test_add_heading_sets_font(self):
        """add_heading should set bold font for headings."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder(title="Test")
        builder.add_heading("Test Heading", level=1)
        # Should not raise; verify PDF has content
        assert builder.pdf.page > 0

    def test_add_paragraph_basic(self):
        """add_paragraph should add text without error."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_paragraph("Simple paragraph text.")
        assert builder.pdf.page > 0

    def test_add_paragraph_bold_italic(self):
        """add_paragraph with bold and italic should not error."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_paragraph("Bold italic text.", bold=True, italic=True)
        assert builder.pdf.page > 0

    def test_add_table_empty_headers_returns_early(self):
        """add_table with empty headers should return without error."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_table([], [])  # Should not raise

    def test_add_table_with_data(self):
        """add_table should render headers and rows."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_table(["Name", "Score"], [["Alice", "0.9"], ["Bob", "0.8"]])
        assert builder.pdf.page > 0

    def test_add_separator(self):
        """add_separator should draw a line without error."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_separator()
        assert builder.pdf.page > 0

    def test_save_creates_file(self, tmp_path):
        """save should create a PDF file on disk."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder(title="Test PDF")
        builder.add_heading("Title", level=1)
        builder.add_paragraph("Content.")

        out = tmp_path / "test.pdf"
        builder.save(str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_raises_on_locked_file(self, tmp_path):
        """save should raise OSError with helpful message when file is locked."""
        import pytest

        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_paragraph("test")

        out = tmp_path / "locked.pdf"
        with patch.object(builder.pdf, "output", side_effect=OSError("locked")):
            with pytest.raises(OSError, match="open in another"):
                builder.save(str(out))

    def test_long_header_truncated_in_table(self):
        """Table headers longer than 20 chars should be truncated."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        long_header = "A" * 25
        builder.add_table([long_header], [["value"]])
        # Should not raise; truncation happens internally

    def test_unicode_in_heading(self):
        """Unicode characters in headings should be sanitized."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_heading("Test \u2014 Heading", level=2)
        assert builder.pdf.page > 0
