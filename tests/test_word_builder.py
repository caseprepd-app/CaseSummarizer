"""
Tests for src/core/export/word_builder.py.

Covers Word document builder using python-docx.
"""

from unittest.mock import patch


class TestWordDocumentBuilder:
    """Tests for WordDocumentBuilder methods."""

    def test_add_heading(self):
        """add_heading should add a heading to the document."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_heading("Test Title", level=1)

        # Document should have the heading
        assert len(builder.doc.paragraphs) > 0

    def test_add_paragraph_basic(self):
        """add_paragraph should add a paragraph with text."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_paragraph("Simple text.")

        para = builder.doc.paragraphs[-1]
        assert para.runs[0].text == "Simple text."

    def test_add_paragraph_bold(self):
        """add_paragraph with bold=True should set run to bold."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_paragraph("Bold text.", bold=True)

        para = builder.doc.paragraphs[-1]
        assert para.runs[0].bold is True

    def test_add_paragraph_italic(self):
        """add_paragraph with italic=True should set run to italic."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_paragraph("Italic text.", italic=True)

        para = builder.doc.paragraphs[-1]
        assert para.runs[0].italic is True

    def test_add_table_empty_headers(self):
        """add_table with empty headers should return without error."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_table([], [])
        assert len(builder.doc.tables) == 0

    def test_add_table_with_data(self):
        """add_table should create a table with headers and data rows."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])

        assert len(builder.doc.tables) == 1
        table = builder.doc.tables[0]
        # Header + 2 data rows = 3 rows
        assert len(table.rows) == 3

    def test_add_table_header_text(self):
        """Table headers should contain the specified text."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_table(["Col1", "Col2"], [["a", "b"]])

        table = builder.doc.tables[0]
        assert table.rows[0].cells[0].text == "Col1"
        assert table.rows[0].cells[1].text == "Col2"

    def test_add_separator(self):
        """add_separator should add a paragraph (separator line)."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        initial_count = len(builder.doc.paragraphs)
        builder.add_separator()
        assert len(builder.doc.paragraphs) > initial_count

    def test_save_creates_file(self, tmp_path):
        """save should create a .docx file on disk."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_heading("Test", level=1)
        builder.add_paragraph("Content.")

        out = tmp_path / "test.docx"
        builder.save(str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_raises_on_locked_file(self, tmp_path):
        """save should raise OSError with helpful message on failure."""
        import pytest

        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_paragraph("test")

        with patch.object(builder.doc, "save", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError, match="open in another"):
                builder.save(str(tmp_path / "locked.docx"))

    def test_default_font_is_calibri(self):
        """Default font should be Calibri 11pt."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        style = builder.doc.styles["Normal"]
        assert style.font.name == "Calibri"
