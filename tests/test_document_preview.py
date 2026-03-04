"""
Tests for the Document Preview Panel feature.

Covers:
1. DocumentPreviewPanel — text source selection, metadata, clear
2. FileReviewTable — on_select callback for data column clicks
3. MainWindow — _on_file_selected data flow, remove/clear guards
"""

from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel():
    """Create a DocumentPreviewPanel via __new__ with mocked internals."""
    from src.ui.document_preview_panel import (
        DocumentPreviewPanel,
    )

    panel = DocumentPreviewPanel.__new__(DocumentPreviewPanel)
    panel._current_filename = None
    panel._sections = []
    panel._current_section = 0
    panel._find_bar = MagicMock()
    panel._header_frame = MagicMock()
    panel._filename_label = MagicMock()
    panel._detail_label = MagicMock()
    panel._textbox = MagicMock()
    panel._status_label = MagicMock()
    panel._nav_frame = MagicMock()
    panel._prev_btn = MagicMock()
    panel._next_btn = MagicMock()
    panel._section_label = MagicMock()
    return panel


def _make_table(on_remove=None, on_select=None):
    """Create a FileReviewTable via __new__ with mocked internals."""
    from src.ui.widgets import FileReviewTable

    table = FileReviewTable.__new__(FileReviewTable)
    table._on_remove = on_remove
    table._on_select = on_select
    table.column_map = {
        "filename": ("Filename", 300),
        "status": ("Status", 100),
        "method": ("Method", 100),
        "confidence": ("Confidence", 100),
        "pages": ("Pages", 50),
        "size": ("Size", 80),
    }
    table.file_item_map = {"test.pdf": "item1"}
    table._result_data = {}
    table._hovered_row = None
    table._tooltip_window = None
    table._remove_icon = MagicMock()
    table._drop_zone = MagicMock()
    table._drop_zone.winfo_ismapped.return_value = False
    table.tree = MagicMock()
    return table


def _sample_result(filename="report.pdf", **overrides):
    """Build a processing result dict with optional overrides."""
    result = {
        "filename": filename,
        "status": "success",
        "confidence": 91,
        "method": "pdfplumber",
        "page_count": 12,
        "word_count": 8450,
        "file_size": 204800,
        "preprocessed_text": "This is the cleaned preprocessed text.",
        "extracted_text": "This is the raw extracted text.",
    }
    result.update(overrides)
    return result


# ===========================================================================
# 1. DocumentPreviewPanel — widget logic
# ===========================================================================


class TestDisplayDocument:
    """Tests for DocumentPreviewPanel.display_document()."""

    def test_display_prefers_preprocessed_text(self):
        """preprocessed_text is used when both fields are available."""
        panel = _make_panel()
        result = _sample_result()

        panel.display_document(result)

        # Textbox should receive preprocessed text
        panel._textbox.insert.assert_called_once()
        inserted_text = panel._textbox.insert.call_args[0][1]
        assert "cleaned preprocessed" in inserted_text

    def test_display_falls_back_to_extracted_text(self):
        """Falls back to extracted_text when preprocessed_text is absent."""
        panel = _make_panel()
        result = _sample_result(preprocessed_text=None)

        panel.display_document(result)

        inserted_text = panel._textbox.insert.call_args[0][1]
        assert "raw extracted" in inserted_text

    def test_display_empty_text_shows_status(self):
        """Empty text shows a 'no text extracted' message."""

        panel = _make_panel()
        result = _sample_result(preprocessed_text="", extracted_text="")

        panel.display_document(result)

        # Status label shown, textbox not populated
        panel._status_label.configure.assert_called()
        status_text = panel._status_label.configure.call_args[1]["text"]
        assert "report.pdf" in status_text

    def test_metadata_format(self):
        """Detail line includes pages, words, confidence, method, and source indicator."""
        panel = _make_panel()
        result = _sample_result()

        panel.display_document(result)

        detail_text = panel._detail_label.configure.call_args[1]["text"]
        assert "12 pages" in detail_text
        assert "8,450 words" in detail_text
        assert "91%" in detail_text
        assert "Pdfplumber" in detail_text
        assert "(Preprocessed)" in detail_text

    def test_clear_resets_state(self):
        """clear() sets current_filename to None and shows placeholder."""
        from src.ui.document_preview_panel import _PLACEHOLDER_TEXT

        panel = _make_panel()
        result = _sample_result()
        panel.display_document(result)
        assert panel.current_filename == "report.pdf"

        panel.clear()

        assert panel.current_filename is None
        panel._status_label.configure.assert_called()
        status_text = panel._status_label.configure.call_args[1]["text"]
        assert status_text == _PLACEHOLDER_TEXT


# ===========================================================================
# 2. FileReviewTable — on_select callback
# ===========================================================================


class TestFileTableSelect:
    """Tests for FileReviewTable click → on_select callback."""

    def test_click_data_column_fires_on_select(self):
        """Clicking a data column (not #0) fires the on_select callback."""
        callback = MagicMock()
        table = _make_table(on_select=callback)
        table.tree.identify_row.return_value = "item1"
        table.tree.identify_column.return_value = "#1"
        table.tree.item.return_value = ("test.pdf", "Ready", "Digital")

        event = MagicMock()
        event.x = 100
        event.y = 20
        table._on_click(event)

        callback.assert_called_once_with("test.pdf")
        table.tree.selection_set.assert_called_once_with("item1")

    def test_click_remove_column_does_not_fire_on_select(self):
        """Clicking column #0 (✕) fires on_remove, not on_select."""
        select_cb = MagicMock()
        remove_cb = MagicMock()
        table = _make_table(on_remove=remove_cb, on_select=select_cb)
        table.tree.identify_row.return_value = "item1"
        table.tree.identify_column.return_value = "#0"
        table.tree.item.return_value = ("test.pdf",)

        event = MagicMock()
        event.x = 10
        event.y = 20
        table._on_click(event)

        remove_cb.assert_called_once_with("test.pdf")
        select_cb.assert_not_called()

    def test_click_empty_row_fires_nothing(self):
        """Clicking on blank area (no row) fires nothing."""
        select_cb = MagicMock()
        remove_cb = MagicMock()
        table = _make_table(on_remove=remove_cb, on_select=select_cb)
        table.tree.identify_row.return_value = ""
        table.tree.identify_column.return_value = "#1"

        event = MagicMock()
        event.x = 100
        event.y = 200
        table._on_click(event)

        select_cb.assert_not_called()
        remove_cb.assert_not_called()


# ===========================================================================
# 3. MainWindow — data flow
# ===========================================================================


class TestFileSelectedFlow:
    """Tests for MainWindow._on_file_selected and remove guards."""

    def test_file_selected_finds_result(self):
        """_on_file_selected looks up result and calls show_document_preview."""
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub.processing_results = [_sample_result("a.pdf"), _sample_result("b.pdf")]
        stub.output_display = MagicMock()

        MainWindow._on_file_selected(stub, "b.pdf")

        stub.output_display.show_document_preview.assert_called_once()
        passed_result = stub.output_display.show_document_preview.call_args[0][0]
        assert passed_result["filename"] == "b.pdf"

    def test_file_selected_missing_no_crash(self):
        """_on_file_selected with unknown filename doesn't crash."""
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub.processing_results = [_sample_result("a.pdf")]
        stub.output_display = MagicMock()

        MainWindow._on_file_selected(stub, "unknown.pdf")

        stub.output_display.show_document_preview.assert_not_called()

    def test_removed_file_clears_preview(self):
        """Removing the currently previewed file clears the Document tab."""

        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub.selected_files = ["/path/to/test.pdf"]
        stub.processing_results = [_sample_result("test.pdf")]
        stub.file_table = MagicMock()
        stub.output_display = MagicMock()
        stub.output_display.document_preview_filename = "test.pdf"
        stub.set_status = MagicMock()

        MainWindow._remove_file(stub, "test.pdf")

        stub.output_display.clear_document_preview.assert_called_once()

    def test_removed_other_file_preserves_preview(self):
        """Removing a different file does NOT clear the preview."""

        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub.selected_files = ["/path/to/a.pdf", "/path/to/b.pdf"]
        stub.processing_results = [_sample_result("a.pdf"), _sample_result("b.pdf")]
        stub.file_table = MagicMock()
        stub.output_display = MagicMock()
        stub.output_display.document_preview_filename = "a.pdf"
        stub.set_status = MagicMock()

        MainWindow._remove_file(stub, "b.pdf")

        stub.output_display.clear_document_preview.assert_not_called()


# ===========================================================================
# 4. DocumentPreviewPanel — pagination
# ===========================================================================


def _long_text(word_count=1000):
    """Generate multi-paragraph text exceeding the default section size."""
    paragraphs = []
    words_left = word_count
    while words_left > 0:
        chunk = min(80, words_left)
        paragraphs.append(" ".join(["word"] * chunk))
        words_left -= chunk
    return "\n".join(paragraphs)


class TestPagination:
    """Tests for DocumentPreviewPanel pagination behavior."""

    def test_short_text_hides_nav_bar(self):
        """A short document hides the navigation bar."""
        panel = _make_panel()
        result = _sample_result(preprocessed_text="Short text here.")
        panel.display_document(result)

        panel._nav_frame.grid_remove.assert_called()

    def test_long_text_shows_nav_bar(self):
        """A long document shows the navigation bar."""
        panel = _make_panel()
        result = _sample_result(preprocessed_text=_long_text(1000))
        panel.display_document(result)

        panel._nav_frame.grid.assert_called()
        assert len(panel._sections) > 1

    def test_section_label_format(self):
        """Section label reads 'Section 1 of N'."""
        panel = _make_panel()
        result = _sample_result(preprocessed_text=_long_text(1000))
        panel.display_document(result)

        label_text = panel._section_label.configure.call_args[1]["text"]
        assert label_text.startswith("Section 1 of ")
        total = len(panel._sections)
        assert label_text == f"Section 1 of {total}"

    def test_next_advances_section(self):
        """Clicking next advances to section 2."""
        panel = _make_panel()
        result = _sample_result(preprocessed_text=_long_text(1000))
        panel.display_document(result)

        panel._show_next_section()

        assert panel._current_section == 1
        label_text = panel._section_label.configure.call_args[1]["text"]
        assert "Section 2 of" in label_text

    def test_prev_goes_back(self):
        """After advancing, prev goes back to section 1."""
        panel = _make_panel()
        result = _sample_result(preprocessed_text=_long_text(1000))
        panel.display_document(result)

        panel._show_next_section()
        panel._show_prev_section()

        assert panel._current_section == 0

    def test_prev_disabled_at_start(self):
        """Prev button is disabled on the first section."""
        panel = _make_panel()
        result = _sample_result(preprocessed_text=_long_text(1000))
        panel.display_document(result)

        # Check the last configure call on _prev_btn
        panel._prev_btn.configure.assert_called()
        last_call = panel._prev_btn.configure.call_args
        assert last_call[1]["state"] == "disabled"

    def test_next_disabled_at_end(self):
        """Next button is disabled on the last section."""
        panel = _make_panel()
        result = _sample_result(preprocessed_text=_long_text(1000))
        panel.display_document(result)

        # Navigate to last section
        while panel._current_section < len(panel._sections) - 1:
            panel._show_next_section()

        last_call = panel._next_btn.configure.call_args
        assert last_call[1]["state"] == "disabled"

    def test_only_current_section_in_textbox(self):
        """Textbox receives only the current section, not the full text."""
        panel = _make_panel()
        result = _sample_result(preprocessed_text=_long_text(1000))
        panel.display_document(result)

        inserted_text = panel._textbox.insert.call_args[0][1]
        assert len(inserted_text.split()) <= 350

    def test_clear_resets_pagination(self):
        """clear() resets sections and hides nav bar."""
        panel = _make_panel()
        result = _sample_result(preprocessed_text=_long_text(1000))
        panel.display_document(result)

        panel.clear()

        assert panel._sections == []
        assert panel._current_section == 0
        panel._nav_frame.grid_remove.assert_called()
