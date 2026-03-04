"""
Document Preview Panel for CasePrepd.

Displays extracted/preprocessed text for a selected document with metadata header.
Integrated into the right panel as the "Document" tab in DynamicOutputWidget.
"""

import logging

import customtkinter as ctk

from src.ui.theme import COLORS, FONTS, TEXTBOX_STYLES

logger = logging.getLogger(__name__)

# Placeholder shown when no document is selected
_PLACEHOLDER_TEXT = "Click a document to preview extracted text"
_NO_TEXT_TEMPLATE = "No text extracted from '{}'"
_NOT_YET_TEMPLATE = "Text not yet available \u2014 run extraction first"


class DocumentPreviewPanel(ctk.CTkFrame):
    """
    Read-only preview of a document's extracted/preprocessed text.

    Layout:
        row 0: TextFindBar (hidden, shown on Ctrl+F)
        row 1: Metadata header (filename + detail line)
        row 2: CTkTextbox (read-only, word wrap) OR status label
    """

    def __init__(self, master, **kwargs):
        """
        Args:
            master: Parent widget (tab frame from CTkTabview).
        """
        super().__init__(master, fg_color="transparent", **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # Textbox expands

        self._current_filename = None

        # Row 0: Find bar (hidden initially)
        from src.ui.text_find_bar import TextFindBar

        self._textbox = ctk.CTkTextbox(self, wrap="word", **TEXTBOX_STYLES["default"])
        self._find_bar = TextFindBar(self, self._textbox)
        self._find_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=(2, 0))
        self._find_bar.grid_remove()

        # Row 1: Metadata header
        self._header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._header_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(8, 2))
        self._header_frame.grid_remove()

        self._filename_label = ctk.CTkLabel(
            self._header_frame, text="", font=FONTS["heading"], anchor="w"
        )
        self._filename_label.pack(anchor="w")

        self._detail_label = ctk.CTkLabel(
            self._header_frame,
            text="",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            anchor="w",
        )
        self._detail_label.pack(anchor="w")

        # Row 2: Textbox (hidden initially)
        self._textbox.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self._textbox.grid_remove()

        # Row 2 alt: Status label (shown initially)
        self._status_label = ctk.CTkLabel(
            self,
            text=_PLACEHOLDER_TEXT,
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            justify="center",
            wraplength=0,
        )
        self._status_label.grid(row=2, column=0, sticky="nsew", padx=20, pady=50)

    @property
    def current_filename(self):
        """The filename of the currently displayed document, or None."""
        return self._current_filename

    def display_document(self, result):
        """
        Populate the panel with document metadata and text.

        Args:
            result: Dict with keys like filename, preprocessed_text,
                    extracted_text, page_count, word_count, confidence, method.
        """
        filename = result.get("filename", "Unknown")
        self._current_filename = filename

        # Hide find bar on document switch
        self._find_bar.grid_remove()

        # Choose text source: preprocessed preferred, fallback to extracted
        text = result.get("preprocessed_text") or result.get("extracted_text") or ""

        if not text.strip():
            self._show_status(_NO_TEXT_TEMPLATE.format(filename))
            return

        # Build metadata detail line
        detail = self._build_detail_line(result)

        # Show header
        self._filename_label.configure(text=filename)
        self._detail_label.configure(text=detail)
        self._header_frame.grid()

        # Show textbox with content
        self._status_label.grid_remove()
        self._textbox.grid()
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.insert("1.0", text)
        self._textbox.configure(state="disabled")

        logger.debug("Previewing document: %s (%d chars)", filename, len(text))

    def _build_detail_line(self, result):
        """
        Build a metadata summary string like:
        '12 pages | 8,450 words | 91% confidence | Pdfplumber | (Preprocessed)'

        Args:
            result: The extraction result dict.

        Returns:
            Formatted detail string.
        """
        parts = []

        page_count = result.get("page_count")
        if page_count:
            parts.append(f"{page_count} pages")

        word_count = result.get("word_count")
        if word_count:
            parts.append(f"{word_count:,} words")

        confidence = result.get("confidence")
        if confidence:
            parts.append(f"{confidence}% confidence")

        method = result.get("method")
        if method:
            parts.append(method.replace("_", " ").title())

        # Indicate which text source is shown
        if result.get("preprocessed_text"):
            parts.append("(Preprocessed)")
        elif result.get("extracted_text"):
            parts.append("(Raw)")

        return " | ".join(parts)

    def clear(self):
        """Reset to empty placeholder state."""
        self._current_filename = None
        self._find_bar.grid_remove()
        self._header_frame.grid_remove()
        self._textbox.grid_remove()
        self._status_label.configure(text=_PLACEHOLDER_TEXT)
        self._status_label.grid()

        # Clear textbox content
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.configure(state="disabled")

    def show_find_bar(self):
        """Show the Ctrl+F find bar."""
        self._find_bar.show()

    def hide_find_bar(self):
        """Hide the find bar."""
        self._find_bar.grid_remove()

    def _show_status(self, message):
        """Show a status message instead of document content."""
        self._header_frame.grid_remove()
        self._textbox.grid_remove()
        self._status_label.configure(text=message)
        self._status_label.grid()
