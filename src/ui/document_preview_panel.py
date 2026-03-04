"""
Document Preview Panel for CasePrepd.

Displays extracted/preprocessed text for a selected document with metadata header.
Integrated into the right panel as the "Document" tab in DynamicOutputWidget.
Large documents are paginated into ~300-word sections to prevent GUI freezes.
"""

import logging

import customtkinter as ctk

from src.ui.document_paginator import split_into_sections
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
        row 3: Navigation bar (hidden when single section)
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
        self._sections = []
        self._current_section = 0

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

        # Row 3: Navigation bar (hidden initially)
        self._nav_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._nav_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 5))
        self._nav_frame.grid_remove()
        self._nav_frame.grid_columnconfigure(1, weight=1)

        self._prev_btn = ctk.CTkButton(
            self._nav_frame, text="\u25c0 Prev", width=80, command=self._show_prev_section
        )
        self._prev_btn.grid(row=0, column=0, padx=(0, 10))

        self._section_label = ctk.CTkLabel(self._nav_frame, text="", font=FONTS["small"])
        self._section_label.grid(row=0, column=1)

        self._next_btn = ctk.CTkButton(
            self._nav_frame, text="Next \u25b6", width=80, command=self._show_next_section
        )
        self._next_btn.grid(row=0, column=2, padx=(10, 0))

    @property
    def current_filename(self):
        """The filename of the currently displayed document, or None."""
        return self._current_filename

    def display_document(self, result):
        """
        Populate the panel with document metadata and paginated text.

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

        # Paginate and show first section
        self._sections = split_into_sections(text)
        self._current_section = 0
        self._status_label.grid_remove()
        self._textbox.grid()
        self._show_current_section()
        self._update_nav_bar()

        logger.debug(
            "Previewing document: %s (%d chars, %d sections)",
            filename,
            len(text),
            len(self._sections),
        )

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

    def _show_current_section(self):
        """Insert the current section's text into the textbox."""
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        if self._sections:
            self._textbox.insert("1.0", self._sections[self._current_section])
        self._textbox.configure(state="disabled")

    def _update_nav_bar(self):
        """Show/hide nav bar and update label + button states."""
        total = len(self._sections)
        if total <= 1:
            self._nav_frame.grid_remove()
            return

        self._nav_frame.grid()
        self._section_label.configure(text=f"Section {self._current_section + 1} of {total}")
        self._prev_btn.configure(state="normal" if self._current_section > 0 else "disabled")
        self._next_btn.configure(
            state="normal" if self._current_section < total - 1 else "disabled"
        )

    def _show_prev_section(self):
        """Navigate to the previous section."""
        if self._current_section > 0:
            self._current_section -= 1
            self._show_current_section()
            self._update_nav_bar()

    def _show_next_section(self):
        """Navigate to the next section."""
        if self._current_section < len(self._sections) - 1:
            self._current_section += 1
            self._show_current_section()
            self._update_nav_bar()

    def clear(self):
        """Reset to empty placeholder state."""
        self._current_filename = None
        self._sections = []
        self._current_section = 0
        self._find_bar.grid_remove()
        self._header_frame.grid_remove()
        self._textbox.grid_remove()
        self._nav_frame.grid_remove()
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
        self._nav_frame.grid_remove()
        self._status_label.configure(text=message)
        self._status_label.grid()
