"""
Term-in-Context Viewer Dialog for CasePrepd.

Shows document excerpts where a vocabulary term appears, grouped by document
with the term highlighted. Opened via right-click "View in Context" on
vocabulary table rows.

The extract_snippets() function is pure text processing with no UI dependencies,
making it independently testable.
"""

import logging
import re

import customtkinter as ctk

from src.ui.base_dialog import BaseModalDialog
from src.ui.text_find_bar import TextFindBar
from src.ui.theme import BUTTON_STYLES, COLORS

logger = logging.getLogger(__name__)

# Tag styles for the context textbox (tuples resolved at call time via resolve_tags)
_CONTEXT_TAGS = {
    "term_highlight": {
        "background": COLORS["term_highlight_bg"],
        "foreground": COLORS["term_highlight_fg"],
    },
    "doc_header": {"foreground": COLORS["doc_header_fg"], "font": ("Segoe UI", 13, "bold")},
    "more_note": {"foreground": COLORS["more_note_fg"], "font": ("Segoe UI", 11, "italic")},
}


def _expand_to_word_boundary(text: str, pos: int, direction: str, max_chars: int) -> int:
    """
    Expand a position left or right to the nearest word boundary.

    Args:
        text: Full document text
        pos: Starting character position
        direction: "left" or "right"
        max_chars: Maximum characters to expand

    Returns:
        Adjusted position at a word boundary (space character).
    """
    if direction == "left":
        target = max(0, pos - max_chars)
        # Walk forward to next space (don't cut mid-word)
        if target > 0:
            while target < pos and text[target] != " ":
                target += 1
        return target
    else:
        target = min(len(text), pos + max_chars)
        # Walk backward to previous space (don't cut mid-word)
        if target < len(text):
            while target > pos and text[target - 1] != " ":
                target -= 1
            # If we couldn't find a space, use the hard limit
            if target == pos:
                target = min(len(text), pos + max_chars)
        return target


def extract_snippets(
    term: str,
    documents: list[dict],
    context_chars: int = 80,
    max_per_doc: int = 10,
) -> list[dict]:
    """
    Extract text snippets showing a term in context across documents.

    Args:
        term: The term to search for
        documents: List of document dicts with 'filename' and
                   'preprocessed_text' or 'extracted_text' keys
        context_chars: Characters of context on each side (~80)
        max_per_doc: Maximum snippets per document (default 10)

    Returns:
        List of result dicts sorted by total_count descending:
        [
            {
                "filename": "complaint.pdf",
                "total_count": 23,
                "snippets": [
                    {"before": "...the ", "term": "negligence", "after": " claim..."},
                ],
                "remaining": 13,
            },
        ]
    """
    if not term or not documents:
        return []

    pattern = re.compile(re.escape(term), re.IGNORECASE)
    results = []

    for doc in documents:
        text = doc.get("preprocessed_text") or doc.get("extracted_text", "")
        if not text:
            continue

        filename = doc.get("filename", "Unknown")
        matches = list(pattern.finditer(text))
        if not matches:
            continue

        total_count = len(matches)
        snippets = []

        for match in matches[:max_per_doc]:
            start, end = match.start(), match.end()

            # Expand context to word boundaries
            ctx_start = _expand_to_word_boundary(text, start, "left", context_chars)
            ctx_end = _expand_to_word_boundary(text, end, "right", context_chars)

            before = text[ctx_start:start]
            matched_term = text[start:end]
            after = text[end:ctx_end]

            # Add ellipsis when not at document boundaries
            if ctx_start > 0:
                before = "..." + before
            if ctx_end < len(text):
                after = after + "..."

            snippets.append(
                {
                    "before": before,
                    "term": matched_term,
                    "after": after,
                }
            )

        remaining = max(0, total_count - max_per_doc)
        results.append(
            {
                "filename": filename,
                "total_count": total_count,
                "snippets": snippets,
                "remaining": remaining,
            }
        )

    # Sort by total_count descending
    results.sort(key=lambda r: r["total_count"], reverse=True)
    return results


class ContextViewerDialog(BaseModalDialog):
    """
    Modal dialog showing term occurrences across loaded documents.

    Displays snippets grouped by document with the term highlighted.
    Includes a TextFindBar for searching within the results.

    Attributes:
        _term_data: Dict with at least 'Term' key from the vocabulary table
        _processing_results: List of document dicts with extracted text
    """

    def __init__(self, parent, term_data: dict, processing_results: list[dict]):
        """
        Initialize the context viewer dialog.

        Args:
            parent: Parent window
            term_data: Term dict from vocabulary table (needs 'Term' key)
            processing_results: List of document result dicts
        """
        self._term_data = term_data
        self._processing_results = processing_results
        self._term = term_data.get("Term", "")

        # Extract snippets before building UI (needed for title)
        self._results = extract_snippets(self._term, processing_results)
        total = sum(r["total_count"] for r in self._results)
        doc_count = len(self._results)

        title = f'Context for "{self._term}" \u2014 {total} in {doc_count} document'
        if doc_count != 1:
            title += "s"

        super().__init__(
            parent=parent,
            title=title,
            width=700,
            height=500,
            min_width=500,
            min_height=350,
        )
        self._create_ui()
        self._populate_content()

    def _create_ui(self):
        """Build the dialog layout: find bar + textbox + close button."""
        # Main content frame
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=12, pady=(8, 4))
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)

        # Textbox (read-only)
        self._textbox = ctk.CTkTextbox(content, wrap="word", font=("Consolas", 12))
        self._textbox.grid(row=1, column=0, sticky="nsew")

        # Find bar (hidden initially, above textbox)
        self._find_bar = TextFindBar(content, self._textbox)
        self._find_bar.grid(row=0, column=0, sticky="ew")
        self._find_bar.grid_remove()

        # Bind Ctrl+F to show find bar
        self.bind("<Control-f>", lambda e: self._find_bar.show())

        # Configure text tags on the underlying tk.Text widget
        text_widget = self._textbox._textbox
        from src.ui.theme import resolve_tags

        for tag_name, tag_conf in resolve_tags(_CONTEXT_TAGS).items():
            text_widget.tag_configure(tag_name, cnf=tag_conf)

        # Close button
        close_btn = ctk.CTkButton(
            self, text="Close", width=100, command=self.close, **BUTTON_STYLES["secondary"]
        )
        close_btn.pack(pady=(4, 12))

    def _populate_content(self):
        """Fill the textbox with extracted snippets."""
        self._textbox.configure(state="normal")

        if not self._results:
            self._textbox.insert("end", "No occurrences found in loaded documents.")
            self._textbox.configure(state="disabled")
            return

        for i, doc_result in enumerate(self._results):
            if i > 0:
                self._textbox.insert("end", "\n")
            self._insert_document_section(doc_result)

        self._textbox.configure(state="disabled")

    def _insert_document_section(self, doc_result: dict):
        """
        Insert a document header and its snippets into the textbox.

        Args:
            doc_result: One entry from extract_snippets() output
        """
        text_widget = self._textbox._textbox
        filename = doc_result["filename"]
        total = doc_result["total_count"]

        # Document header
        header = f"{filename} \u2014 {total} occurrence"
        if total != 1:
            header += "s"
        header += "\n"

        start_idx = text_widget.index("end-1c")
        self._textbox.insert("end", header)
        end_idx = text_widget.index("end-1c")
        text_widget.tag_add("doc_header", start_idx, end_idx)

        # Separator line
        self._textbox.insert("end", "\u2500" * 40 + "\n")

        # Snippets
        for snippet in doc_result["snippets"]:
            self._insert_snippet(snippet)

        # "N more" note
        if doc_result["remaining"] > 0:
            note = f"  ... and {doc_result['remaining']} more occurrence"
            if doc_result["remaining"] != 1:
                note += "s"
            note += " in this document\n"

            start_idx = text_widget.index("end-1c")
            self._textbox.insert("end", note)
            end_idx = text_widget.index("end-1c")
            text_widget.tag_add("more_note", start_idx, end_idx)

    def _insert_snippet(self, snippet: dict):
        """
        Insert a single snippet with term highlighting.

        Args:
            snippet: Dict with 'before', 'term', 'after' keys
        """
        text_widget = self._textbox._textbox

        self._textbox.insert("end", "  ")
        self._textbox.insert("end", snippet["before"])

        # Insert term with highlight tag
        term_start = text_widget.index("end-1c")
        self._textbox.insert("end", snippet["term"])
        term_end = text_widget.index("end-1c")
        text_widget.tag_add("term_highlight", term_start, term_end)

        self._textbox.insert("end", snippet["after"] + "\n")
