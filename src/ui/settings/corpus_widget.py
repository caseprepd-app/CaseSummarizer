"""
Corpus Settings Widget for Settings Dialog.

Widget for corpus management within Settings dialog.
Session 82: Extracted from settings_widgets.py for modularity.

Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │  Corpus Management                                       ⓘ  │
    ├─────────────────────────────────────────────────────────────┤
    │  📚 What is a Corpus?                                       │
    │  A corpus is a collection of YOUR past transcripts...       │
    │                                                             │
    │  ✓ 100% local and offline - never leaves your machine       │
    │  ✓ Powers the BM25 vocabulary algorithm                     │
    ├─────────────────────────────────────────────────────────────┤
    │  Status:                                                    │
    │  • Active corpus: My Transcripts                            │
    │  • Documents: 23 files                                      │
    │  • BM25 Algorithm: ✓ Active (5+ documents)                  │
    ├─────────────────────────────────────────────────────────────┤
    │  [Manage Corpus...]                                         │
    └─────────────────────────────────────────────────────────────┘
"""

import customtkinter as ctk

from src.ui.settings.settings_widgets import TooltipIcon
from src.ui.theme import COLORS, FONTS


class CorpusSettingsWidget(ctk.CTkFrame):
    """
    Widget for corpus management within Settings dialog.

    Session 64: Provides corpus status overview and quick access to full
    corpus management dialog. Shows active corpus, document count, and
    BM25 algorithm status.
    """

    def __init__(self, parent, **kwargs):
        """
        Initialize the corpus settings widget.

        Args:
            parent: Parent widget.
            **kwargs: Additional CTkFrame arguments.
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        # Get corpus registry
        from src.core.vocabulary import get_corpus_registry

        self.registry = get_corpus_registry()

        self._setup_ui()

    def _setup_ui(self):
        """Create the widget layout."""
        self.grid_columnconfigure(0, weight=1)

        # Header with tooltip
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        header_label = ctk.CTkLabel(
            header_frame, text="Corpus Management", font=FONTS["heading"], anchor="w"
        )
        header_label.pack(side="left", padx=(0, 5))

        tooltip = TooltipIcon(
            header_frame,
            tooltip_text=(
                "A corpus is your collection of past transcripts used to identify "
                "case-specific vocabulary.\n\n"
                "The BM25 algorithm compares current documents against your corpus "
                "to find unusual terms - words that appear often in this case but "
                "rarely in your typical work.\n\n"
                "Requires 5+ documents to activate. More documents = better results."
            ),
        )
        tooltip.pack(side="left")

        # Educational section
        edu_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"), corner_radius=6)
        edu_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        edu_title = ctk.CTkLabel(
            edu_frame, text="📚 What is a Corpus?", font=FONTS["heading_sm"], anchor="w"
        )
        edu_title.pack(anchor="w", padx=15, pady=(10, 5))

        edu_text = ctk.CTkLabel(
            edu_frame,
            text=(
                "A corpus is a collection of YOUR past transcripts that helps LocalScribe\n"
                "understand which words are common in your work vs. unusual for a specific case."
            ),
            font=FONTS["body"],
            anchor="w",
            justify="left",
        )
        edu_text.pack(anchor="w", padx=15, pady=(0, 5))

        check1 = ctk.CTkLabel(
            edu_frame,
            text="✓ 100% local and offline - never leaves your machine",
            font=FONTS["body"],
            text_color=(COLORS["success"], COLORS["success_light"]),
            anchor="w",
        )
        check1.pack(anchor="w", padx=15, pady=(0, 2))

        check2 = ctk.CTkLabel(
            edu_frame,
            text="✓ Powers the BM25 vocabulary algorithm",
            font=FONTS["body"],
            text_color=(COLORS["success"], COLORS["success_light"]),
            anchor="w",
        )
        check2.pack(anchor="w", padx=15, pady=(0, 10))

        # Status section
        status_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"), corner_radius=6)
        status_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        status_title = ctk.CTkLabel(
            status_frame, text="Current Status", font=FONTS["heading_sm"], anchor="w"
        )
        status_title.pack(anchor="w", padx=15, pady=(10, 5))

        # Active corpus line
        self.active_corpus_label = ctk.CTkLabel(
            status_frame, text="• Active corpus: Loading...", font=FONTS["body"], anchor="w"
        )
        self.active_corpus_label.pack(anchor="w", padx=15, pady=(0, 2))

        # Document count line
        self.doc_count_label = ctk.CTkLabel(
            status_frame, text="• Documents: Loading...", font=FONTS["body"], anchor="w"
        )
        self.doc_count_label.pack(anchor="w", padx=15, pady=(0, 2))

        # BM25 status line
        self.bm25_status_label = ctk.CTkLabel(
            status_frame, text="• BM25 Algorithm: Loading...", font=FONTS["body"], anchor="w"
        )
        self.bm25_status_label.pack(anchor="w", padx=15, pady=(0, 10))

        # Manage button
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="w", pady=(5, 0))

        self.manage_btn = ctk.CTkButton(
            btn_frame,
            text="Manage Corpus...",
            command=self._open_corpus_dialog,
            width=160,
            height=32,
        )
        self.manage_btn.pack(side="left")

        # Refresh status on load
        self._refresh_status()

    def _refresh_status(self):
        """Update status labels from corpus registry."""
        try:
            active_name = self.registry.get_active_corpus()
            corpora = self.registry.list_corpora()

            # Find active corpus info
            active_corpus = None
            for c in corpora:
                if c.name == active_name:
                    active_corpus = c
                    break

            # Active corpus
            if active_corpus:
                self.active_corpus_label.configure(text=f"• Active corpus: {active_corpus.name}")
                doc_count = active_corpus.doc_count
            else:
                self.active_corpus_label.configure(text="• Active corpus: (none)")
                doc_count = 0

            # Document count
            self.doc_count_label.configure(text=f"• Documents: {doc_count} files")

            # BM25 status
            from src.config import CORPUS_MIN_DOCUMENTS

            if doc_count >= CORPUS_MIN_DOCUMENTS:
                self.bm25_status_label.configure(
                    text=f"• BM25 Algorithm: ✓ Active ({doc_count}/{CORPUS_MIN_DOCUMENTS}+ documents)",
                    text_color=(COLORS["success"], COLORS["success_light"]),
                )
            else:
                needed = CORPUS_MIN_DOCUMENTS - doc_count
                self.bm25_status_label.configure(
                    text=f"• BM25 Algorithm: ○ Inactive (need {needed} more documents)",
                    text_color=(COLORS["warning"], COLORS["warning"]),
                )

        except Exception as e:
            from src.logging_config import debug_log

            debug_log(f"[CorpusSettingsWidget] Error refreshing status: {e}")
            self.active_corpus_label.configure(text="• Active corpus: (error)")
            self.doc_count_label.configure(text="• Documents: (error)")
            self.bm25_status_label.configure(text="• BM25 Algorithm: (error)")

    def _open_corpus_dialog(self):
        """Open the full corpus management dialog."""
        from src.ui.corpus_dialog import CorpusDialog

        # Find the settings dialog (our grandparent) to use as parent
        parent = self.winfo_toplevel()

        dialog = CorpusDialog(parent)
        parent.wait_window(dialog)

        # Refresh status after dialog closes
        if dialog.corpus_changed:
            self._refresh_status()

            # Also notify main window to refresh its corpus dropdown
            try:
                main_window = parent.master
                if main_window and hasattr(main_window, "_refresh_corpus_dropdown"):
                    main_window._refresh_corpus_dropdown()
            except Exception:
                pass  # Main window refresh is best-effort

    def get_value(self):
        """No value to return - changes are made via dialog."""
        return None

    def set_value(self, value):
        """No value to set - loads from registry."""
        pass
