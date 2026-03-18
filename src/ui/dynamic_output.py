"""
Dynamic Output Display Widget for CasePrepd

Displays Vocabulary tables, search results, and key excerpts using tab navigation.
Provides copy/save functionality for export.
The vocabulary display uses an Excel-like Treeview with frozen headers
and right-click context menu for excluding terms from future extractions.

Tab navigation uses CTkTabview with three persistent tabs:
"Vocabulary" | "Search" | "Key Excerpts". Tab switching uses frame
stacking (tkraise) with no widget recreation; all content is preserved
across tab switches (scroll position, state).

Performance optimizations:
- Text truncation prevents row height overflow and improves rendering speed
- Batch insertion with reduced batch size for responsiveness
- Asynchronous batch insertion with after() to yield to event loop
- Pagination with "Load More" button for large datasets
- Window resize debouncing prevents batch insertion conflicts
- Tab navigation eliminates layout recalculation overhead
"""

import csv
import gc
import io
import logging
import os
import re
from tkinter import Menu, filedialog, messagebox

import customtkinter as ctk

from src.config import SORT_WARNING_COLUMNS, USER_VOCAB_EXCLUDE_PATH, VF

logger = logging.getLogger(__name__)
from src.ui.semantic_panel import SemanticPanel
from src.ui.styles import get_vocab_font_specs
from src.ui.theme import (
    BUTTON_STYLES,
    COLORS,
    FONTS,
    FRAME_STYLES,
    get_color,
)

# Import column configuration from centralized module
from src.ui.vocab_table.column_config import (
    ALL_EXPORT_COLUMNS,
    BATCH_INSERT_DELAY_MS,
    BATCH_INSERT_SIZE,
    COLUMN_CONFIG,
    COLUMN_ORDER,
    COLUMN_REGISTRY,
    DISPLAY_TO_DATA_COLUMN,
    GUI_DISPLAY_COLUMNS,
    ROWS_PER_PAGE,
    compute_column_widths,
)
from src.ui.vocab_table.vocab_treeview import VocabTreeview, strip_display_prefix  # noqa: E402
from src.user_preferences import get_user_preferences


class DynamicOutputWidget(ctk.CTkFrame):
    """Widget to dynamically display Vocabulary, Search, or Key Excerpts outputs."""

    def __init__(self, master, **kwargs):
        # Distinct background color for output pane (slightly different to distinguish)
        kwargs.setdefault("fg_color", COLORS["output_pane"])
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)  # Tab view gets all space

        # Create tabview with tabs: Document, Vocabulary, Search, Key Excerpts
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Create tabs — Document first (matches pipeline flow: input → results)
        self.tabview.add("Document")
        self.tabview.add("Vocabulary")
        self.tabview.add("Search")
        self.tabview.add("Key Excerpts")

        # Bind tab change to show/hide appropriate button bar
        self.tabview.configure(command=self._on_tab_changed)

        # Configure tab grids
        self.tabview.tab("Document").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Document").grid_rowconfigure(0, weight=1)
        self.tabview.tab("Vocabulary").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Vocabulary").grid_rowconfigure(0, weight=1)
        self.tabview.tab("Search").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Search").grid_rowconfigure(0, weight=1)
        self.tabview.tab("Key Excerpts").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Key Excerpts").grid_rowconfigure(0, weight=0)

        # Document Tab: Preview panel for extracted text
        from src.ui.document_preview_panel import DocumentPreviewPanel

        self._document_panel = DocumentPreviewPanel(self.tabview.tab("Document"))
        self._document_panel.grid(row=0, column=0, sticky="nsew")

        # Progress Badge - shows data source status for Names & Vocab tab
        self._progress_badge = ctk.CTkLabel(
            self.tabview.tab("Vocabulary"),
            text="",
            font=FONTS["small"],
            text_color=("gray50", "gray70"),
        )
        self._progress_badge.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))

        # Feedback balance hint - nudges users toward balanced keep/skip voting
        self._balance_hint = ctk.CTkLabel(
            self.tabview.tab("Vocabulary"),
            text="",
            font=FONTS["small"],
            text_color=("gray50", "gray70"),
            cursor="hand2",
        )
        self._balance_hint.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 5))
        self._balance_hint.bind("<Button-1>", lambda e: self._dismiss_balance_hint())
        self._balance_hint_dismissed = False

        # Names & Vocab Tab: Treeview frame (initially None, created when needed)
        self.csv_treeview = None  # Backward compat alias to _main_vocab_tv.widget
        self.treeview_frame = None  # Frame to hold treeview and scrollbars
        self._main_vocab_tv: VocabTreeview | None = None

        # Filtered terms (collapsed section below main vocab table)
        self._filtered_treeview = None  # Backward compat alias to _filtered_vocab_tv.widget
        self._filtered_vocab_tv: VocabTreeview | None = None
        self._filtered_frame = None  # Outer frame with header + treeview
        self._filtered_header = None  # Clickable expand/collapse header
        self._filtered_inner_frame = None  # Inner frame (hidden when collapsed)
        self._filtered_expanded = False
        self._filtered_vocab_data_raw: list[dict] = []  # Raw filtered data
        self._filtered_unsorted_data: list[dict] = []  # Unsorted copy for sort reset
        self._filtered_sort_column: str | None = None
        self._filtered_sort_ascending: bool = True

        # Dialog-open guards (prevent opening duplicate dialogs)
        self._dialog_open: dict[str, bool] = {}
        self._exporting_vocab = False  # Re-entrancy guard for vocab export

        # Filter widgets for vocabulary table
        self.filter_frame = None
        self.filter_entry = None
        self.filter_regex_var = None  # BooleanVar for regex checkbox
        self._detached_items = []  # Items hidden by filter (for restore)

        # Right-click context menu for vocabulary exclusion
        self.context_menu = None
        self._selected_term = None

        # Extraction state for feedback blocking
        self._extraction_in_progress = False

        # Workflow status tracking for tab status messages
        from src.ui.workflow_status import TabStatusConfig, WorkflowPhase

        self._workflow_phase = WorkflowPhase.IDLE
        self._tab_status_config = TabStatusConfig()

        # Search Tab: Status placeholder (shown when no results) + search panel
        # See: https://github.com/TomSchimansky/CustomTkinter/issues/1508
        self._semantic_status_label = ctk.CTkLabel(
            self.tabview.tab("Search"),
            text="",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            justify="center",
            wraplength=0,
        )
        self._semantic_status_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=50)

        self._semantic_panel = SemanticPanel(self.tabview.tab("Search"))
        self._semantic_panel.grid(row=0, column=0, sticky="nsew")
        self._semantic_panel.grid_remove()  # Hidden initially, shown when results arrive

        # Summary Tab: Status placeholder (shown when no summary) + textbox
        self._summary_status_label = ctk.CTkLabel(
            self.tabview.tab("Key Excerpts"),
            text="",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            justify="center",
            wraplength=0,
        )
        self._summary_status_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=50)

        self.summary_text_display = ctk.CTkTextbox(
            self.tabview.tab("Key Excerpts"),
            wrap="word",
            font=FONTS["body"],
            fg_color=COLORS["bg_darker"],
            text_color=COLORS["text_primary"],
        )
        self.summary_text_display.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.summary_text_display.grid_remove()  # Hidden initially, shown when results arrive

        # Configure text tags for card-style formatting (same as Search tab)
        self._apply_summary_tags()

        # Re-apply text tags when font size changes live
        self.bind("<<FontChanged>>", lambda e: self._apply_summary_tags())

        # Inline find bar for Summary tab (hidden by default)
        from src.ui.text_find_bar import TextFindBar

        self._summary_find_bar = TextFindBar(
            self.tabview.tab("Key Excerpts"), self.summary_text_display
        )
        self._summary_find_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=(2, 0))
        self._summary_find_bar.grid_remove()

        # Summary tab row 1 (textbox) expands
        self.tabview.tab("Key Excerpts").grid_rowconfigure(1, weight=1)

        # Initialize tab status messages
        self._update_tab_status_messages()

        # Default to Vocabulary tab (Document is just a placeholder until user clicks a file)
        self.tabview.set("Vocabulary")

        # Bind Ctrl+F for find-in-text
        self._bind_find_shortcut()

        # Button bar (below tabs)
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.copy_btn = ctk.CTkButton(
            self.button_frame, text="Copy to Clipboard", command=self.copy_to_clipboard
        )
        self.copy_btn.pack(side="left", padx=5)

        self.save_btn = ctk.CTkButton(
            self.button_frame, text="Save to File...", command=self.save_to_file
        )
        self.save_btn.pack(side="left", padx=5)

        # Column visibility state
        self._column_visibility = self._load_column_visibility()
        self.column_picker_btn = ctk.CTkButton(
            self.button_frame,
            text="Columns...",
            command=self._show_column_menu,
            width=90,
            **BUTTON_STYLES["secondary"],
        )
        self.column_picker_btn.pack(side="left", padx=5)

        # Export dropdown
        # Note: CTkOptionMenu doesn't support hover_color, so only use fg_color
        self.export_dropdown = ctk.CTkOptionMenu(
            self.button_frame,
            values=["Export...", "TXT", "PDF", "HTML"],
            command=self._on_export_format_selected,
            width=120,
            fg_color=BUTTON_STYLES["primary"]["fg_color"],
        )
        self.export_dropdown.pack(side="left", padx=5)

        # Internal storage for outputs
        self._outputs = {
            "Names & Vocabulary": [],  # Primary output - people + terms
            "Search": [],  # Semantic search results
            "Key Excerpts": "",  # Combined summary (replaces "Meta-Summary")
            # Backward compatibility keys
            "Meta-Summary": "",
            "Rare Word List (CSV)": [],
            "Semantic Results": [],
        }
        self._document_summaries = {}  # {filename: summary_text}

        # Data source tracking for progress badge
        self._extraction_source = "none"  # "none", "ner", "both"

        # Pagination state for vocabulary display
        self._vocab_display_offset = 0  # Current offset into vocabulary data
        self._vocab_total_items = 0  # Total items in vocabulary data
        self._load_more_btn = None  # "Load More" button reference
        self._is_loading = False  # Prevents duplicate load operations
        self._insertion_cancelled = False  # Cancels pending async insertions

        # Sort state for vocabulary table
        self._sort_column = None  # Currently sorted column name
        self._sort_ascending = True  # Sort direction
        self._unsorted_vocab_data = []  # Original order for reset

        # Resize event debouncing to prevent glitchiness
        self._resize_after_id = None  # Track pending resize callbacks
        self._batch_insertion_paused = False  # Pause batch insertion during resize

        # Feedback manager for ML learning
        from src.services import VocabularyService

        self._feedback_manager = VocabularyService().get_feedback_manager()

        # Track whether corpus warning has been shown this session
        self._corpus_warning_shown = False

        # Bind to Configure event for resize debouncing
        self.bind("<Configure>", self._on_window_resize)

    def _bind_find_shortcut(self):
        """Bind Ctrl+F to show the find bar for the active text tab."""

        def _on_ctrl_f(event):
            active_tab = self.tabview.get()
            if active_tab == "Document":
                self._document_panel.show_find_bar()
            elif active_tab == "Search":
                self._semantic_panel.show_find_bar()
            elif active_tab == "Key Excerpts":
                self._summary_find_bar.show()
            # Names & Vocab tab has its own filter bar — no-op
            return "break"

        # Bind on the top-level window so it works regardless of focus
        self.winfo_toplevel().bind("<Control-f>", _on_ctrl_f)

    def _on_window_resize(self, event):
        """
        Debounce resize events to prevent batch insertion conflicts.

        When the window is resized/maximized, this pauses batch insertion
        and schedules a callback to resume after resizing stabilizes (100ms).

        Only responds to Configure events on this widget itself, not child
        widgets. Without this filter, every child label/frame/button that
        changes geometry during a resize fires this handler, flooding the
        event loop and creating a feedback loop with update_idletasks().

        Args:
            event: Tkinter Configure event
        """
        # Ignore Configure events from child widgets -- only respond to self
        if event.widget is not self:
            return

        # Cancel any pending resize callback
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)

        # Pause batch insertion during resize
        self._batch_insertion_paused = True

        # Schedule callback for 100ms after resize stops
        self._resize_after_id = self.after(100, self._on_resize_complete)

    def _on_resize_complete(self):
        """Called 100ms after resize events stop - resumes batch insertion."""
        self._batch_insertion_paused = False
        self._resize_after_id = None
        logger.debug("Resize complete - batch insertion resumed")

    # -------------------------------------------------------------------------
    # Workflow Status Methods for Tab Status Messages
    # -------------------------------------------------------------------------

    def _update_tab_status_messages(self):
        """Update status labels in Search and Key Excerpts tabs based on workflow phase and config."""
        from src.ui.workflow_status import get_semantic_tab_status, get_summary_tab_status

        # Update Search tab status
        semantic_status = get_semantic_tab_status(self._workflow_phase, self._tab_status_config)
        self._semantic_status_label.configure(text=semantic_status)

        # Update Summary tab status
        summary_status = get_summary_tab_status(self._workflow_phase, self._tab_status_config)
        self._summary_status_label.configure(text=summary_status)

    def set_workflow_phase(self, phase):
        """
        Update the workflow phase and refresh tab status messages.

        Called by MainWindow when workflow transitions between phases.

        Args:
            phase: WorkflowPhase enum value
        """

        self._workflow_phase = phase
        self._update_tab_status_messages()
        logger.debug("Workflow phase set to: %s", phase.name)

    def set_tab_status_config(self, vocab_enabled=None, semantic_enabled=None, **kwargs):
        """
        Update which features are enabled and refresh tab status messages.

        Called by MainWindow when checkboxes change.

        Args:
            vocab_enabled: Whether vocabulary extraction is enabled (or None to keep current)
            semantic_enabled: Whether semantic search is enabled (or None to keep current)
        """
        if vocab_enabled is not None:
            self._tab_status_config.vocab_enabled = vocab_enabled
        if semantic_enabled is not None:
            self._tab_status_config.semantic_enabled = semantic_enabled

        self._update_tab_status_messages()
        logger.debug(
            "Tab status config updated: vocab=%s, qa=%s",
            self._tab_status_config.vocab_enabled,
            self._tab_status_config.semantic_enabled,
        )

    def show_semantic_content(self):
        """Show the search panel and hide the status label (called when results arrive)."""
        self._semantic_status_label.grid_remove()
        self._semantic_panel.grid()

    def show_semantic_status(self):
        """Show the search status label and hide the panel (called when clearing or before results)."""
        self._semantic_panel.grid_remove()
        self._semantic_status_label.grid()
        self._update_tab_status_messages()

    def _apply_summary_tags(self):
        """Apply or re-apply text tags on the Key Excerpts textbox."""
        from src.ui.theme import SEMANTIC_TEXT_TAGS, resolve_tags

        for tag_name, tag_conf in resolve_tags(SEMANTIC_TEXT_TAGS).items():
            self.summary_text_display.tag_config(tag_name, cnf=tag_conf)

    def show_summary_content(self):
        """Show the summary textbox and hide the status label (called when summary arrives)."""
        self._summary_status_label.grid_remove()
        self.summary_text_display.grid()

    def show_summary_status(self):
        """Show the summary status label and hide the textbox (called when clearing)."""
        self.summary_text_display.grid_remove()
        self._summary_status_label.grid()
        self._update_tab_status_messages()

    def update_key_sentences(self, sentences: list[dict]):
        """
        Display key excerpts in the Key Excerpts tab with card-style formatting.

        Each sentence is displayed as a styled card with source attribution,
        visually consistent with the Search tab layout.

        Args:
            sentences: List of dicts with 'text', 'source_file', 'position', 'score'.
        """
        if not sentences:
            return

        self._key_sentences = sentences

        # Clear and insert with styled tags
        self.summary_text_display.configure(state="normal")
        self.summary_text_display.delete("0.0", "end")

        total = len(sentences)
        for i, sent in enumerate(sentences, 1):
            text = sent.get("text", "")
            source = sent.get("source_file", "")

            # Card header with position context
            self.summary_text_display.insert("end", f"[ Excerpt {i} of {total} ]\n", "question")

            # Excerpt body with breathing room
            self.summary_text_display.insert("end", f"\n{text}\n\n", "citation")

            # Source attribution
            if source:
                self.summary_text_display.insert("end", f"Source: {source}\n", "source")

            # Generous spacing between cards
            if i < total:
                self.summary_text_display.insert("end", "\n")
                self.summary_text_display.insert("end", "─" * 60 + "\n\n\n", "separator")

        self.summary_text_display.configure(state="disabled")
        self.show_summary_content()
        logger.debug("Displayed %d key excerpts in Key Excerpts tab", len(sentences))

    def show_document_preview(self, result):
        """
        Display a document's extracted text in the Document tab.

        Auto-switches to the Document tab.

        Args:
            result: Extraction result dict with filename, preprocessed_text, etc.
        """
        self._document_panel.display_document(result)
        self.tabview.set("Document")

    def clear_document_preview(self):
        """Clear the Document tab back to placeholder state."""
        self._document_panel.clear()

    @property
    def document_preview_filename(self):
        """The filename currently shown in the Document tab, or None."""
        return self._document_panel.current_filename

    def _on_tab_changed(self):
        """
        Handle tab change to show/hide appropriate button bars.

        Hides the shared button bar when Search tab is active (since SemanticPanel
        has its own buttons), shows it for other tabs. Also shows/hides the
        main window's follow-up input frame so it only appears when the Search
        tab is active. Saves column widths when leaving Names & Vocab tab.
        """
        current_tab = self.tabview.get()

        # Brief dim/reveal transition effect
        self._animate_tab_transition()

        # Show/hide main window's follow-up frame based on tab
        main_window = self.winfo_toplevel()
        if hasattr(main_window, "followup_frame"):
            if current_tab == "Search":
                main_window.followup_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
            else:
                main_window.followup_frame.grid_remove()

        if current_tab in ("Search", "Document"):
            # Hide shared button bar - SemanticPanel has its own, Document is read-only
            self.button_frame.grid_remove()
        else:
            # Show shared button bar for Names & Vocab and Summary tabs
            self.button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

            # Show/hide vocab-specific widgets based on tab
            if current_tab == "Vocabulary":
                # Update progress badge when switching to this tab
                self._update_progress_badge(self._extraction_source)
                # Show vocab widgets - use after= to maintain correct order
                if not self.column_picker_btn.winfo_ismapped():
                    self.column_picker_btn.pack(side="left", padx=5, after=self.save_btn)
                if not self.export_dropdown.winfo_ismapped():
                    self.export_dropdown.pack(side="left", padx=5, after=self.column_picker_btn)
            else:
                # Hide vocab-specific widgets on Summary tab
                self.column_picker_btn.pack_forget()
                self.export_dropdown.pack_forget()

        # Save column widths when leaving Names & Vocab tab
        # (Non-critical -- must not block UI updates above)
        self._save_column_widths()

    def _animate_tab_transition(self):
        """Brief dim/reveal effect when switching tabs."""
        current_tab = self.tabview.get()
        tab_frame = self.tabview.tab(current_tab)

        # Step 1: Dim by placing a temporary overlay
        overlay = ctk.CTkFrame(tab_frame, fg_color=COLORS["bg_dark"], corner_radius=0)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.lift()

        # Step 2: Remove overlay after brief delay to create "reveal" effect
        def remove_overlay():
            try:
                overlay.place_forget()
                overlay.destroy()
            except Exception:
                logger.debug("Overlay cleanup skipped (widget destroyed)")

        self.after(100, remove_overlay)

    def _update_progress_badge(self, source: str):
        """
        Update the progress badge to show data source status.

        "Local algorithms" covers NER, RAKE, and BM25 results.

        Args:
            source: "none", "partial" (BM25+RAKE only), or "ner" (all local algorithms)
        """
        if source == "none":
            self._progress_badge.configure(text="")
        elif source == "partial":
            # BM25 + RAKE results shown before NER completes
            self._progress_badge.configure(
                text="Partial results (BM25+RAKE)", text_color=("gray50", "gray70")
            )
        elif source == "ner":
            # Results come from NER, RAKE, BM25
            self._progress_badge.configure(
                text="Results (local algorithms)", text_color=COLORS["progress_complete"]
            )

    def set_extraction_source(self, source: str):
        """
        Set the data source for the current extraction.

        Called by workflow orchestrator to update progress badge.

        Args:
            source: "none", "partial" (BM25+RAKE only), or "ner" (all local algorithms)
        """
        self._extraction_source = source
        # Update badge if Names & Vocabulary tab is currently active
        current_tab = self.tabview.get()
        if current_tab == "Vocabulary":
            self._update_progress_badge(source)

    def set_extraction_in_progress(self, in_progress: bool):
        """
        Set whether vocabulary extraction is in progress.

        When in progress, feedback buttons are visually dimmed and clicks
        show a message asking user to wait for extraction to complete.

        Args:
            in_progress: True to dim buttons, False to re-enable
        """
        self._extraction_in_progress = in_progress

        # Update progress badge to indicate extraction state
        if in_progress:
            self._progress_badge.configure(
                text="Extracting... (feedback disabled)",
                text_color=("gray50", "gray70"),
            )
        elif self._extraction_source:
            self._update_progress_badge(self._extraction_source)

    # -------------------------------------------------------------------------
    # Column Visibility Management
    # -------------------------------------------------------------------------

    def _load_column_visibility(self) -> dict[str, bool]:
        """Load column visibility from user preferences, with defaults."""
        prefs = get_user_preferences()
        saved = prefs.get("vocab_column_visibility", {})

        visibility = {}
        for col_name, col_config in COLUMN_REGISTRY.items():
            # Use saved value if present, else default from registry
            visibility[col_name] = saved.get(col_name, col_config["default"])
        return visibility

    def _save_column_visibility(self):
        """Save current column visibility to user preferences."""
        prefs = get_user_preferences()
        prefs.set("vocab_column_visibility", self._column_visibility)

    def _get_visible_columns(self) -> list[str]:
        """Get list of currently visible column names in display order."""
        return [col for col in COLUMN_ORDER if self._column_visibility.get(col, False)]

    def _get_filtered_vocab_data(self) -> list[dict]:
        """
        Return score-filtered vocab data, with fallback to manual filtering.

        Uses _unsorted_vocab_data (already filtered by score floor during display).
        Falls back to filtering raw output data if display hasn't populated yet.

        Returns:
            List of vocabulary dicts passing the score floor filter
        """
        if self._unsorted_vocab_data:
            return [
                d
                for d in self._unsorted_vocab_data
                if self._feedback_manager.get_rating(d.get(VF.TERM, "")) != -1
            ]
        # Fallback if display hasn't populated _unsorted_vocab_data yet
        raw = self._outputs.get("Names & Vocabulary") or self._outputs.get(
            "Rare Word List (CSV)", []
        )
        if not raw:
            return []
        score_floor = get_user_preferences().get("vocab_score_floor", 55)
        return [
            d
            for d in raw
            if isinstance(d, dict)
            and d.get(VF.QUALITY_SCORE, 0) >= score_floor
            and self._feedback_manager.get_rating(d.get(VF.TERM, "")) != -1
        ]

    # -------------------------------------------------------------------------
    # Column Width Persistence
    # -------------------------------------------------------------------------

    def _load_column_widths(self) -> dict[str, int]:
        """Load saved column widths from user preferences."""
        prefs = get_user_preferences()
        return prefs.get("vocab_column_widths", {})

    def _save_column_widths(self):
        """Save current column widths to user preferences."""
        if self.csv_treeview is None:
            return

        widths = {}
        columns = self._get_visible_columns()
        for col in columns:
            try:
                # Get actual current width from treeview
                width = self.csv_treeview.column(col, "width")
                if isinstance(width, int) and 30 <= width <= 800:
                    widths[col] = width
            except Exception as e:
                logger.debug("Could not read column '%s' width: %s", col, e)

        if widths:
            try:
                prefs = get_user_preferences()
                prefs.set("vocab_column_widths", widths)
            except Exception as e:
                logger.debug("Could not save column widths: %s", e)

    def _get_column_width(self, col_name: str) -> int:
        """
        Get width for a column, preferring saved width over default.

        Args:
            col_name: Column name

        Returns:
            Width in pixels
        """
        saved_widths = self._load_column_widths()
        if col_name in saved_widths:
            return saved_widths[col_name]
        # Fall back to COLUMN_CONFIG default
        return COLUMN_CONFIG.get(col_name, {}).get("width", 100)

    def _autosize_columns_to_content(self, data: list):
        """
        Auto-size columns to fit actual content using DPI-aware font metrics.

        Scans first 100 rows of data and adjusts column widths to fit.
        Only runs when user has no saved column widths for this session.

        Args:
            data: Full vocabulary data list (first 100 rows sampled).
        """
        if self.csv_treeview is None:
            return

        # Skip if user has saved custom widths
        saved_widths = self._load_column_widths()
        if saved_widths:
            return

        content_font, heading_font = get_vocab_font_specs()
        available_w = self.treeview_frame.winfo_width()
        if available_w < 100:
            available_w = 900

        columns = list(self._current_columns)
        computed = compute_column_widths(
            columns, content_font, heading_font, available_w, data_sample=data
        )

        for col in columns:
            if col in computed:
                self.csv_treeview.column(col, width=computed[col])

        logger.debug("Auto-sized %d columns to content", len(computed))

    def _reset_column_widths(self):
        """Clear saved column widths and re-auto-size to content."""
        prefs = get_user_preferences()
        prefs.set("vocab_column_widths", {})
        logger.debug("Cleared saved column widths")

        # Re-auto-size using current data
        vocab_data = self._outputs.get("Names & Vocabulary", [])
        if not vocab_data:
            vocab_data = self._outputs.get("Rare Word List (CSV)", [])
        if vocab_data and self.csv_treeview is not None:
            content_font, heading_font = get_vocab_font_specs()
            available_w = self.treeview_frame.winfo_width()
            if available_w < 100:
                available_w = 900
            columns = list(self._current_columns)
            computed = compute_column_widths(
                columns, content_font, heading_font, available_w, data_sample=vocab_data
            )
            for col in columns:
                if col in computed:
                    self.csv_treeview.column(col, width=computed[col])
            logger.debug("Re-auto-sized columns after reset")

    def _show_column_menu(self, event=None):
        """
        Show column visibility context menu.

        Can be triggered by the Columns button or right-click on header.
        """
        menu = Menu(
            self,
            tearoff=0,
            bg=get_color("menu_bg"),
            fg=get_color("menu_fg"),
            activebackground=get_color("menu_active_bg"),
            activeforeground=get_color("menu_active_fg"),
            font=("Segoe UI", 10),
        )

        for col_name in COLUMN_ORDER:
            col_config = COLUMN_REGISTRY[col_name]

            # Term cannot be hidden
            if not col_config["can_hide"]:
                menu.add_command(label=f"  {col_name} (required)", state="disabled")
            else:
                # Checkmark for visible columns
                is_visible = self._column_visibility.get(col_name, col_config["default"])
                prefix = "\u2713 " if is_visible else "   "  # ✓ or spaces
                menu.add_command(
                    label=f"{prefix}{col_name}", command=lambda c=col_name: self._toggle_column(c)
                )

        menu.add_separator()
        menu.add_command(label="Reset Column Visibility", command=self._reset_column_visibility)
        menu.add_command(label="Reset Column Widths", command=self._reset_column_widths)

        # Position menu at button or event location
        if event:
            menu.tk_popup(event.x_root, event.y_root)
        else:
            # Position below the button
            btn_x = self.column_picker_btn.winfo_rootx()
            btn_y = self.column_picker_btn.winfo_rooty() + self.column_picker_btn.winfo_height()
            menu.tk_popup(btn_x, btn_y)

    def _toggle_column(self, column_name: str):
        """Toggle visibility of a column."""
        current = self._column_visibility.get(column_name, COLUMN_REGISTRY[column_name]["default"])
        self._column_visibility[column_name] = not current
        self._save_column_visibility()
        self._refresh_treeview_columns()

    def _reset_column_visibility(self):
        """Reset column visibility to defaults."""
        self._column_visibility = {col: cfg["default"] for col, cfg in COLUMN_REGISTRY.items()}
        self._save_column_visibility()
        self._refresh_treeview_columns()

    def _auto_hide_single_doc_columns(self, data: list[dict]):
        """
        Auto-hide '# Docs' column when session has only 1 document.

        When all terms come from a single document, '# Docs' always shows 1
        and provides no useful information. Auto-restores when multi-doc
        data is processed next.

        Args:
            data: Vocabulary data list (each item has 'total_docs_in_session')
        """
        if not data:
            return

        total_docs = data[0].get("total_docs_in_session", 1)
        is_single_doc = total_docs <= 1
        currently_visible = self._column_visibility.get(VF.NUM_DOCS, True)

        if is_single_doc and currently_visible:
            self._column_visibility[VF.NUM_DOCS] = False
            self._save_column_visibility()
            logger.debug("Auto-hid '# Docs' column (single document session)")
        elif not is_single_doc and not currently_visible:
            self._column_visibility[VF.NUM_DOCS] = True
            self._save_column_visibility()
            logger.debug("Auto-restored '# Docs' column (multi-document session)")

    def _refresh_treeview_columns(self):
        """Refresh treeview when column visibility changes."""
        vocab_data = self._outputs.get("Names & Vocabulary", [])
        if not vocab_data:
            vocab_data = self._outputs.get("Rare Word List (CSV)", [])
        if vocab_data and self.csv_treeview is not None:
            self.csv_treeview.destroy()
            self.csv_treeview = None
            self._display_csv(vocab_data)

    def _sort_by_column(self, column: str):
        """
        Sort vocabulary table by column.

        Clicking a header sorts ascending; clicking again sorts descending;
        clicking a third time resets to original order. Shows a warning dialog
        when sorting by non-Score columns since those sorts will show
        lower-quality results first.

        Args:
            column: The column name to sort by
        """
        if not self._unsorted_vocab_data or self.csv_treeview is None:
            return

        # Show warning for non-Score columns (first click only)
        # Don't warn when changing direction on same column or resetting
        is_new_column = self._sort_column != column
        if is_new_column and column in SORT_WARNING_COLUMNS:
            result = messagebox.askyesno(
                "Sort Warning",
                f"Sorting by '{column}' will show lower-quality results first.\n\nContinue?",
                icon="warning",
            )
            if not result:
                return  # User cancelled

        # Determine new sort state
        if self._sort_column == column:
            if self._sort_ascending:
                # Second click: descending
                self._sort_ascending = False
            else:
                # Third click: reset to unsorted
                self._sort_column = None
                self._sort_ascending = True
        else:
            # New column: ascending
            self._sort_column = column
            self._sort_ascending = True

        # Prepare sorted data
        if self._sort_column is None:
            # Reset to original order
            sorted_data = list(self._unsorted_vocab_data)
        else:
            # Sort by column
            sorted_data = self._sort_vocab_data(
                self._unsorted_vocab_data, self._sort_column, self._sort_ascending
            )

        # Update header text to show sort indicator
        self._update_sort_headers()

        # Redisplay with sorted data (without resetting sort state)
        self._redisplay_sorted_data(sorted_data)

    def _sort_vocab_data(self, data: list[dict], column: str, ascending: bool) -> list[dict]:
        """
        Sort vocabulary data by column with type-aware comparison.

        Numeric columns (Score, # Docs, Count, etc.) sort numerically.
        Other columns sort alphabetically, case-insensitive.

        After sorting, groups potential duplicates together so they appear adjacent.

        Args:
            data: List of vocabulary dicts
            column: Column name to sort by
            ascending: True for ascending, False for descending

        Returns:
            Sorted list (new list, doesn't modify original)
        """
        # Map display column names to data keys
        column_key_map = {
            "Score": VF.QUALITY_SCORE,
            VF.NUM_DOCS: VF.NUM_DOCS,
            VF.ALGO_COUNT: VF.ALGO_COUNT,
            VF.GOOGLE_RARITY_RANK: VF.GOOGLE_RARITY_RANK,
        }
        data_key = column_key_map.get(column, column)

        # Numeric columns for type-aware sorting
        numeric_columns = {
            "Score",
            VF.QUALITY_SCORE,
            VF.OCCURRENCES,
            VF.NUM_DOCS,
            VF.ALGO_COUNT,
            VF.GOOGLE_RARITY_RANK,
        }
        is_numeric = column in numeric_columns or data_key in numeric_columns

        def sort_key(item):
            value = item.get(data_key, "")
            if is_numeric:
                # Parse numeric value (handle percentage strings like "85%")
                if isinstance(value, (int, float)):
                    return value
                try:
                    # Strip percentage sign and other non-numeric chars
                    clean = str(value).replace("%", "").replace(",", "").strip()
                    return float(clean) if clean else 0
                except (ValueError, TypeError):
                    return 0
                except Exception:
                    logger.error("Unexpected error in sort key", exc_info=True)
                    return 0
            else:
                # String comparison, case-insensitive
                return str(value).lower()

        sorted_data = sorted(data, key=sort_key, reverse=not ascending)

        # Group potential duplicates together after sorting
        return self._group_potential_duplicates(sorted_data)

    def _group_potential_duplicates(self, data: list[dict]) -> list[dict]:
        """
        Reorder list so potential duplicates appear adjacent to their matches.

        Moves items with _potential_duplicate_of to be immediately after their
        matching longer name, making it easy for users to review related names.

        Args:
            data: Sorted vocabulary list

        Returns:
            Reordered list with potential duplicates grouped
        """
        # Build index of term positions
        term_to_index = {item.get(VF.TERM, ""): i for i, item in enumerate(data)}

        # Find items that need to be moved
        items_to_move = []
        for i, item in enumerate(data):
            match = item.get("_potential_duplicate_of")
            if match and match in term_to_index:
                items_to_move.append((i, item, match))

        if not items_to_move:
            return data  # Nothing to group

        # Create result list, skipping items that will be moved
        indices_to_skip = {i for i, _, _ in items_to_move}
        result = []

        for i, item in enumerate(data):
            if i in indices_to_skip:
                continue
            result.append(item)
            # Insert any items that should follow this one
            term = item.get(VF.TERM, "")
            for _, moved_item, match in items_to_move:
                if match == term:
                    result.append(moved_item)

        return result

    def _update_sort_headers(self):
        """Update column header text to show sort indicator (▲/▼)."""
        if self.csv_treeview is None:
            return

        columns = self._get_visible_columns()
        for col in columns:
            if col == self._sort_column:
                indicator = " ▲" if self._sort_ascending else " ▼"
                self.csv_treeview.heading(col, text=f"{col}{indicator}")
            else:
                # Remove indicator from other columns
                self.csv_treeview.heading(col, text=col)

    def _redisplay_sorted_data(self, sorted_data: list):
        """
        Redisplay treeview with sorted data without full rebuild.

        Clears and repopulates treeview rows while preserving structure.
        """
        if self.csv_treeview is None:
            return

        # Clear existing rows
        self.csv_treeview.delete(*self.csv_treeview.get_children())

        # Reset pagination and redisplay
        self._vocab_display_offset = 0
        self._vocab_total_items = len(sorted_data)

        # Calculate how many items to load
        initial_load = min(ROWS_PER_PAGE, self._vocab_total_items)

        # Temporarily store sorted data for async insertion
        self._sorted_display_data = sorted_data

        # Start async batch insertion
        self._async_insert_rows(sorted_data, 0, initial_load)

    # -------------------------------------------------------------------------
    # Text Filter for Vocabulary Table
    # -------------------------------------------------------------------------

    def _on_filter_changed(self, event=None):
        """Handle filter entry text change - filter treeview rows."""
        if self.filter_entry is None:
            return
        filter_text = self.filter_entry.get().strip()
        use_regex = self.filter_regex_var.get() if self.filter_regex_var else False
        self._apply_filter(filter_text, use_regex=use_regex)

    def _clear_filter(self):
        """Clear filter and show all rows."""
        if self.filter_entry is not None:
            self.filter_entry.delete(0, "end")
        self._apply_filter("", use_regex=False)

    def _apply_filter(self, filter_text: str, use_regex: bool = False):
        """
        Apply text filter to treeview rows.

        Uses detach/reattach for performance (preserves item state).
        Filters by Term column (first column), case-insensitive.

        Args:
            filter_text: Text to filter by (empty string shows all)
            use_regex: If True, treat filter_text as regex pattern
        """
        if self.csv_treeview is None:
            return

        # First, restore all previously detached items
        import contextlib

        for item_id in self._detached_items:
            with contextlib.suppress(Exception):
                self.csv_treeview.reattach(item_id, "", "end")
        self._detached_items = []

        if not filter_text:
            return  # No filter, all items visible

        # Compile regex if using regex mode
        regex_pattern = None
        if use_regex:
            try:
                regex_pattern = re.compile(filter_text, re.IGNORECASE)
            except re.error:
                # Invalid regex - don't filter, let user fix it
                return
            except Exception:
                logger.error("Unexpected error compiling filter regex", exc_info=True)
                return

        # Detach non-matching items
        filter_lower = filter_text.lower()
        for item_id in self.csv_treeview.get_children():
            values = self.csv_treeview.item(item_id, "values")
            if values and len(values) >= 1:
                term = str(values[0])  # First column is Term
                if use_regex and regex_pattern:
                    matches = regex_pattern.search(term) is not None
                else:
                    matches = filter_lower in term.lower()

                if not matches:
                    self.csv_treeview.detach(item_id)
                    self._detached_items.append(item_id)

    def cleanup(self):
        """
        Clean up resources when widget is no longer needed.
        Call this to free memory after heavy processing.
        """
        # Clear internal data storage (must match __init__ structure)
        self._outputs = {
            "Names & Vocabulary": [],  # Primary output
            "Search": [],  # Semantic search results
            "Key Excerpts": "",  # Combined summary
            # Backward compatibility keys
            "Meta-Summary": "",
            "Rare Word List (CSV)": [],
            "Semantic Results": [],
        }
        self._document_summaries = {}
        self._extraction_source = "none"  # Reset progress badge state

        # Save column widths before cleanup, then clear sort state
        self._save_column_widths()
        self._unsorted_vocab_data = []
        self._sort_column = None
        self._sort_ascending = True

        # Clear treeview data if it exists
        if self._main_vocab_tv is not None:
            self._main_vocab_tv.clear()

        # Clear filtered section
        if self._filtered_frame is not None:
            self._filtered_frame.destroy()
            self._filtered_frame = None
            self._filtered_treeview = None
            self._filtered_vocab_tv = None
            self._filtered_header = None
            self._filtered_inner_frame = None
        self._filtered_vocab_data_raw = []

        # Force garbage collection
        gc.collect()
        logger.debug("Cleanup completed, memory freed.")

    def update_outputs(
        self,
        meta_summary: str = "",
        vocab_csv_data: list | None = None,
        document_summaries: dict | None = None,
        semantic_results: list | None = None,
        # Combined output parameters
        names_vocab_data: list | None = None,
        summary_text: str = "",
        extraction_source: str | None = None,
        filtered_vocab_data: list | None = None,
    ):
        """
        Updates the internal storage with new outputs and refreshes the dropdown.

        Args:
            meta_summary: The generated meta-summary text (legacy).
            vocab_csv_data: A list of dicts representing vocabulary data (legacy).
            document_summaries: A dictionary of {filename: summary_text}.
            semantic_results: A list of SemanticResult objects from semantic search.
            names_vocab_data: Combined people + vocabulary data.
            summary_text: Combined summary text.
            extraction_source: "ner" or "both" for progress badge.
            filtered_vocab_data: Terms excluded by frequency filters.
        """
        # Primary outputs
        if names_vocab_data is not None:
            self._outputs["Names & Vocabulary"] = names_vocab_data
        if summary_text:
            self._outputs["Key Excerpts"] = summary_text
        if extraction_source:
            self._extraction_source = extraction_source

        # Legacy support
        if meta_summary:
            self._outputs["Meta-Summary"] = meta_summary
            if not self._outputs.get("Key Excerpts"):
                self._outputs["Key Excerpts"] = meta_summary
        if vocab_csv_data is not None:
            self._outputs["Rare Word List (CSV)"] = vocab_csv_data
            if self._outputs.get("Names & Vocabulary") is None:
                self._outputs["Names & Vocabulary"] = vocab_csv_data
        if document_summaries:
            self._document_summaries.update(document_summaries)
        if semantic_results is not None:
            self._outputs["Semantic Results"] = semantic_results
            self._outputs["Search"] = semantic_results
        if filtered_vocab_data is not None:
            self._filtered_vocab_data_raw = filtered_vocab_data

        self._refresh_tabs()

    def _refresh_tabs(self):
        """
        Refresh tabs based on available outputs.

        Enables/disables tabs based on data availability and populates content.
        """
        # Names & Vocabulary tab
        vocab_data = self._outputs.get("Names & Vocabulary") or self._outputs.get(
            "Rare Word List (CSV)"
        )
        if vocab_data:
            self._display_csv(vocab_data)
            self._display_filtered_section(self._filtered_vocab_data_raw)
            self._update_progress_badge(self._extraction_source)

        # Search tab
        search_data = self._outputs.get("Search") or self._outputs.get("Semantic Results")
        if search_data:
            self._display_semantic_results(search_data)

        # Key Excerpts tab
        summary = self._outputs.get("Key Excerpts") or self._outputs.get("Meta-Summary")
        if summary:
            self.summary_text_display.delete("0.0", "end")
            self.summary_text_display.insert("0.0", summary)

        # Individual document summaries (if any) - append to key excerpts tab
        # Batch into single .insert() to reduce Tk text widget recalculations
        if self._document_summaries:
            separator = "=" * 50
            parts = [f"\n\n{separator}\n", "INDIVIDUAL DOCUMENT SUMMARIES\n", f"{separator}\n\n"]
            for doc_name, doc_summary in sorted(self._document_summaries.items()):
                parts.append(f"{doc_name}:\n{doc_summary}\n\n")
            self.summary_text_display.insert("end", "".join(parts))

        # Show summary content if we have any (hide status label)
        if summary or self._document_summaries:
            self.show_summary_content()

        # Only auto-switch to Vocab if user isn't already viewing
        # another tab that has content (prevents Search tab from jumping away)
        if vocab_data:
            current_tab = self.tabview.get()
            has_search = bool(search_data)
            has_summary = bool(summary) or bool(self._document_summaries)
            stay_on_current = (current_tab == "Search" and has_search) or (
                current_tab == "Key Excerpts" and has_summary
            )
            if not stay_on_current:
                self.tabview.set("Vocabulary")

    def _display_csv(self, data: list):
        """
        Displays vocabulary data in an Excel-like Treeview with frozen headers.

        Uses async batch insertion with pagination for GUI responsiveness.
        Initial load shows ROWS_PER_PAGE items, "Load More" button adds more.

        Filters out items with negative feedback (Skip) from display so
        previously skipped items don't reappear in new sessions.

        Args:
            data: List of dicts with keys: Term, Quality Score, Is Person, Found By
        """
        if not data:
            logger.debug("No vocabulary data to display")
            return

        # Filter out items with negative feedback (previously skipped)
        # This prevents old skipped items from appearing even if not in exclusion file
        original_count = len(data)
        data = [
            item
            for item in data
            if self._feedback_manager.get_rating(
                item.get(VF.TERM, "") if isinstance(item, dict) else ""
            )
            != -1
        ]
        filtered_count = original_count - len(data)
        if filtered_count > 0:
            logger.debug("Filtered %s previously skipped items", filtered_count)

        if not data:
            logger.debug("All items were filtered (previously skipped)")
            return

        # Filter by quality score floor
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        score_floor = prefs.get("vocab_score_floor", 55)
        pre_score_count = len(data)
        data = [
            item
            for item in data
            if isinstance(item, dict) and item.get(VF.QUALITY_SCORE, 0) >= score_floor
        ]
        score_filtered = pre_score_count - len(data)
        if score_filtered > 0:
            logger.debug("Filtered %s items below score floor %s", score_filtered, score_floor)

        if not data:
            logger.debug("All items filtered by score floor")
            return

        # Auto-hide "# Docs" column when only 1 document in session
        self._auto_hide_single_doc_columns(data)

        # Store unsorted data for sort operations and reset sort state
        # Group potential duplicates together for easier review
        grouped_data = self._group_potential_duplicates(list(data))
        self._unsorted_vocab_data = grouped_data
        self._sort_column = None
        self._sort_ascending = True

        # Reset pagination state
        self._vocab_display_offset = 0
        self._vocab_total_items = len(grouped_data)
        data = grouped_data  # Use grouped data for display
        self._is_loading = False

        # Create frame to hold treeview and scrollbars
        if self.treeview_frame is None:
            self.treeview_frame = ctk.CTkFrame(
                self.tabview.tab("Vocabulary"), **FRAME_STYLES["card"]
            )

        self.treeview_frame.grid(row=0, column=0, sticky="nsew")
        self.treeview_frame.grid_columnconfigure(0, weight=1)
        self.treeview_frame.grid_rowconfigure(1, weight=1)  # Treeview row expands

        # Create filter bar at top of treeview frame
        if self.filter_frame is None:
            self.filter_frame = ctk.CTkFrame(self.treeview_frame, fg_color="transparent")
            self.filter_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 2))

            self.filter_entry = ctk.CTkEntry(
                self.filter_frame, placeholder_text="Filter terms...", width=250
            )
            self.filter_entry.pack(side="left", padx=(0, 5))
            self.filter_entry.bind("<KeyRelease>", self._on_filter_changed)

            filter_clear_btn = ctk.CTkButton(
                self.filter_frame,
                text="Clear",
                width=60,
                command=self._clear_filter,
                **BUTTON_STYLES["secondary"],
            )
            filter_clear_btn.pack(side="left")

            # Regex checkbox (disabled by default)
            self.filter_regex_var = ctk.BooleanVar(value=False)
            filter_regex_cb = ctk.CTkCheckBox(
                self.filter_frame,
                text="Regex",
                variable=self.filter_regex_var,
                width=60,
                command=self._on_filter_changed,  # Re-apply filter when toggled
            )
            filter_regex_cb.pack(side="left", padx=(10, 0))

        # Get visible columns from user preferences
        columns = tuple(self._get_visible_columns())
        self._current_columns = columns  # Store for use in _async_insert_rows

        # Create or reconfigure treeview
        if self.csv_treeview is None:
            self._main_vocab_tv = VocabTreeview(
                parent=self.treeview_frame,
                columns=columns,
                tag_prefix="",
                feedback_manager=self._feedback_manager,
                on_click_callback=self._on_vocab_tv_click,
                on_right_click_callback=self._on_vocab_tv_right_click,
            )
            self.csv_treeview = self._main_vocab_tv.widget  # Backward compat

            # Configure column widths: saved widths preferred over DPI-aware defaults
            # stretch=False on ALL columns prevents Tk from recalculating widths
            # on layout events, which was causing user-dragged widths to snap back.
            saved_widths = self._load_column_widths()
            if saved_widths:
                computed = None
            else:
                content_font, heading_font = get_vocab_font_specs()
                available_w = self.treeview_frame.winfo_width()
                if available_w < 100:
                    available_w = 900
                computed = compute_column_widths(
                    list(columns), content_font, heading_font, available_w
                )

            # Build combined widths dict
            widths = {}
            for col in columns:
                if saved_widths and col in saved_widths:
                    widths[col] = saved_widths[col]
                elif computed and col in computed:
                    widths[col] = computed[col]

            self._main_vocab_tv.configure_columns(columns, widths)
            self._main_vocab_tv.configure_sortable_columns(columns, self._sort_by_column)

            # Add scrollbars and grid layout (row=1 for filter bar at row=0)
            vsb, hsb = self._main_vocab_tv.add_scrollbars(self.treeview_frame)
            self.csv_treeview.grid(row=1, column=0, sticky="nsew")
            vsb.grid(row=1, column=1, sticky="ns")
            hsb.grid(row=2, column=0, sticky="ew")

            # Create shared context menu
            self._create_context_menu()

        # Cancel any pending async insertion before starting new one
        # Uses a generation counter so old callbacks detect they're stale,
        # even after the new insertion resets _is_loading.
        if self._is_loading:
            logger.debug("Cancelling pending async insertion for new data")
            self._insertion_cancelled = True
            self._is_loading = False
        self._insertion_generation = getattr(self, "_insertion_generation", 0) + 1

        # Clear existing data and item mapping
        self._main_vocab_tv.clear()

        # Calculate how many items to load initially
        initial_load = min(ROWS_PER_PAGE, self._vocab_total_items)

        logger.debug(
            "Showing %s of %s terms (pagination: %s per page)",
            initial_load,
            self._vocab_total_items,
            ROWS_PER_PAGE,
        )

        # Start async batch insertion for initial load
        self._async_insert_rows(data, 0, initial_load)

    def _display_filtered_section(self, filtered_data: list):
        """
        Display a collapsible "Filtered out (N terms)" section below the main vocab table.

        Shows terms excluded by frequency filters in a visually muted treeview.
        Collapsed by default. Uses the same columns and right-click menu as the
        main table. Each VocabTreeview instance owns its own item_to_data dict.

        Args:
            filtered_data: List of term dicts that were filtered out
        """
        vocab_tab = self.tabview.tab("Vocabulary")

        # Remove previous filtered section if it exists
        if self._filtered_frame is not None:
            self._filtered_frame.destroy()
            self._filtered_frame = None
            self._filtered_treeview = None
            self._filtered_vocab_tv = None
            self._filtered_header = None
            self._filtered_inner_frame = None

        if not filtered_data:
            # Reset grid weight — main table gets all space
            vocab_tab.grid_rowconfigure(1, weight=0)
            return

        # Apply filtered-list score floor (hide low-scoring noise)
        from src.user_preferences import get_user_preferences

        filtered_floor = get_user_preferences().get("vocab_filtered_score_floor", 40)
        filtered_data = [
            item
            for item in filtered_data
            if isinstance(item, dict) and item.get(VF.QUALITY_SCORE, 0) >= filtered_floor
        ]
        if not filtered_data:
            vocab_tab.grid_rowconfigure(1, weight=0)
            return

        # Store raw data for expand/collapse
        self._filtered_vocab_data_raw = filtered_data
        self._filtered_expanded = False
        count = len(filtered_data)

        # Outer frame for entire section
        self._filtered_frame = ctk.CTkFrame(vocab_tab, **FRAME_STYLES["card"])
        self._filtered_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._filtered_frame.grid_columnconfigure(0, weight=1)

        # Clickable header bar
        header_text = f"\u25b6  Filtered out ({count} term{'s' if count != 1 else ''})"
        self._filtered_header = ctk.CTkLabel(
            self._filtered_frame,
            text=header_text,
            font=FONTS["small"],
            text_color="#888888",
            anchor="w",
            cursor="hand2",
        )
        self._filtered_header.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 4))
        self._filtered_header.bind("<Button-1>", self._toggle_filtered_section)

        # Inner frame (hidden until expanded)
        self._filtered_inner_frame = ctk.CTkFrame(self._filtered_frame, fg_color="transparent")
        # Not gridded yet — shown on expand

    def _toggle_filtered_section(self, _event=None):
        """Toggle the filtered terms section between collapsed and expanded."""
        if self._filtered_inner_frame is None:
            return

        # Debounce rapid clicks — disable header briefly after each toggle
        if not getattr(self, "_filtered_toggle_ready", True):
            return
        self._filtered_toggle_ready = False
        self.after(300, lambda: setattr(self, "_filtered_toggle_ready", True))

        vocab_tab = self.tabview.tab("Vocabulary")

        if self._filtered_expanded:
            # Collapse
            self._filtered_inner_frame.grid_remove()
            self._filtered_expanded = False
            count = len(self._filtered_vocab_data_raw)
            self._filtered_header.configure(
                text=f"\u25b6  Filtered out ({count} term{'s' if count != 1 else ''})"
            )
            # Give main table all space
            vocab_tab.grid_rowconfigure(1, weight=0)
        else:
            # Expand
            self._filtered_inner_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
            self._filtered_inner_frame.grid_columnconfigure(0, weight=1)
            self._filtered_inner_frame.grid_rowconfigure(0, weight=1)
            self._filtered_expanded = True
            count = len(self._filtered_vocab_data_raw)
            self._filtered_header.configure(
                text=f"\u25bc  Filtered out ({count} term{'s' if count != 1 else ''})"
            )
            # Give filtered section some vertical space
            vocab_tab.grid_rowconfigure(1, weight=1)

            # Build the treeview on first expand (lazy)
            if self._filtered_treeview is None:
                self._build_filtered_treeview()

    def _build_filtered_treeview(self):
        """Build the treeview for filtered terms (called on first expand)."""
        columns = tuple(self._get_visible_columns())

        self._filtered_vocab_tv = VocabTreeview(
            parent=self._filtered_inner_frame,
            columns=columns,
            tag_prefix="filtered_",
            feedback_manager=self._feedback_manager,
            on_click_callback=self._on_vocab_tv_click,
            on_right_click_callback=self._on_vocab_tv_right_click,
        )
        self._filtered_treeview = self._filtered_vocab_tv.widget  # Backward compat
        self._filtered_vocab_tv.configure_columns(columns)
        self._filtered_vocab_tv.configure_sortable_columns(columns, self._sort_filtered_by_column)

        # Add vertical scrollbar only (filtered section is compact)
        vsb, _hsb = self._filtered_vocab_tv.add_scrollbars(self._filtered_inner_frame)
        self._filtered_treeview.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Sort by score descending by default
        display_data = self._sort_vocab_data(
            self._filtered_vocab_data_raw, "Score", ascending=False
        )
        self._filtered_unsorted_data = list(self._filtered_vocab_data_raw)
        self._filtered_sort_column = "Score"
        self._filtered_sort_ascending = False
        self._update_filtered_sort_headers()

        # Insert rows using VocabTreeview (handles values, tags, and data mapping)
        for i, item in enumerate(display_data):
            if not isinstance(item, dict):
                continue
            self._filtered_vocab_tv.insert_row(item, i, columns)

        logger.debug("Filtered section: %s terms displayed", len(self._filtered_vocab_data_raw))

    def _sort_filtered_by_column(self, column: str):
        """Sort filtered vocabulary table by column (mirrors _sort_by_column)."""
        if not self._filtered_unsorted_data or self._filtered_treeview is None:
            return

        # Determine new sort state (cycle: asc → desc → unsorted)
        if self._filtered_sort_column == column:
            if self._filtered_sort_ascending:
                self._filtered_sort_ascending = False
            else:
                self._filtered_sort_column = None
                self._filtered_sort_ascending = True
        else:
            self._filtered_sort_column = column
            self._filtered_sort_ascending = True

        # Prepare sorted data
        if self._filtered_sort_column is None:
            sorted_data = list(self._filtered_unsorted_data)
        else:
            sorted_data = self._sort_vocab_data(
                self._filtered_unsorted_data,
                self._filtered_sort_column,
                self._filtered_sort_ascending,
            )

        self._update_filtered_sort_headers()
        self._redisplay_filtered_data(sorted_data)

    def _update_filtered_sort_headers(self):
        """Update filtered table column headers with sort indicator (▲/▼)."""
        if self._filtered_treeview is None:
            return
        for col in self._get_visible_columns():
            if col == self._filtered_sort_column:
                indicator = " ▲" if self._filtered_sort_ascending else " ▼"
                self._filtered_treeview.heading(col, text=f"{col}{indicator}")
            else:
                self._filtered_treeview.heading(col, text=col)

    def _redisplay_filtered_data(self, sorted_data: list):
        """Clear and repopulate the filtered treeview with sorted data."""
        if self._filtered_treeview is None:
            return
        self._filtered_treeview.delete(*self._filtered_treeview.get_children())
        columns = tuple(self._get_visible_columns())
        for i, item in enumerate(sorted_data):
            if not isinstance(item, dict):
                continue
            self._filtered_vocab_tv.insert_row(item, i, columns)

    def _display_semantic_results(self, results: list):
        """
        Display semantic search results using the SemanticPanel widget.

        Args:
            results: List of SemanticResult objects
        """
        if not results:
            logger.debug("No semantic search results to display")
            return

        # Set up follow-up callback if not already done
        # (must be done after MainWindow is fully initialized, not in __init__)
        if self._semantic_panel.on_ask_followup is None:
            main_window = self.winfo_toplevel()
            if hasattr(main_window, "_ask_followup_for_semantic_panel"):
                self._semantic_panel.set_followup_callback(
                    main_window._ask_followup_for_semantic_panel
                )
                logger.debug("Follow-up callback connected to MainWindow")

        # Display results and show the search panel (hide status label)
        self._semantic_panel.display_results(results)
        self.show_semantic_content()

        logger.debug("Showing %s search results", len(results))

    def _async_insert_rows(self, data: list, start_idx: int, end_idx: int):
        """
        Asynchronously insert rows into the Treeview in small batches.

        Uses after() to yield to the event loop between batches,
        keeping the GUI responsive during large data loads.

        Args:
            data: Full vocabulary data list
            start_idx: Starting index in data
            end_idx: Ending index (exclusive)
        """
        if self._is_loading:
            return
        self._is_loading = True
        self._insertion_cancelled = False  # Reset cancellation flag for new operation
        my_generation = self._insertion_generation  # Capture generation for this insertion

        current_idx = start_idx

        def insert_batch():
            nonlocal current_idx

            # Check if this operation was cancelled (new data arrived)
            # Also check generation counter — if a newer insertion started,
            # this callback is stale even if _insertion_cancelled was reset.
            if self._insertion_cancelled or self._insertion_generation != my_generation:
                logger.debug("Async insertion cancelled - stopping")
                self._is_loading = False
                return

            # If paused by window resize, reschedule and wait
            if self._batch_insertion_paused:
                self.after(BATCH_INSERT_DELAY_MS, insert_batch)
                return

            # Insert a batch of rows
            batch_end = min(current_idx + BATCH_INSERT_SIZE, end_idx)

            # Use stored columns (may be extended with NER/RAKE/BM25)
            current_columns = getattr(self, "_current_columns", GUI_DISPLAY_COLUMNS)

            for i in range(current_idx, batch_end):
                item = data[i]
                if isinstance(item, dict):
                    self._main_vocab_tv.insert_row(item, i, current_columns)
                else:
                    # Handle list format (legacy) — convert to dict for insert_row
                    legacy_dict = {}
                    col_list = list(current_columns)
                    for j, v in enumerate(item):
                        if j < len(col_list):
                            key = DISPLAY_TO_DATA_COLUMN.get(col_list[j], col_list[j])
                            legacy_dict[key] = v
                    self._main_vocab_tv.insert_row(legacy_dict, i, current_columns)

            current_idx = batch_end

            # Check if we need to insert more
            if current_idx < end_idx:
                # Schedule next batch with a small delay to allow UI to breathe
                self.after(BATCH_INSERT_DELAY_MS, insert_batch)
            else:
                # All rows inserted for this page
                self._vocab_display_offset = end_idx
                self._is_loading = False
                self._update_pagination_ui(data)

        # Start the first batch
        insert_batch()

    def _update_pagination_ui(self, data: list):
        """
        Update pagination UI after rows are loaded.

        Shows "Load More" button if more data is available,
        or info label if all data is shown.

        Args:
            data: Full vocabulary data list
        """
        # Auto-size columns to content after first batch of rows loads
        self._autosize_columns_to_content(data)

        total_items = len(data)
        displayed_items = self._vocab_display_offset

        logger.debug("Updating UI: %s/%s displayed", displayed_items, total_items)

        # Create or update "Load More" button
        if displayed_items < total_items:
            remaining = total_items - displayed_items

            if self._load_more_btn is None:
                self._load_more_btn = ctk.CTkButton(
                    self.treeview_frame, text="", height=28, **BUTTON_STYLES["primary"]
                )

            # Always update command to use current data (fixes stale closure)
            self._load_more_btn.configure(
                text=f"Load More ({remaining} remaining)",
                command=lambda d=data: self._load_more_rows(d),
            )
            self._load_more_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
            logger.debug("Load More button shown (%s remaining)", remaining)

            # Update info label
            if not hasattr(self, "vocab_info_label"):
                self.vocab_info_label = ctk.CTkLabel(
                    self.treeview_frame,
                    text="",
                    font=FONTS["small"],
                    text_color=COLORS["text_secondary"],
                )
            self.vocab_info_label.configure(
                text=f"Showing {displayed_items} of {total_items} terms • Full list available via 'Save to File'"
            )
            self.vocab_info_label.grid(
                row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 5)
            )

        else:
            # All items displayed
            logger.debug("All %s items displayed, hiding Load More", total_items)
            if self._load_more_btn is not None:
                self._load_more_btn.grid_remove()

            if hasattr(self, "vocab_info_label"):
                self.vocab_info_label.configure(text=f"Showing all {total_items} terms")
                self.vocab_info_label.grid(
                    row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5
                )

        # Call gc.collect() directly (lightweight operation)
        gc.collect()

    def _load_more_rows(self, data: list):
        """
        Load more rows when "Load More" button is clicked.

        Args:
            data: Full vocabulary data list
        """
        if self._is_loading:
            return

        start_idx = self._vocab_display_offset
        end_idx = min(start_idx + ROWS_PER_PAGE, len(data))

        logger.debug("Loading more: rows %s to %s", start_idx, end_idx)

        # Start async insertion
        self._async_insert_rows(data, start_idx, end_idx)

    def _create_context_menu(self):
        """Create right-click context menu for vocabulary table."""
        self.context_menu = Menu(
            self,
            tearoff=0,
            bg=get_color("menu_bg"),
            fg=get_color("menu_fg"),
            activebackground=get_color("menu_active_bg"),
            activeforeground=get_color("menu_active_fg"),
            font=("Segoe UI", 10),
        )
        self.context_menu.add_command(
            label="Exclude this term from future lists", command=self._exclude_selected_term
        )
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Copy term", command=self._copy_selected_term)

    def _on_vocab_tv_click(self, event, vocab_tv: VocabTreeview):
        """
        Callback from VocabTreeview left-click — delegates to feedback handler.

        Args:
            event: Tkinter event
            vocab_tv: The VocabTreeview instance that was clicked
        """
        self._on_treeview_click(event, vocab_tv=vocab_tv)

    def _on_vocab_tv_right_click(self, event, vocab_tv: VocabTreeview):
        """
        Callback from VocabTreeview right-click — shows context menu.

        Args:
            event: Tkinter event
            vocab_tv: The VocabTreeview instance that was clicked
        """
        self._on_right_click(event, vocab_tv=vocab_tv)

    def _on_right_click(self, event, vocab_tv: VocabTreeview | None = None):
        """Handle right-click on treeview - show column menu for header, context menu for rows."""
        if vocab_tv is None:
            return

        tv = vocab_tv.widget

        # Check if click is on header region
        region = tv.identify_region(event.x, event.y)
        if region == "heading":
            self._show_column_menu(event)
            return

        # Identify the row under cursor
        item_id = tv.identify_row(event.y)
        if item_id:
            tv.selection_set(item_id)
            values = tv.item(item_id, "values")
            if values and len(values) >= 1:
                self._selected_term = strip_display_prefix(values[0])
                self._update_alternatives_menu_item(item_id, vocab_tv)

                try:
                    self.context_menu.tk_popup(event.x_root, event.y_root)
                except Exception:
                    logger.error("Context menu popup failed", exc_info=True)
                finally:
                    self.context_menu.grab_release()

    def _update_alternatives_menu_item(self, item_id: str, vocab_tv: VocabTreeview | None = None):
        """
        Add or update dynamic menu items based on selected term.

        Rebuilds the separator + "View Alternatives" + "View in Context" items
        each time the user right-clicks a row.

        Args:
            item_id: Treeview item ID for the selected row
            vocab_tv: The VocabTreeview instance that owns this item
        """
        # Remove existing dynamic items (always at index 3 onward)
        menu_size = self.context_menu.index("end")
        if menu_size is not None and menu_size >= 3:
            try:
                self.context_menu.delete(3, "end")
            except Exception as e:
                logger.debug("Context menu cleanup failed: %s", e)

        # Look up term data from the VocabTreeview that owns this item
        term_data = vocab_tv.get_item_data(item_id) if vocab_tv else {}
        alternatives = term_data.get("_alternatives", [])
        is_person = str(term_data.get(VF.IS_PERSON, "")).lower() in ("yes", "true", "1")
        if not is_person:
            is_person = str(term_data.get(VF.TYPE, "")).lower() == "person"

        has_alts = bool(alternatives) and is_person
        alt_count = len(alternatives)

        # --- View Alternatives ---
        self.context_menu.add_separator()
        if has_alts:
            label = f"View Alternatives ({alt_count} variant"
            if alt_count != 1:
                label += "s"
            label += ")"
            self.context_menu.add_command(
                label=label,
                command=lambda: self._show_alternatives_dialog(term_data),
            )
        else:
            self.context_menu.add_command(
                label="View Alternatives",
                state="disabled",
            )

        # --- View in Context ---
        occurrences = 0
        doc_count = 0
        try:
            occurrences = int(term_data.get(VF.OCCURRENCES, 0))
            doc_count = int(term_data.get(VF.NUM_DOCS, 0))
        except (ValueError, TypeError):
            logger.debug("Could not parse occurrences/doc_count for context menu")
        except Exception:
            logger.error("Unexpected error parsing context menu data", exc_info=True)

        if occurrences > 0 and doc_count > 0:
            ctx_label = f"View in Context ({doc_count} doc"
            if doc_count != 1:
                ctx_label += "s"
            ctx_label += ")"
            self.context_menu.add_command(
                label=ctx_label,
                command=lambda td=term_data: self._show_context_dialog(td),
            )
        else:
            self.context_menu.add_command(
                label="View in Context",
                state="disabled",
            )

        # --- Why this score? ---
        self._add_score_explanation_menu_item(term_data)

    def _add_score_explanation_menu_item(self, term_data: dict):
        """
        Add "Why this score?" menu item if the ML model is trained.

        Args:
            term_data: Term dict for the selected row
        """
        from src.services.vocabulary_service import VocabularyService

        explanation = VocabularyService().explain_term_score(term_data)
        if explanation is not None:
            term_name = term_data.get("Term", "Unknown")
            self.context_menu.add_command(
                label="Why this score?",
                command=lambda: self._show_score_explanation(term_name, explanation),
            )
        else:
            self.context_menu.add_command(
                label="Why this score?",
                state="disabled",
            )

    def _open_guarded_dialog(self, dialog_key: str, dialog_factory):
        """
        Open a dialog with duplicate-open prevention.

        Tracks open state in self._dialog_open. Clears the flag when
        the dialog window is destroyed (via bind to <Destroy>).

        Args:
            dialog_key: Unique string key for this dialog type
            dialog_factory: Callable that creates and returns the dialog
        """
        if self._dialog_open.get(dialog_key):
            return
        self._dialog_open[dialog_key] = True

        try:
            dlg = dialog_factory()
            if dlg is not None:
                dlg.bind(
                    "<Destroy>",
                    lambda _e, k=dialog_key: self._dialog_open.pop(k, None),
                    add="+",
                )
            else:
                self._dialog_open.pop(dialog_key, None)
        except Exception:
            self._dialog_open.pop(dialog_key, None)
            raise

    def _show_score_explanation(self, term_name: str, explanation: dict):
        """
        Open the ScoreExplanationDialog.

        Args:
            term_name: Display name of the term
            explanation: Dict from score_explainer.explain_score()
        """
        from src.ui.score_explanation_dialog import ScoreExplanationDialog

        self._open_guarded_dialog(
            "score_explanation",
            lambda: ScoreExplanationDialog(self, term_name, explanation),
        )

    def _show_alternatives_dialog(self, term_data: dict):
        """
        Open the AlternativesDialog for a term.

        Args:
            term_data: Term dict with _alternatives and _canonical_reason keys
        """
        from src.ui.alternatives_dialog import AlternativesDialog

        self._open_guarded_dialog(
            "alternatives",
            lambda: AlternativesDialog(self, term_data),
        )

    def _show_context_dialog(self, term_data: dict):
        """
        Open the ContextViewerDialog showing term occurrences across documents.

        Gets processing_results from the main window via winfo_toplevel().

        Args:
            term_data: Term dict with at least 'Term' key
        """
        from src.ui.context_viewer_dialog import ContextViewerDialog

        try:
            main_window = self.winfo_toplevel()
            processing_results = getattr(main_window, "processing_results", [])
        except Exception:
            processing_results = []

        if not processing_results:
            logger.warning("No processing results available for context viewer")
            return

        self._open_guarded_dialog(
            "context_viewer",
            lambda: ContextViewerDialog(self, term_data, processing_results),
        )

    def _exclude_selected_term(self):
        """Exclude the selected term from future vocabulary extractions."""
        if not self._selected_term:
            return

        term = self._selected_term
        lower_term = term.lower().strip()

        # Confirm with user
        result = messagebox.askyesno(
            "Exclude Term",
            f"Exclude '{term}' from future rare word lists?\n\n"
            f"This will also exclude case variations like '{term.upper()}' and '{term.title()}'.\n\n"
            "You can undo this by editing:\n"
            f"{USER_VOCAB_EXCLUDE_PATH}",
            icon="question",
        )

        if not result:
            return

        # Add to exclusion file (dedup handled by _add_to_user_exclusion_list)
        try:
            self._add_to_user_exclusion_list(term)

            # Remove from current display
            selected = self.csv_treeview.selection()
            if selected:
                self.csv_treeview.delete(selected[0])

                # Also remove from internal data (both legacy and primary keys)
                for key in ("Names & Vocabulary", "Rare Word List (CSV)"):
                    self._outputs[key] = [
                        item
                        for item in self._outputs.get(key, [])
                        if isinstance(item, dict) and item.get(VF.TERM, "").lower() != lower_term
                    ]
                # Remove from unsorted cache so re-sorts don't resurrect it
                self._unsorted_vocab_data = [
                    item
                    for item in self._unsorted_vocab_data
                    if isinstance(item, dict) and item.get(VF.TERM, "").lower() != lower_term
                ]

            messagebox.showinfo(
                "Term Excluded",
                f"'{term}' will not appear in future rare word lists.\n\n"
                "Note: This takes effect on the next vocabulary extraction.",
            )

        except Exception as e:
            logger.debug("Failed to save exclusion: %s", e)
            messagebox.showerror(
                "Error", f"Failed to save exclusion: {e}\n\nPlease check file permissions."
            )

    def _copy_selected_term(self):
        """Copy the selected term to clipboard."""
        if self._selected_term:
            try:
                self.clipboard_clear()
                self.clipboard_append(self._selected_term)
            except Exception as e:
                logger.warning("Clipboard copy failed: %s", e)

    def _add_to_user_exclusion_list(self, term: str) -> None:
        """
        Add a term to the user exclusion list (silent, no dialog).

        Called when user gives negative feedback (-1) to a term.
        The term will be filtered out of future vocabulary extractions.

        Includes detailed logging for exclusion persistence debugging.

        Args:
            term: The term to exclude (case-insensitive)
        """
        if not term:
            return

        lower_term = term.lower().strip()
        logger.debug("Adding '%s' to exclusion list at %s", lower_term, USER_VOCAB_EXCLUDE_PATH)

        try:
            # Use Path methods for consistency
            USER_VOCAB_EXCLUDE_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Check if term is already in the list
            existing_terms = set()
            if USER_VOCAB_EXCLUDE_PATH.exists():
                with open(USER_VOCAB_EXCLUDE_PATH, encoding="utf-8") as f:
                    existing_terms = {line.strip().lower() for line in f if line.strip()}
                logger.debug("Existing exclusions: %s terms", len(existing_terms))

            if lower_term in existing_terms:
                logger.debug("Term '%s' already in exclusion list, skipping", term)
                return

            # Append to file
            with open(USER_VOCAB_EXCLUDE_PATH, "a", encoding="utf-8") as f:
                f.write(f"{lower_term}\n")

            # Verify write succeeded
            if USER_VOCAB_EXCLUDE_PATH.exists():
                with open(USER_VOCAB_EXCLUDE_PATH, encoding="utf-8") as f:
                    new_count = sum(1 for line in f if line.strip())
                logger.debug(
                    "Successfully added '%s' to exclusion list (now %s total exclusions)",
                    term,
                    new_count,
                )
            else:
                logger.debug("WARNING: File not found after write: %s", USER_VOCAB_EXCLUDE_PATH)

        except Exception as e:
            logger.debug("ERROR: Failed to add '%s' to exclusion list: %s", term, e)

    def _build_vocab_csv(self, vocab_data: list) -> str:
        """
        Build CSV string from vocabulary data.

        Respects the vocab_export_format user preference:
        - "all": All columns including Quality Score, Occurrences, Google Rarity Rank
        - "basic": Term, Score, Is Person, Found By
        - "terms_only": Just the Term column

        Args:
            vocab_data: List of vocabulary dicts

        Returns:
            CSV formatted string
        """
        if not vocab_data:
            return ""

        prefs = get_user_preferences()
        export_format = prefs.get("vocab_export_format", "basic")

        # Determine columns based on format
        if export_format == "all":
            columns = list(ALL_EXPORT_COLUMNS)
        elif export_format == "terms_only":
            columns = [VF.TERM]
        else:  # "basic" (default)
            columns = list(GUI_DISPLAY_COLUMNS)

        # Respect auto-hide: remove "# Docs" when hidden (single-doc session)
        if not self._column_visibility.get(VF.NUM_DOCS, True) and VF.NUM_DOCS in columns:
            columns.remove(VF.NUM_DOCS)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(columns)

        for item in vocab_data:
            if isinstance(item, dict):
                row = []
                for col in columns:
                    # Map display column to data field if needed (e.g., "Score" -> "Quality Score")
                    data_col = DISPLAY_TO_DATA_COLUMN.get(col, col)
                    row.append(item.get(data_col, ""))
                writer.writerow(row)
            else:
                # Legacy list format
                writer.writerow(item[: len(columns)])

        return output.getvalue()

    def get_current_content_for_export(self):
        """
        Returns the currently displayed content for copy/save operations.

        For vocabulary CSV, respects the vocab_export_format setting:
        - "all": All columns including Quality Score, Occurrences, Google Rarity Rank
        - "basic": Term, Score, Is Person, Found By
        - "terms_only": Just the Term column
        """
        current_tab = self.tabview.get()

        if current_tab == "Vocabulary":
            data = self._get_filtered_vocab_data()
            return self._build_vocab_csv(data)
        elif current_tab == "Search":
            # Get export content from SemanticPanel if available
            if self._semantic_panel is not None:
                return self._semantic_panel.get_export_content()
            return ""
        elif current_tab == "Key Excerpts":
            # Return text from summary display
            return self.summary_text_display.get("0.0", "end").strip()
        return ""

    def copy_to_clipboard(self):
        """Copy currently displayed content to clipboard."""
        content = self.get_current_content_for_export()
        if content:
            try:
                self.clipboard_clear()
                self.clipboard_append(content)
            except Exception as e:
                logger.warning("Clipboard copy failed: %s", e)
                return

            # Brief button flash for immediate feedback
            original_text = self.copy_btn.cget("text")
            self.copy_btn.configure(text="Copied!")

            def _reset_copy_btn():
                try:
                    self.copy_btn.configure(text=original_text)
                except Exception:
                    pass  # Widget destroyed during delay

            self.after(1500, _reset_copy_btn)

            # Status bar confirmation with count
            current_tab = self.tabview.get()
            main_window = self.winfo_toplevel()
            if current_tab == "Vocabulary":
                filtered_data = self._get_filtered_vocab_data()
                main_window.set_status(
                    f"Copied {len(filtered_data)} terms to clipboard", duration_ms=5000
                )
            elif current_tab == "Key Excerpts":
                main_window.set_status("Copied key excerpts to clipboard", duration_ms=5000)
        else:
            messagebox.showwarning("Empty", "No content to copy.")

    def save_to_file(self):
        """Save currently displayed content to file."""
        content = self.get_current_content_for_export()
        if not content:
            messagebox.showwarning("Empty", "No content to save.")
            return

        # Use current tab to determine file type
        current_tab = self.tabview.get()
        default_filename = "output"
        filetypes = [("All Files", "*.*")]

        if current_tab == "Vocabulary":
            default_filename = "names_vocabulary.csv"
            filetypes = [("CSV Files", "*.csv"), ("All Files", "*.*")]
        elif current_tab == "Search":
            default_filename = "semantic_results.txt"
            filetypes = [("Text Files", "*.txt"), ("All Files", "*.*")]
        elif current_tab == "Key Excerpts":
            default_filename = "summary.txt"
            filetypes = [("Text Files", "*.txt"), ("All Files", "*.*")]

        # Remember last export folder
        from src.services import DocumentService

        prefs = get_user_preferences()
        initial_dir = (
            prefs.get("last_export_path") or DocumentService().get_default_documents_folder()
        )

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=filetypes,
            initialfile=default_filename,
            initialdir=initial_dir,
            title="Save Output",
        )
        if filepath:
            from pathlib import Path

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as e:
                logger.error("Failed to save file '%s': %s", filepath, e)
                main_window = self.winfo_toplevel()
                if hasattr(main_window, "set_status_error"):
                    main_window.set_status_error(f"Save failed: {e}")
                return
            except Exception as e:
                logger.error("Unexpected error saving file '%s': %s", filepath, e, exc_info=True)
                return

            # Remember last export folder
            prefs.set("last_export_path", str(Path(filepath).parent))

            # Brief button flash for immediate feedback
            original_text = self.save_btn.cget("text")
            self.save_btn.configure(text="Saved!")

            def _reset_save_btn():
                try:
                    self.save_btn.configure(text=original_text)
                except Exception:
                    pass  # Widget destroyed during delay

            self.after(1500, _reset_save_btn)

            # Status bar confirmation with details
            main_window = self.winfo_toplevel()
            filename = os.path.basename(filepath)
            if current_tab == "Vocabulary":
                filtered_data = self._get_filtered_vocab_data()
                main_window.set_status(
                    f"Saved {len(filtered_data)} terms to {filename}", duration_ms=5000
                )
            elif current_tab == "Search":
                main_window.set_status(f"Saved search results to {filename}", duration_ms=5000)
            elif current_tab == "Key Excerpts":
                main_window.set_status(f"Saved key excerpts to {filename}", duration_ms=5000)

    def _export_vocab(self, format_key: str):
        """
        Export vocabulary in the given format.

        Shared boilerplate: get filtered data, empty check, build path,
        write, save path, status bar, error handling.

        Args:
            format_key: One of "csv", "txt", "word", "pdf", "html"
        """
        if self._exporting_vocab:
            return
        self._exporting_vocab = True

        try:
            self._export_vocab_impl(format_key)
        except Exception:
            logger.error("Vocabulary export failed", exc_info=True)
        finally:
            self._exporting_vocab = False

    def _export_vocab_impl(self, format_key: str):
        """Implementation of _export_vocab, guarded by _exporting_vocab flag."""
        from datetime import datetime
        from pathlib import Path

        from src.services import DocumentService, get_export_service

        ext_map = {
            "csv": ".csv",
            "txt": ".txt",
            "word": ".docx",
            "pdf": ".pdf",
            "html": ".html",
        }
        ext = ext_map[format_key]

        vocab_data = self._get_filtered_vocab_data()
        if not vocab_data:
            messagebox.showwarning(
                "No Data", "No vocabulary data to export.\n\nProcess documents first."
            )
            return

        prefs = get_user_preferences()
        export_path = (
            prefs.get("last_export_path") or DocumentService().get_default_documents_folder()
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vocabulary_{timestamp}{ext}"
        filepath = os.path.join(export_path, filename)

        try:
            error_detail = None
            if format_key == "csv":
                csv_content = self._build_vocab_csv(vocab_data)
                with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
                    f.write(csv_content)
                success = True
            else:
                export_service = get_export_service()
                if format_key in ("word", "pdf"):
                    include_details = any(
                        self._column_visibility.get(col, False)
                        for col in [
                            VF.NER,
                            VF.RAKE,
                            VF.BM25,
                            VF.TOPICRANK,
                            VF.MEDICALNER,
                            VF.YAKE,
                            VF.ALGO_COUNT,
                        ]
                    )
                    is_single_doc = not self._column_visibility.get(VF.NUM_DOCS, True)
                    write_fn = {
                        "word": export_service.export_vocabulary_to_word,
                        "pdf": export_service.export_vocabulary_to_pdf,
                    }
                    success, error_detail = write_fn[format_key](
                        vocab_data, filepath, include_details, is_single_doc=is_single_doc
                    )
                elif format_key == "txt":
                    success, error_detail = export_service.export_vocabulary_to_txt(
                        vocab_data, filepath
                    )
                else:  # html
                    visible_columns = self._get_visible_columns()
                    success, error_detail = export_service.export_vocabulary_to_html(
                        vocab_data, filepath, visible_columns
                    )

            if success:
                prefs.set("last_export_path", str(Path(filepath).parent))
                folder_name = Path(export_path).name
                main_window = self.winfo_toplevel()
                if hasattr(main_window, "set_status"):
                    main_window.set_status(
                        f"Exported {len(vocab_data)} terms to {folder_name}/{filename}",
                        duration_ms=5000,
                    )
                logger.debug("Exported %s terms to %s: %s", len(vocab_data), format_key, filepath)
            else:
                detail = f"\n\n{error_detail}" if error_detail else ""
                messagebox.showerror("Export Failed", f"Could not export to {ext} file.{detail}")

        except Exception as e:
            logger.warning("Vocab export failed: %s", e)
            messagebox.showerror("Export Failed", f"Could not save file:\n{e}")

    # --- Deprecated single-section export shortcuts (Mar 2026) ---
    # Kept for potential future use. Export dropdown now uses _export_all()
    # which exports all sections (vocab, search, key excerpts) together.
    #
    # def _quick_export_vocab_csv(self):
    #     """Quick export vocabulary to CSV file."""
    #     self._export_vocab("csv")
    #
    # def _export_vocab_to_word(self):
    #     """Export vocabulary to Word document."""
    #     self._export_vocab("word")
    #
    # def _export_vocab_to_pdf(self):
    #     """Export vocabulary to PDF document."""
    #     self._export_vocab("pdf")
    #
    # def _export_vocab_to_txt(self):
    #     """Export vocabulary to plain text file."""
    #     self._export_vocab("txt")
    #
    # def _export_vocab_to_html(self):
    #     """Export vocabulary to interactive HTML file."""
    #     self._export_vocab("html")

    # -----------------------------------------------------------------
    # Combined Export (all sections: vocab + search + key excerpts)
    # -----------------------------------------------------------------

    def _get_exportable_semantic_results(self) -> list:
        """Return semantic results that meet the export relevance threshold."""
        results = self._outputs.get("Search") or self._outputs.get("Semantic Results", [])
        return [r for r in results if hasattr(r, "is_exportable") and r.is_exportable]

    def _get_summary_text(self) -> str:
        """Return the key excerpts text for export."""
        return (
            self._outputs.get("Key Excerpts") or self._outputs.get("Meta-Summary") or ""
        ).strip()

    def _export_all(self, format_key: str):
        """
        Export all sections (vocabulary, search, key excerpts) in the given format.

        Args:
            format_key: One of "txt", "pdf", "html"
        """
        if self._exporting_vocab:
            return
        self._exporting_vocab = True
        try:
            self._export_all_impl(format_key)
        except Exception:
            logger.error("Combined export failed", exc_info=True)
        finally:
            self._exporting_vocab = False

    def _export_all_impl(self, format_key: str):
        """Implementation of _export_all, guarded by _exporting_vocab flag."""
        from datetime import datetime

        from src.services import DocumentService

        vocab_data = self._get_filtered_vocab_data()
        semantic_results = self._get_exportable_semantic_results()
        summary_text = self._get_summary_text()

        if not vocab_data and not semantic_results and not summary_text:
            messagebox.showwarning("No Data", "No data to export.\n\nProcess documents first.")
            return

        ext_map = {"txt": ".txt", "pdf": ".pdf", "html": ".html"}
        prefs = get_user_preferences()
        export_path = (
            prefs.get("last_export_path") or DocumentService().get_default_documents_folder()
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"case_report_{timestamp}{ext_map[format_key]}"
        filepath = os.path.join(export_path, filename)

        success, error_detail = self._dispatch_combined_export(
            format_key, vocab_data, semantic_results, summary_text, filepath
        )
        self._handle_export_result(success, error_detail, filepath, export_path, prefs)

    def _dispatch_combined_export(
        self, format_key, vocab_data, semantic_results, summary_text, filepath
    ):
        """Dispatch to the appropriate combined exporter by format."""
        from src.services import get_export_service

        export_service = get_export_service()

        if format_key == "html":
            visible_columns = self._get_visible_columns()
            return export_service.export_combined_html(
                vocab_data, semantic_results, summary_text, filepath, visible_columns
            )
        elif format_key == "pdf":
            include_details = self._should_include_vocab_details()
            return export_service.export_combined_to_pdf(
                vocab_data,
                semantic_results,
                filepath,
                include_vocab_details=include_details,
                summary_text=summary_text,
            )
        else:  # txt
            return export_service.export_combined_to_txt(
                vocab_data, semantic_results, summary_text, filepath
            )

    def _should_include_vocab_details(self) -> bool:
        """Check if algorithm detail columns are visible (for Word/PDF exports)."""
        return any(
            self._column_visibility.get(col, False)
            for col in [
                VF.NER,
                VF.RAKE,
                VF.BM25,
                VF.TOPICRANK,
                VF.MEDICALNER,
                VF.YAKE,
                VF.ALGO_COUNT,
            ]
        )

    def _handle_export_result(self, success, error_detail, filepath, export_path, prefs):
        """Handle post-export status messaging and preference saving."""
        from pathlib import Path

        if success:
            prefs.set("last_export_path", str(Path(filepath).parent))
            folder_name = Path(export_path).name
            filename = Path(filepath).name
            main_window = self.winfo_toplevel()
            if hasattr(main_window, "set_status"):
                main_window.set_status(f"Exported to {folder_name}/{filename}", duration_ms=5000)
            logger.debug("Exported to %s", filepath)
        else:
            ext = Path(filepath).suffix
            detail = f"\n\n{error_detail}" if error_detail else ""
            messagebox.showerror("Export Failed", f"Could not export to {ext} file.{detail}")

    def _on_export_format_selected(self, choice: str):
        """
        Handle export format selection from dropdown.

        Exports all sections (vocabulary, search results, key excerpts)
        in the chosen format.

        Args:
            choice: Selected format ("Export...", "TXT", "PDF", "HTML")
        """
        if choice == "Export...":
            return  # Placeholder, do nothing

        format_map = {"TXT": "txt", "PDF": "pdf", "HTML": "html"}
        if choice in format_map:
            self._export_all(format_map[choice])

        # Reset dropdown to placeholder
        self.export_dropdown.set("Export...")

    def _on_treeview_click(self, event, vocab_tv: VocabTreeview | None = None):
        """
        Handle left-click on treeview for feedback columns.

        Detects clicks on the Keep or Skip columns and toggles the
        feedback state for that term.

        Args:
            event: Tkinter event
            vocab_tv: The VocabTreeview instance that was clicked
        """
        if vocab_tv is None:
            return
        tv = vocab_tv.widget
        column = tv.identify_column(event.x)
        item_id = tv.identify_row(event.y)

        if not item_id:
            return

        # Dynamically find Keep and Skip column indices
        current_columns = getattr(self, "_current_columns", GUI_DISPLAY_COLUMNS)
        try:
            keep_idx = current_columns.index(VF.KEEP) + 1  # 1-based
            skip_idx = current_columns.index(VF.SKIP) + 1  # 1-based
        except ValueError:
            return  # Keep/Skip columns not found
        except Exception:
            logger.error("Unexpected error finding feedback columns", exc_info=True)
            return

        # Check if click was on a feedback column
        if column == f"#{keep_idx}":  # Keep column
            self._toggle_feedback(item_id, +1, vocab_tv)
        elif column == f"#{skip_idx}":  # Skip column
            self._toggle_feedback(item_id, -1, vocab_tv)

    def _check_corpus_and_warn(self) -> bool:
        """
        Check corpus status and warn user if not ready.

        Shows a warning dialog once per session if corpus has < 5 documents.
        User can choose to continue or cancel feedback.

        Returns:
            True to proceed with feedback, False to cancel
        """
        # Already warned this session, proceed silently
        if self._corpus_warning_shown:
            return True

        from src.services import VocabularyService

        corpus_manager = VocabularyService().get_corpus_manager()
        if corpus_manager.is_corpus_ready():
            return True  # Corpus OK, proceed

        # Mark warning as shown (only warn once per session)
        self._corpus_warning_shown = True

        from tkinter import messagebox

        result = messagebox.askyesno(
            "Corpus Not Ready",
            f"Your vocabulary corpus has {corpus_manager.get_document_count()}/5 documents.\n\n"
            "The ML model learns better with a corpus of past transcripts. "
            "Consider adding documents in Settings > Corpus before providing feedback.\n\n"
            "Continue providing feedback anyway?",
            icon="warning",
        )
        return result

    def _toggle_feedback(
        self, item_id: str, feedback_type: int, vocab_tv: VocabTreeview | None = None
    ):
        """
        Toggle feedback state for a vocabulary term.

        If the term already has this feedback, clear it.
        If the term has opposite or no feedback, set the new feedback.

        Args:
            item_id: Treeview item identifier
            feedback_type: +1 for Keep, -1 for Skip
            vocab_tv: VocabTreeview instance that owns this item
        """
        # Block feedback while extraction is in progress
        if self._extraction_in_progress:
            main_window = self.winfo_toplevel()
            if hasattr(main_window, "set_status"):
                main_window.set_status(
                    "Feedback disabled during extraction. Please wait for NER to complete.",
                    duration_ms=3000,
                )
            return

        # Warn about missing corpus (once per session)
        if not self._check_corpus_and_warn():
            return

        # Resolve VocabTreeview if not passed
        if vocab_tv is None:
            vocab_tv = self._vocab_tv_for_item(item_id)
        if vocab_tv is None:
            return

        values = vocab_tv.widget.item(item_id, "values")
        if not values or len(values) < 1:
            return

        term = strip_display_prefix(values[0])
        current_rating = self._feedback_manager.get_rating(term)

        # Toggle logic: if already this rating, clear it; otherwise set it
        new_rating = 0 if current_rating == feedback_type else feedback_type

        # Find full term data from internal storage for ML features
        term_data = self._find_term_data(term)
        if not term_data:
            term_data = {VF.TERM: term}

        # Record feedback (handles both setting and clearing)
        success = self._feedback_manager.record_feedback(term_data, new_rating)

        if success:
            current_columns = getattr(self, "_current_columns", GUI_DISPLAY_COLUMNS)
            vocab_tv.update_feedback_display(item_id, new_rating, current_columns)
            action = "Cleared" if new_rating == 0 else "Set"
            logger.debug("%s feedback for '%s': %s", action, term, new_rating)

            if new_rating == -1:
                self._add_to_user_exclusion_list(term)

            self._update_balance_hint()

    def _update_balance_hint(self):
        """
        Show a hint if user feedback is heavily lopsided (>75% one class).

        Helps users understand that balanced feedback improves ML predictions.
        Only shown after 20+ votes. Dismissable per session.
        """
        if self._balance_hint_dismissed:
            return

        keep_count = len(self._feedback_manager.get_rated_terms(rating_filter=+1))
        skip_count = len(self._feedback_manager.get_rated_terms(rating_filter=-1))
        total = keep_count + skip_count

        if total < 20:
            self._balance_hint.configure(text="")
            return

        majority_pct = max(keep_count, skip_count) / total
        if majority_pct < 0.75:
            self._balance_hint.configure(text="")
            return

        if keep_count > skip_count:
            msg = "Tip: Your feedback is mostly keeps \u2014 voting skip on some terms helps the model learn faster  (click to dismiss)"
        else:
            msg = "Tip: Your feedback is mostly skips \u2014 voting keep on good terms helps the model learn faster  (click to dismiss)"
        self._balance_hint.configure(text=msg, text_color=("gray50", "gray70"))

    def _dismiss_balance_hint(self):
        """Dismiss the balance hint for the rest of this session."""
        self._balance_hint_dismissed = True
        self._balance_hint.configure(text="")

    def _find_term_data(self, term: str) -> dict | None:
        """
        Find full term data from internal storage by term name.

        Searches both main vocab and filtered terms.

        Args:
            term: The term to search for (case-insensitive)

        Returns:
            Dictionary with term data, or None if not found
        """
        # Check primary key first, then legacy key
        vocab_data = self._outputs.get("Names & Vocabulary", [])
        if not vocab_data:
            vocab_data = self._outputs.get("Rare Word List (CSV)", [])

        lower_term = term.lower().strip()

        for item in vocab_data:
            if isinstance(item, dict) and item.get(VF.TERM, "").lower().strip() == lower_term:
                return item

        # Also search filtered terms
        for item in self._filtered_vocab_data_raw:
            if isinstance(item, dict) and item.get(VF.TERM, "").lower().strip() == lower_term:
                return item

        return None

    def _vocab_tv_for_item(self, item_id: str) -> VocabTreeview | None:
        """
        Return the VocabTreeview that owns a given item ID.

        Args:
            item_id: Treeview item identifier

        Returns:
            VocabTreeview instance, or None if not found
        """
        if self._filtered_vocab_tv and self._filtered_vocab_tv.has_item(item_id):
            return self._filtered_vocab_tv
        if self._main_vocab_tv and self._main_vocab_tv.has_item(item_id):
            return self._main_vocab_tv
        return None
