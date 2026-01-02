"""
Dynamic Output Display Widget for LocalScribe (Session 51 Update)

Displays Names & Vocabulary tables, Q&A results, and summaries using tab navigation.
Provides copy/save functionality for export.
The vocabulary display uses an Excel-like Treeview with frozen headers
and right-click context menu for excluding terms from future extractions.

Session 51 Updates (Tab Navigation):
- Replaced dropdown menu with CTkTabview for instant, glitch-free navigation
- Three persistent tabs: "Names & Vocab" | "Ask Questions" | "Summary"
- Tab switching uses frame stacking (tkraise) - no widget recreation
- Removed ~100 lines of visibility management code
- All content preserved across tab switches (scroll position, state)

Session 45 Updates:
- Renamed "Case Briefing" to "Names & Vocabulary" as primary output
- Added progress badge showing data source (NER only → NER + LLM)
- Output pane has distinct background color

Performance Optimizations (Session 14-16, 51):
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
import os
import threading
from tkinter import Menu, filedialog, messagebox, ttk

import customtkinter as ctk

from src.config import (
    USER_VOCAB_EXCLUDE_PATH,
    VOCABULARY_ROWS_PER_PAGE,
    VOCABULARY_BATCH_INSERT_SIZE,
    VOCABULARY_BATCH_INSERT_DELAY_MS,
)
from src.logging_config import debug_log
from src.user_preferences import get_user_preferences
from src.core.vocabulary.feedback_manager import get_feedback_manager
from src.ui.qa_panel import QAPanel
from src.ui.theme import FONTS, COLORS, BUTTON_STYLES, FRAME_STYLES, VOCAB_TABLE_TAGS

# Feedback icons (Unicode for cross-platform compatibility)
# Using checkmark (✓) and X (✗) for clearer approve/reject semantics
THUMB_UP_EMPTY = "☐"      # U+2610 Ballot Box (empty checkbox)
THUMB_UP_FILLED = "✓"     # U+2713 Check Mark (green via tag)
THUMB_DOWN_EMPTY = "☐"    # U+2610 Ballot Box (empty checkbox)
THUMB_DOWN_FILLED = "✗"   # U+2717 Ballot X (red via tag)

# Pagination settings (imported from config.py for centralized tuning)
ROWS_PER_PAGE = VOCABULARY_ROWS_PER_PAGE
BATCH_INSERT_SIZE = VOCABULARY_BATCH_INSERT_SIZE
BATCH_INSERT_DELAY_MS = VOCABULARY_BATCH_INSERT_DELAY_MS


# Column width configuration (in pixels) - controls text truncation
# Approximate character limits based on font size 10 Segoe UI
# Session 23: Added Quality Score, In-Case Freq, Freq Rank columns for filtering
# Session 25: Added feedback columns (Keep/Skip) for ML learning
# Session 43: Added Found By column for NER/LLM reconciliation
# Session 47: Added per-algorithm detection columns (NER, RAKE, BM25)
# Session 52: Removed Type column (unreliable), added Is Person (NER detection)
# Session 54: Removed Definition/Role (useless), added Score for ML transparency
COLUMN_CONFIG = {
    "Term": {"width": 180, "max_chars": 30},
    "Score": {"width": 55, "max_chars": 5},  # Quality Score - ML ranking value
    "Is Person": {"width": 65, "max_chars": 4},  # Yes/No - NER person detection
    "Found By": {"width": 120, "max_chars": 20},  # Algorithm names: NER, RAKE, BM25
    "Keep": {"width": 45, "max_chars": 3},
    "Skip": {"width": 45, "max_chars": 3},
    # Extended view columns (Session 47)
    "NER": {"width": 45, "max_chars": 4},
    "RAKE": {"width": 50, "max_chars": 4},
    "BM25": {"width": 50, "max_chars": 4},
    "Algo Count": {"width": 55, "max_chars": 3},  # Number of algorithms that found term
    # Export-only columns (not shown in GUI)
    "Quality Score": {"width": 85, "max_chars": 8},
    "In-Case Freq": {"width": 80, "max_chars": 8},
    "Freq Rank": {"width": 80, "max_chars": 10},
}

# Columns visible in the GUI Treeview
# Session 54: Simplified - removed Definition/Role, added Score for ML transparency
GUI_DISPLAY_COLUMNS = ("Term", "Score", "Is Person", "Found By", "Keep", "Skip")

# Session 47: Extended columns with per-algorithm detection (shown via "Show Details" button)
# Session 54: Removed Definition/Role, added Score
GUI_DISPLAY_COLUMNS_EXTENDED = ("Term", "Score", "Is Person", "Found By", "NER", "RAKE", "BM25", "Algo Count", "Keep", "Skip")

# All columns available for export (includes ML feature columns)
# Session 54: Removed Definition, kept Quality Score for full data export
ALL_EXPORT_COLUMNS = ("Term", "Quality Score", "Is Person", "Found By", "NER", "RAKE", "BM25", "Algo Count", "In-Case Freq", "Freq Rank")


def truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text to prevent Treeview row overflow.

    Args:
        text: Text to truncate
        max_chars: Maximum characters before truncation

    Returns:
        Truncated text with ellipsis if needed
    """
    if not text:
        return ""
    text = str(text).replace('\n', ' ').replace('\r', '').strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


class DynamicOutputWidget(ctk.CTkFrame):
    """Widget to dynamically display Names & Vocabulary, Q&A, or Summary outputs (Session 45)."""

    def __init__(self, master, **kwargs):
        # Session 45: Set distinct background color for output pane
        # Slightly darker/different than other panes to distinguish
        kwargs.setdefault('fg_color', ('#e8e8e8', '#1a1a2e'))  # Light/dark mode colors
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)  # Tab view gets all space

        # Session 51: Replaced dropdown with CTkTabview for better performance
        # Create tabview with three tabs: Names & Vocab, Q&A, Summary
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Create tabs
        self.tabview.add("Names & Vocab")
        self.tabview.add("Ask Questions")
        self.tabview.add("Summary")

        # Session 68: Bind tab change to show/hide appropriate button bar
        self.tabview.configure(command=self._on_tab_changed)

        # Configure tab grids
        self.tabview.tab("Names & Vocab").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Names & Vocab").grid_rowconfigure(0, weight=1)
        self.tabview.tab("Ask Questions").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Ask Questions").grid_rowconfigure(0, weight=1)
        self.tabview.tab("Summary").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Summary").grid_rowconfigure(0, weight=1)

        # Progress Badge (Session 45) - shows data source status for Names & Vocab tab
        self._progress_badge = ctk.CTkLabel(
            self.tabview.tab("Names & Vocab"),
            text="",
            font=FONTS["small"],
            text_color=("gray50", "gray70")
        )
        self._progress_badge.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))

        # Names & Vocab Tab: Treeview frame (initially None, created when needed)
        self.csv_treeview = None
        self.treeview_frame = None  # Frame to hold treeview and scrollbars

        # Right-click context menu for vocabulary exclusion
        self.context_menu = None
        self._selected_term = None

        # Q&A Tab: Q&A panel (created eagerly to prevent tab switching artifacts)
        # See: https://github.com/TomSchimansky/CustomTkinter/issues/1508
        self._qa_panel = QAPanel(self.tabview.tab("Ask Questions"))
        self._qa_panel.grid(row=0, column=0, sticky="nsew")

        # Summary Tab: Textbox for summaries
        self.summary_text_display = ctk.CTkTextbox(
            self.tabview.tab("Summary"),
            wrap="word"
        )
        self.summary_text_display.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.summary_text_display.insert(
            "0.0",
            "Generated summaries will appear here.\n\nProcess documents and enable 'Summary' to generate content."
        )

        # Button bar (below tabs)
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.copy_btn = ctk.CTkButton(self.button_frame, text="Copy to Clipboard", command=self.copy_to_clipboard)
        self.copy_btn.pack(side="left", padx=5)

        self.save_btn = ctk.CTkButton(self.button_frame, text="Save to File...", command=self.save_to_file)
        self.save_btn.pack(side="left", padx=5)

        # Session 47: Toggle button for showing per-algorithm detection details
        self._show_details = False
        self.detail_toggle_btn = ctk.CTkButton(
            self.button_frame,
            text="Show Details",
            command=self._toggle_detail_view,
            width=100,
            **BUTTON_STYLES["secondary"]
        )
        self.detail_toggle_btn.pack(side="left", padx=5)

        # Session 65: Quick Export CSV button for vocabulary
        self.export_csv_btn = ctk.CTkButton(
            self.button_frame,
            text="Export CSV",
            command=self._quick_export_vocab_csv,
            width=100,
            **BUTTON_STYLES["primary"]
        )
        self.export_csv_btn.pack(side="left", padx=5)

        # Internal storage for outputs (Session 45: Updated naming)
        self._outputs = {
            "Names & Vocabulary": [],  # Session 45: Primary output - people + terms
            "Ask Questions": [],  # Q&A results (replaces "Q&A Results")
            "Summary": "",  # Combined summary (replaces "Meta-Summary")
            # Backward compatibility keys
            "Meta-Summary": "",
            "Rare Word List (CSV)": [],
            "Q&A Results": [],
            "Case Briefing": "",
        }
        self._document_summaries = {}  # {filename: summary_text}
        self._briefing_sections = {}  # Section name -> content for navigation

        # Session 45: Data source tracking for progress badge
        self._extraction_source = "none"  # "none", "ner", "both"

        # Pagination state for vocabulary display
        self._vocab_display_offset = 0  # Current offset into vocabulary data
        self._vocab_total_items = 0  # Total items in vocabulary data
        self._load_more_btn = None  # "Load More" button reference
        self._is_loading = False  # Prevents duplicate load operations

        # Resize event debouncing to prevent glitchiness (Session 51)
        self._resize_after_id = None  # Track pending resize callbacks
        self._batch_insertion_paused = False  # Pause batch insertion during resize

        # Feedback manager for ML learning (Session 25)
        self._feedback_manager = get_feedback_manager()

        # Session 68: Track whether corpus warning has been shown this session
        self._corpus_warning_shown = False

        # Bind to Configure event for resize debouncing
        self.bind("<Configure>", self._on_window_resize)

    def _on_window_resize(self, event):
        """
        Debounce resize events to prevent batch insertion conflicts.

        When the window is resized/maximized, this pauses batch insertion
        and schedules a callback to resume after resizing stabilizes (100ms).

        Args:
            event: Tkinter Configure event
        """
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
        debug_log("[DYNAMIC OUTPUT] Resize complete - batch insertion resumed")

    def _on_tab_changed(self):
        """
        Handle tab change to show/hide appropriate button bars (Session 68).

        Hides the shared button bar when Q&A tab is active (since QAPanel
        has its own buttons), shows it for other tabs.
        """
        current_tab = self.tabview.get()

        if current_tab == "Ask Questions":
            # Hide shared button bar - QAPanel has its own buttons
            self.button_frame.grid_remove()
        else:
            # Show shared button bar for Names & Vocab and Summary tabs
            self.button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

            # Show/hide vocab-specific buttons based on tab
            if current_tab == "Names & Vocab":
                # Update progress badge when switching to this tab
                self._update_progress_badge(self._extraction_source)
                # Show vocab buttons - use after= to maintain correct order
                if not self.detail_toggle_btn.winfo_ismapped():
                    self.detail_toggle_btn.pack(side="left", padx=5, after=self.save_btn)
                if not self.export_csv_btn.winfo_ismapped():
                    self.export_csv_btn.pack(side="left", padx=5, after=self.detail_toggle_btn)
            else:
                # Hide vocab-specific buttons on Summary tab
                self.detail_toggle_btn.pack_forget()
                self.export_csv_btn.pack_forget()

    def _update_progress_badge(self, source: str):
        """
        Update the progress badge to show data source status (Session 45).

        Args:
            source: "none", "ner", or "both"
        """
        if source == "none":
            self._progress_badge.configure(text="")
        elif source == "ner":
            self._progress_badge.configure(
                text="Initial results (NER only)",
                text_color=("orange", "#ffaa00")
            )
        elif source == "both":
            self._progress_badge.configure(
                text="Enhanced results (NER + LLM)",
                text_color=("green", "#00cc66")
            )

    def set_extraction_source(self, source: str):
        """
        Set the data source for the current extraction (Session 45).

        Called by workflow orchestrator to update progress badge.

        Args:
            source: "none", "ner", or "both"
        """
        self._extraction_source = source
        # Update badge if Names & Vocabulary tab is currently active
        current_tab = self.tabview.get()
        if current_tab == "Names & Vocab":
            self._update_progress_badge(source)

    def _toggle_detail_view(self):
        """
        Toggle between basic and detailed column view (Session 47).

        When details are shown, displays NER, RAKE, and BM25 columns
        indicating which algorithms detected each term.
        """
        self._show_details = not self._show_details
        self.detail_toggle_btn.configure(
            text="Hide Details" if self._show_details else "Show Details"
        )
        # Refresh current view to apply new columns if Names & Vocab tab is active
        current_tab = self.tabview.get()
        if current_tab == "Names & Vocab":
            vocab_data = self._outputs.get("Names & Vocabulary", [])
            if not vocab_data:
                vocab_data = self._outputs.get("Rare Word List (CSV)", [])
            # Force treeview recreation to update columns
            if self.csv_treeview is not None:
                self.csv_treeview.destroy()
                self.csv_treeview = None
            self._display_csv(vocab_data)

    def cleanup(self):
        """
        Clean up resources when widget is no longer needed.
        Call this to free memory after heavy processing.
        """
        # Clear internal data storage (must match __init__ structure)
        self._outputs = {
            "Names & Vocabulary": [],  # Session 45: Primary output
            "Ask Questions": [],  # Q&A results
            "Summary": "",  # Combined summary
            # Backward compatibility keys
            "Meta-Summary": "",
            "Rare Word List (CSV)": [],
            "Q&A Results": [],
            "Case Briefing": "",
        }
        self._document_summaries = {}
        self._briefing_sections = {}
        self._extraction_source = "none"  # Reset progress badge state

        # Clear treeview data if it exists
        if self.csv_treeview is not None:
            self.csv_treeview.delete(*self.csv_treeview.get_children())

        # Force garbage collection
        gc.collect()
        debug_log("[VOCAB DISPLAY] Cleanup completed, memory freed.")

    def update_outputs(
        self,
        meta_summary: str = "",
        vocab_csv_data: list = None,
        document_summaries: dict = None,
        qa_results: list = None,
        briefing_text: str = "",
        briefing_sections: dict = None,
        # Session 45 new parameters
        names_vocab_data: list = None,
        summary_text: str = "",
        extraction_source: str = None,
    ):
        """
        Updates the internal storage with new outputs and refreshes the dropdown.

        Args:
            meta_summary: The generated meta-summary text (legacy).
            vocab_csv_data: A list of dicts representing vocabulary data (legacy).
            document_summaries: A dictionary of {filename: summary_text}.
            qa_results: A list of QAResult objects from Q&A processing.
            briefing_text: The formatted Case Briefing Sheet text (legacy).
            briefing_sections: Dict mapping section names to content for navigation.
            names_vocab_data: Session 45 - combined people + vocabulary data.
            summary_text: Session 45 - combined summary text.
            extraction_source: Session 45 - "ner" or "both" for progress badge.
        """
        # Session 45: Primary outputs
        if names_vocab_data is not None:
            self._outputs["Names & Vocabulary"] = names_vocab_data
        if summary_text:
            self._outputs["Summary"] = summary_text
        if extraction_source:
            self._extraction_source = extraction_source

        # Legacy support
        if meta_summary:
            self._outputs["Meta-Summary"] = meta_summary
            if not self._outputs.get("Summary"):
                self._outputs["Summary"] = meta_summary
        if vocab_csv_data is not None:
            self._outputs["Rare Word List (CSV)"] = vocab_csv_data
            if self._outputs.get("Names & Vocabulary") is None:
                self._outputs["Names & Vocabulary"] = vocab_csv_data
        if document_summaries:
            self._document_summaries.update(document_summaries)
        if qa_results is not None:
            self._outputs["Q&A Results"] = qa_results
            self._outputs["Ask Questions"] = qa_results
        if briefing_text:
            self._outputs["Case Briefing"] = briefing_text
        if briefing_sections is not None:
            self._briefing_sections = briefing_sections

        self._refresh_tabs()

    def _refresh_tabs(self):
        """
        Refresh tabs based on available outputs (Session 51).

        Enables/disables tabs based on data availability and populates content.
        """
        # Names & Vocabulary tab
        vocab_data = self._outputs.get("Names & Vocabulary") or self._outputs.get("Rare Word List (CSV)")
        if vocab_data:
            self._display_csv(vocab_data)
            self._update_progress_badge(self._extraction_source)

        # Q&A tab - always enabled if Q&A system is ready
        main_window = self.winfo_toplevel()
        qa_ready = getattr(main_window, '_qa_ready', False)
        qa_data = self._outputs.get("Ask Questions") or self._outputs.get("Q&A Results")
        if qa_data:
            self._display_qa_results(qa_data)

        # Summary tab
        summary = self._outputs.get("Summary") or self._outputs.get("Meta-Summary")
        if summary:
            self.summary_text_display.delete("0.0", "end")
            self.summary_text_display.insert("0.0", summary)
        elif self._outputs.get("Case Briefing"):
            # Legacy Case Briefing support
            self._display_briefing(self._outputs.get("Case Briefing", ""))

        # Individual document summaries (if any) - append to summary tab
        if self._document_summaries:
            self.summary_text_display.insert("end", "\n\n" + "="*50 + "\n")
            self.summary_text_display.insert("end", "INDIVIDUAL DOCUMENT SUMMARIES\n")
            self.summary_text_display.insert("end", "="*50 + "\n\n")
            for doc_name, doc_summary in sorted(self._document_summaries.items()):
                self.summary_text_display.insert("end", f"{doc_name}:\n{doc_summary}\n\n")

        # Set default tab to Names & Vocab if it has data, otherwise Q&A if ready, otherwise Summary
        if vocab_data:
            self.tabview.set("Names & Vocab")
        elif qa_ready:
            self.tabview.set("Ask Questions")
        else:
            self.tabview.set("Summary")

    def _display_csv(self, data: list):
        """
        Displays vocabulary data in an Excel-like Treeview with frozen headers.

        Uses async batch insertion with pagination for GUI responsiveness.
        Initial load shows ROWS_PER_PAGE items, "Load More" button adds more.

        Args:
            data: List of dicts with keys: Term, Quality Score, Is Person, Found By
        """
        if not data:
            debug_log("[VOCAB DISPLAY] No vocabulary data to display")
            return

        # Reset pagination state
        self._vocab_display_offset = 0
        self._vocab_total_items = len(data)
        self._is_loading = False

        # Create frame to hold treeview and scrollbars
        if self.treeview_frame is None:
            self.treeview_frame = ctk.CTkFrame(self.tabview.tab("Names & Vocab"), **FRAME_STYLES["card"])

        self.treeview_frame.grid(row=0, column=0, sticky="nsew")
        self.treeview_frame.grid_columnconfigure(0, weight=1)
        self.treeview_frame.grid_rowconfigure(0, weight=1)

        # Session 47: Choose columns based on detail view toggle
        columns = GUI_DISPLAY_COLUMNS_EXTENDED if self._show_details else GUI_DISPLAY_COLUMNS
        self._current_columns = columns  # Store for use in _async_insert_rows

        # Create or reconfigure treeview
        if self.csv_treeview is None:
            self.csv_treeview = ttk.Treeview(
                self.treeview_frame,
                columns=columns,
                show="headings",
                style="Vocab.Treeview",
                selectmode="browse"
            )

            # Configure column headings and widths using COLUMN_CONFIG
            for col in columns:
                col_config = COLUMN_CONFIG.get(col, {"width": 100})
                self.csv_treeview.heading(col, text=col, anchor='w')
                self.csv_treeview.column(
                    col,
                    width=col_config["width"],
                    minwidth=60,
                    anchor='w',
                    stretch=True if col == "Term" else False  # Term stretches to fill space
                )

            # Add vertical scrollbar
            vsb = ttk.Scrollbar(
                self.treeview_frame,
                orient="vertical",
                command=self.csv_treeview.yview,
                style="Vocab.Vertical.TScrollbar"
            )
            self.csv_treeview.configure(yscrollcommand=vsb.set)

            # Add horizontal scrollbar
            hsb = ttk.Scrollbar(
                self.treeview_frame,
                orient="horizontal",
                command=self.csv_treeview.xview,
                style="Vocab.Horizontal.TScrollbar"
            )
            self.csv_treeview.configure(xscrollcommand=hsb.set)

            # Grid layout
            self.csv_treeview.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")

            # Bind right-click for context menu
            self.csv_treeview.bind("<Button-3>", self._on_right_click)
            self.csv_treeview.bind("<Double-1>", self._on_double_click)
            # Bind left-click for feedback columns (Session 25)
            self.csv_treeview.bind("<Button-1>", self._on_treeview_click)

            # Create context menu
            self._create_context_menu()

            # Configure all vocabulary table tags from centralized theme
            for tag_name, tag_config in VOCAB_TABLE_TAGS.items():
                self.csv_treeview.tag_configure(tag_name, **tag_config)

        # Clear existing data
        self.csv_treeview.delete(*self.csv_treeview.get_children())

        # Calculate how many items to load initially
        initial_load = min(ROWS_PER_PAGE, self._vocab_total_items)

        debug_log(f"[VOCAB DISPLAY] Showing {initial_load} of {self._vocab_total_items} terms "
                  f"(pagination: {ROWS_PER_PAGE} per page)")

        # Start async batch insertion for initial load
        self._async_insert_rows(data, 0, initial_load)

    def _display_qa_results(self, results: list):
        """
        Display Q&A results using the QAPanel widget.

        Args:
            results: List of QAResult objects
        """
        if not results:
            debug_log("[Q&A DISPLAY] No Q&A results to display")
            return

        # Set up follow-up callback if not already done
        # (must be done after MainWindow is fully initialized, not in __init__)
        if self._qa_panel.on_ask_followup is None:
            main_window = self.winfo_toplevel()
            if hasattr(main_window, '_ask_followup_for_qa_panel'):
                self._qa_panel.set_followup_callback(main_window._ask_followup_for_qa_panel)
                debug_log("[Q&A DISPLAY] Follow-up callback connected to MainWindow")

        # Display results
        self._qa_panel.display_results(results)

        debug_log(f"[Q&A DISPLAY] Showing {len(results)} Q&A results")

    def _display_briefing(self, briefing_text: str):
        """
        Display Case Briefing Sheet in the summary textbox.

        The briefing is formatted text with sections like:
        - Case Type
        - Parties Involved
        - Names to Know
        - What Happened (narrative)

        Args:
            briefing_text: Formatted briefing text from BriefingFormatter
        """
        if not briefing_text:
            self.summary_text_display.delete("0.0", "end")
            self.summary_text_display.insert(
                "0.0",
                "Case Briefing not yet generated.\n\n"
                "Case Briefing is generated automatically after document extraction "
                "if enabled in Settings > Q&A/Briefing > Auto-run."
            )
            return

        # Display in textbox
        self.summary_text_display.delete("0.0", "end")
        self.summary_text_display.insert("0.0", briefing_text)

        debug_log(f"[BRIEFING DISPLAY] Showing Case Briefing ({len(briefing_text)} chars)")

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

        current_idx = start_idx

        def insert_batch():
            nonlocal current_idx

            # If paused by window resize, reschedule and wait
            if self._batch_insertion_paused:
                self.after(BATCH_INSERT_DELAY_MS, insert_batch)
                return

            # Insert a batch of rows
            batch_end = min(current_idx + BATCH_INSERT_SIZE, end_idx)

            # Session 47: Use stored columns (may be extended with NER/RAKE/BM25)
            current_columns = getattr(self, '_current_columns', GUI_DISPLAY_COLUMNS)

            for i in range(current_idx, batch_end):
                item = data[i]
                rating = 0  # Default no rating
                if isinstance(item, dict):
                    # Apply text truncation to prevent row overflow
                    # Build values for each column, handling feedback columns specially
                    values = []
                    term = item.get("Term", "")
                    rating = self._feedback_manager.get_rating(term)

                    for col in current_columns:
                        if col == "Keep":
                            values.append(THUMB_UP_FILLED if rating == 1 else THUMB_UP_EMPTY)
                        elif col == "Skip":
                            values.append(THUMB_DOWN_FILLED if rating == -1 else THUMB_DOWN_EMPTY)
                        elif col == "Score":
                            # Map "Score" display column to "Quality Score" data field
                            score = item.get("Quality Score", "")
                            values.append(truncate_text(str(score), COLUMN_CONFIG[col]["max_chars"]))
                        else:
                            values.append(truncate_text(str(item.get(col, "")), COLUMN_CONFIG[col]["max_chars"]))

                    values = tuple(values)
                else:
                    # Handle list format (legacy) - apply truncation, default empty feedback
                    raw_values = tuple(item) if len(item) >= 4 else tuple(item) + ("",) * (4 - len(item))
                    values = tuple(
                        truncate_text(str(v), COLUMN_CONFIG[current_columns[j]]["max_chars"])
                        for j, v in enumerate(raw_values[:4])
                    ) + (THUMB_UP_EMPTY, THUMB_DOWN_EMPTY)

                # Apply tag for row coloring based on existing rating or Found By (Session 43)
                # Session 51: Add alternating row background color
                row_bg_tag = 'oddrow' if i % 2 else 'evenrow'

                if rating == 1:
                    tag = (row_bg_tag, 'rated_up')
                elif rating == -1:
                    tag = (row_bg_tag, 'rated_down')
                else:
                    # No feedback rating - use Found By coloring if available
                    # Session 53: Handle comma-separated algorithm names (e.g., "NER, RAKE")
                    # Session 61: Also handle legacy "Both" value
                    found_by = item.get("Found By", "") if isinstance(item, dict) else ""
                    algos = [a.strip() for a in found_by.split(",")] if found_by else []
                    if len(algos) >= 2 or "Both" in algos:
                        # Multiple algorithms = high confidence (includes legacy "Both")
                        tag = (row_bg_tag, 'found_multi')
                    elif "NER" in algos:
                        tag = (row_bg_tag, 'found_ner')
                    elif "RAKE" in algos:
                        tag = (row_bg_tag, 'found_rake')
                    elif "BM25" in algos:
                        tag = (row_bg_tag, 'found_bm25')
                    elif "LLM" in algos:
                        tag = (row_bg_tag, 'found_llm')
                    else:
                        # Fallback: use multi color for visibility (better than invisible)
                        tag = (row_bg_tag, 'found_multi')
                self.csv_treeview.insert("", "end", values=values, tags=tag)

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
        total_items = len(data)
        displayed_items = self._vocab_display_offset

        # Create or update "Load More" button
        if displayed_items < total_items:
            remaining = total_items - displayed_items

            if self._load_more_btn is None:
                self._load_more_btn = ctk.CTkButton(
                    self.treeview_frame,
                    text="",
                    command=lambda: self._load_more_rows(data),
                    height=28,
                    **BUTTON_STYLES["primary"]
                )

            self._load_more_btn.configure(
                text=f"Load More ({remaining} remaining)"
            )
            self._load_more_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

            # Update info label
            if not hasattr(self, 'vocab_info_label'):
                self.vocab_info_label = ctk.CTkLabel(
                    self.treeview_frame,
                    text="",
                    font=FONTS["small"],
                    text_color=COLORS["text_secondary"]
                )
            self.vocab_info_label.configure(
                text=f"Showing {displayed_items} of {total_items} terms • Full list available via 'Save to File'"
            )
            self.vocab_info_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 5))

        else:
            # All items displayed
            if self._load_more_btn is not None:
                self._load_more_btn.grid_remove()

            if hasattr(self, 'vocab_info_label'):
                self.vocab_info_label.configure(
                    text=f"Showing all {total_items} terms"
                )
                self.vocab_info_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        # Run garbage collection in background thread (non-blocking)
        threading.Thread(target=gc.collect, daemon=True).start()

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

        debug_log(f"[VOCAB DISPLAY] Loading more: rows {start_idx} to {end_idx}")

        # Start async insertion
        self._async_insert_rows(data, start_idx, end_idx)

    def _create_context_menu(self):
        """Create right-click context menu for vocabulary table."""
        self.context_menu = Menu(self, tearoff=0, bg="#404040", fg="white",
                                  activebackground="#505050", activeforeground="white",
                                  font=('Segoe UI', 10))
        self.context_menu.add_command(
            label="Exclude this term from future lists",
            command=self._exclude_selected_term
        )
        self.context_menu.add_separator()
        self.context_menu.add_command(
            label="Copy term",
            command=self._copy_selected_term
        )

    def _on_right_click(self, event):
        """Handle right-click on treeview to show context menu."""
        # Identify the row under cursor
        item_id = self.csv_treeview.identify_row(event.y)
        if item_id:
            # Select the row
            self.csv_treeview.selection_set(item_id)
            # Get the term value (first column)
            values = self.csv_treeview.item(item_id, 'values')
            if values:
                self._selected_term = values[0]  # Term is first column
                # Show context menu at cursor position
                try:
                    self.context_menu.tk_popup(event.x_root, event.y_root)
                finally:
                    self.context_menu.grab_release()

    def _on_double_click(self, event):
        """Handle double-click to copy the term."""
        item_id = self.csv_treeview.identify_row(event.y)
        if item_id:
            values = self.csv_treeview.item(item_id, 'values')
            if values and len(values) >= 1:
                term = values[0]  # Term is first column
                if term:
                    self.clipboard_clear()
                    self.clipboard_append(term)

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
            icon="question"
        )

        if not result:
            return

        # Add to exclusion file
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(USER_VOCAB_EXCLUDE_PATH), exist_ok=True)

            # Append to file
            with open(USER_VOCAB_EXCLUDE_PATH, 'a', encoding='utf-8') as f:
                f.write(f"{lower_term}\n")

            debug_log(f"[VOCAB UI] Added '{term}' to user exclusion list at {USER_VOCAB_EXCLUDE_PATH}")

            # Remove from current display
            selected = self.csv_treeview.selection()
            if selected:
                self.csv_treeview.delete(selected[0])

                # Also remove from internal data
                self._outputs["Rare Word List (CSV)"] = [
                    item for item in self._outputs.get("Rare Word List (CSV)", [])
                    if isinstance(item, dict) and item.get("Term", "").lower() != lower_term
                ]

            messagebox.showinfo(
                "Term Excluded",
                f"'{term}' will not appear in future rare word lists.\n\n"
                "Note: This takes effect on the next vocabulary extraction."
            )

        except Exception as e:
            debug_log(f"[VOCAB UI] Failed to save exclusion: {e}")
            messagebox.showerror(
                "Error",
                f"Failed to save exclusion: {e}\n\n"
                "Please check file permissions."
            )

    def _copy_selected_term(self):
        """Copy the selected term to clipboard."""
        if self._selected_term:
            self.clipboard_clear()
            self.clipboard_append(self._selected_term)

    def _build_vocab_csv(self, vocab_data: list) -> str:
        """
        Build CSV string from vocabulary data.

        Respects the vocab_export_format user preference:
        - "all": All columns including Quality Score, In-Case Freq, Freq Rank
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
            columns = ["Term"]
        else:  # "basic" (default)
            columns = list(GUI_DISPLAY_COLUMNS)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(columns)

        for item in vocab_data:
            if isinstance(item, dict):
                row = []
                for col in columns:
                    # Map "Score" display column to "Quality Score" data field
                    if col == "Score":
                        row.append(item.get("Quality Score", ""))
                    else:
                        row.append(item.get(col, ""))
                writer.writerow(row)
            else:
                # Legacy list format
                writer.writerow(item[:len(columns)])

        return output.getvalue()

    def get_current_content_for_export(self):
        """
        Returns the currently displayed content for copy/save operations.

        For vocabulary CSV, respects the vocab_export_format setting:
        - "all": All columns including Quality Score, In-Case Freq, Freq Rank
        - "basic": Term, Score, Is Person, Found By
        - "terms_only": Just the Term column
        """
        current_tab = self.tabview.get()

        if current_tab == "Names & Vocab":
            data = self._outputs.get("Names & Vocabulary") or self._outputs.get("Rare Word List (CSV)", [])
            return self._build_vocab_csv(data)
        elif current_tab == "Ask Questions":
            # Get export content from QAPanel if available
            if self._qa_panel is not None:
                return self._qa_panel.get_export_content()
            return ""
        elif current_tab == "Summary":
            # Return text from summary display
            return self.summary_text_display.get("0.0", "end").strip()
        return ""

    def copy_to_clipboard(self):
        """Copy currently displayed content to clipboard."""
        content = self.get_current_content_for_export()
        if content:
            self.clipboard_clear()
            self.clipboard_append(content)
            # Brief button flash instead of modal dialog
            original_text = self.copy_btn.cget("text")
            self.copy_btn.configure(text="Copied!")
            self.after(1500, lambda: self.copy_btn.configure(text=original_text))
        else:
            messagebox.showwarning("Empty", "No content to copy.")

    def save_to_file(self):
        """Save currently displayed content to file."""
        content = self.get_current_content_for_export()
        if not content:
            messagebox.showwarning("Empty", "No content to save.")
            return

        # Session 51: Use current tab instead of dropdown
        current_tab = self.tabview.get()
        default_filename = "output"
        filetypes = [("All Files", "*.*")]

        if current_tab == "Names & Vocab":
            default_filename = "names_vocabulary.csv"
            filetypes = [("CSV Files", "*.csv"), ("All Files", "*.*")]
        elif current_tab == "Ask Questions":
            default_filename = "qa_results.txt"
            filetypes = [("Text Files", "*.txt"), ("All Files", "*.*")]
        elif current_tab == "Summary":
            default_filename = "summary.txt"
            filetypes = [("Text Files", "*.txt"), ("All Files", "*.*")]

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=filetypes,
            initialfile=default_filename,
            title="Save Output"
        )
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Saved", f"Output saved to {filepath}")

    def _quick_export_vocab_csv(self):
        """
        Quick export vocabulary to CSV file (Session 65).

        Exports to Documents folder with timestamped filename.
        Shows success message with file path.
        """
        from datetime import datetime

        # Check if we have vocabulary data
        vocab_data = self._outputs.get("Names & Vocabulary") or self._outputs.get("Rare Word List (CSV)", [])
        if not vocab_data:
            messagebox.showwarning("No Data", "No vocabulary data to export.\n\nProcess documents first.")
            return

        # Generate CSV content using shared helper
        csv_content = self._build_vocab_csv(vocab_data)

        # Get default export directory (Documents folder)
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                               r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders") as key:
                documents_path = winreg.QueryValueEx(key, "Personal")[0]
        except Exception:
            # Fallback to home directory
            from pathlib import Path
            documents_path = str(Path.home() / "Documents")

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vocabulary_{timestamp}.csv"
        filepath = os.path.join(documents_path, filename)

        # Save the file
        try:
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write(csv_content)

            debug_log(f"[VOCAB EXPORT] Saved {len(vocab_data)} terms to {filepath}")

            # Show success with option to open folder
            result = messagebox.askyesno(
                "Export Successful",
                f"Vocabulary exported to:\n{filepath}\n\n"
                f"({len(vocab_data)} terms)\n\n"
                "Open containing folder?",
                icon="info"
            )

            if result:
                # Open the folder in Windows Explorer
                os.startfile(documents_path)

        except Exception as e:
            debug_log(f"[VOCAB EXPORT] Failed: {e}")
            messagebox.showerror("Export Failed", f"Could not save file:\n{e}")

    def _on_treeview_click(self, event):
        """
        Handle left-click on treeview for feedback columns.

        Detects clicks on the Keep or Skip columns and toggles the
        feedback state for that term.

        Session 47: Column indices are dynamic based on detail view toggle.
        """
        # Identify which column and row was clicked
        column = self.csv_treeview.identify_column(event.x)
        item_id = self.csv_treeview.identify_row(event.y)

        if not item_id:
            return

        # Session 47: Dynamically find Keep and Skip column indices
        current_columns = getattr(self, '_current_columns', GUI_DISPLAY_COLUMNS)
        try:
            keep_idx = current_columns.index("Keep") + 1  # 1-based
            skip_idx = current_columns.index("Skip") + 1  # 1-based
        except ValueError:
            return  # Keep/Skip columns not found

        # Check if click was on a feedback column
        if column == f"#{keep_idx}":  # Keep column
            self._toggle_feedback(item_id, +1)
        elif column == f"#{skip_idx}":  # Skip column
            self._toggle_feedback(item_id, -1)

    def _check_corpus_and_warn(self) -> bool:
        """
        Check corpus status and warn user if not ready (Session 68).

        Shows a warning dialog once per session if corpus has < 5 documents.
        User can choose to continue or cancel feedback.

        Returns:
            True to proceed with feedback, False to cancel
        """
        # Already warned this session, proceed silently
        if self._corpus_warning_shown:
            return True

        from src.core.vocabulary.corpus_manager import get_corpus_manager

        corpus_manager = get_corpus_manager()
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
            icon="warning"
        )
        return result

    def _toggle_feedback(self, item_id: str, feedback_type: int):
        """
        Toggle feedback state for a vocabulary term.

        If the term already has this feedback, clear it.
        If the term has opposite or no feedback, set the new feedback.

        Args:
            item_id: Treeview item identifier
            feedback_type: +1 for Keep, -1 for Skip
        """
        # Session 68: Warn about missing corpus (once per session)
        if not self._check_corpus_and_warn():
            return  # User cancelled

        # Get the term from the row
        values = self.csv_treeview.item(item_id, 'values')
        if not values:
            return

        term = values[0]  # Term is first column
        current_rating = self._feedback_manager.get_rating(term)

        # Toggle logic: if already this rating, clear it; otherwise set it
        if current_rating == feedback_type:
            new_rating = 0  # Clear the rating
        else:
            new_rating = feedback_type

        # Find full term data from internal storage for ML features
        term_data = self._find_term_data(term)
        if not term_data:
            term_data = {"Term": term}

        # Record feedback (handles both setting and clearing)
        success = self._feedback_manager.record_feedback(term_data, new_rating)

        if success:
            # Update the visual display
            self._update_feedback_display(item_id, new_rating)
            debug_log(f"[FEEDBACK UI] {'Cleared' if new_rating == 0 else 'Set'} "
                      f"feedback for '{term}': {new_rating}")

    def _find_term_data(self, term: str) -> dict | None:
        """
        Find full term data from internal storage by term name.

        Args:
            term: The term to search for (case-insensitive)

        Returns:
            Dictionary with term data, or None if not found
        """
        # Check primary key first (Session 45), then legacy key
        vocab_data = self._outputs.get("Names & Vocabulary", [])
        if not vocab_data:
            vocab_data = self._outputs.get("Rare Word List (CSV)", [])

        lower_term = term.lower().strip()

        for item in vocab_data:
            if isinstance(item, dict):
                if item.get("Term", "").lower().strip() == lower_term:
                    return item

        return None

    def _update_feedback_display(self, item_id: str, rating: int):
        """
        Update the visual display of feedback icons for a term.

        Args:
            item_id: Treeview item identifier
            rating: +1 (Keep filled), -1 (Skip filled), 0 (both empty)
        """
        values = list(self.csv_treeview.item(item_id, 'values'))

        # Session 47: Dynamically find Keep and Skip column indices
        current_columns = getattr(self, '_current_columns', GUI_DISPLAY_COLUMNS)
        try:
            keep_idx = current_columns.index("Keep")  # 0-based for list access
            skip_idx = current_columns.index("Skip")  # 0-based for list access
        except ValueError:
            return  # Keep/Skip columns not found

        if len(values) <= max(keep_idx, skip_idx):
            return

        # Update the icon values at dynamic positions
        if rating == 1:
            values[keep_idx] = THUMB_UP_FILLED
            values[skip_idx] = THUMB_DOWN_EMPTY
            tag = ('rated_up',)
        elif rating == -1:
            values[keep_idx] = THUMB_UP_EMPTY
            values[skip_idx] = THUMB_DOWN_FILLED
            tag = ('rated_down',)
        else:  # rating == 0
            values[keep_idx] = THUMB_UP_EMPTY
            values[skip_idx] = THUMB_DOWN_EMPTY
            tag = ()

        # Update the item with new values and tag for coloring
        self.csv_treeview.item(item_id, values=tuple(values), tags=tag)
