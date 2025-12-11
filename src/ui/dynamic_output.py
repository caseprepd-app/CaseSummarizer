"""
Dynamic Output Display Widget for LocalScribe (Session 45 Update)

Displays Names & Vocabulary tables, Q&A results, and optional summaries.
Provides copy/save functionality for export.
The vocabulary display uses an Excel-like Treeview with frozen headers
and right-click context menu for excluding terms from future extractions.

Session 45 Updates:
- Renamed "Case Briefing" to "Names & Vocabulary" as primary output
- Simplified dropdown: "Names & Vocabulary" | "Q&A" | "Summary"
- Added progress badge showing data source (NER only → NER + LLM)
- Output pane has distinct background color

Performance Optimizations (Session 14):
- Text truncation prevents row height overflow and improves rendering speed
- Batch insertion with reduced batch size for responsiveness
- Garbage collection after large operations
- Deferred Treeview creation until first use

Performance Optimizations (Session 16):
- Asynchronous batch insertion with after() to yield to event loop
- Background garbage collection to avoid blocking main thread
- Pagination with "Load More" button for large datasets
- Optimized single update_idletasks() call after each batch
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
from src.vocabulary.feedback_manager import get_feedback_manager

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
COLUMN_CONFIG = {
    "Term": {"width": 150, "max_chars": 25},
    "Type": {"width": 80, "max_chars": 12},
    "Found By": {"width": 60, "max_chars": 6},  # Both, NER, or LLM
    "Role/Relevance": {"width": 120, "max_chars": 18},
    "Quality Score": {"width": 85, "max_chars": 8},
    "In-Case Freq": {"width": 80, "max_chars": 8},
    "Freq Rank": {"width": 80, "max_chars": 10},
    "Definition": {"width": 240, "max_chars": 38},  # Slightly narrower to fit Found By
    "Keep": {"width": 45, "max_chars": 3},
    "Skip": {"width": 45, "max_chars": 3},
}

# Columns visible in the GUI Treeview (confidence columns hidden for cleaner display)
# Session 23: Users can export all columns via CSV, but GUI shows only essential info
# Session 25: Added feedback columns for ML learning
# Session 43: Added Found By column - terms found by Both (NER+LLM) appear first
GUI_DISPLAY_COLUMNS = ("Term", "Type", "Found By", "Role/Relevance", "Definition", "Keep", "Skip")

# All columns available for export (includes confidence/filtering columns)
# Session 43: Added Found By for export
ALL_EXPORT_COLUMNS = ("Term", "Type", "Found By", "Role/Relevance", "Quality Score", "In-Case Freq", "Freq Rank", "Definition")


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

    # Class-level flag to track if style has been configured (Session 45)
    _style_configured = False

    def __init__(self, master, **kwargs):
        # Session 45: Set distinct background color for output pane
        # Slightly darker/different than other panes to distinguish
        kwargs.setdefault('fg_color', ('#e8e8e8', '#1a1a2e'))  # Light/dark mode colors
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # Row for the dynamic content frame (moved to row 2)

        # Header row with dropdown and progress badge
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        self.header_frame.grid_columnconfigure(1, weight=1)

        # Output Selection Dropdown
        self.output_selector_label = ctk.CTkLabel(
            self.header_frame, text="View Output:",
            font=ctk.CTkFont(weight="bold")
        )
        self.output_selector_label.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.output_selector = ctk.CTkComboBox(
            self.header_frame,
            values=["No outputs yet"],
            command=self._on_output_selection,
            width=200
        )
        self.output_selector.grid(row=0, column=1, sticky="w")
        self.output_selector.set("No outputs yet")

        # Progress Badge (Session 45) - shows data source status
        self._progress_badge = ctk.CTkLabel(
            self.header_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray70")
        )
        self._progress_badge.grid(row=0, column=2, sticky="e", padx=(10, 0))

        # Dynamic Content Frame (to hold either Textbox or Treeview)
        self.dynamic_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.dynamic_content_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.grid_rowconfigure(1, weight=1)  # Make content frame expandable
        self.dynamic_content_frame.grid_columnconfigure(0, weight=1)
        self.dynamic_content_frame.grid_rowconfigure(0, weight=1)

        # Textbox for summaries
        self.summary_text_display = ctk.CTkTextbox(self.dynamic_content_frame, wrap="word")
        self.summary_text_display.grid(row=0, column=0, sticky="nsew")
        self.summary_text_display.insert("0.0", "Generated summaries and rare word lists will appear here. Select an option from the dropdown above.")

        # Treeview for CSV (initially None, created when needed)
        self.csv_treeview = None
        self.treeview_frame = None  # Frame to hold treeview and scrollbars

        # Right-click context menu for vocabulary exclusion
        self.context_menu = None
        self._selected_term = None

        # Button bar
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.copy_btn = ctk.CTkButton(self.button_frame, text="Copy to Clipboard", command=self.copy_to_clipboard)
        self.copy_btn.pack(side="left", padx=5)

        self.save_btn = ctk.CTkButton(self.button_frame, text="Save to File...", command=self.save_to_file)
        self.save_btn.pack(side="left", padx=5)

        # Internal storage for outputs (Session 45: Updated naming)
        self._outputs = {
            "Names & Vocabulary": [],  # Session 45: Primary output - people + terms
            "Q&A": [],  # Q&A results (replaces "Q&A Results")
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

        # Q&A panel (created on first use)
        self._qa_panel = None

        # Pagination state for vocabulary display
        self._vocab_display_offset = 0  # Current offset into vocabulary data
        self._vocab_total_items = 0  # Total items in vocabulary data
        self._load_more_btn = None  # "Load More" button reference
        self._is_loading = False  # Prevents duplicate load operations

        # Feedback manager for ML learning (Session 25)
        self._feedback_manager = get_feedback_manager()

    def _on_output_selection(self, choice):
        """Handle selection change in the output_selector dropdown (Session 45)."""
        self._clear_dynamic_content()

        if choice == "No outputs yet":
            self.summary_text_display.grid(row=0, column=0, sticky="nsew")
            self.summary_text_display.delete("0.0", "end")
            self.summary_text_display.insert(
                "0.0",
                "Names, vocabulary, and Q&A results will appear here.\n"
                "Select documents and click 'Process' to begin."
            )
            self._update_progress_badge("none")
        elif choice.startswith("Names & Vocabulary"):
            # Session 45: Primary output - show combined people + vocabulary table
            vocab_data = self._outputs.get("Names & Vocabulary", [])
            if not vocab_data:
                # Fall back to legacy format
                vocab_data = self._outputs.get("Rare Word List (CSV)", [])
            self._display_csv(vocab_data)
            self._update_progress_badge(self._extraction_source)
        elif choice.startswith("Q&A"):
            # Session 45: Show Q&A panel
            qa_data = self._outputs.get("Q&A", [])
            if not qa_data:
                qa_data = self._outputs.get("Q&A Results", [])
            self._display_qa_results(qa_data)
            self._update_progress_badge("none")  # Q&A doesn't have NER/LLM source
        elif choice.startswith("Summary"):
            # Session 45: Combined summary display
            self.summary_text_display.grid(row=0, column=0, sticky="nsew")
            self.summary_text_display.delete("0.0", "end")
            summary = self._outputs.get("Summary", "") or self._outputs.get("Meta-Summary", "")
            self.summary_text_display.insert(
                "0.0",
                summary if summary else "Summary not yet generated."
            )
            self._update_progress_badge("none")
        # Legacy support for old dropdown options
        elif choice == "Meta-Summary":
            self.summary_text_display.grid(row=0, column=0, sticky="nsew")
            self.summary_text_display.delete("0.0", "end")
            self.summary_text_display.insert("0.0", self._outputs.get("Meta-Summary", "Meta-Summary not yet generated."))
        elif choice.startswith("Case Briefing"):
            self._display_briefing(self._outputs.get("Case Briefing", ""))
        elif choice.startswith("Rare Word List"):
            self._display_csv(self._outputs.get("Rare Word List (CSV)", []))
        elif choice.startswith("Q&A Results"):
            self._display_qa_results(self._outputs.get("Q&A Results", []))
        elif choice.startswith("Summary for "):
            doc_name = choice.replace("Summary for ", "")
            self.summary_text_display.grid(row=0, column=0, sticky="nsew")
            self.summary_text_display.delete("0.0", "end")
            self.summary_text_display.insert("0.0", self._document_summaries.get(doc_name, f"Summary for {doc_name} not yet generated."))

    def _clear_dynamic_content(self):
        """Clears the currently displayed widget in the dynamic content frame."""
        for widget in self.dynamic_content_frame.winfo_children():
            widget.grid_remove()

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
        # Update badge if Names & Vocabulary is currently displayed
        current = self.output_selector.get()
        if current.startswith("Names & Vocabulary"):
            self._update_progress_badge(source)

    def cleanup(self):
        """
        Clean up resources when widget is no longer needed.
        Call this to free memory after heavy processing.
        """
        # Clear internal data storage
        self._outputs = {
            "Meta-Summary": "",
            "Rare Word List (CSV)": [],
            "Q&A Results": [],
            "Case Briefing": "",
        }
        self._document_summaries = {}
        self._briefing_sections = {}

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
            self._outputs["Q&A"] = qa_results
        if briefing_text:
            self._outputs["Case Briefing"] = briefing_text
        if briefing_sections is not None:
            self._briefing_sections = briefing_sections

        self._refresh_dropdown()

    def _refresh_dropdown(self):
        """Refreshes the output selection dropdown based on available outputs (Session 45)."""
        options = []

        # Session 45: Primary output - Names & Vocabulary
        vocab_data = self._outputs.get("Names & Vocabulary")
        if vocab_data is None:
            # Fall back to legacy key
            vocab_data = self._outputs.get("Rare Word List (CSV)")

        if vocab_data is not None:
            vocab_count = len(vocab_data)
            options.append(f"Names & Vocabulary ({vocab_count} items)")

        # Session 45: Q&A is shown when available
        qa_data = self._outputs.get("Q&A")
        if qa_data is None:
            qa_data = self._outputs.get("Q&A Results")

        if qa_data is not None:
            qa_count = len(qa_data)
            options.append(f"Q&A ({qa_count} results)")

        # Session 45: Summary (combined or individual)
        summary = self._outputs.get("Summary") or self._outputs.get("Meta-Summary")
        if summary:
            options.append("Summary")

        # Individual document summaries (if any)
        doc_summary_options = [f"Summary for {name}" for name in self._document_summaries.keys()]
        if doc_summary_options:
            doc_summary_options.sort()
            options.extend(doc_summary_options)

        # Legacy support: Case Briefing (if still present)
        if self._outputs.get("Case Briefing"):
            options.append("Case Briefing (Legacy)")

        # Only include placeholder if no real outputs exist
        if not options:
            options = ["No outputs yet"]

        self.output_selector.configure(values=options)
        if options and options[0] != "No outputs yet":
            self.output_selector.set(options[0])
            self._on_output_selection(options[0])
        else:
            self.output_selector.set("No outputs yet")
            self._on_output_selection("No outputs yet")

    def _create_treeview_style(self):
        """Create and configure the Treeview style (Session 45: only once to avoid UI lockup)."""
        # Only configure style once to prevent UI lockup from repeated theme_use calls
        if DynamicOutputWidget._style_configured:
            return

        style = ttk.Style()
        # Note: theme_use("default") can cause layout thrashing, only do once
        style.theme_use("default")

        # Main treeview styling - dark theme
        # Row height 25px is tight but prevents text wrapping issues
        # Text truncation handles overflow (see truncate_text function)
        style.configure(
            "Vocab.Treeview",
            background="#2b2b2b",
            foreground="white",
            fieldbackground="#2b2b2b",
            borderwidth=0,
            rowheight=25,
            font=('Segoe UI', 10)
        )
        style.map('Vocab.Treeview', background=[('selected', '#3470b6')])

        # Header styling - slightly lighter, bold
        style.configure(
            "Vocab.Treeview.Heading",
            background="#404040",
            foreground="white",
            relief="flat",
            font=('Segoe UI', 10, 'bold'),
            padding=(8, 4)
        )
        style.map("Vocab.Treeview.Heading", background=[('active', '#505050')])

        # Scrollbar styling
        style.configure(
            "Vocab.Vertical.TScrollbar",
            background="#404040",
            troughcolor="#2b2b2b",
            borderwidth=0,
            arrowcolor="white"
        )
        style.configure(
            "Vocab.Horizontal.TScrollbar",
            background="#404040",
            troughcolor="#2b2b2b",
            borderwidth=0,
            arrowcolor="white"
        )

        DynamicOutputWidget._style_configured = True
        debug_log("[DynamicOutput] Treeview style configured")

    def _display_csv(self, data: list):
        """
        Displays vocabulary data in an Excel-like Treeview with frozen headers.

        Uses async batch insertion with pagination for GUI responsiveness.
        Initial load shows ROWS_PER_PAGE items, "Load More" button adds more.

        Args:
            data: List of dicts with keys: Term, Type, Role/Relevance, Definition
        """
        self._clear_dynamic_content()

        if not data:
            self.summary_text_display.grid(row=0, column=0, sticky="nsew")
            self.summary_text_display.delete("0.0", "end")
            self.summary_text_display.insert("0.0", "Rare Word List (CSV) not yet generated or is empty.")
            return

        # Reset pagination state
        self._vocab_display_offset = 0
        self._vocab_total_items = len(data)
        self._is_loading = False

        # Create style if not already done
        self._create_treeview_style()

        # Create frame to hold treeview and scrollbars
        if self.treeview_frame is None:
            self.treeview_frame = ctk.CTkFrame(self.dynamic_content_frame, fg_color="#2b2b2b", corner_radius=6)

        self.treeview_frame.grid(row=0, column=0, sticky="nsew")
        self.treeview_frame.grid_columnconfigure(0, weight=1)
        self.treeview_frame.grid_rowconfigure(0, weight=1)

        # Define columns for GUI display (confidence columns hidden)
        columns = GUI_DISPLAY_COLUMNS

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
                    stretch=True if col == "Definition" else False
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

            # Configure feedback icon tags for coloring
            self.csv_treeview.tag_configure('rated_up', foreground='#28a745')  # Green
            self.csv_treeview.tag_configure('rated_down', foreground='#dc3545')  # Red

            # Configure Found By tags for visual distinction (Session 43)
            self.csv_treeview.tag_configure('found_both', foreground='#28a745')  # Green - highest confidence
            self.csv_treeview.tag_configure('found_ner', foreground='#6c757d')   # Gray - NER only
            self.csv_treeview.tag_configure('found_llm', foreground='#17a2b8')   # Blue - LLM only

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
        self._clear_dynamic_content()

        if not results:
            self.summary_text_display.grid(row=0, column=0, sticky="nsew")
            self.summary_text_display.delete("0.0", "end")
            self.summary_text_display.insert(
                "0.0",
                "Q&A is ready to use.\n\n"
                "Click 'Ask More Questions' below to ask questions about your documents.\n\n"
                "The Q&A system uses semantic search to find relevant passages "
                "and answer your questions."
            )
            return

        # Create QAPanel on first use
        if self._qa_panel is None:
            from src.ui.qa_panel import QAPanel
            self._qa_panel = QAPanel(self.dynamic_content_frame)

            # Set up follow-up callback by finding MainWindow through the widget tree
            # DynamicOutputWidget's master is right_panel, not MainWindow directly
            # Use winfo_toplevel() to get the root window (MainWindow)
            main_window = self.winfo_toplevel()
            if hasattr(main_window, '_ask_followup_for_qa_panel'):
                self._qa_panel.set_followup_callback(main_window._ask_followup_for_qa_panel)
                debug_log("[Q&A DISPLAY] Follow-up callback connected to MainWindow")

        # Display results
        self._qa_panel.display_results(results)
        self._qa_panel.grid(row=0, column=0, sticky="nsew")

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
        self._clear_dynamic_content()

        if not briefing_text:
            self.summary_text_display.grid(row=0, column=0, sticky="nsew")
            self.summary_text_display.delete("0.0", "end")
            self.summary_text_display.insert(
                "0.0",
                "Case Briefing not yet generated.\n\n"
                "Case Briefing is generated automatically after document extraction "
                "if enabled in Settings > Q&A/Briefing > Auto-run."
            )
            return

        # Display in textbox
        self.summary_text_display.grid(row=0, column=0, sticky="nsew")
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

            # Insert a batch of rows
            batch_end = min(current_idx + BATCH_INSERT_SIZE, end_idx)

            for i in range(current_idx, batch_end):
                item = data[i]
                rating = 0  # Default no rating
                if isinstance(item, dict):
                    # Apply text truncation to prevent row overflow
                    # Build values for each column, handling feedback columns specially
                    values = []
                    term = item.get("Term", "")
                    rating = self._feedback_manager.get_rating(term)

                    for col in GUI_DISPLAY_COLUMNS:
                        if col == "Keep":
                            values.append(THUMB_UP_FILLED if rating == 1 else THUMB_UP_EMPTY)
                        elif col == "Skip":
                            values.append(THUMB_DOWN_FILLED if rating == -1 else THUMB_DOWN_EMPTY)
                        else:
                            values.append(truncate_text(item.get(col, ""), COLUMN_CONFIG[col]["max_chars"]))

                    values = tuple(values)
                else:
                    # Handle list format (legacy) - apply truncation, default empty feedback
                    raw_values = tuple(item) if len(item) >= 4 else tuple(item) + ("",) * (4 - len(item))
                    values = tuple(
                        truncate_text(str(v), COLUMN_CONFIG[GUI_DISPLAY_COLUMNS[j]]["max_chars"])
                        for j, v in enumerate(raw_values[:4])
                    ) + (THUMB_UP_EMPTY, THUMB_DOWN_EMPTY)

                # Apply tag for row coloring based on existing rating or Found By (Session 43)
                if rating == 1:
                    tag = ('rated_up',)
                elif rating == -1:
                    tag = ('rated_down',)
                else:
                    # No feedback rating - use Found By coloring if available
                    found_by = item.get("Found By", "") if isinstance(item, dict) else ""
                    if found_by == "Both":
                        tag = ('found_both',)
                    elif found_by == "NER":
                        tag = ('found_ner',)
                    elif found_by == "LLM":
                        tag = ('found_llm',)
                    else:
                        tag = ()
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
                    fg_color="#2d5a87",
                    hover_color="#3d6a97",
                    height=28
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
                    font=ctk.CTkFont(size=11),
                    text_color="#aaaaaa"
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
        """Handle double-click to copy the full definition."""
        item_id = self.csv_treeview.identify_row(event.y)
        if item_id:
            values = self.csv_treeview.item(item_id, 'values')
            if values and len(values) >= 4:
                definition = values[3]  # Definition is fourth column
                if definition and definition != "N/A":
                    self.clipboard_clear()
                    self.clipboard_append(definition)
                    # Brief visual feedback could be added here

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

    def get_current_content_for_export(self):
        """
        Returns the currently displayed content for copy/save operations.

        For vocabulary CSV, respects the vocab_export_format setting:
        - "all": All columns including Quality Score, In-Case Freq, Freq Rank
        - "basic": Term, Type, Role/Relevance, Definition
        - "terms_only": Just the Term column
        """
        current_choice = self.output_selector.get()
        if current_choice == "Meta-Summary":
            return self._outputs.get("Meta-Summary", "")
        elif current_choice.startswith("Case Briefing"):
            return self._outputs.get("Case Briefing", "")
        elif current_choice.startswith("Rare Word List"):
            # Convert list of dicts to CSV string
            data = self._outputs.get("Rare Word List (CSV)", [])
            if not data:
                return ""

            # Get export format from user preferences
            prefs = get_user_preferences()
            export_format = prefs.get("vocab_export_format", "basic")

            # Determine which columns to export based on setting
            if export_format == "all":
                columns = list(ALL_EXPORT_COLUMNS)
            elif export_format == "terms_only":
                columns = ["Term"]
            else:  # "basic" (default)
                columns = list(GUI_DISPLAY_COLUMNS)

            output = io.StringIO()
            writer = csv.writer(output)
            # Write header
            writer.writerow(columns)
            # Write data
            for item in data:
                if isinstance(item, dict):
                    writer.writerow([item.get(col, "") for col in columns])
                else:
                    # Legacy list format - map by position
                    writer.writerow(item[:len(columns)])
            return output.getvalue()
        elif current_choice.startswith("Q&A Results"):
            # Get export content from QAPanel if available
            if self._qa_panel is not None:
                return self._qa_panel.get_export_content()
            return ""
        elif current_choice.startswith("Summary for "):
            doc_name = current_choice.replace("Summary for ", "")
            return self._document_summaries.get(doc_name, "")
        return ""

    def copy_to_clipboard(self):
        """Copy currently displayed content to clipboard."""
        content = self.get_current_content_for_export()
        if content:
            self.clipboard_clear()
            self.clipboard_append(content)
            messagebox.showinfo("Copied", "Content copied to clipboard.")
        else:
            messagebox.showwarning("Empty", "No content to copy.")

    def save_to_file(self):
        """Save currently displayed content to file."""
        content = self.get_current_content_for_export()
        if not content:
            messagebox.showwarning("Empty", "No content to save.")
            return

        current_choice = self.output_selector.get()
        default_filename = "output"
        filetypes = [("All Files", "*.*")]

        if current_choice == "Meta-Summary":
            default_filename = "meta_summary.txt"
            filetypes = [("Text Files", "*.txt"), ("All Files", "*.*")]
        elif current_choice.startswith("Case Briefing"):
            default_filename = "case_briefing.txt"
            filetypes = [("Text Files", "*.txt"), ("All Files", "*.*")]
        elif current_choice.startswith("Rare Word List"):
            default_filename = "rare_word_list.csv"
            filetypes = [("CSV Files", "*.csv"), ("All Files", "*.*")]
        elif current_choice.startswith("Q&A Results"):
            default_filename = "qa_results.txt"
            filetypes = [("Text Files", "*.txt"), ("All Files", "*.*")]
        elif current_choice.startswith("Summary for "):
            doc_name = current_choice.replace("Summary for ", "")
            default_filename = f"{doc_name}_summary.txt"
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

    def _on_treeview_click(self, event):
        """
        Handle left-click on treeview for feedback columns.

        Detects clicks on the Keep or Skip columns and toggles the
        feedback state for that term.

        Column indices (1-based in identify_column):
        - #1: Term, #2: Type, #3: Role/Relevance, #4: Definition
        - #5: Keep, #6: Skip
        """
        # Identify which column and row was clicked
        column = self.csv_treeview.identify_column(event.x)
        item_id = self.csv_treeview.identify_row(event.y)

        if not item_id:
            return

        # Check if click was on a feedback column
        if column == "#5":  # Keep column
            self._toggle_feedback(item_id, +1)
        elif column == "#6":  # Skip column
            self._toggle_feedback(item_id, -1)

    def _toggle_feedback(self, item_id: str, feedback_type: int):
        """
        Toggle feedback state for a vocabulary term.

        If the term already has this feedback, clear it.
        If the term has opposite or no feedback, set the new feedback.

        Args:
            item_id: Treeview item identifier
            feedback_type: +1 for Keep, -1 for Skip
        """
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
        if len(values) < 6:
            return

        # Update the icon values (columns 5 and 6, indices 4 and 5)
        if rating == 1:
            values[4] = THUMB_UP_FILLED
            values[5] = THUMB_DOWN_EMPTY
            tag = ('rated_up',)
        elif rating == -1:
            values[4] = THUMB_UP_EMPTY
            values[5] = THUMB_DOWN_FILLED
            tag = ('rated_down',)
        else:  # rating == 0
            values[4] = THUMB_UP_EMPTY
            values[5] = THUMB_DOWN_EMPTY
            tag = ()

        # Update the item with new values and tag for coloring
        self.csv_treeview.item(item_id, values=tuple(values), tags=tag)
