"""
Dynamic Output Display Widget for CasePrepd (Session 51 Update)

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
import re
from tkinter import Menu, filedialog, messagebox, ttk

import customtkinter as ctk

from src.config import SORT_WARNING_COLUMNS, USER_VOCAB_EXCLUDE_PATH
from src.logging_config import debug_log
from src.ui.qa_panel import QAPanel
from src.ui.theme import BUTTON_STYLES, COLORS, FONTS, FRAME_STYLES, VOCAB_TABLE_TAGS

# Session 82: Import column configuration from centralized module
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
    THUMB_DOWN_EMPTY,
    THUMB_DOWN_FILLED,
    THUMB_UP_EMPTY,
    THUMB_UP_FILLED,
    truncate_text,
)
from src.user_preferences import get_user_preferences


class DynamicOutputWidget(ctk.CTkFrame):
    """Widget to dynamically display Names & Vocabulary, Q&A, or Summary outputs (Session 45)."""

    def __init__(self, master, **kwargs):
        # Session 45: Set distinct background color for output pane
        # Slightly darker/different than other panes to distinguish
        kwargs.setdefault("fg_color", ("#e8e8e8", "#1a1a2e"))  # Light/dark mode colors
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
            text_color=("gray50", "gray70"),
        )
        self._progress_badge.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))

        # Names & Vocab Tab: Treeview frame (initially None, created when needed)
        self.csv_treeview = None
        self.treeview_frame = None  # Frame to hold treeview and scrollbars

        # Session 80b: Filter widgets for vocabulary table
        self.filter_frame = None
        self.filter_entry = None
        self.filter_regex_var = None  # BooleanVar for regex checkbox
        self._detached_items = []  # Items hidden by filter (for restore)

        # Right-click context menu for vocabulary exclusion
        self.context_menu = None
        self._selected_term = None

        # Session 85: Extraction state for feedback blocking
        self._extraction_in_progress = False

        # Q&A Tab: Q&A panel (created eagerly to prevent tab switching artifacts)
        # See: https://github.com/TomSchimansky/CustomTkinter/issues/1508
        self._qa_panel = QAPanel(self.tabview.tab("Ask Questions"))
        self._qa_panel.grid(row=0, column=0, sticky="nsew")

        # Summary Tab: Textbox for summaries
        self.summary_text_display = ctk.CTkTextbox(self.tabview.tab("Summary"), wrap="word")
        self.summary_text_display.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.summary_text_display.insert(
            "0.0",
            "Generated summaries will appear here.\n\nProcess documents and enable 'Summary' to generate content.",
        )

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

        # Session 80: Column visibility state (replaces Show Details toggle)
        self._column_visibility = self._load_column_visibility()
        self.column_picker_btn = ctk.CTkButton(
            self.button_frame,
            text="Columns...",
            command=self._show_column_menu,
            width=90,
            **BUTTON_STYLES["secondary"],
        )
        self.column_picker_btn.pack(side="left", padx=5)

        # Session 72: Export dropdown (replaces separate CSV/Word/PDF buttons)
        # Note: CTkOptionMenu doesn't support hover_color, so only use fg_color
        self.export_dropdown = ctk.CTkOptionMenu(
            self.button_frame,
            values=["Export...", "TXT", "CSV", "Word (.docx)", "PDF", "HTML"],
            command=self._on_export_format_selected,
            width=120,
            fg_color=BUTTON_STYLES["primary"]["fg_color"],
        )
        self.export_dropdown.pack(side="left", padx=5)

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
        self._insertion_cancelled = False  # Cancels pending async insertions

        # Session 80: Sort state for vocabulary table
        self._sort_column = None  # Currently sorted column name
        self._sort_ascending = True  # Sort direction
        self._unsorted_vocab_data = []  # Original order for reset

        # Resize event debouncing to prevent glitchiness (Session 51)
        self._resize_after_id = None  # Track pending resize callbacks
        self._batch_insertion_paused = False  # Pause batch insertion during resize

        # Tooltip for potential duplicates
        self._tooltip = None  # Tooltip window
        self._tooltip_after_id = None  # Delayed tooltip show
        self._item_to_data: dict[str, dict] = {}  # Treeview item ID -> term data mapping

        # Feedback manager for ML learning (Session 25)
        from src.services import VocabularyService

        self._feedback_manager = VocabularyService().get_feedback_manager()

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
        Handle tab change to show/hide appropriate button bars (Session 68, 78, 80).

        Hides the shared button bar when Q&A tab is active (since QAPanel
        has its own buttons), shows it for other tabs.

        Session 78: Also shows/hides the main window's follow-up input frame
        so it only appears when the Q&A tab is active.

        Session 80: Saves column widths when leaving Names & Vocab tab.
        """
        current_tab = self.tabview.get()

        # Session 80: Save column widths when leaving Names & Vocab tab
        # (Widths may have been adjusted by user dragging column separators)
        self._save_column_widths()

        # Session 78: Show/hide main window's follow-up frame based on tab
        main_window = self.winfo_toplevel()
        if hasattr(main_window, "followup_frame"):
            if current_tab == "Ask Questions":
                main_window.followup_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
            else:
                main_window.followup_frame.grid_remove()

        if current_tab == "Ask Questions":
            # Hide shared button bar - QAPanel has its own buttons
            self.button_frame.grid_remove()
        else:
            # Show shared button bar for Names & Vocab and Summary tabs
            self.button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

            # Show/hide vocab-specific widgets based on tab
            if current_tab == "Names & Vocab":
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

    def _update_progress_badge(self, source: str):
        """
        Update the progress badge to show data source status (Session 45).

        Session 80: Changed "NER only" to "Local algorithms" since results
        actually come from NER, RAKE, and BM25 (not just NER). The badge
        indicates extraction phase (local vs LLM-enhanced), not which algorithms.

        Args:
            source: "none", "ner" (local algorithms), or "both" (local + LLM)
        """
        if source == "none":
            self._progress_badge.configure(text="")
        elif source == "partial":
            # Session 85: BM25 + RAKE results shown before NER completes
            self._progress_badge.configure(
                text="Partial results (BM25+RAKE)", text_color=("gray50", "gray70")
            )
        elif source == "ner":
            # Session 80: More accurate label - results come from NER, RAKE, BM25
            self._progress_badge.configure(
                text="Results (local algorithms)", text_color=("orange", "#ffaa00")
            )
        elif source == "both":
            self._progress_badge.configure(
                text="Enhanced results (+ LLM)", text_color=("green", "#00cc66")
            )

    def set_extraction_source(self, source: str):
        """
        Set the data source for the current extraction (Session 45).

        Called by workflow orchestrator to update progress badge.

        Args:
            source: "none", "ner", "partial" (BM25+RAKE only), or "both"
        """
        self._extraction_source = source
        # Update badge if Names & Vocabulary tab is currently active
        current_tab = self.tabview.get()
        if current_tab == "Names & Vocab":
            self._update_progress_badge(source)

    def set_extraction_in_progress(self, in_progress: bool):
        """
        Set whether vocabulary extraction is in progress (Session 85).

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
    # Session 80: Column Visibility Management
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

    # -------------------------------------------------------------------------
    # Session 80: Column Width Persistence
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
                if isinstance(width, int) and 30 <= width <= 500:
                    widths[col] = width
            except Exception:
                pass  # Column may not exist in current view

        if widths:
            prefs = get_user_preferences()
            prefs.set("vocab_column_widths", widths)

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

    def _show_column_menu(self, event=None):
        """
        Show column visibility context menu (Session 80).

        Can be triggered by the Columns button or right-click on header.
        """
        menu = Menu(
            self,
            tearoff=0,
            bg="#404040",
            fg="white",
            activebackground="#505050",
            activeforeground="white",
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
        menu.add_command(label="Reset to Defaults", command=self._reset_column_visibility)

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
        Sort vocabulary table by column (Session 80, 80b).

        Clicking a header sorts ascending; clicking again sorts descending;
        clicking a third time resets to original order.

        Session 80b: Shows warning dialog when sorting by non-Score columns
        since those sorts will show lower-quality results first.

        Args:
            column: The column name to sort by
        """
        if not self._unsorted_vocab_data or self.csv_treeview is None:
            return

        # Session 80b: Show warning for non-Score columns (first click only)
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
            "Score": "Quality Score",
            "# Docs": "# Docs",
            "Count": "Count",
            "Algo Count": "Algo Count",
            "Freq Rank": "Freq Rank",
        }
        data_key = column_key_map.get(column, column)

        # Numeric columns for type-aware sorting
        numeric_columns = {"Score", "Quality Score", "# Docs", "Count", "Algo Count", "Freq Rank"}
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
        term_to_index = {item.get("Term", ""): i for i, item in enumerate(data)}

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
            term = item.get("Term", "")
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
    # Session 80b: Text Filter for Vocabulary Table
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

        # Detach non-matching items
        filter_lower = filter_text.lower()
        for item_id in self.csv_treeview.get_children():
            values = self.csv_treeview.item(item_id, "values")
            if values:
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

        # Session 80: Save column widths before cleanup, then clear sort state
        self._save_column_widths()
        self._unsorted_vocab_data = []
        self._sort_column = None
        self._sort_ascending = True

        # Clear treeview data if it exists
        if self.csv_treeview is not None:
            self.csv_treeview.delete(*self.csv_treeview.get_children())

        # Force garbage collection
        gc.collect()
        debug_log("[VOCAB DISPLAY] Cleanup completed, memory freed.")

    def update_outputs(
        self,
        meta_summary: str = "",
        vocab_csv_data: list | None = None,
        document_summaries: dict | None = None,
        qa_results: list | None = None,
        briefing_text: str = "",
        briefing_sections: dict | None = None,
        # Session 45 new parameters
        names_vocab_data: list | None = None,
        summary_text: str = "",
        extraction_source: str | None = None,
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
        vocab_data = self._outputs.get("Names & Vocabulary") or self._outputs.get(
            "Rare Word List (CSV)"
        )
        if vocab_data:
            self._display_csv(vocab_data)
            self._update_progress_badge(self._extraction_source)

        # Q&A tab - always enabled if Q&A system is ready
        main_window = self.winfo_toplevel()
        getattr(main_window, "_qa_ready", False)
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
            self.summary_text_display.insert("end", "\n\n" + "=" * 50 + "\n")
            self.summary_text_display.insert("end", "INDIVIDUAL DOCUMENT SUMMARIES\n")
            self.summary_text_display.insert("end", "=" * 50 + "\n\n")
            for doc_name, doc_summary in sorted(self._document_summaries.items()):
                self.summary_text_display.insert("end", f"{doc_name}:\n{doc_summary}\n\n")

        # Session 78: Only switch to Vocab tab when there's vocab data
        # Don't switch to Summary/Q&A automatically - user should stay on Vocab tab
        # as it's the most useful for court reporters
        if vocab_data:
            self.tabview.set("Names & Vocab")

    def _display_csv(self, data: list):
        """
        Displays vocabulary data in an Excel-like Treeview with frozen headers.

        Uses async batch insertion with pagination for GUI responsiveness.
        Initial load shows ROWS_PER_PAGE items, "Load More" button adds more.

        Session 80: Filters out items with negative feedback (Skip) from display.
        This ensures previously skipped items don't reappear in new sessions.

        Args:
            data: List of dicts with keys: Term, Quality Score, Is Person, Found By
        """
        if not data:
            debug_log("[VOCAB DISPLAY] No vocabulary data to display")
            return

        # Session 80: Filter out items with negative feedback (previously skipped)
        # This prevents old skipped items from appearing even if not in exclusion file
        original_count = len(data)
        data = [
            item
            for item in data
            if self._feedback_manager.get_rating(
                item.get("Term", "") if isinstance(item, dict) else ""
            )
            != -1
        ]
        filtered_count = original_count - len(data)
        if filtered_count > 0:
            debug_log(f"[VOCAB DISPLAY] Filtered {filtered_count} previously skipped items")

        if not data:
            debug_log("[VOCAB DISPLAY] All items were filtered (previously skipped)")
            return

        # Filter by quality score floor
        # TODO: Test score floor filtering for both GUI display and CSV export
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        score_floor = prefs.get("vocab_score_floor", 55)
        pre_score_count = len(data)
        data = [
            item
            for item in data
            if isinstance(item, dict) and item.get("Quality Score", 0) >= score_floor
        ]
        score_filtered = pre_score_count - len(data)
        if score_filtered > 0:
            debug_log(
                f"[VOCAB DISPLAY] Filtered {score_filtered} items below score floor {score_floor}"
            )

        if not data:
            debug_log("[VOCAB DISPLAY] All items filtered by score floor")
            return

        # Session 80: Store unsorted data for sort operations and reset sort state
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
                self.tabview.tab("Names & Vocab"), **FRAME_STYLES["card"]
            )

        self.treeview_frame.grid(row=0, column=0, sticky="nsew")
        self.treeview_frame.grid_columnconfigure(0, weight=1)
        self.treeview_frame.grid_rowconfigure(1, weight=1)  # Treeview row expands

        # Session 80b: Create filter bar at top of treeview frame
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

        # Session 80: Get visible columns from user preferences
        columns = tuple(self._get_visible_columns())
        self._current_columns = columns  # Store for use in _async_insert_rows

        # Create or reconfigure treeview
        if self.csv_treeview is None:
            self.csv_treeview = ttk.Treeview(
                self.treeview_frame,
                columns=columns,
                show="headings",
                style="Vocab.Treeview",
                selectmode="browse",
            )

            # Configure column headings and widths
            # Session 80: Add click-to-sort functionality and use saved column widths
            for col in columns:
                # Use saved width if available, else default from COLUMN_CONFIG
                col_width = self._get_column_width(col)
                # Lambda capture col by default argument to avoid closure issue
                self.csv_treeview.heading(
                    col, text=col, anchor="w", command=lambda c=col: self._sort_by_column(c)
                )
                self.csv_treeview.column(
                    col,
                    width=col_width,
                    minwidth=60,
                    anchor="w",
                    stretch=col == "Term",  # Term stretches to fill space
                )

            # Add vertical scrollbar
            vsb = ttk.Scrollbar(
                self.treeview_frame,
                orient="vertical",
                command=self.csv_treeview.yview,
                style="Vocab.Vertical.TScrollbar",
            )
            self.csv_treeview.configure(yscrollcommand=vsb.set)

            # Add horizontal scrollbar
            hsb = ttk.Scrollbar(
                self.treeview_frame,
                orient="horizontal",
                command=self.csv_treeview.xview,
                style="Vocab.Horizontal.TScrollbar",
            )
            self.csv_treeview.configure(xscrollcommand=hsb.set)

            # Grid layout (Session 80b: row=1 to make room for filter bar at row=0)
            self.csv_treeview.grid(row=1, column=0, sticky="nsew")
            vsb.grid(row=1, column=1, sticky="ns")
            hsb.grid(row=2, column=0, sticky="ew")

            # Bind right-click for context menu
            self.csv_treeview.bind("<Button-3>", self._on_right_click)
            self.csv_treeview.bind("<Double-1>", self._on_double_click)
            # Bind left-click for feedback columns (Session 25)
            self.csv_treeview.bind("<Button-1>", self._on_treeview_click)
            # Bind hover for potential duplicate tooltip
            self.csv_treeview.bind("<Motion>", self._on_treeview_hover)
            self.csv_treeview.bind("<Leave>", self._hide_tooltip)

            # Create context menu
            self._create_context_menu()

            # Configure all vocabulary table tags from centralized theme
            for tag_name, tag_config in VOCAB_TABLE_TAGS.items():
                self.csv_treeview.tag_configure(tag_name, **tag_config)

        # Cancel any pending async insertion before starting new one
        # This fixes race condition when ner_complete arrives before partial_vocab_complete finishes
        if self._is_loading:
            debug_log("[VOCAB DISPLAY] Cancelling pending async insertion for new data")
            self._insertion_cancelled = True
            self._is_loading = False

        # Clear existing data and item mapping
        self.csv_treeview.delete(*self.csv_treeview.get_children())
        self._item_to_data.clear()

        # Calculate how many items to load initially
        initial_load = min(ROWS_PER_PAGE, self._vocab_total_items)

        debug_log(
            f"[VOCAB DISPLAY] Showing {initial_load} of {self._vocab_total_items} terms "
            f"(pagination: {ROWS_PER_PAGE} per page)"
        )

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
            if hasattr(main_window, "_ask_followup_for_qa_panel"):
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
                "if enabled in Settings > Q&A/Briefing > Auto-run.",
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
        self._insertion_cancelled = False  # Reset cancellation flag for new operation

        current_idx = start_idx

        def insert_batch():
            nonlocal current_idx

            # Check if this operation was cancelled (new data arrived)
            if self._insertion_cancelled:
                debug_log("[VOCAB DISPLAY] Async insertion cancelled - stopping")
                self._is_loading = False
                return

            # If paused by window resize, reschedule and wait
            if self._batch_insertion_paused:
                self.after(BATCH_INSERT_DELAY_MS, insert_batch)
                return

            # Insert a batch of rows
            batch_end = min(current_idx + BATCH_INSERT_SIZE, end_idx)

            # Session 47: Use stored columns (may be extended with NER/RAKE/BM25)
            current_columns = getattr(self, "_current_columns", GUI_DISPLAY_COLUMNS)

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
                        elif col == "Term":
                            # Special handling for Term column - add 🔗 for potential duplicates
                            term_display = item.get("Term", "")
                            if item.get("_potential_duplicate_of"):
                                term_display = f"🔗 {term_display}"
                            values.append(
                                truncate_text(str(term_display), COLUMN_CONFIG[col]["max_chars"])
                            )
                        elif col in DISPLAY_TO_DATA_COLUMN:
                            # Map display column to data field (e.g., "Score" -> "Quality Score")
                            data_col = DISPLAY_TO_DATA_COLUMN[col]
                            value = item.get(data_col, "")
                            values.append(
                                truncate_text(str(value), COLUMN_CONFIG[col]["max_chars"])
                            )
                        else:
                            values.append(
                                truncate_text(
                                    str(item.get(col, "")), COLUMN_CONFIG[col]["max_chars"]
                                )
                            )

                    values = tuple(values)
                else:
                    # Handle list format (legacy) - apply truncation, default empty feedback
                    raw_values = (
                        tuple(item) if len(item) >= 4 else tuple(item) + ("",) * (4 - len(item))
                    )
                    values = (
                        *tuple(
                            truncate_text(str(v), COLUMN_CONFIG[current_columns[j]]["max_chars"])
                            for j, v in enumerate(raw_values[:4])
                        ),
                        THUMB_UP_EMPTY,
                        THUMB_DOWN_EMPTY,
                    )

                # Apply tag for row coloring based on existing rating or Found By (Session 43)
                # Session 51: Add alternating row background color
                row_bg_tag = "oddrow" if i % 2 else "evenrow"

                # Session 78: Row coloring should ONLY reflect feedback status
                # - Thumbs up (rating=1) → green
                # - Thumbs down (rating=-1) → red
                # - No feedback (rating=0) → neutral (just alternating row background)
                # Removed algorithm-based coloring (found_ner, found_rake, etc.) since
                # colors should only indicate user/developer feedback, not detection source.
                # Session 84: Distinguish feedback source (session vs loaded)
                rating_source = self._feedback_manager.get_rating_source(term)
                if rating == 1:
                    if rating_source == "session":
                        tag = (row_bg_tag, "rated_up_session")
                    else:
                        tag = (row_bg_tag, "rated_up_loaded")
                elif rating == -1:
                    if rating_source == "session":
                        tag = (row_bg_tag, "rated_down_session")
                    else:
                        tag = (row_bg_tag, "rated_down_loaded")
                else:
                    # No feedback - neutral coloring (just alternating background)
                    tag = (row_bg_tag,)
                item_id = self.csv_treeview.insert("", "end", values=values, tags=tag)
                # Store mapping for tooltip lookup
                if isinstance(item, dict):
                    self._item_to_data[item_id] = item

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

        debug_log(f"[PAGINATION] Updating UI: {displayed_items}/{total_items} displayed")

        # Create or update "Load More" button
        if displayed_items < total_items:
            remaining = total_items - displayed_items

            if self._load_more_btn is None:
                self._load_more_btn = ctk.CTkButton(
                    self.treeview_frame, text="", height=28, **BUTTON_STYLES["primary"]
                )

            # Session 80: Always update command to use current data (fixes stale closure)
            self._load_more_btn.configure(
                text=f"Load More ({remaining} remaining)",
                command=lambda d=data: self._load_more_rows(d),
            )
            self._load_more_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
            debug_log(f"[PAGINATION] Load More button shown ({remaining} remaining)")

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
            debug_log(f"[PAGINATION] All {total_items} items displayed, hiding Load More")
            if self._load_more_btn is not None:
                self._load_more_btn.grid_remove()

            if hasattr(self, "vocab_info_label"):
                self.vocab_info_label.configure(text=f"Showing all {total_items} terms")
                self.vocab_info_label.grid(
                    row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5
                )

        # PERF-008: Call gc.collect() directly (lightweight operation)
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

        debug_log(f"[VOCAB DISPLAY] Loading more: rows {start_idx} to {end_idx}")

        # Start async insertion
        self._async_insert_rows(data, start_idx, end_idx)

    def _create_context_menu(self):
        """Create right-click context menu for vocabulary table."""
        self.context_menu = Menu(
            self,
            tearoff=0,
            bg="#404040",
            fg="white",
            activebackground="#505050",
            activeforeground="white",
            font=("Segoe UI", 10),
        )
        self.context_menu.add_command(
            label="Exclude this term from future lists", command=self._exclude_selected_term
        )
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Copy term", command=self._copy_selected_term)

    def _on_right_click(self, event):
        """Handle right-click on treeview - show column menu for header, context menu for rows."""
        # Session 80: Check if click is on header region
        region = self.csv_treeview.identify_region(event.x, event.y)
        if region == "heading":
            # Show column visibility menu
            self._show_column_menu(event)
            return

        # Identify the row under cursor
        item_id = self.csv_treeview.identify_row(event.y)
        if item_id:
            # Select the row
            self.csv_treeview.selection_set(item_id)
            # Get the term value (first column)
            values = self.csv_treeview.item(item_id, "values")
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
            values = self.csv_treeview.item(item_id, "values")
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
            icon="question",
        )

        if not result:
            return

        # Add to exclusion file
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(USER_VOCAB_EXCLUDE_PATH), exist_ok=True)

            # Append to file
            with open(USER_VOCAB_EXCLUDE_PATH, "a", encoding="utf-8") as f:
                f.write(f"{lower_term}\n")

            debug_log(
                f"[VOCAB UI] Added '{term}' to user exclusion list at {USER_VOCAB_EXCLUDE_PATH}"
            )

            # Remove from current display
            selected = self.csv_treeview.selection()
            if selected:
                self.csv_treeview.delete(selected[0])

                # Also remove from internal data
                self._outputs["Rare Word List (CSV)"] = [
                    item
                    for item in self._outputs.get("Rare Word List (CSV)", [])
                    if isinstance(item, dict) and item.get("Term", "").lower() != lower_term
                ]

            messagebox.showinfo(
                "Term Excluded",
                f"'{term}' will not appear in future rare word lists.\n\n"
                "Note: This takes effect on the next vocabulary extraction.",
            )

        except Exception as e:
            debug_log(f"[VOCAB UI] Failed to save exclusion: {e}")
            messagebox.showerror(
                "Error", f"Failed to save exclusion: {e}\n\nPlease check file permissions."
            )

    def _copy_selected_term(self):
        """Copy the selected term to clipboard."""
        if self._selected_term:
            self.clipboard_clear()
            self.clipboard_append(self._selected_term)

    def _add_to_user_exclusion_list(self, term: str) -> None:
        """
        Add a term to the user exclusion list (silent, no dialog).

        Called when user gives negative feedback (-1) to a term.
        The term will be filtered out of future vocabulary extractions.

        Session 80: Improved logging to debug exclusion persistence issues.

        Args:
            term: The term to exclude (case-insensitive)
        """
        if not term:
            return

        lower_term = term.lower().strip()
        debug_log(
            f"[FEEDBACK] Adding '{lower_term}' to exclusion list at {USER_VOCAB_EXCLUDE_PATH}"
        )

        try:
            # Session 80: Use Path methods instead of os.path for consistency
            USER_VOCAB_EXCLUDE_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Check if term is already in the list
            existing_terms = set()
            if USER_VOCAB_EXCLUDE_PATH.exists():
                with open(USER_VOCAB_EXCLUDE_PATH, encoding="utf-8") as f:
                    existing_terms = {line.strip().lower() for line in f if line.strip()}
                debug_log(f"[FEEDBACK] Existing exclusions: {len(existing_terms)} terms")

            if lower_term in existing_terms:
                debug_log(f"[FEEDBACK] Term '{term}' already in exclusion list, skipping")
                return

            # Append to file
            with open(USER_VOCAB_EXCLUDE_PATH, "a", encoding="utf-8") as f:
                f.write(f"{lower_term}\n")

            # Verify write succeeded
            if USER_VOCAB_EXCLUDE_PATH.exists():
                with open(USER_VOCAB_EXCLUDE_PATH, encoding="utf-8") as f:
                    new_count = sum(1 for line in f if line.strip())
                debug_log(
                    f"[FEEDBACK] Successfully added '{term}' to exclusion list "
                    f"(now {new_count} total exclusions)"
                )
            else:
                debug_log(
                    f"[FEEDBACK] WARNING: File not found after write: {USER_VOCAB_EXCLUDE_PATH}"
                )

        except Exception as e:
            debug_log(f"[FEEDBACK] ERROR: Failed to add '{term}' to exclusion list: {e}")

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
        - "all": All columns including Quality Score, In-Case Freq, Freq Rank
        - "basic": Term, Score, Is Person, Found By
        - "terms_only": Just the Term column
        """
        current_tab = self.tabview.get()

        if current_tab == "Names & Vocab":
            data = self._outputs.get("Names & Vocabulary") or self._outputs.get(
                "Rare Word List (CSV)", []
            )
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

            # Brief button flash for immediate feedback
            original_text = self.copy_btn.cget("text")
            self.copy_btn.configure(text="Copied!")
            self.after(1500, lambda: self.copy_btn.configure(text=original_text))

            # Status bar confirmation with count (Session 69)
            current_tab = self.tabview.get()
            main_window = self.winfo_toplevel()
            if current_tab == "Names & Vocab":
                vocab_data = self._outputs.get("Names & Vocabulary") or self._outputs.get(
                    "Rare Word List (CSV)", []
                )
                main_window.set_status(
                    f"Copied {len(vocab_data)} terms to clipboard", duration_ms=5000
                )
            elif current_tab == "Summary":
                main_window.set_status("Copied summary to clipboard", duration_ms=5000)
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

        # Session 73: Remember last export folder
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

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            # Session 73: Remember last export folder
            prefs.set("last_export_path", str(Path(filepath).parent))

            # Brief button flash for immediate feedback
            original_text = self.save_btn.cget("text")
            self.save_btn.configure(text="Saved!")
            self.after(1500, lambda: self.save_btn.configure(text=original_text))

            # Status bar confirmation with details (Session 69)
            main_window = self.winfo_toplevel()
            filename = os.path.basename(filepath)
            if current_tab == "Names & Vocab":
                vocab_data = self._outputs.get("Names & Vocabulary") or self._outputs.get(
                    "Rare Word List (CSV)", []
                )
                main_window.set_status(
                    f"Saved {len(vocab_data)} terms to {filename}", duration_ms=5000
                )
            elif current_tab == "Ask Questions":
                main_window.set_status(f"Saved Q&A results to {filename}", duration_ms=5000)
            elif current_tab == "Summary":
                main_window.set_status(f"Saved summary to {filename}", duration_ms=5000)

    def _quick_export_vocab_csv(self):
        """
        Quick export vocabulary to CSV file (Session 65, updated Session 69).

        Exports to last used folder (or Documents) with timestamped filename.
        Uses status bar confirmation instead of modal dialog.
        """
        from datetime import datetime
        from pathlib import Path

        # Check if we have vocabulary data
        vocab_data = self._outputs.get("Names & Vocabulary") or self._outputs.get(
            "Rare Word List (CSV)", []
        )
        if not vocab_data:
            messagebox.showwarning(
                "No Data", "No vocabulary data to export.\n\nProcess documents first."
            )
            return

        # Generate CSV content using shared helper
        csv_content = self._build_vocab_csv(vocab_data)

        # Session 73: Use last export folder or Documents
        from src.services import DocumentService

        prefs = get_user_preferences()
        export_path = (
            prefs.get("last_export_path") or DocumentService().get_default_documents_folder()
        )

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vocabulary_{timestamp}.csv"
        filepath = os.path.join(export_path, filename)

        # Save the file
        try:
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write(csv_content)

            # Session 73: Remember export folder
            prefs.set("last_export_path", str(Path(filepath).parent))

            debug_log(f"[VOCAB EXPORT] Saved {len(vocab_data)} terms to {filepath}")

            # Status bar confirmation (Session 69, updated 73)
            main_window = self.winfo_toplevel()
            folder_name = Path(export_path).name
            main_window.set_status(
                f"Exported {len(vocab_data)} terms to {folder_name}/{filename}", duration_ms=5000
            )

        except Exception as e:
            debug_log(f"[VOCAB EXPORT] Failed: {e}")
            messagebox.showerror("Export Failed", f"Could not save file:\n{e}")

    def _export_vocab_to_word(self):
        """
        Export vocabulary to Word document (Session 71, updated Session 73).
        """
        from datetime import datetime
        from pathlib import Path

        from src.services import DocumentService, get_export_service

        # Session 80: Use fallback to legacy key like other export functions
        vocab_data = self._outputs.get("Names & Vocabulary") or self._outputs.get(
            "Rare Word List (CSV)", []
        )
        if not vocab_data:
            messagebox.showinfo("No Data", "No vocabulary data to export.")
            return

        # Session 73: Use last export folder or Documents
        prefs = get_user_preferences()
        export_path = (
            prefs.get("last_export_path") or DocumentService().get_default_documents_folder()
        )

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vocabulary_{timestamp}.docx"
        filepath = os.path.join(export_path, filename)

        # Export using service
        export_service = get_export_service()
        # Session 80: Include details if any algorithm columns are visible
        include_details = any(
            self._column_visibility.get(col, False) for col in ["NER", "RAKE", "BM25", "Algo Count"]
        )
        success = export_service.export_vocabulary_to_word(vocab_data, filepath, include_details)

        if success:
            # Session 73: Remember export folder
            prefs.set("last_export_path", str(Path(filepath).parent))

            # Status bar confirmation
            main_window = self.winfo_toplevel()
            folder_name = Path(export_path).name
            main_window.set_status(
                f"Exported {len(vocab_data)} terms to {folder_name}/{filename}", duration_ms=5000
            )
        else:
            messagebox.showerror("Export Failed", "Could not export to Word document.")

    def _export_vocab_to_pdf(self):
        """
        Export vocabulary to PDF document (Session 71, updated Session 73).
        """
        from datetime import datetime
        from pathlib import Path

        from src.services import DocumentService, get_export_service

        # Session 80: Use fallback to legacy key like other export functions
        vocab_data = self._outputs.get("Names & Vocabulary") or self._outputs.get(
            "Rare Word List (CSV)", []
        )
        if not vocab_data:
            messagebox.showinfo("No Data", "No vocabulary data to export.")
            return

        # Session 73: Use last export folder or Documents
        prefs = get_user_preferences()
        export_path = (
            prefs.get("last_export_path") or DocumentService().get_default_documents_folder()
        )

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vocabulary_{timestamp}.pdf"
        filepath = os.path.join(export_path, filename)

        # Export using service
        export_service = get_export_service()
        # Session 80: Include details if any algorithm columns are visible
        include_details = any(
            self._column_visibility.get(col, False) for col in ["NER", "RAKE", "BM25", "Algo Count"]
        )
        success = export_service.export_vocabulary_to_pdf(vocab_data, filepath, include_details)

        if success:
            # Session 73: Remember export folder
            prefs.set("last_export_path", str(Path(filepath).parent))

            # Status bar confirmation
            main_window = self.winfo_toplevel()
            folder_name = Path(export_path).name
            main_window.set_status(
                f"Exported {len(vocab_data)} terms to {folder_name}/{filename}", duration_ms=5000
            )
        else:
            messagebox.showerror("Export Failed", "Could not export to PDF document.")

    def _export_vocab_to_txt(self):
        """
        Export vocabulary to plain text file (Session 72, updated Session 73).
        """
        from datetime import datetime
        from pathlib import Path

        from src.services import DocumentService, get_export_service

        # Session 80: Use fallback to legacy key like other export functions
        vocab_data = self._outputs.get("Names & Vocabulary") or self._outputs.get(
            "Rare Word List (CSV)", []
        )
        if not vocab_data:
            messagebox.showinfo("No Data", "No vocabulary data to export.")
            return

        # Session 73: Use last export folder or Documents
        prefs = get_user_preferences()
        export_path = (
            prefs.get("last_export_path") or DocumentService().get_default_documents_folder()
        )

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vocabulary_{timestamp}.txt"
        filepath = os.path.join(export_path, filename)

        # Export using service
        export_service = get_export_service()
        success = export_service.export_vocabulary_to_txt(vocab_data, filepath)

        if success:
            # Session 73: Remember export folder
            prefs.set("last_export_path", str(Path(filepath).parent))

            # Status bar confirmation
            main_window = self.winfo_toplevel()
            folder_name = Path(export_path).name
            main_window.set_status(
                f"Exported {len(vocab_data)} terms to {folder_name}/{filename}", duration_ms=5000
            )
        else:
            messagebox.showerror("Export Failed", "Could not export to text file.")

    def _export_vocab_to_html(self):
        """
        Export vocabulary to interactive HTML file (Session 72, updated Session 80).

        Session 80: Now passes visible columns to HTML export so the exported
        file mirrors the GUI's column visibility settings.
        """
        from datetime import datetime
        from pathlib import Path

        from src.services import DocumentService, get_export_service

        # Session 80: Use fallback to legacy key like other export functions
        vocab_data = self._outputs.get("Names & Vocabulary") or self._outputs.get(
            "Rare Word List (CSV)", []
        )
        if not vocab_data:
            messagebox.showinfo("No Data", "No vocabulary data to export.")
            return

        # Session 73: Use last export folder or Documents
        prefs = get_user_preferences()
        export_path = (
            prefs.get("last_export_path") or DocumentService().get_default_documents_folder()
        )

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vocabulary_{timestamp}.html"
        filepath = os.path.join(export_path, filename)

        # Session 80: Pass current visible columns to HTML export
        visible_columns = self._get_visible_columns()

        # Export using service
        export_service = get_export_service()
        success = export_service.export_vocabulary_to_html(vocab_data, filepath, visible_columns)

        if success:
            # Session 73: Remember export folder
            prefs.set("last_export_path", str(Path(filepath).parent))

            # Status bar confirmation
            main_window = self.winfo_toplevel()
            folder_name = Path(export_path).name
            main_window.set_status(
                f"Exported {len(vocab_data)} terms to {folder_name}/{filename}", duration_ms=5000
            )
        else:
            messagebox.showerror("Export Failed", "Could not export to HTML file.")

    def _on_export_format_selected(self, choice: str):
        """
        Handle export format selection from dropdown (Session 72).

        Args:
            choice: Selected format ("Export...", "TXT", "CSV", "Word (.docx)", "PDF", "HTML")
        """
        if choice == "Export...":
            return  # Placeholder, do nothing

        if choice == "TXT":
            self._export_vocab_to_txt()
        elif choice == "CSV":
            self._quick_export_vocab_csv()
        elif choice == "Word (.docx)":
            self._export_vocab_to_word()
        elif choice == "PDF":
            self._export_vocab_to_pdf()
        elif choice == "HTML":
            self._export_vocab_to_html()

        # Reset dropdown to placeholder
        self.export_dropdown.set("Export...")

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
        current_columns = getattr(self, "_current_columns", GUI_DISPLAY_COLUMNS)
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

    def _toggle_feedback(self, item_id: str, feedback_type: int):
        """
        Toggle feedback state for a vocabulary term.

        If the term already has this feedback, clear it.
        If the term has opposite or no feedback, set the new feedback.

        Args:
            item_id: Treeview item identifier
            feedback_type: +1 for Keep, -1 for Skip
        """
        # Session 85: Block feedback while extraction is in progress
        # Session 86: Use non-blocking status message instead of modal dialog
        # (modal messagebox blocks UI event loop, causing freeze during NER)
        if self._extraction_in_progress:
            main_window = self.winfo_toplevel()
            if hasattr(main_window, "set_status"):
                main_window.set_status(
                    "Feedback disabled during extraction. Please wait for NER to complete.",
                    duration_ms=3000,
                )
            return

        # Session 68: Warn about missing corpus (once per session)
        if not self._check_corpus_and_warn():
            return  # User cancelled

        # Get the term from the row
        values = self.csv_treeview.item(item_id, "values")
        if not values:
            return

        term = values[0]  # Term is first column
        current_rating = self._feedback_manager.get_rating(term)

        # Toggle logic: if already this rating, clear it; otherwise set it
        new_rating = 0 if current_rating == feedback_type else feedback_type

        # Find full term data from internal storage for ML features
        term_data = self._find_term_data(term)
        if not term_data:
            term_data = {"Term": term}

        # Record feedback (handles both setting and clearing)
        success = self._feedback_manager.record_feedback(term_data, new_rating)

        if success:
            # Update the visual display
            self._update_feedback_display(item_id, new_rating)
            debug_log(
                f"[FEEDBACK UI] {'Cleared' if new_rating == 0 else 'Set'} "
                f"feedback for '{term}': {new_rating}"
            )

            # Session 78: Add skipped terms to user exclusion list
            # This filters them out of future extractions
            if new_rating == -1:
                self._add_to_user_exclusion_list(term)

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
            if isinstance(item, dict) and item.get("Term", "").lower().strip() == lower_term:
                return item

        return None

    def _update_feedback_display(self, item_id: str, rating: int):
        """
        Update the visual display of feedback icons for a term.

        Args:
            item_id: Treeview item identifier
            rating: +1 (Keep filled), -1 (Skip filled), 0 (both empty)
        """
        values = list(self.csv_treeview.item(item_id, "values"))

        # Session 47: Dynamically find Keep and Skip column indices
        current_columns = getattr(self, "_current_columns", GUI_DISPLAY_COLUMNS)
        try:
            keep_idx = current_columns.index("Keep")  # 0-based for list access
            skip_idx = current_columns.index("Skip")  # 0-based for list access
        except ValueError:
            return  # Keep/Skip columns not found

        if len(values) <= max(keep_idx, skip_idx):
            return

        # Update the icon values at dynamic positions
        # Session 84: User clicks always use session tags (darker colors)
        if rating == 1:
            values[keep_idx] = THUMB_UP_FILLED
            values[skip_idx] = THUMB_DOWN_EMPTY
            tag = ("rated_up_session",)
        elif rating == -1:
            values[keep_idx] = THUMB_UP_EMPTY
            values[skip_idx] = THUMB_DOWN_FILLED
            tag = ("rated_down_session",)
        else:  # rating == 0
            values[keep_idx] = THUMB_UP_EMPTY
            values[skip_idx] = THUMB_DOWN_EMPTY
            tag = ()

        # Update the item with new values and tag for coloring
        self.csv_treeview.item(item_id, values=tuple(values), tags=tag)

    def _on_treeview_hover(self, event):
        """
        Handle mouse hover over treeview rows.

        Shows a tooltip with potential duplicate information when hovering
        over rows that have been flagged as possible duplicates.
        """
        # Cancel any pending tooltip
        if self._tooltip_after_id:
            self.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None

        # Identify the row under the cursor
        item_id = self.csv_treeview.identify_row(event.y)
        if not item_id:
            self._hide_tooltip(None)
            return

        # Get term data from our mapping
        term_data = self._item_to_data.get(item_id)
        if not term_data:
            self._hide_tooltip(None)
            return

        # Check if this term has a potential duplicate
        duplicate_of = term_data.get("_potential_duplicate_of")
        if not duplicate_of:
            self._hide_tooltip(None)
            return

        # Show tooltip after a short delay (300ms)
        tooltip_text = f"Possible duplicate: {duplicate_of}"
        self._tooltip_after_id = self.after(
            300,
            lambda: self._show_tooltip(tooltip_text, event.x_root + 15, event.y_root + 10),
        )

    def _show_tooltip(self, text: str, x: int, y: int):
        """
        Display a tooltip window at the specified screen coordinates.

        Args:
            text: The text to display in the tooltip
            x: Screen x-coordinate for tooltip position
            y: Screen y-coordinate for tooltip position
        """
        # Hide any existing tooltip first
        self._hide_tooltip(None)

        # Create tooltip as a toplevel window
        self._tooltip = ctk.CTkToplevel(self)
        self._tooltip.wm_overrideredirect(True)  # No window decorations
        self._tooltip.wm_geometry(f"+{x}+{y}")

        # Prevent tooltip from taking focus
        self._tooltip.attributes("-topmost", True)

        # Create tooltip label with styling
        label = ctk.CTkLabel(
            self._tooltip,
            text=text,
            fg_color=("#FFF9C4", "#424242"),  # Light yellow / dark gray
            text_color=("#333333", "#FFFFFF"),
            corner_radius=4,
            padx=8,
            pady=4,
        )
        label.pack()

    def _hide_tooltip(self, event):
        """
        Hide and destroy the tooltip window.

        Args:
            event: The event that triggered hiding (can be None)
        """
        # Cancel any pending tooltip show
        if self._tooltip_after_id:
            self.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None

        # Destroy existing tooltip
        if self._tooltip:
            self._tooltip.destroy()
            self._tooltip = None
