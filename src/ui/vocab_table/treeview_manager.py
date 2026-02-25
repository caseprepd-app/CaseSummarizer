"""
Vocabulary Treeview Manager Mixin.

Contains:
- Column visibility management (load/save, menu, toggle)
- Column width persistence
- Sorting functionality
- Text filtering
- Async row insertion with pagination
"""

import logging
import tkinter as tk
from tkinter import ttk

from src.ui.vocab_table.column_config import (
    BATCH_INSERT_DELAY_MS,
    BATCH_INSERT_SIZE,
    COLUMN_ORDER,
    COLUMN_REGISTRY,
    DISPLAY_TO_DATA_COLUMN,
    truncate_text,
)

logger = logging.getLogger(__name__)


class TreeviewManagerMixin:
    """
    Mixin class providing treeview management functionality.

    Methods in this mixin assume the parent class has:
    - self.csv_tree: ttk.Treeview widget
    - self._vocab_csv_data: List of vocabulary dictionaries
    - self._visible_columns: Set of visible column names
    - self._column_widths: Dict of column name to width
    - self._sort_column: Current sort column name or None
    - self._sort_reverse: Boolean for sort direction
    """

    # =========================================================================
    # Column Visibility Management
    # =========================================================================

    def _load_column_visibility(self):
        """
        Load column visibility settings from user preferences.

        Loads saved visibility or uses COLUMN_REGISTRY defaults.
        """
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        saved = prefs.get("vocab_column_visibility", {})

        self._visible_columns = set()
        for col in COLUMN_ORDER:
            config = COLUMN_REGISTRY.get(col, {})
            # Use saved value if exists, otherwise use default from registry
            default = config.get("default", False)
            is_visible = saved.get(col, default)
            if is_visible:
                self._visible_columns.add(col)

        # Ensure "Term" is always visible (cannot be hidden)
        self._visible_columns.add("Term")

        logger.debug("Loaded visibility: %s columns", len(self._visible_columns))

    def _save_column_visibility(self):
        """Save current column visibility to user preferences."""
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        visibility = {col: (col in self._visible_columns) for col in COLUMN_ORDER}
        prefs.set("vocab_column_visibility", visibility)

    def _show_column_menu(self, event=None):
        """
        Show column visibility popup menu.

        Right-click or button click shows checkable column list.
        """
        menu = tk.Menu(self, tearoff=0)

        for col in COLUMN_ORDER:
            config = COLUMN_REGISTRY.get(col, {})
            can_hide = config.get("can_hide", True)

            # Create checkbutton variable
            var = tk.BooleanVar(value=(col in self._visible_columns))

            if can_hide:
                menu.add_checkbutton(
                    label=col,
                    variable=var,
                    command=lambda c=col, v=var: self._toggle_column(c, v.get()),
                )
            else:
                # Term column - show but disable
                menu.add_checkbutton(
                    label=f"{col} (required)",
                    variable=var,
                    state="disabled",
                )

        # Show menu at appropriate location
        if event:
            menu.tk_popup(event.x_root, event.y_root)
        elif hasattr(self, "columns_btn"):
            x = self.columns_btn.winfo_rootx()
            y = self.columns_btn.winfo_rooty() + self.columns_btn.winfo_height()
            menu.tk_popup(x, y)

    def _toggle_column(self, col_name: str, visible: bool):
        """
        Toggle visibility of a column.

        Args:
            col_name: Column name to toggle
            visible: True to show, False to hide
        """
        config = COLUMN_REGISTRY.get(col_name, {})
        if not config.get("can_hide", True):
            return  # Cannot toggle required columns

        if visible:
            self._visible_columns.add(col_name)
        else:
            self._visible_columns.discard(col_name)

        self._save_column_visibility()
        self._rebuild_treeview_columns()

        logger.debug("Column '%s' visibility: %s", col_name, visible)

    def _rebuild_treeview_columns(self):
        """Rebuild treeview with current visible columns."""
        if not hasattr(self, "csv_tree") or not self.csv_tree:
            return

        # Get current columns in order
        visible_cols = [c for c in COLUMN_ORDER if c in self._visible_columns]

        # Update treeview columns
        self.csv_tree["columns"] = visible_cols
        self.csv_tree["displaycolumns"] = visible_cols

        for col in visible_cols:
            config = COLUMN_REGISTRY.get(col, {})
            width = self._column_widths.get(col, config.get("width", 80))

            self.csv_tree.heading(
                col,
                text=col,
                command=lambda c=col: self._sort_by_column(c),
            )
            self.csv_tree.column(
                col,
                width=width,
                minwidth=40,
                anchor="w" if col == "Term" else "center",
            )

        # Refresh data display
        self._redisplay_sorted_data()

    # =========================================================================
    # Column Width Persistence
    # =========================================================================

    def _load_column_widths(self):
        """Load saved column widths from user preferences."""
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        self._column_widths = prefs.get("vocab_column_widths", {})

        # Fill in defaults for any missing columns
        for col, config in COLUMN_REGISTRY.items():
            if col not in self._column_widths:
                self._column_widths[col] = config.get("width", 80)

    def _save_column_widths(self):
        """Save current column widths to user preferences."""
        from src.user_preferences import get_user_preferences

        if not hasattr(self, "csv_tree") or not self.csv_tree:
            return

        prefs = get_user_preferences()

        # Read current widths from treeview
        for col in self._visible_columns:
            try:
                width = self.csv_tree.column(col, "width")
                if width and 40 <= width <= 500:
                    self._column_widths[col] = width
            except tk.TclError:
                pass

        prefs.set("vocab_column_widths", self._column_widths)

    def _get_column_width(self, col_name: str) -> int:
        """
        Get width for a column (saved or default).

        Args:
            col_name: Column name

        Returns:
            Width in pixels
        """
        if col_name in self._column_widths:
            return self._column_widths[col_name]
        config = COLUMN_REGISTRY.get(col_name, {})
        return config.get("width", 80)

    # =========================================================================
    # Sorting
    # =========================================================================

    def _sort_by_column(self, col_name: str):
        """
        Sort vocabulary data by the specified column.

        Click column header to sort. Click again to reverse.

        Args:
            col_name: Column name to sort by
        """
        if not hasattr(self, "_vocab_csv_data") or not self._vocab_csv_data:
            return

        # Toggle direction if same column, otherwise default to ascending
        if self._sort_column == col_name:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = col_name
            self._sort_reverse = False

        # Sort the data
        self._sort_vocab_data()
        self._update_sort_headers()
        self._redisplay_sorted_data()

        direction = "desc" if self._sort_reverse else "asc"
        logger.debug("Sorted by '%s' %s", col_name, direction)

    def _sort_vocab_data(self):
        """Sort _vocab_csv_data in place by current sort column."""
        if not self._sort_column or not self._vocab_csv_data:
            return

        col = self._sort_column
        data_key = DISPLAY_TO_DATA_COLUMN.get(col, col)

        def sort_key(item):
            val = item.get(data_key, item.get(col, ""))
            # Handle numeric sorting
            if col in (
                "Score",
                "Quality Score",
                "# Docs",
                "NER",
                "RAKE",
                "BM25",
                "Algo Count",
                "Google Rarity Rank",
            ):
                try:
                    return float(val) if val else 0
                except (ValueError, TypeError):
                    return 0
            # String sorting (case-insensitive)
            return str(val).lower()

        self._vocab_csv_data.sort(key=sort_key, reverse=self._sort_reverse)

    def _update_sort_headers(self):
        """Update column headers to show sort indicators."""
        if not hasattr(self, "csv_tree") or not self.csv_tree:
            return

        for col in self._visible_columns:
            if col == self._sort_column:
                indicator = " \u25bc" if self._sort_reverse else " \u25b2"
                self.csv_tree.heading(col, text=f"{col}{indicator}")
            else:
                self.csv_tree.heading(col, text=col)

    def _redisplay_sorted_data(self):
        """Redisplay vocabulary data with current sort and visibility."""
        if not hasattr(self, "csv_tree") or not self.csv_tree:
            return

        # Clear existing rows
        self.csv_tree.delete(*self.csv_tree.get_children())

        # Cancel any pending async inserts
        if hasattr(self, "_async_insert_id") and self._async_insert_id:
            self.after_cancel(self._async_insert_id)
            self._async_insert_id = None

        # Insert rows (async for large datasets)
        if hasattr(self, "_vocab_csv_data") and self._vocab_csv_data:
            self._async_insert_rows(self._vocab_csv_data, 0)

    # =========================================================================
    # Text Filtering
    # =========================================================================

    def _apply_filter(self, filter_text: str):
        """
        Filter vocabulary rows by text match.

        Filters Term column, hides non-matching rows.

        Args:
            filter_text: Text to search for (case-insensitive)
        """
        if not hasattr(self, "csv_tree") or not self.csv_tree:
            return

        filter_lower = filter_text.lower().strip()

        for item_id in self.csv_tree.get_children():
            values = self.csv_tree.item(item_id, "values")
            if not values:
                continue

            # Term is always first column
            term = str(values[0]).lower()

            if not filter_lower or filter_lower in term:
                # Show row by reattaching to tree
                self.csv_tree.reattach(item_id, "", "end")
            else:
                # Hide row by detaching
                self.csv_tree.detach(item_id)

    def _clear_filter(self):
        """Clear text filter and show all rows."""
        self._apply_filter("")

    # =========================================================================
    # Async Row Insertion
    # =========================================================================

    def _async_insert_rows(self, data: list[dict], start_index: int):
        """
        Insert rows asynchronously to keep GUI responsive.

        Batch inserts with delays to prevent UI freeze.

        Args:
            data: Full vocabulary data list
            start_index: Starting index for this batch
        """
        if not hasattr(self, "csv_tree") or not self.csv_tree:
            return

        end_index = min(start_index + BATCH_INSERT_SIZE, len(data))
        visible_cols = [c for c in COLUMN_ORDER if c in self._visible_columns]

        for i in range(start_index, end_index):
            term = data[i]
            values = []

            for col in visible_cols:
                config = COLUMN_REGISTRY.get(col, {})
                max_chars = config.get("max_chars", 50)
                data_key = DISPLAY_TO_DATA_COLUMN.get(col, col)

                raw_val = term.get(data_key, term.get(col, ""))
                display_val = truncate_text(str(raw_val), max_chars)
                values.append(display_val)

            # Insert row
            item_id = self.csv_tree.insert("", "end", values=values)

            # Store reference to original data
            self.csv_tree.set(item_id, "#0", str(i))

        # Schedule next batch if more data
        if end_index < len(data):
            self._async_insert_id = self.after(
                BATCH_INSERT_DELAY_MS,
                lambda: self._async_insert_rows(data, end_index),
            )
        else:
            self._async_insert_id = None
            logger.debug("Inserted %s rows", len(data))

    # =========================================================================
    # Treeview Setup
    # =========================================================================

    def _setup_treeview_columns(self, tree: ttk.Treeview):
        """
        Configure treeview columns based on visibility settings.

        Args:
            tree: Treeview widget to configure
        """
        visible_cols = [c for c in COLUMN_ORDER if c in self._visible_columns]

        tree["columns"] = visible_cols
        tree["displaycolumns"] = visible_cols
        tree["show"] = "headings"

        for col in visible_cols:
            COLUMN_REGISTRY.get(col, {})
            width = self._get_column_width(col)

            tree.heading(
                col,
                text=col,
                command=lambda c=col: self._sort_by_column(c),
            )
            tree.column(
                col,
                width=width,
                minwidth=40,
                anchor="w" if col == "Term" else "center",
            )

        # Bind right-click for column menu
        tree.bind("<Button-3>", self._on_header_right_click)

    def _on_header_right_click(self, event):
        """Handle right-click on treeview header area."""
        # Check if click is in header region (approximate)
        if event.y < 25:  # Header height
            self._show_column_menu(event)
