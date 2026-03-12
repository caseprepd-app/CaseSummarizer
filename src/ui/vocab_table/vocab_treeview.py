"""
Vocabulary Treeview Widget.

Self-contained treeview that bundles the ttk.Treeview widget with its own
item-to-data mapping, row insertion, feedback display, and tooltip logic.

Creating two instances (main + filtered) eliminates the Tk item-ID collision
bug: both treeviews generate IDs starting from I001, so every interaction
must look up the correct data dict. By owning its own dict, each instance
is self-contained and ID collisions are impossible.

NOTE: This file does not conform to the 200-line limit per user instruction —
the treeview logic is complex enough to warrant a single cohesive class.
"""

import logging
from tkinter import ttk

import customtkinter as ctk

from src.config import VF
from src.ui.theme import COLORS, VOCAB_TABLE_TAGS, resolve_tags
from src.ui.vocab_table.column_config import (
    COLUMN_CONFIG,
    DISPLAY_TO_DATA_COLUMN,
    THUMB_DOWN_EMPTY,
    THUMB_DOWN_FILLED,
    THUMB_UP_EMPTY,
    THUMB_UP_FILLED,
    truncate_text,
)

logger = logging.getLogger(__name__)

# Prefix added to potential-duplicate terms in the treeview display
_LINK_EMOJI_PREFIX = "\U0001f517 "


def strip_display_prefix(term: str) -> str:
    """Strip the potential-duplicate link emoji prefix from a display term."""
    if term.startswith(_LINK_EMOJI_PREFIX):
        return term[len(_LINK_EMOJI_PREFIX) :]
    return term


class VocabTreeview:
    """
    Self-contained vocabulary treeview with item-data mapping.

    Each instance owns its own treeview widget and item_to_data dict,
    eliminating the Tk item-ID collision bug between main and filtered tables.

    Args:
        parent: Parent frame to place the treeview in
        columns: Tuple of column names
        tag_prefix: "" for main table, "filtered_" for filtered table
        feedback_manager: FeedbackManager for rating lookups
        on_click_callback: Called on left-click with (event, self)
        on_right_click_callback: Called on right-click with (event, self)
    """

    def __init__(
        self,
        parent,
        columns: tuple,
        tag_prefix: str = "",
        feedback_manager=None,
        on_click_callback=None,
        on_right_click_callback=None,
    ):
        """Create the treeview widget with columns, scrollbars, and event bindings."""
        self._parent = parent
        self._tag_prefix = tag_prefix
        self._feedback_manager = feedback_manager
        self._on_click_callback = on_click_callback
        self._on_right_click_callback = on_right_click_callback
        self.item_to_data: dict[str, dict] = {}

        # Tooltip state
        self._tooltip = None
        self._tooltip_after_id = None

        # Create treeview
        self.widget = ttk.Treeview(
            parent,
            columns=columns,
            show="headings",
            style="Vocab.Treeview",
            selectmode="browse",
        )

        # Configure all vocabulary table tags from centralized theme
        for tag_name, tag_config in resolve_tags(VOCAB_TABLE_TAGS).items():
            self.widget.tag_configure(tag_name, **tag_config)

        # Bind events
        self.widget.bind("<Button-1>", self._on_click)
        self.widget.bind("<Button-3>", self._on_right_click)
        self.widget.bind("<Double-1>", self._on_double_click)
        self.widget.bind("<Motion>", self._on_hover)
        self.widget.bind("<Leave>", self._hide_tooltip)

    def configure_columns(self, columns, widths: dict | None = None):
        """
        Set heading text, widths, and anchor for each column.

        Args:
            columns: Tuple of column names
            widths: Optional dict of {col_name: pixel_width}
        """
        for col in columns:
            col_width = (widths or {}).get(col) or COLUMN_CONFIG.get(col, {}).get("width", 100)
            self.widget.heading(col, text=col, anchor="w")
            self.widget.column(col, width=col_width, minwidth=40, anchor="w", stretch=False)

    def configure_sortable_columns(self, columns, sort_callback):
        """
        Set heading commands for click-to-sort (main table only).

        Args:
            columns: Tuple of column names
            sort_callback: Called with column name when heading clicked
        """
        for col in columns:
            self.widget.heading(col, command=lambda c=col: sort_callback(c))

    def add_scrollbars(self, parent):
        """
        Add vertical and horizontal scrollbars to the treeview.

        Args:
            parent: Frame to place the scrollbars in

        Returns:
            Tuple of (vertical_scrollbar, horizontal_scrollbar)
        """
        vsb = ttk.Scrollbar(
            parent,
            orient="vertical",
            command=self.widget.yview,
            style="Vocab.Vertical.TScrollbar",
        )
        self.widget.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(
            parent,
            orient="horizontal",
            command=self.widget.xview,
            style="Vocab.Horizontal.TScrollbar",
        )
        self.widget.configure(xscrollcommand=hsb.set)

        return vsb, hsb

    def insert_row(self, item_data: dict, row_index: int, columns: tuple) -> str:
        """
        Insert a single row into the treeview.

        Builds the values tuple from item_data, applies feedback icons,
        alternating row tags, and rating color tags.

        Args:
            item_data: Term data dictionary
            row_index: Index for alternating row coloring
            columns: Current visible columns tuple

        Returns:
            The treeview item ID
        """
        term = item_data.get(VF.TERM, "")
        rating = self._feedback_manager.get_rating(term) if self._feedback_manager else 0

        values = self._build_values(item_data, columns, rating)
        tag = self._build_tags(term, row_index, rating)

        item_id = self.widget.insert("", "end", values=tuple(values), tags=tag)
        self.item_to_data[item_id] = item_data
        return item_id

    def _build_values(self, item_data: dict, columns: tuple, rating: int) -> list:
        """
        Build the display values list for a row.

        Args:
            item_data: Term data dictionary
            columns: Current visible columns
            rating: Current feedback rating (+1, -1, 0)

        Returns:
            List of display values
        """
        values = []
        for col in columns:
            if col == VF.KEEP:
                values.append(THUMB_UP_FILLED if rating == 1 else THUMB_UP_EMPTY)
            elif col == VF.SKIP:
                values.append(THUMB_DOWN_FILLED if rating == -1 else THUMB_DOWN_EMPTY)
            elif col == VF.TERM:
                term_display = item_data.get(VF.TERM, "")
                if item_data.get("_potential_duplicate_of"):
                    term_display = f"{_LINK_EMOJI_PREFIX}{term_display}"
                values.append(truncate_text(str(term_display), COLUMN_CONFIG[col]["max_chars"]))
            elif col in DISPLAY_TO_DATA_COLUMN:
                data_col = DISPLAY_TO_DATA_COLUMN[col]
                value = item_data.get(data_col, "")
                values.append(truncate_text(str(value), COLUMN_CONFIG[col]["max_chars"]))
            else:
                values.append(
                    truncate_text(str(item_data.get(col, "")), COLUMN_CONFIG[col]["max_chars"])
                )
        return values

    def _build_tags(self, term: str, row_index: int, rating: int) -> tuple:
        """
        Build the tag tuple for a row (alternating bg + rating color).

        Args:
            term: Term string for rating source lookup
            row_index: Index for alternating row coloring
            rating: Current feedback rating (+1, -1, 0)

        Returns:
            Tuple of tag strings
        """
        prefix = self._tag_prefix
        row_bg_tag = f"{prefix}oddrow" if row_index % 2 else f"{prefix}evenrow"

        if not self._feedback_manager or rating == 0:
            return (row_bg_tag,)

        rating_source = self._feedback_manager.get_rating_source(term)
        suffix = "session" if rating_source == "session" else "loaded"

        if rating == 1:
            return (row_bg_tag, f"{prefix}rated_up_{suffix}")
        elif rating == -1:
            return (row_bg_tag, f"{prefix}rated_down_{suffix}")
        return (row_bg_tag,)

    def get_item_data(self, item_id: str) -> dict:
        """
        Look up term data by item ID.

        Args:
            item_id: Treeview item identifier

        Returns:
            Term data dict, or empty dict if not found
        """
        return self.item_to_data.get(item_id, {})

    def has_item(self, item_id: str) -> bool:
        """Check if this treeview owns the given item ID."""
        return item_id in self.item_to_data

    def update_feedback_display(self, item_id: str, rating: int, columns: tuple):
        """
        Update the visual display of feedback icons for a term.

        Args:
            item_id: Treeview item identifier
            rating: +1 (Keep filled), -1 (Skip filled), 0 (both empty)
            columns: Current visible columns for index lookup
        """
        values = list(self.widget.item(item_id, "values"))

        try:
            keep_idx = list(columns).index(VF.KEEP)
            skip_idx = list(columns).index(VF.SKIP)
        except ValueError:
            return
        except Exception:
            logger.error("Unexpected error finding feedback columns", exc_info=True)
            return

        if len(values) <= max(keep_idx, skip_idx):
            return

        # Preserve the row background tag
        prefix = self._tag_prefix
        existing_tags = self.widget.item(item_id, "tags")
        bg_candidates = (f"{prefix}oddrow", f"{prefix}evenrow")
        row_bg_tag = next((t for t in existing_tags if t in bg_candidates), None)

        # Update icon values — user clicks always use session tags
        if rating == 1:
            values[keep_idx] = THUMB_UP_FILLED
            values[skip_idx] = THUMB_DOWN_EMPTY
            tag = (
                (row_bg_tag, f"{prefix}rated_up_session")
                if row_bg_tag
                else (f"{prefix}rated_up_session",)
            )
        elif rating == -1:
            values[keep_idx] = THUMB_UP_EMPTY
            values[skip_idx] = THUMB_DOWN_FILLED
            tag = (
                (row_bg_tag, f"{prefix}rated_down_session")
                if row_bg_tag
                else (f"{prefix}rated_down_session",)
            )
        else:
            values[keep_idx] = THUMB_UP_EMPTY
            values[skip_idx] = THUMB_DOWN_EMPTY
            tag = (row_bg_tag,) if row_bg_tag else ()

        self.widget.item(item_id, values=tuple(values), tags=tag)

    def clear(self):
        """Delete all rows and clear the item-to-data mapping."""
        self.widget.delete(*self.widget.get_children())
        self.item_to_data.clear()

    def destroy(self):
        """Destroy the treeview widget and clear all state."""
        self._hide_tooltip(None)
        self.item_to_data.clear()
        self.widget.destroy()

    # ─── Event handlers ───────────────────────────────────────────────

    def _on_click(self, event):
        """Delegate left-click to parent callback with self reference."""
        if self._on_click_callback:
            self._on_click_callback(event, self)

    def _on_right_click(self, event):
        """Delegate right-click to parent callback with self reference."""
        if self._on_right_click_callback:
            self._on_right_click_callback(event, self)

    def _on_double_click(self, event):
        """Handle double-click to copy the term to clipboard."""
        item_id = self.widget.identify_row(event.y)
        if not item_id:
            return
        values = self.widget.item(item_id, "values")
        if values and len(values) >= 1:
            term = strip_display_prefix(values[0])
            try:
                self.widget.clipboard_clear()
                self.widget.clipboard_append(term)
                logger.debug("Copied to clipboard: %s", term)
            except Exception:
                logger.error("Failed to copy to clipboard", exc_info=True)

    def _on_hover(self, event):
        """Show tooltip for potential-duplicate terms on hover."""
        # Cancel any pending tooltip
        if self._tooltip_after_id:
            self._parent.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None

        item_id = self.widget.identify_row(event.y)
        if not item_id:
            self._hide_tooltip(None)
            return

        term_data = self.item_to_data.get(item_id)
        if not term_data:
            self._hide_tooltip(None)
            return

        duplicate_of = term_data.get("_potential_duplicate_of")
        if not duplicate_of:
            self._hide_tooltip(None)
            return

        tooltip_text = f"Possible duplicate: {duplicate_of}"
        self._tooltip_after_id = self._parent.after(
            300,
            lambda: self._show_tooltip(tooltip_text, event.x_root + 15, event.y_root + 10),
        )

    def _show_tooltip(self, text: str, x: int, y: int):
        """
        Display a tooltip window at the specified screen coordinates.

        Args:
            text: Tooltip text
            x: Screen x-coordinate
            y: Screen y-coordinate
        """
        self._hide_tooltip(None)

        self._tooltip = ctk.CTkToplevel(self._parent)
        self._tooltip.wm_overrideredirect(True)
        self._tooltip.wm_geometry(f"+{x}+{y}")
        self._tooltip.attributes("-topmost", True)

        label = ctk.CTkLabel(
            self._tooltip,
            text=text,
            fg_color=COLORS["tooltip_bg"],
            text_color=COLORS["tooltip_fg"],
            corner_radius=4,
            padx=8,
            pady=4,
        )
        label.pack()

    def _hide_tooltip(self, _event):
        """Hide and destroy the tooltip window."""
        if self._tooltip_after_id:
            self._parent.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None
        if self._tooltip:
            self._tooltip.destroy()
            self._tooltip = None
