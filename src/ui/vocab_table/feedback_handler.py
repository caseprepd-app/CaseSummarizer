"""
Vocabulary Feedback Handler Mixin.

Session 82: Extracted from dynamic_output.py for modularity.

Contains:
- Feedback UI (Keep/Skip buttons in table)
- Term exclusion list management
- Context menu for term actions
"""

import tkinter as tk
from tkinter import ttk, messagebox

from src.logging_config import debug_log
from src.ui.vocab_table.column_config import (
    THUMB_UP_EMPTY,
    THUMB_UP_FILLED,
    THUMB_DOWN_EMPTY,
    THUMB_DOWN_FILLED,
)


class FeedbackHandlerMixin:
    """
    Mixin class providing vocabulary feedback functionality.

    Methods in this mixin assume the parent class has:
    - self.csv_tree: ttk.Treeview widget
    - self._vocab_csv_data: List of vocabulary dictionaries
    - self._feedback_state: Dict mapping term -> "keep" | "skip" | None
    """

    def _setup_feedback_state(self):
        """Initialize feedback state tracking."""
        self._feedback_state = {}  # term -> "keep" | "skip" | None

    def _on_tree_click(self, event):
        """
        Handle click events on treeview for feedback buttons.

        Detects clicks on Keep/Skip columns and toggles feedback state.
        """
        if not hasattr(self, "csv_tree") or not self.csv_tree:
            return

        # Identify clicked region
        region = self.csv_tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        column = self.csv_tree.identify_column(event.x)
        item = self.csv_tree.identify_row(event.y)

        if not item or not column:
            return

        # Get column name from column ID (#1, #2, etc.)
        try:
            col_index = int(column.replace("#", "")) - 1
            visible_cols = list(self.csv_tree["columns"])
            if col_index < 0 or col_index >= len(visible_cols):
                return
            col_name = visible_cols[col_index]
        except (ValueError, IndexError):
            return

        # Handle Keep/Skip clicks
        if col_name == "Keep":
            self._toggle_feedback(item, "keep")
        elif col_name == "Skip":
            self._toggle_feedback(item, "skip")

    def _toggle_feedback(self, item_id: str, feedback_type: str):
        """
        Toggle feedback state for a vocabulary term.

        Args:
            item_id: Treeview item ID
            feedback_type: "keep" or "skip"
        """
        values = list(self.csv_tree.item(item_id, "values"))
        if not values:
            return

        # Get term (first column)
        term = values[0]

        # Get current state
        current_state = self._feedback_state.get(term)

        # Toggle logic: click same button clears, click different button sets
        if current_state == feedback_type:
            new_state = None
        else:
            new_state = feedback_type

        self._feedback_state[term] = new_state

        # Update display
        self._update_feedback_display(item_id, new_state)

        debug_log(f"[Feedback] Term '{term}': {current_state} -> {new_state}")

    def _update_feedback_display(self, item_id: str, state: str | None):
        """
        Update the Keep/Skip column display for an item.

        Args:
            item_id: Treeview item ID
            state: "keep", "skip", or None
        """
        values = list(self.csv_tree.item(item_id, "values"))
        visible_cols = list(self.csv_tree["columns"])

        # Find Keep/Skip column indices
        keep_idx = visible_cols.index("Keep") if "Keep" in visible_cols else -1
        skip_idx = visible_cols.index("Skip") if "Skip" in visible_cols else -1

        # Update values
        if keep_idx >= 0:
            values[keep_idx] = THUMB_UP_FILLED if state == "keep" else THUMB_UP_EMPTY
        if skip_idx >= 0:
            values[skip_idx] = THUMB_DOWN_FILLED if state == "skip" else THUMB_DOWN_EMPTY

        self.csv_tree.item(item_id, values=values)

        # Apply visual tags
        if state == "keep":
            self.csv_tree.item(item_id, tags=("keep",))
        elif state == "skip":
            self.csv_tree.item(item_id, tags=("skip",))
        else:
            self.csv_tree.item(item_id, tags=())

    def _setup_feedback_tags(self, tree: ttk.Treeview):
        """
        Configure treeview tags for feedback styling.

        Args:
            tree: Treeview widget to configure
        """
        # Note: ttk.Treeview doesn't support foreground/background via tags
        # on Windows reliably. Using styled row tags for future styling.
        try:
            tree.tag_configure("keep", foreground="#228B22")  # Forest green
            tree.tag_configure("skip", foreground="#CD5C5C")  # Indian red
        except tk.TclError:
            pass  # Tags may not be fully supported

    # =========================================================================
    # Context Menu
    # =========================================================================

    def _show_context_menu(self, event):
        """
        Show context menu for selected vocabulary term.

        Provides options like:
        - Add to exclusion list
        - Copy term
        - Mark as keep/skip
        """
        if not hasattr(self, "csv_tree") or not self.csv_tree:
            return

        # Identify clicked item
        item = self.csv_tree.identify_row(event.y)
        if not item:
            return

        # Select the clicked item
        self.csv_tree.selection_set(item)

        # Get term value
        values = self.csv_tree.item(item, "values")
        if not values:
            return
        term = values[0]

        # Create context menu
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label=f"Copy '{term}'", command=lambda: self._copy_term(term))
        menu.add_separator()
        menu.add_command(
            label="Add to Exclusion List",
            command=lambda: self._add_to_exclusion_list(term, item),
        )
        menu.add_separator()

        # Feedback options
        current_state = self._feedback_state.get(term)
        if current_state != "keep":
            menu.add_command(
                label="Mark as Keep",
                command=lambda: self._set_feedback(item, "keep"),
            )
        if current_state != "skip":
            menu.add_command(
                label="Mark as Skip",
                command=lambda: self._set_feedback(item, "skip"),
            )
        if current_state:
            menu.add_command(
                label="Clear Feedback",
                command=lambda: self._set_feedback(item, None),
            )

        menu.tk_popup(event.x_root, event.y_root)

    def _copy_term(self, term: str):
        """Copy term to clipboard."""
        self.clipboard_clear()
        self.clipboard_append(term)
        debug_log(f"[Feedback] Copied to clipboard: {term}")

    def _set_feedback(self, item_id: str, state: str | None):
        """
        Set feedback state directly (from context menu).

        Args:
            item_id: Treeview item ID
            state: "keep", "skip", or None
        """
        values = self.csv_tree.item(item_id, "values")
        if not values:
            return

        term = values[0]
        self._feedback_state[term] = state
        self._update_feedback_display(item_id, state)

    # =========================================================================
    # Exclusion List
    # =========================================================================

    def _add_to_exclusion_list(self, term: str, item_id: str | None = None):
        """
        Add term to user exclusion list and remove from display.

        Args:
            term: Term to exclude
            item_id: Optional treeview item ID to remove
        """
        from src.config import USER_VOCAB_EXCLUDE_PATH

        # Confirm with user
        result = messagebox.askyesno(
            "Add to Exclusion List",
            f"Add '{term}' to your permanent exclusion list?\n\n"
            "This term will be hidden from future extractions.",
        )

        if not result:
            return

        try:
            # Ensure file exists
            USER_VOCAB_EXCLUDE_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Read existing terms
            existing = set()
            if USER_VOCAB_EXCLUDE_PATH.exists():
                with open(USER_VOCAB_EXCLUDE_PATH, "r", encoding="utf-8") as f:
                    existing = {line.strip().lower() for line in f if line.strip()}

            # Add new term
            term_lower = term.strip().lower()
            if term_lower not in existing:
                with open(USER_VOCAB_EXCLUDE_PATH, "a", encoding="utf-8") as f:
                    f.write(f"{term_lower}\n")
                debug_log(f"[Feedback] Added to exclusion list: {term}")

            # Remove from display
            if item_id:
                self.csv_tree.delete(item_id)

            # Remove from data
            if hasattr(self, "_vocab_csv_data"):
                self._vocab_csv_data = [
                    v for v in self._vocab_csv_data if v.get("Term", "").lower() != term_lower
                ]

        except Exception as e:
            debug_log(f"[Feedback] Error adding to exclusion list: {e}")
            messagebox.showerror("Error", f"Failed to add term to exclusion list: {e}")

    def _get_feedback_summary(self) -> dict:
        """
        Get summary of feedback for export/saving.

        Returns:
            Dict with "keep" and "skip" lists of terms
        """
        keep_terms = []
        skip_terms = []

        for term, state in self._feedback_state.items():
            if state == "keep":
                keep_terms.append(term)
            elif state == "skip":
                skip_terms.append(term)

        return {"keep": keep_terms, "skip": skip_terms}

    def _clear_feedback(self):
        """Clear all feedback state."""
        self._feedback_state.clear()

        # Update display
        if hasattr(self, "csv_tree") and self.csv_tree:
            for item_id in self.csv_tree.get_children():
                self._update_feedback_display(item_id, None)
