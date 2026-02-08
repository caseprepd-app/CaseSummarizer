"""
Inline Find Bar for CTkTextbox widgets.

Provides Ctrl+F search with match highlighting, prev/next navigation,
and match count display. Attaches to any CTkTextbox.
"""

import logging

import customtkinter as ctk

from src.ui.theme import BUTTON_STYLES, COLORS, FONTS

logger = logging.getLogger(__name__)

# Highlight tag styles applied to the target textbox
FIND_HIGHLIGHT = {"background": "#FFEB3B", "foreground": "#000000"}
FIND_CURRENT = {"background": "#FF9800", "foreground": "#000000"}


class TextFindBar(ctk.CTkFrame):
    """
    Inline find bar for CTkTextbox widgets.

    Args:
        master: Parent widget (the frame containing the textbox).
        textbox: The CTkTextbox to search within.

    Usage:
        find_bar = TextFindBar(parent_frame, my_textbox)
        find_bar.grid(row=0, column=0, sticky="ew")
        find_bar.grid_remove()  # Hidden initially
        find_bar.show()         # Show on Ctrl+F
    """

    def __init__(self, master, textbox: ctk.CTkTextbox, **kwargs):
        super().__init__(master, fg_color=COLORS["bg_dark"], **kwargs)

        self._textbox = textbox
        self._matches: list[str] = []  # List of text indices for match starts
        self._current_idx = -1
        self._debounce_id = None

        # Configure highlight tags on the target textbox
        self._textbox.tag_config("find_highlight", cnf=FIND_HIGHLIGHT)
        self._textbox.tag_config("find_current", cnf=FIND_CURRENT)

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self._build_ui()

    def _build_ui(self):
        """Build the find bar UI elements."""
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=4, pady=3)

        # Search entry
        self._entry = ctk.CTkEntry(bar, placeholder_text="Find...", font=FONTS["body"], width=200)
        self._entry.pack(side="left", padx=(0, 4))
        self._entry.bind("<KeyRelease>", self._on_key)
        self._entry.bind("<Return>", lambda e: self._next_match())
        self._entry.bind("<Escape>", lambda e: self.hide())

        # Match count label
        self._count_label = ctk.CTkLabel(
            bar,
            text="",
            font=FONTS["small"],
            width=70,
            text_color=COLORS["text_secondary"],
        )
        self._count_label.pack(side="left", padx=(0, 4))

        # Prev button
        prev_btn = ctk.CTkButton(
            bar,
            text="Prev",
            width=50,
            command=self._prev_match,
            **BUTTON_STYLES["secondary"],
        )
        prev_btn.pack(side="left", padx=(0, 2))

        # Next button
        next_btn = ctk.CTkButton(
            bar,
            text="Next",
            width=50,
            command=self._next_match,
            **BUTTON_STYLES["secondary"],
        )
        next_btn.pack(side="left", padx=(0, 4))

        # Close button
        close_btn = ctk.CTkButton(
            bar,
            text="X",
            width=30,
            command=self.hide,
            **BUTTON_STYLES["tertiary"],
        )
        close_btn.pack(side="right")

    def show(self):
        """Show the find bar and focus the entry."""
        self.grid()
        self._entry.focus_set()
        self._entry.select_range(0, "end")

    def hide(self):
        """Hide the find bar and clear highlights."""
        self._clear_highlights()
        self._count_label.configure(text="")
        self.grid_remove()
        self._matches = []
        self._current_idx = -1

    def _on_key(self, event):
        """Handle keystrokes with debounce."""
        if event.keysym == "Escape":
            return
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(150, self._do_search)

    def _do_search(self):
        """Search for all matches and highlight them."""
        self._debounce_id = None
        self._clear_highlights()
        self._matches = []
        self._current_idx = -1

        query = self._entry.get().strip()
        if not query:
            self._count_label.configure(text="")
            return

        # Access underlying tkinter Text widget
        text_widget = self._textbox._textbox
        content = text_widget.get("1.0", "end-1c")
        query_lower = query.lower()
        content_lower = content.lower()

        # Find all match positions
        start = 0
        while True:
            pos = content_lower.find(query_lower, start)
            if pos == -1:
                break
            # Convert character offset to tkinter index
            line = content[:pos].count("\n") + 1
            col = pos - content[:pos].rfind("\n") - 1
            tk_index = f"{line}.{col}"
            self._matches.append(tk_index)
            # Add highlight tag
            end_index = f"{tk_index}+{len(query)}c"
            text_widget.tag_add("find_highlight", tk_index, end_index)
            start = pos + 1

        total = len(self._matches)
        if total > 0:
            self._current_idx = 0
            self._highlight_current()
            self._count_label.configure(text=f"1 of {total}")
        else:
            self._count_label.configure(text="No matches")

        logger.debug("Find: '%s' → %d matches", query, total)

    def _next_match(self):
        """Move to the next match."""
        if not self._matches:
            return
        self._current_idx = (self._current_idx + 1) % len(self._matches)
        self._highlight_current()

    def _prev_match(self):
        """Move to the previous match."""
        if not self._matches:
            return
        self._current_idx = (self._current_idx - 1) % len(self._matches)
        self._highlight_current()

    def _highlight_current(self):
        """Highlight the current match in orange and scroll to it."""
        text_widget = self._textbox._textbox
        query_len = len(self._entry.get().strip())

        # Remove previous current highlight
        text_widget.tag_remove("find_current", "1.0", "end")

        if self._current_idx < 0 or self._current_idx >= len(self._matches):
            return

        idx = self._matches[self._current_idx]
        end_idx = f"{idx}+{query_len}c"
        text_widget.tag_add("find_current", idx, end_idx)

        # Ensure current highlight renders on top of yellow
        text_widget.tag_raise("find_current", "find_highlight")

        # Scroll to current match
        text_widget.see(idx)

        total = len(self._matches)
        self._count_label.configure(text=f"{self._current_idx + 1} of {total}")

    def _clear_highlights(self):
        """Remove all find highlights from the textbox."""
        text_widget = self._textbox._textbox
        text_widget.tag_remove("find_highlight", "1.0", "end")
        text_widget.tag_remove("find_current", "1.0", "end")
