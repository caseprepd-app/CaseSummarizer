"""
Indicator Pattern Widget for Vocabulary Settings.

Lets users define positive and negative indicator strings that become
ML features for the vocabulary preference learner. Users add simple
strings; the app auto-builds OR regexes. A "Show regex" toggle reveals
the generated pattern for advanced editing.

Layout:
    ┌─ Positive indicators ─────────────────────────────┐
    │  [text entry] [Add]                               │
    │  dr.  [x]   plaintiff  [x]   defendant  [x]      │
    ├─ Negative indicators ─────────────────────────────┤
    │  [text entry] [Add]                               │
    │  direct  [x]   redirect  [x]   cross  [x]        │
    ├─ Show regex ──────────────────────────────────────┤
    │  (?i)(?:dr\\.|plaintiff|defendant)    [Edit]      │
    └───────────────────────────────────────────────────┘
"""

import re

import customtkinter as ctk

from src.ui.theme import FONTS


def _build_regex_preview(strings: list[str]) -> str:
    """Build an OR regex preview from a list of strings."""
    escaped = [re.escape(s) for s in strings if s.strip()]
    if not escaped:
        return ""
    return f"(?i)(?:{'|'.join(escaped)})"


def _validate_regex(regex_str: str) -> str | None:
    """Validate a regex string. Returns None if valid, error message if not."""
    if not regex_str or not regex_str.strip():
        return None
    try:
        re.compile(regex_str.strip())
        return None
    except re.error as e:
        return str(e)


class IndicatorPatternWidget(ctk.CTkFrame):
    """
    Settings widget for user-defined vocabulary indicator patterns.

    Manages two string lists (positive/negative) with add/remove UI,
    optional regex preview/override, and triggers retrain on save.
    """

    def __init__(self, parent, **kwargs):
        """
        Initialize the indicator pattern widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        from src.user_preferences import get_user_preferences

        self._prefs = get_user_preferences()
        self._positive_strings: list[str] = list(self._prefs.get("vocab_positive_indicators", []))
        self._negative_strings: list[str] = list(self._prefs.get("vocab_negative_indicators", []))
        self._positive_override: str = self._prefs.get("vocab_positive_regex_override", "")
        self._negative_override: str = self._prefs.get("vocab_negative_regex_override", "")
        self._show_regex = False
        self._regex_error_label = None

        self._setup_ui()

    def _setup_ui(self):
        """Create the widget layout."""
        self.grid_columnconfigure(0, weight=1)
        row = 0

        # === Positive indicators section ===
        pos_label = ctk.CTkLabel(
            self,
            text="Positive indicators (terms to favor)",
            font=FONTS["body"],
            anchor="w",
        )
        pos_label.grid(row=row, column=0, sticky="w", pady=(0, 2))
        row += 1

        pos_input_frame = ctk.CTkFrame(self, fg_color="transparent")
        pos_input_frame.grid(row=row, column=0, sticky="ew", pady=(0, 2))
        pos_input_frame.grid_columnconfigure(0, weight=1)

        self._pos_entry = ctk.CTkEntry(pos_input_frame, placeholder_text="e.g. dr.")
        self._pos_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self._pos_entry.bind("<Return>", lambda e: self._add_positive())

        pos_add_btn = ctk.CTkButton(
            pos_input_frame,
            text="Add",
            width=50,
            command=self._add_positive,
        )
        pos_add_btn.grid(row=0, column=1)
        row += 1

        self._pos_tags_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._pos_tags_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        # === Negative indicators section ===
        neg_label = ctk.CTkLabel(
            self,
            text="Negative indicators (terms to demote)",
            font=FONTS["body"],
            anchor="w",
        )
        neg_label.grid(row=row, column=0, sticky="w", pady=(0, 2))
        row += 1

        neg_input_frame = ctk.CTkFrame(self, fg_color="transparent")
        neg_input_frame.grid(row=row, column=0, sticky="ew", pady=(0, 2))
        neg_input_frame.grid_columnconfigure(0, weight=1)

        self._neg_entry = ctk.CTkEntry(neg_input_frame, placeholder_text="e.g. redirect")
        self._neg_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self._neg_entry.bind("<Return>", lambda e: self._add_negative())

        neg_add_btn = ctk.CTkButton(
            neg_input_frame,
            text="Add",
            width=50,
            command=self._add_negative,
        )
        neg_add_btn.grid(row=0, column=1)
        row += 1

        self._neg_tags_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._neg_tags_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        # === Show regex toggle ===
        self._regex_toggle = ctk.CTkCheckBox(
            self,
            text="Show regex",
            font=FONTS["small"],
            command=self._toggle_regex,
        )
        self._regex_toggle.grid(row=row, column=0, sticky="w", pady=(0, 2))
        row += 1

        self._regex_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._regex_frame.grid(row=row, column=0, sticky="ew")
        self._regex_frame.grid_columnconfigure(0, weight=1)
        self._regex_row = row
        row += 1

        # Error label for regex validation
        self._regex_error_label = ctk.CTkLabel(
            self,
            text="",
            font=FONTS["small"],
            text_color="red",
            anchor="w",
        )
        self._regex_error_label.grid(row=row, column=0, sticky="w")
        row += 1

        # Help text
        help_label = ctk.CTkLabel(
            self,
            text=(
                "Add strings that indicate good or bad vocabulary terms.\n"
                "The ML model learns from your votes whether these patterns "
                "correlate with terms you keep or skip."
            ),
            font=FONTS["small"],
            text_color=("gray40", "gray60"),
            anchor="w",
            justify="left",
        )
        help_label.grid(row=row, column=0, sticky="ew")

        # Initial render
        self._render_tags()
        self._regex_frame.grid_remove()  # Hidden by default
        self._regex_error_label.grid_remove()

    def _add_positive(self):
        """Add a string to the positive indicators list."""
        text = self._pos_entry.get().strip()
        if text and text not in self._positive_strings:
            self._positive_strings.append(text)
            self._pos_entry.delete(0, "end")
            self._render_tags()
            self._update_regex_preview()

    def _add_negative(self):
        """Add a string to the negative indicators list."""
        text = self._neg_entry.get().strip()
        if text and text not in self._negative_strings:
            self._negative_strings.append(text)
            self._neg_entry.delete(0, "end")
            self._render_tags()
            self._update_regex_preview()

    def _remove_positive(self, text: str):
        """Remove a string from the positive indicators list."""
        if text in self._positive_strings:
            self._positive_strings.remove(text)
            self._render_tags()
            self._update_regex_preview()

    def _remove_negative(self, text: str):
        """Remove a string from the negative indicators list."""
        if text in self._negative_strings:
            self._negative_strings.remove(text)
            self._render_tags()
            self._update_regex_preview()

    def _render_tags(self):
        """Render tag chips for both positive and negative lists."""
        self._render_tag_list(self._pos_tags_frame, self._positive_strings, self._remove_positive)
        self._render_tag_list(self._neg_tags_frame, self._negative_strings, self._remove_negative)

    def _render_tag_list(self, parent_frame, strings: list[str], remove_fn):
        """
        Render removable tag chips in a flow layout.

        Args:
            parent_frame: Frame to render tags into.
            strings: List of indicator strings.
            remove_fn: Callback to remove a string.
        """
        for widget in parent_frame.winfo_children():
            widget.destroy()

        if not strings:
            return

        for i, text in enumerate(strings):
            tag = ctk.CTkButton(
                parent_frame,
                text=f"{text}  \u00d7",
                height=24,
                font=FONTS["small"],
                fg_color=("gray80", "gray30"),
                hover_color=("gray70", "gray40"),
                text_color=("gray10", "gray90"),
                command=lambda t=text: remove_fn(t),
            )
            tag.pack(side="left", padx=(0, 4), pady=2)

    def _toggle_regex(self):
        """Toggle regex preview visibility."""
        self._show_regex = not self._show_regex
        if self._show_regex:
            self._build_regex_preview()
            self._regex_frame.grid()
        else:
            self._regex_frame.grid_remove()
            self._regex_error_label.grid_remove()

    def _build_regex_preview(self):
        """Build the regex preview UI with edit capability."""
        for widget in self._regex_frame.winfo_children():
            widget.destroy()

        # Positive regex
        pos_label = ctk.CTkLabel(
            self._regex_frame,
            text="Positive regex:",
            font=FONTS["small"],
            anchor="w",
        )
        pos_label.grid(row=0, column=0, sticky="w", pady=(2, 0))

        pos_preview = self._positive_override or _build_regex_preview(self._positive_strings)
        self._pos_regex_entry = ctk.CTkEntry(
            self._regex_frame,
            font=FONTS.get("mono", FONTS["small"]),
        )
        self._pos_regex_entry.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        if pos_preview:
            self._pos_regex_entry.insert(0, pos_preview)
        if not self._positive_override:
            self._pos_regex_entry.configure(state="disabled")

        pos_edit_btn = ctk.CTkButton(
            self._regex_frame,
            text="Edit" if not self._positive_override else "Auto",
            width=50,
            font=FONTS["small"],
            command=self._toggle_positive_regex_edit,
        )
        pos_edit_btn.grid(row=1, column=1, padx=(4, 0))
        self._pos_edit_btn = pos_edit_btn

        # Negative regex
        neg_label = ctk.CTkLabel(
            self._regex_frame,
            text="Negative regex:",
            font=FONTS["small"],
            anchor="w",
        )
        neg_label.grid(row=2, column=0, sticky="w", pady=(2, 0))

        neg_preview = self._negative_override or _build_regex_preview(self._negative_strings)
        self._neg_regex_entry = ctk.CTkEntry(
            self._regex_frame,
            font=FONTS.get("mono", FONTS["small"]),
        )
        self._neg_regex_entry.grid(row=3, column=0, sticky="ew", pady=(0, 4))
        if neg_preview:
            self._neg_regex_entry.insert(0, neg_preview)
        if not self._negative_override:
            self._neg_regex_entry.configure(state="disabled")

        neg_edit_btn = ctk.CTkButton(
            self._regex_frame,
            text="Edit" if not self._negative_override else "Auto",
            width=50,
            font=FONTS["small"],
            command=self._toggle_negative_regex_edit,
        )
        neg_edit_btn.grid(row=3, column=1, padx=(4, 0))
        self._neg_edit_btn = neg_edit_btn

    def _toggle_positive_regex_edit(self):
        """Toggle between auto-generated and manual regex for positive."""
        if self._positive_override:
            # Switch to auto mode
            self._positive_override = ""
            self._build_regex_preview()
        else:
            # Switch to edit mode — copy current auto value as starting point
            self._positive_override = (
                self._pos_regex_entry.get() if hasattr(self, "_pos_regex_entry") else ""
            )
            self._build_regex_preview()

    def _toggle_negative_regex_edit(self):
        """Toggle between auto-generated and manual regex for negative."""
        if self._negative_override:
            self._negative_override = ""
            self._build_regex_preview()
        else:
            self._negative_override = (
                self._neg_regex_entry.get() if hasattr(self, "_neg_regex_entry") else ""
            )
            self._build_regex_preview()

    def _update_regex_preview(self):
        """Update regex preview if visible."""
        if self._show_regex:
            self._build_regex_preview()

    def get_value(self) -> dict:
        """
        Get the current indicator pattern configuration.

        Returns:
            Dict with positive/negative strings and regex overrides.
        """
        # Read regex overrides from entries if in edit mode
        pos_override = self._positive_override
        neg_override = self._negative_override
        if self._show_regex and pos_override and hasattr(self, "_pos_regex_entry"):
            pos_override = self._pos_regex_entry.get().strip()
        if self._show_regex and neg_override and hasattr(self, "_neg_regex_entry"):
            neg_override = self._neg_regex_entry.get().strip()

        return {
            "positive_strings": list(self._positive_strings),
            "negative_strings": list(self._negative_strings),
            "positive_override": pos_override,
            "negative_override": neg_override,
        }

    def set_value(self, value) -> None:
        """
        Set the indicator pattern configuration.

        Args:
            value: Dict or None. Ignored since we load from prefs in __init__.
        """
        # Values are loaded from preferences in __init__, so this is a no-op
        # unless called with explicit data
        if isinstance(value, dict):
            self._positive_strings = list(value.get("positive_strings", []))
            self._negative_strings = list(value.get("negative_strings", []))
            self._positive_override = value.get("positive_override", "")
            self._negative_override = value.get("negative_override", "")
            self._render_tags()
            self._update_regex_preview()

    def validate(self) -> str | None:
        """
        Validate regex overrides if set.

        Returns:
            None if valid, error message string if invalid.
        """
        data = self.get_value()
        if data["positive_override"]:
            err = _validate_regex(data["positive_override"])
            if err:
                return f"Invalid positive regex: {err}"
        if data["negative_override"]:
            err = _validate_regex(data["negative_override"])
            if err:
                return f"Invalid negative regex: {err}"
        return None
