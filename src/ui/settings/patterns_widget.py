"""
Custom Header/Footer Patterns Widget for Settings Dialog.

Simple text area widget for entering custom patterns to filter
from documents. One pattern per line.

Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │  Custom patterns (one per line)                          ⓘ  │
    ├─────────────────────────────────────────────────────────────┤
    │  SMITH & JONES LLP                                          │
    │  CERTIFIED COURT REPORTER                                   │
    │  ...                                                        │
    └─────────────────────────────────────────────────────────────┘
"""

import customtkinter as ctk

from src.ui.settings.base_settings_widget import BaseSettingsWidget
from src.ui.settings.settings_widgets import TooltipIcon
from src.ui.theme import FONTS


class CustomPatternsWidget(BaseSettingsWidget):
    """
    Widget for entering custom header/footer patterns.

    Patterns are stored as a newline-separated string in preferences.
    The HeaderFooterRemover will match these patterns (case-insensitive)
    in addition to its built-in patterns.
    """

    def __init__(self, parent, tooltip_text: str = "", **kwargs):
        """
        Initialize the custom patterns widget.

        Args:
            parent: Parent widget.
            tooltip_text: Help text for the tooltip.
            **kwargs: Additional CTkFrame arguments.
        """
        from src.user_preferences import get_user_preferences

        self._prefs = get_user_preferences()
        self._tooltip_text = tooltip_text

        # Load current value
        self._initial_value = self._prefs.get("custom_header_footer_patterns", "")

        super().__init__(parent, **kwargs)

    def _setup_ui(self):
        """Create the widget layout."""
        self.grid_columnconfigure(0, weight=1)

        # Header row
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        header_label = ctk.CTkLabel(
            header_frame,
            text="Custom patterns to remove (one per line)",
            font=FONTS["body"],
            anchor="w",
        )
        header_label.pack(side="left", padx=(0, 5))

        if self._tooltip_text:
            tooltip = TooltipIcon(header_frame, tooltip_text=self._tooltip_text)
            tooltip.pack(side="left")

        # Text area for patterns
        self._text_area = ctk.CTkTextbox(
            self,
            height=120,
            font=FONTS["mono"] if "mono" in FONTS else FONTS["body"],
            wrap="none",
        )
        self._text_area.grid(row=1, column=0, sticky="ew", pady=(0, 5))

        # Load initial value
        if self._initial_value:
            self._text_area.insert("1.0", self._initial_value)

        # Help text
        help_text = (
            "Enter text that repeats in headers/footers (firm names, "
            "reporter info, etc.).\nThese will be removed if they appear 3+ times."
        )
        help_label = ctk.CTkLabel(
            self,
            text=help_text,
            font=FONTS["small"],
            text_color=("gray40", "gray60"),
            anchor="w",
            justify="left",
        )
        help_label.grid(row=2, column=0, sticky="ew")

    def get_value(self) -> str:
        """
        Get the current patterns as a newline-separated string.

        Returns:
            Patterns string for storage.
        """
        text = self._text_area.get("1.0", "end-1c")
        # Clean up: remove empty lines and extra whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def set_value(self, value: str) -> None:
        """
        Set the patterns from a newline-separated string.

        Args:
            value: Patterns string from storage.
        """
        self._text_area.delete("1.0", "end")
        if value:
            self._text_area.insert("1.0", value)
