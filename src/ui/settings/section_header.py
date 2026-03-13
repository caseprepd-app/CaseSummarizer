"""
Reusable section header widget with optional tooltip.

Replaces the repeated pattern of creating a transparent frame, label,
and TooltipIcon found in every settings widget.

Usage:
    header = SectionHeader(parent, "Default Searches", tooltip_text="...")
    header.grid(row=0, column=0, sticky="ew", pady=(0, 5))
"""

import customtkinter as ctk

from src.ui.settings.settings_widgets import TooltipIcon
from src.ui.theme import FONTS


class SectionHeader(ctk.CTkFrame):
    """Transparent frame containing a title label and optional tooltip icon.

    Attributes:
        label: The CTkLabel displaying the title text.
        tooltip: The TooltipIcon, if tooltip_text was provided.
    """

    def __init__(
        self,
        parent,
        title: str,
        *,
        tooltip_text: str = "",
        font: tuple | None = None,
        **kwargs,
    ):
        """Create a section header.

        Args:
            parent: Parent widget.
            title: Header text.
            tooltip_text: If non-empty, a TooltipIcon is added beside the label.
            font: Override font; defaults to FONTS["heading_sm"].
            **kwargs: Extra CTkFrame arguments.
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        resolved_font = font or FONTS["heading_sm"]

        self.label = ctk.CTkLabel(
            self,
            text=title,
            font=resolved_font,
            anchor="w",
        )
        self.label.pack(side="left", padx=(0, 5))

        self.tooltip = None
        if tooltip_text:
            self.tooltip = TooltipIcon(self, tooltip_text=tooltip_text)
            self.tooltip.pack(side="left")
