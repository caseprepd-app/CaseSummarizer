"""
Collapsible section widget for the Advanced settings tab.

Provides a clickable header that expands/collapses a group of setting
widgets. Each section includes a "Reset Section" button to restore
defaults for that section only.

Usage:
    section = CollapsibleSection(parent, title="ML Training", on_reset=my_reset_fn)
    section.pack(fill="x", pady=2)
    # Add widgets to section.content_frame
"""

import customtkinter as ctk

from src.ui.theme import FONTS


class CollapsibleSection(ctk.CTkFrame):
    """
    Collapsible section with header and expandable content.

    Clicking the header toggles visibility of the content area.
    All sections start collapsed.

    Attributes:
        content_frame: Frame to add child widgets into.
        is_expanded: Whether the section is currently showing content.
    """

    def __init__(self, parent, title: str, on_reset=None, **kwargs):
        """
        Initialize the collapsible section.

        Args:
            parent: Parent widget.
            title: Section header text.
            on_reset: Optional callback for "Reset Section" button.
        """
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self._title = title
        self._on_reset = on_reset
        self.is_expanded = False

        self._build_header()
        self._build_content()

    def _build_header(self):
        """Create the clickable header row with arrow, title, and reset button."""
        self._header_frame = ctk.CTkFrame(
            self,
            fg_color=("gray88", "gray22"),
            corner_radius=6,
            height=36,
        )
        self._header_frame.grid(row=0, column=0, sticky="ew", pady=(2, 0))
        self._header_frame.grid_columnconfigure(1, weight=1)
        self._header_frame.grid_propagate(True)

        # Arrow indicator
        self._arrow_label = ctk.CTkLabel(
            self._header_frame,
            text="\u25b6",
            font=FONTS["body"],
            width=20,
            anchor="center",
        )
        self._arrow_label.grid(row=0, column=0, padx=(8, 0), pady=6)

        # Title
        self._title_label = ctk.CTkLabel(
            self._header_frame,
            text=self._title,
            font=FONTS["heading_sm"],
            anchor="w",
        )
        self._title_label.grid(row=0, column=1, padx=(4, 10), pady=6, sticky="w")

        # Reset section button
        if self._on_reset:
            self._reset_btn = ctk.CTkButton(
                self._header_frame,
                text="Reset Section",
                command=self._on_reset,
                width=90,
                height=24,
                font=FONTS["small"],
                fg_color="transparent",
                border_width=1,
                text_color=("gray40", "gray70"),
                hover_color=("gray80", "gray30"),
            )
            self._reset_btn.grid(row=0, column=2, padx=(0, 8), pady=6)

        # Bind click to all header elements
        for widget in [self._header_frame, self._arrow_label, self._title_label]:
            widget.bind("<Button-1>", self._toggle)
            widget.configure(cursor="hand2")

    def _build_content(self):
        """Create the content frame (initially hidden)."""
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid_columnconfigure(0, weight=1)
        # Start collapsed - don't grid the content frame

    def _toggle(self, event=None):
        """Toggle the expanded/collapsed state."""
        if self.is_expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self):
        """Show the content frame."""
        self.is_expanded = True
        self._arrow_label.configure(text="\u25bc")
        self.content_frame.grid(row=1, column=0, sticky="ew", padx=(20, 0), pady=(4, 8))

    def collapse(self):
        """Hide the content frame."""
        self.is_expanded = False
        self._arrow_label.configure(text="\u25b6")
        self.content_frame.grid_forget()
