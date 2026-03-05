"""
Column Visibility Widget for Settings Dialog.

Widget for configuring default column visibility in vocabulary table.

Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │  Default Visible Columns                                 ⓘ  │
    ├─────────────────────────────────────────────────────────────┤
    │  Basic:                                                     │
    │  ☑ Term (required)   ☑ Score   ☑ Is Person   ☑ Found By    │
    │                                                             │
    │  Term Sources:                                              │
    │  ☑ # Docs   ☑ Count   ☑ OCR Confidence                        │
    │                                                             │
    │  Algorithm Details:                                         │
    │  ☐ NER  ☐ RAKE  ☐ BM25  ☐ TopicRank  ☐ MedicalNER         │
    │  ☐ YAKE  ☐ Algo Count                                     │
    │                                                             │
    │  Other:                                                     │
    │  ☐ Google Rarity Rank   ☑ Keep   ☑ Skip                             │
    ├─────────────────────────────────────────────────────────────┤
    │  [Reset to Defaults]                                        │
    └─────────────────────────────────────────────────────────────┘
"""

from typing import ClassVar

import customtkinter as ctk

from src.config import VF
from src.ui.settings.settings_widgets import TooltipIcon
from src.ui.theme import COLORS, FONTS


class ColumnVisibilityWidget(ctk.CTkFrame):
    """
    Widget for configuring default column visibility in vocabulary table.

    Provides grouped checkboxes for selecting which columns appear by default.
    Users can also toggle columns via right-click on the table header in the
    main UI.
    """

    # Column groups for organized display
    COLUMN_GROUPS: ClassVar[list[tuple[str, list[str]]]] = [
        ("Basic", [VF.TERM, "Score", VF.IS_PERSON, VF.FOUND_BY]),
        ("Term Sources", [VF.NUM_DOCS, VF.OCCURRENCES, VF.OCR_CONFIDENCE]),
        (
            "Algorithm Details",
            [
                VF.NER,
                VF.RAKE,
                VF.BM25,
                VF.TOPICRANK,
                VF.MEDICALNER,
                VF.YAKE,
                VF.ALGO_COUNT,
            ],
        ),
        ("Other", [VF.GOOGLE_RARITY_RANK, VF.KEEP, VF.SKIP]),
    ]

    def __init__(self, parent, **kwargs):
        """
        Initialize the column visibility widget.

        Args:
            parent: Parent widget.
            **kwargs: Additional CTkFrame arguments.
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._checkboxes: dict[str, tuple[ctk.CTkCheckBox, ctk.BooleanVar]] = {}
        self._setup_ui()
        self._load_values()

    def _setup_ui(self):
        """Create the widget layout."""
        from src.ui.dynamic_output import COLUMN_REGISTRY

        self.grid_columnconfigure(0, weight=1)

        # Header with tooltip
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(header_frame, text="Default Visible Columns", font=FONTS["heading_sm"]).pack(
            side="left"
        )

        tooltip_icon = TooltipIcon(
            header_frame,
            tooltip_text=(
                "Choose which columns appear by default in the vocabulary table.\n\n"
                "You can also toggle columns via:\n"
                "• Right-click on any column header\n"
                "• 'Columns...' button below the table\n\n"
                "Changes are saved automatically and persist between sessions."
            ),
        )
        tooltip_icon.pack(side="left", padx=(5, 0))

        # Create checkbox groups
        row_idx = 1
        for group_name, columns in self.COLUMN_GROUPS:
            # Group label
            group_label = ctk.CTkLabel(
                self,
                text=f"{group_name}:",
                font=FONTS["small"],
                text_color=COLORS["text_secondary"],
            )
            group_label.grid(row=row_idx, column=0, sticky="w", pady=(10, 2))
            row_idx += 1

            # Checkbox row
            checkbox_frame = ctk.CTkFrame(self, fg_color="transparent")
            checkbox_frame.grid(row=row_idx, column=0, sticky="w", padx=(10, 0))
            row_idx += 1

            for col_name in columns:
                col_config = COLUMN_REGISTRY.get(col_name, {})
                var = ctk.BooleanVar(value=col_config.get("default", False))

                cb = ctk.CTkCheckBox(
                    checkbox_frame,
                    text=col_name,
                    variable=var,
                    command=self._on_checkbox_change,
                    width=100,
                    checkbox_width=18,
                    checkbox_height=18,
                    font=FONTS["small"],
                )
                cb.pack(side="left", padx=(0, 15))

                # Disable "Term" checkbox (required column)
                if not col_config.get("can_hide", True):
                    cb.configure(state="disabled")
                    var.set(True)  # Always checked

                self._checkboxes[col_name] = (cb, var)

        # Reset button
        reset_frame = ctk.CTkFrame(self, fg_color="transparent")
        reset_frame.grid(row=row_idx, column=0, sticky="w", pady=(15, 0))

        reset_btn = ctk.CTkButton(
            reset_frame,
            text="Reset to Defaults",
            command=self._reset_to_defaults,
            width=120,
            height=28,
            font=FONTS["small"],
        )
        reset_btn.pack(side="left")

    def _load_values(self):
        """Load current visibility settings from preferences."""
        from src.ui.dynamic_output import COLUMN_REGISTRY
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        saved = prefs.get("vocab_column_visibility", {})

        for col_name, (_cb, var) in self._checkboxes.items():
            # Get default from registry if not saved
            default = COLUMN_REGISTRY.get(col_name, {}).get("default", False)
            value = saved.get(col_name, default)
            var.set(value)

    def _on_checkbox_change(self):
        """Handle checkbox state change (persisted on dialog Save)."""
        pass

    def _save_values(self):
        """Save current checkbox states to preferences."""
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        visibility = {col: var.get() for col, (cb, var) in self._checkboxes.items()}
        prefs.set("vocab_column_visibility", visibility)

    def _reset_to_defaults(self):
        """Reset all checkboxes to their default values."""
        from src.ui.dynamic_output import COLUMN_REGISTRY

        for col_name, (_cb, var) in self._checkboxes.items():
            default = COLUMN_REGISTRY.get(col_name, {}).get("default", False)
            var.set(default)

    def get_value(self) -> dict[str, bool]:
        """Return current checkbox states."""
        return {col: var.get() for col, (cb, var) in self._checkboxes.items()}

    def set_value(self, value: dict[str, bool]):
        """Set checkbox states from dict."""
        if not value:
            return
        for col, is_visible in value.items():
            if col in self._checkboxes:
                _cb, var = self._checkboxes[col]
                var.set(is_visible)
