"""
Custom Log Categories Dialog for CasePrepd.

Modal dialog that lets users pick which log categories produce output
when the logging level is set to "Custom". Errors and warnings are
always logged regardless of selections.

Usage:
    from src.ui.logging_dialog import LoggingDialog
    LoggingDialog(parent=settings_dialog)
"""

import customtkinter as ctk

from src.logging_config import LOG_CATEGORIES, refresh_custom_log_filter
from src.ui.base_dialog import BaseModalDialog
from src.ui.theme import COLORS, FONTS
from src.user_preferences import get_user_preferences


class LoggingDialog(BaseModalDialog):
    """
    Modal dialog with checkboxes for each log category.

    Loads current states from user preferences and saves on OK.
    Errors & Warnings checkbox is always checked and disabled.

    Attributes:
        category_vars: Dict mapping category name -> BooleanVar.
    """

    def __init__(self, parent=None):
        """
        Initialize the logging categories dialog.

        Args:
            parent: Parent window (dialog is modal to this).
        """
        super().__init__(
            parent=parent,
            title="Custom Log Categories",
            width=450,
            height=520,
            min_width=380,
            min_height=420,
        )

        self.category_vars: dict[str, ctk.BooleanVar] = {}
        self._setup_ui()

    def _setup_ui(self):
        """Build the dialog UI with category checkboxes."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkLabel(
            self,
            text="Select which categories to log.",
            font=FONTS["heading_sm"],
            anchor="w",
        )
        header.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 0))

        subheader = ctk.CTkLabel(
            self,
            text="Errors and warnings are always logged.",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            anchor="w",
        )
        subheader.grid(row=1, column=0, sticky="w", padx=20, pady=(2, 10))

        # Scrollable checkbox area
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=2, column=0, sticky="nsew", padx=15, pady=0)
        self.grid_rowconfigure(2, weight=1)

        # Load current states
        prefs = get_user_preferences()
        saved_states = prefs.get_custom_log_categories()

        # Category checkboxes
        for idx, cat_name in enumerate(LOG_CATEGORIES):
            var = ctk.BooleanVar(value=saved_states.get(cat_name, True))
            self.category_vars[cat_name] = var

            cb = ctk.CTkCheckBox(
                scroll,
                text=cat_name,
                variable=var,
                font=FONTS["body"],
                checkbox_width=20,
                checkbox_height=20,
            )
            cb.grid(row=idx, column=0, sticky="w", pady=3, padx=5)

        # Disabled "Errors & Warnings" checkbox (always on)
        always_var = ctk.BooleanVar(value=True)
        always_cb = ctk.CTkCheckBox(
            scroll,
            text="Errors & Warnings (always on)",
            variable=always_var,
            font=FONTS["body"],
            checkbox_width=20,
            checkbox_height=20,
            state="disabled",
        )
        always_cb.grid(row=len(LOG_CATEGORIES), column=0, sticky="w", pady=(10, 3), padx=5)

        # Button bar
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 20))

        # Select All / Deselect All on the left
        ctk.CTkButton(
            btn_frame,
            text="Select All",
            command=self._select_all,
            width=90,
            height=28,
            font=FONTS["body"],
            fg_color="gray40",
            hover_color="gray30",
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            btn_frame,
            text="Deselect All",
            command=self._deselect_all,
            width=90,
            height=28,
            font=FONTS["body"],
            fg_color="gray40",
            hover_color="gray30",
        ).pack(side="left")

        # OK / Cancel on the right
        ctk.CTkButton(
            btn_frame,
            text="OK",
            command=self._on_ok,
            width=80,
            height=28,
        ).pack(side="right", padx=(5, 0))

        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=self.destroy,
            width=80,
            height=28,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
        ).pack(side="right")

    def _select_all(self):
        """Check all category checkboxes."""
        for var in self.category_vars.values():
            var.set(True)

    def _deselect_all(self):
        """Uncheck all category checkboxes."""
        for var in self.category_vars.values():
            var.set(False)

    def _on_ok(self):
        """Save selections to preferences and close."""
        states = {name: var.get() for name, var in self.category_vars.items()}

        prefs = get_user_preferences()
        prefs.set_custom_log_categories(states)
        refresh_custom_log_filter()

        self.destroy()
