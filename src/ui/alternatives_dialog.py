"""
Alternatives Dialog for Person Names.

Shows the chosen canonical name, the reason it was selected,
and all rejected variants with their rejection reasons.

Opened via right-click "View Alternatives" on Person names
in the vocabulary table.
"""

import customtkinter as ctk

from src.ui.base_dialog import BaseModalDialog
from src.ui.theme import COLORS


class AlternativesDialog(BaseModalDialog):
    """
    Modal dialog showing canonical name alternatives.

    Displays the chosen name with its selection reason,
    followed by a list of rejected variants with reasons.

    All data arrives as plain dict keys — no core imports needed.

    Attributes:
        term_data: Dict containing '_alternatives' and '_canonical_reason' keys
    """

    def __init__(self, parent, term_data: dict):
        """
        Initialize the alternatives dialog.

        Args:
            parent: Parent window
            term_data: Term dict with keys:
                - Term: The canonical name string
                - Occurrences: Occurrence count
                - _canonical_reason: Why this name was chosen
                - _alternatives: List of dicts with 'variant', 'reason', 'frequency'
        """
        self._term_data = term_data
        term_name = term_data.get("Term", "Unknown")

        super().__init__(
            parent=parent,
            title=f'Alternatives for "{term_name}"',
            width=520,
            height=400,
            min_width=400,
            min_height=300,
        )
        self._create_ui()

    def _create_ui(self):
        """Build the dialog content."""
        term = self._term_data.get("Term", "Unknown")
        freq = self._term_data.get("Occurrences", 0)
        reason = self._term_data.get("_canonical_reason", "Selected as canonical variant")
        alternatives = self._term_data.get("_alternatives", [])

        # Scrollable frame for content
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=16, pady=(12, 8))

        # --- Chosen name section ---
        chosen_label = ctk.CTkLabel(
            scroll,
            text="CHOSEN",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["dialog_chosen"],
            anchor="w",
        )
        chosen_label.pack(fill="x", pady=(0, 4))

        name_label = ctk.CTkLabel(
            scroll,
            text=term,
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        )
        name_label.pack(fill="x")

        reason_label = ctk.CTkLabel(
            scroll,
            text=reason,
            font=ctk.CTkFont(size=12),
            text_color=COLORS["dialog_subtitle"],
            anchor="w",
        )
        reason_label.pack(fill="x", pady=(2, 0))

        freq_label = ctk.CTkLabel(
            scroll,
            text=f"{freq} occurrences",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["dialog_muted"],
            anchor="w",
        )
        freq_label.pack(fill="x", pady=(2, 8))

        # --- Rejected variants section ---
        if alternatives:
            sep = ctk.CTkFrame(scroll, height=1, fg_color=COLORS["dialog_separator"])
            sep.pack(fill="x", pady=(4, 8))

            count_text = f"{len(alternatives)} Rejected Variant"
            if len(alternatives) != 1:
                count_text += "s"

            header = ctk.CTkLabel(
                scroll,
                text=count_text,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=COLORS["dialog_rejected"],
                anchor="w",
            )
            header.pack(fill="x", pady=(0, 8))

            for alt in alternatives:
                self._add_variant_block(scroll, alt)
        else:
            no_alts = ctk.CTkLabel(
                scroll,
                text="No rejected variants recorded.",
                font=ctk.CTkFont(size=12),
                text_color=COLORS["dialog_muted"],
                anchor="w",
            )
            no_alts.pack(fill="x", pady=(8, 0))

        # --- Close button ---
        close_btn = ctk.CTkButton(
            self,
            text="Close",
            width=100,
            command=self.close,
        )
        close_btn.pack(pady=(4, 12))

    def _add_variant_block(self, parent_frame, alt: dict):
        """
        Add a rejected variant block to the dialog.

        Args:
            parent_frame: Frame to add the block to
            alt: Dict with 'variant', 'reason', 'frequency'
        """
        variant = alt.get("variant", "")
        reason = alt.get("reason", "")
        freq = alt.get("frequency", 0)

        variant_label = ctk.CTkLabel(
            parent_frame,
            text=f'"{variant}"',
            font=ctk.CTkFont(size=13),
            anchor="w",
        )
        variant_label.pack(fill="x", pady=(4, 0))

        # Split reason by semicolons for bullet points
        reason_parts = [r.strip() for r in reason.split(";") if r.strip()]
        for part in reason_parts:
            bullet = ctk.CTkLabel(
                parent_frame,
                text=f"  \u2022 {part}",
                font=ctk.CTkFont(size=11),
                text_color=COLORS["dialog_subtitle"],
                anchor="w",
            )
            bullet.pack(fill="x")

        freq_label = ctk.CTkLabel(
            parent_frame,
            text=f"  \u2022 Appeared {freq} times",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["dialog_muted"],
            anchor="w",
        )
        freq_label.pack(fill="x", pady=(0, 6))
