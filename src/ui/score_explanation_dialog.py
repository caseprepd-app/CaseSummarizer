"""
Score Explanation Dialog for Vocabulary Terms.

Shows a plain-English breakdown of why the scoring system rated a term
the way it did. Combines factors from all three model components:
- Rules: which quality rules fired for this term
- LR: Logistic Regression per-feature contributions
- RF: Random Forest important features (when ensemble is active)

Opened via right-click "Why this score?" on a term in the vocabulary table.
"""

import logging

import customtkinter as ctk

from src.ui.base_dialog import BaseModalDialog
from src.ui.theme import COLORS

logger = logging.getLogger(__name__)

# Muted colors for source tags — lighter than the +/- indicators
_SOURCE_TAG_COLOR = COLORS.get("dialog_muted", "#888888")


class ScoreExplanationDialog(BaseModalDialog):
    """
    Modal dialog showing per-feature score contributions.

    Displays the ML score, overall direction (keep/skip), and the top
    contributing factors from each model component with colored +/-
    indicators and source tags [LR], [RF], [Rules].

    Attributes:
        _explanation: Dict from score_explainer.explain_score()
        _term_name: Display name of the term
    """

    def __init__(self, parent, term_name: str, explanation: dict):
        """
        Initialize the score explanation dialog.

        Args:
            parent: Parent window
            term_name: The term being explained
            explanation: Dict with keys: score, direction, reasons, model_status
        """
        self._explanation = explanation
        self._term_name = term_name

        super().__init__(
            parent=parent,
            title=f'Score Breakdown: "{term_name}"',
            width=480,
            height=400,
            min_width=400,
            min_height=300,
        )
        self._create_ui()

    def _create_ui(self):
        """Build the dialog content."""
        explanation = self._explanation
        score = explanation["score"]
        direction = explanation["direction"]
        reasons = explanation["reasons"]

        # Scrollable content area
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=16, pady=(12, 8))

        # --- Score summary ---
        self._build_score_header(scroll, score, direction)

        # --- Top factors ---
        if reasons:
            self._build_factors_section(scroll, reasons)
        else:
            no_reasons = ctk.CTkLabel(
                scroll,
                text="No significant factors identified.",
                font=ctk.CTkFont(size=12),
                text_color=COLORS["dialog_muted"],
                anchor="w",
            )
            no_reasons.pack(fill="x", pady=(8, 0))

        # --- Model info footer ---
        model_text = (
            "Based on Logistic Regression + Rules"
            if explanation["model_status"] == "lr"
            else "Based on LR + Random Forest + Rules"
        )
        model_label = ctk.CTkLabel(
            scroll,
            text=model_text,
            font=ctk.CTkFont(size=10),
            text_color=COLORS["dialog_muted"],
            anchor="w",
        )
        model_label.pack(fill="x", pady=(12, 0))

        # --- Close button ---
        close_btn = ctk.CTkButton(self, text="Close", width=100, command=self.close)
        close_btn.pack(pady=(4, 12))

    def _build_score_header(self, parent: ctk.CTkScrollableFrame, score: float, direction: str):
        """Build the score value and keep/skip direction labels."""
        score_pct = score * 100
        if direction == "keep":
            direction_text = "Leaning toward KEEP"
            direction_color = COLORS["dialog_chosen"]
        else:
            direction_text = "Leaning toward SKIP"
            direction_color = COLORS["dialog_rejected"]

        score_label = ctk.CTkLabel(
            parent,
            text=f"ML Score: {score_pct:.0f}",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        )
        score_label.pack(fill="x", pady=(0, 2))

        direction_label = ctk.CTkLabel(
            parent,
            text=direction_text,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=direction_color,
            anchor="w",
        )
        direction_label.pack(fill="x", pady=(0, 10))

    def _build_factors_section(self, parent: ctk.CTkScrollableFrame, reasons: list):
        """Build the top factors section with header and reason rows."""
        factors_header = ctk.CTkLabel(
            parent,
            text="TOP FACTORS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["dialog_muted"],
            anchor="w",
        )
        factors_header.pack(fill="x", pady=(0, 6))

        sep = ctk.CTkFrame(parent, height=1, fg_color=COLORS["dialog_separator"])
        sep.pack(fill="x", pady=(0, 8))

        for reason in reasons:
            # Support both old (label, value) and new (label, value, source) formats
            if len(reason) >= 3:
                label, contrib, source = reason[0], reason[1], reason[2]
            else:
                label, contrib, source = reason[0], reason[1], ""
            self._add_reason_row(parent, label, contrib, source)

    def _add_reason_row(self, parent_frame, label: str, contribution: float, source: str = ""):
        """
        Add a single reason row with +/- indicator and source tag.

        Args:
            parent_frame: Frame to add the row to
            label: Human-readable reason text
            contribution: Positive = toward keep, negative = toward skip
            source: Model source tag ("LR", "RF", "Rules", or "")
        """
        if contribution > 0:
            symbol = "+"
            color = COLORS["dialog_chosen"]
        else:
            symbol = "-"
            color = COLORS["dialog_rejected"]

        row = ctk.CTkFrame(parent_frame, fg_color="transparent")
        row.pack(fill="x", pady=2)

        # +/- symbol
        indicator = ctk.CTkLabel(
            row,
            text=symbol,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=color,
            width=20,
            anchor="w",
        )
        indicator.pack(side="left", padx=(0, 6))

        # Reason text
        text = ctk.CTkLabel(
            row,
            text=label,
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        text.pack(side="left", fill="x", expand=True)

        # Source tag (small, muted) — e.g., [LR], [RF], [Rules]
        if source:
            tag = ctk.CTkLabel(
                row,
                text=f"[{source}]",
                font=ctk.CTkFont(size=10),
                text_color=_SOURCE_TAG_COLOR,
                anchor="e",
            )
            tag.pack(side="right", padx=(6, 0))
