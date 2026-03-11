"""
Pipeline Step Indicator Widget

Horizontal strip showing pipeline progression: Extract > Vocab > Search > Key Excerpts.
Each step has visual states: pending, active, done, skipped.
"""

import customtkinter as ctk

from src.ui.theme import COLORS, FONTS

# Step display names
PIPELINE_STEPS = ["Extract", "Vocabulary", "Search", "Key Excerpts"]


class PipelineIndicator(ctk.CTkFrame):
    """
    Horizontal pipeline indicator showing step progression.

    States per step:
        pending: dimmed text, normal weight
        active: blue text, bold
        done: green text with checkmark prefix
        skipped: dimmed text, italic
    """

    def __init__(self, master, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        kwargs.setdefault("height", 28)
        super().__init__(master, **kwargs)

        self._step_labels = {}
        self._step_states = {}

        for i, step in enumerate(PIPELINE_STEPS):
            if i > 0:
                # Arrow separator
                arrow = ctk.CTkLabel(
                    self,
                    text=">",
                    font=FONTS["small"],
                    text_color=COLORS["text_disabled"],
                )
                arrow.pack(side="left", padx=4)

            label = ctk.CTkLabel(
                self,
                text=step,
                font=FONTS["small"],
                text_color=COLORS["text_disabled"],
            )
            label.pack(side="left", padx=2)
            self._step_labels[step] = label
            self._step_states[step] = "pending"

    def set_step_state(self, step_name: str, state: str):
        """
        Update a step's visual state.

        Args:
            step_name: One of PIPELINE_STEPS
            state: One of 'pending', 'active', 'done', 'skipped'
        """
        label = self._step_labels.get(step_name)
        if not label:
            return

        self._step_states[step_name] = state

        if state == "pending":
            label.configure(
                text=step_name,
                font=FONTS["small"],
                text_color=COLORS["text_disabled"],
            )
        elif state == "active":
            label.configure(
                text=step_name,
                font=FONTS["small_bold"],
                text_color=COLORS["info"],
            )
        elif state == "done":
            label.configure(
                text=f"\u2713 {step_name}",
                font=FONTS["small_bold"],
                text_color=COLORS["success"],
            )
        elif state == "skipped":
            label.configure(
                text=step_name,
                font=("Segoe UI", 11, "italic"),
                text_color=COLORS["text_disabled"],
            )

    def set_enabled_steps(self, enabled: list[str]):
        """
        Mark steps as pending (enabled) or skipped (not enabled).

        Args:
            enabled: List of step names that are enabled
        """
        for step in PIPELINE_STEPS:
            if step in enabled:
                self.set_step_state(step, "pending")
            else:
                self.set_step_state(step, "skipped")

    def reset(self):
        """Reset all steps to pending."""
        for step in PIPELINE_STEPS:
            self.set_step_state(step, "pending")
