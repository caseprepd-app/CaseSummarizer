"""
Default Questions Widget for Settings Dialog.

Widget for managing default Q&A questions with checkboxes.
Session 82: Extracted from settings_widgets.py for modularity.

Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │  Default Questions                                       ⓘ  │
    ├─────────────────────────────────────────────────────────────┤
    │  ☑ What is this case about?                            [✕] │
    │  ☑ What are the main allegations?                      [✕] │
    │  ☐ Who are the plaintiffs?                             [✕] │
    │  ...                                                        │
    ├─────────────────────────────────────────────────────────────┤
    │  [+ Add Question]                                           │
    └─────────────────────────────────────────────────────────────┘
"""

import customtkinter as ctk

from src.ui.settings.settings_widgets import TooltipIcon
from src.ui.theme import FONTS


class DefaultQuestionsWidget(ctk.CTkFrame):
    """
    Widget for managing default Q&A questions with checkboxes.

    Session 63c: Provides a scrollable list of questions where each can be
    enabled/disabled via checkbox. Also supports add, edit, delete, and reorder.
    """

    def __init__(self, parent, **kwargs):
        """
        Initialize the default questions widget.

        Args:
            parent: Parent widget.
            **kwargs: Additional CTkFrame arguments.
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        # Get manager instance
        from src.core.qa.default_questions_manager import get_default_questions_manager

        self.manager = get_default_questions_manager()

        # Track checkbox variables
        self._checkboxes: list[tuple[ctk.CTkCheckBox, ctk.BooleanVar]] = []

        self._setup_ui()

    def _setup_ui(self):
        """Create the widget layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header row with label and tooltip
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        header_label = ctk.CTkLabel(
            header_frame, text="Default Questions", font=FONTS["heading_sm"], anchor="w"
        )
        header_label.pack(side="left", padx=(0, 5))

        tooltip = TooltipIcon(
            header_frame,
            tooltip_text=(
                "Questions that are automatically asked after document processing.\n\n"
                "• Check/uncheck to enable/disable questions\n"
                "• Click '✕' to delete a question\n"
                "• Click '+ Add Question' to add new questions\n\n"
                "Disabled questions are saved but won't be asked."
            ),
        )
        tooltip.pack(side="left")

        # Scrollable frame for questions
        self.scroll_frame = ctk.CTkScrollableFrame(
            self, height=180, fg_color=("gray90", "gray17"), corner_radius=6
        )
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        # Populate with existing questions
        self._refresh_question_list()

        # Add question button
        add_frame = ctk.CTkFrame(self, fg_color="transparent")
        add_frame.grid(row=2, column=0, sticky="w")

        self.add_btn = ctk.CTkButton(
            add_frame,
            text="+ Add Question",
            command=self._add_question,
            width=140,
            height=28,
            font=FONTS["body"],
        )
        self.add_btn.pack(side="left")

    def _refresh_question_list(self):
        """Rebuild the question list from manager."""
        # Clear existing widgets
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self._checkboxes.clear()

        # Add each question
        questions = self.manager.get_all_questions()
        for idx, q in enumerate(questions):
            self._add_question_row(idx, q.text, q.enabled)

    def _add_question_row(self, index: int, text: str, enabled: bool):
        """Add a single question row."""
        row_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        row_frame.grid(row=index, column=0, sticky="ew", pady=2, padx=5)
        row_frame.grid_columnconfigure(1, weight=1)

        # Checkbox with question text
        var = ctk.BooleanVar(value=enabled)
        checkbox = ctk.CTkCheckBox(
            row_frame,
            text="",
            variable=var,
            command=lambda i=index: self._on_toggle(i),
            width=24,
            checkbox_width=18,
            checkbox_height=18,
        )
        checkbox.grid(row=0, column=0, sticky="w")

        # Question text label (clickable to edit)
        text_label = ctk.CTkLabel(
            row_frame, text=text, anchor="w", font=FONTS["body"], cursor="hand2"
        )
        text_label.grid(row=0, column=1, sticky="ew", padx=(5, 10))
        text_label.bind("<Button-1>", lambda e, i=index: self._edit_question(i))

        # Delete button
        delete_btn = ctk.CTkButton(
            row_frame,
            text="✕",
            command=lambda i=index: self._delete_question(i),
            width=24,
            height=24,
            fg_color="transparent",
            hover_color=("gray70", "gray30"),
            text_color=("gray40", "gray60"),
            font=FONTS["body"],
        )
        delete_btn.grid(row=0, column=2, sticky="e")

        self._checkboxes.append((checkbox, var))

    def _on_toggle(self, index: int):
        """Handle checkbox toggle."""
        if index < len(self._checkboxes):
            _, var = self._checkboxes[index]
            self.manager.set_enabled(index, var.get())

    def _add_question(self):
        """Show dialog to add a new question."""
        dialog = ctk.CTkInputDialog(text="Enter a new question:", title="Add Question")
        text = dialog.get_input()

        if text and text.strip():
            self.manager.add_question(text.strip())
            self._refresh_question_list()

    def _edit_question(self, index: int):
        """Show dialog to edit a question."""
        questions = self.manager.get_all_questions()
        if index >= len(questions):
            return

        current_text = questions[index].text
        # Show current text in the prompt since CTkInputDialog doesn't support pre-fill
        display_text = current_text[:60] + "..." if len(current_text) > 60 else current_text
        dialog = ctk.CTkInputDialog(
            text=f'Current: "{display_text}"\n\nEnter new text:', title="Edit Question"
        )

        text = dialog.get_input()

        if text and text.strip():
            self.manager.update_question(index, text.strip())
            self._refresh_question_list()

    def _delete_question(self, index: int):
        """Delete a question after confirmation."""
        from tkinter import messagebox

        questions = self.manager.get_all_questions()
        if index >= len(questions):
            return

        question_text = questions[index].text
        # Truncate for display
        display_text = question_text[:50] + "..." if len(question_text) > 50 else question_text

        if messagebox.askyesno("Delete Question", f'Delete this question?\n\n"{display_text}"'):
            self.manager.remove_question(index)
            self._refresh_question_list()

    def get_value(self):
        """Return enabled state - not needed since changes are saved immediately."""
        return None

    def set_value(self, value):
        """Set value - not needed since we load from manager."""
        pass
