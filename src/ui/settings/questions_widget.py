"""
Default Questions Widget for Settings Dialog.

Widget for managing default Q&A questions with checkboxes.
Changes are buffered locally and only persisted when the dialog's
Save button is clicked.

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

    All changes are buffered locally. The dialog calls get_value()
    on Save to retrieve the buffer, which is then persisted via the
    registered setter. Clicking Cancel discards the buffer.
    """

    def __init__(self, parent, **kwargs):
        """
        Initialize the default questions widget.

        Args:
            parent: Parent widget.
            **kwargs: Additional CTkFrame arguments.
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        from src.services import QAService

        qa_service = QAService()
        self._manager = qa_service.get_default_questions_manager()

        # Buffer: list of {"text": str, "enabled": bool} dicts
        self._buffer = [
            {"text": q.text, "enabled": q.enabled} for q in self._manager.get_all_questions()
        ]

        self._checkboxes: list[tuple[ctk.CTkCheckBox, ctk.BooleanVar]] = []
        self._setup_ui()

    def _setup_ui(self):
        """Create the widget layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header row
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

        # Guidance text
        guidance_text = (
            "Tip: Ask one specific question at a time about facts stated in the documents.\n\n"
            'Good: "What injuries were sustained?", "What warnings were given?"\n'
            'Avoid: "What is this case about?", "When and where did the accident happen?"'
        )
        guidance_label = ctk.CTkLabel(
            self,
            text=guidance_text,
            font=FONTS["small"],
            text_color=("gray40", "gray60"),
            anchor="w",
            justify="left",
        )
        guidance_label.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        # Scrollable frame for questions
        self.scroll_frame = ctk.CTkScrollableFrame(
            self, height=180, fg_color=("gray90", "gray17"), corner_radius=6
        )
        self.scroll_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        self._rebuild_ui()

        # Add question button
        add_frame = ctk.CTkFrame(self, fg_color="transparent")
        add_frame.grid(row=3, column=0, sticky="w")

        self.add_btn = ctk.CTkButton(
            add_frame,
            text="+ Add Question",
            command=self._add_question,
            width=140,
            height=28,
            font=FONTS["body"],
        )
        self.add_btn.pack(side="left")

    def _rebuild_ui(self):
        """Rebuild the question list from the local buffer."""
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self._checkboxes.clear()

        for idx, q in enumerate(self._buffer):
            self._add_question_row(idx, q["text"], q["enabled"])

    def _add_question_row(self, index: int, text: str, enabled: bool):
        """Add a single question row to the scroll frame."""
        row_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        row_frame.grid(row=index, column=0, sticky="ew", pady=2, padx=5)
        row_frame.grid_columnconfigure(1, weight=1)

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

        text_label = ctk.CTkLabel(
            row_frame, text=text, anchor="w", font=FONTS["body"], cursor="hand2"
        )
        text_label.grid(row=0, column=1, sticky="ew", padx=(5, 10))
        text_label.bind("<Button-1>", lambda e, i=index: self._edit_question(i))

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
        """Handle checkbox toggle — updates buffer only."""
        if index < len(self._buffer):
            _, var = self._checkboxes[index]
            self._buffer[index]["enabled"] = var.get()

    def _add_question(self):
        """Show dialog to add a new question to the buffer."""
        dialog = ctk.CTkInputDialog(text="Enter a new question:", title="Add Question")
        text = dialog.get_input()

        if text and text.strip():
            self._buffer.append({"text": text.strip(), "enabled": True})
            self._rebuild_ui()

    def _edit_question(self, index: int):
        """Show dialog to edit a question in the buffer."""
        if index >= len(self._buffer):
            return

        current_text = self._buffer[index]["text"]
        display_text = current_text[:60] + "..." if len(current_text) > 60 else current_text
        dialog = ctk.CTkInputDialog(
            text=f'Current: "{display_text}"\n\nEnter new text:', title="Edit Question"
        )
        text = dialog.get_input()

        if text and text.strip():
            self._buffer[index]["text"] = text.strip()
            self._rebuild_ui()

    def _delete_question(self, index: int):
        """Delete a question from the buffer after confirmation."""
        from tkinter import messagebox

        if index >= len(self._buffer):
            return

        question_text = self._buffer[index]["text"]
        display_text = question_text[:50] + "..." if len(question_text) > 50 else question_text

        if messagebox.askyesno("Delete Question", f'Delete this question?\n\n"{display_text}"'):
            self._buffer.pop(index)
            self._rebuild_ui()

    def get_value(self) -> list[dict]:
        """Return the buffered questions list for persistence on Save."""
        return self._buffer

    def set_value(self, value):
        """Set value — reload buffer from provided data."""
        if value and isinstance(value, list):
            self._buffer = value
            self._rebuild_ui()
