"""
Default Searches Widget for Settings Dialog.

Widget for managing default semantic searches with checkboxes.
Changes are buffered locally and only persisted when the dialog's
Save button is clicked.

Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │  Default Searches                                        ⓘ  │
    ├─────────────────────────────────────────────────────────────┤
    │  ☑ What injuries were sustained?                       [✕] │
    │  ☑ What warnings were given?                           [✕] │
    │  ...                                                        │
    ├─────────────────────────────────────────────────────────────┤
    │  [+ Add Search]                                             │
    └─────────────────────────────────────────────────────────────┘
"""

import customtkinter as ctk

from src.ui.settings.base_settings_widget import BaseSettingsWidget
from src.ui.theme import FONTS


class DefaultQuestionsWidget(BaseSettingsWidget):
    """
    Widget for managing default semantic searches with checkboxes.

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
        from src.services import SemanticService

        semantic_service = SemanticService()
        self._manager = semantic_service.get_default_questions_manager()

        # Buffer: list of {"text": str, "enabled": bool} dicts
        self._buffer = [
            {"text": q.text, "enabled": q.enabled} for q in self._manager.get_all_questions()
        ]

        self._checkboxes: list[tuple[ctk.CTkCheckBox, ctk.BooleanVar]] = []
        super().__init__(parent, **kwargs)

    def _setup_ui(self):
        """Create the widget layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header row
        from src.ui.settings.section_header import SectionHeader

        header = SectionHeader(
            self,
            "Default Searches",
            tooltip_text=(
                "Searches that are automatically run after document processing.\n\n"
                "• Check/uncheck to enable/disable searches\n"
                "• Click '✕' to delete a search\n"
                "• Click '+ Add Search' to add new searches\n\n"
                "Disabled searches are saved but won't run."
            ),
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        # Guidance text
        guidance_text = (
            "Tip: Search for one specific topic at a time. Use natural phrases.\n\n"
            'Good: "injuries sustained by the plaintiff", "warnings given before the accident"\n'
            'Avoid: "what is this case about", "injuries and warnings and timeline"'
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
            text="+ Add Search",
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
        """Show dialog to add a new search to the buffer."""
        dialog = ctk.CTkInputDialog(text="Enter a new search:", title="Add Search")
        text = dialog.get_input()

        if text and text.strip():
            self._buffer.append({"text": text.strip(), "enabled": True})
            self._rebuild_ui()

    def _edit_question(self, index: int):
        """Show dialog to edit a search in the buffer."""
        if index >= len(self._buffer):
            return

        current_text = self._buffer[index]["text"]
        display_text = current_text[:60] + "..." if len(current_text) > 60 else current_text
        dialog = ctk.CTkInputDialog(
            text=f'Current: "{display_text}"\n\nEnter new text:', title="Edit Search"
        )
        text = dialog.get_input()

        if text and text.strip():
            self._buffer[index]["text"] = text.strip()
            self._rebuild_ui()

    def _delete_question(self, index: int):
        """Delete a search from the buffer after confirmation."""
        from tkinter import messagebox

        if index >= len(self._buffer):
            return

        question_text = self._buffer[index]["text"]
        display_text = question_text[:50] + "..." if len(question_text) > 50 else question_text

        if messagebox.askyesno("Delete Search", f'Delete this search?\n\n"{display_text}"'):
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
