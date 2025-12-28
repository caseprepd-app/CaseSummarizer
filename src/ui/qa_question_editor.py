"""
Q&A Question Editor Widget for LocalScribe.

GUI for editing the default questions asked of every document.
Questions are stored in config/qa_questions.yaml.

Features:
- List view of current questions
- Add, Edit, Delete functionality
- Save/Cancel with confirmation
- Reset to defaults button
- Accessible from Settings dialog and Q&A panel

The editor modifies a simplified flat list of questions for the MVP.
The full branching question tree in qa_questions.yaml is preserved
but not exposed in this UI.
"""

import shutil
from pathlib import Path
from tkinter import messagebox, ttk

import customtkinter as ctk
import yaml

from src.config import DEBUG_MODE
from src.core.config import load_yaml_with_fallback
from src.logging_config import debug_log
from src.ui.theme import FONTS, COLORS, BUTTON_STYLES

# Default questions YAML path
DEFAULT_QUESTIONS_PATH = Path(__file__).parent.parent.parent / "config" / "qa_questions.yaml"

# Backup path for reset functionality
BACKUP_QUESTIONS_PATH = Path(__file__).parent.parent.parent / "config" / "qa_questions_default.yaml"


class QAQuestionEditor(ctk.CTkToplevel):
    """
    Dialog for editing default Q&A questions.

    Opens as a modal dialog with a list of questions that can be
    added, edited, or deleted. Changes are saved to qa_questions.yaml.

    Example:
        editor = QAQuestionEditor(parent_window)
        editor.wait_window()  # Blocks until closed
        if editor.saved:
            print("Questions were saved")
    """

    def __init__(self, parent, yaml_path: Path | None = None):
        """
        Initialize question editor dialog.

        Args:
            parent: Parent window
            yaml_path: Path to questions YAML (default: config/qa_questions.yaml)
        """
        super().__init__(parent)

        self.yaml_path = yaml_path or DEFAULT_QUESTIONS_PATH
        self.saved = False
        self._original_questions: list[dict] = []
        self._questions: list[dict] = []
        self._has_changes = False

        # Window configuration
        self.title("Edit Default Questions")
        self.geometry("700x500")
        self.minsize(500, 400)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 700) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 500) // 2
        self.geometry(f"+{x}+{y}")

        # Build UI
        self._create_ui()

        # Load questions
        self._load_questions()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if DEBUG_MODE:
            debug_log(f"[QAQuestionEditor] Opened with {len(self._questions)} questions")

    def _create_ui(self):
        """Build the editor UI."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        title = ctk.CTkLabel(
            header,
            text="Default Questions",
            font=FONTS["heading_lg"]
        )
        title.pack(side="left")

        description = ctk.CTkLabel(
            header,
            text="These questions are asked automatically for every document.",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"]
        )
        description.pack(side="left", padx=20)

        # Question list
        list_frame = ctk.CTkFrame(self, fg_color="#2b2b2b", corner_radius=6)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)

        # Treeview for questions (style configured in src/ui/styles.py)
        self.question_tree = ttk.Treeview(
            list_frame,
            columns=("num", "category", "question"),
            show="headings",
            style="QuestionList.Treeview",
            selectmode="browse"
        )

        # Configure columns
        self.question_tree.heading("num", text="#", anchor="w")
        self.question_tree.heading("category", text="Category", anchor="w")
        self.question_tree.heading("question", text="Question", anchor="w")

        self.question_tree.column("num", width=40, stretch=False)
        self.question_tree.column("category", width=120, stretch=False)
        self.question_tree.column("question", width=450, stretch=True)

        self.question_tree.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.question_tree.yview
        )
        self.question_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Double-click to edit
        self.question_tree.bind("<Double-1>", lambda e: self._edit_selected())

        # Edit controls
        edit_frame = ctk.CTkFrame(self, fg_color="transparent")
        edit_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        self.add_btn = ctk.CTkButton(
            edit_frame,
            text="+ Add Question",
            command=self._add_question,
            width=120
        )
        self.add_btn.pack(side="left", padx=(0, 5))

        self.edit_btn = ctk.CTkButton(
            edit_frame,
            text="Edit Selected",
            command=self._edit_selected,
            width=100
        )
        self.edit_btn.pack(side="left", padx=5)

        self.delete_btn = ctk.CTkButton(
            edit_frame,
            text="Delete Selected",
            command=self._delete_selected,
            width=110,
            **BUTTON_STYLES["danger"]
        )
        self.delete_btn.pack(side="left", padx=5)

        self.reset_btn = ctk.CTkButton(
            edit_frame,
            text="Reset to Defaults",
            command=self._reset_to_defaults,
            width=120,
            **BUTTON_STYLES["secondary"]
        )
        self.reset_btn.pack(side="right", padx=5)

        # Move Up/Down buttons
        self.move_up_btn = ctk.CTkButton(
            edit_frame,
            text="↑ Move Up",
            command=self._move_up,
            width=80,
            **BUTTON_STYLES["tertiary"]
        )
        self.move_up_btn.pack(side="right", padx=5)

        self.move_down_btn = ctk.CTkButton(
            edit_frame,
            text="↓ Move Down",
            command=self._move_down,
            width=90,
            **BUTTON_STYLES["tertiary"]
        )
        self.move_down_btn.pack(side="right", padx=5)

        # Bottom buttons (Save/Cancel)
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))

        self.cancel_btn = ctk.CTkButton(
            bottom_frame,
            text="Cancel",
            command=self._on_close,
            width=100,
            **BUTTON_STYLES["secondary"]
        )
        self.cancel_btn.pack(side="right", padx=(5, 0))

        self.save_btn = ctk.CTkButton(
            bottom_frame,
            text="Save Changes",
            command=self._save_and_close,
            width=120
        )
        self.save_btn.pack(side="right", padx=5)

        # Status label
        self.status_label = ctk.CTkLabel(
            bottom_frame,
            text="",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"]
        )
        self.status_label.pack(side="left")

    def _load_questions(self):
        """Load questions from YAML file."""
        import copy

        config = load_yaml_with_fallback(
            self.yaml_path,
            fallback={},
            log_prefix="[QAQuestionEditor]"
        )

        if not config or "questions" not in config:
            self._questions = []
        else:
            # Extract simplified question list
            self._questions = []
            for q in config.get("questions", []):
                self._questions.append({
                    "id": q.get("id", ""),
                    "text": q.get("text", ""),
                    "category": q.get("category", "General"),
                    "type": q.get("type", "extraction")
                })

        # Store original for reset/cancel
        self._original_questions = copy.deepcopy(self._questions)

        self._refresh_list()
        self.status_label.configure(text=f"{len(self._questions)} questions loaded")

    def _refresh_list(self):
        """Refresh the Treeview with current questions."""
        self.question_tree.delete(*self.question_tree.get_children())

        for i, q in enumerate(self._questions, 1):
            self.question_tree.insert(
                "",
                "end",
                iid=str(i - 1),
                values=(i, q.get("category", ""), q.get("text", ""))
            )

    def _add_question(self):
        """Add a new question."""
        dialog = QuestionEditDialog(self, title="Add Question")
        self.wait_window(dialog)

        if dialog.result:
            self._questions.append(dialog.result)
            self._has_changes = True
            self._refresh_list()
            self.status_label.configure(text="Question added (unsaved)")

    def _edit_selected(self):
        """Edit the selected question."""
        selection = self.question_tree.selection()
        if not selection:
            messagebox.showinfo("Select Question", "Please select a question to edit.")
            return

        try:
            index = int(selection[0])
            question = self._questions[index]

            dialog = QuestionEditDialog(self, title="Edit Question", question=question)
            self.wait_window(dialog)

            if dialog.result:
                self._questions[index] = dialog.result
                self._has_changes = True
                self._refresh_list()
                self.status_label.configure(text="Question edited (unsaved)")

        except (ValueError, IndexError):
            pass

    def _delete_selected(self):
        """Delete the selected question."""
        selection = self.question_tree.selection()
        if not selection:
            messagebox.showinfo("Select Question", "Please select a question to delete.")
            return

        if not messagebox.askyesno("Confirm Delete", "Delete this question?"):
            return

        try:
            index = int(selection[0])
            del self._questions[index]
            self._has_changes = True
            self._refresh_list()
            self.status_label.configure(text="Question deleted (unsaved)")

        except (ValueError, IndexError):
            pass

    def _move_up(self):
        """Move selected question up in the list."""
        selection = self.question_tree.selection()
        if not selection:
            return

        try:
            index = int(selection[0])
            if index > 0:
                self._questions[index], self._questions[index - 1] = \
                    self._questions[index - 1], self._questions[index]
                self._has_changes = True
                self._refresh_list()
                self.question_tree.selection_set(str(index - 1))

        except (ValueError, IndexError):
            pass

    def _move_down(self):
        """Move selected question down in the list."""
        selection = self.question_tree.selection()
        if not selection:
            return

        try:
            index = int(selection[0])
            if index < len(self._questions) - 1:
                self._questions[index], self._questions[index + 1] = \
                    self._questions[index + 1], self._questions[index]
                self._has_changes = True
                self._refresh_list()
                self.question_tree.selection_set(str(index + 1))

        except (ValueError, IndexError):
            pass

    def _reset_to_defaults(self):
        """Reset questions to default values."""
        if not messagebox.askyesno(
            "Reset to Defaults",
            "This will restore the original default questions.\n\n"
            "Your custom questions will be lost. Continue?"
        ):
            return

        # Try to restore from backup file
        if BACKUP_QUESTIONS_PATH.exists():
            try:
                shutil.copy(BACKUP_QUESTIONS_PATH, self.yaml_path)
                self._load_questions()
                self._has_changes = True
                self.status_label.configure(text="Reset to defaults (unsaved)")
                debug_log("[QAQuestionEditor] Reset from backup")
                return
            except Exception as e:
                debug_log(f"[QAQuestionEditor] Backup restore failed: {e}")

        # If no backup, restore from original
        import copy
        self._questions = copy.deepcopy(self._original_questions)
        self._has_changes = True
        self._refresh_list()
        self.status_label.configure(text="Reset to defaults (unsaved)")

    def _save_and_close(self):
        """Save changes and close dialog."""
        if not self._save_questions():
            return  # Save failed

        self.saved = True
        self.destroy()

    def _save_questions(self) -> bool:
        """
        Save questions back to YAML file.

        Returns:
            True if save succeeded
        """
        try:
            # Load existing config to preserve structure
            config = load_yaml_with_fallback(
                self.yaml_path,
                fallback={"version": "1.0", "entry_point": "is_court_case"},
                log_prefix="[QAQuestionEditor]"
            )

            # Update questions in config
            config["questions"] = []
            for q in self._questions:
                config["questions"].append({
                    "id": q.get("id", "q_" + str(len(config["questions"]))),
                    "text": q.get("text", ""),
                    "category": q.get("category", "General"),
                    "type": q.get("type", "extraction"),
                    "terminal": True  # All custom questions are terminal for simplicity
                })

            # Write back
            with open(self.yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            debug_log(f"[QAQuestionEditor] Saved {len(self._questions)} questions")
            return True

        except Exception as e:
            debug_log(f"[QAQuestionEditor] Save error: {e}")
            messagebox.showerror("Save Error", f"Failed to save questions: {e}")
            return False

    def _on_close(self):
        """Handle window close."""
        if self._has_changes:
            result = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes.\n\nSave before closing?"
            )
            if result is None:  # Cancel
                return
            if result:  # Yes - save
                if not self._save_questions():
                    return
                self.saved = True
            # No - discard changes

        self.destroy()


class QuestionEditDialog(ctk.CTkToplevel):
    """
    Dialog for editing a single question.

    Used by QAQuestionEditor for add/edit operations.
    """

    def __init__(self, parent, title: str = "Edit Question", question: dict | None = None):
        """
        Initialize question edit dialog.

        Args:
            parent: Parent window
            title: Dialog title
            question: Existing question dict to edit (None for new)
        """
        super().__init__(parent)

        self.result = None
        self._question = question or {}

        # Window configuration
        self.title(title)
        self.geometry("500x300")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 500) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 300) // 2
        self.geometry(f"+{x}+{y}")

        self._create_ui()

        # Focus on question entry
        self.question_entry.focus()

    def _create_ui(self):
        """Build the dialog UI."""
        self.grid_columnconfigure(0, weight=1)

        # Category
        category_label = ctk.CTkLabel(self, text="Category:")
        category_label.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 5))

        self.category_entry = ctk.CTkEntry(self, width=200)
        self.category_entry.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 10))
        self.category_entry.insert(0, self._question.get("category", "General"))

        # Question text
        question_label = ctk.CTkLabel(self, text="Question:")
        question_label.grid(row=2, column=0, sticky="w", padx=20, pady=(10, 5))

        self.question_entry = ctk.CTkTextbox(self, height=100, width=440)
        self.question_entry.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 10))
        self.question_entry.insert("0.0", self._question.get("text", ""))

        # Buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=4, column=0, sticky="ew", padx=20, pady=(10, 20))

        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.destroy,
            width=100,
            **BUTTON_STYLES["secondary"]
        )
        cancel_btn.pack(side="right", padx=(5, 0))

        save_btn = ctk.CTkButton(
            button_frame,
            text="Save",
            command=self._save,
            width=100
        )
        save_btn.pack(side="right", padx=5)

    def _save(self):
        """Save and close dialog."""
        text = self.question_entry.get("0.0", "end").strip()
        if not text:
            messagebox.showwarning("Required", "Please enter a question.")
            return

        self.result = {
            "id": self._question.get("id", ""),
            "text": text,
            "category": self.category_entry.get().strip() or "General",
            "type": "extraction"
        }
        self.destroy()
