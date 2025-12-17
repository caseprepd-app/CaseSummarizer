"""
Q&A Panel Widget for LocalScribe.

Displays Q&A results in a CSV-style table with three columns:
- Question: The question asked
- Quick Answer: AI-synthesized answer from Ollama
- Citation: Raw text excerpts from BM25+/vector retrieval

Features:
- Excel-like Treeview table with frozen headers
- Per-result include/exclude toggles (checkbox column)
- Export to CSV or TXT
- Collapsible follow-up question input pane
"""

import csv
import io
import queue
import threading
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import customtkinter as ctk

from src.config import DEBUG_MODE
from src.logging_config import debug_log
from src.qa.qa_orchestrator import QAResult

# Unicode checkbox icons for toggle display
CHECK_ICON = "☑"  # U+2611 Ballot Box with Check
UNCHECK_ICON = "☐"  # U+2610 Ballot Box

# Column configuration for Q&A table
QA_COLUMN_CONFIG = {
    "Include": {"width": 50, "max_chars": 3},
    "Question": {"width": 180, "max_chars": 35},
    "Quick Answer": {"width": 250, "max_chars": 50},
    "Citation": {"width": 300, "max_chars": 60},
}


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text with ellipsis for table display."""
    if not text:
        return ""
    text = str(text).replace('\n', ' ').replace('\r', '').strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


class QAPanel(ctk.CTkFrame):
    """
    Q&A display panel with CSV-style table layout.

    Features:
    - 3-column table: Question | Quick Answer | Citation
    - Include/exclude checkboxes for export
    - Export to CSV or TXT
    - Follow-up question input pane

    Example:
        panel = QAPanel(parent)
        panel.display_results(qa_results)

        # Handle follow-up questions
        panel.set_followup_callback(lambda q: orchestrator.ask_followup(q))
    """

    def __init__(
        self,
        master,
        on_edit_questions: Callable | None = None,
        on_ask_followup: Callable[[str], QAResult | None] | None = None,
        **kwargs
    ):
        """
        Initialize Q&A panel.

        Args:
            master: Parent widget
            on_edit_questions: Callback when "Edit Questions" is clicked
            on_ask_followup: Callback(question_text) -> QAResult for follow-ups
        """
        super().__init__(master, **kwargs)

        self.on_edit_questions = on_edit_questions
        self.on_ask_followup = on_ask_followup

        # Results storage
        self._results: list[QAResult] = []

        # Async follow-up state
        self._followup_queue: queue.Queue = queue.Queue()
        self._followup_thread: threading.Thread | None = None
        self._polling_active: bool = False

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Table area expands

        # Build UI components
        self._create_header()
        self._create_table_area()
        self._create_button_bar()
        self._create_followup_pane()

        if DEBUG_MODE:
            debug_log("[QAPanel] Initialized with CSV-style table layout")

    def _create_header(self):
        """Create header with title and info."""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))

        title = ctk.CTkLabel(
            header,
            text="Document Q&A (CSV Format)",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(side="left")

        self.info_label = ctk.CTkLabel(
            header,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#aaaaaa"
        )
        self.info_label.pack(side="right")

    def _create_table_area(self):
        """Create main CSV-style table display."""
        # Frame for table with dark theme
        table_frame = ctk.CTkFrame(self, fg_color="#2b2b2b", corner_radius=6)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        # Define columns
        columns = ("Include", "Question", "Quick Answer", "Citation")

        # Create Treeview
        self.qa_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            style="QATable.Treeview",
            selectmode="browse"
        )

        # Configure column headings and widths
        for col in columns:
            col_config = QA_COLUMN_CONFIG.get(col, {"width": 100})
            self.qa_tree.heading(col, text=col, anchor='w')
            # Citation column stretches to fill remaining space
            stretch = True if col == "Citation" else False
            self.qa_tree.column(
                col,
                width=col_config["width"],
                minwidth=50,
                anchor='w',
                stretch=stretch
            )

        # Add vertical scrollbar
        vsb = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.qa_tree.yview,
            style="QATable.Vertical.TScrollbar"
        )
        self.qa_tree.configure(yscrollcommand=vsb.set)

        # Add horizontal scrollbar
        hsb = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.qa_tree.xview,
            style="QATable.Horizontal.TScrollbar"
        )
        self.qa_tree.configure(xscrollcommand=hsb.set)

        # Grid layout
        self.qa_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Bind click for Include column toggle
        self.qa_tree.bind("<Button-1>", self._on_table_click)

        # Bind double-click to show full text
        self.qa_tree.bind("<Double-1>", self._on_double_click)

    def _create_button_bar(self):
        """Create action buttons bar."""
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        # Edit Questions button
        self.edit_btn = ctk.CTkButton(
            button_frame,
            text="⚙ Edit Questions",
            command=self._on_edit_click,
            width=130
        )
        self.edit_btn.pack(side="left", padx=(0, 5))

        # Ask More Questions button
        self.ask_more_btn = ctk.CTkButton(
            button_frame,
            text="Ask More Questions",
            command=self._toggle_followup_pane,
            width=140
        )
        self.ask_more_btn.pack(side="left", padx=5)

        # Export CSV button (primary)
        self.export_csv_btn = ctk.CTkButton(
            button_frame,
            text="Export CSV",
            command=self._export_to_csv,
            width=100
        )
        self.export_csv_btn.pack(side="right", padx=5)

        # Export TXT button (secondary)
        self.export_txt_btn = ctk.CTkButton(
            button_frame,
            text="Export TXT",
            command=self._export_to_txt,
            width=90,
            fg_color="#555555"
        )
        self.export_txt_btn.pack(side="right", padx=5)

        # Select All / Deselect All buttons
        self.select_all_btn = ctk.CTkButton(
            button_frame,
            text="Select All",
            command=lambda: self._set_all_include(True),
            width=80,
            fg_color="#2d5a87"
        )
        self.select_all_btn.pack(side="right", padx=5)

        self.deselect_all_btn = ctk.CTkButton(
            button_frame,
            text="Deselect All",
            command=lambda: self._set_all_include(False),
            width=90,
            fg_color="#555555"
        )
        self.deselect_all_btn.pack(side="right", padx=5)

    def _create_followup_pane(self):
        """Create collapsible follow-up question input pane."""
        self.followup_frame = ctk.CTkFrame(self, fg_color="#333333", corner_radius=6)
        # Initially hidden
        self.followup_visible = False

        # Input field
        self.followup_entry = ctk.CTkEntry(
            self.followup_frame,
            placeholder_text="Type your question here...",
            width=400
        )
        self.followup_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)

        # Bind Enter key
        self.followup_entry.bind("<Return>", lambda e: self._submit_followup())

        # Ask button
        self.followup_ask_btn = ctk.CTkButton(
            self.followup_frame,
            text="Ask",
            command=self._submit_followup,
            width=80
        )
        self.followup_ask_btn.pack(side="right", padx=(5, 10), pady=10)

    def display_results(self, results: list[QAResult]):
        """
        Display Q&A results in the table.

        Args:
            results: List of QAResult objects to display
        """
        self._results = results

        # Clear existing items
        self.qa_tree.delete(*self.qa_tree.get_children())

        # Insert rows
        for i, result in enumerate(results):
            include_icon = CHECK_ICON if result.include_in_export else UNCHECK_ICON
            question = truncate_text(result.question, QA_COLUMN_CONFIG["Question"]["max_chars"])
            quick_answer = truncate_text(result.quick_answer, QA_COLUMN_CONFIG["Quick Answer"]["max_chars"])
            citation = truncate_text(result.citation, QA_COLUMN_CONFIG["Citation"]["max_chars"])

            self.qa_tree.insert(
                "",
                "end",
                iid=str(i),
                values=(include_icon, question, quick_answer, citation)
            )

        # Update info label
        included = sum(1 for r in results if r.include_in_export)
        self.info_label.configure(text=f"{included}/{len(results)} selected for export")

        if DEBUG_MODE:
            debug_log(f"[QAPanel] Displaying {len(results)} results in table")

    def _on_table_click(self, event):
        """Handle click on table to toggle include/exclude."""
        # Check if click was on Include column
        column = self.qa_tree.identify_column(event.x)
        item_id = self.qa_tree.identify_row(event.y)

        if not item_id:
            return

        # Only toggle if clicking the Include column (#1)
        if column == "#1":
            try:
                index = int(item_id)
                if 0 <= index < len(self._results):
                    # Toggle the flag
                    result = self._results[index]
                    result.include_in_export = not result.include_in_export

                    # Update display
                    icon = CHECK_ICON if result.include_in_export else UNCHECK_ICON
                    values = list(self.qa_tree.item(item_id, 'values'))
                    values[0] = icon
                    self.qa_tree.item(item_id, values=tuple(values))

                    # Update info label
                    included = sum(1 for r in self._results if r.include_in_export)
                    self.info_label.configure(text=f"{included}/{len(self._results)} selected for export")

                    if DEBUG_MODE:
                        debug_log(f"[QAPanel] Toggled Q{index + 1}: include={result.include_in_export}")

            except (ValueError, IndexError):
                pass

    def _on_double_click(self, event):
        """Handle double-click to show full text in a popup."""
        column = self.qa_tree.identify_column(event.x)
        item_id = self.qa_tree.identify_row(event.y)

        if not item_id:
            return

        try:
            index = int(item_id)
            if 0 <= index < len(self._results):
                result = self._results[index]

                # Determine which column was clicked
                if column == "#2":  # Question
                    title = "Full Question"
                    content = result.question
                elif column == "#3":  # Quick Answer
                    title = "Full Quick Answer"
                    content = result.quick_answer
                elif column == "#4":  # Citation
                    title = "Full Citation"
                    content = result.citation
                else:
                    return

                # Show in messagebox (simple approach)
                messagebox.showinfo(title, content)

        except (ValueError, IndexError):
            pass

    def _set_all_include(self, include: bool):
        """Set include_in_export for all results."""
        for result in self._results:
            result.include_in_export = include

        # Refresh display
        self.display_results(self._results)

    def _toggle_followup_pane(self):
        """Show/hide the follow-up question input pane."""
        if self.followup_visible:
            self.followup_frame.grid_remove()
            self.ask_more_btn.configure(text="Ask More Questions")
            self.followup_visible = False
        else:
            self.followup_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=(0, 5))
            self.ask_more_btn.configure(text="Hide Question Input")
            self.followup_visible = True
            self.followup_entry.focus()

    def _submit_followup(self):
        """Submit a follow-up question asynchronously."""
        question = self.followup_entry.get().strip()
        if not question:
            return

        if self.on_ask_followup is None:
            messagebox.showwarning(
                "Not Available",
                "Follow-up questions are not available. "
                "Please process a document first."
            )
            return

        # Prevent duplicate submissions
        if self._followup_thread is not None and self._followup_thread.is_alive():
            debug_log("[QAPanel] Follow-up already in progress, ignoring")
            return

        # Clear entry
        self.followup_entry.delete(0, "end")

        # Disable button while processing
        self.followup_ask_btn.configure(state="disabled", text="Asking...")
        self.followup_entry.configure(state="disabled")

        if DEBUG_MODE:
            debug_log(f"[QAPanel] Starting async follow-up: {question[:30]}...")

        # Run callback in background thread
        def run_followup():
            try:
                result = self.on_ask_followup(question)
                self._followup_queue.put(("success", result))
            except Exception as e:
                self._followup_queue.put(("error", str(e)))
                debug_log(f"[QAPanel] Follow-up thread error: {e}")

        self._followup_thread = threading.Thread(target=run_followup, daemon=True)
        self._followup_thread.start()

        # Start polling for results
        self._polling_active = True
        self._poll_followup_result()

    def _poll_followup_result(self):
        """Poll for follow-up result from background thread."""
        if not self._polling_active:
            return

        try:
            msg_type, data = self._followup_queue.get_nowait()

            # Got a result - process it
            self._polling_active = False

            if msg_type == "success" and data is not None:
                # Add to results
                self._results.append(data)
                self.display_results(self._results)

                if DEBUG_MODE:
                    debug_log(f"[QAPanel] Follow-up completed successfully")

            elif msg_type == "error":
                messagebox.showerror("Error", f"Failed to ask question: {data}")

            # Re-enable input
            self.followup_ask_btn.configure(state="normal", text="Ask")
            self.followup_entry.configure(state="normal")
            self.followup_entry.focus()

        except queue.Empty:
            # No result yet, keep polling
            self.after(100, self._poll_followup_result)

    def _on_edit_click(self):
        """Handle Edit Questions button click."""
        if self.on_edit_questions:
            self.on_edit_questions()
        else:
            messagebox.showinfo(
                "Edit Questions",
                "Question editor will be available in Settings > Q&A > Edit Default Questions"
            )

    def _export_to_csv(self):
        """Export selected Q&A results to CSV file."""
        exportable = [r for r in self._results if r.include_in_export]

        if not exportable:
            messagebox.showwarning(
                "No Q&A Selected",
                "Select at least one Q&A pair to export.\n\n"
                "Click the checkboxes in the Include column."
            )
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="document_qa.csv",
            title="Export Q&A Results"
        )

        if not filepath:
            return

        try:
            content = self._format_csv_export(exportable)
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                f.write(content)

            messagebox.showinfo("Exported", f"Q&A results saved to:\n{filepath}")
            debug_log(f"[QAPanel] Exported {len(exportable)} Q&A pairs to CSV: {filepath}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save file: {e}")
            debug_log(f"[QAPanel] Export error: {e}")

    def _export_to_txt(self):
        """Export selected Q&A results to TXT file."""
        exportable = [r for r in self._results if r.include_in_export]

        if not exportable:
            messagebox.showwarning(
                "No Q&A Selected",
                "Select at least one Q&A pair to export.\n\n"
                "Click the checkboxes in the Include column."
            )
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="document_qa.txt",
            title="Export Q&A Results"
        )

        if not filepath:
            return

        try:
            content = self._format_txt_export(exportable)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            messagebox.showinfo("Exported", f"Q&A results saved to:\n{filepath}")
            debug_log(f"[QAPanel] Exported {len(exportable)} Q&A pairs to TXT: {filepath}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save file: {e}")
            debug_log(f"[QAPanel] Export error: {e}")

    def _format_csv_export(self, results: list[QAResult]) -> str:
        """
        Format results as CSV.

        Args:
            results: List of QAResult objects to export

        Returns:
            CSV string with headers
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header row
        writer.writerow(["Question", "Quick Answer", "Citation", "Source"])

        # Data rows
        for result in results:
            writer.writerow([
                result.question,
                result.quick_answer,
                result.citation,
                result.source_summary
            ])

        return output.getvalue()

    def _format_txt_export(self, results: list[QAResult]) -> str:
        """
        Format results for TXT export.

        Args:
            results: List of QAResult objects to export

        Returns:
            Formatted text string
        """
        lines = [
            "=" * 60,
            "DOCUMENT Q&A SUMMARY",
            "=" * 60,
            ""
        ]

        for i, result in enumerate(results, 1):
            lines.append(f"Q{i}: {result.question}")
            lines.append(f"Quick Answer: {result.quick_answer}")
            lines.append("")
            lines.append(f"Citation: {result.citation}")
            if result.source_summary:
                lines.append(f"   [Source: {result.source_summary}]")
            lines.append("")
            lines.append("-" * 60)
            lines.append("")

        return "\n".join(lines)

    def get_export_content(self) -> str:
        """
        Get exportable content as CSV string.

        Used by DynamicOutputWidget for copy/save operations.

        Returns:
            CSV formatted text of selected Q&A pairs
        """
        exportable = [r for r in self._results if r.include_in_export]
        if not exportable:
            return ""
        return self._format_csv_export(exportable)

    def set_followup_callback(self, callback: Callable[[str], QAResult | None]):
        """
        Set callback for follow-up questions.

        Args:
            callback: Function(question_text) -> QAResult
        """
        self.on_ask_followup = callback

    def set_edit_callback(self, callback: Callable):
        """
        Set callback for Edit Questions button.

        Args:
            callback: Function to call when Edit is clicked
        """
        self.on_edit_questions = callback

    def clear(self):
        """Clear all results and reset display."""
        self._results = []
        self.qa_tree.delete(*self.qa_tree.get_children())
        self.info_label.configure(text="")
