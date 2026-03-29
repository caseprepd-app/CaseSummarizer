"""
Semantic Search Panel Widget for CasePrepd.

Displays search results in plain text format:
- Search query text
- Relevance score (cross-encoder)
- Retrieved passages from documents
- Source file attribution

Features:
- Plain text scrollable display (full text, no truncation)
- Include/exclude toggles for export (Select All/Deselect All)
- Export to CSV, TXT, Word, PDF, HTML
"""

import csv
import io
import logging
import os
from collections.abc import Callable
from tkinter import filedialog, messagebox

import customtkinter as ctk

from src.services import SemanticService

logger = logging.getLogger(__name__)

# Get SemanticResult class from service layer (pipeline architecture)
SemanticResult = SemanticService().get_semantic_result_class()
from src.ui.theme import BUTTON_STYLES, COLORS, FONTS, FRAME_STYLES


class SemanticPanel(ctk.CTkFrame):
    """
    Semantic search display panel with plain text layout.

    Features:
    - Plain text display with full content (no truncation)
    - Include/exclude controls for export (Select All/Deselect All)
    - Export to CSV, TXT, Word, PDF, HTML

    Example:
        panel = SemanticPanel(parent)
        panel.display_results(semantic_results)
    """

    def __init__(
        self,
        master,
        on_edit_questions: Callable | None = None,
        **kwargs,
    ):
        """
        Initialize semantic search panel.

        Args:
            master: Parent widget
            on_edit_questions: Callback when "Edit Questions" is clicked
        """
        super().__init__(master, **kwargs)

        self.on_edit_questions = on_edit_questions

        # Results storage
        self._results: list[SemanticResult] = []
        self._exporting_semantic = False  # Re-entrancy guard for semantic export
        self._edit_dialog_open = False  # Re-entrancy guard for edit dialog

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Table area expands

        # Build UI components
        self._create_header()
        self._create_text_display()
        self._create_button_bar()
        # Follow-up input is in the main window, not here

        logger.debug("SemanticPanel initialized with plain text layout")

    def _create_header(self):
        """Create header with title and info."""
        header = ctk.CTkFrame(self, **FRAME_STYLES["transparent"])
        header.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))

        title = ctk.CTkLabel(header, text="Semantic Search Results", font=FONTS["heading"])
        title.pack(side="left")

        self.info_label = ctk.CTkLabel(
            header, text="", font=FONTS["small"], text_color=COLORS["text_secondary"]
        )
        self.info_label.pack(side="right")

    def _create_text_display(self):
        """Create main plain text display area."""
        display_frame = ctk.CTkFrame(self, **FRAME_STYLES["card"])
        display_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_rowconfigure(1, weight=1)  # Textbox row expands

        # Create scrollable textbox (read-only display)
        self.text_display = ctk.CTkTextbox(
            display_frame,
            wrap="word",
            font=FONTS["body"],
            fg_color=COLORS["bg_darker"],
            text_color=COLORS["text_primary"],
            scrollbar_button_color=COLORS["bg_input"],
            scrollbar_button_hover_color=COLORS["bg_hover"],
            state="disabled",  # Read-only - users shouldn't type here
        )
        self.text_display.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # Inline find bar (hidden by default, shown on Ctrl+F)
        from src.ui.text_find_bar import TextFindBar

        self._find_bar = TextFindBar(display_frame, self.text_display)
        self._find_bar.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 0))
        self._find_bar.grid_remove()

        # Configure text tags for formatting
        # IMPORTANT: CTkTextbox forbids 'font' kwarg in tag_config() - use cnf={} instead
        # CTkTextbox ignores font size in insert(); must use tag_config() instead
        self._apply_text_tags()

        # Re-apply text tags when font size changes live
        self.bind("<<FontChanged>>", lambda e: self._apply_text_tags())

    def _apply_text_tags(self):
        """Apply or re-apply text tags on the search results textbox."""
        from src.ui.theme import SEMANTIC_TEXT_TAGS, resolve_tags

        for tag_name, tag_conf in resolve_tags(SEMANTIC_TEXT_TAGS).items():
            self.text_display.tag_config(tag_name, cnf=tag_conf)

    def show_find_bar(self):
        """Show the inline find bar (triggered by Ctrl+F)."""
        self._find_bar.show()

    def _create_button_bar(self):
        """Create action buttons bar."""
        button_frame = ctk.CTkFrame(self, **FRAME_STYLES["transparent"])
        button_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        # Edit Searches button
        self.edit_btn = ctk.CTkButton(
            button_frame, text="Edit Searches", command=self._on_edit_click, width=130
        )
        self.edit_btn.pack(side="left", padx=(0, 5))

        # Copy to Clipboard button
        self.copy_btn = ctk.CTkButton(
            button_frame,
            text="Copy to Clipboard",
            command=self._copy_to_clipboard,
            width=120,
            **BUTTON_STYLES["secondary"],
        )
        self.copy_btn.pack(side="left", padx=5)

        # Export dropdown (CTkOptionMenu doesn't support hover_color, so only use fg_color)
        self.export_dropdown = ctk.CTkOptionMenu(
            button_frame,
            values=["Export...", "TXT", "CSV", "Word (.docx)", "PDF", "HTML"],
            command=self._on_export_format_selected,
            width=120,
            fg_color=BUTTON_STYLES["secondary"]["fg_color"],
        )
        self.export_dropdown.pack(side="right", padx=5)

        # Select All / Deselect All buttons
        self.select_all_btn = ctk.CTkButton(
            button_frame,
            text="Select All",
            command=lambda: self._set_all_include(True),
            width=80,
            **BUTTON_STYLES["primary"],
        )
        self.select_all_btn.pack(side="right", padx=5)

        self.deselect_all_btn = ctk.CTkButton(
            button_frame,
            text="Deselect All",
            command=lambda: self._set_all_include(False),
            width=90,
            **BUTTON_STYLES["secondary"],
        )
        self.deselect_all_btn.pack(side="right", padx=5)

    def display_results(self, results: list[SemanticResult]):
        """
        Display search results as plain text with relevance scores.

        Args:
            results: List of SemanticResult objects to display
        """
        self._results = results

        # Clear existing content
        self.text_display.configure(state="normal")
        self.text_display.delete("1.0", "end")

        for i, result in enumerate(results, 1):
            # Search header with relevance score on the same line
            question_tag = "question_default" if result.is_default_question else "question"
            self.text_display.insert("end", f"Search {i}:", question_tag)
            if result.relevance > 0:
                relevance_pct = int(result.relevance * 100)
                self.text_display.insert("end", f"  [Relevance: {relevance_pct}%]", "score_detail")
            self.text_display.insert("end", "\n")
            self.text_display.insert("end", f"{result.question}\n\n", "answer")

            # Retrieved passages
            self.text_display.insert("end", "Relevant Passages:\n", "label")
            passage_text = result.citation if result.citation else "(no results found)"
            self.text_display.insert("end", f"{passage_text}\n\n", "citation")

            # Source attribution
            self.text_display.insert("end", "Source:\n", "label")
            source_text = result.source_summary if result.source_summary else "(source unknown)"
            self.text_display.insert("end", f"{source_text}\n\n", "source")

            # Separator between results (except after last one)
            if i < len(results):
                separator = "─" * 80 + "\n\n"
                self.text_display.insert("end", separator, "separator")

        # Make read-only
        self.text_display.configure(state="disabled")

        # Update info label
        included = sum(1 for r in results if r.include_in_export)
        self.info_label.configure(text=f"{included}/{len(results)} selected for export")

        logger.debug("Displaying %s search results", len(results))

    def _set_all_include(self, include: bool):
        """Set include_in_export for all results."""
        for result in self._results:
            result.include_in_export = include

        # Refresh display
        self.display_results(self._results)

    def _on_edit_click(self):
        """Handle Edit Searches button click."""
        if self._edit_dialog_open:
            return
        if self.on_edit_questions:
            self._edit_dialog_open = True
            try:
                self.on_edit_questions()
            finally:
                self._edit_dialog_open = False
        else:
            messagebox.showinfo(
                "Edit Searches",
                "You can edit default searches in Settings > Search",
            )

    def _export_semantic(self, format_key: str):
        """
        Export semantic search results in the given format.

        Shared boilerplate: filter exportable, empty check, file dialog,
        write, save path, status bar, error handling.

        Args:
            format_key: One of "csv", "txt", "word", "pdf", "html"
        """
        if self._exporting_semantic:
            return
        self._exporting_semantic = True
        self._set_export_dropdown_enabled(False)

        try:
            self._export_semantic_impl(format_key)
        except Exception:
            logger.error("Semantic export failed", exc_info=True)
        finally:
            self._exporting_semantic = False
            self._set_export_dropdown_enabled(True)

    def _export_semantic_impl(self, format_key: str):
        """Implementation of _export_semantic, guarded by _exporting_semantic flag."""
        from pathlib import Path

        from src.services import DocumentService, get_export_service
        from src.user_preferences import get_user_preferences

        format_info = {
            "csv": (".csv", [("CSV files", "*.csv"), ("All files", "*.*")]),
            "txt": (".txt", [("Text files", "*.txt"), ("All files", "*.*")]),
            "word": (".docx", [("Word documents", "*.docx"), ("All files", "*.*")]),
            "pdf": (".pdf", [("PDF documents", "*.pdf"), ("All files", "*.*")]),
            "html": (".html", [("HTML files", "*.html"), ("All files", "*.*")]),
        }
        ext, filetypes = format_info[format_key]

        exportable = [r for r in self._results if r.include_in_export]

        if not exportable:
            messagebox.showwarning(
                "No Results Selected",
                "Select at least one search result to export.\n\n"
                "Use 'Select All' or click individual results to include them.",
            )
            return

        prefs = get_user_preferences()
        initial_dir = (
            prefs.get("last_export_path") or DocumentService().get_default_documents_folder()
        )

        filepath = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=filetypes,
            initialfile=f"document_search_results{ext}",
            initialdir=initial_dir,
            title="Export Search Results",
        )

        if not filepath:
            return

        try:
            error_detail = None
            # Write the file
            if format_key == "csv":
                content = self._format_csv_export(exportable)
                with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
                    f.write(content)
                success = True
            elif format_key == "txt":
                content = self._format_txt_export(exportable)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                success = True
            else:
                export_service = get_export_service()
                write_fn = {
                    "word": export_service.export_semantic_to_word,
                    "pdf": export_service.export_semantic_to_pdf,
                    "html": export_service.export_semantic_to_html,
                }
                success, error_detail = write_fn[format_key](exportable, filepath)

            if success:
                prefs.set("last_export_path", str(Path(filepath).parent))
                main_window = self.winfo_toplevel()
                if hasattr(main_window, "set_status"):
                    filename = os.path.basename(filepath)
                    result_word = "result" if len(exportable) == 1 else "results"
                    main_window.set_status(
                        f"Exported {len(exportable)} search {result_word} to {filename}",
                        duration_ms=5000,
                    )
                logger.debug(
                    "Exported %s search results to %s: %s", len(exportable), format_key, filepath
                )
            else:
                detail = f"\n\n{error_detail}" if error_detail else ""
                messagebox.showerror("Export Error", f"Failed to create {ext} file.{detail}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save file: {e}")
            logger.debug("Export error: %s", e)

    def _export_to_csv(self):
        """Export selected search results to CSV file."""
        self._export_semantic("csv")

    def _export_to_txt(self):
        """Export selected search results to TXT file."""
        self._export_semantic("txt")

    def _export_to_word(self):
        """Export selected search results to Word document."""
        self._export_semantic("word")

    def _export_to_pdf(self):
        """Export selected search results to PDF document."""
        self._export_semantic("pdf")

    def _export_to_html(self):
        """Export selected search results to interactive HTML file."""
        self._export_semantic("html")

    def _on_export_format_selected(self, choice: str):
        """
        Handle export format selection from dropdown.

        Args:
            choice: Selected format ("Export...", "TXT", "CSV", "Word (.docx)", "PDF", "HTML")
        """
        if choice == "Export...":
            return  # Placeholder, do nothing

        format_map = {
            "TXT": "txt",
            "CSV": "csv",
            "Word (.docx)": "word",
            "PDF": "pdf",
            "HTML": "html",
        }
        if choice in format_map:
            self._export_semantic(format_map[choice])

        # Reset dropdown to placeholder
        self.export_dropdown.set("Export...")

    def _set_export_dropdown_enabled(self, enabled: bool):
        """Enable or disable the export dropdown during export."""
        try:
            state = "normal" if enabled else "disabled"
            self.export_dropdown.configure(state=state)
        except Exception as e:
            logger.debug("Export dropdown state change failed: %s", e)

    def _format_csv_export(self, results: list[SemanticResult]) -> str:
        """
        Format results as CSV.

        Args:
            results: List of SemanticResult objects to export

        Returns:
            CSV string with headers
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header row
        writer.writerow(["Search Query", "Relevant Passages", "Source"])

        # Data rows
        for result in results:
            writer.writerow([result.question, result.citation, result.source_summary])

        return output.getvalue()

    def _format_txt_export(self, results: list[SemanticResult]) -> str:
        """
        Format results for TXT export.

        Args:
            results: List of SemanticResult objects to export

        Returns:
            Formatted text string
        """
        lines = ["=" * 60, "SEMANTIC SEARCH RESULTS", "=" * 60, ""]

        for i, result in enumerate(results, 1):
            lines.append(f"Search {i}: {result.question}")
            lines.append("")
            lines.append(f"Relevant Passages: {result.citation}")
            if result.source_summary:
                lines.append(f"   [Source: {result.source_summary}]")
            lines.append("")
            lines.append("-" * 60)
            lines.append("")

        return "\n".join(lines)

    def _copy_to_clipboard(self):
        """
        Copy selected semantic search results to clipboard.

        Copies in a readable text format suitable for pasting into documents.
        Uses brief button flash + status bar confirmation for better UX.
        """
        exportable = [r for r in self._results if r.include_in_export]

        if not exportable:
            messagebox.showwarning(
                "No Results Selected",
                "Select at least one search result to copy.\n\n"
                "Use 'Select All' to select all results.",
            )
            return

        # Format for clipboard (readable text format)
        lines = []
        for i, result in enumerate(exportable, 1):
            lines.append(f"Search {i}: {result.question}")
            lines.append(f"{result.citation}")
            if result.source_summary:
                lines.append(f"   [Source: {result.source_summary}]")
            lines.append("")

        content = "\n".join(lines)

        # Copy to clipboard
        try:
            self.clipboard_clear()
            self.clipboard_append(content)

            # Brief button flash for immediate feedback
            original_text = self.copy_btn.cget("text")
            self.copy_btn.configure(text=f"Copied {len(exportable)}!")

            def _reset_copy_btn():
                try:
                    self.copy_btn.configure(text=original_text)
                except Exception:
                    logger.debug("Copy button reset skipped (widget destroyed)")

            self.after(1500, _reset_copy_btn)

            # Status bar confirmation
            main_window = self.winfo_toplevel()
            if hasattr(main_window, "set_status"):
                result_word = "result" if len(exportable) == 1 else "results"
                main_window.set_status(
                    f"Copied {len(exportable)} search {result_word} to clipboard",
                    duration_ms=5000,
                )

            logger.debug("Copied %s search results to clipboard", len(exportable))

        except Exception as e:
            messagebox.showerror("Copy Failed", f"Could not copy to clipboard:\n{e}")
            logger.debug("Clipboard copy failed: %s", e)

    def get_export_content(self) -> str:
        """
        Get exportable content as CSV string.

        Used by DynamicOutputWidget for copy/save operations.

        Returns:
            CSV formatted text of selected semantic search results
        """
        exportable = [r for r in self._results if r.include_in_export]
        if not exportable:
            return ""
        return self._format_csv_export(exportable)

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
        self.text_display.configure(state="normal")
        self.text_display.delete("1.0", "end")
        self.text_display.configure(state="disabled")
        self.info_label.configure(text="")
