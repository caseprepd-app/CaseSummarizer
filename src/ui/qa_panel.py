"""
Q&A Panel Widget for CasePrepd.

Displays Q&A results in plain text format:
- Question: The question asked
- Answer: AI-synthesized answer from Ollama
- Citation: Raw text excerpts from BM25+/vector retrieval
- Source: Formatted source metadata (document names, sections)

Features:
- Plain text scrollable display (full text, no truncation)
- Include/exclude toggles for export (Select All/Deselect All)
- Export to CSV or TXT
"""

import csv
import io
import logging
import os
from collections.abc import Callable
from tkinter import filedialog, messagebox

import customtkinter as ctk

from src.services import QAService

logger = logging.getLogger(__name__)

# Get QAResult class from service layer (pipeline architecture)
QAResult = QAService().get_qa_result_class()
from src.ui.theme import BUTTON_STYLES, COLORS, FONTS, FRAME_STYLES, QA_TEXT_TAGS


class QAPanel(ctk.CTkFrame):
    """
    Q&A display panel with plain text layout.

    Features:
    - Plain text display with full content (no truncation)
    - Include/exclude controls for export (Select All/Deselect All)
    - Export to CSV or TXT
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
        **kwargs,
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

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Table area expands

        # Build UI components
        self._create_header()
        self._create_text_display()
        self._create_button_bar()
        # Follow-up input is in the main window, not here

        logger.debug("QAPanel initialized with plain text layout")

    def _create_header(self):
        """Create header with title and info."""
        header = ctk.CTkFrame(self, **FRAME_STYLES["transparent"])
        header.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))

        title = ctk.CTkLabel(header, text="Questions & Answers", font=FONTS["heading"])
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
        # See RESEARCH_LOG.md "Q&A Follow-up Font Scaling Error"
        from src.ui.theme import resolve_tags

        for tag_name, tag_config in resolve_tags(QA_TEXT_TAGS).items():
            self.text_display.tag_config(tag_name, cnf=tag_config)

    def show_find_bar(self):
        """Show the inline find bar (triggered by Ctrl+F)."""
        self._find_bar.show()

    def _create_button_bar(self):
        """Create action buttons bar."""
        button_frame = ctk.CTkFrame(self, **FRAME_STYLES["transparent"])
        button_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        # Edit Questions button
        self.edit_btn = ctk.CTkButton(
            button_frame, text="Edit Questions", command=self._on_edit_click, width=130
        )
        self.edit_btn.pack(side="left", padx=(0, 5))

        # "Ask More Questions" is handled by the follow-up input at bottom of main window

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

    def display_results(self, results: list[QAResult]):
        """
        Display Q&A results as plain text with verification highlighting.

        Args:
            results: List of QAResult objects to display
        """
        self._results = results

        # Clear existing content
        self.text_display.configure(state="normal")
        self.text_display.delete("1.0", "end")

        # Track if any results have verification (for legend)
        has_verification = any(r.verification is not None for r in results)

        # Format and insert each Q&A pair
        for i, result in enumerate(results, 1):
            # Question number and text - use different tag for defaults
            question_tag = "question_default" if result.is_default_question else "question"
            self.text_display.insert("end", f"Question {i}:\n", question_tag)
            self.text_display.insert("end", f"{result.question}\n\n", "answer")

            # Answer section with verification if available
            self.text_display.insert("end", "Answer:\n", "label")

            # Track if answer was rejected (used to hide citation/source)
            answer_rejected = False

            if result.verification:
                # Show retrieval score then reliability header (same font weight)
                retrieval_pct = int(result.confidence * 100)
                self.text_display.insert("end", f"[Retrieval: {retrieval_pct}%]\n", "score_detail")
                self._insert_reliability_header(result.verification)

                if result.verification.answer_rejected:
                    # Show rejection message in unreliable color
                    self.text_display.insert(
                        "end", f"{result.quick_answer}\n\n", "verify_unreliable"
                    )
                    answer_rejected = True
                else:
                    # Show color-coded verified answer
                    self._insert_verified_answer(result.verification)
            else:
                # No verification - show plain answer with retrieval score
                if result.confidence > 0:
                    retrieval_pct = int(result.confidence * 100)
                    self.text_display.insert(
                        "end", f"[Retrieval: {retrieval_pct}%]\n", "score_detail"
                    )
                self.text_display.insert("end", f"{result.quick_answer}\n\n", "answer")

            # Only show citation and source if answer was NOT rejected
            # (showing citation for rejected answers is confusing - implies content exists but is hidden)
            if not answer_rejected:
                # Citation label and text
                self.text_display.insert("end", "Citation:\n", "label")
                citation_text = result.citation if result.citation else "(no citation available)"
                self.text_display.insert("end", f"{citation_text}\n\n", "citation")

                # Source label and text
                self.text_display.insert("end", "Source:\n", "label")
                source_text = result.source_summary if result.source_summary else "(source unknown)"
                self.text_display.insert("end", f"{source_text}\n\n", "source")

            # Separator between Q&A pairs (except after last one)
            if i < len(results):
                separator = "─" * 80 + "\n\n"
                self.text_display.insert("end", separator, "separator")

        # Show verification legend at bottom if any results were verified
        if has_verification:
            self._insert_verification_legend()

        # Make read-only
        self.text_display.configure(state="disabled")

        # Update info label
        included = sum(1 for r in results if r.include_in_export)
        self.info_label.configure(text=f"{included}/{len(results)} selected for export")

        logger.debug("Displaying %s results in plain text format", len(results))

    def _insert_reliability_header(self, verification):
        """
        Insert bold reliability score header before the answer.

        Args:
            verification: VerificationResult with overall_reliability score
        """
        from src.services import QAService

        reliability_pct = int(verification.overall_reliability * 100)
        level = QAService().get_reliability_level(verification.overall_reliability)

        # Map level to tag and label
        tag_map = {
            "high": ("reliability_high", "HIGH"),
            "medium": ("reliability_medium", "MEDIUM"),
            "low": ("reliability_low", "LOW - REJECTED"),
        }
        tag, label = tag_map.get(level, ("reliability_medium", "UNKNOWN"))

        self.text_display.insert("end", f"[Reliability: {reliability_pct}% - {label}]\n", tag)

    def _insert_verified_answer(self, verification):
        """
        Insert answer text with span-level color coding based on hallucination probability.

        Args:
            verification: VerificationResult with spans and probabilities
        """
        from src.services import QAService

        qa_svc = QAService()
        for span in verification.spans:
            category = qa_svc.get_span_category(span.hallucination_prob)
            tag = f"verify_{category}"
            self.text_display.insert("end", span.text, tag)

        self.text_display.insert("end", "\n\n")

    def _insert_verification_legend(self):
        """Insert color legend and score explanations at bottom of display."""
        self.text_display.insert("end", "\n")
        self.text_display.insert("end", "─" * 80 + "\n", "separator")
        self.text_display.insert("end", "Verification Legend: ", "legend_label")

        # Legend items with their tags
        legend_items = [
            ("Verified ", "verify_verified"),
            ("Uncertain ", "verify_uncertain"),
            ("Suspicious ", "verify_suspicious"),
            ("Unreliable ", "verify_unreliable"),
            ("Hallucinated", "verify_hallucinated"),
        ]

        for label, tag in legend_items:
            self.text_display.insert("end", label, tag)
            self.text_display.insert("end", " ", "answer")  # Space between items

        self.text_display.insert("end", "\n\n", "answer")

        # Score explanations
        self.text_display.insert(
            "end",
            "Reliability = How well the answer is supported by the source text "
            "(anti-hallucination check)\n",
            "score_detail",
        )
        self.text_display.insert(
            "end",
            "Retrieval = How semantically relevant the retrieved document "
            "chunks are to the question\n",
            "score_detail",
        )

    def _set_all_include(self, include: bool):
        """Set include_in_export for all results."""
        for result in self._results:
            result.include_in_export = include

        # Refresh display
        self.display_results(self._results)

    def _on_edit_click(self):
        """Handle Edit Questions button click."""
        if self.on_edit_questions:
            self.on_edit_questions()
        else:
            messagebox.showinfo(
                "Edit Questions",
                "You can edit default questions in Settings > Q&A",
            )

    def _export_qa(self, format_key: str):
        """
        Export Q&A results in the given format.

        Shared boilerplate: filter exportable, empty check, file dialog,
        write, save path, status bar, error handling.

        Args:
            format_key: One of "csv", "txt", "word", "pdf", "html"
        """
        if getattr(self, "_exporting_qa", False):
            return
        self._exporting_qa = True

        try:
            self._export_qa_impl(format_key)
        except Exception:
            logger.error("Q&A export failed", exc_info=True)
        finally:
            self._exporting_qa = False

    def _export_qa_impl(self, format_key: str):
        """Implementation of _export_qa, guarded by _exporting_qa flag."""
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
                "No Q&A Selected",
                "Select at least one Q&A pair to export.\n\n"
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
            initialfile=f"document_questions{ext}",
            initialdir=initial_dir,
            title="Export Q&A Results",
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
                    "word": export_service.export_qa_to_word,
                    "pdf": export_service.export_qa_to_pdf,
                    "html": export_service.export_qa_to_html,
                }
                success, error_detail = write_fn[format_key](exportable, filepath)

            if success:
                prefs.set("last_export_path", str(Path(filepath).parent))
                main_window = self.winfo_toplevel()
                if hasattr(main_window, "set_status"):
                    filename = os.path.basename(filepath)
                    pair_word = "pair" if len(exportable) == 1 else "pairs"
                    main_window.set_status(
                        f"Exported {len(exportable)} Q&A {pair_word} to {filename}",
                        duration_ms=5000,
                    )
                logger.debug(
                    "Exported %s Q&A pairs to %s: %s", len(exportable), format_key, filepath
                )
            else:
                detail = f"\n\n{error_detail}" if error_detail else ""
                messagebox.showerror("Export Error", f"Failed to create {ext} file.{detail}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save file: {e}")
            logger.debug("Export error: %s", e)

    def _export_to_csv(self):
        """Export selected Q&A results to CSV file."""
        self._export_qa("csv")

    def _export_to_txt(self):
        """Export selected Q&A results to TXT file."""
        self._export_qa("txt")

    def _export_to_word(self):
        """Export selected Q&A results to Word document."""
        self._export_qa("word")

    def _export_to_pdf(self):
        """Export selected Q&A results to PDF document."""
        self._export_qa("pdf")

    def _export_to_html(self):
        """Export selected Q&A results to interactive HTML file."""
        self._export_qa("html")

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
            self._export_qa(format_map[choice])

        # Reset dropdown to placeholder
        self.export_dropdown.set("Export...")

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
            writer.writerow(
                [result.question, result.quick_answer, result.citation, result.source_summary]
            )

        return output.getvalue()

    def _format_txt_export(self, results: list[QAResult]) -> str:
        """
        Format results for TXT export.

        Args:
            results: List of QAResult objects to export

        Returns:
            Formatted text string
        """
        lines = ["=" * 60, "DOCUMENT QUESTIONS & ANSWERS", "=" * 60, ""]

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

    def _copy_to_clipboard(self):
        """
        Copy selected Q&A results to clipboard.

        Copies in a readable text format suitable for pasting into documents.
        Uses brief button flash + status bar confirmation for better UX.
        """
        exportable = [r for r in self._results if r.include_in_export]

        if not exportable:
            messagebox.showwarning(
                "No Q&A Selected",
                "Select at least one Q&A pair to copy.\n\nUse 'Select All' to select all results.",
            )
            return

        # Format for clipboard (readable text format)
        lines = []
        for _i, result in enumerate(exportable, 1):
            lines.append(f"Q: {result.question}")
            lines.append(f"A: {result.quick_answer}")
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
                pair_word = "pair" if len(exportable) == 1 else "pairs"
                main_window.set_status(
                    f"Copied {len(exportable)} Q&A {pair_word} to clipboard", duration_ms=5000
                )

            logger.debug("Copied %s Q&A pairs to clipboard", len(exportable))

        except Exception as e:
            messagebox.showerror("Copy Failed", f"Could not copy to clipboard:\n{e}")
            logger.debug("Clipboard copy failed: %s", e)

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
        self.text_display.configure(state="normal")
        self.text_display.delete("1.0", "end")
        self.text_display.configure(state="disabled")
        self.info_label.configure(text="")
