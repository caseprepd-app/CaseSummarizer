"""
Export Functions Mixin.

Session 82: Extracted from main_window.py for modularity.

Contains:
- Export All functionality
- Combined Report export (Word/PDF)
"""

import os
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

from src.logging_config import debug_log


class ExportMixin:
    """
    Mixin class providing export functionality.

    Requires parent class to have:
    - self.output_display: DynamicOutputWidget
    - self._qa_results: List of QA results
    - self.export_all_btn: Export All button
    - self.combined_report_btn: Combined Report button
    """

    def _export_all(self):
        """
        Export all results (vocabulary, Q&A, summary) to Documents folder.

        Creates timestamped files for each output type that has data.
        """
        from src.core.utils.text_utils import get_documents_folder

        documents_path = get_documents_folder()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported = []

        # Export vocabulary CSV
        vocab_data = self.output_display._outputs.get("Names & Vocabulary", [])
        if vocab_data:
            csv_content = self.output_display._build_vocab_csv(vocab_data)
            vocab_path = os.path.join(documents_path, f"vocabulary_{timestamp}.csv")
            with open(vocab_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_content)
            exported.append(f"Vocabulary: {len(vocab_data)} terms")

        # Export Q&A results
        if self._qa_results:
            qa_panel = self.output_display._qa_panel
            if qa_panel:
                # Select all for export
                for r in qa_panel._results:
                    r.include_in_export = True
                txt_content = qa_panel._format_txt_export(qa_panel._results)
                qa_path = os.path.join(documents_path, f"qa_results_{timestamp}.txt")
                with open(qa_path, "w", encoding="utf-8") as f:
                    f.write(txt_content)
                exported.append(f"Q&A: {len(qa_panel._results)} answers")

        # Export summary
        summary = self.output_display._outputs.get("Summary", "")
        if not summary:
            summary = self.output_display._outputs.get("Meta-Summary", "")
        if summary and summary.strip():
            summary_path = os.path.join(documents_path, f"summary_{timestamp}.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            exported.append("Summary")

        # Flash button and show result
        if exported:
            self.export_all_btn.configure(text=f"Exported!")
            self.after(1500, lambda: self.export_all_btn.configure(text="Export All"))
            # Status bar with auto-clear (Session 69)
            self.set_status(f"Exported to Documents: {', '.join(exported)}", duration_ms=5000)
            debug_log(f"[MainWindow] Export All: {exported}")
        else:
            messagebox.showwarning("No Data", "No results to export yet.")

    def _export_combined_report(self):
        """
        Export vocabulary and Q&A together in a single Word document.

        Session 73: Combined export feature - creates unified report.
        """
        from src.services import get_export_service
        from src.user_preferences import get_user_preferences
        from src.core.utils.text_utils import get_documents_folder

        # Gather data
        vocab_data = self.output_display._outputs.get("Names & Vocabulary", [])
        qa_results = []
        if self._qa_results:
            qa_panel = self.output_display._qa_panel
            if qa_panel and qa_panel._results:
                qa_results = qa_panel._results

        if not vocab_data and not qa_results:
            messagebox.showwarning("No Data", "No results to export yet.")
            return

        # Get initial directory (last export path or Documents)
        prefs = get_user_preferences()
        initial_dir = prefs.get("last_export_path") or get_documents_folder()

        # Ask for save location with format choice
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = filedialog.asksaveasfilename(
            defaultextension=".docx",
            filetypes=[
                ("Word documents", "*.docx"),
                ("PDF documents", "*.pdf"),
                ("All files", "*.*"),
            ],
            initialfile=f"combined_report_{timestamp}.docx",
            initialdir=initial_dir,
            title="Export Combined Report",
        )

        if not filepath:
            return

        # Determine format from extension
        export_service = get_export_service()
        ext = Path(filepath).suffix.lower()

        if ext == ".pdf":
            success = export_service.export_combined_to_pdf(vocab_data, qa_results, filepath)
        else:
            success = export_service.export_combined_to_word(vocab_data, qa_results, filepath)

        if success:
            # Remember export folder
            prefs.set("last_export_path", str(Path(filepath).parent))

            # Flash button and status
            self.combined_report_btn.configure(text="Exported!")
            self.after(1500, lambda: self.combined_report_btn.configure(text="Combined Report"))

            filename = Path(filepath).name
            term_count = len(vocab_data)
            qa_count = len(qa_results)
            self.set_status(
                f"Combined report: {term_count} terms + {qa_count} Q&A → {filename}",
                duration_ms=5000,
            )
            debug_log(f"[MainWindow] Combined report exported: {filepath}")
        else:
            messagebox.showerror("Export Failed", "Failed to create combined report.")
