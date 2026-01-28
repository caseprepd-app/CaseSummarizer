"""
Export Functions Mixin.

Session 82: Extracted from main_window.py for modularity.

Contains:
- Export All functionality (tabbed HTML export)
"""

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
    """

    def _export_all(self):
        """
        Export all results (vocabulary, Q&A, summary) to a single tabbed HTML file.

        Opens a save dialog defaulting to .html. Gathers score-filtered vocab,
        answered Q&A, and summary text into one combined HTML document.
        """
        from src.services import DocumentService, get_export_service
        from src.user_preferences import get_user_preferences

        # Gather filtered vocabulary data
        vocab_data = self.output_display._get_filtered_vocab_data()

        # Gather answered Q&A results only
        qa_results = []
        if self._qa_results:
            qa_panel = self.output_display._qa_panel
            if qa_panel and qa_panel._results:
                qa_results = [r for r in qa_panel._results if r.is_exportable]

        # Gather summary text
        summary = self.output_display._outputs.get("Summary", "")
        if not summary:
            summary = self.output_display._outputs.get("Meta-Summary", "")
        summary_text = summary.strip() if summary else ""

        # Check we have something to export
        if not vocab_data and not qa_results and not summary_text:
            messagebox.showwarning("No Data", "No results to export yet.")
            return

        # Get initial directory (last export path or Documents)
        prefs = get_user_preferences()
        doc_service = DocumentService()
        initial_dir = prefs.get("last_export_path") or doc_service.get_default_documents_folder()

        # Open save dialog
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[
                ("HTML files", "*.html"),
                ("All files", "*.*"),
            ],
            initialfile=f"case_report_{timestamp}.html",
            initialdir=initial_dir,
            title="Export All Results",
        )

        if not filepath:
            return

        # Get visible columns from vocab display
        visible_columns = self.output_display._get_visible_columns()

        # Export combined HTML
        export_service = get_export_service()
        success = export_service.export_combined_html(
            vocab_data=vocab_data,
            qa_results=qa_results,
            summary_text=summary_text,
            file_path=filepath,
            visible_columns=visible_columns,
        )

        if success:
            # Remember export folder
            prefs.set("last_export_path", str(Path(filepath).parent))

            # Flash button and status
            self.export_all_btn.configure(text="Exported!")
            self.after(1500, lambda: self.export_all_btn.configure(text="Export All"))

            filename = Path(filepath).name
            parts = []
            if vocab_data:
                parts.append(f"{len(vocab_data)} terms")
            if qa_results:
                parts.append(f"{len(qa_results)} Q&A")
            if summary_text:
                parts.append("summary")
            self.set_status(f"Exported {' + '.join(parts)} to {filename}", duration_ms=5000)
            debug_log(f"[MainWindow] Export All HTML: {filepath}")
        else:
            messagebox.showerror("Export Failed", "Failed to create HTML report.")
