"""
Export Service

Provides high-level API for exporting vocabulary and Q&A to Word/PDF/TXT/HTML.

Session 73: Added auto-open exported files feature (configurable).
"""

import os
import sys
from pathlib import Path

from src.core.export import (
    WordDocumentBuilder,
    PdfDocumentBuilder,
    export_vocabulary,
    export_vocabulary_txt,
    export_vocabulary_html,
    export_qa_results,
    export_qa_html,
    export_combined,
)
from src.logging_config import debug_log, error, info


def _auto_open_file(file_path: str) -> None:
    """
    Open a file with the system's default application if auto-open is enabled.

    Session 73: Respects user preference 'auto_open_exports'.

    Args:
        file_path: Path to the file to open
    """
    from src.user_preferences import get_user_preferences

    prefs = get_user_preferences()
    if not prefs.get("auto_open_exports", True):
        return  # Auto-open disabled

    try:
        if sys.platform == "win32":
            os.startfile(file_path)
        elif sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", file_path], check=False)
        else:  # Linux/Unix
            import subprocess
            subprocess.run(["xdg-open", file_path], check=False)
        debug_log(f"[EXPORT] Auto-opened: {file_path}")
    except Exception as e:
        debug_log(f"[EXPORT] Auto-open failed: {e}")


class ExportService:
    """
    Service for exporting data to Word and PDF formats.

    Provides a simple API for UI components to export vocabulary
    and Q&A results without knowing the implementation details.
    """

    def export_vocabulary_to_word(
        self,
        vocab_data: list[dict],
        file_path: str,
        include_details: bool = False
    ) -> bool:
        """
        Export vocabulary to Word document.

        Args:
            vocab_data: List of vocabulary dicts
            file_path: Output file path (.docx)
            include_details: Include algorithm columns

        Returns:
            True if successful, False otherwise
        """
        try:
            debug_log(f"[EXPORT] Exporting {len(vocab_data)} terms to Word: {file_path}")
            builder = WordDocumentBuilder()
            export_vocabulary(vocab_data, builder, include_details)
            builder.save(file_path)
            info(f"Exported vocabulary to {file_path}")
            _auto_open_file(file_path)  # Session 73
            return True
        except Exception as e:
            error(f"Failed to export vocabulary to Word: {e}")
            return False

    def export_vocabulary_to_pdf(
        self,
        vocab_data: list[dict],
        file_path: str,
        include_details: bool = False
    ) -> bool:
        """
        Export vocabulary to PDF document.

        Args:
            vocab_data: List of vocabulary dicts
            file_path: Output file path (.pdf)
            include_details: Include algorithm columns

        Returns:
            True if successful, False otherwise
        """
        try:
            debug_log(f"[EXPORT] Exporting {len(vocab_data)} terms to PDF: {file_path}")
            builder = PdfDocumentBuilder()
            export_vocabulary(vocab_data, builder, include_details)
            builder.save(file_path)
            info(f"Exported vocabulary to {file_path}")
            _auto_open_file(file_path)  # Session 73
            return True
        except Exception as e:
            error(f"Failed to export vocabulary to PDF: {e}")
            return False

    def export_qa_to_word(
        self,
        results: list,
        file_path: str,
        include_verification: bool = True
    ) -> bool:
        """
        Export Q&A results to Word document.

        Args:
            results: List of QAResult objects
            file_path: Output file path (.docx)
            include_verification: Include verification coloring

        Returns:
            True if successful, False otherwise
        """
        try:
            debug_log(f"[EXPORT] Exporting {len(results)} Q&A pairs to Word: {file_path}")
            builder = WordDocumentBuilder()
            export_qa_results(results, builder, include_verification)
            builder.save(file_path)
            info(f"Exported Q&A to {file_path}")
            _auto_open_file(file_path)  # Session 73
            return True
        except Exception as e:
            error(f"Failed to export Q&A to Word: {e}")
            return False

    def export_qa_to_pdf(
        self,
        results: list,
        file_path: str,
        include_verification: bool = True
    ) -> bool:
        """
        Export Q&A results to PDF document.

        Args:
            results: List of QAResult objects
            file_path: Output file path (.pdf)
            include_verification: Include verification coloring

        Returns:
            True if successful, False otherwise
        """
        try:
            debug_log(f"[EXPORT] Exporting {len(results)} Q&A pairs to PDF: {file_path}")
            builder = PdfDocumentBuilder()
            export_qa_results(results, builder, include_verification)
            builder.save(file_path)
            info(f"Exported Q&A to {file_path}")
            _auto_open_file(file_path)  # Session 73
            return True
        except Exception as e:
            error(f"Failed to export Q&A to PDF: {e}")
            return False

    def export_vocabulary_to_txt(
        self,
        vocab_data: list[dict],
        file_path: str
    ) -> bool:
        """
        Export vocabulary to plain text (one term per line).

        Args:
            vocab_data: List of vocabulary dicts
            file_path: Output file path (.txt)

        Returns:
            True if successful, False otherwise
        """
        try:
            debug_log(f"[EXPORT] Exporting {len(vocab_data)} terms to TXT: {file_path}")
            success = export_vocabulary_txt(vocab_data, file_path)
            if success:
                info(f"Exported vocabulary to {file_path}")
                _auto_open_file(file_path)  # Session 73
            return success
        except Exception as e:
            error(f"Failed to export vocabulary to TXT: {e}")
            return False

    def export_vocabulary_to_html(
        self,
        vocab_data: list[dict],
        file_path: str
    ) -> bool:
        """
        Export vocabulary to interactive HTML.

        Args:
            vocab_data: List of vocabulary dicts
            file_path: Output file path (.html)

        Returns:
            True if successful, False otherwise
        """
        try:
            debug_log(f"[EXPORT] Exporting {len(vocab_data)} terms to HTML: {file_path}")
            success = export_vocabulary_html(vocab_data, file_path)
            if success:
                info(f"Exported vocabulary to {file_path}")
                _auto_open_file(file_path)  # Session 73
            return success
        except Exception as e:
            error(f"Failed to export vocabulary to HTML: {e}")
            return False

    def export_qa_to_html(
        self,
        results: list,
        file_path: str,
        include_verification: bool = True
    ) -> bool:
        """
        Export Q&A results to interactive HTML.

        Args:
            results: List of QAResult objects
            file_path: Output file path (.html)
            include_verification: Include verification coloring

        Returns:
            True if successful, False otherwise
        """
        try:
            debug_log(f"[EXPORT] Exporting {len(results)} Q&A pairs to HTML: {file_path}")
            success = export_qa_html(results, file_path, include_verification)
            if success:
                info(f"Exported Q&A to {file_path}")
                _auto_open_file(file_path)  # Session 73
            return success
        except Exception as e:
            error(f"Failed to export Q&A to HTML: {e}")
            return False

    def export_combined_to_word(
        self,
        vocab_data: list[dict],
        qa_results: list,
        file_path: str,
        include_vocab_details: bool = False,
        include_qa_verification: bool = True
    ) -> bool:
        """
        Export vocabulary and Q&A together to a single Word document.

        Session 73: Combined export feature.

        Args:
            vocab_data: List of vocabulary dicts
            qa_results: List of QAResult objects
            file_path: Output file path (.docx)
            include_vocab_details: Include algorithm columns
            include_qa_verification: Include verification coloring

        Returns:
            True if successful, False otherwise
        """
        try:
            debug_log(f"[EXPORT] Combined export to Word: {len(vocab_data)} terms, {len(qa_results)} Q&A pairs")
            builder = WordDocumentBuilder()
            export_combined(vocab_data, qa_results, builder, include_vocab_details, include_qa_verification)
            builder.save(file_path)
            info(f"Exported combined report to {file_path}")
            _auto_open_file(file_path)
            return True
        except Exception as e:
            error(f"Failed to export combined to Word: {e}")
            return False

    def export_combined_to_pdf(
        self,
        vocab_data: list[dict],
        qa_results: list,
        file_path: str,
        include_vocab_details: bool = False,
        include_qa_verification: bool = True
    ) -> bool:
        """
        Export vocabulary and Q&A together to a single PDF document.

        Session 73: Combined export feature.

        Args:
            vocab_data: List of vocabulary dicts
            qa_results: List of QAResult objects
            file_path: Output file path (.pdf)
            include_vocab_details: Include algorithm columns
            include_qa_verification: Include verification coloring

        Returns:
            True if successful, False otherwise
        """
        try:
            debug_log(f"[EXPORT] Combined export to PDF: {len(vocab_data)} terms, {len(qa_results)} Q&A pairs")
            builder = PdfDocumentBuilder()
            export_combined(vocab_data, qa_results, builder, include_vocab_details, include_qa_verification)
            builder.save(file_path)
            info(f"Exported combined report to {file_path}")
            _auto_open_file(file_path)
            return True
        except Exception as e:
            error(f"Failed to export combined to PDF: {e}")
            return False


# Singleton instance
_export_service = None


def get_export_service() -> ExportService:
    """Get or create the export service singleton."""
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
    return _export_service
