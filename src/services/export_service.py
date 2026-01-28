"""
Export Service

Provides high-level API for exporting vocabulary and Q&A to Word/PDF/TXT/HTML.

Session 73: Added auto-open exported files feature (configurable).
Session 75: Refactored with _run_export() helper to reduce duplication (REF-002).
"""

import logging
import os
import sys
from collections.abc import Callable

from src.core.export import (
    PdfDocumentBuilder,
    WordDocumentBuilder,
    export_combined,
    export_qa_html,
    export_qa_results,
    export_vocabulary,
    export_vocabulary_html,
    export_vocabulary_txt,
)

logger = logging.getLogger(__name__)


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
        logger.debug("Auto-opened: %s", file_path)
    except Exception as e:
        logger.debug("Auto-open failed: %s", e)


def _run_export(
    description: str, file_path: str, error_prefix: str, export_fn: Callable[[], bool | None]
) -> bool:
    """
    Helper to run export with standard logging and error handling.

    REF-002: Centralizes the try/except/log pattern used by all export methods.

    Args:
        description: What's being exported (e.g., "10 terms to Word")
        file_path: Output file path for success message
        error_prefix: Prefix for error message (e.g., "vocabulary to Word")
        export_fn: Callable that performs the export, returns True/None on success

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.debug("Exporting %s: %s", description, file_path)
        result = export_fn()
        # Handle functions that return bool vs those that return None
        success = result if result is not None else True
        if success:
            logger.info("Exported to %s", file_path)
            _auto_open_file(file_path)
        return success
    except Exception as e:
        logger.error("Failed to export %s: %s", error_prefix, e)
        return False


class ExportService:
    """
    Service for exporting data to Word and PDF formats.

    Provides a simple API for UI components to export vocabulary
    and Q&A results without knowing the implementation details.
    """

    def export_vocabulary_to_word(
        self, vocab_data: list[dict], file_path: str, include_details: bool = False
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

        def do_export():
            builder = WordDocumentBuilder()
            export_vocabulary(vocab_data, builder, include_details)
            builder.save(file_path)

        return _run_export(
            f"{len(vocab_data)} terms to Word", file_path, "vocabulary to Word", do_export
        )

    def export_vocabulary_to_pdf(
        self, vocab_data: list[dict], file_path: str, include_details: bool = False
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

        def do_export():
            builder = PdfDocumentBuilder()
            export_vocabulary(vocab_data, builder, include_details)
            builder.save(file_path)

        return _run_export(
            f"{len(vocab_data)} terms to PDF", file_path, "vocabulary to PDF", do_export
        )

    def export_qa_to_word(
        self, results: list, file_path: str, include_verification: bool = True
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

        def do_export():
            builder = WordDocumentBuilder()
            export_qa_results(results, builder, include_verification)
            builder.save(file_path)

        return _run_export(f"{len(results)} Q&A pairs to Word", file_path, "Q&A to Word", do_export)

    def export_qa_to_pdf(
        self, results: list, file_path: str, include_verification: bool = True
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

        def do_export():
            builder = PdfDocumentBuilder()
            export_qa_results(results, builder, include_verification)
            builder.save(file_path)

        return _run_export(f"{len(results)} Q&A pairs to PDF", file_path, "Q&A to PDF", do_export)

    def export_vocabulary_to_txt(self, vocab_data: list[dict], file_path: str) -> bool:
        """
        Export vocabulary to plain text (one term per line).

        Args:
            vocab_data: List of vocabulary dicts
            file_path: Output file path (.txt)

        Returns:
            True if successful, False otherwise
        """
        return _run_export(
            f"{len(vocab_data)} terms to TXT",
            file_path,
            "vocabulary to TXT",
            lambda: export_vocabulary_txt(vocab_data, file_path),
        )

    def export_vocabulary_to_html(
        self, vocab_data: list[dict], file_path: str, visible_columns: list[str] | None = None
    ) -> bool:
        """
        Export vocabulary to interactive HTML.

        Session 80: Updated to pass visible columns from GUI to HTML export.
        All columns are included but only visible_columns are shown initially.

        Args:
            vocab_data: List of vocabulary dicts
            file_path: Output file path (.html)
            visible_columns: Columns to show initially (from GUI selection)

        Returns:
            True if successful, False otherwise
        """
        return _run_export(
            f"{len(vocab_data)} terms to HTML",
            file_path,
            "vocabulary to HTML",
            lambda: export_vocabulary_html(vocab_data, file_path, visible_columns),
        )

    def get_vocabulary_html_content(
        self, vocab_data: list[dict], visible_columns: list[str] | None = None
    ) -> str:
        """
        Get vocabulary as HTML content string (without saving to file).

        Session 83: Added for UI components that write the file themselves.

        Args:
            vocab_data: List of vocabulary dicts
            visible_columns: Columns to show initially (from GUI selection)

        Returns:
            HTML content as string
        """
        from src.core.export.html_builder import build_vocabulary_html

        return build_vocabulary_html(vocab_data, visible_columns)

    def export_qa_to_html(
        self, results: list, file_path: str, include_verification: bool = True
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
        return _run_export(
            f"{len(results)} Q&A pairs to HTML",
            file_path,
            "Q&A to HTML",
            lambda: export_qa_html(results, file_path, include_verification),
        )

    def export_combined_html(
        self,
        vocab_data: list[dict],
        qa_results: list,
        summary_text: str,
        file_path: str,
        visible_columns: list[str] | None = None,
        include_verification: bool = True,
    ) -> bool:
        """
        Export vocabulary, Q&A, and summary to a single tabbed HTML file.

        Args:
            vocab_data: List of vocabulary dicts (score-filtered)
            qa_results: List of QAResult objects (answered only)
            summary_text: Summary text string
            file_path: Output file path (.html)
            visible_columns: Columns to show initially in vocab table
            include_verification: Include verification badges in Q&A

        Returns:
            True if successful, False otherwise
        """
        from src.core.export.combined_html_builder import build_combined_html

        def do_export():
            html_content = build_combined_html(
                vocab_data,
                qa_results,
                summary_text,
                visible_columns,
                include_verification,
            )
            from pathlib import Path

            Path(file_path).write_text(html_content, encoding="utf-8")

        term_count = len(vocab_data)
        qa_count = len(qa_results)
        return _run_export(
            f"combined HTML ({term_count} terms, {qa_count} Q&A)",
            file_path,
            "combined HTML",
            do_export,
        )

    def export_combined_to_word(
        self,
        vocab_data: list[dict],
        qa_results: list,
        file_path: str,
        include_vocab_details: bool = False,
        include_qa_verification: bool = True,
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

        def do_export():
            builder = WordDocumentBuilder()
            export_combined(
                vocab_data, qa_results, builder, include_vocab_details, include_qa_verification
            )
            builder.save(file_path)

        return _run_export(
            f"combined ({len(vocab_data)} terms, {len(qa_results)} Q&A) to Word",
            file_path,
            "combined to Word",
            do_export,
        )

    def export_combined_to_pdf(
        self,
        vocab_data: list[dict],
        qa_results: list,
        file_path: str,
        include_vocab_details: bool = False,
        include_qa_verification: bool = True,
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

        def do_export():
            builder = PdfDocumentBuilder()
            export_combined(
                vocab_data, qa_results, builder, include_vocab_details, include_qa_verification
            )
            builder.save(file_path)

        return _run_export(
            f"combined ({len(vocab_data)} terms, {len(qa_results)} Q&A) to PDF",
            file_path,
            "combined to PDF",
            do_export,
        )


# Singleton instance
_export_service = None


def get_export_service() -> ExportService:
    """Get or create the export service singleton."""
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
    return _export_service
