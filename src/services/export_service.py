"""
Export Service

Provides high-level API for exporting vocabulary and Q&A to Word/PDF/TXT/HTML.

Supports auto-open of exported files (configurable).
Uses _run_export() helper to reduce duplication.
"""

import logging
import os
import sys
import threading
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

    Respects user preference 'auto_open_exports'.

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
        logger.warning("Auto-open failed: %s", e)


def _run_export(
    description: str, file_path: str, error_prefix: str, export_fn: Callable[[], bool | None]
) -> tuple[bool, str | None]:
    """
    Helper to run export with standard logging and error handling.

    Centralizes the try/except/log pattern used by all export methods.

    Args:
        description: What's being exported (e.g., "10 terms to Word")
        file_path: Output file path for success message
        error_prefix: Prefix for error message (e.g., "vocabulary to Word")
        export_fn: Callable that performs the export, returns True/None on success

    Returns:
        (True, None) if successful, (False, error_detail) otherwise
    """
    try:
        logger.debug("Exporting %s: %s", description, file_path)
        result = export_fn()
        # Handle functions that return bool vs those that return None
        success = result if result is not None else True
        if success:
            logger.info("Exported to %s", file_path)
            _auto_open_file(file_path)
        return (success, None)
    except Exception as e:
        logger.error("Failed to export %s: %s", error_prefix, e, exc_info=True)
        return (False, str(e))


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
        include_details: bool = False,
        is_single_doc: bool = True,
    ) -> tuple[bool, str | None]:
        """
        Export vocabulary to Word document.

        Args:
            vocab_data: List of vocabulary dicts
            file_path: Output file path (.docx)
            include_details: Include algorithm columns
            is_single_doc: If True, omit "# Docs" column

        Returns:
            (True, None) if successful, (False, error_detail) otherwise
        """

        def do_export():
            builder = WordDocumentBuilder()
            export_vocabulary(vocab_data, builder, include_details, is_single_doc=is_single_doc)
            builder.save(file_path)

        return _run_export(
            f"{len(vocab_data)} terms to Word", file_path, "vocabulary to Word", do_export
        )

    def export_vocabulary_to_pdf(
        self,
        vocab_data: list[dict],
        file_path: str,
        include_details: bool = False,
        is_single_doc: bool = True,
    ) -> tuple[bool, str | None]:
        """
        Export vocabulary to PDF document.

        Args:
            vocab_data: List of vocabulary dicts
            file_path: Output file path (.pdf)
            include_details: Include algorithm columns
            is_single_doc: If True, omit "# Docs" column

        Returns:
            (True, None) if successful, (False, error_detail) otherwise
        """

        def do_export():
            builder = PdfDocumentBuilder()
            export_vocabulary(vocab_data, builder, include_details, is_single_doc=is_single_doc)
            builder.save(file_path)

        return _run_export(
            f"{len(vocab_data)} terms to PDF", file_path, "vocabulary to PDF", do_export
        )

    def export_qa_to_word(
        self, results: list, file_path: str, include_verification: bool = True
    ) -> tuple[bool, str | None]:
        """
        Export Q&A results to Word document.

        Args:
            results: List of QAResult objects
            file_path: Output file path (.docx)
            include_verification: Include verification coloring

        Returns:
            (True, None) if successful, (False, error_detail) otherwise
        """

        def do_export():
            builder = WordDocumentBuilder()
            export_qa_results(results, builder, include_verification)
            builder.save(file_path)

        return _run_export(f"{len(results)} Q&A pairs to Word", file_path, "Q&A to Word", do_export)

    def export_qa_to_pdf(
        self, results: list, file_path: str, include_verification: bool = True
    ) -> tuple[bool, str | None]:
        """
        Export Q&A results to PDF document.

        Args:
            results: List of QAResult objects
            file_path: Output file path (.pdf)
            include_verification: Include verification coloring

        Returns:
            (True, None) if successful, (False, error_detail) otherwise
        """

        def do_export():
            builder = PdfDocumentBuilder()
            export_qa_results(results, builder, include_verification)
            builder.save(file_path)

        return _run_export(f"{len(results)} Q&A pairs to PDF", file_path, "Q&A to PDF", do_export)

    def export_vocabulary_to_txt(
        self, vocab_data: list[dict], file_path: str
    ) -> tuple[bool, str | None]:
        """
        Export vocabulary to plain text (one term per line).

        Args:
            vocab_data: List of vocabulary dicts
            file_path: Output file path (.txt)

        Returns:
            (True, None) if successful, (False, error_detail) otherwise
        """
        return _run_export(
            f"{len(vocab_data)} terms to TXT",
            file_path,
            "vocabulary to TXT",
            lambda: export_vocabulary_txt(vocab_data, file_path),
        )

    def export_vocabulary_to_html(
        self, vocab_data: list[dict], file_path: str, visible_columns: list[str] | None = None
    ) -> tuple[bool, str | None]:
        """
        Export vocabulary to interactive HTML.

        All columns are included but only visible_columns are shown initially.

        Args:
            vocab_data: List of vocabulary dicts
            file_path: Output file path (.html)
            visible_columns: Columns to show initially (from GUI selection)

        Returns:
            (True, None) if successful, (False, error_detail) otherwise
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

        Returns HTML content for UI components that write the file themselves.

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
    ) -> tuple[bool, str | None]:
        """
        Export Q&A results to interactive HTML.

        Args:
            results: List of QAResult objects
            file_path: Output file path (.html)
            include_verification: Include verification coloring

        Returns:
            (True, None) if successful, (False, error_detail) otherwise
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
    ) -> tuple[bool, str | None]:
        """
        Export vocabulary and search results to a single tabbed HTML file.

        Args:
            vocab_data: List of vocabulary dicts (score-filtered)
            qa_results: List of QAResult objects (answered only)
            summary_text: Summary text string
            file_path: Output file path (.html)
            visible_columns: Columns to show initially in vocab table
            include_verification: Include verification badges in search results

        Returns:
            (True, None) if successful, (False, error_detail) otherwise
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
        summary_text: str = "",
    ) -> tuple[bool, str | None]:
        """
        Export vocabulary and search results to a single Word document.

        Args:
            vocab_data: List of vocabulary dicts
            qa_results: List of QAResult objects
            file_path: Output file path (.docx)
            include_vocab_details: Include algorithm columns
            include_qa_verification: Include verification coloring
            summary_text: Summary text to include (empty string to skip)

        Returns:
            (True, None) if successful, (False, error_detail) otherwise
        """

        def do_export():
            builder = WordDocumentBuilder()
            export_combined(
                vocab_data,
                qa_results,
                builder,
                include_vocab_details,
                include_qa_verification,
                summary_text=summary_text,
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
        summary_text: str = "",
    ) -> tuple[bool, str | None]:
        """
        Export vocabulary and search results to a single PDF document.

        Args:
            vocab_data: List of vocabulary dicts
            qa_results: List of QAResult objects
            file_path: Output file path (.pdf)
            include_vocab_details: Include algorithm columns
            include_qa_verification: Include verification coloring
            summary_text: Summary text to include (empty string to skip)

        Returns:
            (True, None) if successful, (False, error_detail) otherwise
        """

        def do_export():
            builder = PdfDocumentBuilder()
            export_combined(
                vocab_data,
                qa_results,
                builder,
                include_vocab_details,
                include_qa_verification,
                summary_text=summary_text,
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
_export_service_lock = threading.Lock()


def get_export_service() -> ExportService:
    """Get or create the export service singleton, thread-safe."""
    global _export_service
    if _export_service is not None:
        return _export_service
    with _export_service_lock:
        if _export_service is None:
            _export_service = ExportService()
    return _export_service


def reset_export_service():
    """Reset the singleton for test isolation."""
    global _export_service
    with _export_service_lock:
        _export_service = None
