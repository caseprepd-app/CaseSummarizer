"""
Document Export Base Classes

Provides abstract base for document generation with Word and PDF implementations.
Both formats share the same interface for consistent output.
"""

from abc import ABC, abstractmethod


class DocumentBuilder(ABC):
    """
    Abstract base for document generation.

    Implementations exist for Word (.docx) and PDF formats.
    Both produce structurally similar output with format-specific styling.
    """

    @abstractmethod
    def add_heading(self, text: str, level: int = 1) -> None:
        """
        Add a heading to the document.

        Args:
            text: Heading text
            level: Heading level (1=title, 2=section, 3=subsection)
        """
        pass

    @abstractmethod
    def add_paragraph(self, text: str, bold: bool = False, italic: bool = False) -> None:
        """
        Add a simple text paragraph.

        Args:
            text: Paragraph text
            bold: Apply bold styling
            italic: Apply italic styling
        """
        pass

    @abstractmethod
    def add_table(self, headers: list[str], rows: list[list[str]]) -> None:
        """
        Add a table to the document.

        Args:
            headers: Column header texts
            rows: List of row data (each row is list of cell texts)
        """
        pass

    @abstractmethod
    def add_separator(self) -> None:
        """Add a visual separator (horizontal rule or spacing)."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the document to file.

        Args:
            path: Output file path
        """
        pass


def format_export_timestamp() -> str:
    """
    Get current timestamp formatted for export documents.

    Returns:
        Timestamp string like "2026-03-04 14:30"
    """
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M")
