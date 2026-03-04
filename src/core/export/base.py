"""
Document Export Base Classes

Provides abstract base for document generation with Word and PDF implementations.
Both formats share the same interface for consistent output.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TextSpan:
    """
    Text segment with optional styling.

    Used for verification-colored Q&A answers where different parts
    have different reliability scores.
    """

    text: str
    color: tuple[int, int, int] | None = None  # RGB tuple
    bold: bool = False
    italic: bool = False
    strikethrough: bool = False


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
    def add_styled_paragraph(self, spans: list[TextSpan]) -> None:
        """
        Add a paragraph with mixed styling (e.g., verification colors).

        Args:
            spans: List of TextSpan objects with individual styling
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


# Verification color constants (from theme.py)
# Used for Q&A answer span coloring
VERIFICATION_COLORS = {
    "verified": (40, 167, 69),  # Green
    "uncertain": (255, 193, 7),  # Yellow
    "suspicious": (253, 126, 20),  # Orange
    "unreliable": (220, 53, 69),  # Red
    "hallucinated": (136, 136, 136),  # Gray (+ strikethrough)
}


def get_verification_color(hallucination_prob: float) -> tuple[tuple[int, int, int], bool, str]:
    """
    Get color and styling for a verification probability.

    Args:
        hallucination_prob: Probability that text is hallucinated (0.0 to 1.0)

    Returns:
        Tuple of (RGB color, strikethrough flag, category name)
    """
    if hallucination_prob < 0.30:
        return VERIFICATION_COLORS["verified"], False, "verified"
    elif hallucination_prob < 0.50:
        return VERIFICATION_COLORS["uncertain"], False, "uncertain"
    elif hallucination_prob < 0.70:
        return VERIFICATION_COLORS["suspicious"], False, "suspicious"
    elif hallucination_prob < 0.85:
        return VERIFICATION_COLORS["unreliable"], False, "unreliable"
    else:
        return VERIFICATION_COLORS["hallucinated"], True, "hallucinated"


def format_export_timestamp() -> str:
    """
    Get current timestamp formatted for export documents.

    Returns:
        Timestamp string like "2026-03-04 14:30"
    """
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M")
