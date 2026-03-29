"""
PDF Document Builder

Implements DocumentBuilder using fpdf2 for PDF export.
"""

import logging
import unicodedata

from fpdf import FPDF

logger = logging.getLogger(__name__)

from src.core.export.base import DocumentBuilder


def _sanitize_for_latin1(text: str) -> str:
    """
    Sanitize text for Helvetica (Latin-1) rendering.

    Normalizes Unicode to closest ASCII equivalent where possible
    (e.g., accented characters), replaces remaining non-Latin-1
    characters with '?' to avoid fpdf2 encoding errors.

    Args:
        text: Input text possibly containing non-Latin-1 characters

    Returns:
        Text safe for Helvetica/Latin-1 rendering
    """
    # Normalize to NFKD (decomposes accented chars into base + combining)
    normalized = unicodedata.normalize("NFKD", text)
    # Strip combining marks (category "Mn") left by NFKD decomposition
    stripped = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    # Encode to latin-1, replacing unencodable chars with '?'
    return stripped.encode("latin-1", errors="replace").decode("latin-1")


class PdfDocumentBuilder(DocumentBuilder):
    """
    PDF document builder using fpdf2.

    Creates PDF files with proper formatting for vocabulary
    and Q&A exports.
    """

    def __init__(self, title: str = "CasePrepd Export"):
        """
        Initialize a new PDF document.

        Args:
            title: Document title (used in metadata)
        """
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()
        self.pdf.set_title(title)

        # Set default font
        self.pdf.set_font("Helvetica", size=11)

        # Track current y position for layout
        self._line_height = 6

    def add_heading(self, text: str, level: int = 1) -> None:
        """Add a heading to the document."""
        sizes = {1: 18, 2: 14, 3: 12}
        size = sizes.get(level, 12)

        self.pdf.set_font("Helvetica", "B", size)
        self.pdf.ln(4 if level > 1 else 0)
        self.pdf.multi_cell(0, size * 0.6, _sanitize_for_latin1(text))
        self.pdf.ln(2)
        self.pdf.set_font("Helvetica", size=11)

    def add_paragraph(self, text: str, bold: bool = False, italic: bool = False) -> None:
        """Add a simple text paragraph."""
        style = ""
        if bold:
            style += "B"
        if italic:
            style += "I"

        self.pdf.set_font("Helvetica", style, 11)
        self.pdf.multi_cell(0, self._line_height, _sanitize_for_latin1(text))
        self.pdf.ln(2)
        self.pdf.set_font("Helvetica", size=11)

    def add_table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Add a table to the document."""
        if not headers:
            return

        # Calculate column widths based on content
        page_width = self.pdf.w - 2 * self.pdf.l_margin
        col_count = len(headers)
        col_width = page_width / col_count

        # Header row
        self.pdf.set_font("Helvetica", "B", 10)
        self.pdf.set_fill_color(240, 240, 240)

        for header in headers:
            header_text = header[:17] + "..." if len(header) > 20 else header
            if len(header) > 20:
                logger.warning("PDF table header truncated: '%s' -> '%s'", header, header_text)
            self.pdf.cell(col_width, 8, _sanitize_for_latin1(header_text), border=1, fill=True)
        self.pdf.ln()

        # Data rows
        self.pdf.set_font("Helvetica", size=10)
        for row_data in rows:
            # Check if we need a new page
            if self.pdf.get_y() > self.pdf.h - 30:
                self.pdf.add_page()
                # Repeat headers on new page
                self.pdf.set_font("Helvetica", "B", 10)
                self.pdf.set_fill_color(240, 240, 240)
                for header in headers:
                    header_text = header[:17] + "..." if len(header) > 20 else header
                    self.pdf.cell(
                        col_width, 8, _sanitize_for_latin1(header_text), border=1, fill=True
                    )
                self.pdf.ln()
                self.pdf.set_font("Helvetica", size=10)

            for i, cell_text in enumerate(row_data):
                if i < col_count:
                    # Truncate long text with ellipsis indicator
                    cell_str = str(cell_text)
                    if len(cell_str) > 25:
                        logger.debug(
                            "PDF table cell truncated from %d to 25 chars: '%.40s...'",
                            len(cell_str),
                            cell_str,
                        )
                        display_text = cell_str[:22] + "..."
                    else:
                        display_text = cell_str
                    self.pdf.cell(col_width, 7, _sanitize_for_latin1(display_text), border=1)
            self.pdf.ln()

        self.pdf.ln(4)
        self.pdf.set_font("Helvetica", size=11)

    def add_separator(self) -> None:
        """Add a visual separator."""
        self.pdf.ln(4)
        self.pdf.set_draw_color(200, 200, 200)
        y = self.pdf.get_y()
        self.pdf.line(self.pdf.l_margin, y, self.pdf.w - self.pdf.r_margin, y)
        self.pdf.ln(6)

    def save(self, path: str) -> None:
        """Save the document to file."""
        try:
            self.pdf.output(path)
        except OSError as e:
            raise OSError(
                f"Could not save PDF to '{path}'. "
                f"The file may be open in another application. "
                f"Please close it and try again."
            ) from e
