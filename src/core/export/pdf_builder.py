"""
PDF Document Builder

Implements DocumentBuilder using fpdf2 for PDF export.
"""

from fpdf import FPDF

from src.core.export.base import DocumentBuilder, TextSpan


class PdfDocumentBuilder(DocumentBuilder):
    """
    PDF document builder using fpdf2.

    Creates PDF files with proper formatting for vocabulary
    and Q&A exports including colored verification spans.
    """

    def __init__(self, title: str = "LocalScribe Export"):
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
        self.pdf.multi_cell(0, size * 0.6, text)
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
        self.pdf.multi_cell(0, self._line_height, text)
        self.pdf.ln(2)
        self.pdf.set_font("Helvetica", size=11)

    def add_styled_paragraph(self, spans: list[TextSpan]) -> None:
        """Add a paragraph with mixed styling (verification colors)."""
        for span in spans:
            # Set color
            if span.color:
                self.pdf.set_text_color(*span.color)
            else:
                self.pdf.set_text_color(0, 0, 0)

            # Set style
            style = ""
            if span.bold:
                style += "B"
            if span.italic:
                style += "I"
            if span.strikethrough:
                style += "S"  # fpdf2 supports strikethrough

            self.pdf.set_font("Helvetica", style, 11)
            self.pdf.write(self._line_height, span.text)

        # Reset to defaults
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_font("Helvetica", size=11)
        self.pdf.ln(self._line_height + 2)

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
            self.pdf.cell(col_width, 8, header[:20], border=1, fill=True)
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
                    self.pdf.cell(col_width, 8, header[:20], border=1, fill=True)
                self.pdf.ln()
                self.pdf.set_font("Helvetica", size=10)

            for i, cell_text in enumerate(row_data):
                if i < col_count:
                    # Truncate long text
                    display_text = (
                        str(cell_text)[:25] if len(str(cell_text)) > 25 else str(cell_text)
                    )
                    self.pdf.cell(col_width, 7, display_text, border=1)
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
        self.pdf.output(path)
