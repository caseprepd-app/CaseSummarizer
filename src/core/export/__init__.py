"""
Document Export Module

Provides Word (.docx) and PDF export for vocabulary and Q&A results.

Usage:
    from src.core.export import WordDocumentBuilder, PdfDocumentBuilder
    from src.core.export import export_vocabulary, export_qa_results

    # Export vocabulary to Word
    builder = WordDocumentBuilder()
    export_vocabulary(vocab_data, builder)
    builder.save("vocabulary.docx")

    # Export Q&A to PDF
    builder = PdfDocumentBuilder()
    export_qa_results(qa_results, builder)
    builder.save("qa_results.pdf")

    # Export combined (Session 73)
    builder = WordDocumentBuilder()
    export_combined(vocab_data, qa_results, builder)
    builder.save("combined_report.docx")
"""

from src.core.export.base import (
    VERIFICATION_COLORS,
    DocumentBuilder,
    TextSpan,
    get_verification_color,
)
from src.core.export.combined_exporter import export_combined
from src.core.export.html_builder import export_qa_html, export_vocabulary_html
from src.core.export.pdf_builder import PdfDocumentBuilder
from src.core.export.qa_exporter import export_qa_results
from src.core.export.vocab_exporter import export_vocabulary, export_vocabulary_txt
from src.core.export.word_builder import WordDocumentBuilder

__all__ = [
    "VERIFICATION_COLORS",
    # Base classes
    "DocumentBuilder",
    "PdfDocumentBuilder",
    "TextSpan",
    # Builders
    "WordDocumentBuilder",
    "export_combined",
    "export_qa_html",
    "export_qa_results",
    # Exporters
    "export_vocabulary",
    "export_vocabulary_html",
    "export_vocabulary_txt",
    "get_verification_color",
]
