"""
Document Export Module

Provides Word (.docx) and PDF export for vocabulary and semantic search results.

Usage:
    from src.core.export import WordDocumentBuilder, PdfDocumentBuilder
    from src.core.export import export_vocabulary, export_semantic_results

    # Export vocabulary to Word
    builder = WordDocumentBuilder()
    export_vocabulary(vocab_data, builder)
    builder.save("vocabulary.docx")

    # Export search results to PDF
    builder = PdfDocumentBuilder()
    export_semantic_results(semantic_results, builder)
    builder.save("semantic_results.pdf")

    # Export combined
    builder = WordDocumentBuilder()
    export_combined(vocab_data, semantic_results, builder)
    builder.save("combined_report.docx")
"""

from src.core.export.base import DocumentBuilder
from src.core.export.combined_exporter import export_combined
from src.core.export.html_builder import export_semantic_html, export_vocabulary_html
from src.core.export.pdf_builder import PdfDocumentBuilder
from src.core.export.semantic_exporter import export_semantic_results
from src.core.export.vocab_exporter import export_vocabulary, export_vocabulary_txt
from src.core.export.word_builder import WordDocumentBuilder

__all__ = [
    "DocumentBuilder",
    "PdfDocumentBuilder",
    "WordDocumentBuilder",
    "export_combined",
    "export_semantic_html",
    "export_semantic_results",
    "export_vocabulary",
    "export_vocabulary_html",
    "export_vocabulary_txt",
]
