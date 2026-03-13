"""
Semantic Search Exporter

Exports semantic search results to Word/PDF using DocumentBuilder interface.
"""

import logging

from src.core.export.base import DocumentBuilder

logger = logging.getLogger(__name__)


def export_semantic_results(
    results: list,  # list[SemanticResult] - avoid circular import
    builder: DocumentBuilder,
    include_verification_colors: bool = True,
    title: str = "Semantic Search Results",
) -> None:
    """
    Export semantic search results using provided document builder.

    Args:
        results: List of SemanticResult objects
        builder: Word or PDF builder instance
        include_verification_colors: Unused (kept for API compatibility)
        title: Document title
    """
    builder.add_heading(title, level=1)

    from src.core.export.base import format_export_timestamp

    timestamp = format_export_timestamp()
    builder.add_paragraph(f"{len(results)} questions answered")
    builder.add_paragraph(f"Generated: {timestamp}", italic=True)
    builder.add_separator()

    if not results:
        builder.add_paragraph("No search results to export.")
        return

    for i, result in enumerate(results, 1):
        builder.add_heading(f"Q{i}: {result.question}", level=2)

        if hasattr(result, "quick_answer") and result.quick_answer:
            builder.add_paragraph("Answer:", bold=True)
            builder.add_paragraph(result.quick_answer)

        if result.citation:
            builder.add_paragraph("Citation:", bold=True)
            builder.add_paragraph(result.citation, italic=True)

        if hasattr(result, "source_summary") and result.source_summary:
            builder.add_paragraph(f"Source: {result.source_summary}", italic=True)

        builder.add_separator()

    builder.add_paragraph("Exported from CasePrepd", italic=True)
