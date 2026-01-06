"""
Q&A Exporter

Exports Q&A results to Word/PDF using DocumentBuilder interface.
Supports verification coloring for hallucination detection.
"""

from datetime import datetime

from src.core.export.base import DocumentBuilder, TextSpan, get_verification_color


def export_qa_results(
    results: list,  # list[QAResult] - avoid circular import
    builder: DocumentBuilder,
    include_verification_colors: bool = True,
    title: str = "Questions & Answers",
) -> None:
    """
    Export Q&A results using provided document builder.

    Args:
        results: List of QAResult objects (should be filtered by include_in_export)
        builder: Word or PDF builder instance
        include_verification_colors: Apply verification coloring to answers
        title: Document title
    """
    # Add title
    builder.add_heading(title, level=1)

    # Add summary
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    builder.add_paragraph(f"{len(results)} questions answered")
    builder.add_paragraph(f"Generated: {timestamp}", italic=True)
    builder.add_separator()

    if not results:
        builder.add_paragraph("No Q&A results to export.")
        return

    # Export each Q&A pair
    for i, result in enumerate(results, 1):
        # Question
        builder.add_heading(f"Q{i}: {result.question}", level=2)

        # Answer with optional verification coloring
        if include_verification_colors and hasattr(result, "verification") and result.verification:
            _add_verified_answer(builder, result)
        else:
            builder.add_paragraph(result.quick_answer)

        # Citation
        if result.citation:
            builder.add_paragraph("Citation:", bold=True)
            builder.add_paragraph(result.citation, italic=True)

        # Source
        if hasattr(result, "source_summary") and result.source_summary:
            builder.add_paragraph(f"Source: {result.source_summary}", italic=True)

        builder.add_separator()

    # Add verification legend if colors were used
    if include_verification_colors:
        _add_verification_legend(builder)

    # Add footer
    builder.add_paragraph("Exported from LocalScribe", italic=True)


def _add_verified_answer(builder: DocumentBuilder, result) -> None:
    """
    Add answer with verification coloring.

    Args:
        builder: Document builder
        result: QAResult with verification data
    """
    verification = result.verification

    # Add reliability badge
    reliability = verification.overall_reliability * 100
    if reliability >= 75:
        level = "HIGH"
    elif reliability >= 50:
        level = "MEDIUM"
    else:
        level = "LOW"

    builder.add_paragraph(f"Reliability: {reliability:.0f}% ({level})", bold=True)

    # Check if answer was rejected
    if verification.answer_rejected:
        builder.add_paragraph(
            "Answer rejected due to low reliability. The AI could not provide "
            "a trustworthy response based on the available documents.",
            italic=True,
        )
        return

    # Build spans with verification colors
    spans = []
    for span in verification.spans:
        color, strikethrough, _ = get_verification_color(span.hallucination_prob)
        spans.append(TextSpan(text=span.text, color=color, strikethrough=strikethrough))

    if spans:
        builder.add_styled_paragraph(spans)
    else:
        # Fallback if no spans
        builder.add_paragraph(result.quick_answer)


def _add_verification_legend(builder: DocumentBuilder) -> None:
    """Add color legend for verification."""
    builder.add_heading("Verification Legend", level=3)

    legend_spans = [
        TextSpan("Verified", color=(40, 167, 69)),
        TextSpan(" | "),
        TextSpan("Uncertain", color=(255, 193, 7)),
        TextSpan(" | "),
        TextSpan("Suspicious", color=(253, 126, 20)),
        TextSpan(" | "),
        TextSpan("Unreliable", color=(220, 53, 69)),
        TextSpan(" | "),
        TextSpan("Hallucinated", color=(136, 136, 136), strikethrough=True),
    ]
    builder.add_styled_paragraph(legend_spans)
