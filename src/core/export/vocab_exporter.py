"""
Vocabulary Exporter

Exports vocabulary data to Word/PDF/TXT using DocumentBuilder interface.
"""

from datetime import datetime
from pathlib import Path

from src.core.export.base import DocumentBuilder


def export_vocabulary(
    vocab_data: list[dict],
    builder: DocumentBuilder,
    include_details: bool = False,
    title: str = "Names & Vocabulary",
) -> None:
    """
    Export vocabulary data using provided document builder.

    Args:
        vocab_data: List of vocabulary dicts with Term, Quality Score, Is Person, etc.
        builder: Word or PDF builder instance
        include_details: Include algorithm detail columns (NER, RAKE, BM25)
        title: Document title
    """
    # Add title
    builder.add_heading(title, level=1)

    # Add summary
    person_count = sum(1 for v in vocab_data if v.get("Is Person") == "Yes")
    term_count = len(vocab_data) - person_count
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    builder.add_paragraph(
        f"Extracted {len(vocab_data)} total entries: {person_count} persons, {term_count} terms"
    )
    builder.add_paragraph(f"Generated: {timestamp}", italic=True)
    builder.add_separator()

    if not vocab_data:
        builder.add_paragraph("No vocabulary data to export.")
        return

    # Build table
    if include_details:
        headers = ["Term", "Score", "Person", "Found By", "NER", "RAKE", "BM25", "Freq"]
        rows = []
        for v in vocab_data:
            rows.append(
                [
                    v.get("Term", ""),
                    str(v.get("Quality Score", "")),
                    v.get("Is Person", ""),
                    v.get("Found By", ""),
                    v.get("NER", ""),
                    v.get("RAKE", ""),
                    v.get("BM25", ""),
                    str(v.get("In-Case Freq", "")),
                ]
            )
    else:
        headers = ["Term", "Score", "Person", "Found By"]
        rows = []
        for v in vocab_data:
            rows.append(
                [
                    v.get("Term", ""),
                    str(v.get("Quality Score", "")),
                    v.get("Is Person", ""),
                    v.get("Found By", ""),
                ]
            )

    builder.add_table(headers, rows)

    # Add footer
    builder.add_separator()
    builder.add_paragraph("Exported from CasePrepd", italic=True)


def export_vocabulary_txt(vocab_data: list[dict], file_path: str) -> bool:
    """
    Export vocabulary as plain text (one term per line).

    Args:
        vocab_data: List of vocabulary dicts with Term key
        file_path: Output file path (.txt)

    Returns:
        True if successful, False otherwise
    """
    try:
        terms = [v.get("Term", "") for v in vocab_data if v.get("Term")]
        content = "\n".join(terms)

        Path(file_path).write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        from src.logging_config import error

        error(f"[Vocab Export] Failed to export vocabulary TXT to '{file_path}': {e}")
        return False
