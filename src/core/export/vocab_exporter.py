"""
Vocabulary Exporter

Exports vocabulary data to Word/PDF/TXT using DocumentBuilder interface.
"""

import logging
from datetime import datetime
from pathlib import Path

from src.core.export.base import DocumentBuilder

logger = logging.getLogger(__name__)


def export_vocabulary(
    vocab_data: list[dict],
    builder: DocumentBuilder,
    include_details: bool = False,
    title: str = "Names & Vocabulary",
    is_single_doc: bool = True,
) -> None:
    """
    Export vocabulary data using provided document builder.

    Args:
        vocab_data: List of vocabulary dicts with Term, Quality Score, Is Person, etc.
        builder: Word or PDF builder instance
        include_details: Include algorithm detail columns (NER, RAKE, BM25)
        title: Document title
        is_single_doc: If True, omit "# Docs" column (always 1, not useful)
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

    # Build column list dynamically
    if include_details:
        headers = ["Term", "Score", "Person", "Found By", "Occurrences"]
        if not is_single_doc:
            headers.append("# Docs")
        headers.extend(
            [
                "NER",
                "RAKE",
                "BM25",
                "TopicRank",
                "MedicalNER",
                "GLiNER",
                "YAKE",
                "KeyBERT",
            ]
        )
    else:
        headers = ["Term", "Score", "Person", "Found By", "Occurrences"]
        if not is_single_doc:
            headers.append("# Docs")

    # Build rows matching headers
    key_map = {
        "Term": "Term",
        "Score": "Quality Score",
        "Person": "Is Person",
        "Found By": "Found By",
        "Occurrences": "Occurrences",
        "# Docs": "# Docs",
        "NER": "NER",
        "RAKE": "RAKE",
        "BM25": "BM25",
        "TopicRank": "TopicRank",
        "MedicalNER": "MedicalNER",
        "GLiNER": "GLiNER",
        "YAKE": "YAKE",
        "KeyBERT": "KeyBERT",
    }
    rows = []
    for v in vocab_data:
        rows.append([str(v.get(key_map[h], "")) for h in headers])

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
        logger.error("Failed to export vocabulary TXT to '%s': %s", file_path, e)
        return False
