"""
Vocabulary Exporter

Exports vocabulary data to Word/PDF/TXT using DocumentBuilder interface.
"""

import logging
from pathlib import Path

from src.core.export.base import DocumentBuilder, format_export_timestamp
from src.core.vocab_schema import VF
from src.core.vocabulary.person_utils import vocab_summary_counts

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
    total, person_count, term_count = vocab_summary_counts(vocab_data)
    timestamp = format_export_timestamp()

    builder.add_paragraph(
        f"Extracted {total} total entries: {person_count} persons, {term_count} terms"
    )
    builder.add_paragraph(f"Generated: {timestamp}", italic=True)
    builder.add_separator()

    if not vocab_data:
        builder.add_paragraph("No vocabulary data to export.")
        return

    # Build column list dynamically
    if include_details:
        headers = [VF.TERM, "Score", "Person", VF.FOUND_BY, VF.OCCURRENCES]
        if not is_single_doc:
            headers.append(VF.NUM_DOCS)
        headers.extend(
            [
                VF.NER,
                VF.RAKE,
                VF.BM25,
                VF.TOPICRANK,
                VF.MEDICALNER,
                VF.YAKE,
            ]
        )
    else:
        headers = [VF.TERM, "Score", "Person", VF.FOUND_BY, VF.OCCURRENCES]
        if not is_single_doc:
            headers.append(VF.NUM_DOCS)

    # Build rows matching headers
    key_map = {
        VF.TERM: VF.TERM,
        "Score": VF.QUALITY_SCORE,
        "Person": VF.IS_PERSON,
        VF.FOUND_BY: VF.FOUND_BY,
        VF.OCCURRENCES: VF.OCCURRENCES,
        VF.NUM_DOCS: VF.NUM_DOCS,
        VF.NER: VF.NER,
        VF.RAKE: VF.RAKE,
        VF.BM25: VF.BM25,
        VF.TOPICRANK: VF.TOPICRANK,
        VF.MEDICALNER: VF.MEDICALNER,
        VF.YAKE: VF.YAKE,
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
        terms = [v.get(VF.TERM, "") for v in vocab_data if v.get(VF.TERM)]
        content = "\n".join(terms)

        Path(file_path).write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        logger.error("Failed to export vocabulary TXT to '%s': %s", file_path, e, exc_info=True)
        return False
