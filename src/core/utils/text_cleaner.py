"""
Text Cleaner for PDF Extraction Output.

Fixes common PDF text artifacts before vocabulary extraction:
- Ligatures (fi, fl, ff, ffi, ffl) fused into single Unicode characters
- Mojibake (encoding errors producing garbage like â€™ instead of ')
- BOM markers (invisible bytes that break string matching)
- Curly quotes normalized to straight quotes

Uses the ftfy library for robust Unicode repair. This runs once on the
full extracted text before any vocabulary algorithms see it.

Usage:
    from src.core.utils.text_cleaner import clean_extracted_text

    cleaned = clean_extracted_text(raw_pdf_text)
"""

import logging

logger = logging.getLogger(__name__)


def clean_extracted_text(text: str) -> str:
    """
    Clean PDF extraction artifacts from text.

    Fixes ligatures, mojibake, BOM markers, and encoding errors that
    cause vocabulary algorithms to misidentify real words as unknown.

    Args:
        text: Raw text from PDF extraction

    Returns:
        Cleaned text with Unicode artifacts resolved
    """
    if not text:
        return text

    try:
        import ftfy

        cleaned = ftfy.fix_text(text)
    except ImportError:
        logger.warning("ftfy not installed; skipping text cleanup")
        return text

    if cleaned != text:
        # Count changes for logging
        diff_chars = sum(1 for a, b in zip(text, cleaned) if a != b)
        diff_chars += abs(len(text) - len(cleaned))
        logger.debug("ftfy cleaned %d character(s) in extracted text", diff_chars)

    return cleaned
