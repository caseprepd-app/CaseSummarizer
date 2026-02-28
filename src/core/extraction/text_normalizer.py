"""
Text Normalization Pipeline for Extracted Documents.

Applies a 4-stage normalization pipeline to raw extracted text:
    1. De-hyphenation - Rejoin words split across lines
    2. Page number removal - Remove page markers (Page 1, -2-, etc.)
    3. Line filtering - Remove short/garbage lines, keep legal headers
    4. Whitespace normalization - Collapse excess blank lines

This is Step 2 of the document processing pipeline (after extraction, before
character sanitization).

Example usage:
    >>> normalizer = TextNormalizer()
    >>> raw_text = "The plain-\\ntiff filed a motion.\\n\\nPage 1\\n\\n"
    >>> normalizer.normalize(raw_text)
    'The plaintiff filed a motion.'
"""

import logging
import re
import time

from src.config import MIN_LINE_LENGTH

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Normalizes raw extracted text through a 4-stage pipeline.

    Each stage is logged with timing and character count changes for
    debugging extraction quality issues.

    Attributes:
        legal_keywords: Set of keywords that identify legal document headers
                       (e.g., COURT, PLAINTIFF). These are preserved even if
                       they're short or all-caps.
    """

    def __init__(self, legal_keywords: set[str] | None = None):
        """
        Initialize the text normalizer.

        Args:
            legal_keywords: Optional set of legal document header keywords.
                          If not provided, uses default set of common terms.
        """
        self.legal_keywords = legal_keywords or {
            "COURT",
            "PLAINTIFF",
            "DEFENDANT",
            "APPEARANCES",
            "SUPREME",
            "MOTION",
            "AFFIDAVIT",
            "EXHIBIT",
            "DEPOSITION",
            "TESTIMONY",
            "COMPLAINT",
            "ANSWER",
            "SUMMONS",
            "NOTICE",
            "ORDER",
            "JUDGE",
            "ATTORNEY",
            "COUNSEL",
            "PARTY",
            "ACTION",
        }

    def normalize(self, text: str) -> str:
        """
        Apply the full 4-stage normalization pipeline.

        Pipeline stages:
            1. De-hyphenation: Rejoin "plain-\\ntiff" -> "plaintiff"
            2. Page numbers: Remove "Page 1", "- 2 -", standalone numbers
            3. Line filtering: Remove short lines, keep legal headers
            4. Whitespace: Collapse multiple blank lines

        Args:
            text: Raw extracted text

        Returns:
            Normalized text ready for character sanitization

        Example:
            >>> normalizer = TextNormalizer()
            >>> normalizer.normalize("The plain-\\ntiff\\nPage 1\\n\\n\\nfiled.")
            'The plaintiff\\nfiled.'
        """
        logger.debug("Applying text normalization rules")

        # Stage 1: De-hyphenation
        text = self._stage_dehyphenation(text)

        # Stage 2: Page number removal
        text = self._stage_page_numbers(text)

        # Stage 3: Line filtering
        text = self._stage_line_filtering(text)

        # Stage 4: Whitespace normalization
        text = self._stage_whitespace(text)

        return text

    def _stage_dehyphenation(self, text: str) -> str:
        """
        Stage 1: Rejoin words split across lines with hyphens.

        Handles patterns like "plain-\\ntiff" -> "plaintiff".
        Must run before line filtering to preserve content.

        Args:
            text: Input text

        Returns:
            Text with hyphenated words rejoined
        """
        logger.debug("Stage 1: De-hyphenation")
        start = time.time()
        original_len = len(text)

        try:
            text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
            duration = time.time() - start
            logger.debug("OK (%.3fs) - Rejoin hyphenated words", duration)
            logger.debug(
                "Input: %d | Output: %d | Delta: %+d",
                original_len,
                len(text),
                len(text) - original_len,
            )
        except Exception as e:
            duration = time.time() - start
            logger.debug("FAILED (%.3fs) - %s: %s", duration, type(e).__name__, e)
            raise

        return text

    def _stage_page_numbers(self, text: str) -> str:
        """
        Stage 2: Remove page number markers.

        Removes common page number patterns:
            - "Page 1", "Page 1 of 10"
            - "- 1 -" (dashed numbers)
            - Standalone numbers (1-4 digits)
            - "P. 1", "Pg. 1"
            - "1/10" (page X of Y)

        Args:
            text: Input text

        Returns:
            Text with page numbers removed
        """
        logger.debug("Stage 2: Page number removal")
        start = time.time()
        original_len = len(text)

        try:
            lines = text.split("\n")
            lines_filtered = []
            removed_count = 0

            for line in lines:
                if not self._is_page_number(line):
                    lines_filtered.append(line)
                else:
                    removed_count += 1
                    logger.debug("Removed page number: %s", line)

            text = "\n".join(lines_filtered)
            duration = time.time() - start
            logger.debug("OK (%.3fs) - Removed %d page markers", duration, removed_count)
            logger.debug(
                "Input: %d | Output: %d | Delta: %+d",
                original_len,
                len(text),
                len(text) - original_len,
            )
        except Exception as e:
            duration = time.time() - start
            logger.debug("FAILED (%.3fs) - %s: %s", duration, type(e).__name__, e)
            raise

        return text

    def _stage_line_filtering(self, text: str) -> str:
        """
        Stage 3: Filter out garbage lines while preserving content.

        Keeps lines that:
            - Have lowercase letters (normal prose)
            - Are legal headers (all-caps + legal keyword)
            - Have more letters than other characters

        Removes lines that:
            - Are too short (< MIN_LINE_LENGTH, usually 3)
            - Have no lowercase letters and aren't legal headers
            - Have more symbols than letters

        Args:
            text: Input text

        Returns:
            Text with garbage lines removed
        """
        logger.debug("Stage 3: Line filtering")
        start = time.time()
        original_len = len(text)

        try:
            normalized_lines = []
            input_lines = text.split("\n")

            for line in input_lines:
                if self._should_keep_line(line):
                    normalized_lines.append(line)

            removed_count = len(input_lines) - len(normalized_lines)
            text = "\n".join(normalized_lines)
            duration = time.time() - start
            logger.debug("OK (%.3fs) - Filtered %d lines", duration, removed_count)
            logger.debug(
                "Input: %d | Output: %d | Delta: %+d",
                original_len,
                len(text),
                len(text) - original_len,
            )
        except Exception as e:
            duration = time.time() - start
            logger.debug("FAILED (%.3fs) - %s: %s", duration, type(e).__name__, e)
            raise

        return text

    def _stage_whitespace(self, text: str) -> str:
        """
        Stage 4: Normalize whitespace.

        Collapses multiple consecutive blank lines into a single blank line
        and strips leading/trailing whitespace.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        logger.debug("Stage 4: Whitespace normalization")
        start = time.time()
        original_len = len(text)

        try:
            # Remove excess blank lines (max 1 between paragraphs)
            text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
            text = text.strip()
            duration = time.time() - start
            logger.debug("OK (%.3fs) - Normalize whitespace", duration)
            logger.debug(
                "Input: %d | Output: %d | Delta: %+d",
                original_len,
                len(text),
                len(text) - original_len,
            )
        except Exception as e:
            duration = time.time() - start
            logger.debug("FAILED (%.3fs) - %s: %s", duration, type(e).__name__, e)
            raise

        return text

    def _is_page_number(self, line: str) -> bool:
        """
        Check if a line is a page number marker.

        Patterns detected:
            - "Page 1", "Page 1 of 10"
            - "- 1 -" (dashed numbers, including en-dash)
            - Standalone numbers: "1", "23" (up to 4 digits)
            - "P. 1", "Pg. 1", "p. 1"
            - "1/10" (page X of Y format)

        Args:
            line: Single line of text

        Returns:
            True if line appears to be a page number

        Example:
            >>> normalizer = TextNormalizer()
            >>> normalizer._is_page_number("Page 5")
            True
            >>> normalizer._is_page_number("- 12 -")
            True
            >>> normalizer._is_page_number("The plaintiff")
            False
        """
        line = line.strip()

        # Pattern 1: "Page X" or "Page X of Y"
        if re.match(r"^Page\s+\d+(\s+of\s+\d+)?$", line, re.IGNORECASE):
            return True

        # Pattern 2: "- X -" or en-dash variants
        if re.match(r"^[-\u2013]\s*\d+\s*[-\u2013]$", line):
            return True

        # Pattern 3: Just a number (but not if it's part of a list like "1.")
        if re.match(r"^\d+$", line) and len(line) <= 4:
            return True

        # Pattern 4: "P. X" or "Pg. X" or "p. X"
        if re.match(r"^P(g)?\.?\s*\d+$", line, re.IGNORECASE):
            return True

        # Pattern 5: "X/Y" (page X of Y)
        return bool(re.match(r"^\d+/\d+$", line))

    def _should_keep_line(self, line: str) -> bool:
        """
        Determine if a line should be kept after filtering.

        A line is kept if it:
            1. Passes minimum length check (or is a legal header)
            2. Has lowercase letters (or is a legal header)
            3. Has more letters than symbols

        Args:
            line: Single line of text

        Returns:
            True if line should be kept
        """
        # Minimum length check
        if len(line) < MIN_LINE_LENGTH:
            # Exception: Allow short legal headers
            return bool(self._is_legal_header(line))

        # Check if line has lowercase letters
        has_lowercase = any(c.islower() for c in line)

        # Check if it's a legal header
        is_header = self._is_legal_header(line)

        # Count character types
        alpha_count = sum(c.isalpha() for c in line)
        other_count = sum(not c.isalpha() and not c.isspace() for c in line)

        # Keep if passes all tests
        return (has_lowercase or is_header) and alpha_count > other_count

    def _is_legal_header(self, line: str) -> bool:
        """
        Check if a line is a legal document header.

        Legal headers are all-caps lines under 50 characters that contain
        one of the legal keywords (COURT, PLAINTIFF, etc.).

        Args:
            line: Single line of text

        Returns:
            True if line appears to be a legal header

        Example:
            >>> normalizer = TextNormalizer()
            >>> normalizer._is_legal_header("SUPREME COURT OF NEW YORK")
            True
            >>> normalizer._is_legal_header("The plaintiff filed")
            False
        """
        return (
            line.isupper()
            and len(line) < 50
            and any(keyword in line for keyword in self.legal_keywords)
        )
