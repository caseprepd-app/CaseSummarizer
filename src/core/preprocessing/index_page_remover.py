"""
Index Page Remover Preprocessor

Detects and removes index/concordance pages from legal transcripts.
Index pages contain word counts like "morning (14)" and page:line
references like "330:7,8,17;331:15" which confuse NER extraction.

These pages always appear at the end of transcripts, so once detected,
the index page AND all subsequent pages are removed.
"""

import logging
import re

from src.config import (
    INDEX_CHAR_WINDOW_SIZE,
    INDEX_DETECTION_WINDOW_SIZE,
    INDEX_ESTIMATED_CHARS_PER_LINE,
    INDEX_MAX_CHECK_LENGTH,
    INDEX_MIN_DENSITY_PERCENT,
    INDEX_MIN_INDEX_LINES,
    INDEX_MIN_TEXT_LENGTH,
    INDEX_PAGE_REF_DIVISOR,
    INDEX_TAIL_CHECK_FRACTION,
)
from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult

logger = logging.getLogger(__name__)


class IndexPageRemover(BasePreprocessor):
    """
    Detects and removes index pages from end of documents.

    Index pages contain word counts like "morning (14)" and page:line
    references like "330:7,8,17;331:15". Once detected, the index page
    and ALL subsequent pages are removed (indexes are always at the end).

    Detection is conservative - thresholds configured in src/config.py.
    """

    name = "Index Page Remover"

    # Thresholds from config (very conservative to avoid false positives)
    MIN_INDEX_LINES = INDEX_MIN_INDEX_LINES
    MIN_DENSITY_PERCENT = INDEX_MIN_DENSITY_PERCENT

    # Patterns for detecting index content
    WORD_WITH_COUNT = re.compile(r"[A-Za-z]+\s*\(\d+\)")  # "morning (14)"
    PAGE_LINE_REF = re.compile(r"\d{2,4}:\d{1,2}")  # "330:7"

    def _split_into_pages(self, text: str) -> list[str]:
        """
        Split text into pages using form feeds or whitespace gaps.

        Args:
            text: Full document text

        Returns:
            List of page-like text chunks
        """
        if "\f" in text:
            return [p for p in text.split("\f") if p.strip()]

        # Fall back to 4+ newlines as page break
        parts = re.split(r"\n{4,}", text)
        return [p for p in parts if p.strip()] if len(parts) > 1 else [text]

    def _is_index_line(self, line: str) -> bool:
        """
        Check if a line looks like an index entry.

        Args:
            line: Single line of text

        Returns:
            True if line matches index patterns
        """
        line = line.strip()
        if not line:
            return False

        # Must have word with count OR 2+ page:line references
        has_word_count = bool(self.WORD_WITH_COUNT.search(line))
        ref_count = len(self.PAGE_LINE_REF.findall(line))

        return has_word_count or ref_count >= 2

    def _is_index_page(self, page_text: str) -> bool:
        """
        Check if a page is an index page.

        Requires both:
        - At least MIN_INDEX_LINES lines matching index patterns
        - At least MIN_DENSITY_PERCENT of non-empty lines matching

        Args:
            page_text: Text content of a single page

        Returns:
            True if page appears to be an index
        """
        lines = [line for line in page_text.split("\n") if line.strip()]
        if len(lines) < self.MIN_INDEX_LINES:
            return False

        index_lines = sum(1 for line in lines if self._is_index_line(line))
        density = (index_lines / len(lines)) * 100

        return index_lines >= self.MIN_INDEX_LINES and density >= self.MIN_DENSITY_PERCENT

    def _check_tail_for_index(self, text: str) -> PreprocessingResult:
        """
        Check the tail of a document for index content.

        Works with both multi-line and single-line text (after reconciliation
        strips newlines). Uses pattern density in character windows.

        Args:
            text: Full document text

        Returns:
            PreprocessingResult with index section removed if found
        """
        lines = text.split("\n")
        total_lines = len(lines)

        # If we have multiple lines, use line-based detection
        if total_lines >= self.MIN_INDEX_LINES:
            return self._check_tail_by_lines(lines)

        # Single line (or few lines) - use character-window detection
        # This handles text where newlines were stripped during extraction
        return self._check_tail_by_chars(text)

    def _check_tail_by_lines(self, lines: list[str]) -> PreprocessingResult:
        """Line-based index detection for text with preserved newlines."""
        total_lines = len(lines)
        window_size = INDEX_DETECTION_WINDOW_SIZE
        index_start_line = None

        # First, check the very last window
        last_window = lines[-window_size:]
        non_empty_last = [line for line in last_window if line.strip()]
        if non_empty_last:
            idx_count = sum(1 for line in non_empty_last if self._is_index_line(line))
            logger.debug(
                "Last %s lines: %s/%s index-like (%.0f%%)",
                window_size,
                idx_count,
                len(non_empty_last),
                idx_count / len(non_empty_last) * 100,
            )

        for start in range(total_lines - window_size, -1, -window_size // 2):
            if start < 0:
                start = 0
            window = lines[start : start + window_size]
            non_empty = [line for line in window if line.strip()]

            if len(non_empty) < self.MIN_INDEX_LINES:
                continue

            index_count = sum(1 for line in non_empty if self._is_index_line(line))
            density = (index_count / len(non_empty)) * 100

            if index_count >= self.MIN_INDEX_LINES and density >= self.MIN_DENSITY_PERCENT:
                index_start_line = start
            else:
                if index_start_line is not None:
                    break

        if index_start_line is None:
            logger.debug("No index section found in tail (line-based)")
            return PreprocessingResult(
                text="\n".join(lines),
                changes_made=0,
                metadata={"pages_analyzed": 1, "index_start": None},
            )

        kept_lines = lines[:index_start_line]
        removed_lines = total_lines - index_start_line

        logger.debug(
            "Found index at line %s, removing %s lines",
            index_start_line,
            removed_lines,
        )

        return PreprocessingResult(
            text="\n".join(kept_lines),
            changes_made=removed_lines,
            metadata={
                "pages_analyzed": 1,
                "index_start_line": index_start_line,
                "lines_removed": removed_lines,
            },
        )

    def _check_tail_by_chars(self, text: str) -> PreprocessingResult:
        """
        Character-window index detection for single-line text.

        When newlines are stripped during extraction, we detect index content
        by looking for high density of index patterns (word counts and page:line refs)
        in character windows at the end of the document.
        """
        text_len = len(text)
        if text_len < INDEX_MIN_TEXT_LENGTH:
            return PreprocessingResult(
                text=text,
                changes_made=0,
                metadata={"pages_analyzed": 1, "index_start": None},
            )

        # Check last 1/N of document for index patterns
        check_len = min(text_len // INDEX_TAIL_CHECK_FRACTION, INDEX_MAX_CHECK_LENGTH)
        tail = text[-check_len:]

        # Count index patterns in the tail
        word_counts = len(self.WORD_WITH_COUNT.findall(tail))
        page_refs = len(self.PAGE_LINE_REF.findall(tail))

        # Estimate "line equivalents" based on chars per line
        estimated_lines = check_len // INDEX_ESTIMATED_CHARS_PER_LINE
        pattern_density = (
            (word_counts + page_refs / INDEX_PAGE_REF_DIVISOR) / max(estimated_lines, 1) * 100
        )

        logger.debug(
            "Tail analysis (%s chars): %s word counts, %s page refs, density=%.0f%%",
            check_len,
            word_counts,
            page_refs,
            pattern_density,
        )

        # If tail has high pattern density, scan backwards to find start
        if pattern_density < self.MIN_DENSITY_PERCENT:
            logger.debug("No index section found in tail (char-based)")
            return PreprocessingResult(
                text=text,
                changes_made=0,
                metadata={"pages_analyzed": 1, "index_start": None},
            )

        # Scan backwards to find where index content starts
        window_size = INDEX_CHAR_WINDOW_SIZE
        index_start_pos = None

        for pos in range(text_len - window_size, 0, -window_size // 2):
            window = text[pos : pos + window_size]
            wc = len(self.WORD_WITH_COUNT.findall(window))
            pr = len(self.PAGE_LINE_REF.findall(window))
            est_lines = window_size // INDEX_ESTIMATED_CHARS_PER_LINE
            density = (wc + pr / INDEX_PAGE_REF_DIVISOR) / max(est_lines, 1) * 100

            if density >= self.MIN_DENSITY_PERCENT:
                index_start_pos = pos
            else:
                # Found non-index content
                if index_start_pos is not None:
                    break

        if index_start_pos is None:
            return PreprocessingResult(
                text=text,
                changes_made=0,
                metadata={"pages_analyzed": 1, "index_start": None},
            )

        # Remove from index_start_pos to end
        kept_text = text[:index_start_pos].rstrip()
        removed_chars = text_len - len(kept_text)

        logger.debug(
            "Found index at char %s, removing %s chars (%s%%)",
            index_start_pos,
            removed_chars,
            removed_chars * 100 // text_len,
        )

        return PreprocessingResult(
            text=kept_text,
            changes_made=removed_chars,
            metadata={
                "pages_analyzed": 1,
                "index_start_char": index_start_pos,
                "chars_removed": removed_chars,
            },
        )

    def process(self, text: str) -> PreprocessingResult:
        """
        Remove index pages and all subsequent pages.

        Args:
            text: Input text potentially containing index pages

        Returns:
            PreprocessingResult with cleaned text and metadata
        """
        if not text:
            return PreprocessingResult(text=text, changes_made=0)

        pages = self._split_into_pages(text)
        logger.debug("Split into %s pages (form_feeds=%s)", len(pages), chr(12) in text)

        if len(pages) <= 1:
            # Single page - check if tail has index content
            return self._check_tail_for_index(text)

        # Find first index page
        index_start = None
        for i, page in enumerate(pages):
            if self._is_index_page(page):
                index_start = i
                debug_log(f"[Index Page Remover] Index detected at page {i + 1}")
                break

        if index_start is None:
            return PreprocessingResult(
                text=text,
                changes_made=0,
                metadata={"pages_analyzed": len(pages), "index_start": None},
            )

        # Keep only pages before the index
        kept_pages = pages[:index_start]
        removed_count = len(pages) - index_start

        debug_log(
            f"[Index Page Remover] Removing {removed_count} pages "
            f"(index starts at page {index_start + 1} of {len(pages)})"
        )

        result = "\n\n".join(kept_pages) if kept_pages else text

        return PreprocessingResult(
            text=result,
            changes_made=removed_count,
            metadata={
                "pages_analyzed": len(pages),
                "index_start": index_start,
                "pages_removed": removed_count,
            },
        )
