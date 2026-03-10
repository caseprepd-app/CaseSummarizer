"""
Title Page Remover Preprocessor

Removes title/cover pages from legal documents.
These pages contain case captions, court information, and formatting
that adds no value to AI summaries.

Two-tiered approach:
1. Page-level: Score whole pages for title characteristics
2. Line-level: Before removing a scored page, scan for content lines.
   A content line is 4+ words with mixed capitalization, confirmed by
   the next 5 substantive lines also being content lines. Look-ahead
   crosses page boundaries.

Common title page patterns:
- Case captions (parties, court, index numbers)
- Attorney information
- Court reporting service headers
- "DEPOSITION OF [NAME]" titles
"""

import logging
import re
from typing import ClassVar

logger = logging.getLogger(__name__)

from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult


class TitlePageRemover(BasePreprocessor):
    """
    Removes title/cover pages from legal documents.

    Strategy:
    1. Split text into page-like chunks (by form feeds or large gaps)
    2. Score each chunk for "title page" characteristics
    3. Before removing, check for content lines mid-page
    4. Remove whole pages or split at content boundary

    Title pages typically have:
    - Case captions with parties
    - Court names and addresses
    - Attorney listings
    - Minimal substantive content
    """

    name = "Title Page Remover"

    # Score threshold - pages scoring >= this are removed
    REMOVAL_THRESHOLD = 4

    # Maximum percentage of text that can be removed (safety limit)
    MAX_REMOVAL_PERCENT = 50

    # Only apply percentage limit for documents larger than this (bytes)
    LARGE_DOC_THRESHOLD = 10000  # 10KB

    # How many substantive lines after a candidate must also be content
    _LOOKAHEAD = 5

    # Fraction of words starting uppercase that signals a title line
    _MOSTLY_CAPS_RATIO = 0.75

    # Patterns and their scores
    TITLE_PAGE_PATTERNS: ClassVar[list[tuple[re.Pattern[str], int]]] = [
        # Court headers
        (re.compile(r"SUPREME\s+COURT", re.IGNORECASE), 2),
        (re.compile(r"CIVIL\s+COURT", re.IGNORECASE), 2),
        (re.compile(r"DISTRICT\s+COURT", re.IGNORECASE), 2),
        (re.compile(r"COURT\s+OF\s+(?:THE\s+)?STATE", re.IGNORECASE), 2),
        (re.compile(r"COUNTY\s+OF\s+[A-Z]+", re.IGNORECASE), 1),
        # Case caption markers
        (re.compile(r"^\s*[-x]+\s*$", re.MULTILINE), 1),  # Separator lines
        (re.compile(r"PLAINTIFF[,\s]", re.IGNORECASE), 2),
        (re.compile(r"DEFENDANT[,\s]", re.IGNORECASE), 2),
        (re.compile(r"\s+-\s*against\s*-\s+", re.IGNORECASE), 2),
        (re.compile(r"^\s*v\.?\s*$", re.MULTILINE | re.IGNORECASE), 1),
        # Index/case numbers
        (re.compile(r"INDEX\s*(?:NO\.?|NUMBER)", re.IGNORECASE), 2),
        (re.compile(r"CASE\s*(?:NO\.?|NUMBER)", re.IGNORECASE), 2),
        (re.compile(r"DOCKET\s*(?:NO\.?|NUMBER)", re.IGNORECASE), 2),
        # Deposition titles
        (re.compile(r"DEPOSITION\s+OF\s+[A-Z]", re.IGNORECASE), 3),
        (re.compile(r"EXAMINATION\s+BEFORE\s+TRIAL", re.IGNORECASE), 3),
        (re.compile(r"ORAL\s+DEPOSITION", re.IGNORECASE), 2),
        # Attorney information
        (re.compile(r"ATTORNEY[S]?\s+FOR", re.IGNORECASE), 2),
        (re.compile(r"COUNSEL\s+FOR", re.IGNORECASE), 2),
        (re.compile(r"LAW\s+OFFICE[S]?\s+OF", re.IGNORECASE), 1),
        (re.compile(r",?\s*(?:ESQ\.?|ESQUIRE)", re.IGNORECASE), 1),
        # Reporter information
        (re.compile(r"COURT\s+REPORTER", re.IGNORECASE), 2),
        (re.compile(r"CERTIFIED\s+SHORTHAND", re.IGNORECASE), 2),
        (re.compile(r"REPORTING\s+(?:SERVICE|COMPANY)", re.IGNORECASE), 1),
        # Appearance markers
        (re.compile(r"APPEARANCES?:", re.IGNORECASE), 2),
        (re.compile(r"ALSO\s+PRESENT:", re.IGNORECASE), 1),
        # Scheduling information (often on title pages)
        (re.compile(r"TAKEN\s+(?:ON|AT)\s+", re.IGNORECASE), 1),
        (re.compile(r"PURSUANT\s+TO", re.IGNORECASE), 1),
    ]

    # Patterns that indicate substantive content (negative score)
    CONTENT_PATTERNS: ClassVar[list[tuple[re.Pattern[str], int]]] = [
        (re.compile(r"^\s*Q[\\.:]", re.MULTILINE), -3),  # Q&A transcript
        (re.compile(r"^\s*A[\\.:]", re.MULTILINE), -3),
        (re.compile(r"THE\s+WITNESS:", re.IGNORECASE), -2),
        (re.compile(r"BY\s+(?:MR\.|MS\.|MRS\.)", re.IGNORECASE), -1),
    ]

    def _split_into_pages(self, text: str) -> list:
        """
        Split text into page-like chunks.

        Uses form feed characters if present, otherwise uses
        large whitespace gaps as page boundaries.

        Args:
            text: Full document text

        Returns:
            List of page-like text chunks
        """
        # Try form feed first
        if "\f" in text:
            pages = text.split("\f")
            return [p for p in pages if p.strip()]

        # Try page break patterns
        page_break = re.compile(r"\n{4,}|\n\s*-{10,}\s*\n")
        parts = page_break.split(text)

        if len(parts) > 1:
            return [p for p in parts if p.strip()]

        # No clear page breaks - return first ~2000 chars as "title page candidate"
        if len(text) > 3000:
            logger.warning(
                "No page breaks found in document (%d chars). "
                "Using arbitrary 2000-char split for title page detection.",
                len(text),
            )
            return [text[:2000], text[2000:]]

        return [text]

    def _score_page(self, page_text: str) -> int:
        """
        Score a page for title page characteristics.

        Higher score = more likely to be a title page.
        Negative scores from content patterns can offset.

        Args:
            page_text: Text of a single page

        Returns:
            Integer score
        """
        score = 0

        for pattern, points in self.TITLE_PAGE_PATTERNS:
            if pattern.search(page_text):
                score += points

        for pattern, points in self.CONTENT_PATTERNS:
            if pattern.search(page_text):
                score += points  # points are negative

        # Short pages with high scores are more likely title pages
        if len(page_text.strip()) < 500 and score > 0:
            score += 1

        return score

    def _is_mostly_caps(self, words: list[str]) -> bool:
        """
        Check if a word list is predominantly capitalized.

        Title page lines (firm names, party names, addresses) are mostly
        uppercase or title-case. Proceedings text has a natural mix.

        Args:
            words: Split word list from a line

        Returns:
            True if >= 75% of alphabetic words start uppercase
        """
        alpha_words = [w for w in words if w and w[0].isalpha()]
        if not alpha_words:
            return True  # No alpha words = not prose
        caps = sum(1 for w in alpha_words if w[0].isupper())
        return caps / len(alpha_words) >= self._MOSTLY_CAPS_RATIO

    def _is_content_line(self, line: str) -> bool:
        """
        Check whether a line looks like proceedings content.

        A content line has 4+ words with mixed capitalization — not
        75%+ uppercase-starting words. This distinguishes proceedings
        text from ALL-CAPS or Title Case title page lines.

        Args:
            line: A single line of text

        Returns:
            True if the line looks like proceedings content
        """
        stripped = line.strip()
        if not stripped:
            return False

        words = stripped.split()
        if len(words) < 4:
            return False

        return not self._is_mostly_caps(words)

    @staticmethod
    def _char_offset(lines: list[str], target_idx: int) -> int:
        """
        Calculate the character offset of a line within the original text.

        Args:
            lines: Lines from text.split('\\n')
            target_idx: Index of the target line

        Returns:
            Character position of the start of that line
        """
        offset = 0
        for i in range(target_idx):
            offset += len(lines[i]) + 1  # +1 for the \n
        return offset

    def _collect_substantive_lines(
        self, page_lines: list[str], start: int, remaining_pages: list[str]
    ) -> list[str]:
        """
        Gather substantive (non-blank) lines starting after a given index.

        Crosses page boundaries using remaining_pages so look-ahead
        isn't limited to the current page.

        Args:
            page_lines: Lines from the current page
            start: Index to start scanning from (exclusive)
            remaining_pages: Subsequent page texts for cross-page look-ahead

        Returns:
            List of up to _LOOKAHEAD substantive lines
        """
        substantive: list[str] = []

        # Lines remaining on the current page
        for j in range(start + 1, len(page_lines)):
            if page_lines[j].strip():
                substantive.append(page_lines[j])
                if len(substantive) >= self._LOOKAHEAD:
                    return substantive

        # Cross into remaining pages if needed
        for page_text in remaining_pages:
            for line in page_text.split("\n"):
                if line.strip():
                    substantive.append(line)
                    if len(substantive) >= self._LOOKAHEAD:
                        return substantive

        return substantive

    def _find_proceedings_start(self, page_text: str, remaining_pages: list[str]) -> int | None:
        """
        Scan a title page for the start of proceedings content.

        A line is confirmed as proceedings start when it is a content line
        AND the next 5 substantive (non-blank) lines are also content lines.
        Look-ahead crosses page boundaries.

        Args:
            page_text: Text of a single page
            remaining_pages: Subsequent pages for cross-page look-ahead

        Returns:
            Character offset where proceedings begin, or None
        """
        lines = page_text.split("\n")
        content_indices = [i for i, line in enumerate(lines) if self._is_content_line(line)]

        if not content_indices:
            return None

        for idx in content_indices:
            ahead = self._collect_substantive_lines(lines, idx, remaining_pages)

            if len(ahead) < self._LOOKAHEAD:
                continue  # Not enough lines to confirm

            all_content = all(self._is_content_line(line) for line in ahead)
            if all_content:
                logger.debug(
                    "Proceedings at line %d (%d/%d content lines ahead)",
                    idx,
                    len(ahead),
                    self._LOOKAHEAD,
                )
                return self._char_offset(lines, idx)

        return None

    def process(self, text: str) -> PreprocessingResult:
        """
        Remove title pages from text.

        For each candidate title page, checks for proceedings content
        before removing. If proceedings are found mid-page, only the
        title portion is discarded and all subsequent content is kept.

        Args:
            text: Input text potentially containing title pages

        Returns:
            PreprocessingResult with cleaned text and metadata
        """
        if not text:
            return PreprocessingResult(text=text, changes_made=0)

        pages = self._split_into_pages(text)

        # If only one "page", don't remove it
        if len(pages) <= 1:
            return PreprocessingResult(
                text=text, changes_made=0, metadata={"pages_analyzed": 1, "pages_removed": 0}
            )

        kept_pages, removed_count, removed_scores, removed_chars = self._filter_pages(
            pages, len(text)
        )

        # Rejoin pages
        result = "\n\n".join(kept_pages)

        # Safety check: never return empty text if input had content
        if not result.strip() and text.strip():
            largest_page = max(pages, key=len)
            result = largest_page
            removed_count = len(pages) - 1

        return PreprocessingResult(
            text=result,
            changes_made=removed_count,
            metadata={
                "pages_analyzed": len(pages),
                "pages_removed": removed_count,
                "removed_scores": removed_scores,
            },
        )

    def _filter_pages(
        self, pages: list[str], total_input_len: int
    ) -> tuple[list[str], int, list[int], int]:
        """
        Score and filter pages, preserving proceedings content.

        Only analyzes the first 3 pages. For each title-scoring page,
        checks for proceedings content before removing. If proceedings
        are found mid-page, splits at that boundary and keeps all
        subsequent pages.

        Args:
            pages: List of page text chunks
            total_input_len: Length of the original full text

        Returns:
            Tuple of (kept_pages, removed_count, removed_scores, removed_chars)
        """
        kept_pages = []
        removed_count = 0
        removed_scores = []
        removed_chars = 0

        for i, page in enumerate(pages):
            if i >= 3:
                kept_pages.extend(pages[i:])
                break

            score = self._score_page(page)

            # Percentage safety limit for large documents
            would_exceed_limit = False
            if total_input_len > self.LARGE_DOC_THRESHOLD:
                would_exceed_limit = (
                    (removed_chars + len(page)) / total_input_len * 100
                ) > self.MAX_REMOVAL_PERCENT

            if score < self.REMOVAL_THRESHOLD or would_exceed_limit:
                kept_pages.append(page)
                kept_pages.extend(pages[i + 1 :])
                break

            # Page scores as title — check for proceedings content
            remaining = pages[i + 1 :]
            proc_offset = self._find_proceedings_start(page, remaining)
            if proc_offset is not None:
                kept_pages.append(page[proc_offset:])
                removed_count += 1
                removed_scores.append(score)
                removed_chars += proc_offset
                kept_pages.extend(remaining)
                logger.info(
                    "Page %d (score=%d): proceedings at offset %d, "
                    "partial removal (%d/%d chars kept)",
                    i,
                    score,
                    proc_offset,
                    len(page) - proc_offset,
                    len(page),
                )
                break

            # No proceedings — remove entire page
            removed_count += 1
            removed_scores.append(score)
            removed_chars += len(page)
            logger.debug("Page %d removed (score=%d)", i, score)

        return kept_pages, removed_count, removed_scores, removed_chars
