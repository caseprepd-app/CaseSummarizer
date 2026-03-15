"""
Title Page Remover Preprocessor

Removes title/cover pages from legal documents.
These pages contain case captions, court information, and formatting
that adds no value to search results or key excerpts.

Page-level approach: a page either scores above the threshold (removed whole)
or below (kept, and all subsequent pages are kept too).

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
    3. Remove whole pages that exceed the threshold; stop at first kept page

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

    def process(self, text: str) -> PreprocessingResult:
        """
        Remove title pages from text.

        Each page is scored independently. Pages scoring at or above the
        threshold are removed whole. The first page that does not score as
        a title page stops further removal — all remaining pages are kept.

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

        kept_pages, removed_count, removed_scores = self._filter_pages(pages, len(text))

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
    ) -> tuple[list[str], int, list[int]]:
        """
        Score and filter pages using page-level decisions only.

        Only analyzes the first 3 pages. A page scoring at or above the
        threshold is removed whole. The first page that does not meet the
        threshold stops removal — it and all subsequent pages are kept.

        Args:
            pages: List of page text chunks
            total_input_len: Length of the original full text

        Returns:
            Tuple of (kept_pages, removed_count, removed_scores)
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

            # Remove entire page
            removed_count += 1
            removed_scores.append(score)
            removed_chars += len(page)
            logger.debug("Page %d removed (score=%d)", i, score)

        return kept_pages, removed_count, removed_scores
