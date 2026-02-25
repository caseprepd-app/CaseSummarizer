"""
Header/Footer Remover Preprocessor

Detects and removes repetitive headers and footers from legal documents.
Uses frequency analysis to identify lines that appear on multiple pages.

Common patterns:
- Company/firm names repeated at top of each page
- Case captions repeated throughout
- Page numbers with document titles ("SMITH DEPOSITION - Page 12")
- Confidentiality notices repeated on every page

Configuration (via Settings > Text Preprocessing):
- Custom patterns: User-defined patterns to always remove
- Short-line detection: Toggle aggressive keyword matching for short lines
- Min occurrences: How many times a line must repeat to be removed
"""

import logging
import re
from collections import Counter
from typing import ClassVar

from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult

logger = logging.getLogger(__name__)


class HeaderFooterRemover(BasePreprocessor):
    """
    Removes repetitive headers and footers from legal documents.

    Uses frequency analysis: lines that appear N+ times AND
    match header/footer patterns are removed.

    Strategy:
    1. Split text into lines
    2. Count frequency of each normalized line
    3. Lines appearing N+ times AND matching patterns -> remove
    4. Preserve unique content lines

    This is conservative by design - would rather keep some headers
    than accidentally remove important content.

    Configuration is loaded from user preferences on initialization.
    """

    name = "Header/Footer Remover"

    # Default minimum occurrences (can be overridden by user prefs)
    DEFAULT_MIN_OCCURRENCES = 3

    # Maximum line length for headers/footers (longer = likely content)
    MAX_HEADER_LENGTH = 120

    # Shorter threshold for keyword-only matches (e.g., just "PLAINTIFF")
    MAX_SHORT_HEADER_LENGTH = 70

    # Patterns that indicate a line is likely a header/footer
    HEADER_FOOTER_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        # Page numbers with text
        re.compile(r"^\s*-?\s*\d+\s*-?\s*$"),  # Just a number: "- 12 -" or "12"
        re.compile(r"page\s+\d+\s*(of\s+\d+)?", re.IGNORECASE),
        re.compile(r"\bpage\s*:?\s*\d+", re.IGNORECASE),
        # Case captions
        re.compile(r"^\s*v\.?\s*$", re.IGNORECASE),  # Just "v." or "vs"
        re.compile(r"plaintiff.*defendant", re.IGNORECASE),
        re.compile(r"index\s*(?:no\.?|number)", re.IGNORECASE),
        re.compile(r"case\s*(?:no\.?|number)", re.IGNORECASE),
        re.compile(r"docket\s*(?:no\.?|number)", re.IGNORECASE),
        # Court/reporting headers
        re.compile(r"supreme\s+court", re.IGNORECASE),
        re.compile(r"court\s+of\s+(?:the\s+)?state", re.IGNORECASE),
        re.compile(r"reporting\s+(?:service|company)", re.IGNORECASE),
        re.compile(r"certified\s+court\s+reporter", re.IGNORECASE),
        # Confidentiality notices
        re.compile(r"confidential", re.IGNORECASE),
        re.compile(r"privileged", re.IGNORECASE),
        re.compile(r"not\s+for\s+(?:public\s+)?distribution", re.IGNORECASE),
        # Document titles
        re.compile(r"deposition\s+of", re.IGNORECASE),
        re.compile(r"examination\s+before\s+trial", re.IGNORECASE),
        re.compile(r"transcript\s+of", re.IGNORECASE),
        # Exhibit references in headers
        re.compile(r"exhibit\s+\d+", re.IGNORECASE),
        # Firm names (ends with common suffixes)
        re.compile(r",?\s*(?:LLP|PLLC|P\.?C\.?|LLC|L\.L\.C\.)\s*$", re.IGNORECASE),
        # Transcript section headers
        re.compile(r"\b(?:direct|cross|redirect|recross)\s*examination\b", re.IGNORECASE),
        re.compile(r"\bopening\s+statements?\b", re.IGNORECASE),
        re.compile(r"\bclosing\s+(?:statements?|arguments?)\b", re.IGNORECASE),
        re.compile(r"^\s*proceedings?\s*$", re.IGNORECASE),
    ]

    # Keywords that indicate header ONLY on short lines (≤ MAX_SHORT_HEADER_LENGTH)
    # These words appear in content too, so we only match when they dominate the line
    SHORT_LINE_KEYWORDS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"\bplaintiff\b", re.IGNORECASE),
        re.compile(r"\bdefendant\b", re.IGNORECASE),
        re.compile(r"\bproceedings?\b", re.IGNORECASE),
        re.compile(r"\b(?:direct|cross|redirect|recross)\b", re.IGNORECASE),
    ]

    def __init__(self):
        """
        Initialize the header/footer remover with user preferences.

        Loads configuration from user preferences:
        - custom_header_footer_patterns: User-defined patterns (one per line)
        - header_footer_short_line_detection: Enable aggressive keyword matching
        - header_footer_min_occurrences: Minimum repeats to consider header/footer
        """
        super().__init__()
        self._load_settings()

    def _load_settings(self) -> None:
        """Load settings from user preferences."""
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()

        # Load min occurrences
        self._min_occurrences = prefs.get(
            "header_footer_min_occurrences", self.DEFAULT_MIN_OCCURRENCES
        )

        # Load short-line detection toggle
        self._short_line_detection = prefs.get("header_footer_short_line_detection", True)

        # Load and compile custom patterns
        custom_patterns_str = prefs.get("custom_header_footer_patterns", "")
        self._custom_patterns: list[re.Pattern[str]] = []

        if custom_patterns_str:
            for line in custom_patterns_str.split("\n"):
                pattern_text = line.strip()
                if pattern_text:
                    try:
                        # Escape special regex chars and make case-insensitive
                        escaped = re.escape(pattern_text)
                        self._custom_patterns.append(re.compile(escaped, re.IGNORECASE))
                    except re.error as e:
                        logger.warning("Invalid custom pattern '%s': %s", pattern_text, e)

        if self._custom_patterns:
            logger.debug("Loaded %d custom header/footer patterns", len(self._custom_patterns))

    def _normalize_line(self, line: str) -> str:
        """
        Normalize a line for comparison purposes.

        Removes extra whitespace and page numbers to group
        variants of the same header together.

        Args:
            line: Raw line of text

        Returns:
            Normalized version for frequency counting
        """
        # Strip whitespace
        normalized = line.strip()

        # Remove trailing page numbers (common in headers)
        normalized = re.sub(r"\s*-?\s*\d+\s*-?\s*$", "", normalized)

        # Collapse whitespace
        normalized = " ".join(normalized.split())

        return normalized.lower()

    def _is_header_footer_candidate(self, line: str) -> bool:
        """
        Check if a line matches header/footer patterns.

        Uses three-tier matching:
        1. Custom user patterns (always checked first)
        2. Standard patterns match up to MAX_HEADER_LENGTH (120 chars)
        3. Short-line keywords only match up to MAX_SHORT_HEADER_LENGTH (70 chars)
           (can be disabled via settings)

        This prevents false positives like "The plaintiff testified..." while
        still catching short header lines like "PLAINTIFF / MR. KAUFER".

        Args:
            line: Line to check

        Returns:
            True if line matches any header/footer pattern
        """
        stripped = line.strip()
        line_len = len(stripped)

        # Empty lines are not headers
        if not stripped:
            return False

        # Too long for any header/footer pattern
        if line_len > self.MAX_HEADER_LENGTH:
            return False

        # Check custom user patterns first (these always take priority)
        if self._custom_patterns:
            if any(pattern.search(line) for pattern in self._custom_patterns):
                return True

        # Check standard patterns (up to MAX_HEADER_LENGTH)
        if any(pattern.search(line) for pattern in self.HEADER_FOOTER_PATTERNS):
            return True

        # Check short-line keywords (only if enabled and line is short enough)
        # These keywords appear in prose too, so we're more conservative
        if self._short_line_detection and line_len <= self.MAX_SHORT_HEADER_LENGTH:
            if any(pattern.search(line) for pattern in self.SHORT_LINE_KEYWORDS):
                # Extra check: line should not look like prose
                # Prose typically ends with sentence punctuation and has many words
                words = stripped.split()
                ends_with_sentence = stripped.endswith((".", "?", "!"))
                # Short lines with few words and no sentence ending = likely header
                if len(words) <= 8 and not ends_with_sentence:
                    return True

        return False

    def process(self, text: str) -> PreprocessingResult:
        """
        Remove repetitive headers and footers from text.

        Args:
            text: Input text potentially containing headers/footers

        Returns:
            PreprocessingResult with cleaned text and metadata
        """
        if not text:
            return PreprocessingResult(text=text, changes_made=0)

        lines = text.split("\n")

        # Pre-compute all normalizations ONCE (optimization)
        # Previously _normalize_line() was called 3x per line
        line_to_normalized: dict[int, str] = {}
        for i, line in enumerate(lines):
            line_to_normalized[i] = self._normalize_line(line)

        # Count normalized line frequencies
        line_counts: Counter = Counter()
        for normalized in line_to_normalized.values():
            if normalized:  # Don't count empty lines
                line_counts[normalized] += 1

        # Find lines that appear frequently AND match patterns
        lines_to_remove: set[str] = set()
        for normalized_line, count in line_counts.items():
            if count >= self._min_occurrences:
                # Check if any original line with this normalization matches patterns
                for i, line in enumerate(lines):
                    if line_to_normalized[
                        i
                    ] == normalized_line and self._is_header_footer_candidate(line):
                        lines_to_remove.add(normalized_line)
                        break

        # Remove matching lines
        result_lines = []
        removed_count = 0
        removed_examples = []

        for i, line in enumerate(lines):
            normalized = line_to_normalized[i]
            if normalized in lines_to_remove:
                removed_count += 1
                # Track first few examples for debugging
                if len(removed_examples) < 5:
                    removed_examples.append(line.strip()[:50])
            else:
                result_lines.append(line)

        return PreprocessingResult(
            text="\n".join(result_lines),
            changes_made=removed_count,
            metadata={
                "unique_patterns_removed": len(lines_to_remove),
                "total_lines_removed": removed_count,
                "examples": removed_examples,
            },
        )
