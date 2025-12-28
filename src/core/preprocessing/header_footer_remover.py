"""
Header/Footer Remover Preprocessor

Detects and removes repetitive headers and footers from legal documents.
Uses frequency analysis to identify lines that appear on multiple pages.

Common patterns:
- Company/firm names repeated at top of each page
- Case captions repeated throughout
- Page numbers with document titles ("SMITH DEPOSITION - Page 12")
- Confidentiality notices repeated on every page
"""

import re
from collections import Counter

from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult


class HeaderFooterRemover(BasePreprocessor):
    """
    Removes repetitive headers and footers from legal documents.

    Uses frequency analysis: lines that appear 3+ times AND
    match header/footer patterns are removed.

    Strategy:
    1. Split text into lines
    2. Count frequency of each normalized line
    3. Lines appearing 3+ times AND matching patterns → remove
    4. Preserve unique content lines

    This is conservative by design - would rather keep some headers
    than accidentally remove important content.
    """

    name = "Header/Footer Remover"

    # Minimum occurrences to consider something a header/footer
    MIN_OCCURRENCES = 3

    # Maximum line length for headers/footers (longer = likely content)
    MAX_HEADER_LENGTH = 120

    # Patterns that indicate a line is likely a header/footer
    HEADER_FOOTER_PATTERNS = [
        # Page numbers with text
        re.compile(r'^\s*-?\s*\d+\s*-?\s*$'),  # Just a number: "- 12 -" or "12"
        re.compile(r'page\s+\d+\s*(of\s+\d+)?', re.IGNORECASE),
        re.compile(r'\bpage\s*:?\s*\d+', re.IGNORECASE),

        # Case captions
        re.compile(r'^\s*v\.?\s*$', re.IGNORECASE),  # Just "v." or "vs"
        re.compile(r'plaintiff.*defendant', re.IGNORECASE),
        re.compile(r'index\s*(?:no\.?|number)', re.IGNORECASE),
        re.compile(r'case\s*(?:no\.?|number)', re.IGNORECASE),
        re.compile(r'docket\s*(?:no\.?|number)', re.IGNORECASE),

        # Court/reporting headers
        re.compile(r'supreme\s+court', re.IGNORECASE),
        re.compile(r'court\s+of\s+(?:the\s+)?state', re.IGNORECASE),
        re.compile(r'reporting\s+(?:service|company)', re.IGNORECASE),
        re.compile(r'certified\s+court\s+reporter', re.IGNORECASE),

        # Confidentiality notices
        re.compile(r'confidential', re.IGNORECASE),
        re.compile(r'privileged', re.IGNORECASE),
        re.compile(r'not\s+for\s+(?:public\s+)?distribution', re.IGNORECASE),

        # Document titles
        re.compile(r'deposition\s+of', re.IGNORECASE),
        re.compile(r'examination\s+before\s+trial', re.IGNORECASE),
        re.compile(r'transcript\s+of', re.IGNORECASE),

        # Exhibit references in headers
        re.compile(r'exhibit\s+\d+', re.IGNORECASE),

        # Firm names (ends with common suffixes)
        re.compile(r',?\s*(?:LLP|PLLC|P\.?C\.?|LLC|L\.L\.C\.)\s*$', re.IGNORECASE),
    ]

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
        normalized = re.sub(r'\s*-?\s*\d+\s*-?\s*$', '', normalized)

        # Collapse whitespace
        normalized = ' '.join(normalized.split())

        return normalized.lower()

    def _is_header_footer_candidate(self, line: str) -> bool:
        """
        Check if a line matches header/footer patterns.

        Args:
            line: Line to check

        Returns:
            True if line matches any header/footer pattern
        """
        # Too long for typical header/footer
        if len(line.strip()) > self.MAX_HEADER_LENGTH:
            return False

        # Empty lines are not headers
        if not line.strip():
            return False

        # Check against patterns
        for pattern in self.HEADER_FOOTER_PATTERNS:
            if pattern.search(line):
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

        lines = text.split('\n')

        # Count normalized line frequencies
        line_counts: Counter = Counter()
        for line in lines:
            normalized = self._normalize_line(line)
            if normalized:  # Don't count empty lines
                line_counts[normalized] += 1

        # Find lines that appear frequently AND match patterns
        lines_to_remove: set[str] = set()
        for normalized_line, count in line_counts.items():
            if count >= self.MIN_OCCURRENCES:
                # Check if any original line with this normalization matches patterns
                for line in lines:
                    if self._normalize_line(line) == normalized_line:
                        if self._is_header_footer_candidate(line):
                            lines_to_remove.add(normalized_line)
                            break

        # Remove matching lines
        result_lines = []
        removed_count = 0
        removed_examples = []

        for line in lines:
            normalized = self._normalize_line(line)
            if normalized in lines_to_remove:
                removed_count += 1
                # Track first few examples for debugging
                if len(removed_examples) < 5:
                    removed_examples.append(line.strip()[:50])
            else:
                result_lines.append(line)

        return PreprocessingResult(
            text='\n'.join(result_lines),
            changes_made=removed_count,
            metadata={
                'unique_patterns_removed': len(lines_to_remove),
                'total_lines_removed': removed_count,
                'examples': removed_examples,
            }
        )
