"""
Case Number Extraction for Legal Documents.

Extracts case numbers, index numbers, and docket numbers from legal document
text using regex patterns for common US court formats.

Supported formats:
    - Federal: "Case No. 1:23-cv-12345"
    - NY Index: "Index No. 123456/2024"
    - Docket: "Docket No. 2024-12345"
    - Generic: "Case No.: 12345"

Example usage:
    >>> extractor = CaseNumberExtractor()
    >>> text = "SUPREME COURT Index No. 123456/2024 JOHN DOE v. ACME CORP"
    >>> extractor.extract(text)
    ['Index No. 123456/2024']
"""

import re
from typing import ClassVar


class CaseNumberExtractor:
    """
    Extracts case numbers from legal document text.

    Uses regex patterns to identify common case number formats used in
    US federal and state courts. Results are deduplicated.

    Example:
        >>> extractor = CaseNumberExtractor()
        >>> extractor.extract("Case No. 1:23-cv-12345-ABC")
        ['Case No. 1:23-cv-12345']
    """

    # Compiled regex patterns for common case number formats
    PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        # Federal court: "Case No. 1:23-cv-12345" or "Case No. 1:23-cv-12345-ABC"
        re.compile(r"Case\s+No\.?\s*:?\s*\d+:\d+-\w+-\d+", re.IGNORECASE),
        # NY Index Number: "Index No. 123456/2024"
        re.compile(r"Index\s+No\.?\s*:?\s*\d+/\d{4}", re.IGNORECASE),
        # Generic docket: "Docket No. 2024-12345"
        re.compile(r"Docket\s+No\.?\s*:?\s*\d+-\d+", re.IGNORECASE),
        # Generic case number: "Case No.: 12345"
        re.compile(r"Case\s+No\.?\s*:?\s*\d+", re.IGNORECASE),
    ]

    def extract(self, text: str) -> list[str]:
        """
        Extract all case numbers from text.

        Searches for multiple case number patterns and returns unique matches.
        More specific patterns (federal, index) are matched before generic ones.

        Args:
            text: Document text to search

        Returns:
            List of unique case number strings found in the text

        Example:
            >>> extractor = CaseNumberExtractor()
            >>> text = '''
            ... SUPREME COURT OF NEW YORK
            ... Index No. 123456/2024
            ... Case No. 789
            ... '''
            >>> sorted(extractor.extract(text))
            ['Case No. 789', 'Index No. 123456/2024']
        """
        case_numbers = []

        for pattern in self.PATTERNS:
            matches = pattern.findall(text)
            case_numbers.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for cn in case_numbers:
            if cn not in seen:
                seen.add(cn)
                unique.append(cn)

        return unique
