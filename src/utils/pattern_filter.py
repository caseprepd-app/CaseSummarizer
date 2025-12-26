"""
Pattern Filter Utility

Provides reusable pattern matching filters for NER and vocabulary extraction.
Centralizes regex pattern definitions and matching logic to reduce duplication.

Usage:
    from src.utils.pattern_filter import (
        PatternFilter, MatchMethod,
        ADDRESS_FILTER, LEGAL_BOILERPLATE_FILTER, ...
    )

    # Check if text matches any pattern in filter
    if ADDRESS_FILTER.matches("123 Main Street"):
        print("Looks like an address")
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class MatchMethod(Enum):
    """How to apply regex patterns."""
    MATCH = 'match'       # Match at start of string (re.match)
    SEARCH = 'search'     # Match anywhere in string (re.search)
    FULLMATCH = 'full'    # Match entire string (re.fullmatch)


@dataclass
class PatternFilter:
    """
    Reusable pattern matching filter.

    Compiles regex patterns once and provides efficient matching.

    Attributes:
        patterns: Tuple of regex pattern strings
        method: How to apply the patterns (match, search, fullmatch)
        case_sensitive: Whether matching is case-sensitive (default False)
    """
    patterns: tuple[str, ...]
    method: MatchMethod = MatchMethod.SEARCH
    case_sensitive: bool = False
    _compiled: tuple[re.Pattern, ...] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Compile patterns on initialization."""
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._compiled = tuple(re.compile(p, flags) for p in self.patterns)

    def matches(self, text: str) -> bool:
        """
        Check if text matches any pattern in the filter.

        Args:
            text: Text to check

        Returns:
            True if text matches any pattern
        """
        for pattern in self._compiled:
            if self.method == MatchMethod.MATCH:
                if pattern.match(text):
                    return True
            elif self.method == MatchMethod.SEARCH:
                if pattern.search(text):
                    return True
            elif self.method == MatchMethod.FULLMATCH:
                if pattern.fullmatch(text):
                    return True
        return False


# =============================================================================
# Pre-built Filters for NER Algorithm
# =============================================================================

# Word variation patterns (plurals, possessives, etc.)
VARIATION_FILTER = PatternFilter(
    patterns=(
        r'^[a-z]+\(s\)$',          # plaintiff(s), defendant(s)
        r'^[a-z]+s\(s\)$',         # defendants(s)
        r'^[a-z]+\([a-z]+\)$',     # word(variant)
        r"^[a-z]+'s$",             # plaintiff's
        r'^[a-z]+-[a-z]+$',        # hyphenated words
    ),
    method=MatchMethod.MATCH,
    case_sensitive=True,  # These patterns expect lowercase
)

# OCR error patterns (line breaks, digit-letter mixups, punctuation errors)
OCR_ERROR_FILTER = PatternFilter(
    patterns=(
        r'^[A-Za-z]+-[A-Z][a-z]',     # Line-break artifacts: "Hos-pital"
        r'.*[0-9][A-Za-z]{2,}[0-9]',  # Digit-letter-digit: "3ohn5mith"
        r'[A-Za-z]+[0-9]+[A-Za-z]+',  # Digit(s) embedded: "Joh3n", "sp1ne"
        r'^[0-9]+[A-Za-z]+',          # Leading digit(s): "1earn", "3ohn"
        r'[A-Za-z]+[0-9]+$',          # Trailing digit(s): "learn1"
        r'[A-Za-z]+[;:][A-Za-z]+',    # Punctuation errors: "John;Smith"
    ),
    method=MatchMethod.SEARCH,
    case_sensitive=True,
)

# Legal citation patterns
LEGAL_CITATION_FILTER = PatternFilter(
    patterns=(
        r'^[A-Z]{2,}\s+(?:SS|§)\s*\d+',
        r'^\w+\s+Law\s+(?:SS|§)\s*\d+',
        r'^\d+\s+[A-Z]+\s+\d+',
        r'^[A-Z]{2,}\s+\d+',
    ),
    method=MatchMethod.MATCH,
    case_sensitive=True,
)

# Legal boilerplate phrases
LEGAL_BOILERPLATE_FILTER = PatternFilter(
    patterns=(
        r'Verified\s+(?:Bill|Answer|Complaint|Petition)',
        r'Notice\s+of\s+Commencement',
        r'Cause\s+of\s+Action',
        r'Honorable\s+Court',
        r'Answering\s+Defendant',
    ),
    method=MatchMethod.SEARCH,
    case_sensitive=False,
)

# Case citation pattern (X v. Y)
CASE_CITATION_FILTER = PatternFilter(
    patterns=(
        r'^[A-Z][a-zA-Z]+\s+v\.?\s+[A-Z][a-zA-Z]+$',
    ),
    method=MatchMethod.MATCH,
    case_sensitive=True,
)

# Geographic code patterns (ZIP codes, etc.)
GEOGRAPHIC_CODE_FILTER = PatternFilter(
    patterns=(
        r'^\d{5}(?:-\d{4})?$',   # ZIP codes
        r'^[A-Z]{2}\s+\d{5}$',   # State + ZIP
    ),
    method=MatchMethod.MATCH,
    case_sensitive=True,
)

# Address fragment patterns
ADDRESS_FILTER = PatternFilter(
    patterns=(
        r'\d+(?:st|nd|rd|th)\s+Floor',
        r'\b(?:Street|Drive|Avenue|Road|Lane|Court|Boulevard|Place|Way)\b',
        r'^\d+\s+[A-Z]',
    ),
    method=MatchMethod.SEARCH,
    case_sensitive=False,
)

# Document fragment patterns
DOCUMENT_FRAGMENT_FILTER = PatternFilter(
    patterns=(
        r'^(?:SUPREME|CIVIL|FAMILY|DISTRICT)\s+COURT',
        r'^NOTICE\s+OF',
        r'Attorneys?\s+for\s+(?:Plaintiff|Defendant)',
        r'Services?\s+-\s+(?:Plaintiff|Defendant|None)',
        r"^(?:Plaintiff|Defendant)(?:'s)?$",
        r'^(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+CAUSE',
        r'^\d+\s+of\s+\d+$',
    ),
    method=MatchMethod.SEARCH,
    case_sensitive=False,
)

# Acronym pattern (2+ uppercase letters)
ACRONYM_FILTER = PatternFilter(
    patterns=(
        r'[A-Z]{2,}',
    ),
    method=MatchMethod.FULLMATCH,
    case_sensitive=True,
)

# Title abbreviations (as a frozen set for fast lookup)
TITLE_ABBREVIATIONS: frozenset[str] = frozenset({
    'dr', 'mr', 'mrs', 'ms', 'md', 'phd', 'esq', 'jr', 'sr', 'ii', 'iii', 'iv',
    'dds', 'dvm', 'od', 'do', 'rn', 'lpn', 'np', 'pa', 'pt', 'ot', 'cpa',
    'jd', 'llm', 'mba', 'cfa', 'pe', 'ra',
})

# Entity length limits
MIN_ENTITY_LENGTH = 3
MAX_ENTITY_LENGTH = 60


def matches_entity_filter(entity_text: str) -> bool:
    """
    Check if an entity should be filtered out based on pattern matching.

    Combines multiple filters for entity validation.

    Args:
        entity_text: The entity text to check

    Returns:
        True if entity should be filtered OUT (excluded)
    """
    if len(entity_text) < MIN_ENTITY_LENGTH:
        return True
    if len(entity_text) > MAX_ENTITY_LENGTH:
        return True

    if ADDRESS_FILTER.matches(entity_text):
        return True
    if DOCUMENT_FRAGMENT_FILTER.matches(entity_text):
        return True
    if LEGAL_BOILERPLATE_FILTER.matches(entity_text):
        return True
    if CASE_CITATION_FILTER.matches(entity_text):
        return True

    return False


def matches_token_filter(token_text: str) -> bool:
    """
    Check if a token should be filtered out based on pattern matching.

    Used for single-token filtering in NER extraction.

    Args:
        token_text: The token text to check

    Returns:
        True if token should be filtered OUT (excluded)
    """
    if LEGAL_CITATION_FILTER.matches(token_text):
        return True
    if LEGAL_BOILERPLATE_FILTER.matches(token_text):
        return True
    if CASE_CITATION_FILTER.matches(token_text):
        return True
    if GEOGRAPHIC_CODE_FILTER.matches(token_text):
        return True
    if OCR_ERROR_FILTER.matches(token_text):
        return True

    return False


def is_valid_acronym(text: str) -> bool:
    """
    Check if text is a valid acronym (2+ uppercase letters, not a title).

    Args:
        text: Text to check

    Returns:
        True if text is a valid acronym
    """
    if not ACRONYM_FILTER.matches(text):
        return False
    if text.lower() in TITLE_ABBREVIATIONS:
        return False
    return True
