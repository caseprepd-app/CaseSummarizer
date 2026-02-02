"""
String Utilities for Vocabulary Processing

Shared string manipulation functions used across vocabulary extraction modules.
Centralizes common operations to avoid code duplication (DRY principle).

Functions:
    edit_distance: Levenshtein edit distance between two strings
    normalize_term: Lowercase and strip whitespace from a term
    get_term_words: Split a normalized term into words

Session 82: Extracted from artifact_filter.py, name_regularizer.py, and
gibberish_filter.py to eliminate duplicate implementations.
"""


def fuzzy_match(s1: str, s2: str, threshold: float = 0.8) -> tuple[bool, float]:
    """
    Check if two strings are similar using SequenceMatcher ratio.

    Centralizes the repeated pattern of SequenceMatcher(None, a, b).ratio()
    compared against a threshold.

    Args:
        s1: First string
        s2: Second string
        threshold: Minimum similarity ratio to consider a match (0-1)

    Returns:
        Tuple of (is_match, ratio) where is_match = ratio >= threshold

    Examples:
        >>> fuzzy_match("Smith", "Smitb", 0.8)
        (True, 0.8)
        >>> fuzzy_match("hello", "world", 0.8)
        (False, 0.2)
    """
    from difflib import SequenceMatcher

    ratio = SequenceMatcher(None, s1, s2).ratio()
    return (ratio >= threshold, ratio)


def edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.

    Edit distance = minimum number of single-character edits (insertions,
    deletions, or substitutions) required to change one string into another.

    This is the standard dynamic programming implementation with O(mn) time
    and O(min(m,n)) space complexity.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Integer edit distance (0 = identical strings)

    Examples:
        >>> edit_distance("kitten", "sitting")
        3
        >>> edit_distance("Smith", "Smitb")
        1
        >>> edit_distance("Jenkins", "Jenidns")
        2
        >>> edit_distance("hello", "hello")
        0
    """
    # Ensure s1 is the longer string for space optimization
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    # Base case: empty string
    if len(s2) == 0:
        return len(s1)

    # Use single row for space optimization
    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_term(term: str) -> str:
    """
    Normalize a term for consistent comparison.

    Applies lowercase and strips leading/trailing whitespace.
    This is the canonical normalization for term matching.

    Args:
        term: Raw term string

    Returns:
        Normalized term (lowercase, stripped)

    Examples:
        >>> normalize_term("  John Smith  ")
        'john smith'
        >>> normalize_term("RADICULOPATHY")
        'radiculopathy'
    """
    return term.lower().strip()


def get_term_words(term: str) -> list[str]:
    """
    Split a term into normalized words.

    Applies normalize_term() then splits on whitespace.

    Args:
        term: Raw term string

    Returns:
        List of lowercase words

    Examples:
        >>> get_term_words("John Smith")
        ['john', 'smith']
        >>> get_term_words("  Di Leo  ")
        ['di', 'leo']
    """
    return normalize_term(term).split()
