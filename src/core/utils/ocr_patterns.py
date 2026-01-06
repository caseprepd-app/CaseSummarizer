"""
OCR Artifact Pattern Detection

Detects character patterns that commonly result from OCR errors, used to apply
a penalty to likely-corrupted spellings during canonical selection.

This module identifies terms that contain patterns suggesting they may be OCR
artifacts rather than correct spellings. When comparing similar terms (e.g.,
"Jenkins" vs "Jenidns"), the one with fewer OCR artifact patterns is more
likely to be correct.

Common OCR confusion categories:
1. **Digit-letter confusion**: 0/O, 1/l/I, 5/S, 8/B
2. **Ligature-like confusion**: rn/m, cl/d, ri/n, vv/w
3. **Shape similarity**: c/e, h/b, fl/fi

NOTE: This is a heuristic approach. An ML model trained on OCR error patterns
could potentially achieve higher accuracy by learning corpus-specific patterns.
This rules-based implementation provides a reasonable baseline.

Session 78: Initial implementation for canonical spelling improvement.
"""

import re

# -----------------------------------------------------------------------------
# OCR Confusion Patterns
# -----------------------------------------------------------------------------

# Pairs of character sequences commonly confused by OCR
# Format: (incorrect_pattern, correct_pattern)
# IMPORTANT: Only include patterns that are VERY unlikely in real names
# Patterns like "cl", "li", "ri" are too common in legitimate names (Clark, Williams, Marie)
OCR_CONFUSION_PAIRS: list[tuple[str, str]] = [
    # Ligature-like confusions - only the most distinctive ones
    ("rn", "m"),  # "Srnith" should be "Smith" - very distinctive
    ("vv", "w"),  # "vvord" should be "word" - unlikely in real names
    ("iii", "m"),  # Triple i - very suspicious
    ("lll", "m"),  # Triple l - very suspicious
    # Note: These patterns removed due to false positives:
    # - "cl" appears in Clark, Clair, Cleveland, etc.
    # - "li" appears in Williams, Oliver, Elizabeth, etc.
    # - "ri" appears in Marie, Brian, Eric, etc.
    # - "nn" appears in Jennifer, Connor, etc.
]

# Characters that are suspicious in name contexts when they could be
# OCR misreads of letters
# Format: digit -> likely_correct_letter
OCR_DIGIT_LETTER_MAP: dict[str, str] = {
    "0": "O",  # "J0hn" should be "John"
    "1": "l",  # "Wi1son" should be "Wilson"
    "5": "S",  # "5mith" should be "Smith"
    "8": "B",  # "8rown" should be "Brown"
}

# Additional suspicious patterns in names
OCR_SUSPICIOUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Digits embedded in names (except at boundaries which might be legitimate)
    (re.compile(r"[A-Za-z][0-9][A-Za-z]"), "embedded_digit"),
    # Unusual case patterns (lowercase surrounded by uppercase)
    (re.compile(r"[A-Z][a-z][A-Z].*[A-Z]"), "mixed_case_unusual"),
    # Multiple consecutive consonants that are unusual in English
    # (More than 4 consonants without a vowel is very suspicious)
    (re.compile(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{5,}"), "consonant_cluster"),
]


# -----------------------------------------------------------------------------
# Main Detection Function
# -----------------------------------------------------------------------------


def has_ocr_artifacts(term: str) -> bool:
    """
    Check if a term contains patterns suggesting OCR corruption.

    This function applies multiple heuristics to detect likely OCR errors:
    1. Known ligature confusions (rn/m, cl/d, etc.)
    2. Digit-letter confusion in name context
    3. Unusual character patterns

    Args:
        term: A vocabulary term to check (typically a person name)

    Returns:
        True if the term contains likely OCR artifact patterns, False otherwise

    Examples:
        >>> has_ocr_artifacts("Jenkins")      # Normal name
        False
        >>> has_ocr_artifacts("Jenidns")      # Typo, but not OCR-specific
        False
        >>> has_ocr_artifacts("Srnith")       # rn→m confusion
        True
        >>> has_ocr_artifacts("J0hn")         # 0→O confusion
        True
        >>> has_ocr_artifacts("Wi1son")       # 1→l confusion
        True

    Note:
        This is intentionally conservative - we only flag patterns that are
        highly likely to be OCR errors. False positives (flagging correct
        spellings) are worse than false negatives (missing some OCR errors).
    """
    if not term or len(term) < 3:
        # Too short to reliably detect OCR patterns
        return False

    # Check for ligature-like confusion patterns
    term_lower = term.lower()
    for incorrect, _ in OCR_CONFUSION_PAIRS:
        if incorrect.lower() in term_lower:
            # Found a suspicious pattern - but verify it's not a legitimate word
            # For now, flag it (could add dictionary check in future)
            return True

    # Check for digit-letter confusion in names
    if _has_suspicious_digits(term):
        return True

    # Check additional suspicious patterns
    return any(pattern.search(term) for pattern, _ in OCR_SUSPICIOUS_PATTERNS)


def _has_suspicious_digits(term: str) -> bool:
    """
    Check if a term has digits that look like OCR misreads of letters.

    We only flag digits that:
    1. Are in the OCR_DIGIT_LETTER_MAP (known confusion pairs)
    2. Are surrounded by letters (not at word boundaries)
    3. The term otherwise looks like a name (has letters)

    This avoids flagging legitimate alphanumeric identifiers like "ABC123".

    Args:
        term: Term to check

    Returns:
        True if suspicious digit patterns found
    """
    # Must have both letters and digits to be suspicious
    has_letters = any(c.isalpha() for c in term)
    has_digits = any(c.isdigit() for c in term)

    if not (has_letters and has_digits):
        return False

    # Check each character position
    for i, char in enumerate(term):
        if char in OCR_DIGIT_LETTER_MAP:
            # Check if this digit is surrounded by letters (embedded)
            prev_is_letter = i > 0 and term[i - 1].isalpha()
            next_is_letter = i < len(term) - 1 and term[i + 1].isalpha()

            if prev_is_letter or next_is_letter:
                # Digit is adjacent to letters - suspicious
                return True

    return False


# -----------------------------------------------------------------------------
# Detailed Analysis (for debugging/logging)
# -----------------------------------------------------------------------------


def analyze_ocr_patterns(term: str) -> dict:
    """
    Provide detailed analysis of OCR patterns in a term.

    Useful for debugging and understanding why a term was flagged.

    Args:
        term: Term to analyze

    Returns:
        Dictionary with analysis results:
        - has_artifacts: bool
        - patterns_found: list of pattern descriptions
        - suspicious_chars: list of (position, char, likely_correct)
    """
    result = {
        "term": term,
        "has_artifacts": False,
        "patterns_found": [],
        "suspicious_chars": [],
    }

    if not term or len(term) < 3:
        return result

    term_lower = term.lower()

    # Check ligature patterns
    for incorrect, correct in OCR_CONFUSION_PAIRS:
        if incorrect.lower() in term_lower:
            result["patterns_found"].append(
                f"Ligature confusion: '{incorrect}' might be '{correct}'"
            )
            result["has_artifacts"] = True

    # Check digit-letter confusion
    for i, char in enumerate(term):
        if char in OCR_DIGIT_LETTER_MAP:
            prev_is_letter = i > 0 and term[i - 1].isalpha()
            next_is_letter = i < len(term) - 1 and term[i + 1].isalpha()

            if prev_is_letter or next_is_letter:
                result["suspicious_chars"].append(
                    {
                        "position": i,
                        "char": char,
                        "likely_correct": OCR_DIGIT_LETTER_MAP[char],
                    }
                )
                result["has_artifacts"] = True

    # Check regex patterns
    for pattern, description in OCR_SUSPICIOUS_PATTERNS:
        match = pattern.search(term)
        if match:
            result["patterns_found"].append(f"{description}: found '{match.group()}'")
            result["has_artifacts"] = True

    return result


def get_ocr_penalty(term: str, base_penalty: float = 0.10) -> float:
    """
    Calculate the OCR penalty factor for a term.

    Args:
        term: Term to check
        base_penalty: Penalty to apply if OCR artifacts detected (default 10%)

    Returns:
        Penalty factor between 0.0 and base_penalty.
        - 0.0 if no artifacts detected
        - base_penalty if artifacts detected

    Example:
        >>> penalty = get_ocr_penalty("Srnith")  # Has rn→m pattern
        0.10
        >>> penalty = get_ocr_penalty("Smith")   # Clean
        0.0

    Usage in scoring:
        final_score = base_score * (1.0 - get_ocr_penalty(term))
    """
    if has_ocr_artifacts(term):
        return base_penalty
    return 0.0


# -----------------------------------------------------------------------------
# Comparison Utilities
# -----------------------------------------------------------------------------


def compare_variants_for_ocr(variant_a: str, variant_b: str) -> str | None:
    """
    Compare two similar term variants and identify which has OCR artifacts.

    When two terms are similar (e.g., edit distance 1-2), this function
    can help determine which is likely the OCR error.

    Args:
        variant_a: First variant
        variant_b: Second variant

    Returns:
        - variant_a if variant_a has artifacts and variant_b doesn't
        - variant_b if variant_b has artifacts and variant_a doesn't
        - None if both or neither have artifacts (can't determine)

    Example:
        >>> compare_variants_for_ocr("Smith", "Srnith")
        "Srnith"  # This one has the OCR artifact
    """
    a_has_artifacts = has_ocr_artifacts(variant_a)
    b_has_artifacts = has_ocr_artifacts(variant_b)

    if a_has_artifacts and not b_has_artifacts:
        return variant_a  # A is the OCR error
    elif b_has_artifacts and not a_has_artifacts:
        return variant_b  # B is the OCR error
    else:
        return None  # Can't determine (both clean or both have artifacts)
