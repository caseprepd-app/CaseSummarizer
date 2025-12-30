"""
Name Regularization for Vocabulary Extraction

Post-processing filter that removes name fragments and OCR typo variants
by comparing against high-frequency canonical terms.

Two main filters:

1. Fragment Filter:
   - "Di Leo" in top quartile → remove "Di", "Leo" from bottom 3/4
   - Handles spaCy splitting multi-word names into separate entities

2. Typo Filter (1-character edit distance):
   - "Barbra Jenkins" in top quartile → remove "Barbr Jenkins", "Barbra Jenkinss"
   - Catches common OCR errors where one character is wrong

These filters run AFTER artifact_filter.py (which catches longer superstrings)
and work on the opposite direction (catching shorter fragments and typos).

Example:
    Top quartile: ["Di Leo", "Barbra Jenkins", "Memorial Hospital"]
    Bottom 3/4:   ["Di", "Leo", "Barbr Jenkins", "Barbra Jenkinss", "Hospital"]

    After filtering:
    - "Di" removed (fragment of "Di Leo")
    - "Leo" removed (fragment of "Di Leo")
    - "Barbr Jenkins" removed (1-char typo of "Barbra Jenkins")
    - "Barbra Jenkinss" removed (1-char typo of "Barbra Jenkins")
    - "Hospital" NOT removed (not a fragment, "Memorial Hospital" ≠ "Hospital")
"""

from src.logging_config import debug_log


def _edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.

    Edit distance = minimum number of single-character edits (insertions,
    deletions, or substitutions) required to change one string into another.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Integer edit distance
    """
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

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


def _is_fragment_of(fragment: str, canonical: str) -> bool:
    """
    Check if fragment is a word-level fragment of canonical.

    A fragment is a proper subset of words from the canonical term.
    "Di" is a fragment of "Di Leo" (one word from a two-word name).
    "Hospital" is NOT a fragment of "Memorial Hospital" (partial word match not allowed).

    Args:
        fragment: Potential fragment term (shorter)
        canonical: Canonical term to check against (longer)

    Returns:
        True if fragment is a word-level subset of canonical
    """
    fragment_lower = fragment.lower().strip()
    canonical_lower = canonical.lower().strip()

    # Split into words
    fragment_words = set(fragment_lower.split())
    canonical_words = set(canonical_lower.split())

    # Fragment must have fewer words than canonical
    if len(fragment_words) >= len(canonical_words):
        return False

    # All fragment words must appear in canonical words
    if not fragment_words.issubset(canonical_words):
        return False

    # Must be a PROPER subset (not equal)
    if fragment_words == canonical_words:
        return False

    return True


def filter_name_fragments(
    vocabulary: list[dict],
    top_fraction: float = 0.25,
    count_key: str = "In-Case Freq",
    term_key: str = "Term",
) -> list[dict]:
    """
    Remove terms that are word-level fragments of high-frequency canonical terms.

    Terms in the top fraction (by count) are considered canonical. Terms in
    the bottom portion are checked to see if they're fragments of any canonical term.

    Note: Top-quartile terms are NEVER removed, even if they're fragments of
    each other. This preserves high-frequency terms that users likely want.

    Example:
        "Di Leo" (count=50) in top quartile
        "Di" (count=3) in bottom → removed (fragment of "Di Leo")
        "Leo" (count=2) in bottom → removed (fragment of "Di Leo")

    Args:
        vocabulary: List of term dictionaries
        top_fraction: Fraction of terms to consider canonical (default 0.25 = top quartile)
        count_key: Dictionary key for term frequency count
        term_key: Dictionary key for the term string

    Returns:
        Filtered vocabulary list with fragments removed
    """
    if not vocabulary or len(vocabulary) < 4:
        return vocabulary

    # Sort by count descending
    sorted_vocab = sorted(
        vocabulary,
        key=lambda x: int(x.get(count_key, 0) or 0),
        reverse=True
    )

    # Split into top quartile and bottom 3/4
    split_index = max(1, int(len(sorted_vocab) * top_fraction))
    top_terms = sorted_vocab[:split_index]
    bottom_terms = sorted_vocab[split_index:]

    # Build set of canonical terms (multi-word only, since fragments are subsets)
    canonical_multiword = []
    for term_dict in top_terms:
        term = term_dict.get(term_key, "")
        if term and len(term.split()) > 1:  # Multi-word terms only
            canonical_multiword.append(term)

    if not canonical_multiword:
        return vocabulary

    debug_log(f"[NAME-REG] Fragment filter: {len(canonical_multiword)} multi-word canonical terms")

    # Filter ONLY bottom terms (top terms are always preserved)
    filtered_bottom = []
    removed_count = 0

    for term_dict in bottom_terms:
        term = term_dict.get(term_key, "")
        is_fragment = False

        for canonical in canonical_multiword:
            if _is_fragment_of(term, canonical):
                debug_log(f"[NAME-REG] Removing fragment '{term}' (subset of '{canonical}')")
                is_fragment = True
                removed_count += 1
                break

        if not is_fragment:
            filtered_bottom.append(term_dict)

    debug_log(f"[NAME-REG] Fragment filter removed {removed_count} terms")

    return top_terms + filtered_bottom


def filter_typo_variants(
    vocabulary: list[dict],
    top_fraction: float = 0.25,
    max_edit_distance: int = 1,
    min_term_length: int = 5,
    count_key: str = "In-Case Freq",
    term_key: str = "Term",
) -> list[dict]:
    """
    Remove terms that are 1-character typos of high-frequency canonical terms.

    Terms in the top fraction (by count) are considered canonical. Terms in
    the bottom portion are checked for edit distance to canonical terms.

    Example:
        "Barbra Jenkins" (count=50) in top quartile
        "Barbr Jenkins" (count=2) in bottom → removed (edit distance = 1)
        "Barbra Jenkinss" (count=1) in bottom → removed (edit distance = 1)

    Args:
        vocabulary: List of term dictionaries
        top_fraction: Fraction of terms to consider canonical (default 0.25 = top quartile)
        max_edit_distance: Maximum edit distance to consider a typo (default 1)
        min_term_length: Minimum term length to apply typo filter (default 5)
        count_key: Dictionary key for term frequency count
        term_key: Dictionary key for the term string

    Returns:
        Filtered vocabulary list with typo variants removed
    """
    if not vocabulary or len(vocabulary) < 4:
        return vocabulary

    # Sort by count descending
    sorted_vocab = sorted(
        vocabulary,
        key=lambda x: int(x.get(count_key, 0) or 0),
        reverse=True
    )

    # Split into top quartile and bottom 3/4
    split_index = max(1, int(len(sorted_vocab) * top_fraction))
    top_terms = sorted_vocab[:split_index]
    bottom_terms = sorted_vocab[split_index:]

    # Build list of canonical terms long enough for typo checking
    canonical_terms = []
    for term_dict in top_terms:
        term = term_dict.get(term_key, "")
        if term and len(term) >= min_term_length:
            canonical_terms.append(term.lower())

    if not canonical_terms:
        return vocabulary

    debug_log(f"[NAME-REG] Typo filter: {len(canonical_terms)} canonical terms (len >= {min_term_length})")

    # Filter bottom terms
    filtered_bottom = []
    removed_count = 0

    for term_dict in bottom_terms:
        term = term_dict.get(term_key, "")
        term_lower = term.lower()

        # Skip short terms (too prone to false positives)
        if len(term) < min_term_length:
            filtered_bottom.append(term_dict)
            continue

        is_typo = False

        for canonical in canonical_terms:
            # Skip if exact match (shouldn't happen but safety check)
            if term_lower == canonical:
                continue

            # Skip if lengths differ by more than max_edit_distance
            # (optimization: edit distance can't be less than length difference)
            if abs(len(term_lower) - len(canonical)) > max_edit_distance:
                continue

            distance = _edit_distance(term_lower, canonical)

            if distance <= max_edit_distance:
                debug_log(
                    f"[NAME-REG] Removing typo '{term}' "
                    f"(distance={distance} from '{canonical}')"
                )
                is_typo = True
                removed_count += 1
                break

        if not is_typo:
            filtered_bottom.append(term_dict)

    debug_log(f"[NAME-REG] Typo filter removed {removed_count} terms")

    return top_terms + filtered_bottom


def _single_pass_regularize(
    vocabulary: list[dict],
    top_fraction: float,
    min_canonical_count: int,
    count_key: str,
    term_key: str,
) -> tuple[list[dict], int, int]:
    """
    Single pass of name regularization.

    Args:
        vocabulary: List of term dictionaries
        top_fraction: Fraction of terms to consider canonical
        min_canonical_count: Minimum number of canonical terms
        count_key: Dictionary key for term frequency count
        term_key: Dictionary key for the term string

    Returns:
        Tuple of (filtered vocabulary, fragments_removed, typos_removed)
    """
    if not vocabulary or len(vocabulary) < 4:
        return vocabulary, 0, 0

    # Sort by count descending to determine canonical terms
    sorted_vocab = sorted(
        vocabulary,
        key=lambda x: int(x.get(count_key, 0) or 0),
        reverse=True
    )

    # Determine split point - use fraction OR min count, whichever gives more canonical terms
    # But never take more than 75% of the vocabulary (must leave some for filtering)
    fraction_index = int(len(sorted_vocab) * top_fraction)
    max_canonical = int(len(sorted_vocab) * 0.75)  # Cap at 75% of vocab
    split_index = min(max(min_canonical_count, fraction_index), max_canonical)
    # But don't exceed the vocabulary size
    split_index = min(split_index, len(sorted_vocab))

    top_terms = sorted_vocab[:split_index]
    bottom_terms = sorted_vocab[split_index:]

    # Build canonical term sets for both filters
    canonical_multiword = []
    canonical_for_typo = []

    for term_dict in top_terms:
        term = term_dict.get(term_key, "")
        if term:
            if len(term.split()) > 1:
                canonical_multiword.append(term)
            if len(term) >= 5:  # min_term_length for typo filter
                canonical_for_typo.append(term.lower())

    # Filter bottom terms only
    filtered_bottom = []
    fragment_removed = 0
    typo_removed = 0

    for term_dict in bottom_terms:
        term = term_dict.get(term_key, "")
        term_lower = term.lower()

        # Check 1: Is this a fragment of a canonical multi-word term?
        is_fragment = False
        for canonical in canonical_multiword:
            if _is_fragment_of(term, canonical):
                is_fragment = True
                fragment_removed += 1
                break

        if is_fragment:
            continue

        # Check 2: Is this a typo of a canonical term?
        is_typo = False
        if len(term) >= 5:  # Only check terms long enough
            for canonical in canonical_for_typo:
                if term_lower == canonical:
                    continue
                if abs(len(term_lower) - len(canonical)) > 1:
                    continue
                if _edit_distance(term_lower, canonical) <= 1:
                    is_typo = True
                    typo_removed += 1
                    break

        if is_typo:
            continue

        filtered_bottom.append(term_dict)

    return top_terms + filtered_bottom, fragment_removed, typo_removed


def regularize_names(
    vocabulary: list[dict],
    top_fraction: float = 0.25,
    min_canonical_count: int = 10,
    num_passes: int = 3,
    count_key: str = "In-Case Freq",
    term_key: str = "Term",
) -> list[dict]:
    """
    Apply name regularization filters with multiple passes.

    This function applies fragment and typo filtering multiple times. Each pass:
    1. Removes fragments (e.g., "Di" when "Di Leo" exists in top quartile)
    2. Removes typos (e.g., "Barbr Jenkins" when "Barbra Jenkins" exists)

    Multiple passes allow legitimate terms to "bubble up" as noise is removed.
    For example, if pass 1 removes 5 typos, a term that was #26 might become #21,
    moving into the top quartile for pass 2 where its typos can be caught.

    Args:
        vocabulary: List of term dictionaries
        top_fraction: Fraction of terms to consider canonical (default 0.25)
        min_canonical_count: Minimum number of canonical terms, regardless of
            percentage (default 10). Ensures small vocabularies have enough
            canonical terms to catch typos.
        num_passes: Number of filtering passes (default 3)
        count_key: Dictionary key for term frequency count
        term_key: Dictionary key for the term string

    Returns:
        Filtered vocabulary list
    """
    if not vocabulary or len(vocabulary) < 4:
        return vocabulary

    debug_log(f"[NAME-REG] Starting regularization on {len(vocabulary)} terms ({num_passes} passes)")

    result = vocabulary
    total_fragments = 0
    total_typos = 0

    for pass_num in range(1, num_passes + 1):
        prev_count = len(result)
        result, fragments, typos = _single_pass_regularize(
            result, top_fraction, min_canonical_count, count_key, term_key
        )
        total_fragments += fragments
        total_typos += typos

        removed_this_pass = prev_count - len(result)
        debug_log(f"[NAME-REG] Pass {pass_num}: removed {fragments} fragments, {typos} typos ({removed_this_pass} total)")

        # Early exit if no changes this pass
        if removed_this_pass == 0:
            debug_log(f"[NAME-REG] No changes in pass {pass_num}, stopping early")
            break

    debug_log(f"[NAME-REG] Regularization complete: removed {total_fragments} fragments, {total_typos} typos total")
    debug_log(f"[NAME-REG] Final: {len(result)} terms remaining")

    return result
