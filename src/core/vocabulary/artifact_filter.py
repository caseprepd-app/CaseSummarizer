"""
Artifact Filter for Vocabulary Extraction

Post-processing filter that removes false positive terms by detecting
substring containment. When a high-frequency "canonical" term appears
within a lower-frequency term, the longer term is likely an OCR or
formatting artifact.

Example:
    - "Ms. Di Leo" (count=50) is canonical
    - "Ms. Di Leo:" (count=2) contains canonical → removed
    - "4 Ms. Di Leo" (count=1) contains canonical → removed

Session 80b: Added common-word prefix/suffix detection for Person entities.
Removes terms like "Luigi Napolitano Patient" when "Luigi Napolitano" exists,
by detecting that "Patient" is a common English word (top 200K in Google dataset).

This filter runs after NER/RAKE/BM25 extraction but before final display.
It complements (does not replace) name_deduplicator.py which handles
transcript-specific artifacts like Q/A notation.
"""

from src.logging_config import debug_log
from src.core.vocabulary.rarity_filter import is_common_word
from src.core.vocabulary.person_utils import is_person_entry

# Default number of top terms to use as canonical candidates
DEFAULT_CANONICAL_COUNT = 25

# Default threshold for common word detection (top N words in Google dataset)
COMMON_WORD_THRESHOLD = 200000


def _edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.

    Copied from name_regularizer.py to avoid circular imports.
    """
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _get_trailing_words(term: str, canonical: str) -> list[str] | None:
    """
    Return trailing words if term starts with canonical, else None.

    Example:
        _get_trailing_words("Luigi Napolitano Patient", "Luigi Napolitano")
        → ["patient"]
    """
    term_words = term.lower().split()
    canonical_words = canonical.lower().split()

    if len(term_words) <= len(canonical_words):
        return None

    # Check exact prefix match
    if term_words[:len(canonical_words)] == canonical_words:
        return term_words[len(canonical_words):]

    return None


def _get_leading_words(term: str, canonical: str) -> list[str] | None:
    """
    Return leading words if term ends with canonical, else None.

    Example:
        _get_leading_words("Patient Luigi Napolitano", "Luigi Napolitano")
        → ["patient"]
    """
    term_words = term.lower().split()
    canonical_words = canonical.lower().split()

    if len(term_words) <= len(canonical_words):
        return None

    # Check exact suffix match
    if term_words[-len(canonical_words):] == canonical_words:
        return term_words[:-len(canonical_words)]

    return None


def _is_common_word_variant(term: str, canonical: str) -> bool:
    """
    Check if term is canonical + common word(s) (leading or trailing).

    Uses Google word frequency dataset to determine if extra words are
    common English words (unlikely to be part of a real name).

    Args:
        term: The longer term to check (e.g., "Luigi Napolitano Patient")
        canonical: The canonical term (e.g., "Luigi Napolitano")

    Returns:
        True if term is canonical + common word(s), False otherwise
    """
    # Check trailing: "Luigi Napolitano Patient"
    trailing = _get_trailing_words(term, canonical)
    if trailing and all(is_common_word(w, COMMON_WORD_THRESHOLD) for w in trailing):
        return True

    # Check leading: "Patient Luigi Napolitano"
    leading = _get_leading_words(term, canonical)
    if leading and all(is_common_word(w, COMMON_WORD_THRESHOLD) for w in leading):
        return True

    return False


def _is_fuzzy_common_word_variant(term: str, canonical: str, max_edit: int = 2) -> bool:
    """
    Check if term is fuzzy-canonical + common word(s).

    Handles typos like "Luigi Napontano Dob" when canonical is "Luigi Napolitano".

    Args:
        term: The longer term to check
        canonical: The canonical term
        max_edit: Maximum edit distance for fuzzy matching

    Returns:
        True if term is a fuzzy variant with common word suffix/prefix
    """
    term_words = term.lower().split()
    canonical_words = canonical.lower().split()

    if len(term_words) <= len(canonical_words):
        return False

    # Try fuzzy match on prefix (trailing common words)
    term_prefix = " ".join(term_words[:len(canonical_words)])
    canonical_str = " ".join(canonical_words)

    if _edit_distance(term_prefix, canonical_str) <= max_edit:
        trailing = term_words[len(canonical_words):]
        if all(is_common_word(w, COMMON_WORD_THRESHOLD) for w in trailing):
            return True

    # Try fuzzy match on suffix (leading common words)
    term_suffix = " ".join(term_words[-len(canonical_words):])

    if _edit_distance(term_suffix, canonical_str) <= max_edit:
        leading = term_words[:-len(canonical_words)]
        if all(is_common_word(w, COMMON_WORD_THRESHOLD) for w in leading):
            return True

    return False


def filter_substring_artifacts(
    vocabulary: list[dict],
    canonical_count: int = DEFAULT_CANONICAL_COUNT,
    count_key: str = "In-Case Freq",
    term_key: str = "Term",
) -> list[dict]:
    """
    Remove terms that contain higher-frequency canonical terms as substrings.

    Terms with the highest counts are assumed to be canonical (correct) forms.
    Lower-count terms that contain these canonical forms are likely artifacts
    from OCR errors, formatting, or extraction noise.

    Session 80b: Also removes Person terms that are canonical + common word(s).
    Example: "Luigi Napolitano Patient" removed when "Luigi Napolitano" exists.

    Args:
        vocabulary: List of term dictionaries from vocabulary extraction
        canonical_count: Number of top terms to use as canonical candidates
        count_key: Dictionary key for term frequency count
        term_key: Dictionary key for the term string

    Returns:
        Filtered vocabulary list with artifacts removed
    """
    if not vocabulary:
        return vocabulary

    # Sort by count descending to identify canonical terms
    sorted_vocab = sorted(
        vocabulary,
        key=lambda x: int(x.get(count_key, 0) or 0),
        reverse=True
    )

    # Extract canonical terms (top N by count)
    # Only use multi-word canonical terms to avoid false positives where
    # single-word terms like "john" would incorrectly filter "John Smith"
    canonical_terms = []
    canonical_person_terms = []  # Session 80b: Also track Person terms separately

    for term_dict in sorted_vocab[:canonical_count]:
        term = term_dict.get(term_key, "")
        # Only include multi-word terms (2+ words) as canonical
        # Single-word canonicals cause false positives with multi-word phrases
        if term and " " in term:
            canonical_terms.append(term.lower())
            # Track Person terms for common-word variant detection
            if is_person_entry(term_dict):
                canonical_person_terms.append(term.lower())

    # Session 80b: Also collect ALL Person terms (not just top N) for
    # common-word variant detection, since we want to catch cases where
    # "Luigi Napolitano" exists anywhere in the list
    all_person_terms = set()
    for term_dict in vocabulary:
        if is_person_entry(term_dict):
            term = term_dict.get(term_key, "")
            if term and " " in term:  # Multi-word only
                all_person_terms.add(term.lower())

    if not canonical_terms and not all_person_terms:
        return vocabulary

    debug_log(f"[ARTIFACT-FILTER] Using {len(canonical_terms)} canonical terms, "
              f"{len(all_person_terms)} person terms for common-word detection")

    # Filter out terms that contain canonical terms as substrings
    filtered = []
    removed_count = 0
    removed_common_word = 0

    for term_dict in vocabulary:
        term = term_dict.get(term_key, "")
        term_lower = term.lower()
        term_is_person = is_person_entry(term_dict)

        # Check if this term contains any canonical term as a proper substring
        # (not equal to it - we keep the canonical terms themselves)
        is_artifact = False

        for canonical in canonical_terms:
            # Skip if this IS the canonical term (exact match)
            if term_lower == canonical:
                continue

            # Check if canonical is a substring of this term
            if canonical in term_lower:
                debug_log(
                    f"[ARTIFACT-FILTER] Removing '{term}' "
                    f"(contains canonical '{canonical}')"
                )
                is_artifact = True
                removed_count += 1
                break

        # Session 80b: Check for common-word variants (Person entities only)
        if not is_artifact and term_is_person and " " in term:
            for canonical in all_person_terms:
                # Skip if this IS the canonical term
                if term_lower == canonical:
                    continue

                # Skip if term is shorter (canonical can't be prefix/suffix of shorter term)
                if len(term_lower.split()) <= len(canonical.split()):
                    continue

                # Check exact common-word variant
                if _is_common_word_variant(term, canonical):
                    debug_log(
                        f"[ARTIFACT-FILTER] Removing '{term}' "
                        f"(common-word variant of '{canonical}')"
                    )
                    is_artifact = True
                    removed_common_word += 1
                    break

                # Check fuzzy common-word variant (for typos)
                if _is_fuzzy_common_word_variant(term, canonical):
                    debug_log(
                        f"[ARTIFACT-FILTER] Removing '{term}' "
                        f"(fuzzy common-word variant of '{canonical}')"
                    )
                    is_artifact = True
                    removed_common_word += 1
                    break

        if not is_artifact:
            filtered.append(term_dict)

    debug_log(f"[ARTIFACT-FILTER] Removed {removed_count} substring artifacts, "
              f"{removed_common_word} common-word variants, "
              f"{len(filtered)} terms remaining")

    return filtered
