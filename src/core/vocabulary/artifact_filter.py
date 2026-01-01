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

This filter runs after NER/RAKE/BM25 extraction but before final display.
It complements (does not replace) name_deduplicator.py which handles
transcript-specific artifacts like Q/A notation.
"""

from src.logging_config import debug_log

# Default number of top terms to use as canonical candidates
DEFAULT_CANONICAL_COUNT = 25


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
    for term_dict in sorted_vocab[:canonical_count]:
        term = term_dict.get(term_key, "")
        # Only include multi-word terms (2+ words) as canonical
        # Single-word canonicals cause false positives with multi-word phrases
        if term and " " in term:
            canonical_terms.append(term.lower())

    if not canonical_terms:
        return vocabulary

    debug_log(f"[ARTIFACT-FILTER] Using {len(canonical_terms)} canonical terms")

    # Filter out terms that contain canonical terms as substrings
    filtered = []
    removed_count = 0

    for term_dict in vocabulary:
        term = term_dict.get(term_key, "")
        term_lower = term.lower()

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

        if not is_artifact:
            filtered.append(term_dict)

    debug_log(f"[ARTIFACT-FILTER] Removed {removed_count} artifacts, "
              f"{len(filtered)} terms remaining")

    return filtered
