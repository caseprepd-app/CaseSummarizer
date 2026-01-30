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

import logging

from src.config import (
    ARTIFACT_FILTER_COMMON_WORD_THRESHOLD,
    ARTIFACT_FILTER_FUZZY_MAX_EDIT_DISTANCE,
    TRANSCRIPT_SECTION_KEYWORDS,
)
from src.core.vocabulary.person_utils import is_person_entry
from src.core.vocabulary.rarity_filter import is_common_word
from src.core.vocabulary.string_utils import edit_distance

logger = logging.getLogger(__name__)

# Default number of top terms to use as canonical candidates
DEFAULT_CANONICAL_COUNT = 25

# Threshold for common word detection (top N words in Google dataset)
# From config with fallback
COMMON_WORD_THRESHOLD = ARTIFACT_FILTER_COMMON_WORD_THRESHOLD

# Maximum edit distance for fuzzy matching
FUZZY_MAX_EDIT_DISTANCE = ARTIFACT_FILTER_FUZZY_MAX_EDIT_DISTANCE


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
    if term_words[: len(canonical_words)] == canonical_words:
        return term_words[len(canonical_words) :]

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
    if term_words[-len(canonical_words) :] == canonical_words:
        return term_words[: -len(canonical_words)]

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
    return bool(leading and all(is_common_word(w, COMMON_WORD_THRESHOLD) for w in leading))


def _is_fuzzy_common_word_variant(
    term: str, canonical: str, max_edit: int = FUZZY_MAX_EDIT_DISTANCE
) -> bool:
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
    term_prefix = " ".join(term_words[: len(canonical_words)])
    canonical_str = " ".join(canonical_words)

    if edit_distance(term_prefix, canonical_str) <= max_edit:
        trailing = term_words[len(canonical_words) :]
        if all(is_common_word(w, COMMON_WORD_THRESHOLD) for w in trailing):
            return True

    # Try fuzzy match on suffix (leading common words)
    term_suffix = " ".join(term_words[-len(canonical_words) :])

    if edit_distance(term_suffix, canonical_str) <= max_edit:
        leading = term_words[: -len(canonical_words)]
        if all(is_common_word(w, COMMON_WORD_THRESHOLD) for w in leading):
            return True

    return False


def _remove_component_names(
    vocabulary: list[dict],
    term_key: str = "Term",
) -> list[dict]:
    """
    Remove single-word Person names that are components of multi-word Person names.

    When a full name like "Arthur Jenkins" exists in the vocabulary, we don't want
    "Arthur" or "Jenkins" appearing as separate entries since they're just parts
    of the full name, not standalone entities.

    Session 84: Only applies to Person entities, not vocabulary terms.

    Args:
        vocabulary: List of term dictionaries
        term_key: Dictionary key for the term string

    Returns:
        Filtered vocabulary list with component names removed

    Example:
        If "Arthur Jenkins" exists as Person:
        - "Arthur" (Person) → removed (is first name of Arthur Jenkins)
        - "Jenkins" (Person) → removed (is last name of Arthur Jenkins)
        - "Arthur" (Vocabulary) → kept (not a Person entity)
    """
    if not vocabulary:
        return vocabulary

    # Collect all multi-word Person names and their components
    multi_word_person_components: set[str] = set()
    for term_dict in vocabulary:
        if not is_person_entry(term_dict):
            continue
        term = term_dict.get(term_key, "")
        words = term.lower().split()
        if len(words) >= 2:
            # Add all components (first name, middle names, last name)
            for word in words:
                if word:  # Skip empty strings
                    multi_word_person_components.add(word)

    if not multi_word_person_components:
        return vocabulary

    # Filter out single-word Person terms that are components of multi-word names
    filtered = []
    removed_count = 0

    for term_dict in vocabulary:
        term = term_dict.get(term_key, "")
        term_lower = term.lower().strip()

        # Only filter single-word Person entities
        if is_person_entry(term_dict) and " " not in term:
            if term_lower in multi_word_person_components:
                logger.debug("Removing '%s' (component of multi-word Person name)", term)
                removed_count += 1
                continue

        filtered.append(term_dict)

    if removed_count > 0:
        logger.debug("Removed %d component names", removed_count)

    return filtered


def _remove_header_artifacts(
    vocabulary: list[dict],
    term_key: str = "Term",
) -> list[dict]:
    """
    Remove transcript header artifacts like "Smith - Direct" or "Jones Cross".

    Detects Person entries that are combinations of a known name component
    plus a transcript section keyword (direct, cross, redirect, etc.).

    Args:
        vocabulary: List of term dictionaries
        term_key: Dictionary key for the term string

    Returns:
        Filtered vocabulary with header artifacts removed

    Examples removed:
        - "Smith - Direct" → "Smith" is component of "John Smith", "Direct" is keyword
        - "Di Leo - Redirect" → "Di Leo" is canonical, "Redirect" is keyword
        - "Smith Plaintiff" → "Smith" is component, "Plaintiff" is common word
    """
    if not vocabulary:
        return vocabulary

    # Collect name components from multi-word Person names
    name_components: set[str] = set()
    canonical_names: set[str] = set()
    for term_dict in vocabulary:
        if not is_person_entry(term_dict):
            continue
        term = term_dict.get(term_key, "")
        words = term.lower().split()
        if len(words) >= 2:
            canonical_names.add(term.lower())
            for word in words:
                if word:
                    name_components.add(word)
            # Also add multi-word sub-sequences (for "Di Leo" style names)
            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    sub = " ".join(words[i:j])
                    if sub:
                        name_components.add(sub)

    if not name_components:
        return vocabulary

    section_keywords = TRANSCRIPT_SECTION_KEYWORDS

    filtered = []
    removed_count = 0

    for term_dict in vocabulary:
        if not is_person_entry(term_dict):
            filtered.append(term_dict)
            continue

        term = term_dict.get(term_key, "")
        # Normalize dashes: "Smith - Direct" → "Smith Direct"
        normalized = term.replace(" - ", " ").replace("-", " ")
        words_lower = normalized.lower().split()

        if len(words_lower) < 2:
            filtered.append(term_dict)
            continue

        # Skip if this is itself a canonical multi-word name
        if term.lower() in canonical_names:
            filtered.append(term_dict)
            continue

        # Try splitting into name_part + keyword_part (in both directions)
        is_header = False

        for split_pos in range(1, len(words_lower)):
            left = " ".join(words_lower[:split_pos])
            right = " ".join(words_lower[split_pos:])

            # Check: left = name component, right = section keyword(s)
            if left in name_components:
                right_words = words_lower[split_pos:]
                if all(
                    w in section_keywords or is_common_word(w, COMMON_WORD_THRESHOLD)
                    for w in right_words
                ):
                    logger.debug(
                        "Removing header artifact '%s' (name='%s', keyword='%s')",
                        term,
                        left,
                        right,
                    )
                    is_header = True
                    break

            # Check: left = section keyword(s), right = name component
            if right in name_components:
                left_words = words_lower[:split_pos]
                if all(
                    w in section_keywords or is_common_word(w, COMMON_WORD_THRESHOLD)
                    for w in left_words
                ):
                    logger.debug(
                        "Removing header artifact '%s' (keyword='%s', name='%s')",
                        term,
                        left,
                        right,
                    )
                    is_header = True
                    break

        if is_header:
            removed_count += 1
        else:
            filtered.append(term_dict)

    if removed_count > 0:
        logger.debug("Removed %d header artifacts", removed_count)

    return filtered


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
    sorted_vocab = sorted(vocabulary, key=lambda x: int(x.get(count_key) or 0), reverse=True)

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

    logger.debug(
        "Using %d canonical terms, %d person terms for common-word detection",
        len(canonical_terms),
        len(all_person_terms),
    )

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
                logger.debug("Removing '%s' (contains canonical '%s')", term, canonical)
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
                    logger.debug("Removing '%s' (common-word variant of '%s')", term, canonical)
                    is_artifact = True
                    removed_common_word += 1
                    break

                # Check fuzzy common-word variant (for typos)
                if _is_fuzzy_common_word_variant(term, canonical):
                    logger.debug(
                        "Removing '%s' (fuzzy common-word variant of '%s')", term, canonical
                    )
                    is_artifact = True
                    removed_common_word += 1
                    break

        if not is_artifact:
            filtered.append(term_dict)

    logger.debug(
        "Removed %d substring artifacts, %d common-word variants, %d terms remaining",
        removed_count,
        removed_common_word,
        len(filtered),
    )

    # Session 84: Remove single-word Person names that are components of full names
    filtered = _remove_component_names(filtered, term_key=term_key)

    # Session 140: Remove transcript header artifacts like "Smith - Direct"
    filtered = _remove_header_artifacts(filtered, term_key=term_key)

    return filtered
