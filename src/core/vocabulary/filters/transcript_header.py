"""
Transcript Header Filter

Removes transcript header/footer artifacts from vocabulary.
Detects terms that contain transcript section keywords (direct, cross, etc.)
and terms that combine canonical person names with section keywords or
concatenate two canonical names.

Priority 25: runs after ExtractionArtifactFilter (20), before NameRegularizerFilter (30).
"""

import logging
import re

from src.config import TRANSCRIPT_SECTION_KEYWORDS
from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult
from src.core.vocabulary.person_utils import is_person_entry

logger = logging.getLogger(__name__)


def _build_canonical_names(vocabulary: list[dict]) -> set[str]:
    """
    Extract canonical person names from vocabulary.

    Returns:
        Set of lowercase canonical person names (multi-word only).
    """
    names = set()
    for term_data in vocabulary:
        if is_person_entry(term_data):
            term = term_data.get("Term", "")
            if len(term.split()) >= 2:
                names.add(term.lower())
    return names


def _contains_section_keyword(term_lower: str) -> bool:
    """
    Check if term contains a transcript section keyword.

    Args:
        term_lower: Lowercase term text.

    Returns:
        True if any section keyword is found as a whole word.
    """
    words = set(term_lower.split())
    return bool(words & TRANSCRIPT_SECTION_KEYWORDS)


def _has_name_plus_keyword(term_lower: str, canonical_names: set[str]) -> bool:
    """
    Check if term is a canonical name combined with a section keyword.

    Matches patterns like "Jones - Direct", "Smith Cross Examination".
    The name and keyword may be separated by hyphens, dashes, or spaces.

    Args:
        term_lower: Lowercase term text.
        canonical_names: Set of lowercase canonical person names.

    Returns:
        True if term contains both a canonical name and a section keyword.
    """
    # Strip separators to normalize "Jones - Direct" → "Jones Direct"
    normalized = re.sub(r"[\-–—]+", " ", term_lower)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    norm_words = set(normalized.split())

    has_keyword = bool(norm_words & TRANSCRIPT_SECTION_KEYWORDS)
    if not has_keyword:
        return False

    # Check if any canonical name's words are present
    for name in canonical_names:
        name_words = set(name.split())
        if name_words.issubset(norm_words):
            return True
    return False


def _has_two_canonical_names(term_lower: str, canonical_names: set[str]) -> bool:
    """
    Check if term concatenates two distinct canonical person names.

    Catches header artifacts like "Jones Smith" where two party names
    got extracted together.

    Args:
        term_lower: Lowercase term text.
        canonical_names: Set of lowercase canonical person names.

    Returns:
        True if term contains words from two different canonical names.
    """
    normalized = re.sub(r"[\-–—]+", " ", term_lower)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    norm_words = set(normalized.split())

    matched_names = []
    for name in canonical_names:
        name_words = set(name.split())
        if name_words.issubset(norm_words):
            matched_names.append(name)

    if len(matched_names) < 2:
        return False

    # Verify they are distinct names (not overlapping)
    for i in range(len(matched_names)):
        for j in range(i + 1, len(matched_names)):
            words_i = set(matched_names[i].split())
            words_j = set(matched_names[j].split())
            if not words_i & words_j:
                return True
    return False


def is_transcript_header_artifact(term: str, canonical_names: set[str]) -> bool:
    """
    Determine if a term is a transcript header/footer artifact.

    Args:
        term: The term text.
        canonical_names: Set of lowercase canonical person names.

    Returns:
        True if term should be removed as a header artifact.
    """
    term_lower = term.lower()

    # Rule 1: Contains section keyword (strong signal)
    if _contains_section_keyword(term_lower):
        return True

    # Rule 2: Canonical name + section keyword with separators
    if canonical_names and _has_name_plus_keyword(term_lower, canonical_names):
        return True

    # Rule 3: Two canonical names concatenated
    if canonical_names and _has_two_canonical_names(term_lower, canonical_names):
        return True

    return False


class TranscriptHeaderFilter(BaseVocabularyFilter):
    """
    Removes transcript header/footer artifacts from vocabulary.

    Detects three patterns:
    1. Terms containing section keywords (direct, cross, redirect, recross)
    2. Canonical name + section keyword ("Jones - Direct")
    3. Two canonical names concatenated ("Jones Smith")
    """

    name = "Transcript Header Filter"
    priority = 25
    exempt_persons = False

    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """Filter transcript header artifacts from vocabulary."""
        canonical_names = _build_canonical_names(vocabulary)

        kept = []
        removed = []
        for term_data in vocabulary:
            term = term_data.get("Term", "")
            term_lower = term.lower()

            # Rule 1 check: keyword-only removal is skipped for person entries
            # (a person named "Cross" or "Dirk Cross" should not be removed)
            if _contains_section_keyword(term_lower) and not is_person_entry(term_data):
                removed.append(term)
                logger.debug("[Transcript Header Filter] Removed: '%s'", term)
                continue

            # Rules 2 & 3: name+keyword and two-name concatenation (apply to all)
            if canonical_names and (
                _has_name_plus_keyword(term_lower, canonical_names)
                or _has_two_canonical_names(term_lower, canonical_names)
            ):
                removed.append(term)
                logger.debug("[Transcript Header Filter] Removed: '%s'", term)
            else:
                kept.append(term_data)

        if removed:
            logger.info(
                "[Transcript Header Filter] Removed %d header artifacts",
                len(removed),
            )

        return FilterResult(
            vocabulary=kept,
            removed_count=len(removed),
            removed_terms=removed,
        )
