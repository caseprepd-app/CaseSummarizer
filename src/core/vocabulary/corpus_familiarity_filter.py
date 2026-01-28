"""
Corpus Familiarity Filtering for Vocabulary Extraction (Session 68)

Filters vocabulary terms based on how frequently they appear across the user's
corpus of past transcripts. Terms appearing in many documents are likely common
vocabulary the user already knows.

TWO THRESHOLD MODES (user-configurable):
1. Percentage mode: Filter if term appears in >= X% of documents (default 75%)
2. Count mode: Filter if term appears in >= N documents (default 10)

DUAL PURPOSE:
1. Hard filter: Terms above threshold are removed entirely (too common)
2. ML feature: Terms below threshold get corpus_familiarity_score for ML ranking

PERSON NAME EXEMPTION:
By default, person names are exempt from filtering (configurable in Settings).
Names in legal documents are always case-specific, even if common names.

WHY THIS EXISTS:
A court reporter who has processed 100 transcripts doesn't need to learn
"plaintiff", "defendant", or "counsel" every time. But a rare medical term
appearing in only 2 of 100 transcripts is worth highlighting.
"""

import logging

from src.core.vocabulary.person_utils import is_person_entry

logger = logging.getLogger(__name__)


def calculate_corpus_familiarity(term: str) -> float:
    """
    Calculate corpus familiarity score for a term.

    Familiarity = doc_freq(term) / total_docs

    Higher values mean the term appears in more of the user's past documents,
    suggesting they already know it.

    Args:
        term: The term to evaluate (case-insensitive)

    Returns:
        Float from 0.0 (never seen) to 1.0 (in every document).
        Returns 0.0 if corpus is not ready or term not found.
    """
    try:
        from src.core.vocabulary.corpus_manager import get_corpus_manager

        corpus_manager = get_corpus_manager()
        total_docs = corpus_manager.get_total_docs_indexed()

        if total_docs == 0:
            return 0.0

        doc_freq = corpus_manager.get_doc_freq(term)
        return doc_freq / total_docs

    except Exception as e:
        logger.debug("Error calculating familiarity for '%s': %s", term, e)
        return 0.0


def should_filter_corpus_familiar(term: str, is_person: bool = False) -> bool:
    """
    Determine if a term should be filtered based on corpus familiarity.

    Uses thresholds from user preferences:
    - corpus_familiarity_threshold (default 0.75 = 75%)
    - corpus_familiarity_min_docs (default 10)
    - corpus_familiarity_exempt_persons (default True)

    Filtering logic:
    1. If exempt_persons and is_person: never filter
    2. If familiarity >= threshold: filter (term is too common)
    3. If doc_freq >= min_docs: filter (alternative threshold mode)

    Args:
        term: The term to evaluate
        is_person: Whether NER detected this as a person name

    Returns:
        True if the term should be FILTERED OUT (removed)
        False if the term should be KEPT
    """
    try:
        from src.core.vocabulary.corpus_manager import get_corpus_manager
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()

        # Get user-configurable thresholds
        threshold = prefs.get("corpus_familiarity_threshold", 0.75)
        min_docs = prefs.get("corpus_familiarity_min_docs", 10)
        exempt_persons = prefs.get("corpus_familiarity_exempt_persons", True)

        # Check person exemption first
        if exempt_persons and is_person:
            return False  # Never filter exempt person names

        # Get corpus data
        corpus_manager = get_corpus_manager()
        total_docs = corpus_manager.get_total_docs_indexed()

        if total_docs == 0:
            return False  # No corpus data, can't filter

        doc_freq = corpus_manager.get_doc_freq(term)

        # Check percentage threshold
        familiarity = doc_freq / total_docs
        if familiarity >= threshold:
            logger.debug(
                "Filtering '%s' - familiarity %.1f%% >= threshold %.0f%%",
                term,
                familiarity * 100,
                threshold * 100,
            )
            return True

        # Check count threshold (alternative mode)
        if min_docs > 0 and doc_freq >= min_docs:
            logger.debug("Filtering '%s' - doc_freq %d >= min_docs %d", term, doc_freq, min_docs)
            return True

        return False  # Keep the term

    except Exception as e:
        logger.debug("Error checking filter for '%s': %s", term, e)
        return False  # On error, don't filter


def _is_person_term(term_data: dict) -> bool:
    """
    Check if a term data dict represents a person.

    Session 70: Now delegates to centralized is_person_entry() utility.

    Args:
        term_data: Term dictionary from vocabulary extraction

    Returns:
        True if the term is a person name
    """
    return is_person_entry(term_data)


def filter_corpus_familiar_terms(
    vocabulary: list[dict],
    term_key: str = "Term",
) -> list[dict]:
    """
    Filter vocabulary list and add corpus_familiarity_score to remaining terms.

    This is the main entry point for corpus familiarity filtering. Should be called
    after rarity filtering but before gibberish filtering.

    For each term:
    1. Calculate corpus_familiarity_score
    2. If above threshold (and not exempt person): remove term
    3. If below threshold: keep term and add corpus_familiarity_score to dict

    Args:
        vocabulary: List of term dictionaries from vocabulary extraction
        term_key: Dictionary key for term text (default "Term")

    Returns:
        Filtered vocabulary list with corpus_familiarity_score added to each term
    """
    if not vocabulary:
        return vocabulary

    try:
        from src.core.vocabulary.corpus_manager import get_corpus_manager

        corpus_manager = get_corpus_manager()
        total_docs = corpus_manager.get_total_docs_indexed()

        # If no corpus data, just add 0.0 scores and return
        if total_docs == 0:
            logger.debug("No corpus data available, skipping filter")
            for term_data in vocabulary:
                term_data["corpus_familiarity_score"] = 0.0
            return vocabulary

    except Exception as e:
        logger.debug("Error accessing corpus: %s", e)
        for term_data in vocabulary:
            term_data["corpus_familiarity_score"] = 0.0
        return vocabulary

    filtered_count = 0
    result = []

    for term_data in vocabulary:
        term = term_data.get(term_key, "")
        is_person = _is_person_term(term_data)

        # Calculate familiarity score
        familiarity = calculate_corpus_familiarity(term)

        # Check if should filter
        if should_filter_corpus_familiar(term, is_person):
            filtered_count += 1
            continue  # Remove term

        # Keep term and add score for ML
        term_data["corpus_familiarity_score"] = familiarity
        result.append(term_data)

    if filtered_count > 0:
        logger.debug("Filtered %d corpus-familiar terms, kept %d", filtered_count, len(result))

    return result
