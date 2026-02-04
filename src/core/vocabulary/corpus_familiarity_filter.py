"""
Corpus Familiarity Filtering for Vocabulary Extraction (Session 68)

Provides a binary ML feature indicating if a term is common in the user's corpus.

SIMPLIFIED DESIGN (Session 147):
- Single binary feature: corpus_common_term (True/False)
- True if term appears in >= 64% of corpus docs AND >= 5 total occurrences
- False otherwise (including when no corpus is available)
- Maximum 25 documents allowed in corpus

The ML model learns to deprioritize terms where corpus_common_term=True,
as these are likely domain vocabulary the user already knows.

WHY THIS EXISTS:
A court reporter who has processed 25 transcripts doesn't need to learn
"plaintiff", "defendant", or "counsel" every time. But a rare medical term
appearing in only 2 of 25 transcripts is worth highlighting.
"""

import logging

logger = logging.getLogger(__name__)


def is_corpus_common_term(term: str) -> bool:
    """
    Check if a term is common in the user's corpus.

    A term is "common" if it appears in >= 64% of corpus documents
    AND has >= 5 total document occurrences.

    This is a binary ML feature - the model learns to deprioritize
    terms where this returns True.

    Args:
        term: The term to evaluate (case-insensitive)

    Returns:
        True if the term is common in the corpus.
        False if not common, corpus disabled, or no corpus available.
    """
    try:
        from src.core.vocabulary.corpus_manager import get_corpus_manager

        corpus_manager = get_corpus_manager()
        return corpus_manager.is_corpus_common_term(term)

    except Exception as e:
        logger.debug("Error checking corpus common term for '%s': %s", term, e)
        return False


def add_corpus_common_feature(
    vocabulary: list[dict],
    term_key: str = "Term",
) -> list[dict]:
    """
    Add corpus_common_term feature to vocabulary terms.

    This is the main entry point for corpus feature extraction. Should be called
    during vocabulary processing to add the ML feature.

    For each term:
    - Sets corpus_common_term = True if term appears in >= 64% of corpus docs
      AND has >= 5 total occurrences
    - Sets corpus_common_term = False otherwise

    Note: This does NOT filter terms. The ML model learns to use this feature
    to deprioritize common domain vocabulary.

    Args:
        vocabulary: List of term dictionaries from vocabulary extraction
        term_key: Dictionary key for term text (default "Term")

    Returns:
        Same vocabulary list with corpus_common_term added to each term
    """
    if not vocabulary:
        return vocabulary

    common_count = 0

    for term_data in vocabulary:
        term = term_data.get(term_key, "")
        is_common = is_corpus_common_term(term)
        term_data["corpus_common_term"] = is_common

        if is_common:
            common_count += 1

    if common_count > 0:
        logger.debug(
            "Marked %d/%d terms as corpus-common",
            common_count,
            len(vocabulary),
        )

    return vocabulary


# Backwards compatibility alias
def filter_corpus_familiar_terms(
    vocabulary: list[dict],
    term_key: str = "Term",
) -> list[dict]:
    """
    Backwards compatibility wrapper for add_corpus_common_feature.

    Previously this function filtered terms AND added a float score.
    Now it just adds the binary corpus_common_term feature without filtering.

    Args:
        vocabulary: List of term dictionaries from vocabulary extraction
        term_key: Dictionary key for term text (default "Term")

    Returns:
        Same vocabulary list with corpus_common_term added to each term
    """
    return add_corpus_common_feature(vocabulary, term_key)
