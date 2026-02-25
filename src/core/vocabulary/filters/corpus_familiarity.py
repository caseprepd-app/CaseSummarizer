"""
Corpus Familiarity Feature Adder

Wraps the add_corpus_common_feature() function for FilterChain integration.
Adds the corpus_common_term ML feature to terms.

This no longer filters terms - it just adds the binary feature
for the ML model to learn from.
"""

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult


class CorpusFamiliarityFilter(BaseVocabularyFilter):
    """
    Adds corpus_common_term feature to vocabulary terms.

    No longer filters terms. Instead, adds a binary ML feature:
    - corpus_common_term = True if term in >= 64% of corpus docs AND >= 5 occurrences
    - corpus_common_term = False otherwise

    The ML model learns to deprioritize terms where this is True.
    """

    name = "Corpus Familiarity"
    priority = 50  # After rarity filter
    exempt_persons = True

    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """Add corpus_common_term feature to all terms (no filtering)."""
        from src.core.vocabulary.corpus_familiarity_filter import add_corpus_common_feature

        # This no longer filters - just adds the feature
        result = add_corpus_common_feature(vocabulary)

        return FilterResult(
            vocabulary=result,
            removed_count=0,  # No terms are removed
        )
