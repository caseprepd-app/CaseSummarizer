"""
Corpus Familiarity Filter

Wraps the existing filter_corpus_familiar_terms() function for FilterChain integration.
Removes terms that appear too frequently in the user's past documents.
"""

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult


class CorpusFamiliarityFilter(BaseVocabularyFilter):
    """
    Filters terms based on corpus familiarity.

    Terms appearing in >= X% of past documents are considered "known"
    and filtered out. Also adds corpus_familiarity_score to remaining terms.

    Person names are exempt by default (configurable).
    """

    name = "Corpus Familiarity"
    priority = 50  # After rarity filter
    exempt_persons = True

    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """Filter corpus-familiar terms and add familiarity scores."""
        from src.core.vocabulary.corpus_familiarity_filter import filter_corpus_familiar_terms

        original_count = len(vocabulary)
        filtered = filter_corpus_familiar_terms(vocabulary)

        return FilterResult(
            vocabulary=filtered,
            removed_count=original_count - len(filtered),
        )
