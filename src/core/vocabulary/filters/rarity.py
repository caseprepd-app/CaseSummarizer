"""
Rarity Filter

Wraps the existing filter_common_phrases() function for FilterChain integration.
Removes terms whose component words are too common in English.
"""

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult


class RarityFilter(BaseVocabularyFilter):
    """
    Filters terms based on word frequency rarity.

    Uses rank-based percentile scoring:
    - Single words: Filtered if in top X% of English vocabulary
    - Multi-word phrases: Filtered if ALL words are too common

    Person names are exempt (names like "John Smith" use common words).
    """

    name = "Rarity Filter"
    priority = 40  # After list-level filters
    exempt_persons = True

    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """Filter common phrases from vocabulary."""
        from src.core.vocabulary.rarity_filter import filter_common_phrases

        original_count = len(vocabulary)
        filtered = filter_common_phrases(vocabulary)

        return FilterResult(
            vocabulary=filtered,
            removed_count=original_count - len(filtered),
        )
