"""
Name Deduplication Filter

Wraps the existing deduplicate_names() function for FilterChain integration.
Merges similar Person names (OCR variants, transcript artifacts).
"""

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult


class NameDeduplicationFilter(BaseVocabularyFilter):
    """
    Merges similar Person names based on artifact removal and fuzzy matching.

    Handles:
    - Transcript artifacts: "DI LEO 1 Q", "Di Leo: Objection" -> "Di Leo"
    - OCR variants: "Arthur Jenkins", "Anhur Jenkins" -> merged
    """

    name = "Name Deduplication"
    priority = 10  # Must run first
    exempt_persons = False  # Operates on persons only

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """Deduplicate person names in vocabulary."""
        from src.core.vocabulary.name_deduplicator import deduplicate_names

        original_count = len(vocabulary)
        filtered = deduplicate_names(vocabulary, self.similarity_threshold)

        return FilterResult(
            vocabulary=filtered,
            removed_count=original_count - len(filtered),
            metadata={"similarity_threshold": self.similarity_threshold},
        )
