"""
Name Deduplication Filter

Wraps the existing deduplicate_names() function for FilterChain integration.
Merges similar Person names (OCR variants, transcript artifacts).
"""

from src.config import NAME_SIMILARITY_THRESHOLD
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

    def __init__(self, similarity_threshold: float | None = None):
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else NAME_SIMILARITY_THRESHOLD
        )

    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """Deduplicate person names in vocabulary."""
        from src.core.vocabulary.name_deduplicator import (
            deduplicate_names,
            find_potential_duplicates,
        )

        original_count = len(vocabulary)
        filtered = deduplicate_names(vocabulary, self.similarity_threshold)

        # Find potential duplicates (names where one is subset of another)
        # These are flagged for user review, not auto-merged
        potential_dupes = find_potential_duplicates(filtered)

        # Attach metadata to flagged terms
        for term in filtered:
            term_name = term.get("Term", "")
            if term_name in potential_dupes:
                term["_potential_duplicate_of"] = potential_dupes[term_name]

        return FilterResult(
            vocabulary=filtered,
            removed_count=original_count - len(filtered),
            metadata={
                "similarity_threshold": self.similarity_threshold,
                "potential_duplicates": len(potential_dupes),
            },
        )
