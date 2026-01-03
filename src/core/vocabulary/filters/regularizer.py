"""
Name Regularizer Filter

Wraps the existing regularize_names() function for FilterChain integration.
Removes name fragments and 1-char typo variants.
"""

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult


class NameRegularizerFilter(BaseVocabularyFilter):
    """
    Removes name fragments and typo variants.

    - Fragments: "Di" removed if "Di Leo" exists in top quartile
    - Typos: "Barbr Jenkins" removed if "Barbra Jenkins" exists (edit distance 1)
    """

    name = "Name Regularizer"
    priority = 30  # After artifact filter
    exempt_persons = False

    def __init__(self, top_fraction: float = 0.25, num_passes: int = 3):
        self.top_fraction = top_fraction
        self.num_passes = num_passes

    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """Filter name fragments and typos."""
        from src.core.vocabulary.name_regularizer import regularize_names

        original_count = len(vocabulary)
        filtered = regularize_names(
            vocabulary,
            top_fraction=self.top_fraction,
            num_passes=self.num_passes
        )

        return FilterResult(
            vocabulary=filtered,
            removed_count=original_count - len(filtered),
            metadata={
                'top_fraction': self.top_fraction,
                'num_passes': self.num_passes
            }
        )
