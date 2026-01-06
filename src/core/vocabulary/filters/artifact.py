"""
Artifact Filter

Wraps the existing filter_substring_artifacts() function for FilterChain integration.
Removes terms that contain higher-frequency canonical terms as substrings.
"""

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult


class ArtifactFilter(BaseVocabularyFilter):
    """
    Removes substring artifacts from vocabulary.

    If a high-frequency term (e.g., "Ms. Di Leo") exists, removes longer
    terms containing it as a substring (e.g., "Ms. Di Leo:").
    """

    name = "Artifact Filter"
    priority = 20  # After name dedup
    exempt_persons = False

    def __init__(self, canonical_count: int = 25):
        self.canonical_count = canonical_count

    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """Filter substring artifacts from vocabulary."""
        from src.core.vocabulary.artifact_filter import filter_substring_artifacts

        original_count = len(vocabulary)
        filtered = filter_substring_artifacts(vocabulary, canonical_count=self.canonical_count)

        return FilterResult(
            vocabulary=filtered,
            removed_count=original_count - len(filtered),
            metadata={"canonical_count": self.canonical_count},
        )
