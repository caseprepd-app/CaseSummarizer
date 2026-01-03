"""
Gibberish Filter

Wraps the existing is_gibberish() function for FilterChain integration.
Removes nonsense/random character sequences.
"""

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult
from src.core.vocabulary.person_utils import is_person_entry


class GibberishFilter(BaseVocabularyFilter):
    """
    Filters gibberish terms using spell-checking.

    Uses SpellChecker to detect nonsense strings with no valid
    dictionary matches or corrections.

    Person names are exempt (foreign names may look unusual).
    """

    name = "Gibberish Filter"
    priority = 60  # Last filter
    exempt_persons = True

    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """Filter gibberish terms from vocabulary."""
        from src.core.utils.gibberish_filter import is_gibberish

        filtered = []
        removed_count = 0
        removed_terms = []

        for term_data in vocabulary:
            term = term_data.get("Term", "")

            # Person names exempt
            if is_person_entry(term_data):
                filtered.append(term_data)
                continue

            # Check if gibberish
            if is_gibberish(term):
                removed_count += 1
                removed_terms.append(term)
                continue

            filtered.append(term_data)

        return FilterResult(
            vocabulary=filtered,
            removed_count=removed_count,
            removed_terms=removed_terms,
        )
