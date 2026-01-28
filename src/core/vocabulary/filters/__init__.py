"""
Vocabulary Filter Chain Module

Provides a unified filtering pipeline for vocabulary extraction.
Filters are executed in priority order with statistics tracking.

Usage:
    from src.core.vocabulary.filters import create_default_filter_chain

    chain = create_default_filter_chain()
    result = chain.run(vocabulary)
    cleaned_vocabulary = result.vocabulary
    print(f"Removed {result.removed_count} terms")

    # Access per-filter stats
    for name, stats in result.metadata['per_filter'].items():
        print(f"  {name}: {stats['removed_count']} removed")
"""

from src.core.vocabulary.filters.artifact import ExtractionArtifactFilter
from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult
from src.core.vocabulary.filters.combined_per_term import UnifiedPerTermFilter
from src.core.vocabulary.filters.corpus_familiarity import CorpusFamiliarityFilter
from src.core.vocabulary.filters.filter_chain import FilterChainStats, VocabularyFilterChain
from src.core.vocabulary.filters.gibberish import GibberishFilter
from src.core.vocabulary.filters.name_dedup import NameDeduplicationFilter
from src.core.vocabulary.filters.rarity import RarityFilter
from src.core.vocabulary.filters.regex_exclusion import RegexExclusionFilter
from src.core.vocabulary.filters.regularizer import NameRegularizerFilter


def create_default_filter_chain() -> VocabularyFilterChain:
    """
    Create a filter chain with all standard vocabulary filters.

    Order (by priority):
    1. NameDeduplicationFilter (10) - Merge similar person names
    2. RegexExclusionFilter (15) - Remove user-defined pattern matches
    3. ExtractionArtifactFilter (20) - Remove substring artifacts
    4. NameRegularizerFilter (30) - Remove fragments and typos
    5. RarityFilter (40) - Remove common-word terms
    6. CorpusFamiliarityFilter (50) - Remove corpus-familiar terms
    7. GibberishFilter (60) - Remove nonsense strings

    Returns:
        Configured VocabularyFilterChain instance
    """
    return VocabularyFilterChain(
        [
            NameDeduplicationFilter(),
            RegexExclusionFilter(),
            ExtractionArtifactFilter(),
            NameRegularizerFilter(),
            RarityFilter(),
            CorpusFamiliarityFilter(),
            GibberishFilter(),
        ]
    )


def create_optimized_filter_chain() -> VocabularyFilterChain:
    """
    Create a filter chain with combined per-term filters for better performance.

    Uses UnifiedPerTermFilter to run Rarity, Corpus Familiarity, and Gibberish
    checks in a single pass instead of three separate passes.

    Order (by priority):
    1. NameDeduplicationFilter (10) - Merge similar person names
    2. RegexExclusionFilter (15) - Remove user-defined pattern matches
    3. ExtractionArtifactFilter (20) - Remove substring artifacts
    4. NameRegularizerFilter (30) - Remove fragments and typos
    5. UnifiedPerTermFilter (40) - Rarity + Corpus + Gibberish in one pass

    Returns:
        Configured VocabularyFilterChain instance (optimized)
    """
    return VocabularyFilterChain(
        [
            NameDeduplicationFilter(),
            RegexExclusionFilter(),
            ExtractionArtifactFilter(),
            NameRegularizerFilter(),
            UnifiedPerTermFilter(),
        ]
    )


def create_partial_results_filter_chain() -> VocabularyFilterChain:
    """
    Create a filter chain for partial results (BM25+RAKE).

    For progressive extraction, BM25 and RAKE run first.
    RarityFilter included because common phrases like "once per year"
    were appearing in partial results without it.

    Order (by priority):
    1. RegexExclusionFilter (15) - Remove user-defined pattern matches
    2. ExtractionArtifactFilter (20) - Remove substring artifacts
    3. RarityFilter (40) - Remove common-word terms
    4. GibberishFilter (60) - Remove nonsense strings

    Returns:
        Configured VocabularyFilterChain instance
    """
    return VocabularyFilterChain(
        [
            RegexExclusionFilter(),
            ExtractionArtifactFilter(),
            RarityFilter(),
            GibberishFilter(),
        ]
    )


__all__ = [
    "ExtractionArtifactFilter",
    # Base classes
    "BaseVocabularyFilter",
    "UnifiedPerTermFilter",
    "CorpusFamiliarityFilter",
    "FilterChainStats",
    "FilterResult",
    "GibberishFilter",
    # Filters
    "NameDeduplicationFilter",
    "NameRegularizerFilter",
    "RarityFilter",
    "RegexExclusionFilter",
    "VocabularyFilterChain",
    # Factory functions
    "create_default_filter_chain",
    "create_optimized_filter_chain",
    "create_partial_results_filter_chain",
]
