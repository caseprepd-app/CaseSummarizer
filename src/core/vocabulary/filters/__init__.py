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

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult
from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain, FilterChainStats
from src.core.vocabulary.filters.name_dedup import NameDeduplicationFilter
from src.core.vocabulary.filters.artifact import ArtifactFilter
from src.core.vocabulary.filters.regularizer import NameRegularizerFilter
from src.core.vocabulary.filters.rarity import RarityFilter
from src.core.vocabulary.filters.corpus_familiarity import CorpusFamiliarityFilter
from src.core.vocabulary.filters.gibberish import GibberishFilter
from src.core.vocabulary.filters.combined_per_term import CombinedPerTermFilter


def create_default_filter_chain() -> VocabularyFilterChain:
    """
    Create a filter chain with all standard vocabulary filters.

    Order (by priority):
    1. NameDeduplicationFilter (10) - Merge similar person names
    2. ArtifactFilter (20) - Remove substring artifacts
    3. NameRegularizerFilter (30) - Remove fragments and typos
    4. RarityFilter (40) - Remove common-word terms
    5. CorpusFamiliarityFilter (50) - Remove corpus-familiar terms
    6. GibberishFilter (60) - Remove nonsense strings

    Returns:
        Configured VocabularyFilterChain instance
    """
    return VocabularyFilterChain([
        NameDeduplicationFilter(),
        ArtifactFilter(),
        NameRegularizerFilter(),
        RarityFilter(),
        CorpusFamiliarityFilter(),
        GibberishFilter(),
    ])


def create_optimized_filter_chain() -> VocabularyFilterChain:
    """
    Create a filter chain with combined per-term filters for better performance.

    Uses CombinedPerTermFilter to run Rarity, Corpus Familiarity, and Gibberish
    checks in a single pass instead of three separate passes.

    Order (by priority):
    1. NameDeduplicationFilter (10) - Merge similar person names
    2. ArtifactFilter (20) - Remove substring artifacts
    3. NameRegularizerFilter (30) - Remove fragments and typos
    4. CombinedPerTermFilter (40) - Rarity + Corpus + Gibberish in one pass

    Returns:
        Configured VocabularyFilterChain instance (optimized)
    """
    return VocabularyFilterChain([
        NameDeduplicationFilter(),
        ArtifactFilter(),
        NameRegularizerFilter(),
        CombinedPerTermFilter(),
    ])


__all__ = [
    # Base classes
    'BaseVocabularyFilter',
    'FilterResult',
    'VocabularyFilterChain',
    'FilterChainStats',
    # Filters
    'NameDeduplicationFilter',
    'ArtifactFilter',
    'NameRegularizerFilter',
    'RarityFilter',
    'CorpusFamiliarityFilter',
    'GibberishFilter',
    'CombinedPerTermFilter',
    # Factory functions
    'create_default_filter_chain',
    'create_optimized_filter_chain',
]
