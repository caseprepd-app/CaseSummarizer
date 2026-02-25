"""
Unified Per-Term Filter

Optimized single-pass filter that unifies Rarity and Gibberish checks
into one iteration through the vocabulary.

This replaces multiple separate filters that each iterated the full list.

Also includes Person validity check to catch garbage terms that spaCy
NER incorrectly marks as Person (e.g., "ModMess Quanny Desortpdon").

Corpus familiarity is no longer used as a hard filter.
Instead, corpus_common_term is added as an ML feature for the model to
learn from.
"""

import logging

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult
from src.core.vocabulary.name_deduplicator import _word_validity_score
from src.core.vocabulary.person_utils import is_person_entry

logger = logging.getLogger(__name__)


class UnifiedPerTermFilter(BaseVocabularyFilter):
    """
    Combines per-term filters into a single pass.

    Instead of iterating the vocabulary multiple times, this filter runs
    all checks in one loop.

    Checks performed (in order):
    1. Rarity - Filter terms with overly common component words
    2. Gibberish - Filter nonsense/random character sequences

    Both checks exempt person names by default.

    Also adds corpus_common_term ML feature (binary: True if term appears
    in >= 64% of corpus docs AND >= 5 total occurrences).
    """

    name = "Combined Per-Term"
    priority = 40  # Replaces Rarity(40), Corpus(50), Gibberish(60)
    exempt_persons = True

    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """
        Run all per-term checks in a single pass.

        Args:
            vocabulary: Input vocabulary list

        Returns:
            FilterResult with filtered vocabulary and per-sub-filter stats
        """
        # Import filter functions
        import time

        from src.core.utils.gibberish_filter import is_gibberish
        from src.core.vocabulary.corpus_familiarity_filter import is_corpus_common_term
        from src.core.vocabulary.rarity_filter import (
            should_filter_phrase,
            should_passthrough_non_ner_term,
        )

        filtered = []
        removed_by_rarity = 0
        removed_by_gibberish = 0
        removed_terms = []

        # Track iteration for periodic GIL yield
        iteration_count = 0
        for term_data in vocabulary:
            term = term_data.get("Term", "")
            is_person = is_person_entry(term_data)

            # === CHECK 1: Rarity Filter ===
            # Person names exempt; RAKE/BM25 terms pass through if sufficiently rare
            if (
                not is_person
                and not should_passthrough_non_ner_term(term, term_data)
                and should_filter_phrase(term, is_person=False)
            ):
                removed_by_rarity += 1
                removed_terms.append(term)
                continue

            # Add corpus_common_term ML feature (simplified binary feature)
            # True if term appears in >= 64% of corpus docs AND >= 5 occurrences
            # No filtering here - let the ML model learn to deprioritize common terms
            term_data["corpus_common_term"] = is_corpus_common_term(term)

            # === CHECK 2: Gibberish Filter ===
            # Person names exempt (foreign names may look unusual)
            if not is_person and is_gibberish(term):
                removed_by_gibberish += 1
                removed_terms.append(term)
                logger.debug("Filtered gibberish: '%s'", term)
                continue

            # === CHECK 4: Person Validity Filter ===
            # Person names are exempt from gibberish filter, but we catch
            # complete garbage that spaCy NER incorrectly marks as Person.
            # If NONE of the words are in the known dictionary, it's garbage.
            # Legitimate foreign names typically have at least one recognizable word.
            if is_person and len(term) >= 10:  # Only check longer names (2+ words)
                validity = _word_validity_score(term)
                if validity == 0.0:
                    removed_by_gibberish += 1  # Count as gibberish
                    removed_terms.append(term)
                    logger.debug("Filtered Person garbage (0%% validity): '%s'", term)
                    continue

            # Term passed all checks
            filtered.append(term_data)

            # Yield GIL every 50 terms to keep GUI responsive
            iteration_count += 1
            if iteration_count % 50 == 0:
                time.sleep(0)

        total_removed = removed_by_rarity + removed_by_gibberish

        return FilterResult(
            vocabulary=filtered,
            removed_count=total_removed,
            removed_terms=removed_terms,
            metadata={
                "rarity_removed": removed_by_rarity,
                "gibberish_removed": removed_by_gibberish,
            },
        )
