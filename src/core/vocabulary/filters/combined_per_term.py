"""
Combined Per-Term Filter

Optimized single-pass filter that combines Rarity, Corpus Familiarity,
and Gibberish checks into one iteration through the vocabulary.

This replaces three separate filters that each iterated the full list.
Performance improvement: 3 passes → 1 pass.

Session 79: Added Person validity check to catch garbage terms that spaCy
NER incorrectly marks as Person (e.g., "ModMess Quanny Desortpdon").
"""

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult
from src.core.vocabulary.name_deduplicator import _word_validity_score
from src.core.vocabulary.person_utils import is_person_entry
from src.logging_config import debug_log


class CombinedPerTermFilter(BaseVocabularyFilter):
    """
    Combines per-term filters into a single pass.

    Instead of iterating the vocabulary 3 times (once for rarity, corpus
    familiarity, and gibberish), this filter runs all three checks in one loop.

    Checks performed (in order):
    1. Rarity - Filter terms with overly common component words
    2. Corpus Familiarity - Filter terms seen in too many past documents
    3. Gibberish - Filter nonsense/random character sequences

    All three checks exempt person names by default.

    Also adds corpus_familiarity_score to remaining terms for ML features.
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
        from src.core.vocabulary.corpus_familiarity_filter import (
            calculate_corpus_familiarity,
            should_filter_corpus_familiar,
        )
        from src.core.vocabulary.rarity_filter import should_filter_phrase

        filtered = []
        removed_by_rarity = 0
        removed_by_corpus = 0
        removed_by_gibberish = 0
        removed_terms = []

        # Session 80: Track iteration for periodic GIL yield
        iteration_count = 0
        for term_data in vocabulary:
            term = term_data.get("Term", "")
            is_person = is_person_entry(term_data)

            # === CHECK 1: Rarity Filter ===
            # Person names exempt
            if not is_person and should_filter_phrase(term, is_person=False):
                removed_by_rarity += 1
                removed_terms.append(term)
                continue

            # === CHECK 2: Corpus Familiarity Filter ===
            # Check if too familiar (person exemption handled inside)
            if should_filter_corpus_familiar(term, is_person):
                removed_by_corpus += 1
                removed_terms.append(term)
                continue

            # Add corpus familiarity score for ML features (even if not filtered)
            # This is what the original filter did - always add the score
            familiarity_score = calculate_corpus_familiarity(term)
            term_data["corpus_familiarity_score"] = familiarity_score

            # === CHECK 3: Gibberish Filter ===
            # Person names exempt (foreign names may look unusual)
            if not is_person and is_gibberish(term):
                removed_by_gibberish += 1
                removed_terms.append(term)
                debug_log(f"[COMBINED] Filtered gibberish: '{term}'")
                continue

            # === CHECK 4: Person Validity Filter (Session 79) ===
            # Person names are exempt from gibberish filter, but we catch
            # complete garbage that spaCy NER incorrectly marks as Person.
            # If NONE of the words are in the known dictionary, it's garbage.
            # Legitimate foreign names typically have at least one recognizable word.
            if is_person and len(term) >= 10:  # Only check longer names (2+ words)
                validity = _word_validity_score(term)
                if validity == 0.0:
                    removed_by_gibberish += 1  # Count as gibberish
                    removed_terms.append(term)
                    debug_log(f"[COMBINED] Filtered Person garbage (0% validity): '{term}'")
                    continue

            # Term passed all checks
            filtered.append(term_data)

            # Session 80: Yield GIL every 50 terms to keep GUI responsive
            iteration_count += 1
            if iteration_count % 50 == 0:
                time.sleep(0)

        total_removed = removed_by_rarity + removed_by_corpus + removed_by_gibberish

        return FilterResult(
            vocabulary=filtered,
            removed_count=total_removed,
            removed_terms=removed_terms,
            metadata={
                "rarity_removed": removed_by_rarity,
                "corpus_removed": removed_by_corpus,
                "gibberish_removed": removed_by_gibberish,
            },
        )
