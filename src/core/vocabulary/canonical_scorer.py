"""
Canonical Spelling Scorer

Selects the correct spelling from a group of similar variants (1-2 edit distance)
using a branching strategy based on dictionary/name dataset presence.

Strategy:
1. If exactly ONE variant is fully in dictionary → it wins (100% confidence)
2. If ZERO variants are known → weighted score decides (exotic name scenario)
3. If MULTIPLE variants are known → weighted score as tiebreaker (Smith vs Smyth)

The weighted score combines:
- Document confidence (OCR quality)
- Occurrence frequency (with ^1.1 exponent for spread)
- OCR artifact penalty (10% reduction if artifacts detected)

NOTE: An ML model trained on user feedback could potentially outperform this
rules-based approach by learning patterns specific to the user's document types
(e.g., medical terminology, regional name spellings). This implementation provides
a reasonable baseline that works without training data. Future enhancement could
replace this with a learned model once sufficient feedback is collected.

"""

import logging

from src.core.utils.ocr_patterns import has_ocr_artifacts
from src.core.vocabulary.term_sources import TermSources

logger = logging.getLogger(__name__)


class CanonicalScorer:
    """
    Score-based canonical spelling selector.

    Uses dictionary/name presence to branch between "known name" and "exotic name"
    scenarios, with confidence-weighted scoring for the latter.

    Attributes:
        known_words: Set of known words (Google freq list + international names)
        ocr_penalty: Penalty applied to terms with OCR artifacts (default 10%)

    Example:
        scorer = CanonicalScorer(known_words)

        # Compare similar variants
        variants = [
            {"Term": "Jenkins", "sources": sources1},
            {"Term": "Jenidns", "sources": sources2},
        ]
        canonical = scorer.select_canonical(variants)
        # Returns the Jenkins entry (known word wins)
    """

    def __init__(
        self,
        known_words: set[str],
        ocr_penalty: float = 0.10,
    ):
        """
        Initialize the canonical scorer.

        Args:
            known_words: Set of lowercase known words/names for dictionary lookup
            ocr_penalty: Penalty factor for OCR artifacts (0.0-1.0, default 0.10)
        """
        self.known_words = known_words
        self.ocr_penalty = ocr_penalty

    def is_fully_known(self, term: str) -> bool:
        """
        Check if ALL words in a term are in the known words dictionary.

        A term like "John Smith" is fully known if both "john" and "smith"
        are in the dictionary. "Djamel Boualem" would not be fully known
        if neither name appears in the dictionary.

        Args:
            term: A term (possibly multi-word) to check

        Returns:
            True if every word in the term is known, False otherwise
        """
        if not term:
            return False

        words = term.lower().split()
        if not words:
            return False

        # Strip common punctuation from each word
        return all(word.strip(".,;:'\"()-") in self.known_words for word in words)

    def calculate_score(self, term: str, sources: TermSources, original_casing: str = "") -> float:
        """
        Calculate the canonical score for a term.

        Formula:
            base_score = sources.weighted_score  # (conf * count)^1.1
            ocr_factor = 1.0 - ocr_penalty if has_ocr_artifacts else 1.0
            casing_bonus = 1.3 if title case, 1.0 if ALL CAPS
            final_score = base_score * ocr_factor * casing_bonus

        Args:
            term: The term string
            sources: TermSources tracking document origins
            original_casing: Original term before normalization, for casing preference

        Returns:
            Float score (higher = more likely to be canonical)
        """
        if sources is None:
            # Fallback for terms without source tracking
            logger.debug("No sources for '%s', using default score", term)
            return 0.0

        base_score = sources.weighted_score

        # Apply OCR artifact penalty if detected
        if has_ocr_artifacts(term):
            penalty = self.ocr_penalty
            base_score *= 1.0 - penalty
            logger.debug(
                "OCR penalty applied to '%s': -%.0f%% -> score=%.2f",
                term,
                penalty * 100,
                base_score,
            )

        # Apply title-case bonus if original casing is available
        # Transcripts often have "BY MR. JONES:" inflating ALL CAPS frequency
        if original_casing:
            words = original_casing.split()
            multi_char = [w for w in words if len(w) > 1]
            is_title = multi_char and all(w[0].isupper() and w[1:].islower() for w in multi_char)
            if is_title:
                base_score *= 1.3
            elif original_casing == original_casing.upper() and len(original_casing) > 1:
                base_score *= 0.8  # Penalize ALL CAPS

        return base_score

    def select_canonical(
        self,
        variants: list[dict],
        term_key: str = "Term",
        sources_key: str = "sources",
    ) -> dict:
        """
        Select the canonical spelling from a group of similar variants.

        Uses a branching strategy:
        1. If exactly ONE variant is fully known → it wins
        2. If ZERO variants are known → highest weighted score wins
        3. If MULTIPLE variants are known → highest weighted score as tiebreaker

        Args:
            variants: List of term dictionaries with at least 'Term' and 'sources'
            term_key: Key for the term string in each dict
            sources_key: Key for the TermSources in each dict

        Returns:
            The selected canonical variant dict (with merged sources)
        """
        if not variants:
            raise ValueError("Cannot select canonical from empty list")

        if len(variants) == 1:
            return variants[0]

        # Classify variants by known status
        known_variants = []
        unknown_variants = []

        for v in variants:
            term = v.get(term_key, "")
            if self.is_fully_known(term):
                known_variants.append(v)
            else:
                unknown_variants.append(v)

        logger.debug(
            "%d variants: %d known, %d unknown",
            len(variants),
            len(known_variants),
            len(unknown_variants),
        )

        # Branch based on how many are known
        if len(known_variants) == 1:
            # Exactly one known → it wins decisively
            canonical = known_variants[0]
            logger.debug("'%s' wins (only known variant)", canonical.get(term_key))
            result = self._merge_into_canonical(canonical, variants, term_key, sources_key)
            result["_selection_branch"] = "single_known"
            return result

        elif len(known_variants) == 0:
            # None known → exotic name scenario, use weighted scoring
            logger.debug("No known variants, using weighted scores")
            result = self._select_by_score(variants, term_key, sources_key)
            result["_selection_branch"] = "none_known"
            return result

        else:
            # Multiple known → use weighted score as tiebreaker
            logger.debug(
                "Multiple known variants (%s), using weighted scores",
                [v.get(term_key) for v in known_variants],
            )
            result = self._select_by_score(known_variants, term_key, sources_key)
            result["_selection_branch"] = "multiple_known"

            # Merge unknown variants' frequencies into the canonical
            unknown_variants = [v for v in variants if v not in known_variants]
            if unknown_variants:
                extra_freq = sum(
                    v.get("Occurrences", 0) or v.get("occurrences", 0) or 0
                    for v in unknown_variants
                )
                result["Occurrences"] = result.get("Occurrences", 0) + extra_freq
                if "occurrences" in result:
                    result["occurrences"] = result.get("occurrences", 0) + extra_freq

            return result

    def _select_by_score(
        self,
        variants: list[dict],
        term_key: str,
        sources_key: str,
    ) -> dict:
        """
        Select canonical by weighted score (for exotic names or tiebreakers).

        Args:
            variants: List of variant dicts to compare
            term_key: Key for term string
            sources_key: Key for TermSources

        Returns:
            Highest-scoring variant with merged sources
        """
        scored = []
        for v in variants:
            term = v.get(term_key, "")
            sources = v.get(sources_key)

            # Handle missing sources gracefully
            if sources is None:
                sources = TermSources.create_legacy(
                    v.get("Occurrences", 1),
                    v.get("source_doc_confidence", 100) / 100.0,  # 0-100 scale → 0-1
                )

            original_casing = v.get("_original_casing", "")
            score = self.calculate_score(term, sources, original_casing)
            scored.append((v, score))

            logger.debug("'%s' score=%.2f", term, score)

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Winner is the highest score
        canonical, best_score = scored[0]
        logger.debug("'%s' wins with score=%.2f", canonical.get(term_key), best_score)

        # Return with merged sources from all variants
        result = self._merge_into_canonical(
            canonical,
            [v for v, _ in scored],
            term_key,
            sources_key,
        )
        result["_scored_variants"] = [(v.get(term_key, ""), s) for v, s in scored]
        return result

    def _merge_into_canonical(
        self,
        canonical: dict,
        all_variants: list[dict],
        term_key: str,
        sources_key: str,
    ) -> dict:
        """
        Merge frequency and sources from all variants into the canonical.

        The canonical keeps its term spelling but accumulates the frequency
        counts and source tracking from all variants it subsumes.

        Args:
            canonical: The winning variant dict
            all_variants: All variants being merged (including canonical)
            term_key: Key for term string
            sources_key: Key for TermSources

        Returns:
            New dict with merged data
        """
        result = canonical.copy()

        # Merge frequencies
        total_freq = sum(
            v.get("Occurrences", 0) or v.get("occurrences", 0) or 0 for v in all_variants
        )
        result["Occurrences"] = total_freq
        if "occurrences" in result:
            result["occurrences"] = total_freq

        # Merge TermSources
        merged_sources = TermSources()
        for v in all_variants:
            sources = v.get(sources_key)
            if sources:
                merged_sources = merged_sources.merge_with(sources)

        result[sources_key] = merged_sources

        # Log what was merged
        if len(all_variants) > 1:
            others = [
                v.get(term_key) for v in all_variants if v.get(term_key) != canonical.get(term_key)
            ]
            if others:
                logger.debug(
                    "Merged %d variants into '%s': %s%s",
                    len(others),
                    canonical.get(term_key),
                    others[:5],
                    "..." if len(others) > 5 else "",
                )

        return result


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def create_canonical_scorer(known_words: set[str] | None = None) -> CanonicalScorer:
    """
    Create a CanonicalScorer with the standard known words dataset.

    If known_words is not provided, loads from the default sources:
    - Google word frequency list (top 50k)
    - International names dataset

    Args:
        known_words: Optional pre-loaded known words set

    Returns:
        Configured CanonicalScorer instance
    """
    if known_words is None:
        # Load from the same source as name_regularizer
        from src.core.vocabulary.name_regularizer import _load_known_words

        known_words = _load_known_words()

    return CanonicalScorer(known_words)


def select_canonical_spelling(
    variants: list[dict],
    known_words: set[str] | None = None,
) -> dict:
    """
    Convenience function to select canonical spelling from variants.

    Creates a scorer and selects the canonical in one call.

    Args:
        variants: List of similar term variants
        known_words: Optional known words set (loads default if None)

    Returns:
        The canonical variant dict

    Example:
        canonical = select_canonical_spelling([
            {"Term": "Jenkins", "Occurrences": 5, "sources": sources1},
            {"Term": "Jenidns", "Occurrences": 8, "sources": sources2},
        ])
        # Returns Jenkins entry
    """
    scorer = create_canonical_scorer(known_words)
    return scorer.select_canonical(variants)
