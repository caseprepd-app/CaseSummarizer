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

Session 78: Initial implementation for canonical spelling improvement.
"""

from src.core.utils.ocr_patterns import has_ocr_artifacts
from src.core.vocabulary.term_sources import TermSources
from src.logging_config import debug_log


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

    def calculate_score(self, term: str, sources: TermSources) -> float:
        """
        Calculate the canonical score for a term.

        Formula:
            base_score = sources.weighted_score  # (conf * count)^1.1
            ocr_factor = 1.0 - ocr_penalty if has_ocr_artifacts else 1.0
            final_score = base_score * ocr_factor

        Args:
            term: The term string
            sources: TermSources tracking document origins

        Returns:
            Float score (higher = more likely to be canonical)
        """
        if sources is None:
            # Fallback for terms without source tracking
            debug_log(f"[CANONICAL] No sources for '{term}', using default score")
            return 0.0

        base_score = sources.weighted_score

        # Apply OCR artifact penalty if detected
        if has_ocr_artifacts(term):
            penalty = self.ocr_penalty
            base_score *= 1.0 - penalty
            debug_log(
                f"[CANONICAL] OCR penalty applied to '{term}': "
                f"-{penalty * 100:.0f}% → score={base_score:.2f}"
            )

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

        debug_log(
            f"[CANONICAL] {len(variants)} variants: "
            f"{len(known_variants)} known, {len(unknown_variants)} unknown"
        )

        # Branch based on how many are known
        if len(known_variants) == 1:
            # Exactly one known → it wins decisively
            canonical = known_variants[0]
            debug_log(f"[CANONICAL] '{canonical.get(term_key)}' wins (only known variant)")
            return self._merge_into_canonical(canonical, variants, term_key, sources_key)

        elif len(known_variants) == 0:
            # None known → exotic name scenario, use weighted scoring
            debug_log("[CANONICAL] No known variants, using weighted scores")
            return self._select_by_score(variants, term_key, sources_key)

        else:
            # Multiple known → use weighted score as tiebreaker
            debug_log(
                f"[CANONICAL] Multiple known variants "
                f"({[v.get(term_key) for v in known_variants]}), "
                f"using weighted scores"
            )
            result = self._select_by_score(known_variants, term_key, sources_key)

            # Merge unknown variants' frequencies into the canonical
            unknown_variants = [v for v in variants if v not in known_variants]
            if unknown_variants:
                extra_freq = sum(
                    v.get("In-Case Freq", 0) or v.get("in_case_freq", 0) or 0
                    for v in unknown_variants
                )
                result["In-Case Freq"] = result.get("In-Case Freq", 0) + extra_freq
                if "in_case_freq" in result:
                    result["in_case_freq"] = result.get("in_case_freq", 0) + extra_freq

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
                    v.get("In-Case Freq", 1),
                    v.get("source_doc_confidence", 1.0) / 100.0,  # Normalize if needed
                )

            score = self.calculate_score(term, sources)
            scored.append((v, score))

            debug_log(f"[CANONICAL] '{term}' score={score:.2f}")

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Winner is the highest score
        canonical, best_score = scored[0]
        debug_log(f"[CANONICAL] '{canonical.get(term_key)}' wins with score={best_score:.2f}")

        # Return with merged sources from all variants
        return self._merge_into_canonical(
            canonical,
            [v for v, _ in scored],
            term_key,
            sources_key,
        )

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
            v.get("In-Case Freq", 0) or v.get("in_case_freq", 0) or 0 for v in all_variants
        )
        result["In-Case Freq"] = total_freq
        if "in_case_freq" in result:
            result["in_case_freq"] = total_freq

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
                debug_log(
                    f"[CANONICAL] Merged {len(others)} variants into "
                    f"'{canonical.get(term_key)}': {others[:5]}"
                    f"{'...' if len(others) > 5 else ''}"
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
            {"Term": "Jenkins", "In-Case Freq": 5, "sources": sources1},
            {"Term": "Jenidns", "In-Case Freq": 8, "sources": sources2},
        ])
        # Returns Jenkins entry
    """
    scorer = create_canonical_scorer(known_words)
    return scorer.select_canonical(variants)
