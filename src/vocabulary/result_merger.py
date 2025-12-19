"""
Result Merger for Multi-Algorithm Vocabulary Extraction

Merges and deduplicates results from multiple extraction algorithms.
When the same term is found by multiple algorithms, their confidence scores
are combined and the term's sources are tracked for ML training.

Merge Strategy:
1. Normalize term text (lowercase for matching, preserve original casing)
2. Group candidates by normalized term
3. Combine confidence scores weighted by algorithm weights
4. Resolve type conflicts (NER type takes precedence over RAKE)
5. Sum frequencies across all occurrences
6. Track which algorithms found each term (for ML features)
"""

from dataclasses import dataclass, field
from typing import Any

from src.vocabulary.algorithms.base import AlgorithmResult, CandidateTerm


@dataclass
class MergedTerm:
    """
    A term after merging results from multiple algorithms.

    This represents the consensus view of a term after combining input from
    all algorithms that detected it. Used as input to role detection and scoring.

    Attributes:
        term: The canonical term text (preserves original casing)
        sources: List of algorithm names that found this term
        combined_confidence: Weighted average confidence across algorithms
        final_type: Resolved type (Person, Medical, Technical, etc.)
        frequency: Total occurrences across all algorithm detections
        metadata: Combined metadata from all sources (for ML training)
    """
    term: str
    sources: list[str]
    combined_confidence: float
    final_type: str
    frequency: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ResultMerger:
    """
    Merges and deduplicates results from multiple extraction algorithms.

    The merger groups candidates by normalized (lowercase) term text,
    then combines their confidence scores using algorithm weights.

    Type Resolution Rules (in order of precedence):
    1. NER-detected type (Person, Place, etc.) - highest confidence
    2. First non-None suggested_type from any algorithm
    3. Default to "Technical"

    Example:
        merger = ResultMerger(algorithm_weights={"NER": 1.0, "RAKE": 0.7})
        merged = merger.merge([ner_result, rake_result])
    """

    # Type precedence: NER types are most trustworthy
    TYPE_PRECEDENCE = {
        "Person": 10,
        "Place": 9,
        "Medical": 8,
        "Organization": 7,
        "Technical": 5,
        "Unknown": 1,
    }

    def __init__(self, algorithm_weights: dict[str, float] | None = None):
        """
        Initialize merger with algorithm weights.

        Args:
            algorithm_weights: Mapping of algorithm name to weight (0.0-1.0+).
                              Higher weight = more influence on final confidence.
                              If None, all algorithms weighted equally at 1.0.
        """
        self.algorithm_weights = algorithm_weights or {}

    def merge(self, results: list[AlgorithmResult]) -> list[MergedTerm]:
        """
        Merge results from multiple algorithms.

        Groups candidates by normalized term text, combines confidence scores,
        resolves type conflicts, and sums frequencies.

        Args:
            results: List of AlgorithmResult from different algorithms

        Returns:
            List of deduplicated MergedTerm objects, ready for further processing
        """
        # Group by normalized term
        term_groups: dict[str, list[CandidateTerm]] = {}

        for result in results:
            for candidate in result.candidates:
                # Normalize: lowercase, strip whitespace
                key = candidate.term.lower().strip()
                if not key:
                    continue  # Skip empty terms

                if key not in term_groups:
                    term_groups[key] = []
                term_groups[key].append(candidate)

        # Merge each group
        merged = []
        for normalized_key, candidates in term_groups.items():
            merged_term = self._merge_group(candidates)
            merged.append(merged_term)

        return merged

    def _merge_group(self, candidates: list[CandidateTerm]) -> MergedTerm:
        """
        Merge a group of candidates for the same term.

        Args:
            candidates: All candidates with the same normalized term text

        Returns:
            Single MergedTerm combining all candidate information
        """
        # Canonical term: use the most frequent capitalization variant
        canonical_term = self._select_canonical_casing(candidates)

        # Collect unique sources (which algorithms found this term)
        sources = list(set(c.source_algorithm for c in candidates))

        # Weighted confidence combination
        combined_confidence = self._calculate_weighted_confidence(candidates)

        # Resolve type conflicts
        final_type = self._resolve_type(candidates)

        # Sum frequencies (term appeared this many times total)
        total_frequency = sum(c.frequency for c in candidates)

        # Merge metadata for ML training (track per-algorithm details)
        merged_metadata = {
            "source_details": [
                {
                    "algorithm": c.source_algorithm,
                    "confidence": c.confidence,
                    "suggested_type": c.suggested_type,
                    **c.metadata
                }
                for c in candidates
            ],
            "algorithm_count": len(sources),
        }

        return MergedTerm(
            term=canonical_term,
            sources=sources,
            combined_confidence=combined_confidence,
            final_type=final_type,
            frequency=total_frequency,
            metadata=merged_metadata,
        )

    def _calculate_weighted_confidence(self, candidates: list[CandidateTerm]) -> float:
        """
        Calculate weighted average confidence across algorithms.

        Args:
            candidates: All candidates for the same term

        Returns:
            Weighted average confidence (0.0-1.0)
        """
        total_weight = 0.0
        weighted_confidence = 0.0

        for candidate in candidates:
            weight = self.algorithm_weights.get(candidate.source_algorithm, 1.0)
            weighted_confidence += candidate.confidence * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_confidence / total_weight
        return 0.5  # Default if no weights

    def _resolve_type(self, candidates: list[CandidateTerm]) -> str:
        """
        Resolve type conflicts when multiple algorithms suggest different types.

        Uses precedence rules: NER types > other types > default.

        Args:
            candidates: All candidates for the same term

        Returns:
            Resolved type string
        """
        best_type = "Technical"  # Default
        best_precedence = 0

        for candidate in candidates:
            if candidate.suggested_type is None:
                continue

            # Higher precedence from NER algorithm
            precedence = self.TYPE_PRECEDENCE.get(candidate.suggested_type, 1)

            # NER algorithm gets bonus precedence
            if candidate.source_algorithm == "NER":
                precedence += 5

            if precedence > best_precedence:
                best_precedence = precedence
                best_type = candidate.suggested_type

        return best_type

    def _select_canonical_casing(self, candidates: list[CandidateTerm]) -> str:
        """
        Select the most frequent capitalization variant as the canonical form.

        Tracks each unique casing variant weighted by its occurrence frequency.
        Returns the variant that appears most often across the document.

        Example:
            "JAMES LUCAS" (freq 2), "James Lucas" (freq 10), "james lucas" (freq 1)
            → Returns "James Lucas" (highest frequency)

        Args:
            candidates: All candidates for the same normalized term

        Returns:
            The most frequently occurring capitalization variant
        """
        if len(candidates) == 1:
            return candidates[0].term

        # Count frequency of each capitalization variant
        casing_counts: dict[str, int] = {}
        for candidate in candidates:
            term = candidate.term
            freq = candidate.frequency
            casing_counts[term] = casing_counts.get(term, 0) + freq

        # Return the variant with highest frequency
        # Tie-breaker: prefer mixed case over ALL CAPS (more readable)
        def sort_key(item):
            term, count = item
            # Primary: highest count
            # Secondary: prefer mixed case (not all upper, not all lower)
            is_mixed = not term.isupper() and not term.islower()
            return (count, is_mixed, term)

        sorted_variants = sorted(casing_counts.items(), key=sort_key, reverse=True)
        return sorted_variants[0][0]

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Update algorithm weights (e.g., from ML learner).

        Args:
            new_weights: New weight mapping to use for future merges
        """
        self.algorithm_weights.update(new_weights)
