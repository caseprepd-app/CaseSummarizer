"""
Per-Document Term Source Tracking

Tracks which documents contributed each vocabulary term occurrence, enabling:
1. Confidence-weighted canonical spelling selection
2. Richer ML features for preference learning
3. Better understanding of term provenance

This module provides the TermSources dataclass which stores document IDs,
confidence scores, and occurrence counts for each term.

NOTE: An ML model trained on user feedback could potentially leverage these
signals more effectively than the rules-based canonical scorer. This
infrastructure provides the foundation for future ML enhancements.

"""

from dataclasses import dataclass, field
from statistics import median as stats_median


@dataclass
class TermSources:
    """
    Tracks which documents contributed each term occurrence.

    This enables confidence-weighted scoring for canonical spelling selection
    and provides richer signals for ML preference learning.

    Attributes:
        doc_ids: List of document identifiers (typically SHA256 hashes)
        confidences: OCR/extraction confidence for each document (0.0-1.0)
        counts_per_doc: Number of times term appeared in each document

    Example:
        sources = TermSources(
            doc_ids=["hash1", "hash2"],
            confidences=[0.95, 0.60],
            counts_per_doc=[5, 3]
        )
        # "term" appeared 5 times in high-confidence doc, 3 times in low-confidence doc
        # sources.total_count == 8
        # sources.weighted_score considers confidence * count
    """

    doc_ids: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    counts_per_doc: list[int] = field(default_factory=list)

    def add_document(self, doc_id: str, confidence: float, count: int) -> None:
        """
        Add a document's contribution to this term.

        Args:
            doc_id: Document identifier (typically SHA256 hash)
            confidence: OCR/extraction confidence for the document (0.0-1.0)
            count: Number of times term appeared in this document
        """
        # Check if document already exists (update count if so)
        if doc_id in self.doc_ids:
            idx = self.doc_ids.index(doc_id)
            self.counts_per_doc[idx] += count
        else:
            self.doc_ids.append(doc_id)
            self.confidences.append(confidence)
            self.counts_per_doc.append(count)

    # -------------------------------------------------------------------------
    # Basic Aggregate Properties
    # -------------------------------------------------------------------------

    @property
    def total_count(self) -> int:
        """Total occurrences across all documents."""
        return sum(self.counts_per_doc)

    @property
    def num_documents(self) -> int:
        """Number of distinct documents containing this term."""
        return len(self.doc_ids)

    # -------------------------------------------------------------------------
    # Confidence Statistics
    # -------------------------------------------------------------------------

    @property
    def mean_confidence(self) -> float:
        """
        Count-weighted mean confidence.

        Weights each document's confidence by how many times the term
        appeared in that document. A term appearing 10 times in a
        high-confidence doc contributes more than one appearing once.
        """
        if not self.counts_per_doc or self.total_count == 0:
            return 0.0
        weighted = sum(c * n for c, n in zip(self.confidences, self.counts_per_doc, strict=False))
        return weighted / self.total_count

    @property
    def median_confidence(self) -> float:
        """
        Median confidence - robust to outliers.

        Expands confidences by count (e.g., conf=0.9 with count=3 becomes
        [0.9, 0.9, 0.9]) then takes the median. This prevents a single
        low-confidence document from dominating the metric.
        """
        if not self.confidences:
            return 0.0

        # Weight-expand confidences by count, then take median
        expanded = []
        for c, n in zip(self.confidences, self.counts_per_doc, strict=False):
            expanded.extend([c] * n)

        if not expanded:
            return 0.0

        return stats_median(expanded)

    @property
    def confidence_std_dev(self) -> float:
        """
        Standard deviation of document confidences (consistency signal).

        Low std_dev means term appears in consistently high-quality (or
        consistently low-quality) documents. High std_dev means mixed sources.

        Note: This is population std dev of document-level confidences,
        NOT weighted by count. Measures source diversity.
        """
        if len(self.confidences) < 2:
            return 0.0

        mean = sum(self.confidences) / len(self.confidences)
        variance = sum((c - mean) ** 2 for c in self.confidences) / len(self.confidences)
        return variance**0.5

    @property
    def min_confidence(self) -> float:
        """Minimum confidence across source documents."""
        return min(self.confidences) if self.confidences else 0.0

    @property
    def max_confidence(self) -> float:
        """Maximum confidence across source documents."""
        return max(self.confidences) if self.confidences else 0.0

    # -------------------------------------------------------------------------
    # Quality Flag Properties (for ML features)
    # -------------------------------------------------------------------------

    @property
    def high_conf_doc_ratio(self) -> float:
        """
        Proportion of source documents with confidence > 0.80.

        A term appearing in many high-confidence documents is more likely
        to be correctly spelled than one appearing only in poor scans.
        """
        if not self.confidences:
            return 0.0
        high_conf_count = sum(1 for c in self.confidences if c > 0.80)
        return high_conf_count / len(self.confidences)

    @property
    def all_low_conf(self) -> bool:
        """
        True if ALL source documents have confidence < 0.60.

        This is a red flag - the term may be an OCR artifact that only
        appears in poorly-scanned documents.
        """
        if not self.confidences:
            return True
        return all(c < 0.60 for c in self.confidences)

    def doc_diversity_ratio(self, total_docs_in_session: int) -> float:
        """
        Proportion of session documents containing this term.

        Args:
            total_docs_in_session: Total number of documents being processed

        Returns:
            Ratio from 0.0 to 1.0 (1.0 = term appears in all documents)

        A term appearing in many documents is more likely to be a real
        entity than one appearing in only one document.
        """
        if total_docs_in_session <= 0:
            return 0.0
        return self.num_documents / total_docs_in_session

    # -------------------------------------------------------------------------
    # Canonical Scoring
    # -------------------------------------------------------------------------

    @property
    def weighted_score(self) -> float:
        """
        Confidence-weighted frequency score for canonical spelling selection.

        Formula: sum(confidence * count) ^ 1.1

        The ^1.1 exponent boosts higher counts to be more impactful:
        - count=5  → 5^1.1  ≈ 6.0  (+20%)
        - count=10 → 10^1.1 ≈ 12.6 (+26%)
        - count=50 → 50^1.1 ≈ 69.6 (+39%)

        This creates meaningful separation - a term appearing 50 times
        isn't just 10x better than one appearing 5 times, it's ~11.6x better.

        The confidence weighting means terms from high-quality documents
        score higher than those from poor OCR scans.
        """
        if not self.counts_per_doc:
            return 0.0

        weighted_sum = sum(
            c * n for c, n in zip(self.confidences, self.counts_per_doc, strict=False)
        )
        return weighted_sum**1.1

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization or ML feature extraction.

        Returns dict with all computed properties for easy access.
        """
        return {
            "doc_ids": self.doc_ids,
            "confidences": self.confidences,
            "counts_per_doc": self.counts_per_doc,
            "total_count": self.total_count,
            "num_documents": self.num_documents,
            "mean_confidence": self.mean_confidence,
            "median_confidence": self.median_confidence,
            "confidence_std_dev": self.confidence_std_dev,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "high_conf_doc_ratio": self.high_conf_doc_ratio,
            "all_low_conf": self.all_low_conf,
            "weighted_score": self.weighted_score,
        }

    def merge_with(self, other: "TermSources") -> "TermSources":
        """
        Merge another TermSources into this one.

        Used when combining results from multiple extraction algorithms
        that may have found the same term in the same documents.

        Args:
            other: Another TermSources for the same term

        Returns:
            New TermSources with combined data
        """
        merged = TermSources(
            doc_ids=self.doc_ids.copy(),
            confidences=self.confidences.copy(),
            counts_per_doc=self.counts_per_doc.copy(),
        )

        for doc_id, conf, count in zip(
            other.doc_ids, other.confidences, other.counts_per_doc, strict=False
        ):
            merged.add_document(doc_id, conf, count)

        return merged

    @classmethod
    def from_single_document(cls, doc_id: str, confidence: float, count: int) -> "TermSources":
        """
        Create TermSources for a term found in a single document.

        Convenience constructor for the common case.

        Args:
            doc_id: Document identifier
            confidence: Document's OCR/extraction confidence (0.0-1.0)
            count: Number of occurrences in this document

        Returns:
            New TermSources instance
        """
        return cls(
            doc_ids=[doc_id],
            confidences=[confidence],
            counts_per_doc=[count],
        )

    @classmethod
    def create_legacy(cls, total_count: int, doc_confidence: float = 1.0) -> "TermSources":
        """
        Create TermSources from legacy data (no per-document tracking).

        Used for backward compatibility when processing single documents
        or when document source information is not available.

        Args:
            total_count: Total occurrence count
            doc_confidence: Single confidence value to use

        Returns:
            TermSources with a synthetic single document
        """
        return cls(
            doc_ids=["legacy"],
            confidences=[doc_confidence],
            counts_per_doc=[total_count],
        )


def merge_term_sources_dict(
    sources_by_term: dict[str, TermSources],
    new_sources: dict[str, TermSources],
) -> dict[str, TermSources]:
    """
    Merge two dictionaries of TermSources by term.

    Used when combining extraction results from multiple algorithms.

    Args:
        sources_by_term: Existing term → TermSources mapping
        new_sources: New term → TermSources mapping to merge in

    Returns:
        Merged dictionary with combined TermSources for each term
    """
    result = {term: sources.merge_with(TermSources()) for term, sources in sources_by_term.items()}

    for term, new_src in new_sources.items():
        if term in result:
            result[term] = result[term].merge_with(new_src)
        else:
            result[term] = new_src

    return result
