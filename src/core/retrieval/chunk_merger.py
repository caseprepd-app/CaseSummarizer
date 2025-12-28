"""
Chunk Merger for Multi-Algorithm Retrieval.

Merges and ranks results from multiple retrieval algorithms.
When the same chunk is found by multiple algorithms, their relevance scores
are combined using weighted averaging.

Merge Strategy:
1. Collect all retrieved chunks from all algorithms
2. Group by chunk_id (same chunk from different algorithms)
3. Combine relevance scores weighted by algorithm weights
4. Rank by combined score
5. Track which algorithms found each chunk (for ML features)

This mirrors the vocabulary extraction ResultMerger pattern for consistency.
"""

from dataclasses import dataclass, field
from typing import Any

from src.core.retrieval.base import AlgorithmRetrievalResult, RetrievedChunk


@dataclass
class MergedChunk:
    """
    A chunk after merging results from multiple algorithms.

    This represents the consensus view of a chunk's relevance after combining
    input from all algorithms that retrieved it.

    Attributes:
        chunk_id: Unique identifier for the chunk
        text: The chunk text content
        combined_score: Weighted average relevance across algorithms
        sources: List of algorithm names that retrieved this chunk
        filename: Source document filename
        chunk_num: Chunk number within the document
        section_name: Section name (if available)
        metadata: Combined metadata from all sources (for ML training)
    """

    chunk_id: str
    text: str
    combined_score: float
    sources: list[str]
    filename: str
    chunk_num: int = 0
    section_name: str = "N/A"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MergedRetrievalResult:
    """
    Result after merging from all algorithms.

    Contains the final ranked list of chunks and processing metadata.

    Attributes:
        chunks: List of MergedChunk, sorted by combined_score descending
        total_algorithms: Number of algorithms that contributed
        processing_time_ms: Total time for retrieval + merging
        query: The original query string
        metadata: Merge-level statistics
    """

    chunks: list[MergedChunk]
    total_algorithms: int
    processing_time_ms: float = 0.0
    query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of merged chunks."""
        return len(self.chunks)


class ChunkMerger:
    """
    Merges and ranks results from multiple retrieval algorithms.

    The merger groups chunks by chunk_id, then combines their relevance scores
    using algorithm weights. Chunks found by multiple algorithms get boosted
    scores, reflecting higher confidence.

    Scoring Strategy:
    1. Weighted average of relevance scores from each algorithm
    2. Bonus for being found by multiple algorithms (+0.1 per additional algo)
    3. Final score clamped to 0-1 range

    Example:
        merger = ChunkMerger(algorithm_weights={"BM25+": 1.0, "FAISS": 0.5})
        merged = merger.merge([bm25_result, faiss_result], k=10)
    """

    def __init__(self, algorithm_weights: dict[str, float] | None = None):
        """
        Initialize merger with algorithm weights.

        Args:
            algorithm_weights: Mapping of algorithm name to weight (0.0-1.0+).
                              Higher weight = more influence on final score.
                              If None, all algorithms weighted equally at 1.0.
        """
        self.algorithm_weights = algorithm_weights or {}

        # Bonus score for each additional algorithm that finds the chunk
        # This reflects higher confidence when multiple methods agree
        self.multi_algo_bonus = 0.1

    def merge(
        self,
        results: list[AlgorithmRetrievalResult],
        k: int | None = None
    ) -> MergedRetrievalResult:
        """
        Merge results from multiple algorithms.

        Groups chunks by chunk_id, combines scores, and ranks.

        Args:
            results: List of AlgorithmRetrievalResult from different algorithms
            k: Maximum number of chunks to return (None = return all)

        Returns:
            MergedRetrievalResult with ranked chunks
        """
        import time

        start_time = time.perf_counter()

        # Group by chunk_id
        chunk_groups: dict[str, list[RetrievedChunk]] = {}

        for result in results:
            for chunk in result.chunks:
                key = chunk.chunk_id
                if key not in chunk_groups:
                    chunk_groups[key] = []
                chunk_groups[key].append(chunk)

        # Merge each group
        merged_chunks = []
        for chunk_id, chunks in chunk_groups.items():
            merged = self._merge_group(chunks)
            merged_chunks.append(merged)

        # Sort by combined score (descending)
        merged_chunks.sort(key=lambda c: c.combined_score, reverse=True)

        # Limit to top k if specified
        if k is not None:
            merged_chunks = merged_chunks[:k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Get query from first result
        query = results[0].query if results else ""

        return MergedRetrievalResult(
            chunks=merged_chunks,
            total_algorithms=len(results),
            processing_time_ms=elapsed_ms,
            query=query,
            metadata={
                "algorithm_weights": self.algorithm_weights,
                "multi_algo_bonus": self.multi_algo_bonus,
                "total_unique_chunks": len(chunk_groups),
                "chunks_returned": len(merged_chunks),
            }
        )

    def _merge_group(self, chunks: list[RetrievedChunk]) -> MergedChunk:
        """
        Merge a group of chunks (same chunk_id from different algorithms).

        Args:
            chunks: All retrievals of the same chunk from different algorithms

        Returns:
            Single MergedChunk combining all algorithm scores
        """
        # Use first chunk as template
        first = chunks[0]

        # Collect unique sources
        sources = list(set(c.source_algorithm for c in chunks))

        # Calculate weighted score
        combined_score = self._calculate_weighted_score(chunks)

        # Apply multi-algorithm bonus
        if len(sources) > 1:
            bonus = (len(sources) - 1) * self.multi_algo_bonus
            combined_score = min(1.0, combined_score + bonus)

        # Merge metadata for ML training
        merged_metadata = {
            "source_details": [
                {
                    "algorithm": c.source_algorithm,
                    "relevance_score": c.relevance_score,
                    "raw_score": c.raw_score,
                    **c.metadata
                }
                for c in chunks
            ],
            "algorithm_count": len(sources),
            "multi_algo_bonus_applied": len(sources) > 1,
        }

        return MergedChunk(
            chunk_id=first.chunk_id,
            text=first.text,
            combined_score=combined_score,
            sources=sources,
            filename=first.filename,
            chunk_num=first.chunk_num,
            section_name=first.section_name,
            metadata=merged_metadata,
        )

    def _calculate_weighted_score(self, chunks: list[RetrievedChunk]) -> float:
        """
        Calculate weighted average score across algorithms.

        Args:
            chunks: All retrievals of the same chunk

        Returns:
            Weighted average score (0.0-1.0)
        """
        total_weight = 0.0
        weighted_score = 0.0

        for chunk in chunks:
            weight = self.algorithm_weights.get(chunk.source_algorithm, 1.0)
            weighted_score += chunk.relevance_score * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_score / total_weight
        return 0.5  # Default if no weights

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Update algorithm weights (e.g., from ML learner).

        Args:
            new_weights: New weight mapping to use for future merges
        """
        self.algorithm_weights.update(new_weights)
