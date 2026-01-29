"""
Chunk Merger for Multi-Algorithm Retrieval.

Merges and ranks results from multiple retrieval algorithms using
Reciprocal Rank Fusion (RRF). Each algorithm's results are ranked
independently, then RRF combines ranks across algorithms.

RRF Formula: score(chunk) = sum(1 / (k + rank_i)) for each algorithm i
where k=60 (standard constant that prevents high-ranked items from dominating).

Why RRF over weighted averaging:
- Score distributions differ between algorithms (FAISS cosine vs BM25+ TF-IDF)
- RRF uses rank positions only, so score scale doesn't matter
- Chunks found by multiple algorithms naturally accumulate higher RRF scores
- No weight tuning needed (k=60 is the standard default)

Reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods"
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
        combined_score: RRF score across algorithms (higher = better)
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
    Merges and ranks results from multiple retrieval algorithms using RRF.

    Reciprocal Rank Fusion assigns each chunk a score based on its rank
    position within each algorithm's result list. Chunks found by multiple
    algorithms accumulate scores from each list, naturally boosting
    consensus results without explicit multi-algorithm bonuses.

    Example:
        merger = ChunkMerger()
        merged = merger.merge([bm25_result, faiss_result], k=10)
    """

    def __init__(self, algorithm_weights: dict[str, float] | None = None):
        """
        Initialize merger.

        Args:
            algorithm_weights: Legacy parameter, kept for backward compatibility
                              and metadata logging. Does not affect RRF scoring.
        """
        from src.config import RRF_K

        self.algorithm_weights = algorithm_weights or {}
        self.rrf_k = RRF_K

    def merge(
        self, results: list[AlgorithmRetrievalResult], k: int | None = None
    ) -> MergedRetrievalResult:
        """
        Merge results from multiple algorithms using Reciprocal Rank Fusion.

        For each algorithm, chunks are ranked by relevance_score descending.
        Each chunk receives RRF score = 1/(k + rank) from each algorithm
        that found it. Scores are summed across algorithms.

        Args:
            results: List of AlgorithmRetrievalResult from different algorithms
            k: Maximum number of chunks to return (None = return all)

        Returns:
            MergedRetrievalResult with ranked chunks
        """
        import time

        start_time = time.perf_counter()

        # Step 1: Compute RRF scores per chunk across all algorithms
        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, list[RetrievedChunk]] = {}

        for result in results:
            # Sort this algorithm's chunks by relevance_score descending
            sorted_chunks = sorted(result.chunks, key=lambda c: c.relevance_score, reverse=True)
            for rank, chunk in enumerate(sorted_chunks, start=1):
                cid = chunk.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
                if cid not in chunk_lookup:
                    chunk_lookup[cid] = []
                chunk_lookup[cid].append(chunk)

        # Step 2: Build MergedChunk for each unique chunk_id
        merged_chunks = []
        for chunk_id, chunks in chunk_lookup.items():
            merged = self._merge_group(chunks, rrf_scores[chunk_id])
            merged_chunks.append(merged)

        # Sort by RRF score (descending)
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
                "merge_strategy": "reciprocal_rank_fusion",
                "rrf_k": self.rrf_k,
                "algorithm_weights_legacy": self.algorithm_weights,
                "total_unique_chunks": len(chunk_lookup),
                "chunks_returned": len(merged_chunks),
            },
        )

    def _merge_group(self, chunks: list[RetrievedChunk], rrf_score: float) -> MergedChunk:
        """
        Build a MergedChunk from a group of chunks (same chunk_id).

        Args:
            chunks: All retrievals of the same chunk from different algorithms
            rrf_score: Pre-computed RRF score for this chunk

        Returns:
            Single MergedChunk with RRF score and source metadata
        """
        first = chunks[0]
        sources = list({c.source_algorithm for c in chunks})

        # Merge metadata for ML training
        merged_metadata = {
            "source_details": [
                {
                    "algorithm": c.source_algorithm,
                    "relevance_score": c.relevance_score,
                    "raw_score": c.raw_score,
                    **c.metadata,
                }
                for c in chunks
            ],
            "algorithm_count": len(sources),
            "rrf_score": rrf_score,
        }

        return MergedChunk(
            chunk_id=first.chunk_id,
            text=first.text,
            combined_score=rrf_score,
            sources=sources,
            filename=first.filename,
            chunk_num=first.chunk_num,
            section_name=first.section_name,
            metadata=merged_metadata,
        )

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Update algorithm weights (legacy, kept for backward compatibility).

        Args:
            new_weights: New weight mapping (stored for metadata only)
        """
        self.algorithm_weights.update(new_weights)
