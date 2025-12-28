"""
Retrieval Package for LocalScribe Q&A System.

Multi-algorithm document retrieval with weighted result merging.
Mirrors the vocabulary extraction architecture for consistency.

Architecture:
- BaseRetrievalAlgorithm: ABC for all retrieval algorithms
- BM25PlusRetriever: Lexical search with BM25+ scoring
- FAISSRetriever: Semantic search with embeddings
- ChunkMerger: Combines results from multiple algorithms
- HybridRetriever: Coordinates algorithm execution and merging

Example:
    from src.core.retrieval import HybridRetriever

    retriever = HybridRetriever(documents)
    results = retriever.retrieve("Who are the plaintiffs?", k=5)

    for chunk in results.chunks:
        print(f"{chunk.text[:100]}... (score: {chunk.combined_score:.2f})")
"""

from src.core.retrieval.base import (
    BaseRetrievalAlgorithm,
    RetrievedChunk,
    AlgorithmRetrievalResult,
    DocumentChunk,
)
from src.core.retrieval.chunk_merger import ChunkMerger, MergedChunk
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.query_transformer import QueryTransformer, QueryTransformResult

__all__ = [
    # Base classes
    "BaseRetrievalAlgorithm",
    "RetrievedChunk",
    "AlgorithmRetrievalResult",
    "DocumentChunk",
    # Merger
    "ChunkMerger",
    "MergedChunk",
    # Main retriever
    "HybridRetriever",
    # Query transformation (LlamaIndex)
    "QueryTransformer",
    "QueryTransformResult",
]
