"""
Retrieval Package for CasePrepd Q&A System.

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
    AlgorithmRetrievalResult,
    BaseRetrievalAlgorithm,
    DocumentChunk,
    RetrievedChunk,
)
from src.core.retrieval.chunk_merger import ChunkMerger, MergedChunk
from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.query_transformer import QueryTransformer, QueryTransformResult

__all__ = [
    "AlgorithmRetrievalResult",
    # Base classes
    "BaseRetrievalAlgorithm",
    # Merger
    "ChunkMerger",
    # Cross-encoder reranking
    "CrossEncoderReranker",
    "DocumentChunk",
    # Main retriever
    "HybridRetriever",
    "MergedChunk",
    "QueryTransformResult",
    # Query transformation (LlamaIndex)
    "QueryTransformer",
    "RetrievedChunk",
]
