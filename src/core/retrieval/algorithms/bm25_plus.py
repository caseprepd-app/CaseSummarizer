"""
BM25+ Retrieval Algorithm for CasePrepd Q&A.

Implements lexical (keyword-based) retrieval using BM25+, an improved version
of the classic BM25 algorithm that addresses term frequency saturation issues.

BM25+ vs BM25:
- Standard BM25 has diminishing returns for term frequency (TF saturation)
- BM25+ adds a lower bound (delta) to the TF component, improving recall
- Better for legal documents where exact terminology matters

Why BM25+ for Legal Documents:
- Legal language is precise - exact terms matter ("plaintiff" vs "claimant")
- No neural model needed - faster, deterministic, no GPU required
- Works out-of-the-box without domain-specific training
- Handles rare legal terminology that embedding models may not understand

Reference:
    Lv, Y., & Zhai, C. (2011). "Lower-bounding term frequency normalization"
    CIKM '11: Proceedings of the 20th ACM international conference on Information
    and knowledge management.
"""

import logging
import time
from typing import Any

import numpy as np  # PERF-004: Move to module level
from rank_bm25 import BM25Plus

from src.config import BM25_B, BM25_DELTA, BM25_K1
from src.core.retrieval.algorithms import register_algorithm
from src.core.retrieval.base import (
    AlgorithmRetrievalResult,
    BaseRetrievalAlgorithm,
    DocumentChunk,
    RetrievedChunk,
)
from src.core.utils.tokenizer import tokenize_simple

logger = logging.getLogger(__name__)


@register_algorithm
class BM25PlusRetriever(BaseRetrievalAlgorithm):
    """
    BM25+ retrieval algorithm for lexical/keyword search.

    Uses the BM25Plus algorithm from rank_bm25 library for term-based retrieval.
    Scores are normalized to 0-1 range for compatibility with other algorithms.

    Attributes:
        name: Algorithm identifier ("BM25+")
        weight: Default weight for merging (0.2 - secondary algorithm)
        enabled: Whether this algorithm is active

    Example:
        retriever = BM25PlusRetriever()
        retriever.index_documents(chunks)
        results = retriever.retrieve("Who are the plaintiffs?", k=5)
    """

    name: str = "BM25+"
    weight: float = 0.2  # Secondary weight - exact term matching for precision
    enabled: bool = True

    def __init__(self):
        """Initialize BM25+ retriever."""
        self._index: BM25Plus | None = None
        self._chunks: list[DocumentChunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def index_documents(self, chunks: list[DocumentChunk]) -> None:
        """
        Build BM25+ index from document chunks.

        Tokenizes each chunk and creates the BM25+ index for retrieval.

        Args:
            chunks: List of DocumentChunk objects to index

        Raises:
            ValueError: If chunks is empty
        """
        start_time = time.perf_counter()

        if not chunks:
            raise ValueError("Cannot index empty chunk list")

        self._chunks = chunks

        # Tokenize all chunks
        self._tokenized_corpus = [tokenize_simple(chunk.text) for chunk in chunks]

        # Build BM25+ index with validated config parameters
        # Standard defaults: k1=1.2-2.0, b=0.0-1.0, delta=0.5-1.0
        k1 = max(0.0, min(float(BM25_K1), 5.0))  # Clamp to sensible range
        b = max(0.0, min(float(BM25_B), 1.0))  # Must be 0-1
        delta = max(0.0, min(float(BM25_DELTA), 3.0))  # Clamp to sensible range
        self._index = BM25Plus(self._tokenized_corpus, k1=k1, b=b, delta=delta)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug("Indexed %d chunks in %.1fms", len(chunks), elapsed_ms)
        avg_tokens = sum(len(t) for t in self._tokenized_corpus) / len(chunks)
        logger.debug("Average tokens per chunk: %.1f", avg_tokens)

    def retrieve(self, query: str, k: int = 5) -> AlgorithmRetrievalResult:
        """
        Retrieve top-k relevant chunks using BM25+ scoring.

        Args:
            query: The search query string
            k: Maximum number of chunks to retrieve

        Returns:
            AlgorithmRetrievalResult with ranked chunks

        Raises:
            RuntimeError: If index_documents() hasn't been called
        """
        start_time = time.perf_counter()

        if not self.is_indexed:
            raise RuntimeError("Index not built. Call index_documents() first.")

        # Tokenize query
        query_tokens = tokenize_simple(query)

        logger.debug("Query: '%s...' -> %d tokens", query[:50], len(query_tokens))

        # Get BM25+ scores for all documents
        raw_scores = self._index.get_scores(query_tokens)

        # Get top-k indices (sorted by score descending)
        # PERF-004: Use module-level numpy import
        top_k_indices = np.argsort(raw_scores)[::-1][:k]

        # Build result chunks
        # Fixed sigmoid constant for BM25 score normalization
        # K=2.0: a raw BM25 score of ~2 maps to 0.5 (decent match)
        # Low scores stay low instead of inflating to ~0.5
        BM25_NORM_K = 2.0

        retrieved_chunks = []
        for idx in top_k_indices:
            raw_score = raw_scores[idx]

            # Skip zero-score chunks (no query terms found)
            if raw_score <= 0:
                continue

            chunk = self._chunks[idx]

            # Normalize score to 0-1 range using fixed sigmoid
            # Unlike raw/(raw+max), this doesn't inflate when all scores are similar
            normalized_score = raw_score / (raw_score + BM25_NORM_K)

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    relevance_score=normalized_score,
                    raw_score=raw_score,
                    source_algorithm=self.name,
                    filename=chunk.filename,
                    chunk_num=chunk.chunk_num,
                    section_name=chunk.section_name,
                    metadata={
                        "query_tokens": query_tokens,
                        "chunk_tokens": len(self._tokenized_corpus[idx]),
                    },
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug("Retrieved %d chunks in %.1fms", len(retrieved_chunks), elapsed_ms)
        for i, chunk in enumerate(retrieved_chunks[:3]):
            logger.debug(
                "  [%d] score=%.2f -> %.3f | %s",
                i + 1,
                chunk.raw_score,
                chunk.relevance_score,
                chunk.filename,
            )

        return AlgorithmRetrievalResult(
            chunks=retrieved_chunks,
            processing_time_ms=elapsed_ms,
            query=query,
            metadata={
                "algorithm": self.name,
                "index_size": len(self._chunks),
                "max_raw_score": float(max(raw_scores)) if len(raw_scores) > 0 else 0.0,
                "query_token_count": len(query_tokens),
            },
        )

    @property
    def is_indexed(self) -> bool:
        """Check if the BM25+ index is built."""
        return self._index is not None and len(self._chunks) > 0

    def get_config(self) -> dict[str, Any]:
        """Return BM25+ configuration."""
        config = super().get_config()
        config.update(
            {
                "index_size": len(self._chunks) if self._chunks else 0,
                "algorithm_variant": "BM25Plus",
                "parameters": {
                    "k1": BM25_K1,
                    "b": BM25_B,
                    "delta": BM25_DELTA,
                },
            }
        )
        return config
