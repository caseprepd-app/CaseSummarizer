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

import time
from typing import Any

import numpy as np  # PERF-004: Move to module level
from rank_bm25 import BM25Plus

from src.config import BM25_B, BM25_DELTA, BM25_K1, DEBUG_MODE
from src.core.retrieval.algorithms import register_algorithm
from src.core.retrieval.base import (
    AlgorithmRetrievalResult,
    BaseRetrievalAlgorithm,
    DocumentChunk,
    RetrievedChunk,
)
from src.core.utils.tokenizer import tokenize_simple
from src.logging_config import debug_log


@register_algorithm
class BM25PlusRetriever(BaseRetrievalAlgorithm):
    """
    BM25+ retrieval algorithm for lexical/keyword search.

    Uses the BM25Plus algorithm from rank_bm25 library for term-based retrieval.
    Scores are normalized to 0-1 range for compatibility with other algorithms.

    Attributes:
        name: Algorithm identifier ("BM25+")
        weight: Default weight for merging (1.0 - primary algorithm)
        enabled: Whether this algorithm is active

    Example:
        retriever = BM25PlusRetriever()
        retriever.index_documents(chunks)
        results = retriever.retrieve("Who are the plaintiffs?", k=5)
    """

    name: str = "BM25+"
    weight: float = 1.0
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

        # Build BM25+ index
        # BM25Plus uses library defaults (k1=1.5, b=0.75, delta=1)
        # Our config constants match these: BM25_K1, BM25_B, BM25_DELTA
        self._index = BM25Plus(self._tokenized_corpus)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if DEBUG_MODE:
            debug_log(f"[BM25+] Indexed {len(chunks)} chunks in {elapsed_ms:.1f}ms")
            avg_tokens = sum(len(t) for t in self._tokenized_corpus) / len(chunks)
            debug_log(f"[BM25+] Average tokens per chunk: {avg_tokens:.1f}")

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

        if DEBUG_MODE:
            debug_log(f"[BM25+] Query: '{query[:50]}...' -> {len(query_tokens)} tokens")

        # Get BM25+ scores for all documents
        raw_scores = self._index.get_scores(query_tokens)

        # Get top-k indices (sorted by score descending)
        # PERF-004: Use module-level numpy import
        top_k_indices = np.argsort(raw_scores)[::-1][:k]

        # Normalize scores to 0-1 range
        # BM25 scores are unbounded positive values
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0

        # Build result chunks
        retrieved_chunks = []
        for idx in top_k_indices:
            raw_score = raw_scores[idx]

            # Skip zero-score chunks (no query terms found)
            if raw_score <= 0:
                continue

            chunk = self._chunks[idx]

            # Normalize score to 0-1 range
            # Using sigmoid-like normalization to handle score distribution
            normalized_score = raw_score / (raw_score + max_score) if max_score > 0 else 0

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

        if DEBUG_MODE:
            debug_log(f"[BM25+] Retrieved {len(retrieved_chunks)} chunks in {elapsed_ms:.1f}ms")
            for i, chunk in enumerate(retrieved_chunks[:3]):
                debug_log(
                    f"  [{i + 1}] score={chunk.raw_score:.2f} -> {chunk.relevance_score:.3f} | {chunk.filename}"
                )

        return AlgorithmRetrievalResult(
            chunks=retrieved_chunks,
            processing_time_ms=elapsed_ms,
            query=query,
            metadata={
                "algorithm": self.name,
                "index_size": len(self._chunks),
                "max_raw_score": max_score,
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
