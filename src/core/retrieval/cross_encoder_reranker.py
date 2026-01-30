"""
Cross-Encoder Reranker for Q&A Retrieval.

Uses Alibaba-NLP/gte-reranker-modernbert-base (149M params, 8192-token context)
to rerank candidate chunks after hybrid retrieval. The 8192-token limit means
full chunks are scored without truncation.

Architecture:
- Lazy-loads model on first use to avoid startup overhead (~300MB)
- Takes top-K candidates from hybrid retrieval
- Reranks using cross-encoder and returns top-N
- Preserves original scores in metadata for debugging

Usage:
    reranker = CrossEncoderReranker()
    reranked_chunks = reranker.rerank(
        query="Who are the plaintiffs?",
        chunks=hybrid_result.chunks,
        top_k=5
    )
"""

import logging
import os

from src.config import (
    HF_CACHE_DIR,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL_LOCAL_PATH,
    RERANKER_MODEL_NAME,
)

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks chunks using cross-encoder for improved precision.

    Cross-encoders see query and document together, enabling more nuanced
    relevance judgments than bi-encoders (which encode them separately).

    The model is lazy-loaded on first use to avoid startup delay.
    Downloads to project's models/.hf_cache folder on first use (~300MB).
    """

    def __init__(self):
        """Initialize reranker (model loaded on first use)."""
        self._model = None

    def _load_model(self) -> None:
        """
        Load cross-encoder model (called once on first use).

        Uses bundled model if available, otherwise downloads from HuggingFace.
        """
        # Set HuggingFace cache to project folder
        os.environ["HF_HOME"] = str(HF_CACHE_DIR)
        os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)

        from sentence_transformers import CrossEncoder

        # Check for bundled model first
        if RERANKER_MODEL_LOCAL_PATH.exists():
            model_path = str(RERANKER_MODEL_LOCAL_PATH)
            logger.debug("Using bundled model: %s", model_path)
        else:
            model_path = RERANKER_MODEL_NAME
            logger.debug("Downloading model: %s", model_path)
            logger.debug("Cache directory: %s", HF_CACHE_DIR)

        self._model = CrossEncoder(model_path, max_length=RERANKER_MAX_LENGTH)

        logger.debug("Cross-encoder model loaded successfully")

    # Minimum sigmoid score to consider a chunk genuinely relevant.
    # sigmoid(0)=0.5, so 0.3 means the cross-encoder must give a positive
    # signal. Without this, irrelevant chunks pass through when the document
    # simply doesn't contain the answer.
    MIN_RELEVANCE_SCORE = 0.3

    def rerank(self, query: str, chunks: list, top_k: int = 5) -> list:
        """
        Rerank chunks by cross-encoder relevance score.

        Filters out chunks below MIN_RELEVANCE_SCORE to prevent irrelevant
        context from reaching the LLM when the answer isn't in the documents.

        Args:
            query: The user's question
            chunks: List of MergedChunk objects from hybrid retrieval
            top_k: Number of top chunks to return after reranking

        Returns:
            List of top_k MergedChunk objects, sorted by reranker score.
            May return fewer than top_k if chunks are below relevance threshold.
        """
        if not chunks:
            return []

        # Lazy-load model on first use
        if self._model is None:
            try:
                self._load_model()
            except Exception as e:
                logger.error("Failed to load cross-encoder model: %s", e)
                return list(chunks[:top_k])

        # Build query-document pairs for cross-encoder
        pairs = [[query, chunk.text] for chunk in chunks]

        # Warn if any pair approaches the model's token limit (~4 chars/token estimate)
        char_limit = RERANKER_MAX_LENGTH * 4  # conservative chars-to-tokens estimate
        for chunk in chunks:
            pair_len = len(query) + len(chunk.text)
            if pair_len > char_limit * 0.8:
                logger.warning(
                    "Chunk %s #%d is %.0f%% of reranker max length (%d chars / ~%d limit)",
                    chunk.filename,
                    chunk.chunk_num,
                    (pair_len / char_limit) * 100,
                    pair_len,
                    char_limit,
                )

        logger.debug("Reranking %d chunks for query: %s...", len(chunks), query[:50])

        # Get cross-encoder scores (raw logits)
        raw_scores = self._model.predict(pairs)

        # Normalize scores to 0-1 using sigmoid
        # Cross-encoder logits can be any value; sigmoid maps to (0, 1)
        import numpy as np

        normalized_scores = 1 / (1 + np.exp(-np.array(raw_scores)))

        # Pair scores with chunks and sort by score descending
        scored_chunks = list(zip(normalized_scores, raw_scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Take top_k, filtering out chunks below minimum relevance
        reranked = []
        filtered_count = 0
        for i, (norm_score, raw_score, chunk) in enumerate(scored_chunks[:top_k]):
            # Filter out genuinely irrelevant chunks
            if float(norm_score) < self.MIN_RELEVANCE_SCORE:
                filtered_count += 1
                logger.debug(
                    "FILTERED #%d: %s chunk %d - rerank=%.3f < threshold %s",
                    i + 1,
                    chunk.filename,
                    chunk.chunk_num,
                    norm_score,
                    self.MIN_RELEVANCE_SCORE,
                )
                continue

            # Store original hybrid score for debugging/analysis
            if "original_hybrid_score" not in chunk.metadata:
                chunk.metadata["original_hybrid_score"] = chunk.combined_score

            # Store both raw and normalized reranker scores
            chunk.metadata["reranker_score_raw"] = float(raw_score)
            chunk.metadata["reranker_score"] = float(norm_score)
            chunk.metadata["rerank_position"] = len(reranked) + 1

            # Replace combined_score with normalized reranker score
            chunk.combined_score = float(norm_score)

            reranked.append(chunk)

            logger.debug(
                "#%d: %s chunk %d - hybrid=%.3f -> rerank=%.3f (raw=%.3f)",
                len(reranked),
                chunk.filename,
                chunk.chunk_num,
                chunk.metadata["original_hybrid_score"],
                norm_score,
                raw_score,
            )

        if filtered_count > 0:
            logger.debug(
                "Filtered %d chunks below relevance threshold %s",
                filtered_count,
                self.MIN_RELEVANCE_SCORE,
            )

        logger.debug("Returned %d of %d chunks", len(reranked), len(chunks))

        return reranked

    def is_available(self) -> bool:
        """
        Check if cross-encoder can be loaded.

        Returns:
            True if model can be loaded, False otherwise
        """
        try:
            if self._model is None:
                self._load_model()
            return self._model is not None
        except Exception as e:
            logger.debug("Model not available: %s", e)
            return False
