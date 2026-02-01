"""
Chunk Scoring Utilities for Redundancy Detection.

Detects near-duplicate chunks using cosine similarity on embedding vectors.
Used by the summarization pipeline to skip redundant LLM calls.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkScores:
    """
    Per-chunk redundancy flags.

    Attributes:
        skip: True = redundant, skip this chunk during summarization.
        skip_reason: Human-readable reason (e.g. "redundant with chunk 3") or "".
    """

    skip: list[bool] = field(default_factory=list)
    skip_reason: list[str] = field(default_factory=list)


def detect_redundant_chunks(
    embeddings: list[list[float]],
    threshold: float = 0.98,
) -> ChunkScores:
    """
    Flag chunks that are near-duplicates of earlier chunks.

    For each chunk, computes cosine similarity against all earlier chunks.
    If any similarity exceeds threshold, the later chunk is flagged as skip.
    The first occurrence is always kept.

    Conservative threshold (0.98) avoids skipping chunks that are similar
    but contain different facts.

    Args:
        embeddings: List of embedding vectors (one per chunk).
        threshold: Cosine similarity threshold for flagging duplicates.

    Returns:
        ChunkScores with skip flags and reasons for each chunk.
    """
    n = len(embeddings)
    if n == 0:
        return ChunkScores()

    # Normalize embeddings for cosine similarity via dot product
    matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms

    skip = [False] * n
    skip_reason = [""] * n
    skipped_count = 0

    for i in range(1, n):
        # Dot product with all earlier chunks (already normalized = cosine sim)
        similarities = matrix[:i] @ matrix[i]
        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        if max_sim >= threshold:
            skip[i] = True
            skip_reason[i] = f"redundant with chunk {max_idx + 1} (sim={max_sim:.3f})"
            skipped_count += 1

    if skipped_count > 0:
        logger.info(
            "Redundancy detection: %d/%d chunks flagged as redundant (threshold=%.2f)",
            skipped_count,
            n,
            threshold,
        )

    return ChunkScores(skip=skip, skip_reason=skip_reason)
