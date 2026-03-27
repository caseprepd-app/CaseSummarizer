"""
Key Excerpts Extraction via K-Means Clustering

Extracts the most representative passages from a document set using
semantic embeddings and K-means clustering. Each cluster represents a
distinct topic; the passage closest to each centroid is selected.

Uses extract_key_passages() which reuses pre-computed chunk embeddings from
FAISS (no re-splitting or re-embedding).

No new dependencies — uses scikit-learn KMeans already in the venv.
"""

import logging
import re
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Boilerplate patterns common in legal documents
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*page\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*exhibit\s+[a-z0-9]+\s*$", re.IGNORECASE),
    re.compile(r"^\s*table\s+of\s+contents\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),  # Just a number
    re.compile(r"^\s*[ivxlcdm]+\.\s*$", re.IGNORECASE),  # Roman numeral only
    re.compile(r"^\s*cc:\s", re.IGNORECASE),
    re.compile(r"^\s*bcc:\s", re.IGNORECASE),
    re.compile(r"^\s*re:\s", re.IGNORECASE),
    re.compile(r"^\s*dated?\s*:\s*\w", re.IGNORECASE),
]

_MIN_WORDS = 5
_MAX_WORDS = 150


@dataclass
class KeySentence:
    """A single key sentence extracted from the document set."""

    text: str
    source_file: str
    position: int  # Global sentence index (for ordering)
    score: float  # Interestingness score (higher = more useful to user)


def compute_sentence_count(total_pages: int) -> int:
    """
    Scale K (number of key sentences) with document length.

    Rule: 1 key sentence per ~5 pages, min 5, max 15.

    Args:
        total_pages: Total pages across all documents.

    Returns:
        int: Number of key sentences to extract.
    """
    k = max(5, total_pages // 5)
    return min(k, 15)


def extract_key_passages(
    chunk_texts: list[str],
    chunk_embeddings: np.ndarray,
    chunk_metadata: list[dict],
    n: int | None = None,
    total_pages: int = 0,
    scorer_inputs: dict | None = None,
) -> list[KeySentence]:
    """
    Extract the most representative passages from pre-computed chunk data.

    Reuses chunk embeddings from FAISS vector store — no re-splitting or
    re-embedding needed. Chunks (after recursive sentence splitting) naturally
    contain speaker turns and Q&A context.

    Passages are sorted by interestingness score (most useful first) using
    vocab overlap, named entities, word rarity, and rejected term penalties.

    Args:
        chunk_texts: List of chunk text strings
        chunk_embeddings: (num_chunks, embedding_dim) array from FAISS builder
        chunk_metadata: List of dicts with 'source_file' and 'chunk_num'
        n: Number of key passages. If None, auto-scales with page count.
        total_pages: Total pages across all documents (for auto-scaling K).
        scorer_inputs: Pre-built dict with keys: vocab_terms, person_terms,
            rejected_terms, frequency_rank_map. Built by caller to avoid
            cross-module imports.

    Returns:
        List of KeySentence objects sorted by interestingness (best first).
    """
    if not chunk_texts or len(chunk_embeddings) == 0:
        logger.warning("No chunk data for key passages extraction")
        return []

    embeddings_array = np.array(chunk_embeddings, dtype=np.float32)
    logger.debug("Key passages: %d chunks provided", len(chunk_texts))

    # Filter out very short chunks (< 3 words)
    valid_indices = []
    for i, text in enumerate(chunk_texts):
        if len(text.split()) >= 3:
            valid_indices.append(i)

    if not valid_indices:
        logger.warning("No valid chunks after filtering")
        return []

    filtered_texts = [chunk_texts[i] for i in valid_indices]
    filtered_embeddings = embeddings_array[valid_indices]
    filtered_metadata = [chunk_metadata[i] for i in valid_indices]

    # Determine K — use 3-voter ensemble or explicit override
    if n is None:
        from src.core.summarization.k_selection import select_k

        fallback_k = compute_sentence_count(total_pages)
        n = select_k(filtered_embeddings, fallback_k)
    n = min(n, len(filtered_texts))

    # Cluster and select
    selected_indices = _cluster_and_select(filtered_embeddings, n)

    # Score and rank by interestingness
    results = _score_and_rank(
        selected_indices,
        filtered_texts,
        filtered_metadata,
        scorer_inputs,
    )

    logger.debug("Key passages: returning %d passages", len(results))
    return results


def _filter_sentences(sentences: list[dict]) -> list[dict]:
    """
    Remove short, long, and boilerplate sentences.

    Args:
        sentences: List of sentence dicts with 'text' key.

    Returns:
        Filtered list of sentence dicts.
    """
    filtered = []
    for sent in sentences:
        text = sent["text"]
        word_count = len(text.split())

        if word_count < _MIN_WORDS:
            continue
        if word_count > _MAX_WORDS:
            continue
        if any(pat.search(text) for pat in _BOILERPLATE_PATTERNS):
            continue

        filtered.append(sent)

    return filtered


def _cluster_and_select(embeddings: np.ndarray, n: int) -> list[int]:
    """
    K-means cluster sentence embeddings and select closest-to-centroid.

    Args:
        embeddings: (num_sentences, embedding_dim) array.
        n: Number of clusters / sentences to select.

    Returns:
        List of indices into the embeddings array.
    """
    from sklearn.cluster import KMeans

    if len(embeddings) <= n:
        return list(range(len(embeddings)))

    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    kmeans.fit(embeddings)

    selected = []
    for cluster_idx in range(n):
        # Find sentences in this cluster
        mask = kmeans.labels_ == cluster_idx
        cluster_indices = np.where(mask)[0]

        if len(cluster_indices) == 0:
            continue

        # Find closest to centroid
        centroid = kmeans.cluster_centers_[cluster_idx]
        cluster_embeddings = embeddings[cluster_indices]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        best_local = np.argmin(distances)
        selected.append(int(cluster_indices[best_local]))

    return selected


def _score_and_rank(
    selected_indices: list[int],
    texts: list[str],
    metadata: list[dict],
    scorer_inputs: dict | None,
) -> list[KeySentence]:
    """
    Score selected excerpts by interestingness and sort best-first.

    Falls back to document position order if scoring is unavailable.

    Args:
        selected_indices: Indices chosen by K-means clustering
        texts: All filtered chunk texts
        metadata: All filtered chunk metadata
        scorer_inputs: Pre-built dict with scoring data (may be None)

    Returns:
        List of KeySentence sorted by score (highest first)
    """
    from src.core.summarization.excerpt_scorer import score_excerpt

    has_inputs = scorer_inputs and (
        scorer_inputs.get("vocab_terms") or scorer_inputs.get("frequency_rank_map")
    )

    results = []
    for idx in selected_indices:
        meta = metadata[idx]
        chunk_text = texts[idx]

        if has_inputs:
            interestingness = score_excerpt(
                chunk_text,
                scorer_inputs.get("vocab_terms", {}),
                scorer_inputs.get("person_terms", set()),
                scorer_inputs.get("rejected_terms", set()),
                scorer_inputs.get("frequency_rank_map", {}),
            )
        else:
            interestingness = 0.0

        results.append(
            KeySentence(
                text=chunk_text,
                source_file=meta.get("source_file", "unknown"),
                position=meta.get("chunk_num", idx),
                score=interestingness,
            )
        )

    # Sort by interestingness (highest first)
    results.sort(key=lambda s: s.score, reverse=True)
    return results
