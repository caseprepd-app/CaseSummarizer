"""
Key Excerpts Extraction via K-Means Clustering

Extracts the most representative passages from a document set using
semantic embeddings and K-means clustering. Each cluster represents a
distinct topic; the passage closest to each centroid is selected.

Two modes:
- extract_key_passages(): Reuses pre-computed chunk embeddings from FAISS
  (no re-splitting or re-embedding). Preferred.
- extract_key_sentences(): Legacy — re-splits and re-embeds from scratch.
  Kept for backward compatibility.

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
    score: float  # Distance to cluster centroid (lower = more representative)


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
) -> list[KeySentence]:
    """
    Extract the most representative passages from pre-computed chunk data.

    Reuses chunk embeddings from FAISS vector store — no re-splitting or
    re-embedding needed. Chunks (after recursive sentence splitting) naturally
    contain speaker turns and Q&A context.

    Args:
        chunk_texts: List of chunk text strings
        chunk_embeddings: (num_chunks, embedding_dim) array from FAISS builder
        chunk_metadata: List of dicts with 'source_file' and 'chunk_num'
        n: Number of key passages. If None, auto-scales with page count.
        total_pages: Total pages across all documents (for auto-scaling K).

    Returns:
        List of KeySentence objects sorted by chunk position.
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

    # Build result sorted by chunk position
    results = []
    for idx in selected_indices:
        meta = filtered_metadata[idx]
        results.append(
            KeySentence(
                text=filtered_texts[idx],
                source_file=meta.get("source_file", "unknown"),
                position=meta.get("chunk_num", idx),
                score=0.0,
            )
        )

    results.sort(key=lambda s: s.position)
    logger.debug("Key passages: returning %d passages", len(results))
    return results


def extract_key_sentences(
    documents: list[dict],
    embeddings_model,
    n: int | None = None,
) -> list[KeySentence]:
    """
    Extract the most representative sentences from a set of documents.

    Args:
        documents: List of dicts with 'filename' and either
                   'preprocessed_text' or 'extracted_text'.
        embeddings_model: A model with embed_documents(texts) -> list[list[float]].
        n: Number of key sentences. If None, auto-scales with page count.

    Returns:
        List of KeySentence objects sorted by document position.
    """
    from src.core.utils.sentence_splitter import split_sentences

    # Collect all sentences with metadata
    all_sentences: list[dict] = []
    total_pages = 0

    for doc in documents:
        filename = doc.get("filename", "unknown")
        text = doc.get("preprocessed_text") or doc.get("extracted_text", "")
        total_pages += doc.get("page_count", 0)

        if not text.strip():
            continue

        sentences = split_sentences(text)
        for sent in sentences:
            all_sentences.append(
                {
                    "text": sent.strip(),
                    "source_file": filename,
                    "position": len(all_sentences),
                }
            )

    logger.debug(
        "Key sentences: %d raw sentences from %d documents", len(all_sentences), len(documents)
    )

    # Filter sentences
    filtered = _filter_sentences(all_sentences)
    logger.debug("Key sentences: %d after filtering", len(filtered))

    if not filtered:
        logger.warning("No valid sentences after filtering")
        return []

    # Embed all filtered sentences
    texts = [s["text"] for s in filtered]
    try:
        embeddings = embeddings_model.embed_documents(texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)
    except Exception as e:
        logger.error("Key sentences embedding failed: %s", e, exc_info=True)
        return []

    # Determine K — use 3-voter ensemble or explicit override
    if n is None:
        from src.core.summarization.k_selection import select_k

        fallback_k = compute_sentence_count(total_pages)
        n = select_k(embeddings_array, fallback_k)
    n = min(n, len(filtered))

    # Cluster and select
    selected_indices = _cluster_and_select(embeddings_array, n)

    # Build result sorted by document position
    results = []
    for idx in selected_indices:
        sent = filtered[idx]
        results.append(
            KeySentence(
                text=sent["text"],
                source_file=sent["source_file"],
                position=sent["position"],
                score=0.0,  # Updated by _cluster_and_select if needed
            )
        )

    results.sort(key=lambda s: s.position)
    logger.debug("Key sentences: returning %d sentences", len(results))
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
