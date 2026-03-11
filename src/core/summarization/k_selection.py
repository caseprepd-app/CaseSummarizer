"""
3-Voter K Selection for K-Means Clustering

Determines the optimal number of clusters (K) by having three
independent statistical methods each vote for their preferred K:

1. Silhouette Score — sentence-level fit (closer to own group?)
2. BIC via GMM — model complexity tradeoff (more groups worth it?)
3. Calinski-Harabasz — group-level separation (tight and far apart?)

Majority wins. If all three disagree, the median K is chosen.

Reference: Ullmann et al. (2022) — cluster stability across methods
is more robust than any single metric.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_MIN_K = 2  # Need at least 2 groups for metrics to work
_MAX_K = 15  # More than 15 topics is noise, not signal
_MIN_SENTENCES_FOR_VOTING = 10  # Below this, not enough data for reliable voting


def select_k(embeddings: np.ndarray, fallback_k: int) -> int:
    """
    Select optimal K using 3-voter ensemble.

    Args:
        embeddings: (num_sentences, embedding_dim) float32 array.
        fallback_k: K to use if too few sentences for voting.

    Returns:
        int: The chosen K value (2 to 15).
    """
    n_sentences = len(embeddings)

    if n_sentences < _MIN_SENTENCES_FOR_VOTING:
        logger.debug("K selection: %d sentences, using fallback K=%d", n_sentences, fallback_k)
        return min(fallback_k, n_sentences)

    # Can't have more groups than half the sentences (each group needs 2+ members)
    max_k = min(_MAX_K, n_sentences // 2)
    if max_k < _MIN_K:
        return min(fallback_k, n_sentences)

    # Try every K from 2 to max_k; cache KMeans results so Silhouette
    # and Calinski-Harabasz don't repeat the same clustering work
    k_range = range(_MIN_K, max_k + 1)
    labels_cache = _run_kmeans_sweep(embeddings, k_range)

    # Each voter independently picks its best K
    vote_sil = _vote_silhouette(embeddings, labels_cache)
    vote_bic = _vote_bic(embeddings, k_range)
    vote_ch = _vote_calinski(embeddings, labels_cache)

    votes = [vote_sil, vote_bic, vote_ch]
    chosen = _resolve_votes(votes)

    logger.info(
        "K selection — Silhouette: %d, BIC: %d, Calinski-Harabasz: %d → chosen K=%d",
        vote_sil,
        vote_bic,
        vote_ch,
        chosen,
    )
    return chosen


def _resolve_votes(votes: list[int]) -> int:
    """
    Majority wins. If all three disagree, pick the median.

    Args:
        votes: List of exactly 3 K values.

    Returns:
        int: The winning K.
    """
    # Check if any K got 2+ votes (majority)
    for v in votes:
        if votes.count(v) >= 2:
            return v
    # All three disagree — pick the middle value as a safe compromise
    return sorted(votes)[1]


def _run_kmeans_sweep(embeddings: np.ndarray, k_range: range) -> dict[int, np.ndarray]:
    """
    Run KMeans for each K and cache label arrays.

    Args:
        embeddings: (num_sentences, embedding_dim) array.
        k_range: Range of K values to try.

    Returns:
        Dict mapping K -> cluster labels array.
    """
    from sklearn.cluster import KMeans

    cache = {}
    for k in k_range:
        # n_init=5: run KMeans 5 times per K, keep the best (less than the
        # final clustering's n_init=10, since we're just comparing K values)
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        cache[k] = km.fit_predict(embeddings)
    return cache


def _vote_silhouette(embeddings: np.ndarray, labels_cache: dict[int, np.ndarray]) -> int:
    """
    Pick K with highest silhouette score (cosine metric).

    Args:
        embeddings: (num_sentences, embedding_dim) array.
        labels_cache: Dict mapping K -> cluster labels.

    Returns:
        int: Best K according to silhouette.
    """
    from sklearn.metrics import silhouette_score

    best_k = min(labels_cache)
    best_score = -1.0
    for k, labels in labels_cache.items():
        # Cosine metric: correct for sentence embeddings which live on a
        # hypersphere (direction matters more than magnitude)
        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _vote_bic(embeddings: np.ndarray, k_range: range) -> int:
    """
    Pick K with lowest BIC from Gaussian Mixture Model.

    Uses diagonal covariance to handle high-dimensional embeddings
    without requiring huge sample counts.

    Args:
        embeddings: (num_sentences, embedding_dim) array.
        k_range: Range of K values to try.

    Returns:
        int: Best K according to BIC.
    """
    from sklearn.mixture import GaussianMixture

    best_k = k_range[0]
    best_bic = float("inf")
    for k in k_range:
        # covariance_type="diag": each dimension independent — needed because
        # full covariance would require more sentences than dimensions (384)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=42,
            n_init=3,
        )
        gmm.fit(embeddings)
        # Lower BIC = better model (penalizes extra groups that don't help)
        bic = gmm.bic(embeddings)
        if bic < best_bic:
            best_k, best_bic = k, bic
    return best_k


def _vote_calinski(embeddings: np.ndarray, labels_cache: dict[int, np.ndarray]) -> int:
    """
    Pick K with highest Calinski-Harabasz index.

    Args:
        embeddings: (num_sentences, embedding_dim) array.
        labels_cache: Dict mapping K -> cluster labels.

    Returns:
        int: Best K according to Calinski-Harabasz.
    """
    from sklearn.metrics import calinski_harabasz_score

    best_k = min(labels_cache)
    best_score = -1.0
    for k, labels in labels_cache.items():
        # Higher score = groups are tight internally and far apart from each other
        score = calinski_harabasz_score(embeddings, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k
