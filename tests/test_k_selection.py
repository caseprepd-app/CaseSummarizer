"""
Tests for 3-voter K selection ensemble.

Covers: vote resolution, individual voters, integration with embeddings,
edge cases (few sentences, unanimous/split votes).
"""

import numpy as np
import pytest

from src.core.summarization.k_selection import (
    _resolve_votes,
    _run_kmeans_sweep,
    _vote_bic,
    _vote_calinski,
    _vote_silhouette,
    select_k,
)

# =========================================================================
# _resolve_votes
# =========================================================================


class TestResolveVotes:
    """Tests for the voting tiebreaker logic."""

    def test_unanimous(self):
        """All three agree → that value wins."""
        assert _resolve_votes([5, 5, 5]) == 5

    def test_majority_first_two(self):
        """Two agree, one disagrees → majority wins."""
        assert _resolve_votes([3, 3, 7]) == 3

    def test_majority_last_two(self):
        """Two agree, one disagrees → majority wins."""
        assert _resolve_votes([9, 4, 4]) == 4

    def test_majority_first_and_last(self):
        """Two agree, one disagrees → majority wins."""
        assert _resolve_votes([6, 2, 6]) == 6

    def test_all_disagree_picks_median(self):
        """All three differ → pick the middle value."""
        assert _resolve_votes([3, 7, 5]) == 5

    def test_all_disagree_already_sorted(self):
        """Median works regardless of input order."""
        assert _resolve_votes([2, 4, 10]) == 4

    def test_all_disagree_reverse_sorted(self):
        """Median works regardless of input order."""
        assert _resolve_votes([10, 4, 2]) == 4


# =========================================================================
# select_k — edge cases
# =========================================================================


class TestSelectKEdgeCases:
    """Tests for fallback behavior with few sentences."""

    def test_too_few_sentences_uses_fallback(self):
        """Below 10 sentences, uses fallback K."""
        embeddings = np.random.randn(8, 384).astype(np.float32)
        result = select_k(embeddings, fallback_k=5)
        assert result == 5

    def test_fallback_capped_to_sentence_count(self):
        """Fallback K can't exceed number of sentences."""
        embeddings = np.random.randn(3, 384).astype(np.float32)
        result = select_k(embeddings, fallback_k=10)
        assert result == 3

    def test_returns_int(self):
        """Result should always be an integer."""
        embeddings = np.random.randn(30, 384).astype(np.float32)
        result = select_k(embeddings, fallback_k=5)
        assert isinstance(result, int)

    def test_result_within_bounds(self):
        """Result should be between 2 and 15."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 384).astype(np.float32)
        result = select_k(embeddings, fallback_k=5)
        assert 2 <= result <= 15


# =========================================================================
# select_k — with distinct clusters
# =========================================================================


class TestSelectKWithClusters:
    """Tests with synthetic data that has clear cluster structure."""

    def _make_clustered_data(self, n_clusters, per_cluster=20, dim=384):
        """Create embeddings with clear cluster structure."""
        np.random.seed(42)
        clusters = []
        for i in range(n_clusters):
            center = np.zeros(dim)
            center[i % dim] = 10.0  # Offset each cluster
            points = np.random.randn(per_cluster, dim).astype(np.float32) * 0.5
            points += center
            clusters.append(points)
        return np.vstack(clusters)

    def test_finds_clear_clusters(self):
        """With 3 well-separated clusters, should pick K near 3."""
        embeddings = self._make_clustered_data(3)
        result = select_k(embeddings, fallback_k=5)
        # Should be close to 3 (within 1 is acceptable)
        assert 2 <= result <= 4

    def test_finds_more_clusters(self):
        """With 5 well-separated clusters, should pick K near 5."""
        embeddings = self._make_clustered_data(5)
        result = select_k(embeddings, fallback_k=5)
        assert 4 <= result <= 6


# =========================================================================
# Individual voters
# =========================================================================


class TestIndividualVoters:
    """Tests that each voter returns a valid K."""

    @pytest.fixture()
    def embeddings_and_cache(self):
        """Create test embeddings and KMeans cache."""
        np.random.seed(42)
        embeddings = np.random.randn(30, 384).astype(np.float32)
        k_range = range(2, 8)
        cache = _run_kmeans_sweep(embeddings, k_range)
        return embeddings, k_range, cache

    def test_silhouette_returns_valid_k(self, embeddings_and_cache):
        """Silhouette voter returns K in range."""
        embeddings, _, cache = embeddings_and_cache
        result = _vote_silhouette(embeddings, cache)
        assert result in cache

    def test_bic_returns_valid_k(self, embeddings_and_cache):
        """BIC voter returns K in range."""
        embeddings, k_range, _ = embeddings_and_cache
        result = _vote_bic(embeddings, k_range)
        assert result in k_range

    def test_calinski_returns_valid_k(self, embeddings_and_cache):
        """Calinski-Harabasz voter returns K in range."""
        embeddings, _, cache = embeddings_and_cache
        result = _vote_calinski(embeddings, cache)
        assert result in cache

    def test_kmeans_cache_has_correct_keys(self, embeddings_and_cache):
        """KMeans sweep cache should have all K values."""
        _, k_range, cache = embeddings_and_cache
        assert set(cache.keys()) == set(k_range)

    def test_kmeans_cache_labels_shape(self, embeddings_and_cache):
        """Each cached label array should match sentence count."""
        embeddings, _, cache = embeddings_and_cache
        for labels in cache.values():
            assert len(labels) == len(embeddings)
