"""
Tests for the adjusted mean rarity calculator.

Covers compute_adjusted_mean() from src.core.vocabulary.adjusted_mean.
"""

from src.core.vocabulary.adjusted_mean import compute_adjusted_mean


class TestComputeAdjustedMean:
    """Tests for compute_adjusted_mean()."""

    def test_empty_scores_returns_zero(self):
        """Empty input returns 0.0."""
        assert compute_adjusted_mean([], 0.10) == 0.0

    def test_single_score_above_floor(self):
        """Single score above floor returns that score."""
        assert compute_adjusted_mean([0.75], 0.10) == 0.75

    def test_single_score_below_floor_falls_back(self):
        """Single score below floor falls back to full mean."""
        assert compute_adjusted_mean([0.05], 0.10) == 0.05

    def test_filters_common_words(self):
        """Scores below floor are excluded from the mean."""
        # "rare" (0.75) + 3 common words — only rare word counts
        result = compute_adjusted_mean([0.75, 0.00003, 0.000003, 0.05], 0.10)
        assert result == 0.75

    def test_all_below_floor_falls_back_to_full_mean(self):
        """When all scores are below floor, returns full mean."""
        scores = [0.01, 0.02, 0.03]
        expected = sum(scores) / len(scores)
        assert compute_adjusted_mean(scores, 0.10) == expected

    def test_all_above_floor(self):
        """When all scores are above floor, returns normal mean."""
        assert abs(compute_adjusted_mean([0.5, 0.6, 0.7], 0.10) - 0.6) < 1e-9

    def test_separate_filter_scores(self):
        """Filter by linear scores, average log-transformed scores."""
        linear = [0.75, 0.00003, 0.000003]
        log = [0.90, 0.25, 0.10]
        # Only linear[0] >= 0.10, so only log[0] is averaged
        result = compute_adjusted_mean(log, 0.10, filter_scores=linear)
        assert result == 0.90

    def test_filter_scores_all_below_floor(self):
        """Separate filter scores all below floor -> full mean of scores."""
        linear = [0.01, 0.02]
        log = [0.5, 0.6]
        result = compute_adjusted_mean(log, 0.10, filter_scores=linear)
        assert result == 0.55

    def test_zero_floor_includes_everything(self):
        """Floor of 0.0 includes all scores."""
        scores = [0.1, 0.2, 0.3]
        expected = sum(scores) / len(scores)
        assert abs(compute_adjusted_mean(scores, 0.0) - expected) < 1e-9

    def test_mixed_scores(self):
        """Mix of above and below floor scores."""
        scores = [0.8, 0.05, 0.9, 0.01]
        # Only 0.8 and 0.9 are >= 0.10
        expected = (0.8 + 0.9) / 2
        assert abs(compute_adjusted_mean(scores, 0.10) - expected) < 1e-9
