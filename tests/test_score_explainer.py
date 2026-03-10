"""
Tests for score_explainer.py — per-term ML score explanations.

Tests the explain_score() function that provides human-readable
explanations for why the ML model scored a term a certain way.
"""

from unittest.mock import MagicMock, patch

import numpy as np


def _make_term_data(term="Dr. Smith", occurrences=10, algorithms="ner bm25", is_person=1):
    """Create a minimal term data dict for testing."""
    return {
        "Term": term,
        "occurrences": occurrences,
        "algorithms": algorithms,
        "is_person": is_person,
        "total_word_count": 5000,
        "source_doc_confidence": 95,
    }


def _make_mock_learner(is_trained=True, is_ensemble=False, n_features=53):
    """Create a mock learner with LR model."""
    learner = MagicMock()
    learner.is_trained = is_trained
    learner.is_ensemble = is_ensemble
    learner.predict_preference.return_value = 0.82

    if is_trained:
        # Mock LR model with coefficients
        lr_model = MagicMock()
        lr_model.coef_ = np.random.randn(1, n_features) * 0.5
        lr_model.intercept_ = np.array([0.1])
        learner._lr_model = lr_model

        # Mock scaler
        scaler = MagicMock()
        scaler.transform.return_value = np.random.randn(1, n_features)
        learner._scaler = scaler
    else:
        learner._lr_model = None
        learner._scaler = None

    return learner


class TestExplainScore:
    """Tests for the explain_score function."""

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_returns_none_when_not_trained(self, mock_get_learner):
        """explain_score returns None when no model is trained."""
        from src.core.vocabulary.score_explainer import explain_score

        mock_get_learner.return_value = _make_mock_learner(is_trained=False)
        result = explain_score(_make_term_data())
        assert result is None

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_returns_dict_when_trained(self, mock_get_learner):
        """explain_score returns a dict with expected keys when model is trained."""
        from src.core.vocabulary.score_explainer import explain_score

        mock_get_learner.return_value = _make_mock_learner()
        result = explain_score(_make_term_data())

        assert result is not None
        assert "score" in result
        assert "direction" in result
        assert "reasons" in result
        assert "model_status" in result

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_direction_keep_when_score_above_half(self, mock_get_learner):
        """Direction is 'keep' when ML score >= 0.5."""
        from src.core.vocabulary.score_explainer import explain_score

        learner = _make_mock_learner()
        learner.predict_preference.return_value = 0.75
        mock_get_learner.return_value = learner

        result = explain_score(_make_term_data())
        assert result["direction"] == "keep"

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_direction_skip_when_score_below_half(self, mock_get_learner):
        """Direction is 'skip' when ML score < 0.5."""
        from src.core.vocabulary.score_explainer import explain_score

        learner = _make_mock_learner()
        learner.predict_preference.return_value = 0.3
        mock_get_learner.return_value = learner

        result = explain_score(_make_term_data())
        assert result["direction"] == "skip"

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_reasons_sorted_by_absolute_contribution(self, mock_get_learner):
        """Reasons are sorted by absolute contribution, highest first."""
        from src.core.vocabulary.score_explainer import explain_score

        mock_get_learner.return_value = _make_mock_learner()
        result = explain_score(_make_term_data())

        if result and result["reasons"]:
            contribs = [abs(c) for _, c in result["reasons"]]
            assert contribs == sorted(contribs, reverse=True)

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_max_reasons_respected(self, mock_get_learner):
        """Number of reasons does not exceed max_reasons."""
        from src.core.vocabulary.score_explainer import explain_score

        mock_get_learner.return_value = _make_mock_learner()
        result = explain_score(_make_term_data(), max_reasons=3)

        assert result is not None
        assert len(result["reasons"]) <= 3

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_model_status_lr_when_no_ensemble(self, mock_get_learner):
        """model_status is 'lr' when ensemble is not enabled."""
        from src.core.vocabulary.score_explainer import explain_score

        mock_get_learner.return_value = _make_mock_learner(is_ensemble=False)
        result = explain_score(_make_term_data())

        assert result["model_status"] == "lr"

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_model_status_ensemble_when_enabled(self, mock_get_learner):
        """model_status is 'ensemble' when ensemble is enabled."""
        from src.core.vocabulary.score_explainer import explain_score

        mock_get_learner.return_value = _make_mock_learner(is_ensemble=True)
        result = explain_score(_make_term_data())

        assert result["model_status"] == "ensemble"

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_reasons_are_tuples_of_str_and_float(self, mock_get_learner):
        """Each reason is a (str, float) tuple."""
        from src.core.vocabulary.score_explainer import explain_score

        mock_get_learner.return_value = _make_mock_learner()
        result = explain_score(_make_term_data())

        for label, contrib in result["reasons"]:
            assert isinstance(label, str)
            assert isinstance(contrib, float)

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_negligible_contributions_filtered(self, mock_get_learner):
        """Contributions with abs value < 0.01 are filtered out."""
        from src.core.vocabulary.score_explainer import explain_score

        learner = _make_mock_learner()
        # Set all coefficients to near-zero
        learner._lr_model.coef_ = np.full((1, 53), 0.001)
        learner._scaler.transform.return_value = np.full((1, 53), 0.001)
        mock_get_learner.return_value = learner

        result = explain_score(_make_term_data())
        assert result is not None
        # All contributions should be ~0.000001, filtered out
        assert len(result["reasons"]) == 0


class TestVocabularyServiceExplain:
    """Tests for VocabularyService.explain_term_score passthrough."""

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_service_delegates_to_explainer(self, mock_get_learner):
        """VocabularyService.explain_term_score delegates to score_explainer."""
        from src.services.vocabulary_service import VocabularyService

        mock_get_learner.return_value = _make_mock_learner()
        service = VocabularyService()
        result = service.explain_term_score(_make_term_data())

        assert result is not None
        assert "score" in result
        assert "reasons" in result

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_service_returns_none_when_untrained(self, mock_get_learner):
        """VocabularyService.explain_term_score returns None when untrained."""
        from src.services.vocabulary_service import VocabularyService

        mock_get_learner.return_value = _make_mock_learner(is_trained=False)
        service = VocabularyService()
        result = service.explain_term_score(_make_term_data())

        assert result is None
