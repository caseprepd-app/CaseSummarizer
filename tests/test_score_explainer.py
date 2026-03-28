"""
Tests for score_explainer.py — per-term score explanations.

Tests the explain_score() function and its component extractors:
- LR contributions (coefficient x scaled value)
- RF contributions (importance x |scaled value|, when ensemble active)
- Rules contributions (which quality rules fired)
- Merge/dedup logic across all three
"""

from unittest.mock import MagicMock, patch

import numpy as np

from src.core.vocabulary.preference_learner_features import FEATURE_NAMES

_N_FEATURES = len(FEATURE_NAMES)


def _make_term_data(term="Dr. Smith", occurrences=10, algorithms="ner bm25", is_person=1):
    """Create a minimal term data dict for testing."""
    return {
        "Term": term,
        "occurrences": occurrences,
        "algorithms": algorithms,
        "is_person": is_person,
        "total_word_count": 5000,
        "source_doc_confidence": 95,
        "rarity_rank": 0,
    }


def _make_mock_learner(is_trained=True, is_ensemble=False, n_features=_N_FEATURES):
    """Create a mock learner with LR model (and optionally RF)."""
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

        # Mock RF model (only used when is_ensemble=True)
        if is_ensemble:
            rf_model = MagicMock()
            rf_model.feature_importances_ = np.random.rand(n_features)
            rf_model.feature_importances_ /= rf_model.feature_importances_.sum()
            learner._rf_model = rf_model
        else:
            learner._rf_model = None
    else:
        learner._lr_model = None
        learner._scaler = None
        learner._rf_model = None

    return learner


# ---------------------------------------------------------------------------
# Main explain_score function
# ---------------------------------------------------------------------------


class TestExplainScore:
    """Tests for the explain_score orchestrator."""

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
    def test_reasons_are_3_tuples(self, mock_get_learner):
        """Each reason is a (label, value, source) 3-tuple."""
        from src.core.vocabulary.score_explainer import explain_score

        mock_get_learner.return_value = _make_mock_learner()
        result = explain_score(_make_term_data())

        for reason in result["reasons"]:
            assert len(reason) == 3
            label, value, source = reason
            assert isinstance(label, str)
            assert isinstance(value, float)
            assert source in ("LR", "RF", "Rules")

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_reasons_include_rules_source(self, mock_get_learner):
        """At least one reason should come from Rules (term has rarity_rank=0)."""
        from src.core.vocabulary.score_explainer import explain_score

        mock_get_learner.return_value = _make_mock_learner()
        result = explain_score(_make_term_data())

        sources = {reason[2] for reason in result["reasons"]}
        assert "Rules" in sources

    @patch("src.core.vocabulary.score_explainer.get_meta_learner")
    def test_reasons_include_lr_source(self, mock_get_learner):
        """At least one reason should come from LR."""
        from src.core.vocabulary.score_explainer import explain_score

        mock_get_learner.return_value = _make_mock_learner()
        result = explain_score(_make_term_data())

        sources = {reason[2] for reason in result["reasons"]}
        assert "LR" in sources

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
    def test_ensemble_includes_rf_source(self, mock_get_learner):
        """When ensemble is active, reasons should include RF contributions."""
        from src.core.vocabulary.score_explainer import explain_score

        mock_get_learner.return_value = _make_mock_learner(is_ensemble=True)
        result = explain_score(_make_term_data())

        sources = {reason[2] for reason in result["reasons"]}
        assert "RF" in sources


# ---------------------------------------------------------------------------
# Merge / dedup logic
# ---------------------------------------------------------------------------


class TestMergeContributions:
    """Tests for the _merge_contributions dedup logic."""

    def test_dedup_removes_duplicate_feature_keys(self):
        """Same feature_key from different sources is shown only once."""
        from src.core.vocabulary.score_explainer import Contribution, _merge_contributions

        lr = [Contribution("log_count", "Appears frequently", 0.5, "LR")]
        rf = [Contribution("log_count", "Appears frequently", 0.3, "RF")]
        rules = [Contribution("word_log_rarity_score", "Rare word (+20)", 20.0, "Rules")]

        merged = _merge_contributions(lr, rf, rules)
        keys = [c.feature_key for c in merged]

        # log_count should appear only once (from LR, which comes first)
        assert keys.count("log_count") == 1
        assert "word_log_rarity_score" in keys

    def test_interleave_gives_each_source_fair_share(self):
        """All three sources get representation when features are unique."""
        from src.core.vocabulary.score_explainer import Contribution, _merge_contributions

        lr = [
            Contribution("has_ner", "Found by NER", 0.5, "LR"),
            Contribution("log_count", "Frequent", 0.3, "LR"),
        ]
        rf = [
            Contribution("is_person", "Person name", 0.4, "RF"),
            Contribution("has_rake", "Found by RAKE", 0.2, "RF"),
        ]
        rules = [
            Contribution("word_log_rarity_score", "Rare (+20)", 20.0, "Rules"),
            Contribution("_rule_multi_algo", "2 algos (+4)", 4.0, "Rules"),
        ]

        merged = _merge_contributions(lr, rf, rules)
        sources = [c.source for c in merged]

        # All 6 unique features should be present
        assert len(merged) == 6
        # Each source should contribute
        assert "LR" in sources
        assert "RF" in sources
        assert "Rules" in sources

    def test_empty_rf_when_no_ensemble(self):
        """When RF list is empty, only LR and Rules contribute."""
        from src.core.vocabulary.score_explainer import Contribution, _merge_contributions

        lr = [Contribution("has_ner", "Found by NER", 0.5, "LR")]
        rules = [Contribution("log_count", "Appears 10 times (+18)", 18.0, "Rules")]

        merged = _merge_contributions(lr, [], rules)
        sources = {c.source for c in merged}

        assert "RF" not in sources
        assert "LR" in sources
        assert "Rules" in sources


# ---------------------------------------------------------------------------
# Rules evaluator
# ---------------------------------------------------------------------------


class TestEvaluateRules:
    """Tests for the rules-based contribution evaluator."""

    def test_occurrence_boost_fires(self):
        """Terms with multiple occurrences get an occurrence boost."""
        from src.core.vocabulary.score_explainer_rules import evaluate_rules

        results = evaluate_rules(_make_term_data(occurrences=10))
        keys = [r.feature_key for r in results]
        assert "log_count" in keys

    def test_rare_word_boost_fires(self):
        """Terms not in Google dataset get a rarity boost."""
        from src.core.vocabulary.score_explainer_rules import evaluate_rules

        results = evaluate_rules(_make_term_data())
        keys = [r.feature_key for r in results]
        assert "word_log_rarity_score" in keys

    def test_person_name_boost_fires(self):
        """Person names get a person boost rule."""
        from src.core.vocabulary.score_explainer_rules import evaluate_rules

        results = evaluate_rules(_make_term_data(is_person=1))
        keys = [r.feature_key for r in results]
        assert "is_person" in keys

    def test_multi_algo_boost_fires(self):
        """Terms found by 2+ algorithms get multi-algo boost."""
        from src.core.vocabulary.score_explainer_rules import evaluate_rules

        results = evaluate_rules(_make_term_data(algorithms="ner bm25"))
        keys = [r.feature_key for r in results]
        assert "_rule_multi_algo" in keys

    def test_artifact_penalty_all_caps(self):
        """All-caps terms get an artifact penalty."""
        from src.core.vocabulary.score_explainer_rules import evaluate_rules

        results = evaluate_rules(_make_term_data(term="PLAINTIFF"))
        keys = [r.feature_key for r in results]
        assert "is_all_caps" in keys

    def test_artifact_penalty_single_letter(self):
        """Single letter terms get a heavy penalty."""
        from src.core.vocabulary.score_explainer_rules import evaluate_rules

        results = evaluate_rules(_make_term_data(term="Q", is_person=0))
        penalties = [r for r in results if r.points < 0]
        keys = [r.feature_key for r in penalties]
        assert "is_single_letter" in keys

    def test_sorted_by_absolute_points(self):
        """Results are sorted by absolute point value descending."""
        from src.core.vocabulary.score_explainer_rules import evaluate_rules

        results = evaluate_rules(_make_term_data())
        points = [abs(r.points) for r in results]
        assert points == sorted(points, reverse=True)

    def test_trailing_punctuation_penalty(self):
        """Terms ending with punctuation get a penalty."""
        from src.core.vocabulary.score_explainer_rules import evaluate_rules

        results = evaluate_rules(_make_term_data(term="Smith:", is_person=0))
        keys = [r.feature_key for r in results]
        assert "has_trailing_punctuation" in keys


# ---------------------------------------------------------------------------
# VocabularyService passthrough
# ---------------------------------------------------------------------------


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
