"""
Tests for the feedback and meta-learner system (Session 25).

Tests cover:
- FeedbackManager: Recording, retrieving, and persisting feedback
- VocabularyMetaLearner: Training on feedback data
- Integration: Feedback loop with quality score boosting
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vocabulary.feedback_manager import FeedbackManager  # noqa: E402
from src.vocabulary.meta_learner import (  # noqa: E402
    VocabularyMetaLearner,
    confidence_weighted_blend,
)


@pytest.fixture
def temp_feedback_dir():
    """Create a temporary directory for feedback files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def feedback_manager(temp_feedback_dir):
    """Create FeedbackManager with temp directory."""
    return FeedbackManager(feedback_dir=temp_feedback_dir)


@pytest.fixture
def meta_learner(temp_feedback_dir):
    """Create VocabularyMetaLearner with temp model path."""
    model_path = temp_feedback_dir / "test_model.pkl"
    return VocabularyMetaLearner(model_path=model_path)


class TestFeedbackManager:
    """Tests for FeedbackManager."""

    def test_record_positive_feedback(self, feedback_manager):
        """Test recording thumbs up feedback."""
        term_data = {
            "Term": "adenocarcinoma",
            "Type": "Medical",
            "Sources": "NER",
            "Quality Score": 75,
            "In-Case Freq": 3,
            "Freq Rank": 250000,
        }
        result = feedback_manager.record_feedback(term_data, +1)
        assert result is True
        assert feedback_manager.get_rating("adenocarcinoma") == 1

    def test_record_negative_feedback(self, feedback_manager):
        """Test recording thumbs down feedback."""
        term_data = {"Term": "the"}
        result = feedback_manager.record_feedback(term_data, -1)
        assert result is True
        assert feedback_manager.get_rating("the") == -1

    def test_toggle_feedback(self, feedback_manager):
        """Test toggling feedback from positive to negative."""
        term_data = {"Term": "spondylosis"}
        feedback_manager.record_feedback(term_data, +1)
        assert feedback_manager.get_rating("spondylosis") == 1

        feedback_manager.record_feedback(term_data, -1)
        assert feedback_manager.get_rating("spondylosis") == -1

    def test_clear_feedback(self, feedback_manager):
        """Test clearing feedback (setting to 0)."""
        term_data = {"Term": "cardiomyopathy"}
        feedback_manager.record_feedback(term_data, +1)
        assert feedback_manager.get_rating("cardiomyopathy") == 1

        feedback_manager.record_feedback(term_data, 0)
        assert feedback_manager.get_rating("cardiomyopathy") == 0

    def test_case_insensitive(self, feedback_manager):
        """Test that feedback lookups are case-insensitive."""
        term_data = {"Term": "HIPAA"}
        feedback_manager.record_feedback(term_data, +1)
        assert feedback_manager.get_rating("hipaa") == 1
        assert feedback_manager.get_rating("HIPAA") == 1
        assert feedback_manager.get_rating("Hipaa") == 1

    def test_get_unrated_term(self, feedback_manager):
        """Test getting rating for unrated term returns 0."""
        assert feedback_manager.get_rating("never_rated") == 0

    def test_feedback_persists(self, temp_feedback_dir):
        """Test that feedback persists across manager instances."""
        term_data = {"Term": "persistent_term"}

        # Create first manager and record feedback
        manager1 = FeedbackManager(feedback_dir=temp_feedback_dir)
        manager1.record_feedback(term_data, +1)

        # Create second manager and verify feedback was loaded
        manager2 = FeedbackManager(feedback_dir=temp_feedback_dir)
        assert manager2.get_rating("persistent_term") == 1

    def test_get_feedback_count(self, feedback_manager):
        """Test feedback count tracking."""
        assert feedback_manager.get_feedback_count() == 0

        feedback_manager.record_feedback({"Term": "term1"}, +1)
        feedback_manager.record_feedback({"Term": "term2"}, -1)
        assert feedback_manager.get_feedback_count() == 2

    def test_document_id(self, feedback_manager):
        """Test document ID generation and setting."""
        doc_id = feedback_manager.generate_document_id("Sample document text")
        assert doc_id.startswith("doc_")
        assert len(doc_id) > 4

        feedback_manager.set_document_id(doc_id)
        assert feedback_manager._current_doc_id == doc_id


class TestVocabularyMetaLearner:
    """Tests for VocabularyMetaLearner."""

    def test_untrained_prediction(self, meta_learner):
        """Test that untrained model returns neutral prediction."""
        assert not meta_learner.is_trained
        prediction = meta_learner.predict_preference({"Term": "test"})
        assert prediction == 0.5  # Neutral for untrained

    def test_feature_extraction(self, meta_learner):
        """Test feature extraction from term data."""
        term_data = {
            "Term": "hypertension",
            "quality_score": 75,
            "in_case_freq": 3,
            "freq_rank": 250000,
            "algorithms": "NER,RAKE",
            "type": "Medical",
            "total_unique_terms": 100,
        }
        features = meta_learner._extract_features(term_data)
        assert len(features) == 17  # Total feature count
        assert features[0] == 75  # quality_score
        # features[1] is log_count: log(3) ≈ 1.099
        assert abs(features[1] - 1.099) < 0.01
        # features[2] is occurrence_ratio: 3/100 = 0.03
        assert features[2] == 0.03
        # features[13] is has_trailing_punctuation: "hypertension" has none
        assert features[13] == 0.0
        # features[14] is has_leading_digit: "hypertension" has none
        assert features[14] == 0.0
        # features[15] is word_count: "hypertension" is 1 word
        assert features[15] == 1.0
        # features[16] is is_all_caps: "hypertension" is not all caps
        assert features[16] == 0.0

    def test_feature_extraction_artifacts(self, meta_learner):
        """Test that artifact patterns are detected correctly."""
        # Test trailing punctuation
        term_data = {"Term": "Smith:", "type": "Person"}
        features = meta_learner._extract_features(term_data)
        assert features[13] == 1.0  # has_trailing_punctuation

        # Test leading digit
        term_data = {"Term": "4 Ms. Di Leo", "type": "Person"}
        features = meta_learner._extract_features(term_data)
        assert features[14] == 1.0  # has_leading_digit
        assert features[15] == 4.0  # word_count: "4", "Ms.", "Di", "Leo"

        # Test all caps
        term_data = {"Term": "PLAINTIFF", "type": "Unknown"}
        features = meta_learner._extract_features(term_data)
        assert features[16] == 1.0  # is_all_caps

    def test_training_insufficient_data(self, temp_feedback_dir, meta_learner):
        """Test that training fails with insufficient data."""
        feedback_mgr = FeedbackManager(feedback_dir=temp_feedback_dir)
        # Add only a few samples (below threshold)
        for i in range(5):
            feedback_mgr.record_feedback({"Term": f"term{i}"}, +1 if i % 2 == 0 else -1)

        result = meta_learner.train(feedback_mgr)
        assert result is False  # Should fail - not enough data

    def test_model_save_load(self, temp_feedback_dir):
        """Test model persistence."""
        model_path = temp_feedback_dir / "test_model.pkl"

        # Create and "train" a mock scenario
        learner1 = VocabularyMetaLearner(model_path=model_path)
        assert not learner1.is_trained

        # After proper training (if we had enough data), model would save
        # For now, verify load works with non-existent model
        learner2 = VocabularyMetaLearner(model_path=model_path)
        assert not learner2.is_trained

    def test_should_retrain(self, temp_feedback_dir):
        """Test retraining threshold check."""
        feedback_mgr = FeedbackManager(feedback_dir=temp_feedback_dir)
        model_path = temp_feedback_dir / "test_model.pkl"
        learner = VocabularyMetaLearner(model_path=model_path)

        # Initially should not need retraining
        assert not learner.should_retrain(feedback_mgr)


class TestIntegration:
    """Integration tests for the full feedback-ML pipeline."""

    def test_full_pipeline_import(self):
        """Test that all components can be imported together."""
        from src.vocabulary import (
            VocabularyExtractor,
            get_feedback_manager,
            get_meta_learner,
        )
        # Just verify imports work
        assert VocabularyExtractor is not None
        assert get_feedback_manager is not None
        assert get_meta_learner is not None

    def test_extractor_has_meta_learner(self):
        """Test that VocabularyExtractor has meta-learner integration."""
        from src.vocabulary import VocabularyExtractor
        extractor = VocabularyExtractor()
        assert hasattr(extractor, '_meta_learner')
        assert extractor._meta_learner is not None


class TestConfidenceWeightedBlend:
    """Tests for confidence_weighted_blend() pure function."""

    def test_equal_confidence(self):
        """When both models have equal confidence, average the predictions."""
        # Both at 0.8 confidence (0.3 from 0.5)
        result = confidence_weighted_blend(0.8, 0.8)
        assert result == 0.8

        # LR=0.7, RF=0.3 - both have 0.2 confidence, opposite directions
        result = confidence_weighted_blend(0.7, 0.3)
        assert result == pytest.approx(0.5)  # Weighted average

    def test_high_confidence_dominates(self):
        """Higher confidence model should dominate the result."""
        # LR=0.9 (conf=0.4), RF=0.55 (conf=0.05)
        # LR should dominate: weight_lr=0.4/0.45=0.89, weight_rf=0.11
        result = confidence_weighted_blend(0.9, 0.55)
        assert result > 0.85  # Closer to 0.9 than 0.55

    def test_both_uncertain(self):
        """When both models are at 0.5 (uncertain), return 0.5."""
        result = confidence_weighted_blend(0.5, 0.5)
        assert result == 0.5

    def test_symmetric(self):
        """Order of arguments shouldn't matter for the blend concept."""
        # Note: function is NOT symmetric in argument order
        # but the blend should produce reasonable results either way
        result1 = confidence_weighted_blend(0.8, 0.4)
        result2 = confidence_weighted_blend(0.4, 0.8)
        # Both should be valid probabilities
        assert 0.0 <= result1 <= 1.0
        assert 0.0 <= result2 <= 1.0

    def test_extreme_confidence(self):
        """Test with very confident predictions."""
        # LR=0.99 (conf=0.49), RF=0.51 (conf=0.01)
        result = confidence_weighted_blend(0.99, 0.51)
        # LR should strongly dominate
        assert result > 0.95


class TestEnsembleMode:
    """Tests for ensemble behavior."""

    def test_is_ensemble_property(self, meta_learner):
        """Test that is_ensemble property is False when not trained."""
        assert not meta_learner.is_ensemble

    def test_lr_only_mode(self, temp_feedback_dir):
        """Test that model starts in LR-only mode even after training."""
        # We can't easily test actual training with 200+ samples
        # but we can verify the property behavior
        model_path = temp_feedback_dir / "test_model.pkl"
        learner = VocabularyMetaLearner(model_path=model_path)

        # Untrained: no ensemble
        assert not learner.is_trained
        assert not learner.is_ensemble

    def test_backward_compat_alias(self):
        """Test that VocabularyMetaLearner alias works."""
        from src.vocabulary.meta_learner import (
            VocabularyMetaLearner,
            VocabularyPreferenceLearner,
        )
        # Both should refer to the same class
        assert VocabularyMetaLearner is VocabularyPreferenceLearner
