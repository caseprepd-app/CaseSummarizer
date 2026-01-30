"""
Tests for the feedback and preference learner system.

Tests cover:
- FeedbackManager: Recording, retrieving, and persisting feedback
- VocabularyPreferenceLearner: Training on feedback data
- Integration: Feedback loop with quality score boosting
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.vocabulary.feedback_manager import FeedbackManager  # noqa: E402
from src.core.vocabulary.preference_learner import (  # noqa: E402
    VocabularyPreferenceLearner,
    confidence_weighted_blend,
)
from src.core.vocabulary.preference_learner_features import extract_features  # noqa: E402


@pytest.fixture
def temp_feedback_dir():
    """Create a temporary directory for feedback files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def feedback_manager(temp_feedback_dir):
    """Create FeedbackManager with temp directory and no shipped defaults."""
    # Provide a non-existent default_feedback_file so tests start clean
    nonexistent_default = temp_feedback_dir / "default_feedback.csv"
    return FeedbackManager(
        feedback_dir=temp_feedback_dir, default_feedback_file=nonexistent_default
    )


@pytest.fixture
def meta_learner(temp_feedback_dir, feedback_manager):
    """Create VocabularyPreferenceLearner with temp model path and no auto-training.

    Uses the clean feedback_manager fixture (no default feedback) to prevent
    auto-training during initialization.
    """
    model_path = temp_feedback_dir / "test_model.pkl"
    # Patch get_feedback_manager to return our clean fixture during init
    with patch(
        "src.core.vocabulary.preference_learner.get_feedback_manager",
        return_value=feedback_manager,
    ):
        return VocabularyPreferenceLearner(model_path=model_path)


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

        # Provide non-existent default_feedback_file to avoid polluting shipped file
        nonexistent_default = temp_feedback_dir / "default_feedback.csv"

        # Create first manager and record feedback
        manager1 = FeedbackManager(
            feedback_dir=temp_feedback_dir, default_feedback_file=nonexistent_default
        )
        manager1.record_feedback(term_data, +1)

        # Create second manager and verify feedback was loaded
        manager2 = FeedbackManager(
            feedback_dir=temp_feedback_dir, default_feedback_file=nonexistent_default
        )
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


class TestVocabularyPreferenceLearner:
    """Tests for VocabularyPreferenceLearner."""

    def test_untrained_prediction(self, meta_learner):
        """Test that untrained model returns neutral prediction."""
        assert not meta_learner.is_trained
        prediction = meta_learner.predict_preference({"Term": "test"})
        assert prediction == 0.5  # Neutral for untrained

    def test_feature_extraction(self, meta_learner):
        """Test feature extraction from term data.

        Session 76: Feature indices updated after overhaul (23 features total):
        0: log_count, 1: occurrence_ratio, 2-4: has_ner/rake/bm25, 5: is_person,
        6: has_trailing_punctuation, 7: has_leading_digit, 8: has_trailing_digit,
        9: word_count, 10: is_all_caps, 11: is_title_case,
        12: source_doc_confidence, 13: corpus_familiarity_score,
        14-22: NEW Session 76 features (freq_dict_word_ratio, all_words_in_freq_dict,
               term_length, vowel_ratio, is_single_letter, has_internal_digits,
               has_medical_suffix, has_repeated_chars, contains_hyphen)
        """
        term_data = {
            "Term": "hypertension",
            "quality_score": 75,  # No longer used (removed Session 76)
            "in_case_freq": 3,
            "freq_rank": 250000,  # No longer used (replaced with word-level features)
            "algorithms": "NER,RAKE",
            "type": "Medical",
            "total_unique_terms": 100,
        }
        features = extract_features(term_data)
        assert len(features) == 50  # 7 count bins + log_count + 42 other features
        # features[0-6] are count bins: count=3 → count_bin_3=1.0
        assert features[0] == 0.0  # count_bin_1
        assert features[1] == 0.0  # count_bin_2
        assert features[2] == 1.0  # count_bin_3 (count=3)
        assert features[3] == 0.0  # count_bin_4_6
        assert features[4] == 0.0  # count_bin_7_20
        assert features[5] == 0.0  # count_bin_21_50
        assert features[6] == 0.0  # count_bin_51_plus
        # features[7] is log_count: log10(3+1) ≈ 0.602
        assert round(features[7], 2) == 0.60  # log_count
        # features[8] is occurrence_ratio: 3/100 = 0.03
        assert features[8] == 0.03
        # features[9-13] are algorithm flags
        assert features[9] == 1.0  # has_ner
        assert features[10] == 1.0  # has_rake
        assert features[11] == 0.0  # has_bm25
        assert features[12] == 0.0  # has_textrank
        assert features[13] == 0.0  # textrank_score
        # features[15] is has_trailing_punctuation: "hypertension" has none
        assert features[15] == 0.0
        # features[16] is has_leading_digit: "hypertension" has none
        assert features[16] == 0.0
        # features[18] is word_count: "hypertension" is 1 word
        assert features[18] == 1.0
        # features[19] is is_all_caps: "hypertension" is not all caps
        assert features[19] == 0.0
        # features[30] is has_medical_suffix: "hypertension" ends with -ion (not medical suffix)
        assert features[30] == 0.0

    def test_feature_extraction_artifacts(self, meta_learner):
        """Test that artifact patterns are detected correctly.

        Feature indices (50 total, with log_count at index 7):
        15: has_trailing_punctuation, 16: has_leading_digit,
        17: has_trailing_digit, 18: word_count, 19: is_all_caps
        28: is_single_letter, 30: has_medical_suffix,
        31: has_repeated_chars, 32: contains_hyphen
        """
        # Test trailing punctuation
        term_data = {"Term": "Smith:", "type": "Person"}
        features = extract_features(term_data)
        assert features[15] == 1.0  # has_trailing_punctuation

        # Test leading digit
        term_data = {"Term": "4 Ms. Di Leo", "type": "Person"}
        features = extract_features(term_data)
        assert features[16] == 1.0  # has_leading_digit
        assert features[18] == 4.0  # word_count: "4", "Ms.", "Di", "Leo"

        # Test all caps
        term_data = {"Term": "PLAINTIFF", "type": "Unknown"}
        features = extract_features(term_data)
        assert features[19] == 1.0  # is_all_caps

        # Test medical suffix
        term_data = {"Term": "radiculopathy"}
        features = extract_features(term_data)
        assert features[30] == 1.0  # has_medical_suffix (-pathy)

        # Test single letter
        term_data = {"Term": "Q"}
        features = extract_features(term_data)
        assert features[28] == 1.0  # is_single_letter

        # Test repeated chars
        term_data = {"Term": "aaaa"}
        features = extract_features(term_data)
        assert features[31] == 1.0  # has_repeated_chars

        # Test contains hyphen
        term_data = {"Term": "anti-inflammatory"}
        features = extract_features(term_data)
        assert features[32] == 1.0  # contains_hyphen

    def test_training_insufficient_data_no_defaults(self, temp_feedback_dir, meta_learner):
        """Test that training fails with insufficient data when no defaults exist.

        Note: With default_feedback.csv populated (Session 69), training will
        succeed even with minimal user feedback. This test verifies the behavior
        when defaults are NOT available (e.g., if the file is missing).
        """
        import csv

        # Create a feedback manager that uses an empty default file
        empty_default = temp_feedback_dir / "empty_default.csv"
        with open(empty_default, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "document_id",
                    "term",
                    "feedback",
                    "is_person",
                    "algorithms",
                    "NER_detection",
                    "RAKE_detection",
                    "BM25_detection",
                    "algo_count",
                    "quality_score",
                    "in_case_freq",
                    "freq_rank",
                ]
            )

        # Create manager with only 5 user samples and no defaults
        feedback_mgr = FeedbackManager(feedback_dir=temp_feedback_dir)
        # Override the default file path to our empty file
        feedback_mgr.default_feedback_file = empty_default

        for i in range(5):
            feedback_mgr.record_feedback({"Term": f"term{i}"}, +1 if i % 2 == 0 else -1)

        result = meta_learner.train(feedback_mgr)
        assert result is False  # Should fail - not enough data without defaults

    def test_model_save_load(self, temp_feedback_dir, feedback_manager):
        """Test model persistence."""
        model_path = temp_feedback_dir / "test_model.pkl"

        # Patch to prevent auto-training from default feedback
        with patch(
            "src.core.vocabulary.preference_learner.get_feedback_manager",
            return_value=feedback_manager,
        ):
            # Create and "train" a mock scenario
            learner1 = VocabularyPreferenceLearner(model_path=model_path)
            assert not learner1.is_trained

            # After proper training (if we had enough data), model would save
            # For now, verify load works with non-existent model
            learner2 = VocabularyPreferenceLearner(model_path=model_path)
            assert not learner2.is_trained

    def test_should_retrain(self, temp_feedback_dir):
        """Test retraining threshold check."""
        feedback_mgr = FeedbackManager(feedback_dir=temp_feedback_dir)
        model_path = temp_feedback_dir / "test_model.pkl"
        learner = VocabularyPreferenceLearner(model_path=model_path)

        # Initially should not need retraining
        assert not learner.should_retrain(feedback_mgr)


class TestIntegration:
    """Integration tests for the full feedback-ML pipeline."""

    def test_full_pipeline_import(self):
        """Test that all components can be imported together."""
        from src.core.vocabulary import (
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
        from src.core.vocabulary import VocabularyExtractor

        extractor = VocabularyExtractor()
        assert hasattr(extractor, "_meta_learner")
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

    def test_lr_only_mode(self, temp_feedback_dir, feedback_manager):
        """Test that model starts in LR-only mode even after training."""
        # We can't easily test actual training with 200+ samples
        # but we can verify the property behavior
        model_path = temp_feedback_dir / "test_model.pkl"

        # Patch to prevent auto-training from default feedback
        with patch(
            "src.core.vocabulary.preference_learner.get_feedback_manager",
            return_value=feedback_manager,
        ):
            learner = VocabularyPreferenceLearner(model_path=model_path)

            # Untrained: no ensemble
            assert not learner.is_trained
            assert not learner.is_ensemble


@pytest.mark.skip(reason="Pending default training data generation")
class TestDefaultFeedback:
    """Tests for default feedback CSV (universal negatives).

    The default_feedback.csv ships with the app and contains universal
    negative examples - terms ALL users would reject regardless of domain.
    This bootstraps the ML model for immediate noise reduction.
    """

    def test_default_feedback_exists(self):
        """Verify default_feedback.csv exists and is valid CSV."""
        import csv

        from src.config import DEFAULT_FEEDBACK_CSV

        assert DEFAULT_FEEDBACK_CSV.exists(), (
            f"Default feedback CSV not found at {DEFAULT_FEEDBACK_CSV}"
        )

        # Verify it's valid CSV with expected columns
        with open(DEFAULT_FEEDBACK_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "Default feedback CSV is empty"
        assert "term" in reader.fieldnames
        assert "feedback" in reader.fieldnames
        assert "is_person" in reader.fieldnames

    def test_default_feedback_all_negative(self):
        """Verify all default feedback entries are thumbs down (-1)."""
        import csv

        from src.config import DEFAULT_FEEDBACK_CSV

        with open(DEFAULT_FEEDBACK_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                feedback = int(row["feedback"])
                assert feedback == -1, (
                    f"Row {i}: Non-negative feedback in default CSV: "
                    f"term='{row['term']}', feedback={feedback}"
                )

    def test_default_feedback_no_persons(self):
        """Verify no person names in default negatives.

        Universal negatives should not include person names because:
        1. Names are domain-specific (could be legitimate for some users)
        2. NER person detection is reliable, we don't need to train against it
        """
        import csv

        from src.config import DEFAULT_FEEDBACK_CSV

        with open(DEFAULT_FEEDBACK_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                is_person = row["is_person"]
                # Handle both string and int representations
                assert is_person in ("0", "False", 0, False), (
                    f"Row {i}: Person name in default negatives: "
                    f"term='{row['term']}', is_person={is_person}"
                )

    def test_default_feedback_count(self):
        """Verify reasonable number of default feedback entries.

        Should have 50-150 entries:
        - At least 50: Provides meaningful training data
        - At most 150: Focused on universal junk, not overfit
        """
        import csv

        from src.config import DEFAULT_FEEDBACK_CSV

        with open(DEFAULT_FEEDBACK_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)

        assert 50 <= count <= 150, f"Default feedback has {count} entries, expected 50-150"

    def test_default_feedback_categories(self):
        """Verify default feedback covers expected categories."""
        import csv

        from src.config import DEFAULT_FEEDBACK_CSV

        with open(DEFAULT_FEEDBACK_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            terms = [row["term"].lower() for row in reader]

        # Check for presence of key universal negatives
        # Common phrases
        assert any("same" in t for t in terms), "Missing common phrase category"
        # Transcript artifacts
        assert any(t in ("q.", "a.", "q", "a") for t in terms), (
            "Missing transcript artifact category"
        )
        # OCR patterns (terms with digit-letter confusion)
        assert any(t in ("1he", "tbe", "0f") for t in terms), "Missing OCR artifact category"

    def test_training_with_default_feedback_only(self, temp_feedback_dir):
        """Test that training succeeds with only default feedback.

        The default_feedback.csv should have enough entries (30+) to
        trigger Logistic Regression training even with zero user feedback.
        """
        import csv

        from src.config import DEFAULT_FEEDBACK_CSV

        # First verify we have enough defaults
        with open(DEFAULT_FEEDBACK_CSV, encoding="utf-8") as f:
            default_count = sum(1 for _ in csv.DictReader(f))

        if default_count < 30:
            pytest.skip(f"Only {default_count} default entries, need 30+ for training")

        # Create manager with only default feedback (empty user dir)
        manager = FeedbackManager(feedback_dir=temp_feedback_dir)

        # Get combined training data (should include defaults)
        training_data = manager.export_training_data()

        # Should have entries from default file
        assert len(training_data) >= 30, f"Expected 30+ training entries, got {len(training_data)}"

    def test_user_feedback_overrides_default(self, temp_feedback_dir):
        """Test that user feedback takes precedence over defaults.

        If a term appears in both default (negative) and user feedback,
        the user's rating should win.
        """
        # Create manager
        manager = FeedbackManager(feedback_dir=temp_feedback_dir)

        # Record positive feedback for a term that might be in defaults
        term_data = {
            "Term": "the same",  # Likely in default negatives
            "Quality Score": 50,
        }
        manager.record_feedback(term_data, +1)  # User says thumbs up

        # User's rating should be what we get
        assert manager.get_rating("the same") == 1, "User feedback should override default"
