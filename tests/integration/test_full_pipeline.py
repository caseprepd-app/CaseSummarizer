"""
Integration tests for the full vocabulary extraction pipeline.

These tests verify that the complete flow works:
- Text processing -> vocabulary extraction
- ML model training and prediction
- Feature extraction consistency
"""

import pytest


class TestFeatureExtraction:
    """Tests for feature extraction consistency."""

    def test_feature_count_matches_names(self):
        """Verify FEATURE_NAMES matches extracted feature count."""
        from src.core.vocabulary.preference_learner_features import FEATURE_NAMES, extract_features

        # Create a mock term with all required fields
        mock_term = {
            "Term": "TestName",
            "term": "TestName",
            "in_case_freq": 5,
            "algorithms": "NER,RAKE",
            "is_person": 1,
            "source_doc_confidence": 95,
            "corpus_common_term": False,
            "total_unique_terms": 100,
            "total_docs_in_session": 3,
            "timestamp": "2024-01-01T00:00:00",
            "feedback": "+1",
        }

        features = extract_features(mock_term)

        assert len(features) == len(FEATURE_NAMES), (
            f"Feature count mismatch: extracted {len(features)} features "
            f"but FEATURE_NAMES has {len(FEATURE_NAMES)} entries"
        )

    def test_extract_features_requires_term(self):
        """Verify extract_features raises ValueError for missing term."""
        from src.core.vocabulary.preference_learner_features import extract_features

        with pytest.raises(ValueError, match="must contain 'Term' or 'term' key"):
            extract_features({"in_case_freq": 5})

    def test_extract_features_requires_dict(self):
        """Verify extract_features raises ValueError for non-dict input."""
        from src.core.vocabulary.preference_learner_features import extract_features

        with pytest.raises(ValueError, match="must be dict"):
            extract_features("not a dict")

    def test_feature_values_in_valid_range(self):
        """Verify extracted features are in expected ranges."""
        from src.core.vocabulary.preference_learner_features import FEATURE_NAMES, extract_features

        mock_term = {
            "Term": "cervical radiculopathy",
            "in_case_freq": 10,
            "algorithms": "NER,RAKE,BM25",
            "is_person": 0,
            "source_doc_confidence": 85,
            "total_unique_terms": 50,
        }

        features = extract_features(mock_term)

        # All binary features should be 0 or 1
        binary_features = [
            "has_ner",
            "has_rake",
            "has_bm25",
            "is_person",
            "has_trailing_punctuation",
            "has_leading_digit",
            "has_trailing_digit",
            "is_all_caps",
            "is_title_case",
            "is_single_letter",
            "has_internal_digits",
            "has_medical_suffix",
            "has_repeated_chars",
            "contains_hyphen",
            "all_low_conf",
            "is_in_names_dataset",
            "has_legal_suffix",
            "has_title_prefix",
            "has_professional_suffix",
        ]

        for i, name in enumerate(FEATURE_NAMES):
            if name in binary_features:
                assert features[i] in (0.0, 1.0), f"Binary feature {name} has value {features[i]}"

        # All count bins should be 0 or 1 (one-hot encoded)
        count_bin_indices = [
            i for i, name in enumerate(FEATURE_NAMES) if name.startswith("count_bin_")
        ]
        count_bin_sum = sum(features[i] for i in count_bin_indices)
        assert count_bin_sum == 1.0, f"Count bins should sum to 1, got {count_bin_sum}"


class TestMLModelTraining:
    """Tests for ML model training and prediction."""

    def test_train_with_mock_feedback(self):
        """Verify ML model can train on mock feedback data."""
        from src.core.vocabulary.preference_learner_training import train_models

        # Create mock feedback records (need at least 30 for training)
        mock_records = []
        for i in range(40):
            mock_records.append(
                {
                    "term": f"term_{i}",
                    "Term": f"term_{i}",
                    "in_case_freq": i + 1,
                    "algorithms": "NER" if i % 2 == 0 else "RAKE",
                    "is_person": i % 3 == 0,
                    "source_doc_confidence": 80 + (i % 20),
                    "feedback": "+1" if i % 2 == 0 else "-1",
                    "timestamp": "2024-01-01T00:00:00",
                    "source": "user",
                }
            )

        lr_model, rf_model, scaler, ensemble_enabled, user_count, total_count = train_models(
            mock_records
        )

        assert lr_model is not None, "LR model should be trained"
        assert scaler is not None, "Scaler should be fitted"
        assert total_count == 40, f"Expected 40 samples, got {total_count}"

    def test_prediction_returns_valid_probability(self):
        """Verify predictions are valid probabilities between 0 and 1."""
        from src.core.vocabulary.preference_learner_features import extract_features
        from src.core.vocabulary.preference_learner_training import train_models

        # Train a model first
        mock_records = []
        for i in range(40):
            mock_records.append(
                {
                    "term": f"term_{i}",
                    "Term": f"term_{i}",
                    "in_case_freq": i + 1,
                    "algorithms": "NER",
                    "is_person": 0,
                    "feedback": "+1" if i % 2 == 0 else "-1",
                    "timestamp": "2024-01-01T00:00:00",
                    "source": "user",
                }
            )

        lr_model, _, scaler, _, _, _ = train_models(mock_records)

        if lr_model is not None and scaler is not None:
            # Make a prediction
            test_term = {
                "Term": "test_term",
                "in_case_freq": 5,
                "algorithms": "NER",
                "is_person": 0,
            }
            features = extract_features(test_term)
            X_scaled = scaler.transform(features.reshape(1, -1))
            prob = lr_model.predict_proba(X_scaled)[0][1]

            assert 0.0 <= prob <= 1.0, f"Prediction should be 0-1, got {prob}"


class TestTextAnalysis:
    """Tests for text analysis helper functions."""

    def test_max_consonant_run(self):
        """Verify consonant run detection works correctly."""
        from src.core.vocabulary.preference_learner_text_analysis import _max_consonant_run

        assert _max_consonant_run("strengths") == 5  # "ngths" has 5 consonants
        assert _max_consonant_run("hello") == 2  # "ll"
        assert _max_consonant_run("aeiou") == 0  # All vowels
        assert _max_consonant_run("xyz") == 3  # All consonants

    def test_log_rarity_score_ordering(self):
        """Verify log rarity scores preserve rank ordering."""
        from src.core.vocabulary.preference_learner_text_analysis import _log_rarity_score

        # More common words (lower linear score) should have lower log score
        score_common = _log_rarity_score(0.001)  # rank ~333 (very common)
        score_medium = _log_rarity_score(0.01)  # rank ~3330
        score_rare = _log_rarity_score(0.5)  # rank ~166500

        assert score_common < score_medium < score_rare, (
            f"Expected ordering: {score_common} < {score_medium} < {score_rare}"
        )


class TestConfigMagicNumbers:
    """Tests for config values loaded correctly."""

    def test_debug_mode_default_false(self):
        """Verify DEBUG_MODE defaults to False when env var not set."""
        import os

        # Temporarily remove DEBUG env var if set
        orig_debug = os.environ.get("DEBUG")
        if "DEBUG" in os.environ:
            del os.environ["DEBUG"]

        # Reimport to get fresh value
        import importlib

        import src.config

        importlib.reload(src.config)

        from src.config import DEBUG_MODE

        # Restore original env var
        if orig_debug is not None:
            os.environ["DEBUG"] = orig_debug

        assert DEBUG_MODE is False, "DEBUG_MODE should default to False"

    def test_index_page_remover_config_exists(self):
        """Verify index page remover config values exist."""
        from src.config import (
            INDEX_CHAR_WINDOW_SIZE,
            INDEX_DETECTION_WINDOW_SIZE,
            INDEX_ESTIMATED_CHARS_PER_LINE,
            INDEX_MAX_CHECK_LENGTH,
            INDEX_MIN_DENSITY_PERCENT,
            INDEX_MIN_INDEX_LINES,
            INDEX_MIN_TEXT_LENGTH,
            INDEX_PAGE_REF_DIVISOR,
            INDEX_TAIL_CHECK_FRACTION,
        )

        assert INDEX_MIN_INDEX_LINES > 0
        assert INDEX_MIN_DENSITY_PERCENT > 0
        assert INDEX_DETECTION_WINDOW_SIZE > 0
        assert INDEX_MIN_TEXT_LENGTH > 0
        assert INDEX_TAIL_CHECK_FRACTION > 0
        assert INDEX_MAX_CHECK_LENGTH > 0
        assert INDEX_ESTIMATED_CHARS_PER_LINE > 0
        assert INDEX_PAGE_REF_DIVISOR > 0
        assert INDEX_CHAR_WINDOW_SIZE > 0

    def test_semantic_chunker_model_config_exists(self):
        """Verify semantic chunker model config exists."""
        from src.config import SEMANTIC_CHUNKER_EMBEDDING_MODEL

        assert SEMANTIC_CHUNKER_EMBEDDING_MODEL == "all-MiniLM-L6-v2"
