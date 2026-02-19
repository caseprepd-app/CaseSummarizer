"""
Tests for ML training and text analysis modules.

Covers:
- preference_learner_training.py: confidence_weighted_blend, calculate_sample_weight,
  train_models, save_model, load_model, reset_to_default
- preference_learner_text_analysis.py: _max_consonant_run, _log_rarity_score,
  _load_names_datasets
"""

from datetime import datetime, timedelta

# ============================================================================
# confidence_weighted_blend
# ============================================================================


class TestConfidenceWeightedBlend:
    """Tests for confidence_weighted_blend()."""

    def test_both_uncertain_returns_neutral(self):
        from src.core.vocabulary.preference_learner_training import confidence_weighted_blend

        assert confidence_weighted_blend(0.5, 0.5) == 0.5

    def test_one_confident_dominates(self):
        from src.core.vocabulary.preference_learner_training import confidence_weighted_blend

        result = confidence_weighted_blend(0.9, 0.5)
        assert result > 0.7  # LR is confident, RF is neutral -> skews toward LR

    def test_both_agree_high(self):
        from src.core.vocabulary.preference_learner_training import confidence_weighted_blend

        result = confidence_weighted_blend(0.8, 0.9)
        assert result > 0.8  # Both confident and agree

    def test_both_agree_low(self):
        from src.core.vocabulary.preference_learner_training import confidence_weighted_blend

        result = confidence_weighted_blend(0.1, 0.2)
        assert result < 0.2

    def test_disagree_uses_confidence(self):
        from src.core.vocabulary.preference_learner_training import confidence_weighted_blend

        # LR very confident positive, RF slightly negative
        result = confidence_weighted_blend(0.95, 0.4)
        # LR confidence = 0.45, RF confidence = 0.1 -> LR dominates
        assert result > 0.7

    def test_symmetric_confidence(self):
        from src.core.vocabulary.preference_learner_training import confidence_weighted_blend

        # Equal confidence on both sides
        result = confidence_weighted_blend(0.8, 0.2)
        # Both equidistant from 0.5 -> equal weight -> average = 0.5
        assert abs(result - 0.5) < 0.01


# ============================================================================
# calculate_sample_weight
# ============================================================================


class TestCalculateSampleWeight:
    """Tests for calculate_sample_weight()."""

    def test_recent_sample_has_high_weight(self):
        from src.core.vocabulary.preference_learner_training import calculate_sample_weight

        now = datetime.now().isoformat()
        weight = calculate_sample_weight(now, "user", 0)
        assert weight > 0.9

    def test_old_sample_has_lower_weight(self):
        from src.core.vocabulary.preference_learner_training import calculate_sample_weight

        old = (datetime.now() - timedelta(days=365)).isoformat()
        weight = calculate_sample_weight(old, "user", 0)
        recent = calculate_sample_weight(datetime.now().isoformat(), "user", 0)
        assert weight < recent

    def test_weight_has_floor(self):
        from src.core.vocabulary.preference_learner_training import calculate_sample_weight

        very_old = (datetime.now() - timedelta(days=10000)).isoformat()
        weight = calculate_sample_weight(very_old, "user", 0)
        assert weight > 0  # Floor prevents zero weight

    def test_invalid_timestamp_uses_default(self):
        from src.core.vocabulary.preference_learner_training import calculate_sample_weight

        weight = calculate_sample_weight("not-a-date", "user", 0)
        assert weight > 0  # Should use default 0.75 * source_weight

    def test_user_source_weighting(self):
        from src.core.vocabulary.preference_learner_training import calculate_sample_weight

        now = datetime.now().isoformat()
        # With many user samples, user weight should differ from default
        user_w = calculate_sample_weight(now, "user", 50)
        default_w = calculate_sample_weight(now, "default", 50)
        # Both should be positive
        assert user_w > 0
        assert default_w > 0


# ============================================================================
# train_models
# ============================================================================


class TestTrainModels:
    """Tests for train_models()."""

    def _make_records(self, n_pos=10, n_neg=10):
        """Create synthetic feedback records for training."""
        records = []
        now = datetime.now().isoformat()
        for i in range(n_pos):
            records.append(
                {
                    "term": f"good_term_{i}",
                    "feedback": "+1",
                    "timestamp": now,
                    "source": "user",
                    "confidence": 0.8,
                    "frequency": 5,
                    "is_person": False,
                    "word_count": 2,
                    "score": 75,
                }
            )
        for i in range(n_neg):
            records.append(
                {
                    "term": f"bad_term_{i}",
                    "feedback": "-1",
                    "timestamp": now,
                    "source": "user",
                    "confidence": 0.3,
                    "frequency": 1,
                    "is_person": False,
                    "word_count": 1,
                    "score": 20,
                }
            )
        return records

    def test_insufficient_data_returns_none(self):
        from src.core.vocabulary.preference_learner_training import train_models

        lr, rf, scaler, ensemble, user_n, total_n = train_models([])
        assert lr is None
        assert rf is None
        assert scaler is None
        assert ensemble is False

    def test_insufficient_class_diversity(self):
        from src.core.vocabulary.preference_learner_training import train_models

        # All positive, no negative
        records = self._make_records(n_pos=20, n_neg=0)
        lr, rf, scaler, ensemble, _, _ = train_models(records)
        assert lr is None

    def test_trains_lr_model(self):
        from src.core.vocabulary.preference_learner_training import train_models

        records = self._make_records(n_pos=20, n_neg=20)
        lr, rf, scaler, ensemble, user_n, total_n = train_models(records)
        assert lr is not None
        assert scaler is not None
        assert total_n == 40

    def test_lr_can_predict(self):
        from src.core.vocabulary.preference_learner_training import train_models

        records = self._make_records(n_pos=20, n_neg=20)
        lr, rf, scaler, ensemble, _, _ = train_models(records)
        assert lr is not None
        # Predict on a new sample
        from src.core.vocabulary.preference_learner_features import extract_features

        features = extract_features(records[0])
        X = scaler.transform([features])
        proba = lr.predict_proba(X)[0]
        assert len(proba) == 2
        assert 0 <= proba[0] <= 1

    def test_ignores_cleared_feedback(self):
        from src.core.vocabulary.preference_learner_training import train_models

        records = self._make_records(n_pos=20, n_neg=20)
        # Add cleared records that should be ignored
        records.append(
            {"term": "cleared", "feedback": "0", "timestamp": datetime.now().isoformat()}
        )
        records.append(
            {"term": "cleared2", "feedback": None, "timestamp": datetime.now().isoformat()}
        )
        lr, rf, scaler, ensemble, _, total_n = train_models(records)
        assert total_n == 40  # Only labeled records counted


# ============================================================================
# save_model / load_model
# ============================================================================


class TestModelPersistence:
    """Tests for save_model and load_model."""

    def test_save_and_load_roundtrip(self, tmp_path):
        from src.core.vocabulary.preference_learner_training import (
            load_model,
            save_model,
            train_models,
        )

        # Train a real model (need >= 30 samples = ML_MIN_SAMPLES)
        records = []
        now = datetime.now().isoformat()
        for i in range(20):
            records.append(
                {
                    "term": f"good_{i}",
                    "feedback": "+1",
                    "timestamp": now,
                    "source": "user",
                    "confidence": 0.8,
                    "frequency": 5,
                    "is_person": False,
                    "word_count": 2,
                    "score": 75,
                }
            )
        for i in range(20):
            records.append(
                {
                    "term": f"bad_{i}",
                    "feedback": "-1",
                    "timestamp": now,
                    "source": "user",
                    "confidence": 0.3,
                    "frequency": 1,
                    "is_person": False,
                    "word_count": 1,
                    "score": 20,
                }
            )

        lr, rf, scaler, ensemble, user_n, total_n = train_models(records)
        assert lr is not None

        model_path = tmp_path / "test_model.pkl"
        success = save_model(model_path, lr, rf, scaler, ensemble, user_n, total_n)
        assert success is True
        assert model_path.exists()

        # Load it back
        lr2, rf2, scaler2, ens2, u_n2, t_n2, loaded = load_model(model_path)
        assert loaded is True
        assert lr2 is not None
        assert scaler2 is not None
        assert t_n2 == total_n

    def test_load_nonexistent_file(self, tmp_path):
        from src.core.vocabulary.preference_learner_training import load_model

        lr, rf, scaler, ens, u_n, t_n, loaded = load_model(tmp_path / "nope.pkl")
        assert loaded is False
        assert lr is None

    def test_load_corrupted_file(self, tmp_path):
        from src.core.vocabulary.preference_learner_training import load_model

        bad_file = tmp_path / "bad.pkl"
        bad_file.write_bytes(b"not a pickle")
        lr, rf, scaler, ens, u_n, t_n, loaded = load_model(bad_file)
        assert loaded is False


# ============================================================================
# reset_to_default
# ============================================================================


class TestResetToDefault:
    """Tests for reset_to_default()."""

    def test_deletes_model_file(self, tmp_path):
        from src.core.vocabulary.preference_learner_training import reset_to_default

        model = tmp_path / "model.pkl"
        model.write_bytes(b"data")
        assert model.exists()
        result = reset_to_default(model)
        assert result is True
        assert not model.exists()

    def test_no_file_succeeds(self, tmp_path):
        from src.core.vocabulary.preference_learner_training import reset_to_default

        result = reset_to_default(tmp_path / "nope.pkl")
        assert result is True


# ============================================================================
# _max_consonant_run
# ============================================================================


class TestMaxConsonantRun:
    """Tests for _max_consonant_run()."""

    def test_normal_word(self):
        from src.core.vocabulary.preference_learner_text_analysis import _max_consonant_run

        # s(c) t(c) r(c) e(v) n(c) g(c) t(c) h(c) s(c) -> max run = 5 (ngths)
        assert _max_consonant_run("strengths") == 5

    def test_all_vowels(self):
        from src.core.vocabulary.preference_learner_text_analysis import _max_consonant_run

        assert _max_consonant_run("aeiou") == 0

    def test_all_consonants(self):
        from src.core.vocabulary.preference_learner_text_analysis import _max_consonant_run

        assert _max_consonant_run("bcdfg") == 5

    def test_empty(self):
        from src.core.vocabulary.preference_learner_text_analysis import _max_consonant_run

        assert _max_consonant_run("") == 0

    def test_mixed_case(self):
        from src.core.vocabulary.preference_learner_text_analysis import _max_consonant_run

        assert _max_consonant_run("STRong") == 3  # S, T, R are consonants


# ============================================================================
# _log_rarity_score
# ============================================================================


class TestLogRarityScore:
    """Tests for _log_rarity_score()."""

    def test_zero_returns_zero(self):
        from src.core.vocabulary.preference_learner_text_analysis import _log_rarity_score

        assert _log_rarity_score(0.0) == 0.0

    def test_negative_returns_zero(self):
        from src.core.vocabulary.preference_learner_text_analysis import _log_rarity_score

        assert _log_rarity_score(-0.1) == 0.0

    def test_common_word_low_score(self):
        from src.core.vocabulary.preference_learner_text_analysis import _log_rarity_score

        # Rank ~100 out of 333000 -> very common
        score = _log_rarity_score(100 / 333000)
        assert score < 0.5

    def test_rare_word_high_score(self):
        from src.core.vocabulary.preference_learner_text_analysis import _log_rarity_score

        # Rank ~300000 out of 333000 -> very rare
        score = _log_rarity_score(300000 / 333000)
        assert score > 0.8

    def test_monotonic_increasing(self):
        from src.core.vocabulary.preference_learner_text_analysis import _log_rarity_score

        scores = [_log_rarity_score(r / 333000) for r in [10, 100, 1000, 10000, 100000]]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]


# ============================================================================
# _load_names_datasets
# ============================================================================


class TestLoadNamesDatasets:
    """Tests for _load_names_datasets()."""

    def test_returns_two_sets(self):
        from src.core.vocabulary.preference_learner_text_analysis import _load_names_datasets

        forenames, surnames = _load_names_datasets()
        assert isinstance(forenames, set)
        assert isinstance(surnames, set)

    def test_forenames_not_empty(self):
        from src.core.vocabulary.preference_learner_text_analysis import _load_names_datasets

        forenames, _ = _load_names_datasets()
        assert len(forenames) > 100

    def test_surnames_not_empty(self):
        from src.core.vocabulary.preference_learner_text_analysis import _load_names_datasets

        _, surnames = _load_names_datasets()
        assert len(surnames) > 100

    def test_names_are_lowercase(self):
        from src.core.vocabulary.preference_learner_text_analysis import _load_names_datasets

        forenames, surnames = _load_names_datasets()
        for name in list(forenames)[:20]:
            assert name == name.lower()

    def test_caching_returns_same_objects(self):
        from src.core.vocabulary.preference_learner_text_analysis import _load_names_datasets

        f1, s1 = _load_names_datasets()
        f2, s2 = _load_names_datasets()
        assert f1 is f2
        assert s1 is s2

    def test_country_data_accessible(self):
        from src.core.vocabulary.preference_learner_text_analysis import _get_name_country_data

        counts, total = _get_name_country_data()
        assert isinstance(counts, dict)
        assert total >= 1
