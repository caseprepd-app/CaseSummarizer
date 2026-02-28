from unittest.mock import patch

import pytest

# Need to mock several dependencies
MOCK_FREQ = {"the": 0.0, "age": 0.002, "john": 0.01, "smith": 0.05, "radiculopathy": 0.8}


class MockPrefs:
    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)


@pytest.fixture
def mock_deps():
    """Mock all external dependencies."""
    with (
        patch(
            "src.core.vocabulary.preference_learner_features._load_scaled_frequencies",
            return_value=MOCK_FREQ,
        ),
        patch(
            "src.core.vocabulary.preference_learner_features._load_names_datasets",
            return_value=({"john", "james"}, {"smith", "jones"}),
        ),
        patch(
            "src.core.vocabulary.preference_learner_features._get_name_country_data",
            return_value=({"john": 5, "smith": 3}, 20),
        ),
        patch(
            "src.core.vocabulary.preference_learner_features.get_user_preferences",
            return_value=MockPrefs(),
        ),
        patch(
            "src.core.vocabulary.preference_learner_features._log_rarity_score",
            side_effect=lambda x: x,
        ),
        patch(
            "src.core.vocabulary.preference_learner_features.compute_adjusted_mean",
            return_value=0.5,
        ),
    ):
        # Mock is_common_word for stop word features (imported inside extract_features)
        with (
            patch(
                "src.core.vocabulary.rarity_filter.is_common_word",
                side_effect=lambda word, top_n=200000: word in ("the", "and", "a", "of"),
            ),
            patch(
                "src.core.vocabulary.indicator_patterns.get_user_preferences",
                return_value=MockPrefs(),
            ),
        ):
            from src.core.vocabulary.preference_learner_features import (
                FEATURE_NAMES,
                extract_features,
            )

            yield extract_features, FEATURE_NAMES


def test_feature_vector_length(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "John Smith", "occurrences": 5, "algorithms": "NER", "is_person": 1}
    features = extract_features(term_data)
    assert len(features) == len(FEATURE_NAMES)
    assert len(features) == 56


def test_person_feature(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "John Smith", "occurrences": 1, "algorithms": "NER", "is_person": 1}
    features = extract_features(term_data)
    idx = FEATURE_NAMES.index("is_person")
    assert features[idx] == 1.0


def test_algorithm_features(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "test term", "occurrences": 1, "algorithms": "NER, RAKE, BM25"}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("has_ner")] == 1.0
    assert features[FEATURE_NAMES.index("has_rake")] == 1.0
    assert features[FEATURE_NAMES.index("has_bm25")] == 1.0
    assert features[FEATURE_NAMES.index("has_topicrank")] == 0.0


def test_missing_term_raises(mock_deps):
    extract_features, _ = mock_deps
    with pytest.raises(ValueError):
        extract_features({"occurrences": 1})


def test_non_dict_raises(mock_deps):
    extract_features, _ = mock_deps
    with pytest.raises(ValueError):
        extract_features("not a dict")


def test_artifact_features(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "Smith:", "occurrences": 1, "algorithms": ""}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("has_trailing_punctuation")] == 1.0


def test_medical_suffix(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "radiculopathy", "occurrences": 1, "algorithms": "NER"}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("has_medical_suffix")] == 1.0


def test_starts_with_stop_word(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "the same", "occurrences": 1, "algorithms": "RAKE"}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("starts_with_stop_word")] == 1.0
    assert features[FEATURE_NAMES.index("ends_with_stop_word")] == 0.0


def test_ends_with_stop_word(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "Smith and", "occurrences": 1, "algorithms": "NER"}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("starts_with_stop_word")] == 0.0
    assert features[FEATURE_NAMES.index("ends_with_stop_word")] == 1.0


def test_yake_feature(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "legal term", "occurrences": 1, "algorithms": "YAKE", "yake_score": 0.8}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("has_yake")] == 1.0
    assert features[FEATURE_NAMES.index("yake_score")] == 0.8


def test_keybert_feature(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {
        "Term": "legal term",
        "occurrences": 1,
        "algorithms": "KeyBERT",
        "keybert_score": 0.75,
    }
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("has_keybert")] == 1.0
    assert features[FEATURE_NAMES.index("keybert_score")] == 0.75


def test_yake_keybert_absent(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "legal term", "occurrences": 1, "algorithms": "NER"}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("has_yake")] == 0.0
    assert features[FEATURE_NAMES.index("has_keybert")] == 0.0
    assert features[FEATURE_NAMES.index("yake_score")] == 0.0
    assert features[FEATURE_NAMES.index("keybert_score")] == 0.0


def test_rake_score_feature(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "legal term", "occurrences": 1, "algorithms": "RAKE", "rake_score": 7.5}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("rake_score")] == pytest.approx(0.5, abs=0.01)


def test_bm25_score_feature(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "legal term", "occurrences": 1, "algorithms": "BM25", "bm25_score": 15.0}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("bm25_score")] == pytest.approx(1.0, abs=0.01)


def test_rake_bm25_absent(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "legal term", "occurrences": 1, "algorithms": "NER"}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("rake_score")] == 0.0
    assert features[FEATURE_NAMES.index("bm25_score")] == 0.0


def test_count_bin_2_3(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "test", "occurrences": 2, "algorithms": "NER"}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("count_bin_1")] == 0.0
    assert features[FEATURE_NAMES.index("count_bin_2_3")] == 1.0


def test_count_bin_7_20(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "test", "occurrences": 15, "algorithms": "NER"}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("count_bin_7_20")] == 1.0
    assert features[FEATURE_NAMES.index("count_bin_21_plus")] == 0.0


def test_count_bin_21_plus(mock_deps):
    extract_features, FEATURE_NAMES = mock_deps
    term_data = {"Term": "test", "occurrences": 55, "algorithms": "NER"}
    features = extract_features(term_data)
    assert features[FEATURE_NAMES.index("count_bin_7_20")] == 0.0
    assert features[FEATURE_NAMES.index("count_bin_21_plus")] == 1.0
