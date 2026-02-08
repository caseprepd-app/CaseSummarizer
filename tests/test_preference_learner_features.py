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
        with patch(
            "src.core.vocabulary.rarity_filter.is_common_word",
            side_effect=lambda word, top_n=200000: word in ("the", "and", "a", "of"),
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
    assert len(features) == 50


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
    assert features[FEATURE_NAMES.index("has_textrank")] == 0.0


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
