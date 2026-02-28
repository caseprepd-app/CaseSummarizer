"""Tests for user-defined indicator pattern compilation and matching."""

from unittest.mock import patch

import pytest


class MockPrefs:
    """Mock UserPreferencesManager for testing."""

    def __init__(self, data=None):
        self._data = data or {}

    def get(self, key, default=None):
        return self._data.get(key, default)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the indicator pattern cache before each test."""
    from src.core.vocabulary.indicator_patterns import invalidate_cache

    invalidate_cache()
    yield
    invalidate_cache()


class TestBuildRegexPreview:
    """Tests for build_regex_preview."""

    def test_empty_list(self):
        from src.core.vocabulary.indicator_patterns import build_regex_preview

        assert build_regex_preview([]) == ""

    def test_single_string(self):
        from src.core.vocabulary.indicator_patterns import build_regex_preview

        result = build_regex_preview(["hello"])
        assert result == "(?i)(?:hello)"

    def test_multiple_strings(self):
        from src.core.vocabulary.indicator_patterns import build_regex_preview

        result = build_regex_preview(["direct", "redirect", "cross"])
        assert result == "(?i)(?:direct|redirect|cross)"

    def test_special_chars_escaped(self):
        from src.core.vocabulary.indicator_patterns import build_regex_preview

        result = build_regex_preview(["dr.", "u.s.c."])
        assert r"dr\." in result
        assert r"u\.s\.c\." in result

    def test_empty_strings_filtered(self):
        from src.core.vocabulary.indicator_patterns import build_regex_preview

        result = build_regex_preview(["hello", "", "  ", "world"])
        assert result == "(?i)(?:hello|world)"

    def test_all_empty_strings(self):
        from src.core.vocabulary.indicator_patterns import build_regex_preview

        assert build_regex_preview(["", "  "]) == ""


class TestValidateRegex:
    """Tests for validate_regex."""

    def test_valid_regex(self):
        from src.core.vocabulary.indicator_patterns import validate_regex

        assert validate_regex("(?i)hello") is None

    def test_invalid_regex(self):
        from src.core.vocabulary.indicator_patterns import validate_regex

        result = validate_regex("[invalid")
        assert result is not None
        assert isinstance(result, str)

    def test_empty_string(self):
        from src.core.vocabulary.indicator_patterns import validate_regex

        assert validate_regex("") is None

    def test_whitespace_only(self):
        from src.core.vocabulary.indicator_patterns import validate_regex

        assert validate_regex("   ") is None


class TestMatchesPositive:
    """Tests for matches_positive."""

    def test_no_patterns_defined(self):
        """When no patterns are defined, nothing matches."""
        with patch(
            "src.core.vocabulary.indicator_patterns.get_user_preferences",
            return_value=MockPrefs(),
        ):
            from src.core.vocabulary.indicator_patterns import matches_positive

            assert matches_positive("Dr. Smith") is False

    def test_matches_string_pattern(self):
        """String patterns should match case-insensitively."""
        prefs = MockPrefs({"vocab_positive_indicators": ["dr.", "plaintiff"]})
        with patch(
            "src.core.vocabulary.indicator_patterns.get_user_preferences",
            return_value=prefs,
        ):
            from src.core.vocabulary.indicator_patterns import matches_positive

            assert matches_positive("Dr. Smith") is True
            assert matches_positive("dr. jones") is True
            assert matches_positive("the plaintiff") is True
            assert matches_positive("some random term") is False

    def test_regex_override(self):
        """Regex override should be used instead of string list."""
        prefs = MockPrefs(
            {
                "vocab_positive_indicators": ["dr."],
                "vocab_positive_regex_override": "(?i)custom_pattern",
            }
        )
        with patch(
            "src.core.vocabulary.indicator_patterns.get_user_preferences",
            return_value=prefs,
        ):
            from src.core.vocabulary.indicator_patterns import matches_positive

            # Should use override, not string list
            assert matches_positive("custom_pattern here") is True
            assert matches_positive("Dr. Smith") is False

    def test_invalid_regex_override_falls_back(self):
        """Invalid regex override should log warning and return False."""
        prefs = MockPrefs(
            {
                "vocab_positive_indicators": ["dr."],
                "vocab_positive_regex_override": "[invalid",
            }
        )
        with patch(
            "src.core.vocabulary.indicator_patterns.get_user_preferences",
            return_value=prefs,
        ):
            from src.core.vocabulary.indicator_patterns import matches_positive

            # Invalid regex override — no match
            assert matches_positive("Dr. Smith") is False


class TestMatchesNegative:
    """Tests for matches_negative."""

    def test_matches_negative_strings(self):
        """Negative patterns should match correctly."""
        prefs = MockPrefs(
            {
                "vocab_negative_indicators": ["direct", "redirect", "cross"],
            }
        )
        with patch(
            "src.core.vocabulary.indicator_patterns.get_user_preferences",
            return_value=prefs,
        ):
            from src.core.vocabulary.indicator_patterns import matches_negative

            assert matches_negative("REDIRECT EXAMINATION") is True
            assert matches_negative("CROSS EXAMINATION") is True
            assert matches_negative("DIRECT EXAMINATION") is True
            assert matches_negative("John Smith") is False


class TestCaching:
    """Tests for pattern caching behavior."""

    def test_cache_invalidation(self):
        """Cache should be invalidated when invalidate_cache is called."""
        from src.core.vocabulary.indicator_patterns import _cache, invalidate_cache

        prefs = MockPrefs({"vocab_positive_indicators": ["test"]})
        with patch(
            "src.core.vocabulary.indicator_patterns.get_user_preferences",
            return_value=prefs,
        ):
            from src.core.vocabulary.indicator_patterns import matches_positive

            matches_positive("test")
            assert len(_cache) > 0

            invalidate_cache()
            assert len(_cache) == 0

    def test_cache_auto_updates_on_preference_change(self):
        """Cache should auto-update when preferences change."""
        prefs1 = MockPrefs({"vocab_positive_indicators": ["old"]})
        with patch(
            "src.core.vocabulary.indicator_patterns.get_user_preferences",
            return_value=prefs1,
        ):
            from src.core.vocabulary.indicator_patterns import matches_positive

            assert matches_positive("old") is True
            assert matches_positive("new") is False

        prefs2 = MockPrefs({"vocab_positive_indicators": ["new"]})
        with patch(
            "src.core.vocabulary.indicator_patterns.get_user_preferences",
            return_value=prefs2,
        ):
            from src.core.vocabulary.indicator_patterns import matches_positive

            assert matches_positive("new") is True
            assert matches_positive("old") is False


class TestFeatureExtraction:
    """Tests for indicator pattern features in extract_features."""

    @pytest.fixture
    def mock_all_deps(self):
        """Mock all dependencies for extract_features."""
        mock_prefs = MockPrefs(
            {
                "vocab_positive_indicators": ["dr."],
                "vocab_negative_indicators": ["redirect", "cross"],
            }
        )
        MOCK_FREQ = {"the": 0.0, "john": 0.01, "smith": 0.05}

        with (
            patch(
                "src.core.vocabulary.preference_learner_features._load_scaled_frequencies",
                return_value=MOCK_FREQ,
            ),
            patch(
                "src.core.vocabulary.preference_learner_features._load_names_datasets",
                return_value=({"john"}, {"smith"}),
            ),
            patch(
                "src.core.vocabulary.preference_learner_features._get_name_country_data",
                return_value=({"john": 5, "smith": 3}, 20),
            ),
            patch(
                "src.core.vocabulary.preference_learner_features.get_user_preferences",
                return_value=mock_prefs,
            ),
            patch(
                "src.core.vocabulary.preference_learner_features._log_rarity_score",
                side_effect=lambda x: x,
            ),
            patch(
                "src.core.vocabulary.preference_learner_features.compute_adjusted_mean",
                return_value=0.5,
            ),
            patch(
                "src.core.vocabulary.rarity_filter.is_common_word",
                side_effect=lambda word, top_n=200000: word in ("the", "and"),
            ),
            patch(
                "src.core.vocabulary.indicator_patterns.get_user_preferences",
                return_value=mock_prefs,
            ),
        ):
            from src.core.vocabulary.preference_learner_features import (
                FEATURE_NAMES,
                extract_features,
            )

            yield extract_features, FEATURE_NAMES

    def test_positive_indicator_feature(self, mock_all_deps):
        """Terms matching positive indicators should have feature=1.0."""
        extract_features, FEATURE_NAMES = mock_all_deps
        term_data = {"Term": "Dr. Smith", "occurrences": 1, "algorithms": "NER"}
        features = extract_features(term_data)
        idx = FEATURE_NAMES.index("matches_positive_indicator")
        assert features[idx] == 1.0

    def test_negative_indicator_feature(self, mock_all_deps):
        """Terms matching negative indicators should have feature=1.0."""
        extract_features, FEATURE_NAMES = mock_all_deps
        term_data = {"Term": "REDIRECT EXAMINATION", "occurrences": 3, "algorithms": ""}
        features = extract_features(term_data)
        idx = FEATURE_NAMES.index("matches_negative_indicator")
        assert features[idx] == 1.0

    def test_no_indicator_match(self, mock_all_deps):
        """Terms not matching any indicators should have both features=0.0."""
        extract_features, FEATURE_NAMES = mock_all_deps
        term_data = {"Term": "John Smith", "occurrences": 1, "algorithms": "NER"}
        features = extract_features(term_data)
        pos_idx = FEATURE_NAMES.index("matches_positive_indicator")
        neg_idx = FEATURE_NAMES.index("matches_negative_indicator")
        assert features[pos_idx] == 0.0
        assert features[neg_idx] == 0.0

    def test_feature_vector_has_56_elements(self, mock_all_deps):
        """Feature vector should now have 56 elements."""
        extract_features, FEATURE_NAMES = mock_all_deps
        term_data = {"Term": "test", "occurrences": 1, "algorithms": ""}
        features = extract_features(term_data)
        assert len(features) == 56
        assert len(FEATURE_NAMES) == 56
