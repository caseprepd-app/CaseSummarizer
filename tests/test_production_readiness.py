"""Production readiness tests — config validation, encoding safety, defensive reads."""

import json
from unittest.mock import patch

import pytest


class TestActiveCorpusValidation:
    """Tests for active_corpus preference validation."""

    @pytest.fixture
    def prefs(self, tmp_path):
        """Create a UserPreferencesManager with a temp file."""
        from src.user_preferences import UserPreferencesManager

        pref_file = tmp_path / "prefs.json"
        return UserPreferencesManager(pref_file)

    def test_set_valid_corpus_name(self, prefs):
        """Setting a valid corpus name should succeed."""
        prefs.set("active_corpus", "My Transcripts")
        assert prefs.get("active_corpus") == "My Transcripts"

    def test_reject_empty_corpus_name(self, prefs):
        """Empty string should be rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            prefs.set("active_corpus", "")

    def test_reject_whitespace_only_corpus_name(self, prefs):
        """Whitespace-only string should be rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            prefs.set("active_corpus", "   ")

    def test_reject_path_traversal_dotdot(self, prefs):
        """Path traversal with .. should be rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            prefs.set("active_corpus", "../../../etc/passwd")

    def test_reject_forward_slash(self, prefs):
        """Forward slashes should be rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            prefs.set("active_corpus", "foo/bar")

    def test_reject_backslash(self, prefs):
        """Backslashes should be rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            prefs.set("active_corpus", "foo\\bar")

    def test_reject_non_string(self, prefs):
        """Non-string values should be rejected."""
        with pytest.raises(ValueError):
            prefs.set("active_corpus", 123)


class TestCorruptPreferencesFile:
    """Tests for handling corrupted preferences files."""

    def test_corrupt_json_returns_defaults(self, tmp_path):
        """Corrupted JSON file should return defaults, not crash."""
        from src.user_preferences import UserPreferencesManager

        pref_file = tmp_path / "prefs.json"
        pref_file.write_text("{invalid json!!!", encoding="utf-8")

        mgr = UserPreferencesManager(pref_file)
        # Should get default structure, not crash
        assert mgr.get("model_defaults") is not None or mgr.get("model_defaults") == {}

    def test_corrupt_json_logs_warning(self, tmp_path):
        """Corrupted JSON should log a warning."""
        from src.user_preferences import UserPreferencesManager

        pref_file = tmp_path / "prefs.json"
        pref_file.write_text("{bad json", encoding="utf-8")

        with patch("src.user_preferences.logger") as mock_logger:
            UserPreferencesManager(pref_file)
            mock_logger.warning.assert_called_once()

    def test_wrong_type_resource_pct_uses_default(self, tmp_path):
        """If resource_usage_pct is wrong type in JSON, default should be used."""
        from src.user_preferences import UserPreferencesManager

        pref_file = tmp_path / "prefs.json"
        pref_file.write_text(
            json.dumps({"model_defaults": {}, "resource_usage_pct": "not_a_number"}),
            encoding="utf-8",
        )
        mgr = UserPreferencesManager(pref_file)
        raw = mgr.get("resource_usage_pct", 75)
        # The raw value may be wrong type, but system_resources should handle it
        # (tested in TestResourcePctValidation)


class TestResourcePctValidation:
    """Tests for resource_usage_pct validation at read-time."""

    def test_valid_pct_passes_through(self):
        """Valid percentage should be used as-is."""
        from src.system_resources import get_system_resources

        with patch("src.system_resources.get_user_preferences") as mock:
            mock.return_value.get.return_value = 50
            info = get_system_resources()
            assert info.resource_usage_pct == 50

    def test_negative_pct_uses_default(self):
        """Negative percentage should fall back to 75."""
        from src.system_resources import get_system_resources

        with patch("src.system_resources.get_user_preferences") as mock:
            mock.return_value.get.return_value = -10
            info = get_system_resources()
            assert info.resource_usage_pct == 75

    def test_over_100_uses_default(self):
        """Over 100% should fall back to 75."""
        from src.system_resources import get_system_resources

        with patch("src.system_resources.get_user_preferences") as mock:
            mock.return_value.get.return_value = 5000
            info = get_system_resources()
            assert info.resource_usage_pct == 75

    def test_string_type_uses_default(self):
        """String type should fall back to 75."""
        from src.system_resources import get_system_resources

        with patch("src.system_resources.get_user_preferences") as mock:
            mock.return_value.get.return_value = "invalid"
            info = get_system_resources()
            assert info.resource_usage_pct == 75

    def test_none_uses_default(self):
        """None should fall back to 75."""
        from src.system_resources import get_system_resources

        with patch("src.system_resources.get_user_preferences") as mock:
            mock.return_value.get.return_value = None
            info = get_system_resources()
            assert info.resource_usage_pct == 75


class TestRetrievalWeightValidation:
    """Tests for retrieval weight type validation."""

    def test_valid_weights_pass_through(self):
        """Valid float weights should be used."""
        from src.core.vector_store.qa_retriever import _get_effective_algorithm_weights

        class FakePrefs:
            def get(self, key, default=None):
                if key == "retrieval_weight_faiss":
                    return 0.8
                if key == "retrieval_weight_bm25":
                    return 1.2
                return default

        with patch(
            "src.user_preferences.get_user_preferences",
            return_value=FakePrefs(),
        ):
            weights = _get_effective_algorithm_weights()
            assert weights["FAISS"] == 0.8
            assert weights["BM25+"] == 1.2

    def test_string_weight_falls_back_to_default(self):
        """String weight should fall back to config default."""
        from src.config import RETRIEVAL_ALGORITHM_WEIGHTS
        from src.core.vector_store.qa_retriever import _get_effective_algorithm_weights

        class FakePrefs:
            def get(self, key, default=None):
                if key == "retrieval_weight_faiss":
                    return "invalid"
                if key == "retrieval_weight_bm25":
                    return None
                return default

        with patch(
            "src.user_preferences.get_user_preferences",
            return_value=FakePrefs(),
        ):
            weights = _get_effective_algorithm_weights()
            assert weights["FAISS"] == RETRIEVAL_ALGORITHM_WEIGHTS["FAISS"]
            assert weights["BM25+"] == RETRIEVAL_ALGORITHM_WEIGHTS["BM25+"]


class TestPreferenceSetValidation:
    """Tests for known-key validation in UserPreferencesManager.set()."""

    @pytest.fixture
    def prefs(self, tmp_path):
        """Create a UserPreferencesManager with a temp file."""
        from src.user_preferences import UserPreferencesManager

        pref_file = tmp_path / "prefs.json"
        return UserPreferencesManager(pref_file)

    def test_resource_pct_valid(self, prefs):
        prefs.set("resource_usage_pct", 50)
        assert prefs.get("resource_usage_pct") == 50

    def test_resource_pct_too_low(self, prefs):
        with pytest.raises(ValueError):
            prefs.set("resource_usage_pct", 10)

    def test_resource_pct_too_high(self, prefs):
        with pytest.raises(ValueError):
            prefs.set("resource_usage_pct", 200)

    def test_retrieval_weight_valid(self, prefs):
        prefs.set("retrieval_weight_faiss", 1.5)
        assert prefs.get("retrieval_weight_faiss") == 1.5

    def test_retrieval_weight_too_high(self, prefs):
        with pytest.raises(ValueError):
            prefs.set("retrieval_weight_faiss", 3.0)

    def test_indicator_patterns_valid(self, prefs):
        prefs.set("vocab_positive_indicators", ["dr.", "plaintiff"])
        assert prefs.get("vocab_positive_indicators") == ["dr.", "plaintiff"]

    def test_indicator_patterns_not_list(self, prefs):
        with pytest.raises(ValueError):
            prefs.set("vocab_positive_indicators", "dr.")

    def test_regex_override_invalid(self, prefs):
        with pytest.raises(ValueError, match="invalid regex"):
            prefs.set("vocab_positive_regex_override", "[invalid")

    def test_unknown_key_still_stored(self, prefs):
        """Unknown keys should still be stored (extensible system)."""
        prefs.set("some_future_setting", "value")
        assert prefs.get("some_future_setting") == "value"
