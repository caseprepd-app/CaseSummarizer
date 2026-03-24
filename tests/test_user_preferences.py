"""
Tests for user_preferences.py — UserPreferencesManager.

Covers loading, saving, JSON corruption recovery, migration, and validation.
"""

import json

import pytest

from src.user_preferences import UserPreferencesManager


@pytest.fixture()
def prefs_file(tmp_path):
    """Path to a temp preferences file."""
    return tmp_path / "user_preferences.json"


@pytest.fixture()
def manager(prefs_file):
    """UserPreferencesManager with a fresh temp file."""
    return UserPreferencesManager(prefs_file)


# =========================================================================
# Loading
# =========================================================================


class TestLoading:
    """Tests for _load_preferences()."""

    def test_creates_defaults_when_no_file(self, prefs_file):
        """Returns default structure when file doesn't exist."""
        mgr = UserPreferencesManager(prefs_file)
        assert mgr.get_cpu_fraction() == 0.5
        assert mgr.get("last_used_model") is None

    def test_loads_existing_file(self, prefs_file):
        """Reads preferences from existing JSON file."""
        prefs_file.write_text(
            json.dumps({"model_defaults": {}, "custom_key": "value"}),
            encoding="utf-8",
        )
        mgr = UserPreferencesManager(prefs_file)
        assert mgr.get("custom_key") == "value"

    def test_adds_missing_model_defaults(self, prefs_file):
        """Adds model_defaults key if missing from file."""
        prefs_file.write_text(json.dumps({"custom": 1}), encoding="utf-8")
        mgr = UserPreferencesManager(prefs_file)
        assert mgr.get("model_defaults") == {}

    def test_corrupt_json_recovers(self, prefs_file):
        """Corrupted JSON file is renamed and defaults are used."""
        prefs_file.write_text("{invalid json!!!}", encoding="utf-8")
        mgr = UserPreferencesManager(prefs_file)
        assert mgr.get_cpu_fraction() == 0.5

        corrupt = prefs_file.with_suffix(".json.corrupt")
        assert corrupt.exists()
        assert corrupt.read_text(encoding="utf-8") == "{invalid json!!!}"


# =========================================================================
# Migration
# =========================================================================


class TestMigration:
    """Tests for _migrate_qa_to_semantic()."""

    def test_renames_qa_keys(self, prefs_file):
        """Old qa_ keys are renamed to semantic_ keys."""
        prefs_file.write_text(
            json.dumps(
                {
                    "model_defaults": {},
                    "qa_retrieval_k": 10,
                    "qa_max_tokens": 500,
                }
            ),
            encoding="utf-8",
        )
        mgr = UserPreferencesManager(prefs_file)
        assert mgr.get("semantic_retrieval_k") == 10
        assert mgr.get("semantic_max_tokens") == 500
        assert mgr.get("qa_retrieval_k") is None

    def test_removes_dead_keys(self, prefs_file):
        """Dead keys like summary_temperature are removed."""
        prefs_file.write_text(
            json.dumps(
                {
                    "model_defaults": {},
                    "summary_temperature": 0.7,
                    "hallucination_verification_enabled": True,
                }
            ),
            encoding="utf-8",
        )
        mgr = UserPreferencesManager(prefs_file)
        assert mgr.get("summary_temperature") is None
        assert mgr.get("hallucination_verification_enabled") is None

    def test_no_migration_when_new_key_exists(self, prefs_file):
        """Doesn't overwrite new key if both old and new exist."""
        prefs_file.write_text(
            json.dumps(
                {
                    "model_defaults": {},
                    "qa_retrieval_k": 5,
                    "semantic_retrieval_k": 20,
                }
            ),
            encoding="utf-8",
        )
        mgr = UserPreferencesManager(prefs_file)
        assert mgr.get("semantic_retrieval_k") == 20


# =========================================================================
# CPU fraction
# =========================================================================


class TestCpuFraction:
    """Tests for get/set_cpu_fraction."""

    def test_default(self, manager):
        """Default CPU fraction is 0.5."""
        assert manager.get_cpu_fraction() == 0.5

    def test_set_valid(self, manager):
        """Can set valid fractions."""
        for frac in (0.25, 0.5, 0.75):
            manager.set_cpu_fraction(frac)
            assert manager.get_cpu_fraction() == frac

    def test_set_invalid_raises(self, manager):
        """Invalid fractions raise ValueError."""
        with pytest.raises(ValueError, match="CPU fraction"):
            manager.set_cpu_fraction(0.3)
        with pytest.raises(ValueError):
            manager.set_cpu_fraction(1.0)


# =========================================================================
# Logging level
# =========================================================================


class TestLoggingLevel:
    """Tests for get/set_logging_level."""

    def test_default(self, manager):
        """Default logging level is comprehensive."""
        assert manager.get_logging_level() == "comprehensive"

    def test_set_valid_levels(self, manager):
        """Can set all valid logging levels."""
        for level in ("off", "brief", "comprehensive", "custom"):
            manager.set_logging_level(level)
            assert manager.get_logging_level() == level

    def test_set_invalid_raises(self, manager):
        """Invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid logging level"):
            manager.set_logging_level("verbose")

    def test_invalid_stored_value_returns_default(self, prefs_file):
        """If stored value is invalid, returns 'comprehensive'."""
        prefs_file.write_text(
            json.dumps({"model_defaults": {}, "logging_level": "garbage"}),
            encoding="utf-8",
        )
        mgr = UserPreferencesManager(prefs_file)
        assert mgr.get_logging_level() == "comprehensive"


# =========================================================================
# Generic get/set with validation
# =========================================================================


class TestGenericGetSet:
    """Tests for generic get() and set() with validation."""

    def test_get_missing_key(self, manager):
        """get() returns default for missing key."""
        assert manager.get("nonexistent") is None
        assert manager.get("nonexistent", 42) == 42

    def test_set_and_get(self, manager):
        """set() stores value, get() retrieves it."""
        manager.set("custom_key", "hello")
        assert manager.get("custom_key") == "hello"

    def test_summary_words_validation(self, manager):
        """summary_words must be int 50-2000."""
        manager.set("summary_words", 100)
        assert manager.get("summary_words") == 100

        with pytest.raises(ValueError, match="summary_words"):
            manager.set("summary_words", 10)
        with pytest.raises(ValueError, match="summary_words"):
            manager.set("summary_words", 5000)

    def test_active_corpus_validation(self, manager):
        """active_corpus must be non-empty string without path traversal."""
        manager.set("active_corpus", "my_corpus")
        assert manager.get("active_corpus") == "my_corpus"

        with pytest.raises(ValueError, match="non-empty"):
            manager.set("active_corpus", "")
        with pytest.raises(ValueError, match="invalid characters"):
            manager.set("active_corpus", "../etc/passwd")
        with pytest.raises(ValueError, match="invalid characters"):
            manager.set("active_corpus", "path\\traversal")

    def test_font_size_offset_validation(self, manager):
        """font_size_offset must be int -4 to 10."""
        manager.set("font_size_offset", 0)
        with pytest.raises(ValueError):
            manager.set("font_size_offset", -5)
        with pytest.raises(ValueError):
            manager.set("font_size_offset", 11)

    def test_rarity_threshold_validation(self, manager):
        """Rarity thresholds must be float 0.1-0.9."""
        manager.set("single_word_rarity_threshold", 0.5)
        with pytest.raises(ValueError):
            manager.set("single_word_rarity_threshold", 0.0)
        with pytest.raises(ValueError):
            manager.set("single_word_rarity_threshold", 1.0)

    def test_regex_override_validation(self, manager):
        """Regex overrides must compile without error."""
        manager.set("vocab_positive_regex_override", r"\btest\b")
        with pytest.raises(ValueError, match="invalid regex"):
            manager.set("vocab_positive_regex_override", "[invalid")


# =========================================================================
# Persistence
# =========================================================================


class TestPersistence:
    """Tests for save/load round-trip."""

    def test_round_trip(self, prefs_file):
        """Values survive save and reload."""
        mgr1 = UserPreferencesManager(prefs_file)
        mgr1.set_cpu_fraction(0.75)
        mgr1.set_logging_level("brief")

        mgr2 = UserPreferencesManager(prefs_file)
        assert mgr2.get_cpu_fraction() == 0.75
        assert mgr2.get_logging_level() == "brief"
