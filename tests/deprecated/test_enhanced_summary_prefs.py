"""
Tests for enhanced summary mode preferences in UserPreferencesManager.

Covers:
- get/set summary_enhanced_mode tri-state ("auto"/"yes"/"no")
- is_enhanced_summary_enabled resolves "auto" via GPU detection
- Validation rejects invalid values
- Default is "auto"
"""

import pytest

from src.user_preferences import UserPreferencesManager


@pytest.fixture
def prefs_manager(tmp_path):
    """Create a UserPreferencesManager with a temp preferences file."""
    prefs_file = tmp_path / "test_prefs.json"
    prefs_file.write_text("{}", encoding="utf-8")
    return UserPreferencesManager(prefs_file)


class TestEnhancedSummaryModeGetSet:
    """Test get/set for summary_enhanced_mode."""

    def test_default_is_auto(self, prefs_manager):
        """Default value should be 'auto'."""
        assert prefs_manager.get_summary_enhanced_mode() == "auto"

    def test_set_yes(self, prefs_manager):
        """Setting to 'yes' should persist."""
        prefs_manager.set_summary_enhanced_mode("yes")
        assert prefs_manager.get_summary_enhanced_mode() == "yes"

    def test_set_no(self, prefs_manager):
        """Setting to 'no' should persist."""
        prefs_manager.set_summary_enhanced_mode("no")
        assert prefs_manager.get_summary_enhanced_mode() == "no"

    def test_set_auto(self, prefs_manager):
        """Setting to 'auto' should persist."""
        prefs_manager.set_summary_enhanced_mode("yes")
        prefs_manager.set_summary_enhanced_mode("auto")
        assert prefs_manager.get_summary_enhanced_mode() == "auto"

    def test_invalid_value_raises(self, prefs_manager):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            prefs_manager.set_summary_enhanced_mode("maybe")

    def test_generic_set_validates(self, prefs_manager):
        """The generic set() method should also validate."""
        with pytest.raises(ValueError, match="summary_enhanced_mode"):
            prefs_manager.set("summary_enhanced_mode", "invalid")

    def test_invalid_stored_value_returns_auto(self, prefs_manager):
        """If stored value is corrupted, default to 'auto'."""
        prefs_manager._preferences["summary_enhanced_mode"] = "garbage"
        assert prefs_manager.get_summary_enhanced_mode() == "auto"


class TestIsEnhancedSummaryEnabled:
    """Test the resolved boolean from is_enhanced_summary_enabled."""

    def test_yes_returns_true(self, prefs_manager):
        """'yes' mode should always return True."""
        prefs_manager.set_summary_enhanced_mode("yes")
        assert prefs_manager.is_enhanced_summary_enabled() is True

    def test_no_returns_false(self, prefs_manager):
        """'no' mode should always return False."""
        prefs_manager.set_summary_enhanced_mode("no")
        assert prefs_manager.is_enhanced_summary_enabled() is False

    def test_auto_resolves_via_gpu_detection(self, prefs_manager):
        """'auto' mode should call has_dedicated_gpu and return its result."""
        prefs_manager.set_summary_enhanced_mode("auto")

        # Mock the local import inside is_enhanced_summary_enabled
        import src.core.utils.gpu_detector as gpu_mod

        original = gpu_mod.has_dedicated_gpu

        try:
            # Test with GPU present
            gpu_mod.has_dedicated_gpu = lambda: True
            assert prefs_manager.is_enhanced_summary_enabled() is True

            # Test without GPU
            gpu_mod.has_dedicated_gpu = lambda: False
            assert prefs_manager.is_enhanced_summary_enabled() is False
        finally:
            gpu_mod.has_dedicated_gpu = original
