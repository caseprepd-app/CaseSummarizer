"""
Tests for dark/light theme toggle feature.

Covers:
- COLORS dict structure (all values are (light, dark) tuples)
- get_color() resolves tuples for current mode
- resolve_tags() converts tag dicts for tkinter compatibility
- Settings registry has theme dropdown
- Startup reads appearance_mode from preferences
"""

from unittest.mock import patch


class TestColorsStructure:
    """All COLORS entries should be (light, dark) tuples."""

    def test_all_colors_are_tuples(self):
        """Every value in COLORS must be a 2-tuple of strings."""
        from src.ui.theme import COLORS

        for key, value in COLORS.items():
            assert isinstance(value, tuple), f"COLORS['{key}'] should be a tuple, got {type(value)}"
            assert len(value) == 2, f"COLORS['{key}'] should have 2 elements, got {len(value)}"
            assert isinstance(value[0], str), f"COLORS['{key}'][0] (light) should be str"
            assert isinstance(value[1], str), f"COLORS['{key}'][1] (dark) should be str"

    def test_attention_colors_preserved_in_dark(self):
        """Red/orange/yellow attention colors should still be vivid in dark mode."""
        from src.ui.theme import COLORS

        danger_dark = COLORS["danger"][1]
        warning_dark = COLORS["warning"][1]
        assert danger_dark.startswith("#"), "danger dark should be hex"
        assert warning_dark.startswith("#"), "warning dark should be hex"
        assert danger_dark.lower() not in ("#ffffff", "#cccccc", "#888888")
        assert warning_dark.lower() not in ("#ffffff", "#cccccc", "#888888")

    def test_light_mode_not_pure_white(self):
        """Light mode backgrounds should be off-white, not pure #ffffff."""
        from src.ui.theme import COLORS

        bg_light = COLORS["bg_dark"][0]  # Main panel background
        assert bg_light != "#ffffff", "Light bg should not be pure white"

    def test_semantic_color_keys_exist(self):
        """Key semantic color names should be defined."""
        from src.ui.theme import COLORS

        required = [
            "bg_dark",
            "bg_darker",
            "text_primary",
            "text_secondary",
            "success",
            "danger",
            "warning",
            "tooltip_bg",
            "tooltip_fg",
            "placeholder_golden",
            "placeholder_red",
            "corpus_error_text",
        ]
        for key in required:
            assert key in COLORS, f"COLORS['{key}'] should exist"


class TestGetColor:
    """get_color() should resolve tuples based on current mode."""

    @patch("src.ui.theme.ctk")
    def test_get_color_dark_mode(self, mock_ctk):
        """In dark mode, get_color returns second tuple element."""
        mock_ctk.get_appearance_mode.return_value = "Dark"
        from src.ui.theme import COLORS, get_color

        for key in ("bg_dark", "text_primary", "danger"):
            result = get_color(key)
            assert result == COLORS[key][1], f"get_color('{key}') should return dark value"

    @patch("src.ui.theme.ctk")
    def test_get_color_light_mode(self, mock_ctk):
        """In light mode, get_color returns first tuple element."""
        mock_ctk.get_appearance_mode.return_value = "Light"
        from src.ui.theme import COLORS, get_color

        for key in ("bg_dark", "text_primary", "danger"):
            result = get_color(key)
            assert result == COLORS[key][0], f"get_color('{key}') should return light value"


class TestResolveTags:
    """resolve_tags() should flatten color tuples for tkinter tag_config."""

    @patch("src.ui.theme.ctk")
    def test_resolve_tags_dark(self, mock_ctk):
        """In dark mode, color tuples resolve to their second element."""
        mock_ctk.get_appearance_mode.return_value = "Dark"
        from src.ui.theme import resolve_tags

        tags = {
            "test_tag": {
                "foreground": ("#light", "#dark"),
                "font": ("Segoe UI", 12),
            }
        }
        result = resolve_tags(tags)
        assert result["test_tag"]["foreground"] == "#dark"
        # Font tuples (str, int) should NOT be resolved as color pairs
        assert result["test_tag"]["font"] == ("Segoe UI", 12)

    @patch("src.ui.theme.ctk")
    def test_resolve_tags_light(self, mock_ctk):
        """In light mode, color tuples resolve to their first element."""
        mock_ctk.get_appearance_mode.return_value = "Light"
        from src.ui.theme import resolve_tags

        tags = {"tag1": {"foreground": ("#aaa", "#bbb"), "background": ("#ccc", "#ddd")}}
        result = resolve_tags(tags)
        assert result["tag1"]["foreground"] == "#aaa"
        assert result["tag1"]["background"] == "#ccc"

    @patch("src.ui.theme.ctk")
    def test_resolve_tags_preserves_non_color_props(self, mock_ctk):
        """Non-color properties (font, overstrike) should pass through unchanged."""
        mock_ctk.get_appearance_mode.return_value = "Dark"
        from src.ui.theme import resolve_tags

        tags = {
            "bold_tag": {
                "foreground": ("#111", "#222"),
                "font": ("Consolas", 10, "bold"),
                "overstrike": True,
            }
        }
        result = resolve_tags(tags)
        assert result["bold_tag"]["font"] == ("Consolas", 10, "bold")
        assert result["bold_tag"]["overstrike"] is True

    @patch("src.ui.theme.ctk")
    def test_resolve_tags_qa(self, mock_ctk):
        """QA_TEXT_TAGS resolves without errors and all foregrounds are strings."""
        mock_ctk.get_appearance_mode.return_value = "Dark"
        from src.ui.theme import QA_TEXT_TAGS, resolve_tags

        result = resolve_tags(QA_TEXT_TAGS)
        for tag_name, config in result.items():
            if "foreground" in config:
                assert isinstance(config["foreground"], str), (
                    f"QA tag '{tag_name}' foreground should be a string after resolve"
                )

    @patch("src.ui.theme.ctk")
    def test_resolve_tags_vocab(self, mock_ctk):
        """VOCAB_TABLE_TAGS resolves without errors."""
        mock_ctk.get_appearance_mode.return_value = "Light"
        from src.ui.theme import VOCAB_TABLE_TAGS, resolve_tags

        result = resolve_tags(VOCAB_TABLE_TAGS)
        for tag_name, config in result.items():
            for prop, val in config.items():
                if prop in ("foreground", "background"):
                    assert isinstance(val, str), (
                        f"VOCAB tag '{tag_name}'.{prop} should be resolved string"
                    )

    @patch("src.ui.theme.ctk")
    def test_resolve_tags_file_status(self, mock_ctk):
        """FILE_STATUS_TAGS resolves without errors."""
        mock_ctk.get_appearance_mode.return_value = "Dark"
        from src.ui.theme import FILE_STATUS_TAGS, resolve_tags

        result = resolve_tags(FILE_STATUS_TAGS)
        for tag_name, config in result.items():
            fg = config.get("foreground", "")
            assert isinstance(fg, str), f"FILE_STATUS tag '{tag_name}' foreground not resolved"


class TestColorPair:
    """color_pair() should return (light, dark) tuples."""

    def test_returns_tuple(self):
        """color_pair() always returns a 2-tuple."""
        from src.ui.theme import color_pair

        result = color_pair("bg_dark")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestThemeSettingsDropdown:
    """Settings registry should include theme dropdown."""

    def test_appearance_mode_registered(self):
        """appearance_mode setting should be registered in SettingsRegistry."""
        from src.ui.settings.settings_registry import SettingsRegistry

        categories = SettingsRegistry.get_categories()
        assert "Appearance" in categories, "Appearance category should exist"

        settings = SettingsRegistry.get_settings_for_category("Appearance")
        keys = [s.key for s in settings]
        assert "appearance_mode" in keys, "appearance_mode should be registered"

    def test_appearance_mode_has_dark_light_options(self):
        """appearance_mode should offer Dark, Light, and System options."""
        from src.ui.settings.settings_registry import SettingsRegistry

        settings = SettingsRegistry.get_settings_for_category("Appearance")
        theme_setting = next(s for s in settings if s.key == "appearance_mode")
        option_values = [opt[1] if isinstance(opt, tuple) else opt for opt in theme_setting.options]
        assert "Dark" in option_values
        assert "Light" in option_values
        assert theme_setting.default == "Dark"


class TestStartupAppearance:
    """Startup should read appearance_mode from preferences."""

    def test_default_appearance_is_dark(self):
        """Default appearance_mode preference should be 'Dark'."""
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        mode = prefs.get("appearance_mode", "Dark")
        assert mode == "Dark"


class TestReinitializeStyles:
    """reinitialize_styles() should exist for live theme switching."""

    def test_reinitialize_styles_exists(self):
        """styles.py should export reinitialize_styles function."""
        from src.ui.styles import reinitialize_styles

        assert callable(reinitialize_styles)
