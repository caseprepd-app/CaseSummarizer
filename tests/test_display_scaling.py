"""
Tests for display scaling functionality.

Covers all code changes from the Universal Display Scaling implementation:
- src/ui/scaling.py (new): scale_value, get_effective_font_offset, get_effective_ui_scale, apply_scaling
- src/ui/theme.py: scale_fonts(offset: int) with 8pt floor
- src/ui/styles.py: initialize_all_styles(scale_factor)
- src/config.py: scale_column_widths(), derived constants
- src/user_preferences.py: validation for font_size_offset, ui_scale_pct
- src/main.py: apply_scaling() call placement
- src/ui/main_window.py: geometry scaling, old scale_fonts removed, styles with scale factor
- src/ui/base_dialog.py: auto-scales dimensions
- src/ui/semantic_question_editor.py: scaled geometry
- src/ui/window_layout.py: scale_value imports in all layout methods
- src/ui/settings/settings_registry.py: new settings, old removed, dialog scaled
"""

import inspect
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ============================================================================
# Helper
# ============================================================================


def _read_source(module_path: str) -> str:
    """Read a source file relative to project root."""
    root = Path(__file__).parent.parent
    return (root / module_path).read_text(encoding="utf-8")


def _make_prefs(data=None):
    """Create a UserPreferencesManager with temporary file."""
    from src.user_preferences import UserPreferencesManager

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(data or {}, tmp)
    tmp.close()
    return UserPreferencesManager(Path(tmp.name))


def _save_and_restore_column_defs():
    """Context manager to save/restore global COLUMN_DEFINITIONS."""
    import src.config as cfg

    original = list(cfg.COLUMN_DEFINITIONS)

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            cfg.COLUMN_DEFINITIONS = original
            cfg.COLUMN_NAMES = tuple(c.name for c in original)
            cfg.PROTECTED_COLUMNS = frozenset(c.name for c in original if not c.can_hide)
            cfg.SORT_WARNING_COLUMNS = frozenset(
                c.name for c in original if c.triggers_sort_warning
            )
            cfg.NUMERIC_COLUMNS = frozenset(c.name for c in original if c.is_numeric)
            cfg.DISPLAY_TO_DATA_KEY = {c.name: c.data_key for c in original if c.name != c.data_key}

    return _Ctx()


# ============================================================================
# 1. src/ui/scaling.py — scale_value()
# ============================================================================


class TestScaleValue:
    """Tests for scale_value() pixel scaling function."""

    def test_at_100_pct(self):
        """Returns original value at 100% scale."""
        with patch("src.ui.scaling.get_effective_ui_scale", return_value=1.0):
            from src.ui.scaling import scale_value

            assert scale_value(100) == 100
            assert scale_value(50) == 50

    def test_at_125_pct(self):
        """Scales up by 1.25 at 125%."""
        with patch("src.ui.scaling.get_effective_ui_scale", return_value=1.25):
            from src.ui.scaling import scale_value

            assert scale_value(100) == 125
            assert scale_value(80) == 100

    def test_at_150_pct(self):
        """Scales up by 1.5 at 150%."""
        with patch("src.ui.scaling.get_effective_ui_scale", return_value=1.5):
            from src.ui.scaling import scale_value

            assert scale_value(100) == 150
            assert scale_value(200) == 300

    def test_at_75_pct(self):
        """Scales down at 75%."""
        with patch("src.ui.scaling.get_effective_ui_scale", return_value=0.75):
            from src.ui.scaling import scale_value

            assert scale_value(100) == 75

    def test_at_200_pct(self):
        """Doubles at 200%."""
        with patch("src.ui.scaling.get_effective_ui_scale", return_value=2.0):
            from src.ui.scaling import scale_value

            assert scale_value(100) == 200

    def test_returns_int(self):
        """Always returns an integer (truncated, not rounded)."""
        with patch("src.ui.scaling.get_effective_ui_scale", return_value=1.33):
            from src.ui.scaling import scale_value

            result = scale_value(100)
            assert isinstance(result, int)
            assert result > 0  # scaled pixel values should be positive
            assert result == 133

    def test_zero_input(self):
        """Handles zero input."""
        with patch("src.ui.scaling.get_effective_ui_scale", return_value=1.5):
            from src.ui.scaling import scale_value

            assert scale_value(0) == 0

    def test_negative_input(self):
        """Handles negative input (shouldn't happen, but doesn't crash)."""
        with patch("src.ui.scaling.get_effective_ui_scale", return_value=1.5):
            from src.ui.scaling import scale_value

            assert scale_value(-10) == -15


# ============================================================================
# 2. src/ui/scaling.py — get_effective_font_offset()
# ============================================================================


class TestGetEffectiveFontOffset:
    """Tests for get_effective_font_offset()."""

    def test_reads_font_size_offset(self):
        """Reads new font_size_offset key from preferences."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.side_effect = lambda key, default=None: {
                "font_size_offset": 4,
            }.get(key, default)

            from src.ui.scaling import get_effective_font_offset

            assert get_effective_font_offset() == 4

    def test_migration_small(self):
        """Old 'small' migrates to -2."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.side_effect = lambda key, default=None: {
                "font_size_offset": None,
                "font_size": "small",
            }.get(key, default)

            from src.ui.scaling import get_effective_font_offset

            assert get_effective_font_offset() == -2

    def test_migration_medium(self):
        """Old 'medium' migrates to 0."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.side_effect = lambda key, default=None: {
                "font_size_offset": None,
                "font_size": "medium",
            }.get(key, default)

            from src.ui.scaling import get_effective_font_offset

            assert get_effective_font_offset() == 0

    def test_migration_large(self):
        """Old 'large' migrates to +2."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.side_effect = lambda key, default=None: {
                "font_size_offset": None,
                "font_size": "large",
            }.get(key, default)

            from src.ui.scaling import get_effective_font_offset

            assert get_effective_font_offset() == 2

    def test_new_key_takes_precedence(self):
        """When font_size_offset exists, old font_size is ignored."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.side_effect = lambda key, default=None: {
                "font_size_offset": 5,
                "font_size": "small",
            }.get(key, default)

            from src.ui.scaling import get_effective_font_offset

            assert get_effective_font_offset() == 5

    def test_no_keys_defaults_to_zero(self):
        """When neither key exists, defaults to 0."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.return_value = None

            from src.ui.scaling import get_effective_font_offset

            assert get_effective_font_offset() == 0

    def test_corrupted_value_returns_zero(self):
        """Non-castable value returns 0."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.side_effect = lambda key, default=None: {
                "font_size_offset": "garbage",
            }.get(key, default)

            from src.ui.scaling import get_effective_font_offset

            assert get_effective_font_offset() == 0

    def test_float_value_truncates(self):
        """Float value is truncated to int."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.side_effect = lambda key, default=None: {
                "font_size_offset": 3.7,
            }.get(key, default)

            from src.ui.scaling import get_effective_font_offset

            assert get_effective_font_offset() == 3

    def test_unknown_old_font_size_defaults_to_zero(self):
        """Unknown old font_size string defaults to 0."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.side_effect = lambda key, default=None: {
                "font_size_offset": None,
                "font_size": "extra-large",
            }.get(key, default)

            from src.ui.scaling import get_effective_font_offset

            assert get_effective_font_offset() == 0


# ============================================================================
# 3. src/ui/scaling.py — get_effective_ui_scale()
# ============================================================================


class TestGetEffectiveUIScale:
    """Tests for get_effective_ui_scale()."""

    def test_default_returns_1(self):
        """Default (100%) returns 1.0."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.return_value = 100
            from src.ui.scaling import get_effective_ui_scale

            assert get_effective_ui_scale() == 1.0

    def test_125_pct(self):
        """125% returns 1.25."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.return_value = 125
            from src.ui.scaling import get_effective_ui_scale

            assert get_effective_ui_scale() == 1.25

    def test_75_pct(self):
        """75% returns 0.75."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.return_value = 75
            from src.ui.scaling import get_effective_ui_scale

            assert get_effective_ui_scale() == 0.75

    def test_200_pct(self):
        """200% returns 2.0."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.return_value = 200
            from src.ui.scaling import get_effective_ui_scale

            assert get_effective_ui_scale() == 2.0

    def test_clamps_low(self):
        """Values below 75 are clamped to 0.75."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.return_value = 10
            from src.ui.scaling import get_effective_ui_scale

            assert get_effective_ui_scale() == 0.75

    def test_clamps_high(self):
        """Values above 200 are clamped to 2.0."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.return_value = 500
            from src.ui.scaling import get_effective_ui_scale

            assert get_effective_ui_scale() == 2.0

    def test_invalid_value_returns_1(self):
        """Non-numeric value returns 1.0."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.return_value = "invalid"
            from src.ui.scaling import get_effective_ui_scale

            assert get_effective_ui_scale() == 1.0

    def test_none_value_returns_1(self):
        """None value returns 1.0."""
        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value.get.return_value = None
            from src.ui.scaling import get_effective_ui_scale

            assert get_effective_ui_scale() == 1.0


# ============================================================================
# 4. src/ui/scaling.py — apply_scaling()
# ============================================================================


class TestApplyScaling:
    """Tests for apply_scaling() integration."""

    def test_calls_scale_fonts_ctk_and_columns(self):
        """apply_scaling calls scale_fonts, set_widget_scaling, scale_column_widths."""
        with (
            patch("src.ui.scaling.get_effective_font_offset", return_value=2),
            patch("src.ui.scaling.get_effective_ui_scale", return_value=1.25),
            patch("src.ui.theme.scale_fonts") as mock_fonts,
            patch("src.ui.scaling.ctk") as mock_ctk,
            patch("src.config.scale_column_widths") as mock_cols,
        ):
            from src.ui.scaling import apply_scaling

            apply_scaling()

            mock_fonts.assert_called_once_with(2)
            mock_ctk.set_widget_scaling.assert_called_once_with(1.25)
            mock_cols.assert_called_once_with(1.25)

    def test_does_not_call_initialize_all_styles(self):
        """apply_scaling does NOT call initialize_all_styles (needs Tk root)."""
        source = _read_source("src/ui/scaling.py")
        # The function body should not call initialize_all_styles
        # (it's called from MainWindow.__init__ instead)
        func_body = source.split("def apply_scaling")[1].split("\ndef ")[0]
        assert "initialize_all_styles" not in func_body.replace(
            "# NOTE: Do NOT call initialize_all_styles", ""
        ).replace("# MainWindow.__init__() calls initialize_all_styles", "")

    def test_apply_scaling_at_default_values(self):
        """At default values (offset=0, scale=100%), still calls all functions."""
        with (
            patch("src.ui.scaling.get_effective_font_offset", return_value=0),
            patch("src.ui.scaling.get_effective_ui_scale", return_value=1.0),
            patch("src.ui.theme.scale_fonts") as mock_fonts,
            patch("src.ui.scaling.ctk") as mock_ctk,
            patch("src.config.scale_column_widths") as mock_cols,
        ):
            from src.ui.scaling import apply_scaling

            apply_scaling()

            mock_fonts.assert_called_once_with(0)
            mock_ctk.set_widget_scaling.assert_called_once_with(1.0)
            mock_cols.assert_called_once_with(1.0)


# ============================================================================
# 5. src/ui/theme.py — scale_fonts()
# ============================================================================


class TestScaleFonts:
    """Tests for theme.scale_fonts() with integer offset."""

    def test_zero_offset_unchanged(self):
        """Zero offset leaves fonts unchanged."""
        from src.ui.theme import _BASE_FONTS, FONTS, scale_fonts

        scale_fonts(0)
        for key, base in _BASE_FONTS.items():
            assert FONTS[key][1] == base[1]

    def test_positive_offset(self):
        """Positive offset increases font sizes."""
        from src.ui.theme import _BASE_FONTS, FONTS, scale_fonts

        scale_fonts(4)
        for key, base in _BASE_FONTS.items():
            assert FONTS[key][1] == max(8, base[1] + 4)

    def test_negative_offset(self):
        """Negative offset decreases font sizes."""
        from src.ui.theme import _BASE_FONTS, FONTS, scale_fonts

        scale_fonts(-2)
        for key, base in _BASE_FONTS.items():
            assert FONTS[key][1] == max(8, base[1] - 2)

    def test_8pt_floor(self):
        """Font sizes never go below 8pt even with extreme negative offset."""
        from src.ui.theme import FONTS, scale_fonts

        scale_fonts(-20)
        for key, font in FONTS.items():
            assert font[1] >= 8, f"Font '{key}' has size {font[1]} < 8"

    def test_preserves_family(self):
        """Font family is preserved."""
        from src.ui.theme import _BASE_FONTS, FONTS, scale_fonts

        scale_fonts(2)
        for key, base in _BASE_FONTS.items():
            assert FONTS[key][0] == base[0]

    def test_preserves_weight(self):
        """Font weight/style is preserved."""
        from src.ui.theme import _BASE_FONTS, FONTS, scale_fonts

        scale_fonts(2)
        for key, base in _BASE_FONTS.items():
            if len(base) == 3:
                assert len(FONTS[key]) == 3
                assert FONTS[key][2] == base[2]

    def test_two_element_fonts_stay_two(self):
        """Fonts without weight stay as 2-tuples."""
        from src.ui.theme import _BASE_FONTS, FONTS, scale_fonts

        scale_fonts(2)
        for key, base in _BASE_FONTS.items():
            if len(base) == 2:
                assert len(FONTS[key]) == 2

    def test_large_positive_offset(self):
        """Large positive offset works (max +10)."""
        from src.ui.theme import _BASE_FONTS, FONTS, scale_fonts

        scale_fonts(10)
        for key, base in _BASE_FONTS.items():
            assert FONTS[key][1] == base[1] + 10

    def test_resets_from_base_not_cumulative(self):
        """Calling scale_fonts again resets from base sizes, not cumulative."""
        from src.ui.theme import _BASE_FONTS, FONTS, scale_fonts

        scale_fonts(5)
        scale_fonts(0)
        for key, base in _BASE_FONTS.items():
            assert FONTS[key][1] == base[1]

    def test_signature_accepts_int(self):
        """scale_fonts has an int parameter, not a string."""
        sig = inspect.signature(__import__("src.ui.theme", fromlist=["scale_fonts"]).scale_fonts)
        param = list(sig.parameters.values())[0]
        assert param.name == "offset"
        assert param.default == 0

    def test_old_string_keys_still_defined(self):
        """FONT_SIZE_OPTIONS and FONT_SIZE_OFFSETS exist (deprecated but kept for migration)."""
        from src.ui.theme import FONT_SIZE_OFFSETS, FONT_SIZE_OPTIONS

        assert len(FONT_SIZE_OPTIONS) == 3
        assert FONT_SIZE_OFFSETS == {"small": -2, "medium": 0, "large": 2}


# ============================================================================
# 6. src/ui/styles.py — initialize_all_styles(scale_factor, font_offset)
# ============================================================================


class TestInitializeAllStyles:
    """Tests for styles.initialize_all_styles() with scale factor and font offset."""

    def test_accepts_scale_factor_kwarg(self):
        """initialize_all_styles accepts scale_factor parameter."""
        import src.ui.styles as styles_mod

        styles_mod._styles_initialized = False
        try:
            styles_mod.initialize_all_styles(scale_factor=1.5)
        except Exception:
            pass  # No Tk root
        finally:
            styles_mod._styles_initialized = False

    def test_accepts_font_offset_kwarg(self):
        """initialize_all_styles accepts font_offset parameter."""
        import src.ui.styles as styles_mod

        styles_mod._styles_initialized = False
        try:
            styles_mod.initialize_all_styles(scale_factor=1.0, font_offset=4)
        except Exception:
            pass  # No Tk root
        finally:
            styles_mod._styles_initialized = False

    def test_default_scale_factor_is_one(self):
        """Default scale factor is 1.0."""
        from src.ui.styles import initialize_all_styles

        sig = inspect.signature(initialize_all_styles)
        assert sig.parameters["scale_factor"].default == 1.0

    def test_default_font_offset_is_zero(self):
        """Default font_offset is 0."""
        from src.ui.styles import initialize_all_styles

        sig = inspect.signature(initialize_all_styles)
        assert sig.parameters["font_offset"].default == 0

    def test_internal_helpers_accept_scale_and_offset(self):
        """All _configure_* helpers accept sf and font_offset parameters."""
        import src.ui.styles as mod

        for name in [
            "_configure_vocab_treeview_style",
            "_configure_qa_table_style",
            "_configure_file_review_style",
            "_configure_question_list_style",
        ]:
            func = getattr(mod, name)
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            assert "sf" in params, f"{name} missing 'sf' parameter"
            assert "font_offset" in params, f"{name} missing 'font_offset' parameter"

    def test_noop_after_first_call(self):
        """Second call is a no-op (guard flag)."""
        import src.ui.styles as styles_mod

        styles_mod._styles_initialized = True
        try:
            # Should return immediately without error (even without Tk root)
            styles_mod.initialize_all_styles(scale_factor=2.0)
        finally:
            styles_mod._styles_initialized = False


# ============================================================================
# 7. src/config.py — scale_column_widths()
# ============================================================================


class TestScaleColumnWidths:
    """Tests for config.scale_column_widths()."""

    def test_1x_is_noop(self):
        """Scaling at 1.0 does nothing."""
        from src.config import COLUMN_DEFINITIONS, scale_column_widths

        original_widths = [c.width for c in COLUMN_DEFINITIONS]
        scale_column_widths(1.0)
        assert [c.width for c in COLUMN_DEFINITIONS] == original_widths

    def test_1_5x_multiplies_widths(self):
        """Scaling at 1.5x multiplies widths."""
        import src.config as cfg

        original_widths = [c.width for c in cfg.COLUMN_DEFINITIONS]
        with _save_and_restore_column_defs():
            cfg.scale_column_widths(1.5)
            for i, c in enumerate(cfg.COLUMN_DEFINITIONS):
                assert c.width == int(original_widths[i] * 1.5)

    def test_0_75x_shrinks_widths(self):
        """Scaling at 0.75x shrinks widths."""
        import src.config as cfg

        original_widths = [c.width for c in cfg.COLUMN_DEFINITIONS]
        with _save_and_restore_column_defs():
            cfg.scale_column_widths(0.75)
            for i, c in enumerate(cfg.COLUMN_DEFINITIONS):
                assert c.width == int(original_widths[i] * 0.75)

    def test_preserves_column_count(self):
        """Scaling preserves the number of columns."""
        import src.config as cfg

        count = len(cfg.COLUMN_DEFINITIONS)
        with _save_and_restore_column_defs():
            cfg.scale_column_widths(1.25)
            assert len(cfg.COLUMN_DEFINITIONS) == count

    def test_rebuilds_COLUMN_NAMES(self):
        """Scaling rebuilds COLUMN_NAMES tuple."""
        import src.config as cfg

        with _save_and_restore_column_defs():
            cfg.scale_column_widths(1.25)
            assert len(cfg.COLUMN_NAMES) == len(cfg.COLUMN_DEFINITIONS)
            assert all(c.name in cfg.COLUMN_NAMES for c in cfg.COLUMN_DEFINITIONS)

    def test_rebuilds_PROTECTED_COLUMNS(self):
        """Scaling rebuilds PROTECTED_COLUMNS frozenset."""
        import src.config as cfg

        with _save_and_restore_column_defs():
            cfg.scale_column_widths(1.5)
            expected = frozenset(c.name for c in cfg.COLUMN_DEFINITIONS if not c.can_hide)
            assert expected == cfg.PROTECTED_COLUMNS

    def test_rebuilds_NUMERIC_COLUMNS(self):
        """Scaling rebuilds NUMERIC_COLUMNS frozenset."""
        import src.config as cfg

        with _save_and_restore_column_defs():
            cfg.scale_column_widths(1.5)
            expected = frozenset(c.name for c in cfg.COLUMN_DEFINITIONS if c.is_numeric)
            assert expected == cfg.NUMERIC_COLUMNS

    def test_rebuilds_DISPLAY_TO_DATA_KEY(self):
        """Scaling rebuilds DISPLAY_TO_DATA_KEY dict."""
        import src.config as cfg

        with _save_and_restore_column_defs():
            cfg.scale_column_widths(1.5)
            expected = {c.name: c.data_key for c in cfg.COLUMN_DEFINITIONS if c.name != c.data_key}
            assert expected == cfg.DISPLAY_TO_DATA_KEY

    def test_preserves_non_width_fields(self):
        """Scaling only changes width, not other fields."""
        import src.config as cfg

        originals = list(cfg.COLUMN_DEFINITIONS)
        with _save_and_restore_column_defs():
            cfg.scale_column_widths(2.0)
            for i, c in enumerate(cfg.COLUMN_DEFINITIONS):
                o = originals[i]
                assert c.name == o.name
                assert c.data_key == o.data_key
                assert c.max_chars == o.max_chars
                assert c.default_visible == o.default_visible
                assert c.can_hide == o.can_hide
                assert c.triggers_sort_warning == o.triggers_sort_warning
                assert c.is_numeric == o.is_numeric

    def test_get_column_by_name_works_after_scaling(self):
        """get_column_by_name still works after scaling."""
        import src.config as cfg

        with _save_and_restore_column_defs():
            cfg.scale_column_widths(1.5)
            col = cfg.get_column_by_name("Term")
            assert col is not None
            assert col.name == "Term"
            assert col.width == int(180 * 1.5)

    def test_build_column_registry_after_scaling(self):
        """build_column_registry returns scaled widths."""
        import src.config as cfg

        with _save_and_restore_column_defs():
            cfg.scale_column_widths(1.5)
            registry = cfg.build_column_registry()
            assert registry["Term"]["width"] == int(180 * 1.5)
            assert registry["Score"]["width"] == int(55 * 1.5)


# ============================================================================
# 8. src/user_preferences.py — validation
# ============================================================================


class TestPreferenceValidation:
    """Tests for font_size_offset and ui_scale_pct validation."""

    def test_font_size_offset_valid_range(self):
        """Valid font_size_offset values (-4 to 10) are accepted."""
        prefs = _make_prefs()
        for val in [-4, -2, 0, 2, 5, 10]:
            prefs.set("font_size_offset", val)
            assert prefs.get("font_size_offset") == val

    def test_font_size_offset_rejects_below_minus_4(self):
        """font_size_offset below -4 is rejected."""
        prefs = _make_prefs()
        with pytest.raises(ValueError, match="font_size_offset"):
            prefs.set("font_size_offset", -5)

    def test_font_size_offset_rejects_above_10(self):
        """font_size_offset above 10 is rejected."""
        prefs = _make_prefs()
        with pytest.raises(ValueError, match="font_size_offset"):
            prefs.set("font_size_offset", 11)

    def test_font_size_offset_rejects_float(self):
        """font_size_offset rejects float values."""
        prefs = _make_prefs()
        with pytest.raises(ValueError, match="font_size_offset"):
            prefs.set("font_size_offset", 2.5)

    def test_font_size_offset_rejects_string(self):
        """font_size_offset rejects string values."""
        prefs = _make_prefs()
        with pytest.raises(ValueError, match="font_size_offset"):
            prefs.set("font_size_offset", "medium")

    def test_font_size_offset_boundary_minus_4(self):
        """font_size_offset accepts boundary value -4."""
        prefs = _make_prefs()
        prefs.set("font_size_offset", -4)
        assert prefs.get("font_size_offset") == -4

    def test_font_size_offset_boundary_10(self):
        """font_size_offset accepts boundary value 10."""
        prefs = _make_prefs()
        prefs.set("font_size_offset", 10)
        assert prefs.get("font_size_offset") == 10

    def test_ui_scale_pct_valid_range(self):
        """Valid ui_scale_pct values (75 to 200) are accepted."""
        prefs = _make_prefs()
        for val in [75, 100, 125, 150, 175, 200]:
            prefs.set("ui_scale_pct", val)
            assert prefs.get("ui_scale_pct") == val

    def test_ui_scale_pct_rejects_below_75(self):
        """ui_scale_pct below 75 is rejected."""
        prefs = _make_prefs()
        with pytest.raises(ValueError, match="ui_scale_pct"):
            prefs.set("ui_scale_pct", 50)

    def test_ui_scale_pct_rejects_above_200(self):
        """ui_scale_pct above 200 is rejected."""
        prefs = _make_prefs()
        with pytest.raises(ValueError, match="ui_scale_pct"):
            prefs.set("ui_scale_pct", 250)

    def test_ui_scale_pct_rejects_float(self):
        """ui_scale_pct rejects float values."""
        prefs = _make_prefs()
        with pytest.raises(ValueError, match="ui_scale_pct"):
            prefs.set("ui_scale_pct", 125.5)

    def test_ui_scale_pct_accepts_non_step_values(self):
        """ui_scale_pct accepts any int in range, not just step multiples."""
        prefs = _make_prefs()
        prefs.set("ui_scale_pct", 137)
        assert prefs.get("ui_scale_pct") == 137

    def test_ui_scale_pct_boundary_75(self):
        """ui_scale_pct accepts boundary value 75."""
        prefs = _make_prefs()
        prefs.set("ui_scale_pct", 75)
        assert prefs.get("ui_scale_pct") == 75

    def test_ui_scale_pct_boundary_200(self):
        """ui_scale_pct accepts boundary value 200."""
        prefs = _make_prefs()
        prefs.set("ui_scale_pct", 200)
        assert prefs.get("ui_scale_pct") == 200

    def test_both_preferences_persist(self):
        """Both scaling preferences can be set and read back."""
        prefs = _make_prefs()
        prefs.set("font_size_offset", 3)
        prefs.set("ui_scale_pct", 150)
        assert prefs.get("font_size_offset") == 3
        assert prefs.get("ui_scale_pct") == 150


# ============================================================================
# 9. src/main.py — apply_scaling() call placement
# ============================================================================


class TestMainPyScalingCall:
    """Verify apply_scaling() is called in main.py at the right place."""

    def test_apply_scaling_imported_in_main(self):
        """main.py imports apply_scaling from src.ui.scaling."""
        source = _read_source("src/main.py")
        assert "from src.ui.scaling import apply_scaling" in source

    def test_apply_scaling_called_before_mainwindow(self):
        """apply_scaling() is called before MainWindow() in main()."""
        source = _read_source("src/main.py")
        apply_pos = source.index("apply_scaling()")
        mainwindow_pos = source.index("app = MainWindow(")
        assert apply_pos < mainwindow_pos

    def test_apply_scaling_called_after_color_theme(self):
        """apply_scaling() is called after set_default_color_theme()."""
        source = _read_source("src/main.py")
        theme_pos = source.index('set_default_color_theme("blue")')
        apply_pos = source.index("apply_scaling()")
        assert theme_pos < apply_pos


# ============================================================================
# 10. src/ui/main_window.py — geometry scaling, old call removed
# ============================================================================


class TestMainWindowChanges:
    """Verify main_window.py changes."""

    def test_geometry_uses_scale_value(self):
        """MainWindow geometry uses scale_value(), not hardcoded '1200x750'."""
        source = _read_source("src/ui/main_window.py")
        assert 'geometry("1200x750")' not in source
        assert "scale_value(1200)" in source
        assert "scale_value(750)" in source

    def test_minsize_uses_scale_value(self):
        """MainWindow minsize uses scale_value(), not hardcoded 900, 600."""
        source = _read_source("src/ui/main_window.py")
        assert "minsize(900, 600)" not in source
        assert "scale_value(900)" in source
        assert "scale_value(600)" in source

    def test_old_scale_fonts_call_removed(self):
        """Old scale_fonts(_prefs.get('font_size', 'medium')) call is gone."""
        source = _read_source("src/ui/main_window.py")
        assert 'scale_fonts(_prefs.get("font_size"' not in source
        assert "scale_fonts(_prefs.get('font_size'" not in source

    def test_initialize_all_styles_gets_scale_and_offset(self):
        """initialize_all_styles is called with both scale and font offset."""
        source = _read_source("src/ui/main_window.py")
        assert (
            "initialize_all_styles(get_effective_ui_scale(), get_effective_font_offset())" in source
        )

    def test_imports_scaling_functions(self):
        """MainWindow imports both get_effective_ui_scale and get_effective_font_offset."""
        source = _read_source("src/ui/main_window.py")
        assert "get_effective_ui_scale" in source
        assert "get_effective_font_offset" in source


# ============================================================================
# 11. src/ui/base_dialog.py — auto-scales dimensions
# ============================================================================


class TestBaseDialogScaling:
    """Verify base_dialog.py scales width/height/min_width/min_height."""

    def test_imports_scale_value(self):
        """base_dialog.py uses scale_value."""
        source = _read_source("src/ui/base_dialog.py")
        assert "scale_value" in source

    def test_scales_width_and_height(self):
        """geometry uses scaled width and height."""
        source = _read_source("src/ui/base_dialog.py")
        assert "width = scale_value(width)" in source
        assert "height = scale_value(height)" in source

    def test_scales_minsize(self):
        """minsize uses scale_value."""
        source = _read_source("src/ui/base_dialog.py")
        assert "scale_value(min_width)" in source
        assert "scale_value(min_height)" in source


# ============================================================================
# 13. src/ui/semantic_question_editor.py — scaled geometry
# ============================================================================


class TestQAQuestionEditorScaling:
    """Verify semantic_question_editor.py uses scale_value for geometry."""

    def test_geometry_scaled(self):
        """Question editor geometry uses scale_value, not hardcoded '500x300'."""
        source = _read_source("src/ui/semantic_question_editor.py")
        assert 'geometry("500x300")' not in source
        assert "scale_value(500)" in source
        assert "scale_value(300)" in source

    def test_centering_uses_scaled_values(self):
        """Centering calculation uses scaled width/height variables."""
        source = _read_source("src/ui/semantic_question_editor.py")
        # Should use _w and _h variables, not hardcoded 500/300
        assert "- 500)" not in source
        assert "- 300)" not in source


# ============================================================================
# 14. src/ui/window_layout.py — scale_value in all methods
# ============================================================================


class TestWindowLayoutScaling:
    """Verify window_layout.py imports and uses scale_value in all layout methods."""

    def _get_method_source(self, method_name: str) -> str:
        """Extract a single method's source from WindowLayoutMixin."""
        from src.ui.window_layout import WindowLayoutMixin

        method = getattr(WindowLayoutMixin, method_name)
        return inspect.getsource(method)

    def test_create_header_imports_scale_value(self):
        """_create_header has its own scale_value import."""
        src = self._get_method_source("_create_header")
        assert "from src.ui.scaling import scale_value" in src

    def test_create_left_panel_imports_scale_value(self):
        """_create_left_panel has its own scale_value import."""
        src = self._get_method_source("_create_left_panel")
        assert "from src.ui.scaling import scale_value" in src

    def test_create_right_panel_imports_scale_value(self):
        """_create_right_panel has its own scale_value import."""
        src = self._get_method_source("_create_right_panel")
        assert "from src.ui.scaling import scale_value" in src

    def test_create_status_bar_imports_scale_value(self):
        """_create_status_bar has a scale_value import."""
        src = self._get_method_source("_create_status_bar")
        assert "scaling import scale_value" in src

    def test_header_frame_height_scaled(self):
        """Header frame height=50 is scaled."""
        src = self._get_method_source("_create_header")
        assert "scale_value(50)" in src

    def test_settings_button_scaled(self):
        """Settings button width is scaled."""
        src = self._get_method_source("_create_header")
        assert "scale_value(100)" in src

    def test_corpus_dropdown_scaled(self):
        """Corpus dropdown width is scaled."""
        src = self._get_method_source("_create_header")
        assert "scale_value(150)" in src

    def test_manage_button_scaled(self):
        """Manage button width is scaled."""
        src = self._get_method_source("_create_header")
        assert "scale_value(70)" in src

    def test_add_files_button_scaled(self):
        """Add Files button width is scaled."""
        src = self._get_method_source("_create_left_panel")
        assert "scale_value(100)" in src

    def test_clear_all_button_scaled(self):
        """Clear All button width is scaled."""
        src = self._get_method_source("_create_left_panel")
        assert "scale_value(80)" in src

    def test_generate_button_height_scaled(self):
        """Generate button height is scaled."""
        src = self._get_method_source("_create_left_panel")
        assert "scale_value(40)" in src

    def test_followup_entry_height_scaled(self):
        """Followup entry height is scaled."""
        src = self._get_method_source("_create_right_panel")
        assert "scale_value(35)" in src

    def test_followup_button_width_scaled(self):
        """Followup button width is scaled."""
        src = self._get_method_source("_create_right_panel")
        assert "scale_value(60)" in src

    def test_status_bar_height_scaled(self):
        """Status bar frame height is scaled."""
        src = self._get_method_source("_create_status_bar")
        # Uses _sv alias
        assert "_sv(30)" in src

    def test_export_button_scaled(self):
        """Export All button dimensions are scaled."""
        src = self._get_method_source("_create_status_bar")
        assert "_sv(90)" in src
        assert "_sv(24)" in src

    def test_no_unscaled_width_heights_in_header(self):
        """No hardcoded width= or height= values remain unscaled in _create_header."""
        src = self._get_method_source("_create_header")
        import re

        # Find width=<int> patterns that aren't inside scale_value()
        # Exclude width=0 and border_width and corner_radius
        matches = re.findall(r"(?<!scale_value\()(?<!corner_radius=)\bwidth=(\d+)", src)
        # Filter out 0 values (border_width=0, etc.)
        nonzero = [m for m in matches if int(m) > 0]
        assert nonzero == [], f"Unscaled widths in _create_header: {nonzero}"


# ============================================================================
# 15. src/ui/settings/settings_registry.py — new settings, old removed
# ============================================================================


class TestSettingsRegistryChanges:
    """Verify settings_registry.py has new scaling settings."""

    def test_old_font_size_dropdown_removed(self):
        """Old font_size DROPDOWN setting is gone."""
        source = _read_source("src/ui/settings/settings_registry.py")
        assert 'key="font_size"' not in source

    def test_font_size_offset_spinbox_registered(self):
        """font_size_offset is registered as a SPINBOX."""
        source = _read_source("src/ui/settings/settings_registry.py")
        assert 'key="font_size_offset"' in source
        # Check it's in the SPINBOX type
        idx = source.index('key="font_size_offset"')
        nearby = source[max(0, idx - 200) : idx + 200]
        assert "SPINBOX" in nearby

    def test_ui_scale_pct_slider_registered(self):
        """ui_scale_pct is registered as a SLIDER."""
        source = _read_source("src/ui/settings/settings_registry.py")
        assert 'key="ui_scale_pct"' in source
        idx = source.index('key="ui_scale_pct"')
        nearby = source[max(0, idx - 200) : idx + 200]
        assert "SLIDER" in nearby

    def test_font_size_offset_range(self):
        """font_size_offset range is -4 to 10."""
        source = _read_source("src/ui/settings/settings_registry.py")
        idx = source.index('key="font_size_offset"')
        nearby = source[idx : idx + 700]
        assert "min_value=-4" in nearby
        assert "max_value=10" in nearby

    def test_ui_scale_pct_range(self):
        """ui_scale_pct range is 75 to 200, step 25."""
        source = _read_source("src/ui/settings/settings_registry.py")
        idx = source.index('key="ui_scale_pct"')
        nearby = source[idx : idx + 900]
        assert "min_value=75" in nearby
        assert "max_value=200" in nearby
        assert "step=25" in nearby

    def test_both_settings_in_appearance_category(self):
        """Both scaling settings are in the Appearance category."""
        source = _read_source("src/ui/settings/settings_registry.py")

        idx1 = source.index('key="font_size_offset"')
        nearby1 = source[max(0, idx1 - 100) : idx1 + 100]
        assert '"Appearance"' in nearby1

        idx2 = source.index('key="ui_scale_pct"')
        nearby2 = source[max(0, idx2 - 100) : idx2 + 100]
        assert '"Appearance"' in nearby2

    def test_ui_scale_tooltip_mentions_restart(self):
        """UI scale setting mentions restart (font applies immediately)."""
        source = _read_source("src/ui/settings/settings_registry.py")

        idx2 = source.index('key="ui_scale_pct"')
        block2 = source[idx2 : idx2 + 900]
        assert "restart" in block2.lower()

    def test_font_applies_immediately(self):
        """Font size setting applies live (no restart mention)."""
        source = _read_source("src/ui/settings/settings_registry.py")

        idx1 = source.index('key="font_size_offset"')
        block1 = source[idx1 : idx1 + 600]
        assert "restart" not in block1.lower()

    def test_no_FONT_SIZE_OPTIONS_import(self):
        """settings_registry no longer imports FONT_SIZE_OPTIONS."""
        source = _read_source("src/ui/settings/settings_registry.py")
        assert "FONT_SIZE_OPTIONS" not in source
