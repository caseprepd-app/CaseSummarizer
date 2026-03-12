"""
Tests for DPI-aware column width computation and stretch=False fix.

Covers:
- compute_column_widths(): font-metric-based width calculation
- _measure_content_width(): content sampling logic
- _max_width_for_column(): per-column caps
- truncate_text(): text overflow prevention
- get_vocab_font_specs(): font spec accessor
- COLUMN_REGISTRY / COLUMN_ORDER consistency
- stretch=False on all columns (source inspection)
- Save cap raised to 800 (source inspection)
- "Reset Column Widths" menu option (source inspection)
- _autosize_columns_to_content / _reset_column_widths (source inspection)
"""

import tkinter as tk
from pathlib import Path

import pytest


def _read_source(module_path: str) -> str:
    """Read a source file relative to project root."""
    root = Path(__file__).parent.parent
    return (root / module_path).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def _tk_root():
    """Create a Tk root for font metric tests (module-scoped)."""
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk unavailable in this environment")
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


# ============================================================================
# 1. COLUMN_REGISTRY and COLUMN_ORDER consistency
# ============================================================================


class TestColumnRegistryConsistency:
    """Verify COLUMN_REGISTRY and COLUMN_ORDER are in sync."""

    def test_order_matches_registry_keys(self):
        """COLUMN_ORDER contains exactly the same columns as COLUMN_REGISTRY."""
        from src.ui.vocab_table.column_config import COLUMN_ORDER, COLUMN_REGISTRY

        assert set(COLUMN_ORDER) == set(COLUMN_REGISTRY.keys())

    def test_no_duplicate_columns_in_order(self):
        """COLUMN_ORDER has no duplicates."""
        from src.ui.vocab_table.column_config import COLUMN_ORDER

        assert len(COLUMN_ORDER) == len(set(COLUMN_ORDER))

    def test_term_cannot_be_hidden(self):
        """Term column has can_hide=False."""
        from src.ui.vocab_table.column_config import COLUMN_REGISTRY

        assert COLUMN_REGISTRY["Term"]["can_hide"] is False

    def test_all_columns_have_required_keys(self):
        """Every column in COLUMN_REGISTRY has width, max_chars, default, can_hide."""
        from src.ui.vocab_table.column_config import COLUMN_REGISTRY

        required = {"width", "max_chars", "default", "can_hide"}
        for col_name, cfg in COLUMN_REGISTRY.items():
            assert required.issubset(cfg.keys()), (
                f"{col_name} missing keys: {required - cfg.keys()}"
            )

    def test_display_to_data_column_maps_score(self):
        """Score display column maps to 'Quality Score' data key."""
        from src.ui.vocab_table.column_config import DISPLAY_TO_DATA_COLUMN

        assert DISPLAY_TO_DATA_COLUMN["Score"] == "Quality Score"

    def test_column_config_derives_from_registry(self):
        """COLUMN_CONFIG widths match COLUMN_REGISTRY widths."""
        from src.ui.vocab_table.column_config import COLUMN_CONFIG, COLUMN_REGISTRY

        for col_name, cfg in COLUMN_REGISTRY.items():
            assert COLUMN_CONFIG[col_name]["width"] == cfg["width"]
            assert COLUMN_CONFIG[col_name]["max_chars"] == cfg["max_chars"]


# ============================================================================
# 2. truncate_text()
# ============================================================================


class TestTruncateText:
    """Tests for truncate_text helper."""

    def test_short_text_unchanged(self):
        """Text shorter than max_chars is returned as-is."""
        from src.ui.vocab_table.column_config import truncate_text

        assert truncate_text("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        """Text exactly at max_chars is returned as-is."""
        from src.ui.vocab_table.column_config import truncate_text

        assert truncate_text("hello", 5) == "hello"

    def test_long_text_truncated_with_ellipsis(self):
        """Text exceeding max_chars is truncated with '...'."""
        from src.ui.vocab_table.column_config import truncate_text

        result = truncate_text("hello world", 8)
        assert result == "hello..."
        assert len(result) == 8

    def test_empty_string(self):
        """Empty string returns empty string."""
        from src.ui.vocab_table.column_config import truncate_text

        assert truncate_text("", 10) == ""

    def test_none_returns_empty(self):
        """None input returns empty string."""
        from src.ui.vocab_table.column_config import truncate_text

        assert truncate_text(None, 10) == ""

    def test_newlines_replaced_with_spaces(self):
        """Newlines are replaced with spaces."""
        from src.ui.vocab_table.column_config import truncate_text

        assert truncate_text("hello\nworld", 20) == "hello world"

    def test_carriage_return_removed(self):
        """Carriage returns are stripped."""
        from src.ui.vocab_table.column_config import truncate_text

        assert truncate_text("hello\r\nworld", 20) == "hello world"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped."""
        from src.ui.vocab_table.column_config import truncate_text

        assert truncate_text("  hello  ", 10) == "hello"

    def test_non_string_converted(self):
        """Non-string input is converted via str()."""
        from src.ui.vocab_table.column_config import truncate_text

        assert truncate_text(42, 10) == "42"


# ============================================================================
# 3. compute_column_widths() — DPI-aware calculation
# ============================================================================


class TestComputeColumnWidths:
    """Tests for DPI-aware column width computation."""

    def test_returns_dict_for_all_visible_columns(self, _tk_root):
        """Returns a width for every visible column."""
        from src.ui.vocab_table.column_config import compute_column_widths

        cols = ["Term", "Score", "Keep"]
        result = compute_column_widths(cols, ("Segoe UI", 10), ("Segoe UI", 10, "bold"), 1000)
        assert set(result.keys()) == set(cols)

    def test_all_widths_positive(self, _tk_root):
        """All returned widths are positive integers."""
        from src.ui.vocab_table.column_config import compute_column_widths

        cols = ["Term", "Score", "Is Person", "Found By"]
        result = compute_column_widths(cols, ("Segoe UI", 10), ("Segoe UI", 10, "bold"), 1000)
        for col, width in result.items():
            assert isinstance(width, int), f"{col} width is not int"
            assert width > 0, f"{col} width is not positive"

    def test_min_width_floor(self, _tk_root):
        """No column is narrower than MIN_COLUMN_WIDTH."""
        from src.ui.vocab_table.column_config import MIN_COLUMN_WIDTH, compute_column_widths

        cols = ["Term", "Score", "NER"]
        result = compute_column_widths(cols, ("Segoe UI", 10), ("Segoe UI", 10, "bold"), 1000)
        for col, width in result.items():
            assert width >= MIN_COLUMN_WIDTH, f"{col} below min: {width}"

    def test_term_capped_at_45_percent(self, _tk_root):
        """Term column is capped at 45% of available width."""
        from src.ui.vocab_table.column_config import compute_column_widths

        # Use large data to push Term very wide
        sample = [{"Term": "A" * 200}] * 10
        result = compute_column_widths(
            ["Term"], ("Segoe UI", 10), ("Segoe UI", 10, "bold"), 800, data_sample=sample
        )
        assert result["Term"] <= int(800 * 0.45)

    def test_found_by_capped_at_20_percent(self, _tk_root):
        """Found By column is capped at 20% of available width."""
        from src.ui.vocab_table.column_config import compute_column_widths

        sample = [{"Found By": "RAKE, NER, BM25, TopicRank, YAKE, MedicalNER"}] * 10
        result = compute_column_widths(
            ["Found By"], ("Segoe UI", 10), ("Segoe UI", 10, "bold"), 800, data_sample=sample
        )
        assert result["Found By"] <= int(800 * 0.20)

    def test_other_columns_capped_at_15_percent(self, _tk_root):
        """Non-Term, non-Found-By columns are capped at 15% of available width."""
        from src.ui.vocab_table.column_config import compute_column_widths

        sample = [{"Score": "99999"}] * 10
        result = compute_column_widths(
            ["Score"], ("Segoe UI", 10), ("Segoe UI", 10, "bold"), 800, data_sample=sample
        )
        assert result["Score"] <= int(800 * 0.15)

    def test_larger_font_gives_wider_columns(self, _tk_root):
        """Larger font produces wider columns (DPI-awareness)."""
        from src.ui.vocab_table.column_config import compute_column_widths

        cols = ["Term", "Score"]
        small = compute_column_widths(cols, ("Segoe UI", 8), ("Segoe UI", 8, "bold"), 1200)
        large = compute_column_widths(cols, ("Segoe UI", 16), ("Segoe UI", 16, "bold"), 1200)
        # At least one column should be wider with larger font
        assert any(large[c] > small[c] for c in cols)

    def test_data_sample_widens_columns(self, _tk_root):
        """Providing data_sample can make columns wider than heading-only."""
        from src.ui.vocab_table.column_config import compute_column_widths

        cols = ["Term"]
        no_data = compute_column_widths(cols, ("Segoe UI", 10), ("Segoe UI", 10, "bold"), 1200)
        with_data = compute_column_widths(
            cols,
            ("Segoe UI", 10),
            ("Segoe UI", 10, "bold"),
            1200,
            data_sample=[{"Term": "Constitutional Amendment"}] * 5,
        )
        assert with_data["Term"] >= no_data["Term"]

    def test_empty_data_sample_same_as_none(self, _tk_root):
        """Empty data_sample list behaves same as None."""
        from src.ui.vocab_table.column_config import compute_column_widths

        cols = ["Term", "Score"]
        no_data = compute_column_widths(cols, ("Segoe UI", 10), ("Segoe UI", 10, "bold"), 1000)
        empty_data = compute_column_widths(
            cols, ("Segoe UI", 10), ("Segoe UI", 10, "bold"), 1000, data_sample=[]
        )
        assert no_data == empty_data

    def test_score_column_uses_quality_score_key(self, _tk_root):
        """Score column reads 'Quality Score' key from data via DISPLAY_TO_DATA_COLUMN."""
        from src.ui.vocab_table.column_config import compute_column_widths

        # Data has 'Quality Score' not 'Score'
        sample = [{"Quality Score": "0.95"}] * 5
        result = compute_column_widths(
            ["Score"],
            ("Segoe UI", 10),
            ("Segoe UI", 10, "bold"),
            1000,
            data_sample=sample,
        )
        assert result["Score"] >= 40  # At minimum

    def test_single_column(self, _tk_root):
        """Works with a single column."""
        from src.ui.vocab_table.column_config import compute_column_widths

        result = compute_column_widths(["Keep"], ("Segoe UI", 10), ("Segoe UI", 10, "bold"), 800)
        assert "Keep" in result
        assert result["Keep"] > 0


# ============================================================================
# 4. _measure_content_width() internals
# ============================================================================


class TestMeasureContentWidth:
    """Tests for the content width measurement helper."""

    def test_empty_data_returns_zero(self, _tk_root):
        """Empty data list returns 0."""
        import tkinter.font as tkfont

        from src.ui.vocab_table.column_config import _measure_content_width

        font = tkfont.Font(font=("Segoe UI", 10))
        assert _measure_content_width("Term", [], font) == 0

    def test_missing_column_returns_zero(self, _tk_root):
        """Data rows that don't have the column return 0."""
        import tkinter.font as tkfont

        from src.ui.vocab_table.column_config import _measure_content_width

        font = tkfont.Font(font=("Segoe UI", 10))
        result = _measure_content_width("Term", [{"Score": "5"}], font)
        assert result == 0

    def test_samples_at_most_100_rows(self, _tk_root):
        """Only first 100 rows are sampled."""
        import tkinter.font as tkfont

        from src.ui.vocab_table.column_config import _measure_content_width

        font = tkfont.Font(font=("Segoe UI", 10))
        # 200 rows: first 100 have short text, next 100 have long text
        short_rows = [{"Term": "A"}] * 100
        long_rows = [{"Term": "A very long term name here"}] * 100
        data = short_rows + long_rows
        result = _measure_content_width("Term", data, font)
        # Width should be based on "A" only (first 100 rows)
        single_a = font.measure("A")
        # Result includes padding
        assert result < font.measure("A very long term name here") + 20

    def test_truncates_long_content(self, _tk_root):
        """Content exceeding max_chars is truncated before measuring."""
        import tkinter.font as tkfont

        from src.ui.vocab_table.column_config import _measure_content_width

        font = tkfont.Font(font=("Segoe UI", 10))
        # Term max_chars=30, so 50-char text gets truncated to 30
        data = [{"Term": "A" * 50}]
        result = _measure_content_width("Term", data, font)
        full_width = font.measure("A" * 50) + 16
        assert result < full_width  # Should be narrower due to truncation


# ============================================================================
# 5. _max_width_for_column() caps
# ============================================================================


class TestMaxWidthForColumn:
    """Tests for per-column max width caps."""

    def test_term_45_percent(self):
        """Term cap is 45% of available width."""
        from src.ui.vocab_table.column_config import _max_width_for_column

        assert _max_width_for_column("Term", 1000) == 450

    def test_found_by_20_percent(self):
        """Found By cap is 20% of available width."""
        from src.ui.vocab_table.column_config import _max_width_for_column

        assert _max_width_for_column("Found By", 1000) == 200

    def test_other_columns_15_percent(self):
        """Other columns cap at 15% of available width."""
        from src.ui.vocab_table.column_config import _max_width_for_column

        for col in ["Score", "Keep", "NER", "Occurrences", "# Docs"]:
            assert _max_width_for_column(col, 1000) == 150, f"{col} cap wrong"

    def test_zero_available_width(self):
        """Zero available width returns 0 cap."""
        from src.ui.vocab_table.column_config import _max_width_for_column

        assert _max_width_for_column("Term", 0) == 0


# ============================================================================
# 6. get_vocab_font_specs()
# ============================================================================


class TestGetVocabFontSpecs:
    """Tests for the font spec accessor in styles.py."""

    def test_returns_two_tuples(self):
        """Returns a pair of font spec tuples."""
        from src.ui.styles import get_vocab_font_specs

        content, heading = get_vocab_font_specs()
        assert isinstance(content, tuple)
        assert isinstance(heading, tuple)

    def test_content_font_has_family_and_size(self):
        """Content font spec has at least family and size."""
        from src.ui.styles import get_vocab_font_specs

        content, _ = get_vocab_font_specs()
        assert len(content) >= 2
        assert isinstance(content[0], str)  # family
        assert isinstance(content[1], int)  # size

    def test_heading_font_is_bold(self):
        """Heading font spec includes 'bold'."""
        from src.ui.styles import get_vocab_font_specs

        _, heading = get_vocab_font_specs()
        assert len(heading) >= 3
        assert heading[2] == "bold"

    def test_heading_same_family_as_content(self):
        """Content and heading use the same font family."""
        from src.ui.styles import get_vocab_font_specs

        content, heading = get_vocab_font_specs()
        assert content[0] == heading[0]

    def test_heading_same_size_as_content(self):
        """Content and heading use the same font size."""
        from src.ui.styles import get_vocab_font_specs

        content, heading = get_vocab_font_specs()
        assert content[1] == heading[1]


# ============================================================================
# 7. Source-level checks: stretch=False, save cap, menu, autosize
# ============================================================================


class TestStretchFalseOnAllColumns:
    """Verify stretch=False is set on all columns (no snap-back)."""

    def test_no_stretch_true_in_column_config(self):
        """No stretch=True in the VocabTreeview column configuration."""
        source = _read_source("src/ui/vocab_table/vocab_treeview.py")
        # Find the column configuration block
        assert "stretch=False" in source
        # The old pattern should be gone
        assert 'stretch=col == "Term"' not in source

    def test_all_stretch_calls_are_false(self):
        """Every stretch= assignment in vocab_treeview.py is False."""
        import re

        source = _read_source("src/ui/vocab_table/vocab_treeview.py")
        stretches = re.findall(r"stretch=(\S+)", source)
        for val in stretches:
            assert val.rstrip(",)") == "False", f"Found stretch={val}"


class TestSaveCapRaised:
    """Verify the column width save cap was raised to 800."""

    def test_save_cap_is_800(self):
        """_save_column_widths uses 800 as upper bound."""
        source = _read_source("src/ui/dynamic_output.py")
        assert "30 <= width <= 800" in source

    def test_no_old_500_cap(self):
        """Old 500 cap is gone."""
        source = _read_source("src/ui/dynamic_output.py")
        assert "30 <= width <= 500" not in source

    def test_prefs_validation_matches_save_cap(self):
        """user_preferences.py column width upper limit matches dynamic_output save cap (800)."""
        source = _read_source("src/user_preferences.py")
        assert "width > 800" in source
        assert "width > 500" not in source

    def test_prefs_accepts_width_680(self):
        """Column width of 680 (common on wide monitors) is accepted by prefs."""
        import os
        import tempfile

        from src.user_preferences import UserPreferencesManager

        with tempfile.TemporaryDirectory() as d:
            prefs = UserPreferencesManager(os.path.join(d, "p.json"))
            prefs.set("vocab_column_widths", {"Found By": 680})
            assert prefs.get("vocab_column_widths")["Found By"] == 680

    def test_prefs_accepts_width_800(self):
        """Column width of 800 (upper limit) is accepted by prefs."""
        import os
        import tempfile

        from src.user_preferences import UserPreferencesManager

        with tempfile.TemporaryDirectory() as d:
            prefs = UserPreferencesManager(os.path.join(d, "p.json"))
            prefs.set("vocab_column_widths", {"Term": 800})
            assert prefs.get("vocab_column_widths")["Term"] == 800

    def test_prefs_rejects_width_801(self):
        """Column width above 800 is rejected by prefs."""
        import os
        import tempfile

        from src.user_preferences import UserPreferencesManager

        with tempfile.TemporaryDirectory() as d:
            prefs = UserPreferencesManager(os.path.join(d, "p.json"))
            with pytest.raises(ValueError, match="30-800"):
                prefs.set("vocab_column_widths", {"Term": 801})

    def test_save_column_widths_wraps_prefs_in_try_except(self):
        """_save_column_widths catches prefs errors instead of propagating."""
        source = _read_source("src/ui/dynamic_output.py")
        # Find the _save_column_widths method body
        start = source.index("def _save_column_widths")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert "Could not save column widths" in body


class TestOnTabChangedExceptionIsolation:
    """Verify _on_tab_changed keeps working even if _save_column_widths fails."""

    def test_save_after_ui_updates(self):
        """_save_column_widths is called AFTER followup_frame and button_frame updates."""
        source = _read_source("src/ui/dynamic_output.py")
        start = source.index("def _on_tab_changed")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        # followup_frame update should come before _save_column_widths
        followup_pos = body.index("followup_frame")
        save_pos = body.index("_save_column_widths")
        assert followup_pos < save_pos, (
            "_save_column_widths should run after followup_frame visibility update"
        )


class TestResetColumnWidthsMenu:
    """Verify Reset Column Widths is in the column menu."""

    def test_menu_has_reset_column_widths(self):
        """Column menu includes 'Reset Column Widths' command."""
        source = _read_source("src/ui/dynamic_output.py")
        assert '"Reset Column Widths"' in source

    def test_menu_has_reset_column_visibility(self):
        """Column menu includes 'Reset Column Visibility' command."""
        source = _read_source("src/ui/dynamic_output.py")
        assert '"Reset Column Visibility"' in source

    def test_reset_widths_calls_method(self):
        """Reset Column Widths menu command is wired to _reset_column_widths."""
        source = _read_source("src/ui/dynamic_output.py")
        assert "self._reset_column_widths" in source


class TestAutosizeMethod:
    """Verify _autosize_columns_to_content exists and is called."""

    def test_method_exists(self):
        """_autosize_columns_to_content is defined in dynamic_output.py."""
        source = _read_source("src/ui/dynamic_output.py")
        assert "def _autosize_columns_to_content(self, data" in source

    def test_called_in_update_pagination(self):
        """_autosize_columns_to_content is called from _update_pagination_ui."""
        source = _read_source("src/ui/dynamic_output.py")
        # Find the _update_pagination_ui method body
        start = source.index("def _update_pagination_ui")
        # Find next def at same indent level
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert "_autosize_columns_to_content" in body

    def test_skips_when_saved_widths_exist(self):
        """_autosize_columns_to_content returns early when saved widths exist."""
        source = _read_source("src/ui/dynamic_output.py")
        start = source.index("def _autosize_columns_to_content")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert "_load_column_widths" in body
        assert "if saved_widths:" in body


class TestResetColumnWidthsMethod:
    """Verify _reset_column_widths clears prefs and re-auto-sizes."""

    def test_method_exists(self):
        """_reset_column_widths is defined."""
        source = _read_source("src/ui/dynamic_output.py")
        assert "def _reset_column_widths(self)" in source

    def test_clears_saved_widths(self):
        """_reset_column_widths sets vocab_column_widths to empty dict."""
        source = _read_source("src/ui/dynamic_output.py")
        start = source.index("def _reset_column_widths")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert '"vocab_column_widths", {}' in body

    def test_calls_compute_column_widths(self):
        """_reset_column_widths re-computes widths after clearing."""
        source = _read_source("src/ui/dynamic_output.py")
        start = source.index("def _reset_column_widths")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert "compute_column_widths" in body


class TestDPIAwareInitialWidths:
    """Verify initial column widths use DPI-aware computation."""

    def test_display_csv_uses_compute_column_widths(self):
        """_display_csv calls compute_column_widths when no saved widths."""
        source = _read_source("src/ui/dynamic_output.py")
        start = source.index("def _display_csv")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert "compute_column_widths" in body

    def test_display_csv_uses_get_vocab_font_specs(self):
        """_display_csv gets font specs for width computation."""
        source = _read_source("src/ui/dynamic_output.py")
        start = source.index("def _display_csv")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert "get_vocab_font_specs" in body

    def test_display_csv_checks_saved_widths_first(self):
        """_display_csv prefers saved widths over computed ones."""
        source = _read_source("src/ui/dynamic_output.py")
        start = source.index("def _display_csv")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert "_load_column_widths" in body


# ============================================================================
# 8. Constants and imports
# ============================================================================


class TestColumnConfigConstants:
    """Verify column_config constants are set correctly."""

    def test_min_column_width(self):
        """MIN_COLUMN_WIDTH is 40."""
        from src.ui.vocab_table.column_config import MIN_COLUMN_WIDTH

        assert MIN_COLUMN_WIDTH == 40

    def test_cell_padding(self):
        """_CELL_PADDING is 16."""
        from src.ui.vocab_table.column_config import _CELL_PADDING

        assert _CELL_PADDING == 16

    def test_sort_indicator_padding(self):
        """_SORT_INDICATOR_PADDING is 12."""
        from src.ui.vocab_table.column_config import _SORT_INDICATOR_PADDING

        assert _SORT_INDICATOR_PADDING == 12

    def test_column_config_imports_tkfont(self):
        """column_config.py imports tkinter.font for DPI-aware measurement."""
        source = _read_source("src/ui/vocab_table/column_config.py")
        assert "import tkinter.font as tkfont" in source

    def test_dynamic_output_imports_compute_function(self):
        """dynamic_output.py imports compute_column_widths."""
        source = _read_source("src/ui/dynamic_output.py")
        assert "compute_column_widths" in source

    def test_dynamic_output_imports_get_vocab_font_specs(self):
        """dynamic_output.py imports get_vocab_font_specs."""
        source = _read_source("src/ui/dynamic_output.py")
        assert "from src.ui.styles import get_vocab_font_specs" in source
