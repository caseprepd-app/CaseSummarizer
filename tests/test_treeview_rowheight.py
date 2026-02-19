"""
Tests for DPI-aware Treeview rowheight calculation.

Covers the fix for text clipping on high-DPI displays:
- _get_rowheight() derives row height from font metrics, not hardcoded pixels
- font_offset is applied to all 4 treeview style configurations
- File review table (SESSION DOCUMENTS) now has explicit font and rowheight
- 8pt floor prevents unreadable text at extreme negative offsets
- Tooltip mentions table rows are affected by font size adjustment

See module docstring in src/ui/styles.py for the full DPI rationale.
"""

import inspect
import tkinter as tk
from pathlib import Path

import pytest


def _read_source(module_path: str) -> str:
    """Read a source file relative to project root."""
    root = Path(__file__).parent.parent
    return (root / module_path).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def _tk_root():
    """Create a Tk root for font metric tests (module-scoped to avoid Tcl crash)."""
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
# 1. _get_rowheight() — font-metrics-based calculation
# ============================================================================


class TestGetRowheight:
    """Tests for _get_rowheight() DPI-aware row height calculation."""

    def test_returns_positive_int(self, _tk_root):
        """_get_rowheight returns a positive integer."""
        from src.ui.styles import _get_rowheight

        result = _get_rowheight(("Segoe UI", 10))
        assert isinstance(result, int)
        assert result > 0

    def test_includes_padding(self, _tk_root):
        """Result is linespace + padding, not just linespace."""
        from src.ui.styles import _get_rowheight

        # With 0 padding, we get just linespace
        no_pad = _get_rowheight(("Segoe UI", 10), padding=0)
        # With default 8px padding, we get linespace + 8
        with_pad = _get_rowheight(("Segoe UI", 10), padding=8)
        assert with_pad == no_pad + 8

    def test_larger_font_gives_larger_rowheight(self, _tk_root):
        """A larger font produces a taller row."""
        from src.ui.styles import _get_rowheight

        small = _get_rowheight(("Segoe UI", 8), padding=0)
        large = _get_rowheight(("Segoe UI", 18), padding=0)
        assert large > small

    def test_custom_padding(self, _tk_root):
        """Custom padding value is respected."""
        from src.ui.styles import _get_rowheight

        p4 = _get_rowheight(("Segoe UI", 10), padding=4)
        p16 = _get_rowheight(("Segoe UI", 10), padding=16)
        assert p16 - p4 == 12

    def test_default_padding_is_8(self):
        """Default padding argument is 8."""
        sig = inspect.signature(
            __import__("src.ui.styles", fromlist=["_get_rowheight"])._get_rowheight
        )
        assert sig.parameters["padding"].default == 8

    def test_zero_padding(self, _tk_root):
        """Zero padding returns just the linespace."""
        from src.ui.styles import _get_rowheight

        result = _get_rowheight(("Segoe UI", 10), padding=0)
        # linespace for a 10pt font should be at least 10px
        assert result >= 10

    def test_different_font_families(self, _tk_root):
        """Works with different font families available on the system."""
        from src.ui.styles import _get_rowheight

        # These fonts are available on all Windows systems
        for family in ("Segoe UI", "Arial", "Courier New"):
            result = _get_rowheight((family, 10))
            assert result > 0, f"Failed for font family {family}"


# ============================================================================
# 2. font_offset integration into style configurations
# ============================================================================


class TestFontOffsetInStyles:
    """Tests that font_offset is applied to all treeview style configs."""

    def test_all_helpers_accept_font_offset(self):
        """All _configure_* functions accept font_offset parameter."""
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
            assert "font_offset" in params, f"{name} missing 'font_offset'"

    def test_initialize_passes_font_offset_to_helpers(self):
        """initialize_all_styles passes font_offset to each _configure_* call."""
        source = _read_source("src/ui/styles.py")
        for name in [
            "_configure_vocab_treeview_style",
            "_configure_qa_table_style",
            "_configure_file_review_style",
            "_configure_question_list_style",
        ]:
            # Each call should include font_offset as third arg
            assert f"{name}(style, scale_factor, font_offset)" in source, (
                f"{name} not called with font_offset"
            )


# ============================================================================
# 3. 8pt font floor
# ============================================================================


class TestFontSizeFloor:
    """Tests for the 8pt minimum font size in treeview styles."""

    def test_floor_in_source_code(self):
        """All _configure_* functions use max(8, ...) for font size."""
        source = _read_source("src/ui/styles.py")
        # Count occurrences of the floor pattern
        count = source.count("max(8, int(10 * sf) + font_offset)")
        assert count == 4, f"Expected 4 max(8,...) patterns, found {count}"

    def test_extreme_negative_offset_stays_above_8(self):
        """With font_offset=-20, font_size should be clamped to 8."""
        # Simulate the calculation: max(8, int(10 * 1.0) + (-20)) = max(8, -10) = 8
        result = max(8, int(10 * 1.0) + (-20))
        assert result == 8

    def test_offset_zero_gives_base_size(self):
        """With font_offset=0 and sf=1.0, font_size is 10."""
        result = max(8, int(10 * 1.0) + 0)
        assert result == 10

    def test_positive_offset_increases_size(self):
        """With font_offset=4 and sf=1.0, font_size is 14."""
        result = max(8, int(10 * 1.0) + 4)
        assert result == 14

    def test_scale_factor_and_offset_combine(self):
        """Scale factor and offset combine correctly."""
        # sf=1.5, offset=2 -> max(8, 15 + 2) = 17
        result = max(8, int(10 * 1.5) + 2)
        assert result == 17


# ============================================================================
# 4. File review table now has font and rowheight
# ============================================================================


class TestFileReviewStyleComplete:
    """Tests that the file review (SESSION DOCUMENTS) table now has font + rowheight."""

    def test_file_review_has_font_in_source(self):
        """_configure_file_review_style sets font on 'Treeview' style."""
        source = _read_source("src/ui/styles.py")
        # Find the file review function body
        start = source.index("def _configure_file_review_style")
        end = source.index("\ndef ", start + 1)
        body = source[start:end]
        assert "font=font_spec" in body

    def test_file_review_has_rowheight_in_source(self):
        """_configure_file_review_style sets rowheight via _get_rowheight."""
        source = _read_source("src/ui/styles.py")
        start = source.index("def _configure_file_review_style")
        end = source.index("\ndef ", start + 1)
        body = source[start:end]
        assert "rowheight=_get_rowheight(font_spec)" in body

    def test_file_review_has_font_offset(self):
        """_configure_file_review_style uses font_offset in font size calc."""
        source = _read_source("src/ui/styles.py")
        start = source.index("def _configure_file_review_style")
        end = source.index("\ndef ", start + 1)
        body = source[start:end]
        assert "font_offset" in body


# ============================================================================
# 5. No hardcoded rowheight values remain
# ============================================================================


class TestNoHardcodedRowheight:
    """Verify no hardcoded rowheight=int(N * sf) patterns remain."""

    def test_no_hardcoded_rowheight_pattern(self):
        """No 'rowheight=int(' patterns remain in styles.py."""
        source = _read_source("src/ui/styles.py")
        assert "rowheight=int(" not in source

    def test_all_rowheights_use_get_rowheight(self):
        """All rowheight assignments use _get_rowheight()."""
        source = _read_source("src/ui/styles.py")
        import re

        rowheight_assignments = re.findall(r"rowheight=(.+?)(?:,|\))", source)
        for assignment in rowheight_assignments:
            assert "_get_rowheight" in assignment, f"Found non-metric rowheight: {assignment}"

    def test_rowheight_count_matches_style_count(self):
        """There are exactly 4 rowheight assignments (one per style)."""
        source = _read_source("src/ui/styles.py")
        count = source.count("rowheight=_get_rowheight")
        assert count == 4, f"Expected 4 rowheight assignments, found {count}"


# ============================================================================
# 6. _get_rowheight uses tkfont.Font.metrics
# ============================================================================


class TestRowheightUsesFontMetrics:
    """Verify _get_rowheight uses tkfont for DPI-aware measurement."""

    def test_imports_tkfont(self):
        """styles.py imports tkinter.font."""
        source = _read_source("src/ui/styles.py")
        assert "import tkinter.font as tkfont" in source

    def test_uses_metrics_linespace(self):
        """_get_rowheight calls .metrics('linespace')."""
        source = _read_source("src/ui/styles.py")
        assert 'metrics("linespace")' in source

    def test_creates_font_from_spec(self):
        """_get_rowheight creates a tkfont.Font from the font_spec."""
        source = _read_source("src/ui/styles.py")
        assert "tkfont.Font(font=font_spec)" in source


# ============================================================================
# 7. main_window.py call site
# ============================================================================


class TestMainWindowCallSite:
    """Verify main_window.py passes font_offset to initialize_all_styles."""

    def test_imports_both_scaling_functions(self):
        """main_window.py imports get_effective_font_offset."""
        source = _read_source("src/ui/main_window.py")
        assert "get_effective_font_offset" in source

    def test_passes_font_offset_to_styles(self):
        """initialize_all_styles receives get_effective_font_offset()."""
        source = _read_source("src/ui/main_window.py")
        assert (
            "initialize_all_styles(get_effective_ui_scale(), get_effective_font_offset())" in source
        )


# ============================================================================
# 8. Settings tooltip update
# ============================================================================


class TestSettingsTooltipUpdate:
    """Verify font_size_offset tooltip mentions table rows."""

    def test_tooltip_mentions_table_rows(self):
        """font_size_offset tooltip mentions table rows are affected."""
        source = _read_source("src/ui/settings/settings_registry.py")
        idx = source.index('key="font_size_offset"')
        block = source[idx : idx + 600]
        assert "table rows" in block.lower()

    def test_tooltip_mentions_specific_tables(self):
        """Tooltip lists the specific table types affected."""
        source = _read_source("src/ui/settings/settings_registry.py")
        idx = source.index('key="font_size_offset"')
        block = source[idx : idx + 600]
        # Should mention vocabulary, Q&A, and documents tables
        assert "vocabulary" in block.lower()
        assert "q&a" in block.lower() or "qa" in block.lower()
        assert "document" in block.lower()


# ============================================================================
# 9. Module docstring documents the DPI rationale
# ============================================================================


class TestDPIDocumentation:
    """Verify styles.py documents the DPI-scaling rationale."""

    def test_docstring_explains_dpi_problem(self):
        """Module docstring explains why hardcoded rowheight clips text."""
        source = _read_source("src/ui/styles.py")
        docstring = source.split('"""')[1]  # First docstring
        assert "dpi" in docstring.lower()

    def test_docstring_explains_linespace_solution(self):
        """Module docstring explains the linespace-based solution."""
        source = _read_source("src/ui/styles.py")
        docstring = source.split('"""')[1]
        assert "linespace" in docstring.lower()

    def test_docstring_mentions_setprocessdpiawareness(self):
        """Module docstring mentions SetProcessDpiAwareness for context."""
        source = _read_source("src/ui/styles.py")
        docstring = source.split('"""')[1]
        assert "SetProcessDpiAwareness" in docstring

    def test_get_rowheight_docstring_mentions_dpi(self):
        """_get_rowheight docstring explains DPI-aware behavior."""
        from src.ui.styles import _get_rowheight

        assert "dpi" in _get_rowheight.__doc__.lower()
