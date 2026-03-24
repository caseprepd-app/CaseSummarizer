"""
Tests for v1.0.19 tooltip refactor.

Validates that all tooltip implementations follow best practices:
- <ButtonPress> binding hides tooltip on click
- <Destroy> binding cleans up on widget destruction
- Timer IDs stored and cancelled on every event handler entry
- TooltipIcon no longer uses straddling/delayed-hide logic
- VocabTreeview integrates with tooltip_manager
- All bindings use add="+" to preserve existing handlers
"""

import inspect
from unittest.mock import MagicMock, patch

# ===========================================================================
# tooltip_helper.py: create_tooltip event bindings
# ===========================================================================


class TestCreateTooltipBindings:
    """create_tooltip should bind Enter, Leave, ButtonPress, and Destroy."""

    def test_binds_button_press(self):
        """<ButtonPress> binding exists so clicks dismiss the tooltip."""
        source = inspect.getsource(
            __import__("src.ui.tooltip_helper", fromlist=["create_tooltip"]).create_tooltip
        )
        assert "<ButtonPress>" in source

    def test_binds_destroy(self):
        """<Destroy> binding exists so widget destruction cleans up tooltip."""
        source = inspect.getsource(
            __import__("src.ui.tooltip_helper", fromlist=["create_tooltip"]).create_tooltip
        )
        assert "<Destroy>" in source

    def test_binds_enter(self):
        """<Enter> binding exists for showing tooltip."""
        source = inspect.getsource(
            __import__("src.ui.tooltip_helper", fromlist=["create_tooltip"]).create_tooltip
        )
        assert "<Enter>" in source

    def test_binds_leave(self):
        """<Leave> binding exists for hiding tooltip."""
        source = inspect.getsource(
            __import__("src.ui.tooltip_helper", fromlist=["create_tooltip"]).create_tooltip
        )
        assert "<Leave>" in source

    def test_all_bindings_use_add_plus(self):
        """All bind calls use add='+' to preserve existing handlers."""
        source = inspect.getsource(
            __import__("src.ui.tooltip_helper", fromlist=["create_tooltip"]).create_tooltip
        )
        # Count bind calls and add="+" usage
        bind_lines = [line for line in source.splitlines() if "widget.bind(" in line]
        add_plus_lines = [line for line in bind_lines if 'add="+"' in line or "add='+'" in line]
        assert len(bind_lines) >= 4, f"Expected >= 4 bind calls, got {len(bind_lines)}"
        assert len(bind_lines) == len(add_plus_lines), (
            f"{len(bind_lines)} bind calls but only {len(add_plus_lines)} use add='+'"
        )

    def test_cancel_show_in_on_enter(self):
        """on_enter cancels pending timers before scheduling new one."""
        source = inspect.getsource(
            __import__("src.ui.tooltip_helper", fromlist=["create_tooltip"]).create_tooltip
        )
        # The on_enter function should call cancel_show()
        assert "cancel_show()" in source

    def test_cancel_show_in_hide_tooltip(self):
        """hide_tooltip cancels pending timers."""
        source = inspect.getsource(
            __import__("src.ui.tooltip_helper", fromlist=["create_tooltip"]).create_tooltip
        )
        # hide_tooltip should call cancel_show()
        hide_fn_start = source.index("def hide_tooltip")
        hide_fn_body = source[hide_fn_start : hide_fn_start + 200]
        assert "cancel_show()" in hide_fn_body

    def test_checks_winfo_exists_before_showing(self):
        """show_tooltip_at_cursor verifies widget still exists."""
        source = inspect.getsource(
            __import__("src.ui.tooltip_helper", fromlist=["create_tooltip"]).create_tooltip
        )
        assert "winfo_exists" in source


# ===========================================================================
# TooltipIcon: no straddling, proper timer management
# ===========================================================================


class TestTooltipIconRefactor:
    """TooltipIcon should use delay + immediate hide (no straddling)."""

    def test_no_check_hide_tooltip(self):
        """Straddling method _check_hide_tooltip should be removed."""
        from src.ui.settings.settings_widgets import TooltipIcon

        assert not hasattr(TooltipIcon, "_check_hide_tooltip")

    def test_no_delayed_hide_check(self):
        """Delayed hide check method should be removed."""
        from src.ui.settings.settings_widgets import TooltipIcon

        assert not hasattr(TooltipIcon, "_delayed_hide_check")

    def test_has_show_delay(self):
        """TooltipIcon should have a show delay constant."""
        from src.ui.settings.settings_widgets import TooltipIcon

        assert hasattr(TooltipIcon, "_SHOW_DELAY_MS")
        assert TooltipIcon._SHOW_DELAY_MS > 0

    def test_show_delay_reasonable(self):
        """Show delay should be 200-600ms (industry best practice)."""
        from src.ui.settings.settings_widgets import TooltipIcon

        assert 200 <= TooltipIcon._SHOW_DELAY_MS <= 600

    def test_has_cancel_show(self):
        """TooltipIcon should have a _cancel_show method for timer cleanup."""
        from src.ui.settings.settings_widgets import TooltipIcon

        assert hasattr(TooltipIcon, "_cancel_show")

    def test_has_on_destroy(self):
        """TooltipIcon should have a _on_destroy cleanup handler."""
        from src.ui.settings.settings_widgets import TooltipIcon

        assert hasattr(TooltipIcon, "_on_destroy")

    def test_binds_button_press(self):
        """Source should bind <ButtonPress> to hide tooltip on click."""
        from src.ui.settings.settings_widgets import TooltipIcon

        source = inspect.getsource(TooltipIcon.__init__)
        assert "<ButtonPress>" in source

    def test_binds_destroy(self):
        """Source should bind <Destroy> to clean up tooltip."""
        from src.ui.settings.settings_widgets import TooltipIcon

        source = inspect.getsource(TooltipIcon.__init__)
        assert "<Destroy>" in source

    def test_all_bindings_use_add_plus(self):
        """All bind calls in __init__ should use add='+'."""
        from src.ui.settings.settings_widgets import TooltipIcon

        source = inspect.getsource(TooltipIcon.__init__)
        bind_lines = [
            line for line in source.splitlines() if '.bind("' in line or ".bind('<" in line
        ]
        for line in bind_lines:
            assert 'add="+"' in line or "add='+'" in line, f"Missing add='+' in: {line.strip()}"

    def test_on_leave_calls_cancel_show(self):
        """_on_leave should cancel pending show timer."""
        from src.ui.settings.settings_widgets import TooltipIcon

        source = inspect.getsource(TooltipIcon._on_leave)
        assert "_cancel_show" in source

    def test_on_leave_calls_force_hide(self):
        """_on_leave should immediately hide tooltip (no delay)."""
        from src.ui.settings.settings_widgets import TooltipIcon

        source = inspect.getsource(TooltipIcon._on_leave)
        assert "_force_hide_tooltip" in source
        # Should NOT contain after() delay
        assert "after(" not in source

    def test_show_tooltip_checks_winfo_exists(self):
        """_show_tooltip should verify widget exists before creating tooltip."""
        from src.ui.settings.settings_widgets import TooltipIcon

        source = inspect.getsource(TooltipIcon._show_tooltip)
        assert "winfo_exists" in source

    def test_show_tooltip_sets_toolwindow(self):
        """_show_tooltip should set -toolwindow attribute (Windows)."""
        from src.ui.settings.settings_widgets import TooltipIcon

        source = inspect.getsource(TooltipIcon._show_tooltip)
        assert "-toolwindow" in source

    def test_force_hide_unregisters_from_manager(self):
        """_force_hide_tooltip should unregister from tooltip_manager."""
        from src.ui.settings.settings_widgets import TooltipIcon

        source = inspect.getsource(TooltipIcon._force_hide_tooltip)
        assert "unregister" in source


# ===========================================================================
# FileReviewTable: <ButtonPress> binding
# ===========================================================================


class TestFileReviewTableTooltip:
    """FileReviewTable should hide tooltip on click."""

    def test_binds_button_press(self):
        """Source should bind <ButtonPress> to hide tooltip."""
        from src.ui.widgets import FileReviewTable

        # Binding is in _create_treeview (called from __init__)
        source = inspect.getsource(FileReviewTable._create_treeview)
        assert "<ButtonPress>" in source

    def test_button_press_calls_hide(self):
        """<ButtonPress> binding should call _hide_tooltip."""
        from src.ui.widgets import FileReviewTable

        source = inspect.getsource(FileReviewTable._create_treeview)
        # Filter to actual bind() calls (not comments)
        button_lines = [l for l in source.splitlines() if "<ButtonPress>" in l and ".bind(" in l]
        assert len(button_lines) >= 1
        assert "_hide_tooltip" in button_lines[0]


# ===========================================================================
# VocabTreeview: tooltip_manager integration + <ButtonPress>
# ===========================================================================


class TestVocabTreeviewTooltip:
    """VocabTreeview should integrate with tooltip_manager."""

    def test_binds_button_press(self):
        """Source should bind <ButtonPress> to hide tooltip."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        source = inspect.getsource(VocabTreeview.__init__)
        assert "<ButtonPress>" in source

    def test_show_tooltip_calls_manager_close(self):
        """_show_tooltip should close existing tooltips via manager."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        source = inspect.getsource(VocabTreeview._show_tooltip)
        assert "tooltip_manager" in source
        assert "close_active" in source

    def test_show_tooltip_registers_with_manager(self):
        """_show_tooltip should register the new tooltip with manager."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        source = inspect.getsource(VocabTreeview._show_tooltip)
        assert "register" in source

    def test_hide_tooltip_unregisters_from_manager(self):
        """_hide_tooltip should unregister from tooltip_manager."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        source = inspect.getsource(VocabTreeview._hide_tooltip)
        assert "unregister" in source

    def test_show_tooltip_sets_toolwindow(self):
        """_show_tooltip should set -toolwindow for Windows."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        source = inspect.getsource(VocabTreeview._show_tooltip)
        assert "-toolwindow" in source

    def test_hide_tooltip_exception_safe(self):
        """_hide_tooltip should handle exceptions from destroy()."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        source = inspect.getsource(VocabTreeview._hide_tooltip)
        assert "except" in source


# ===========================================================================
# TooltipManager: singleton behavior (existing, verify still works)
# ===========================================================================


class TestTooltipManagerSingleton:
    """TooltipManager singleton should coordinate all tooltip sources."""

    def test_singleton_instance(self):
        """Two imports should yield the same instance."""
        from src.ui.tooltip_manager import tooltip_manager as tm1
        from src.ui.tooltip_manager import tooltip_manager as tm2

        assert tm1 is tm2

    def test_close_active_safe_when_empty(self):
        """close_active should not raise when no tooltip is active."""
        from src.ui.tooltip_manager import tooltip_manager

        tooltip_manager._active_tooltip = None
        tooltip_manager.close_active()  # Should not raise

    def test_unregister_clears_matching(self):
        """unregister should clear reference when tooltip matches."""
        from src.ui.tooltip_manager import tooltip_manager

        sentinel = MagicMock()
        tooltip_manager._active_tooltip = sentinel
        tooltip_manager._active_owner = "test"
        tooltip_manager.unregister(sentinel)
        assert tooltip_manager._active_tooltip is None

    def test_unregister_ignores_non_matching(self):
        """unregister should not clear if tooltip doesn't match."""
        from src.ui.tooltip_manager import tooltip_manager

        sentinel = MagicMock()
        other = MagicMock()
        tooltip_manager._active_tooltip = sentinel
        tooltip_manager.unregister(other)
        assert tooltip_manager._active_tooltip is sentinel
        # Cleanup
        tooltip_manager._active_tooltip = None


# ===========================================================================
# Cross-cutting: all tooltip sources use tooltip_manager
# ===========================================================================


class TestAllTooltipSourcesUseManager:
    """Every tooltip implementation should integrate with TooltipManager."""

    def test_tooltip_helper_uses_manager(self):
        """create_tooltip should use tooltip_manager."""
        from src.ui.tooltip_helper import create_tooltip

        source = inspect.getsource(create_tooltip)
        assert "tooltip_manager" in source

    def test_tooltip_icon_uses_manager(self):
        """TooltipIcon should use tooltip_manager."""
        from src.ui.settings.settings_widgets import TooltipIcon

        show_src = inspect.getsource(TooltipIcon._show_tooltip)
        hide_src = inspect.getsource(TooltipIcon._force_hide_tooltip)
        assert "tooltip_manager" in show_src
        assert "tooltip_manager" in hide_src

    def test_file_review_table_uses_manager(self):
        """FileReviewTable tooltips should use tooltip_manager."""
        from src.ui.widgets import FileReviewTable

        show_src = inspect.getsource(FileReviewTable._show_tooltip)
        hide_src = inspect.getsource(FileReviewTable._hide_tooltip)
        assert "tooltip_manager" in show_src
        assert "tooltip_manager" in hide_src

    def test_vocab_treeview_uses_manager(self):
        """VocabTreeview tooltips should use tooltip_manager."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        show_src = inspect.getsource(VocabTreeview._show_tooltip)
        hide_src = inspect.getsource(VocabTreeview._hide_tooltip)
        assert "tooltip_manager" in show_src
        assert "tooltip_manager" in hide_src


# ===========================================================================
# Behavioral: TooltipIcon timer cancellation
# ===========================================================================


class TestTooltipIconBehavior:
    """Behavioral tests for TooltipIcon using MagicMock stub."""

    def _make_icon_stub(self):
        """Create a stub with TooltipIcon attributes."""
        from src.ui.settings.settings_widgets import TooltipIcon

        stub = MagicMock()
        stub.tooltip_text = "Test tooltip"
        stub.tooltip_window = None
        stub._show_timer = None
        stub._SHOW_DELAY_MS = TooltipIcon._SHOW_DELAY_MS
        return stub

    def test_cancel_show_with_active_timer(self):
        """_cancel_show should call after_cancel when timer is active."""
        from src.ui.settings.settings_widgets import TooltipIcon

        stub = self._make_icon_stub()
        stub._show_timer = 12345

        TooltipIcon._cancel_show(stub)

        stub.after_cancel.assert_called_once_with(12345)
        assert stub._show_timer is None

    def test_cancel_show_with_no_timer(self):
        """_cancel_show should be safe when no timer is active."""
        from src.ui.settings.settings_widgets import TooltipIcon

        stub = self._make_icon_stub()
        stub._show_timer = None

        TooltipIcon._cancel_show(stub)  # Should not raise

        stub.after_cancel.assert_not_called()

    def test_on_enter_schedules_show(self):
        """_on_enter should schedule _show_tooltip after delay."""
        from src.ui.settings.settings_widgets import TooltipIcon

        stub = self._make_icon_stub()
        stub._show_timer = None
        stub.after.return_value = 99

        TooltipIcon._on_enter(stub)

        # Verify after() was called with the delay and show method
        stub.after.assert_called_once()
        call_args = stub.after.call_args
        assert call_args[0][0] == TooltipIcon._SHOW_DELAY_MS
        assert stub._show_timer == 99

    def test_on_leave_cancels_and_hides(self):
        """_on_leave should cancel timer and force-hide tooltip."""
        from src.ui.settings.settings_widgets import TooltipIcon

        stub = self._make_icon_stub()
        stub._show_timer = 42

        TooltipIcon._on_leave(stub)

        stub._cancel_show.assert_called_once()
        stub._force_hide_tooltip.assert_called_once()

    def test_force_hide_destroys_window(self):
        """_force_hide_tooltip should destroy and unregister tooltip."""
        from src.ui.settings.settings_widgets import TooltipIcon

        stub = self._make_icon_stub()
        mock_window = MagicMock()
        stub.tooltip_window = mock_window

        with patch("src.ui.settings.settings_widgets.tooltip_manager") as mock_mgr:
            TooltipIcon._force_hide_tooltip(stub)

        mock_mgr.unregister.assert_called_once_with(mock_window)
        mock_window.destroy.assert_called_once()
        assert stub.tooltip_window is None

    def test_force_hide_noop_when_no_window(self):
        """_force_hide_tooltip should be safe when no tooltip exists."""
        from src.ui.settings.settings_widgets import TooltipIcon

        stub = self._make_icon_stub()
        stub.tooltip_window = None

        with patch("src.ui.settings.settings_widgets.tooltip_manager") as mock_mgr:
            TooltipIcon._force_hide_tooltip(stub)

        mock_mgr.unregister.assert_not_called()
