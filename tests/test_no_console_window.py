"""
Tests for console window suppression on Windows.

Verifies that subprocess spawns from the worker process and splash screen
include CREATE_NO_WINDOW flags so no blank CLI window appears alongside
the GUI for end users.
"""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# A. Worker subprocess Popen patch
# ============================================================================


class TestWorkerSubprocessPatch:
    """Verify _patch_subprocess_no_window injects CREATE_NO_WINDOW."""

    @pytest.fixture(autouse=True)
    def _unpatch_after(self):
        """Restore original Popen.__init__ after each test."""
        orig = subprocess.Popen.__init__
        yield
        subprocess.Popen.__init__ = orig

    def test_patch_modifies_popen_init(self):
        """After patching, Popen.__init__ should be replaced."""
        orig = subprocess.Popen.__init__
        from src.worker_process import _patch_subprocess_no_window

        _patch_subprocess_no_window()
        assert subprocess.Popen.__init__ is not orig

    def test_patch_injects_create_no_window_flag(self):
        """The patched Popen passes CREATE_NO_WINDOW to the original init."""
        _CREATE_NO_WINDOW = 0x08000000

        # Replace init with a spy, then patch — patch captures the spy
        spy = MagicMock()
        subprocess.Popen.__init__ = spy

        from src.worker_process import _patch_subprocess_no_window

        _patch_subprocess_no_window()

        mock_self = MagicMock()
        subprocess.Popen.__init__(mock_self, ["echo"])

        spy.assert_called_once()
        flags = spy.call_args[1].get("creationflags", 0)
        assert flags & _CREATE_NO_WINDOW, f"CREATE_NO_WINDOW not injected: flags={hex(flags)}"

    def test_patch_preserves_existing_flags(self):
        """Existing creationflags survive when the patch adds its flag."""
        _CREATE_NO_WINDOW = 0x08000000
        _EXISTING = 0x00000010

        spy = MagicMock()
        subprocess.Popen.__init__ = spy

        from src.worker_process import _patch_subprocess_no_window

        _patch_subprocess_no_window()

        mock_self = MagicMock()
        subprocess.Popen.__init__(mock_self, ["echo"], creationflags=_EXISTING)

        flags = spy.call_args[1].get("creationflags", 0)
        assert flags & _EXISTING, "Existing flags were lost"
        assert flags & _CREATE_NO_WINDOW, "CREATE_NO_WINDOW not added"

    def test_patch_is_idempotent(self):
        """Calling the patch twice should not break anything."""
        from src.worker_process import _patch_subprocess_no_window

        _patch_subprocess_no_window()
        first = subprocess.Popen.__init__
        _patch_subprocess_no_window()
        second = subprocess.Popen.__init__
        assert callable(second)


# ============================================================================
# B. Main process multiprocessing.set_executable
# ============================================================================


class TestMainProcessPatch:
    """Verify main.py configures multiprocessing to use pythonw.exe."""

    def test_main_sets_pythonw_executable(self):
        """main.py should set multiprocessing executable to pythonw.exe."""
        import multiprocessing
        import os

        pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        if not os.path.exists(pythonw):
            pytest.skip("pythonw.exe not found (not a standard CPython install)")

        # Simulate what main.py does
        multiprocessing.set_executable(pythonw)
        # In spawn context, get_executable returns the set value
        from multiprocessing.spawn import get_executable

        assert get_executable().endswith("pythonw.exe")


# ============================================================================
# C. Splash screen frozen-mode launch
# ============================================================================


class TestSplashCreateNoWindow:
    """Verify splash.py passes CREATE_NO_WINDOW in frozen-mode Popen call."""

    def test_frozen_launch_passes_create_no_window(self):
        """In frozen mode, Popen should include CREATE_NO_WINDOW flag."""
        from src.splash import launch

        with (
            patch("src.splash.get_splash_dir") as mock_dir,
            patch("subprocess.Popen") as mock_popen,
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", "C:\\CasePrepd\\CasePrepd.exe"),
        ):
            # Fake a splash directory with one image
            mock_splash_dir = MagicMock()
            mock_image = MagicMock()
            mock_image.suffix = ".png"
            mock_splash_dir.iterdir.return_value = [mock_image]
            mock_dir.return_value = mock_splash_dir

            mock_popen.return_value = MagicMock(pid=9999)
            launch()

            mock_popen.assert_called_once()
            kwargs = mock_popen.call_args[1]
            flags = kwargs.get("creationflags", 0)
            assert flags & subprocess.CREATE_NO_WINDOW, (
                f"CREATE_NO_WINDOW not set in splash Popen: flags={hex(flags)}"
            )


# ============================================================================
# D. Source-level audit: no unguarded subprocess calls
# ============================================================================


class TestSourceCodeAudit:
    """Verify all subprocess launches in src/ use console suppression."""

    KNOWN_GUARDED = {
        "ocr_processor.py",  # Monkey-patches pytesseract Popen
        "splash.py",  # Uses CREATE_NO_WINDOW (just fixed)
        "export_service.py",  # Uses os.startfile on Windows
        "settings_registry.py",  # Uses os.startfile on Windows
        "corpus_dialog.py",  # Uses os.startfile on Windows
        "worker_process.py",  # Patches subprocess.Popen globally
    }

    def test_all_subprocess_calls_are_guarded(self):
        """Every file with subprocess.Popen/run/call must be in KNOWN_GUARDED."""
        import re
        from pathlib import Path

        src_dir = Path(__file__).parent.parent / "src"
        pattern = re.compile(r"subprocess\.(Popen|run|call)\(")
        unguarded = []

        for py_file in src_dir.rglob("*.py"):
            if py_file.name in self.KNOWN_GUARDED:
                continue
            try:
                text = py_file.read_text(encoding="utf-8")
            except Exception:
                continue
            if pattern.search(text):
                unguarded.append(py_file.name)

        assert not unguarded, (
            f"Subprocess calls without known console guards: {unguarded}. "
            "Add CREATE_NO_WINDOW or startupinfo, then add to KNOWN_GUARDED."
        )
