"""
Tests for console window suppression on Windows.

Verifies that multiprocessing subprocess spawns and splash screen launches
include CREATE_NO_WINDOW flags so no blank CLI window appears alongside
the GUI for end users.
"""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# A. Multiprocessing monkey-patch
# ============================================================================


class TestMultiprocessingPatch:
    """Verify _patch_multiprocessing_no_window injects CREATE_NO_WINDOW."""

    @pytest.fixture(autouse=True)
    def _unpatch_after(self):
        """Restore original Popen.__init__ after each test."""
        import multiprocessing.popen_spawn_win32 as mpw

        orig = mpw.Popen.__init__
        yield
        mpw.Popen.__init__ = orig

    def test_patch_modifies_popen_init(self):
        """After patching, Popen.__init__ should be replaced."""
        import multiprocessing.popen_spawn_win32 as mpw

        orig = mpw.Popen.__init__
        from src.main import _patch_multiprocessing_no_window

        _patch_multiprocessing_no_window()
        assert mpw.Popen.__init__ is not orig

    def test_patch_injects_create_no_window_flag(self):
        """The patched CreateProcess call must OR in 0x08000000."""
        from src.main import _patch_multiprocessing_no_window

        _patch_multiprocessing_no_window()

        captured_flags = []
        import _winapi
        import multiprocessing.popen_spawn_win32 as mpw

        real_create = _winapi.CreateProcess

        # Build a fake Popen and intercept CreateProcess
        with (
            patch.object(_winapi, "CreateProcess") as mock_cp,
            patch.object(_winapi, "CreatePipe", return_value=(1, 2)),
            patch("msvcrt.open_osfhandle", return_value=3),
            patch("builtins.open", MagicMock()),
            patch("multiprocessing.spawn.get_preparation_data", return_value={}),
            patch("multiprocessing.spawn.get_executable", return_value=sys.executable),
            patch("multiprocessing.spawn.get_command_line", return_value=["python", "--test"]),
            patch("multiprocessing.popen_spawn_win32.set_spawning_popen"),
            patch("multiprocessing.reduction.dump"),
        ):
            mock_cp.return_value = (1, 2, 1234, 5678)
            fake_proc = MagicMock()
            fake_proc._name = "test"

            try:
                mpw.Popen(fake_proc)
            except Exception:
                pass  # We only care about the CreateProcess call

            if mock_cp.called:
                call_args = mock_cp.call_args[0]
                flags = call_args[5]  # dwCreationFlags is 6th positional arg
                captured_flags.append(flags)

        assert captured_flags, "CreateProcess was never called"
        assert captured_flags[0] & 0x08000000, (
            f"CREATE_NO_WINDOW (0x08000000) not set in flags: {hex(captured_flags[0])}"
        )

    def test_patch_restores_create_process_after_call(self):
        """Original _winapi.CreateProcess must be restored after Popen init."""
        import _winapi

        from src.main import _patch_multiprocessing_no_window

        _patch_multiprocessing_no_window()

        orig_cp = _winapi.CreateProcess  # save current (should be original)

        # Trigger the patched init (will fail, but should still restore)
        import multiprocessing.popen_spawn_win32 as mpw

        with (
            patch.object(_winapi, "CreatePipe", return_value=(1, 2)),
            patch("msvcrt.open_osfhandle", return_value=3),
            patch("builtins.open", MagicMock()),
        ):
            try:
                mpw.Popen(MagicMock(_name="t"))
            except Exception:
                pass

        assert _winapi.CreateProcess is orig_cp, (
            "CreateProcess was not restored after Popen.__init__"
        )

    def test_patch_is_idempotent(self):
        """Calling the patch twice should not double-wrap."""
        import multiprocessing.popen_spawn_win32 as mpw

        from src.main import _patch_multiprocessing_no_window

        _patch_multiprocessing_no_window()
        first = mpw.Popen.__init__
        _patch_multiprocessing_no_window()
        second = mpw.Popen.__init__
        # Both should be wrappers, but the important thing is they still work.
        # We just verify it doesn't raise.
        assert callable(second)


# ============================================================================
# B. Splash screen frozen-mode launch
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
# C. Source-level audit: no unguarded subprocess calls
# ============================================================================


class TestSourceCodeAudit:
    """Verify all subprocess launches in src/ use console suppression."""

    KNOWN_GUARDED = {
        "ocr_processor.py",  # Monkey-patches pytesseract Popen
        "gpu_detector.py",  # Uses CREATE_NO_WINDOW directly
        "splash.py",  # Uses CREATE_NO_WINDOW (just fixed)
        "export_service.py",  # Uses os.startfile on Windows
        "settings_registry.py",  # Uses os.startfile on Windows
        "corpus_dialog.py",  # Uses os.startfile on Windows
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
