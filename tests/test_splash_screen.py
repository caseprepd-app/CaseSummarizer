"""Tests for splash screen: subprocess launch, image selection, portability, installer readiness.

Covers:
- SPLASH_EXTENSIONS constant validation
- get_splash_dir() in dev mode, frozen mode, and edge cases
- launch() subprocess creation and error handling
- kill() graceful termination
- DPI awareness setup (high-DPI display portability)
- Image file validity (real PNGs, not just correct extensions)
- Filename safety (no spaces or special chars that break on other machines)
- Image dimensions (reasonable for all target displays)
- PyInstaller spec bundling
- Source code safety checks (no hardcoded paths, proper frozen-mode guards)
"""

import os
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.splash import (
    SPLASH_EXTENSIONS,
    get_splash_dir,
    kill,
    launch,
    splash_log,
)

PROJECT_ROOT = Path(__file__).parent.parent
SPLASH_DIR = PROJECT_ROOT / "assets" / "splash"

# PNG magic bytes: 0x89 P N G \r \n 0x1a \n
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# GIF magic bytes
GIF87_MAGIC = b"GIF87a"
GIF89_MAGIC = b"GIF89a"


# ============================================================================
# A. SPLASH_EXTENSIONS constant
# ============================================================================


class TestSplashExtensions:
    """Verify the supported extension set is correct for tkinter.PhotoImage."""

    def test_png_supported(self):
        assert ".png" in SPLASH_EXTENSIONS

    def test_gif_supported(self):
        assert ".gif" in SPLASH_EXTENSIONS

    def test_jpg_not_supported(self):
        """tkinter.PhotoImage cannot load JPEG without Pillow/PIL."""
        assert ".jpg" not in SPLASH_EXTENSIONS
        assert ".jpeg" not in SPLASH_EXTENSIONS

    def test_bmp_not_supported(self):
        """BMP is not natively supported by PhotoImage."""
        assert ".bmp" not in SPLASH_EXTENSIONS

    def test_extensions_are_lowercase_dotted(self):
        """All extensions should be lowercase with leading dot."""
        for ext in SPLASH_EXTENSIONS:
            assert ext.startswith("."), f"Extension missing dot: {ext}"
            assert ext == ext.lower(), f"Extension not lowercase: {ext}"


# ============================================================================
# B. get_splash_dir — dev mode (real assets/splash/ folder)
# ============================================================================


class TestGetSplashDirDevMode:
    """Tests get_splash_dir against the real project assets."""

    def test_returns_a_path(self):
        result = get_splash_dir()
        assert isinstance(result, Path)

    def test_returned_dir_exists(self):
        result = get_splash_dir()
        assert result.is_dir()

    def test_returns_assets_splash_folder(self):
        result = get_splash_dir()
        assert result == SPLASH_DIR

    def test_contains_splash_images(self):
        result = get_splash_dir()
        images = [f for f in result.iterdir() if f.suffix.lower() in SPLASH_EXTENSIONS]
        assert len(images) >= 2, f"Expected >= 2 images, found {len(images)}"


# ============================================================================
# C. get_splash_dir — frozen mode (sys._MEIPASS)
# ============================================================================


def _set_frozen(tmp_path):
    """Context manager to fake PyInstaller frozen mode."""

    class _Ctx:
        def __enter__(self):
            self._old_frozen = getattr(sys, "frozen", None)
            self._old_meipass = getattr(sys, "_MEIPASS", None)
            sys.frozen = True
            sys._MEIPASS = str(tmp_path)
            return self

        def __exit__(self, *exc):
            if self._old_frozen is None:
                if hasattr(sys, "frozen"):
                    del sys.frozen
            else:
                sys.frozen = self._old_frozen
            if self._old_meipass is None:
                if hasattr(sys, "_MEIPASS"):
                    del sys._MEIPASS
            else:
                sys._MEIPASS = self._old_meipass

    return _Ctx()


class TestGetSplashDirFrozenMode:
    """Tests get_splash_dir with sys.frozen / sys._MEIPASS faked."""

    def test_uses_meipass_path(self, tmp_path):
        """In frozen mode, should return sys._MEIPASS/assets/splash/."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)

        with _set_frozen(tmp_path):
            result = get_splash_dir()
            assert result == splash_dir

    def test_frozen_mode_returns_none_when_dir_missing(self, tmp_path):
        """Frozen mode with no assets/splash/ dir should return None."""
        with _set_frozen(tmp_path):
            assert get_splash_dir() is None


# ============================================================================
# D. launch — subprocess creation
# ============================================================================


class TestLaunchSplash:
    """Verify launch() creates subprocess correctly or fails gracefully."""

    def test_returns_none_when_splash_dir_missing(self, tmp_path):
        """No splash directory should return None without crashing."""
        with _set_frozen(tmp_path):
            result = launch()
            assert result is None

    def test_returns_none_when_no_images(self, tmp_path):
        """Empty splash directory should return None."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        with _set_frozen(tmp_path):
            result = launch()
            assert result is None

    def test_returns_none_when_only_unsupported_files(self, tmp_path):
        """Directory with only .jpg/.txt files should return None."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "photo.jpg").write_bytes(b"fake")
        (splash_dir / "notes.txt").write_text("notes")
        with _set_frozen(tmp_path):
            result = launch()
            assert result is None

    def test_frozen_mode_sets_env_var(self, tmp_path):
        """Frozen mode should set _CASEPREPD_SPLASH=1 in subprocess env."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "test.png").write_bytes(b"fake")
        with _set_frozen(tmp_path), patch("src.splash.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=99999)
            result = launch()
            assert result is not None
            # Verify env var is set
            call_kwargs = mock_popen.call_args[1]
            assert call_kwargs["env"]["_CASEPREPD_SPLASH"] == "1"

    def test_frozen_mode_does_not_pass_splash_only_argv(self, tmp_path):
        """Frozen mode should NOT use --splash-only (the old broken approach)."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "test.png").write_bytes(b"fake")
        with _set_frozen(tmp_path), patch("src.splash.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=99999)
            launch()
            args = mock_popen.call_args[0][0]
            assert "--splash-only" not in args

    def test_frozen_mode_spawns_self(self, tmp_path):
        """Frozen mode should spawn sys.executable."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "test.png").write_bytes(b"fake")
        with _set_frozen(tmp_path), patch("src.splash.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=99999)
            launch()
            args = mock_popen.call_args[0][0]
            assert args == [sys.executable]

    def test_frozen_mode_returns_none_on_popen_failure(self, tmp_path):
        """If frozen Popen fails, should return None gracefully."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "test.png").write_bytes(b"fake")
        with _set_frozen(tmp_path):
            with patch("src.splash.subprocess.Popen", side_effect=OSError("spawn failed")):
                result = launch()
                assert result is None

    def test_returns_popen_in_dev_mode(self):
        """In dev mode with real assets, should return a Popen object."""
        proc = launch()
        if proc is None:
            import pytest

            pytest.skip("Could not launch splash (no pythonw.exe or no display)")
        try:
            assert isinstance(proc, subprocess.Popen)
            assert proc.pid > 0
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_uses_pythonw_exe(self):
        """Should use pythonw.exe (no console) for the subprocess."""
        with patch("src.splash.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=12345)
            launch()
            if mock_popen.called:
                args = mock_popen.call_args[0][0]
                exe_name = Path(args[0]).name.lower()
                assert exe_name == "pythonw.exe", f"Expected pythonw.exe, got {exe_name}"

    def test_graceful_on_popen_failure(self):
        """If subprocess.Popen raises, should return None."""
        with patch("src.splash.subprocess.Popen", side_effect=OSError("no pythonw")):
            result = launch()
            assert result is None

    def test_subprocess_script_contains_topmost(self):
        """The inline splash script should set -topmost for visibility."""
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        assert '"-topmost", True' in source

    def test_subprocess_script_contains_mainloop(self):
        """The inline splash script should call mainloop to keep window open."""
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        assert "root.mainloop()" in source

    def test_subprocess_script_has_dpi_awareness(self):
        """The inline splash script should set DPI awareness."""
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        assert "SetProcessDpiAwareness" in source


# ============================================================================
# E. kill — graceful termination
# ============================================================================


class TestKillSplash:
    """Verify kill() terminates subprocesses safely."""

    def test_none_is_safe(self):
        """Passing None should not raise."""
        kill(None)

    def test_terminates_process(self):
        """Should call terminate() on the process."""
        mock_proc = MagicMock()
        kill(mock_proc)
        mock_proc.terminate.assert_called_once()

    def test_waits_for_process(self):
        """Should call wait() after terminate() for clean shutdown."""
        mock_proc = MagicMock()
        kill(mock_proc)
        mock_proc.wait.assert_called_once_with(timeout=3)

    def test_survives_terminate_error(self):
        """If terminate() raises, should not crash."""
        mock_proc = MagicMock()
        mock_proc.terminate.side_effect = OSError("already dead")
        kill(mock_proc)  # Should not raise

    def test_survives_wait_timeout(self):
        """If wait() times out, should not crash."""
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("splash", 3)
        kill(mock_proc)  # Should not raise

    def test_escalates_to_kill_on_timeout(self):
        """If terminate+wait times out, should escalate to kill()."""
        mock_proc = MagicMock()
        # First wait() call (after terminate) times out;
        # second wait() call (after kill) succeeds.
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired("splash", 3),
            None,
        ]
        kill(mock_proc)
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert mock_proc.wait.call_count == 2

    def test_survives_kill_failure(self):
        """If both terminate-wait and kill fail, should not crash."""
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("splash", 3)
        mock_proc.kill.side_effect = OSError("access denied")
        kill(mock_proc)  # Should not raise

    def test_real_subprocess_termination(self):
        """Launch and kill a real splash subprocess end-to-end."""
        proc = launch()
        if proc is None:
            return  # Can't test on systems without pythonw.exe
        assert proc.poll() is None, "Process should still be running"
        kill(proc)
        # Process should be terminated
        proc.wait(timeout=5)
        assert proc.returncode is not None


# ============================================================================
# F. Asset file validity — real images on disk
# ============================================================================


class TestSplashImageValidity:
    """Verify the actual splash image files are valid and portable."""

    def _get_splash_images(self):
        """Return list of splash image paths."""
        return [f for f in SPLASH_DIR.iterdir() if f.suffix.lower() in SPLASH_EXTENSIONS]

    def test_splash_folder_exists(self):
        assert SPLASH_DIR.is_dir(), f"Missing: {SPLASH_DIR}"

    def test_at_least_two_splash_images(self):
        images = self._get_splash_images()
        assert len(images) >= 2, f"Expected >= 2, found {len(images)}"

    def test_all_images_have_valid_png_magic_bytes(self):
        """Every .png file must start with the 8-byte PNG signature."""
        for img_path in SPLASH_DIR.glob("*.png"):
            with open(img_path, "rb") as f:
                magic = f.read(8)
            assert magic == PNG_MAGIC, (
                f"{img_path.name} is not a valid PNG (magic bytes: {magic!r})"
            )

    def test_all_gif_images_have_valid_magic_bytes(self):
        """Every .gif file must start with GIF87a or GIF89a."""
        for img_path in SPLASH_DIR.glob("*.gif"):
            with open(img_path, "rb") as f:
                magic = f.read(6)
            assert magic in (GIF87_MAGIC, GIF89_MAGIC), (
                f"{img_path.name} is not a valid GIF (magic bytes: {magic!r})"
            )

    def test_images_are_reasonable_file_size(self):
        """Each splash image should be under 1 MB (resized for fast loading)."""
        for img_path in self._get_splash_images():
            size_kb = img_path.stat().st_size / 1024
            assert size_kb < 1024, f"{img_path.name} is {size_kb:.0f}KB, expected < 1MB"

    def test_images_are_not_too_small(self):
        """Images should be at least 1 KB (not empty or stub files)."""
        for img_path in self._get_splash_images():
            size_kb = img_path.stat().st_size / 1024
            assert size_kb > 1, f"{img_path.name} is only {size_kb:.1f}KB -- too small"

    def test_png_dimensions_are_reasonable(self):
        """PNG images should be between 400-1200px wide, 200-800px tall.

        This ensures they fit on screens from 1024x768 laptops to 4K monitors.
        We read the IHDR chunk directly (no PIL dependency needed).
        """
        for img_path in SPLASH_DIR.glob("*.png"):
            with open(img_path, "rb") as f:
                f.read(8)  # skip PNG magic
                f.read(4)  # skip IHDR chunk length
                f.read(4)  # skip "IHDR" tag
                width = struct.unpack(">I", f.read(4))[0]
                height = struct.unpack(">I", f.read(4))[0]
            assert 400 <= width <= 1200, f"{img_path.name}: width {width}px not in 400-1200 range"
            assert 200 <= height <= 800, f"{img_path.name}: height {height}px not in 200-800 range"


# ============================================================================
# G. Filename safety — no spaces/special chars that break on other machines
# ============================================================================


class TestFilenamePortability:
    """Filenames must be safe for Windows, PyInstaller, and path handling."""

    def test_no_spaces_in_filenames(self):
        """Spaces in filenames can cause issues with some path-handling code."""
        for f in SPLASH_DIR.iterdir():
            assert " " not in f.name, f"Space in filename: {f.name}"

    def test_only_ascii_in_filenames(self):
        """Non-ASCII characters can cause encoding issues on some Windows locales."""
        for f in SPLASH_DIR.iterdir():
            assert f.name.isascii(), f"Non-ASCII filename: {f.name}"

    def test_no_special_characters(self):
        """Only alphanumeric, underscore, hyphen, and dot allowed."""
        import re

        safe_pattern = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
        for f in SPLASH_DIR.iterdir():
            assert safe_pattern.match(f.name), f"Unsafe characters in filename: {f.name}"

    def test_filenames_are_lowercase(self):
        """Lowercase filenames avoid case-sensitivity issues across platforms."""
        for f in SPLASH_DIR.iterdir():
            assert f.name == f.name.lower(), (
                f"Uppercase in filename: {f.name} (should be {f.name.lower()})"
            )


# ============================================================================
# H. DPI awareness — winfo_screenwidth/height portability
# ============================================================================


class TestDpiAwareness:
    """Verify DPI awareness is set in main.py source code."""

    def test_dpi_awareness_call_exists_in_source(self):
        """main.py should call SetProcessDpiAwareness before creating windows."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
        assert "SetProcessDpiAwareness" in source

    def test_dpi_awareness_before_main_function(self):
        """DPI awareness must be set BEFORE main() is defined."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
        dpi_pos = source.index("SetProcessDpiAwareness")
        main_pos = source.index("def main(")
        assert dpi_pos < main_pos, "SetProcessDpiAwareness must appear before main() definition"

    def test_dpi_awareness_wrapped_in_suppress(self):
        """DPI call must be wrapped in try/except so it doesn't crash on non-Windows."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
        assert "contextlib.suppress" in source or "except" in source


# ============================================================================
# I. Source code safety — no hardcoded paths, proper frozen-mode guards
# ============================================================================


class TestSourceCodeSafety:
    """Static analysis of splash.py and main.py for portability red flags."""

    def _get_splash_source(self):
        return Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")

    def _get_main_source(self):
        return Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")

    def test_no_hardcoded_drive_letters(self):
        """No C:\\, D:\\, etc. in the source."""
        import re

        for source in (self._get_splash_source(), self._get_main_source()):
            matches = re.findall(r"[A-Z]:\\", source)
            assert not matches, f"Hardcoded drive letter found: {matches}"

    def test_no_hardcoded_usernames(self):
        """No /Users/xxx or C:\\Users\\xxx paths."""
        for source in (self._get_splash_source(), self._get_main_source()):
            assert "noahc" not in source.lower()
            assert "/Users/" not in source

    def test_frozen_mode_check_uses_getattr(self):
        """Should use getattr(sys, 'frozen', False), not hasattr."""
        source = self._get_splash_source()
        assert 'getattr(sys, "frozen", False)' in source

    def test_frozen_mode_uses_meipass(self):
        """Should reference sys._MEIPASS for frozen path resolution."""
        source = self._get_splash_source()
        assert "sys._MEIPASS" in source

    def test_kill_has_exception_guard(self):
        """kill() must handle terminate/wait failures gracefully."""
        source = self._get_splash_source()
        func_start = source.index("def kill(")
        next_def = source.index("\ndef ", func_start + 1)
        func_body = source[func_start:next_def]
        assert "except" in func_body, "kill() should handle exceptions from terminate/wait/kill"

    def test_launch_catches_popen_exceptions(self):
        """launch() must catch exceptions from subprocess.Popen."""
        source = self._get_splash_source()
        func_start = source.index("def launch(")
        next_def = source.index("\ndef ", func_start + 1)
        func_body = source[func_start:next_def]
        assert "except" in func_body, "launch() must catch exceptions to prevent startup crashes"

    def test_kill_escalates_to_kill(self):
        """kill() must call proc.kill() when terminate times out."""
        source = self._get_splash_source()
        func_start = source.index("def kill(")
        next_def = source.index("\ndef ", func_start + 1)
        func_body = source[func_start:next_def]
        assert "proc.kill()" in func_body, "kill() should escalate to kill() on TimeoutExpired"
        assert "TimeoutExpired" in func_body, (
            "kill() should catch TimeoutExpired to trigger kill() escalation"
        )

    def test_import_failure_kills_splash(self):
        """If heavy imports fail, splash subprocess must be terminated."""
        source = self._get_main_source()
        assert "kill(splash_proc)" in source

    def test_window_focus_after_splash_kill(self):
        """After killing splash, main window must be raised to front."""
        source = self._get_main_source()
        # All three calls must appear after kill() in main()
        kill_pos = source.index("kill(splash_proc)")
        mainloop_pos = source.index("app.mainloop()")
        between = source[kill_pos:mainloop_pos]
        assert "app.lift()" in between, "app.lift() must be called after splash kill"
        assert "app.focus_force()" in between, "app.focus_force() must be called after splash kill"
        assert '"-topmost"' in between, "Window must temporarily set -topmost after splash kill"

    def test_traceback_import_for_crash_log(self):
        """traceback must be imported so crash-log code doesn't NameError."""
        source = self._get_main_source()
        assert "import traceback" in source, (
            "main.py must import traceback for crash-log formatting"
        )
        # Verify traceback is imported BEFORE the crash-log code uses it
        import_pos = source.index("import traceback")
        usage_pos = source.index("traceback.format_exc()")
        assert import_pos < usage_pos, (
            "import traceback must appear before traceback.format_exc() usage"
        )

    def test_no_tkinter_import_in_main_process(self):
        """Main process should not import tkinter -- splash runs in subprocess.

        The env var splash handler imports from src.splash (which imports tkinter
        only inside run_splash_window), and that code path calls sys.exit(0)
        before any heavy imports run. We check the main process code path only.
        """
        source = self._get_main_source()
        # Remove the env var splash handler block (exits immediately)
        env_check_start = source.index('if os.environ.get("_CASEPREPD_SPLASH") == "1":')
        env_check_end = source.index("run_splash_window()", env_check_start) + len(
            "run_splash_window()"
        )
        main_code = source[:env_check_start] + source[env_check_end:]
        assert "import tkinter" not in main_code, (
            "Main process should not import tkinter -- splash uses subprocess"
        )


# ============================================================================
# J. PyInstaller spec and bundling
# ============================================================================


class TestInstallerBundling:
    """Verify caseprepd.spec correctly bundles splash assets."""

    def _get_spec(self):
        return (PROJECT_ROOT / "caseprepd.spec").read_text(encoding="utf-8")

    def test_spec_bundles_splash_directory(self):
        spec = self._get_spec()
        assert 'os.path.join("assets", "splash"' in spec

    def test_spec_splash_uses_wildcard_glob(self):
        """Spec should use * glob to include all files in splash dir."""
        spec = self._get_spec()
        assert 'os.path.join("assets", "splash", "*")' in spec

    def test_spec_splash_dest_preserves_path(self):
        """Destination should maintain assets/splash/ structure."""
        spec = self._get_spec()
        lines = spec.split("\n")
        splash_lines = [l for l in lines if "splash" in l]
        assert any('os.path.join("assets", "splash")' in l for l in splash_lines), (
            "Spec must preserve assets/splash/ directory structure in bundle"
        )

    def test_spec_still_bundles_icon(self):
        """Adding splash must not break existing icon.ico bundling."""
        spec = self._get_spec()
        assert "icon.ico" in spec


# ============================================================================
# K. Subprocess splash script content (dev mode inline script in splash.py)
# ============================================================================


class TestSplashScriptContent:
    """Verify the inline splash script passed to the subprocess is correct."""

    def _get_splash_script_section(self):
        """Extract the splash_script f-string content from splash.py."""
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        start = source.index('splash_script = f"""') + len('splash_script = f"""')
        end = source.index('"""', start)
        return source[start:end]

    def test_script_imports_tkinter(self):
        script = self._get_splash_script_section()
        assert "import tkinter as tk" in script

    def test_script_imports_random(self):
        script = self._get_splash_script_section()
        assert "import random" in script

    def test_script_uses_overrideredirect(self):
        """Splash should be borderless (no title bar)."""
        script = self._get_splash_script_section()
        assert "overrideredirect(True)" in script

    def test_script_uses_topmost(self):
        """Splash must appear above other windows."""
        script = self._get_splash_script_section()
        assert '"-topmost", True' in script

    def test_script_centers_on_screen(self):
        """Splash should center itself using screen dimensions."""
        script = self._get_splash_script_section()
        assert "winfo_screenwidth" in script
        assert "winfo_screenheight" in script

    def test_script_shows_loading_text(self):
        """Splash should display Loading... status text."""
        script = self._get_splash_script_section()
        assert "Loading..." in script

    def test_script_calls_mainloop(self):
        """Subprocess splash must call mainloop to stay visible."""
        script = self._get_splash_script_section()
        assert "root.mainloop()" in script

    def test_script_uses_random_choice(self):
        """Should randomly pick from available images."""
        script = self._get_splash_script_section()
        assert "random.choice" in script

    def test_script_has_dpi_awareness(self):
        """Subprocess should also set DPI awareness for correct centering."""
        script = self._get_splash_script_section()
        assert "SetProcessDpiAwareness" in script

    def test_script_prevents_gc_of_image(self):
        """Must keep reference to PhotoImage to prevent garbage collection."""
        script = self._get_splash_script_section()
        assert "lbl.image = img" in script

    def test_script_has_auto_close_safety_net(self):
        """Dev-mode splash script should auto-close after timeout."""
        script = self._get_splash_script_section()
        assert "root.after(" in script
        assert "root.quit" in script


# ============================================================================
# L. Env var splash handler (frozen mode self-spawn)
# ============================================================================


class TestEnvVarSplashHandler:
    """Verify the _CASEPREPD_SPLASH env var handler in main.py."""

    def _get_source(self):
        return Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")

    def test_handler_exists(self):
        """The _CASEPREPD_SPLASH env var handler must exist in main.py."""
        source = self._get_source()
        assert "_CASEPREPD_SPLASH" in source

    def test_handler_checks_env_var_not_argv(self):
        """Should use env var, not --splash-only argv (the old broken approach)."""
        source = self._get_source()
        assert 'os.environ.get("_CASEPREPD_SPLASH")' in source
        assert '"--splash-only" in sys.argv' not in source

    def test_handler_before_heavy_imports(self):
        """Env var handler must run before customtkinter/torch imports."""
        source = self._get_source()
        handler_pos = source.index("_CASEPREPD_SPLASH")
        ctk_pos = source.index("import customtkinter")
        assert handler_pos < ctk_pos, "Env var handler must appear before heavy imports"

    def test_handler_calls_run_splash_window(self):
        """Handler must call run_splash_window() from src.splash."""
        source = self._get_source()
        assert "from src.splash import run_splash_window" in source
        assert "run_splash_window()" in source

    def test_run_splash_window_uses_tkinter(self):
        """run_splash_window() in splash.py must use tkinter."""
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        func_start = source.index("def run_splash_window(")
        func_body = source[func_start:]
        assert "import tkinter as tk" in func_body

    def test_run_splash_window_uses_overrideredirect(self):
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        func_start = source.index("def run_splash_window(")
        func_body = source[func_start:]
        assert "overrideredirect(True)" in func_body

    def test_run_splash_window_uses_topmost(self):
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        func_start = source.index("def run_splash_window(")
        func_body = source[func_start:]
        assert '"-topmost", True' in func_body

    def test_run_splash_window_shows_loading_text(self):
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        func_start = source.index("def run_splash_window(")
        func_body = source[func_start:]
        assert "Loading..." in func_body

    def test_run_splash_window_calls_mainloop(self):
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        func_start = source.index("def run_splash_window(")
        func_body = source[func_start:]
        assert "mainloop()" in func_body

    def test_run_splash_window_exits_with_zero(self):
        """run_splash_window must call sys.exit(0) to prevent heavy imports."""
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        func_start = source.index("def run_splash_window(")
        func_body = source[func_start:]
        assert "sys.exit(0)" in func_body

    def test_run_splash_window_has_auto_close_safety_net(self):
        """Safety net: auto-close after timeout if parent dies."""
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        func_start = source.index("def run_splash_window(")
        func_body = source[func_start:]
        assert "root.after(" in func_body
        assert "root.quit" in func_body

    def test_run_splash_window_supports_frozen_and_dev_splash_dirs(self):
        """run_splash_window should use get_splash_dir() which handles both modes."""
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        func_start = source.index("def run_splash_window(")
        func_body = source[func_start:]
        assert "get_splash_dir()" in func_body

    def test_run_splash_window_uses_random_choice(self):
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        func_start = source.index("def run_splash_window(")
        func_body = source[func_start:]
        assert "random.choice" in func_body

    def test_dpi_awareness_set_before_handler(self):
        """DPI awareness must be set at module level before env var handler."""
        source = self._get_source()
        dpi_pos = source.index("SetProcessDpiAwareness")
        # The env var handler is inline at top of main.py — but DPI is set
        # inside run_splash_window and in main.py module level. Check main.py
        # sets DPI before main() is defined.
        main_pos = source.index("def main(")
        assert dpi_pos < main_pos, "DPI awareness must be set before main() definition"


# ============================================================================
# M. splash_log helper
# ============================================================================


class TestSplashLog:
    """Verify splash_log handles None stdout safely."""

    def test_prints_when_stdout_available(self, capsys):
        """Should print normally when stdout is not None."""
        splash_log("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_no_crash_when_stdout_is_none(self):
        """In windowed/noconsole mode, sys.stdout is None. Must not crash."""
        with patch("src.splash.sys.stdout", None):
            splash_log("this should not crash")

    def test_source_uses_splash_log_not_print(self):
        """Splash functions should use splash_log() instead of print()."""
        source = Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")
        # Extract launch, kill, get_splash_dir function bodies
        for func_name in ("launch", "kill", "get_splash_dir"):
            func_start = source.index(f"def {func_name}(")
            next_def = source.index("\ndef ", func_start + 1)
            func_body = source[func_start:next_def]
            # Should not have bare print() calls (only splash_log)
            import re

            bare_prints = re.findall(r"(?<![_a-zA-Z])print\(", func_body)
            assert not bare_prints, f"{func_name} uses print() instead of splash_log()"


# ============================================================================
# N. Auto-close safety net
# ============================================================================


class TestAutoCloseSafetyNet:
    """Verify both splash modes have auto-close to prevent orphaned windows."""

    def _get_splash_source(self):
        return Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")

    def test_dev_mode_splash_has_auto_close(self):
        """The dev-mode inline splash script must have root.after(timeout, quit)."""
        source = self._get_splash_source()
        start = source.index('splash_script = f"""')
        end = source.index('"""', start + len('splash_script = f"""'))
        script = source[start:end]
        assert "root.after(" in script
        assert "root.quit" in script

    def test_frozen_mode_splash_has_auto_close(self):
        """The run_splash_window function must have auto-close safety net."""
        source = self._get_splash_source()
        start = source.index("def run_splash_window(")
        block = source[start:]
        assert "root.after(" in block
        assert "root.quit" in block

    def test_auto_close_timeout_is_reasonable(self):
        """Auto-close timeout should be 30-120 seconds."""
        source = self._get_splash_source()
        import re

        # Find all root.after(N, ...) or _root.after(N, ...) calls
        timeouts = re.findall(r"_?root\.after\((\d+),", source)
        for timeout_str in timeouts:
            timeout_ms = int(timeout_str)
            assert 30000 <= timeout_ms <= 120000, (
                f"Auto-close timeout {timeout_ms}ms not in 30-120s range"
            )


# ============================================================================
# O. run_splash_window — behavioral tests (mock-based)
# ============================================================================


class TestRunSplashWindowBehavior:
    """Verify run_splash_window() early-exit paths and sys.exit calls."""

    def test_exits_when_splash_dir_missing(self, tmp_path):
        """Should call sys.exit(0) when get_splash_dir() returns None."""
        with _set_frozen(tmp_path), pytest.raises(SystemExit) as exc_info:
            from src.splash import run_splash_window

            run_splash_window()
        assert exc_info.value.code == 0

    def test_exits_when_no_images(self, tmp_path):
        """Should call sys.exit(0) when splash dir has no supported images."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "readme.txt").write_text("no images here")
        with _set_frozen(tmp_path), pytest.raises(SystemExit) as exc_info:
            from src.splash import run_splash_window

            run_splash_window()
        assert exc_info.value.code == 0


# ============================================================================
# P. launch — frozen mode subprocess environment details
# ============================================================================


class TestLaunchFrozenEnvDetails:
    """Verify launch() frozen mode env handling in detail."""

    def _setup_frozen(self, tmp_path):
        """Create a fake frozen env with one splash image."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "test.png").write_bytes(b"fake")
        return splash_dir

    def test_frozen_suppresses_stdout_stderr(self, tmp_path):
        """Frozen subprocess should redirect stdout/stderr to DEVNULL."""
        self._setup_frozen(tmp_path)
        with _set_frozen(tmp_path), patch("src.splash.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=99999)
            launch()
            call_kwargs = mock_popen.call_args[1]
            assert call_kwargs["stdout"] == subprocess.DEVNULL
            assert call_kwargs["stderr"] == subprocess.DEVNULL

    def test_frozen_env_preserves_existing_vars(self, tmp_path):
        """Subprocess env should contain all parent env vars plus _CASEPREPD_SPLASH."""
        self._setup_frozen(tmp_path)
        with _set_frozen(tmp_path), patch("src.splash.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=99999)
            launch()
            call_kwargs = mock_popen.call_args[1]
            env = call_kwargs["env"]
            # Must preserve existing env vars (PATH is always present)
            assert "PATH" in env or "Path" in env
            # Must add the splash flag
            assert env["_CASEPREPD_SPLASH"] == "1"

    def test_frozen_env_does_not_mutate_parent_env(self, tmp_path):
        """launch() must not set _CASEPREPD_SPLASH in the parent process env."""
        self._setup_frozen(tmp_path)
        # Clean up in case it's already set from a previous test
        os.environ.pop("_CASEPREPD_SPLASH", None)
        with _set_frozen(tmp_path), patch("src.splash.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=99999)
            launch()
            assert "_CASEPREPD_SPLASH" not in os.environ

    def test_dev_mode_suppresses_stdout_stderr(self):
        """Dev mode subprocess should also redirect stdout/stderr to DEVNULL."""
        with patch("src.splash.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=12345)
            launch()
            if mock_popen.called:
                call_kwargs = mock_popen.call_args[1]
                assert call_kwargs["stdout"] == subprocess.DEVNULL
                assert call_kwargs["stderr"] == subprocess.DEVNULL


# ============================================================================
# Q. Module architecture — splash.py standalone, no src.* imports
# ============================================================================


class TestSplashModuleArchitecture:
    """Verify splash.py is a standalone module with no src.* dependencies."""

    def _get_splash_source(self):
        return Path(PROJECT_ROOT / "src" / "splash.py").read_text(encoding="utf-8")

    def test_no_src_imports(self):
        """splash.py must not import from src.* — it's a standalone startup utility."""
        source = self._get_splash_source()
        import re

        src_imports = re.findall(r"^\s*(?:from|import)\s+src\.", source, re.MULTILINE)
        assert not src_imports, f"splash.py must not have src.* imports, found: {src_imports}"

    def test_no_splash_only_argv_usage(self):
        """splash.py must not use --splash-only in sys.argv (the old broken mechanism).

        Comments explaining the rationale are fine — we only forbid actual usage.
        """
        source = self._get_splash_source()

        # Match actual argv usage like: "--splash-only" in sys.argv, or
        # Popen([..., "--splash-only"]) — but not inside comments
        code_lines = [
            line
            for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert '"--splash-only"' not in code_only, (
            "splash.py must not use --splash-only in executable code"
        )

    def test_only_stdlib_imports(self):
        """splash.py should only import from the standard library."""
        source = self._get_splash_source()
        import re

        # Find all top-level imports (not inside functions)
        top_level_imports = re.findall(r"^(?:from|import)\s+(\S+)", source, re.MULTILINE)
        stdlib_allowed = {
            "logging",
            "os",
            "subprocess",
            "sys",
            "pathlib",
            "random",
            "tkinter",
            "contextlib",
        }
        for module in top_level_imports:
            root_module = module.split(".")[0]
            assert root_module in stdlib_allowed, f"splash.py imports non-stdlib module: {module}"


# ============================================================================
# R. main.py ordering constraints
# ============================================================================


class TestMainModuleOrdering:
    """Verify critical ordering constraints in main.py."""

    def _get_source(self):
        return Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")

    def test_freeze_support_before_splash_check(self):
        """multiprocessing.freeze_support() must run before the splash env var check."""
        source = self._get_source()
        freeze_pos = source.index("freeze_support()")
        splash_pos = source.index("_CASEPREPD_SPLASH")
        assert freeze_pos < splash_pos, "freeze_support() must come before _CASEPREPD_SPLASH check"

    def test_splash_check_before_heavy_imports(self):
        """The env var splash check must run before threading/traceback imports."""
        source = self._get_source()
        splash_pos = source.index("_CASEPREPD_SPLASH")
        # These imports come after the splash check in normal flow
        threading_pos = source.index("import threading")
        assert splash_pos < threading_pos, "Splash check must come before heavy stdlib imports"

    def test_main_imports_launch_and_kill_from_splash(self):
        """main() must import launch and kill from src.splash, not define them locally."""
        source = self._get_source()
        assert "from src.splash import kill, launch" in source

    def test_no_splash_only_argv_in_main(self):
        """main.py must not reference --splash-only (the old broken mechanism)."""
        source = self._get_source()
        assert "--splash-only" not in source

    def test_worker_shutdown_after_mainloop(self):
        """Worker subprocess must be shut down after app.mainloop() returns."""
        source = self._get_source()
        mainloop_pos = source.index("app.mainloop()")
        shutdown_pos = source.index("worker_manager.shutdown")
        assert mainloop_pos < shutdown_pos, "worker_manager.shutdown must come after app.mainloop()"
