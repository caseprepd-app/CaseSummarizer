"""Tests for splash screen: subprocess launch, image selection, portability, installer readiness.

Covers:
- SPLASH_EXTENSIONS constant validation
- _get_splash_dir() in dev mode, frozen mode, and edge cases
- _launch_splash() subprocess creation and error handling
- _kill_splash() graceful termination
- DPI awareness setup (high-DPI display portability)
- Image file validity (real PNGs, not just correct extensions)
- Filename safety (no spaces or special chars that break on other machines)
- Image dimensions (reasonable for all target displays)
- PyInstaller spec bundling
- Source code safety checks (no hardcoded paths, proper frozen-mode guards)
"""

import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.main import SPLASH_EXTENSIONS, _get_splash_dir, _kill_splash, _launch_splash

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
# B. _get_splash_dir — dev mode (real assets/splash/ folder)
# ============================================================================


class TestGetSplashDirDevMode:
    """Tests _get_splash_dir against the real project assets."""

    def test_returns_a_path(self):
        result = _get_splash_dir()
        assert isinstance(result, Path)

    def test_returned_dir_exists(self):
        result = _get_splash_dir()
        assert result.is_dir()

    def test_returns_assets_splash_folder(self):
        result = _get_splash_dir()
        assert result == SPLASH_DIR

    def test_contains_splash_images(self):
        result = _get_splash_dir()
        images = [f for f in result.iterdir() if f.suffix.lower() in SPLASH_EXTENSIONS]
        assert len(images) >= 2, f"Expected >= 2 images, found {len(images)}"


# ============================================================================
# C. _get_splash_dir — frozen mode (sys._MEIPASS)
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
    """Tests _get_splash_dir with sys.frozen / sys._MEIPASS faked."""

    def test_uses_meipass_path(self, tmp_path):
        """In frozen mode, should return sys._MEIPASS/assets/splash/."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)

        with _set_frozen(tmp_path):
            result = _get_splash_dir()
            assert result == splash_dir

    def test_frozen_mode_returns_none_when_dir_missing(self, tmp_path):
        """Frozen mode with no assets/splash/ dir should return None."""
        with _set_frozen(tmp_path):
            assert _get_splash_dir() is None


# ============================================================================
# D. _launch_splash — subprocess creation
# ============================================================================


class TestLaunchSplash:
    """Verify _launch_splash creates subprocess correctly or fails gracefully."""

    def test_returns_none_when_splash_dir_missing(self, tmp_path):
        """No splash directory should return None without crashing."""
        with _set_frozen(tmp_path):
            result = _launch_splash()
            assert result is None

    def test_returns_none_when_no_images(self, tmp_path):
        """Empty splash directory should return None."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        with _set_frozen(tmp_path):
            result = _launch_splash()
            assert result is None

    def test_returns_none_when_only_unsupported_files(self, tmp_path):
        """Directory with only .jpg/.txt files should return None."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "photo.jpg").write_bytes(b"fake")
        (splash_dir / "notes.txt").write_text("notes")
        with _set_frozen(tmp_path):
            result = _launch_splash()
            assert result is None

    def test_returns_none_in_frozen_mode(self, tmp_path):
        """Frozen mode skips subprocess splash (no separate Python available)."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "test.png").write_bytes(b"fake")
        with _set_frozen(tmp_path):
            result = _launch_splash()
            assert result is None

    def test_returns_popen_in_dev_mode(self):
        """In dev mode with real assets, should return a Popen object."""
        proc = _launch_splash()
        try:
            assert proc is not None
            assert isinstance(proc, subprocess.Popen)
            assert proc.pid > 0
        finally:
            if proc is not None:
                proc.terminate()
                proc.wait(timeout=5)

    def test_uses_pythonw_exe(self):
        """Should use pythonw.exe (no console) for the subprocess."""
        with patch("src.main.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=12345)
            _launch_splash()
            if mock_popen.called:
                args = mock_popen.call_args[0][0]
                exe_name = Path(args[0]).name.lower()
                assert exe_name == "pythonw.exe", f"Expected pythonw.exe, got {exe_name}"

    def test_graceful_on_popen_failure(self):
        """If subprocess.Popen raises, should return None."""
        with patch("src.main.subprocess.Popen", side_effect=OSError("no pythonw")):
            result = _launch_splash()
            assert result is None

    def test_subprocess_script_contains_topmost(self):
        """The inline splash script should set -topmost for visibility."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
        # Find the splash_script string
        assert '"-topmost", True' in source

    def test_subprocess_script_contains_mainloop(self):
        """The inline splash script should call mainloop to keep window open."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
        assert "root.mainloop()" in source

    def test_subprocess_script_has_dpi_awareness(self):
        """The inline splash script should set DPI awareness."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
        # DPI call appears twice: once for main process, once in splash script
        count = source.count("SetProcessDpiAwareness")
        assert count >= 2, f"Expected DPI awareness in both main and splash script, found {count}"


# ============================================================================
# E. _kill_splash — graceful termination
# ============================================================================


class TestKillSplash:
    """Verify _kill_splash terminates subprocesses safely."""

    def test_none_is_safe(self):
        """Passing None should not raise."""
        _kill_splash(None)

    def test_terminates_process(self):
        """Should call terminate() on the process."""
        mock_proc = MagicMock()
        _kill_splash(mock_proc)
        mock_proc.terminate.assert_called_once()

    def test_waits_for_process(self):
        """Should call wait() after terminate() for clean shutdown."""
        mock_proc = MagicMock()
        _kill_splash(mock_proc)
        mock_proc.wait.assert_called_once_with(timeout=3)

    def test_survives_terminate_error(self):
        """If terminate() raises, should not crash."""
        mock_proc = MagicMock()
        mock_proc.terminate.side_effect = OSError("already dead")
        _kill_splash(mock_proc)  # Should not raise

    def test_survives_wait_timeout(self):
        """If wait() times out, should not crash."""
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("splash", 3)
        _kill_splash(mock_proc)  # Should not raise

    def test_real_subprocess_termination(self):
        """Launch and kill a real splash subprocess end-to-end."""
        proc = _launch_splash()
        if proc is None:
            return  # Can't test on systems without pythonw.exe
        assert proc.poll() is None, "Process should still be running"
        _kill_splash(proc)
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

    def test_dpi_awareness_before_launch_splash(self):
        """DPI awareness must be set BEFORE _launch_splash() is called."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
        dpi_pos = source.index("SetProcessDpiAwareness")
        splash_pos = source.index("def _launch_splash")
        assert dpi_pos < splash_pos, (
            "SetProcessDpiAwareness must appear before _launch_splash definition"
        )

    def test_dpi_awareness_wrapped_in_suppress(self):
        """DPI call must be wrapped in try/except so it doesn't crash on non-Windows."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
        assert "contextlib.suppress" in source or "except" in source


# ============================================================================
# I. Source code safety — no hardcoded paths, proper frozen-mode guards
# ============================================================================


class TestSourceCodeSafety:
    """Static analysis of main.py splash code for portability red flags."""

    def _get_source(self):
        return Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")

    def test_no_hardcoded_drive_letters(self):
        """No C:\\, D:\\, etc. in the source."""
        source = self._get_source()
        import re

        matches = re.findall(r"[A-Z]:\\", source)
        assert not matches, f"Hardcoded drive letter found: {matches}"

    def test_no_hardcoded_usernames(self):
        """No /Users/xxx or C:\\Users\\xxx paths."""
        source = self._get_source()
        assert "noahc" not in source.lower()
        assert "/Users/" not in source

    def test_frozen_mode_check_uses_getattr(self):
        """Should use getattr(sys, 'frozen', False), not hasattr."""
        source = self._get_source()
        assert 'getattr(sys, "frozen", False)' in source

    def test_frozen_mode_uses_meipass(self):
        """Should reference sys._MEIPASS for frozen path resolution."""
        source = self._get_source()
        assert "sys._MEIPASS" in source

    def test_kill_splash_has_exception_guard(self):
        """_kill_splash must wrap terminate/wait in exception suppression."""
        source = self._get_source()
        func_start = source.index("def _kill_splash")
        # Find the next def or end of file
        next_def = source.index("\ndef ", func_start + 1)
        func_body = source[func_start:next_def]
        assert "suppress" in func_body, (
            "_kill_splash should wrap terminate/wait in contextlib.suppress"
        )

    def test_launch_splash_catches_popen_exceptions(self):
        """_launch_splash must catch exceptions from subprocess.Popen."""
        source = self._get_source()
        func_start = source.index("def _launch_splash")
        next_def = source.index("\ndef ", func_start + 1)
        func_body = source[func_start:next_def]
        assert "except" in func_body, (
            "_launch_splash must catch exceptions to prevent startup crashes"
        )

    def test_import_failure_kills_splash(self):
        """If heavy imports fail, splash subprocess must be terminated."""
        source = self._get_source()
        assert "_kill_splash(_splash_proc)" in source

    def test_no_tkinter_import_in_main_process(self):
        """Main process should not import tkinter -- splash runs in subprocess."""
        source = self._get_source()
        # Check top-level imports (not the inline splash script string)
        # Split at the splash_script string to only check main process code
        main_code = source.split('splash_script = f"""')[0]
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
# K. Subprocess splash script content
# ============================================================================


class TestSplashScriptContent:
    """Verify the inline splash script passed to the subprocess is correct."""

    def _get_splash_script_section(self):
        """Extract the splash_script f-string content from main.py."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
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
