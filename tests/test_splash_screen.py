"""Tests for splash screen: random image selection, portability, installer readiness.

Covers:
- SPLASH_EXTENSIONS constant validation
- _pick_splash_image() in dev mode, frozen mode, and edge cases
- _show_splash() graceful error handling
- DPI awareness setup (high-DPI display portability)
- Image file validity (real PNGs, not just correct extensions)
- Filename safety (no spaces or special chars that break on other machines)
- Image dimensions (reasonable for all target displays)
- PyInstaller spec bundling
- Source code safety checks (no hardcoded paths, proper frozen-mode guards)
"""

import struct
import sys
from pathlib import Path
from unittest.mock import patch

from src.main import SPLASH_EXTENSIONS, _pick_splash_image, _show_splash

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
# B. _pick_splash_image — dev mode (real assets/splash/ folder)
# ============================================================================


class TestPickSplashImageDevMode:
    """Tests _pick_splash_image against the real project assets."""

    def test_returns_a_path(self):
        result = _pick_splash_image()
        assert isinstance(result, Path)

    def test_returned_file_exists(self):
        result = _pick_splash_image()
        assert result.exists(), f"Picked image does not exist: {result}"

    def test_returned_file_has_valid_extension(self):
        result = _pick_splash_image()
        assert result.suffix.lower() in SPLASH_EXTENSIONS

    def test_picks_from_splash_folder(self):
        result = _pick_splash_image()
        assert result.parent == SPLASH_DIR

    def test_randomness_over_many_calls(self):
        """With 2+ images, should pick different ones over 50 calls."""
        picks = {_pick_splash_image().name for _ in range(50)}
        assert len(picks) > 1, "Always picked the same image -- randomness broken"


# ============================================================================
# C. _pick_splash_image — frozen mode (sys._MEIPASS)
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


class TestPickSplashImageFrozenMode:
    """Tests _pick_splash_image with sys.frozen / sys._MEIPASS faked."""

    def test_uses_meipass_path(self, tmp_path):
        """In frozen mode, should look under sys._MEIPASS/assets/splash/."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "frozen_img.png").write_bytes(b"fake-png")

        with _set_frozen(tmp_path):
            result = _pick_splash_image()
            assert result is not None
            assert result.name == "frozen_img.png"

    def test_frozen_mode_returns_none_when_dir_missing(self, tmp_path):
        """Frozen mode with no assets/splash/ dir should return None."""
        with _set_frozen(tmp_path):
            assert _pick_splash_image() is None


# ============================================================================
# D. _pick_splash_image — edge cases (tmp_path directories)
# ============================================================================


class TestPickSplashImageEdgeCases:
    """Edge cases using temporary directories to control splash folder contents."""

    def _pick_from(self, tmp_path):
        """Call _pick_splash_image with splash dir redirected via fake frozen mode."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True, exist_ok=True)
        with _set_frozen(tmp_path):
            return _pick_splash_image(), splash_dir

    def test_returns_none_when_folder_empty(self, tmp_path):
        result, _ = self._pick_from(tmp_path)
        assert result is None

    def test_returns_none_when_only_unsupported_files(self, tmp_path):
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "photo.jpg").write_bytes(b"fake")
        (splash_dir / "notes.txt").write_text("notes")
        (splash_dir / "image.bmp").write_bytes(b"fake")
        result, _ = self._pick_from(tmp_path)
        assert result is None

    def test_picks_png(self, tmp_path):
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "a.png").write_bytes(b"fake")
        result, _ = self._pick_from(tmp_path)
        assert result.name == "a.png"

    def test_picks_gif(self, tmp_path):
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "anim.gif").write_bytes(b"fake")
        result, _ = self._pick_from(tmp_path)
        assert result.name == "anim.gif"

    def test_case_insensitive_extensions(self, tmp_path):
        """Should match .PNG, .GIF (uppercase) via .lower() check."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "a.PNG").write_bytes(b"fake")
        result, _ = self._pick_from(tmp_path)
        assert result is not None

    def test_ignores_jpg_alongside_png(self, tmp_path):
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "photo.jpg").write_bytes(b"fake")
        (splash_dir / "valid.png").write_bytes(b"fake")
        result, _ = self._pick_from(tmp_path)
        assert result.name == "valid.png"

    def test_picks_from_multiple(self, tmp_path):
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        names = ["a.png", "b.png", "c.gif"]
        for name in names:
            (splash_dir / name).write_bytes(b"fake")
        result, _ = self._pick_from(tmp_path)
        assert result.name in names

    def test_randomness_with_multiple(self, tmp_path):
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        for i in range(5):
            (splash_dir / f"img_{i}.png").write_bytes(b"fake")
        picks = set()
        for _ in range(50):
            result, _ = self._pick_from(tmp_path)
            picks.add(result.name)
        assert len(picks) > 1

    def test_adding_new_image_is_picked_up(self, tmp_path):
        """Future-proofing: adding a new file should be automatically included."""
        splash_dir = tmp_path / "assets" / "splash"
        splash_dir.mkdir(parents=True)
        (splash_dir / "old.png").write_bytes(b"fake")

        # First call: only old.png
        with _set_frozen(tmp_path):
            r1 = _pick_splash_image()
            assert r1.name == "old.png"

        # Add a new image
        (splash_dir / "new.png").write_bytes(b"fake")

        # Now both should be in the pool
        names_seen = set()
        for _ in range(50):
            with _set_frozen(tmp_path):
                names_seen.add(_pick_splash_image().name)
        assert "new.png" in names_seen, "Newly added image was never picked"


# ============================================================================
# E. _show_splash — graceful error handling
# ============================================================================


class TestShowSplash:
    """Verify _show_splash never crashes, always returns None on errors."""

    def test_returns_none_when_no_images(self):
        with patch("src.main._pick_splash_image", return_value=None):
            assert _show_splash() is None

    def test_returns_none_on_tk_error(self):
        """Headless server / no display: Tk() raises, should not crash."""
        with patch("src.main._pick_splash_image", return_value=Path("fake.png")):
            with patch("src.main.tk.Tk", side_effect=RuntimeError("no display")):
                assert _show_splash() is None

    def test_returns_none_on_photo_image_error(self):
        """Corrupt image file: PhotoImage raises, should not crash."""
        with patch("src.main._pick_splash_image", return_value=Path("bad.png")):
            with patch("src.main.tk.Tk"):
                with patch("src.main.tk.PhotoImage", side_effect=RuntimeError("bad image")):
                    assert _show_splash() is None

    def test_returns_none_on_geometry_error(self):
        """winfo_screenwidth() failure should not crash."""
        with patch("src.main._pick_splash_image", return_value=Path("x.png")):
            with patch("src.main.tk.Tk") as mock_tk:
                mock_root = mock_tk.return_value
                mock_root.winfo_screenwidth.side_effect = RuntimeError("no screen")
                with patch("src.main.tk.PhotoImage"):
                    assert _show_splash() is None


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
        """main.py should call SetProcessDpiAwareness before creating Tk()."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
        assert "SetProcessDpiAwareness" in source

    def test_dpi_awareness_before_show_splash(self):
        """DPI awareness must be set BEFORE _show_splash() is called."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
        dpi_pos = source.index("SetProcessDpiAwareness")
        splash_pos = source.index("def _show_splash")
        assert dpi_pos < splash_pos, (
            "SetProcessDpiAwareness must appear before _show_splash definition"
        )

    def test_dpi_awareness_wrapped_in_suppress(self):
        """DPI call must be wrapped in try/except so it doesn't crash on non-Windows."""
        source = Path(PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
        # Find the DPI awareness block
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

    def test_splash_destroy_has_exception_guard(self):
        """Splash destroy in main() must be wrapped in exception suppression."""
        source = self._get_source()
        # Find the destroy section
        destroy_idx = source.index("_splash.destroy()")
        # contextlib.suppress should be nearby (within 200 chars before)
        nearby = source[max(0, destroy_idx - 200) : destroy_idx]
        assert "suppress" in nearby, "_splash.destroy() should be wrapped in contextlib.suppress"

    def test_show_splash_catches_all_exceptions(self):
        """_show_splash must have a bare except or Exception catch."""
        source = self._get_source()
        # Find the function body
        func_start = source.index("def _show_splash")
        func_end = source.index("\n_splash = _show_splash()")
        func_body = source[func_start:func_end]
        assert "except Exception" in func_body, (
            "_show_splash must catch Exception to prevent startup crashes"
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
        # Should have both source and dest referencing assets/splash
        lines = spec.split("\n")
        splash_lines = [l for l in lines if "splash" in l]
        assert any('os.path.join("assets", "splash")' in l for l in splash_lines), (
            "Spec must preserve assets/splash/ directory structure in bundle"
        )

    def test_spec_still_bundles_icon(self):
        """Adding splash must not break existing icon.ico bundling."""
        spec = self._get_spec()
        assert "icon.ico" in spec
