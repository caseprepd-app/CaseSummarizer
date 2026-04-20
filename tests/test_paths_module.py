"""Tests for src/core/paths.py — centralized path resolution.

src.core.paths is the single source of truth for base/config/data/assets
directory detection in both dev and PyInstaller-frozen modes.

Covers:
- get_base_dir(): project root in dev, sys._MEIPASS when frozen
- get_config_dir(), get_data_dir(), get_assets_dir(): compose off base
"""

import sys
from pathlib import Path
from unittest.mock import patch


class TestGetBaseDirDevMode:
    """get_base_dir() returns the project root in development mode."""

    def test_dev_mode_returns_existing_directory(self):
        """In dev mode (not frozen) the resolved base dir exists on disk."""
        from src.core.paths import get_base_dir

        with patch.object(sys, "frozen", False, create=True):
            base = get_base_dir()
        assert base.exists(), f"Base dir should exist in dev: {base}"

    def test_dev_mode_contains_src_subdirectory(self):
        """Dev-mode base dir is the project root, so src/ lives beneath it."""
        from src.core.paths import get_base_dir

        # When running tests, the module is unfrozen by default.
        # Ensure `frozen` attribute absent or False.
        if hasattr(sys, "frozen"):
            # Force treat as unfrozen for this check
            with patch.object(sys, "frozen", False):
                base = get_base_dir()
        else:
            base = get_base_dir()
        assert (base / "src").is_dir(), "Project root should contain src/ folder"

    def test_dev_mode_returns_path_object(self):
        """get_base_dir() returns a pathlib.Path."""
        from src.core.paths import get_base_dir

        assert isinstance(get_base_dir(), Path)


class TestGetBaseDirFrozenMode:
    """When sys.frozen is True, get_base_dir() uses sys._MEIPASS."""

    def test_frozen_mode_uses_meipass(self, tmp_path):
        """Frozen binaries read the base dir from sys._MEIPASS."""
        from src.core.paths import get_base_dir

        fake_meipass = str(tmp_path)
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "_MEIPASS", fake_meipass, create=True):
                base = get_base_dir()
        assert base == Path(fake_meipass)


class TestGetConfigDir:
    """get_config_dir() returns <base>/config."""

    def test_is_subdir_of_base(self):
        """Config dir is the base dir with /config appended."""
        from src.core.paths import get_base_dir, get_config_dir

        assert get_config_dir() == get_base_dir() / "config"

    def test_returns_path_object(self):
        """Return type is Path."""
        from src.core.paths import get_config_dir

        assert isinstance(get_config_dir(), Path)


class TestGetDataDir:
    """get_data_dir() returns <base>/data."""

    def test_is_subdir_of_base(self):
        """Data dir is the base dir with /data appended."""
        from src.core.paths import get_base_dir, get_data_dir

        assert get_data_dir() == get_base_dir() / "data"

    def test_returns_path_object(self):
        """Return type is Path."""
        from src.core.paths import get_data_dir

        assert isinstance(get_data_dir(), Path)


class TestGetAssetsDir:
    """get_assets_dir() returns <base>/assets."""

    def test_is_subdir_of_base(self):
        """Assets dir is the base dir with /assets appended."""
        from src.core.paths import get_assets_dir, get_base_dir

        assert get_assets_dir() == get_base_dir() / "assets"

    def test_returns_path_object(self):
        """Return type is Path."""
        from src.core.paths import get_assets_dir

        assert isinstance(get_assets_dir(), Path)


class TestPathConsistency:
    """All derived dirs share the same base."""

    def test_all_dirs_share_common_parent(self):
        """config/, data/, assets/ all have the same parent (the base dir)."""
        from src.core.paths import (
            get_assets_dir,
            get_base_dir,
            get_config_dir,
            get_data_dir,
        )

        base = get_base_dir()
        assert get_config_dir().parent == base
        assert get_data_dir().parent == base
        assert get_assets_dir().parent == base

    def test_dirs_are_distinct(self):
        """config, data, and assets resolve to different directories."""
        from src.core.paths import get_assets_dir, get_config_dir, get_data_dir

        dirs = {get_config_dir(), get_data_dir(), get_assets_dir()}
        assert len(dirs) == 3
