"""
Tests for model_loader.py — model path resolution and HF cache setup.

Covers bundled path resolution, frozen-mode enforcement, and env var setup.
"""

import os
import sys
from unittest.mock import patch

import pytest

from src.core.utils.model_loader import resolve_model_path, set_hf_cache_env

# =========================================================================
# resolve_model_path
# =========================================================================


class TestResolveModelPath:
    """Tests for resolve_model_path()."""

    def test_bundled_path_exists(self, tmp_path):
        """Returns local path when bundled model directory exists."""
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()

        path, is_local = resolve_model_path(model_dir, "org/model-name")
        assert path == str(model_dir)
        assert is_local is True

    def test_dev_mode_falls_back_to_hf_name(self, tmp_path):
        """Returns HF model name when bundled path missing in dev mode."""
        missing = tmp_path / "nonexistent_model"

        with patch.object(sys, "frozen", False, create=True):
            path, is_local = resolve_model_path(missing, "org/model-name")

        assert path == "org/model-name"
        assert is_local is False

    def test_frozen_mode_raises_when_missing(self, tmp_path):
        """Raises RuntimeError when bundled model missing in frozen mode."""
        missing = tmp_path / "nonexistent_model"

        with patch.object(sys, "frozen", True, create=True):
            with pytest.raises(RuntimeError, match="not found"):
                resolve_model_path(missing, "org/model-name")

    def test_bundled_path_preferred_even_in_frozen(self, tmp_path):
        """Bundled path returned even when frozen, if it exists."""
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()

        with patch.object(sys, "frozen", True, create=True):
            path, is_local = resolve_model_path(model_dir, "org/model-name")

        assert path == str(model_dir)
        assert is_local is True


# =========================================================================
# set_hf_cache_env
# =========================================================================


class TestSetHfCacheEnv:
    """Tests for set_hf_cache_env()."""

    def test_sets_hf_home(self, tmp_path):
        """Sets HF_HOME environment variable."""
        cache_dir = tmp_path / "hf_cache"
        old = os.environ.get("HF_HOME")

        try:
            set_hf_cache_env(cache_dir)
            assert os.environ["HF_HOME"] == str(cache_dir)
        finally:
            if old is not None:
                os.environ["HF_HOME"] = old
            else:
                os.environ.pop("HF_HOME", None)

    def test_sets_transformers_cache(self, tmp_path):
        """Sets TRANSFORMERS_CACHE environment variable."""
        cache_dir = tmp_path / "hf_cache"
        old = os.environ.get("TRANSFORMERS_CACHE")

        try:
            set_hf_cache_env(cache_dir)
            assert os.environ["TRANSFORMERS_CACHE"] == str(cache_dir)
        finally:
            if old is not None:
                os.environ["TRANSFORMERS_CACHE"] = old
            else:
                os.environ.pop("TRANSFORMERS_CACHE", None)

    def test_overwrites_existing_values(self, tmp_path):
        """Overwrites previously set HF env vars."""
        cache1 = tmp_path / "cache1"
        cache2 = tmp_path / "cache2"
        old_home = os.environ.get("HF_HOME")
        old_cache = os.environ.get("TRANSFORMERS_CACHE")

        try:
            set_hf_cache_env(cache1)
            set_hf_cache_env(cache2)
            assert os.environ["HF_HOME"] == str(cache2)
            assert os.environ["TRANSFORMERS_CACHE"] == str(cache2)
        finally:
            if old_home is not None:
                os.environ["HF_HOME"] = old_home
            else:
                os.environ.pop("HF_HOME", None)
            if old_cache is not None:
                os.environ["TRANSFORMERS_CACHE"] = old_cache
            else:
                os.environ.pop("TRANSFORMERS_CACHE", None)
