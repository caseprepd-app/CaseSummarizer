"""
Tests for the unified config loader (src/core/config/loader.py).

Covers:
- YAML loading with success/failure/parse-error paths
- Default fallback behavior
- Save functionality
- Encoding edge cases
"""

from pathlib import Path

import pytest
import yaml

# ============================================================================
# A. load_yaml
# ============================================================================


class TestLoadYaml:
    """Tests for load_yaml()."""

    def test_loads_valid_yaml(self, tmp_path):
        """Successfully loads and parses a valid YAML file."""
        from src.core.config.loader import load_yaml

        config_file = tmp_path / "test.yaml"
        config_file.write_text("key: value\ncount: 42\n", encoding="utf-8")

        result = load_yaml(config_file)
        assert result == {"key": "value", "count": 42}

    def test_loads_nested_yaml(self, tmp_path):
        """Handles nested YAML structures."""
        from src.core.config.loader import load_yaml

        config_file = tmp_path / "nested.yaml"
        config_file.write_text(
            "parent:\n  child: value\n  list:\n    - a\n    - b\n",
            encoding="utf-8",
        )

        result = load_yaml(config_file)
        assert result["parent"]["child"] == "value"
        assert result["parent"]["list"] == ["a", "b"]

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """Empty YAML file returns empty dict, not None."""
        from src.core.config.loader import load_yaml

        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")

        result = load_yaml(config_file)
        assert result == {}

    def test_file_not_found_raises(self):
        """Missing file raises FileNotFoundError by default."""
        from src.core.config.loader import load_yaml

        with pytest.raises(FileNotFoundError):
            load_yaml("/nonexistent/config.yaml")

    def test_file_not_found_returns_default(self):
        """Missing file returns default when raise_on_error=False."""
        from src.core.config.loader import load_yaml

        result = load_yaml(
            "/nonexistent/config.yaml", default={"fallback": True}, raise_on_error=False
        )
        assert result == {"fallback": True}

    def test_file_not_found_no_default_returns_empty_dict(self):
        """Missing file with no default returns empty dict."""
        from src.core.config.loader import load_yaml

        result = load_yaml("/nonexistent/config.yaml", raise_on_error=False)
        assert result == {}

    def test_invalid_yaml_raises(self, tmp_path):
        """Malformed YAML raises YAMLError by default."""
        from src.core.config.loader import load_yaml

        config_file = tmp_path / "bad.yaml"
        config_file.write_text("key: [invalid: yaml: here", encoding="utf-8")

        with pytest.raises(yaml.YAMLError):
            load_yaml(config_file)

    def test_invalid_yaml_returns_default(self, tmp_path):
        """Malformed YAML returns default when raise_on_error=False."""
        from src.core.config.loader import load_yaml

        config_file = tmp_path / "bad.yaml"
        config_file.write_text("key: [invalid: yaml: here", encoding="utf-8")

        result = load_yaml(config_file, default={"safe": True}, raise_on_error=False)
        assert result == {"safe": True}

    def test_utf8_content(self, tmp_path):
        """Handles UTF-8 content (accented characters, etc.)."""
        from src.core.config.loader import load_yaml

        config_file = tmp_path / "utf8.yaml"
        config_file.write_text("name: caf\u00e9\nlabel: \u00a7 2.1\n", encoding="utf-8")

        result = load_yaml(config_file)
        assert result["name"] == "caf\u00e9"
        assert result["label"] == "\u00a7 2.1"

    def test_path_object_accepted(self, tmp_path):
        """Accepts both str and Path objects."""
        from src.core.config.loader import load_yaml

        config_file = tmp_path / "test.yaml"
        config_file.write_text("key: value\n", encoding="utf-8")

        # Path object
        result1 = load_yaml(config_file)
        # String path
        result2 = load_yaml(str(config_file))
        assert result1 == result2 == {"key": "value"}


# ============================================================================
# B. load_yaml_with_fallback
# ============================================================================


class TestLoadYamlWithFallback:
    """Tests for load_yaml_with_fallback()."""

    def test_loads_valid_file(self, tmp_path):
        """Returns parsed content when file exists."""
        from src.core.config.loader import load_yaml_with_fallback

        config_file = tmp_path / "test.yaml"
        config_file.write_text("setting: enabled\n", encoding="utf-8")

        result = load_yaml_with_fallback(config_file, fallback={"setting": "off"})
        assert result == {"setting": "enabled"}

    def test_returns_fallback_on_missing(self):
        """Returns fallback when file doesn't exist."""
        from src.core.config.loader import load_yaml_with_fallback

        result = load_yaml_with_fallback("/nonexistent/file.yaml", fallback={"default": True})
        assert result == {"default": True}

    def test_returns_fallback_on_parse_error(self, tmp_path):
        """Returns fallback when YAML is malformed."""
        from src.core.config.loader import load_yaml_with_fallback

        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{ bad yaml [", encoding="utf-8")

        result = load_yaml_with_fallback(config_file, fallback=[])
        assert result == []

    def test_never_raises(self, tmp_path):
        """Never raises exceptions regardless of input."""
        from src.core.config.loader import load_yaml_with_fallback

        # Missing file
        result1 = load_yaml_with_fallback("/nonexistent", fallback="safe")
        assert result1 == "safe"

        # Bad YAML (tab characters in wrong context cause parse errors)
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [invalid: yaml: here", encoding="utf-8")
        result2 = load_yaml_with_fallback(bad, fallback="safe")
        assert result2 == "safe"


# ============================================================================
# C. save_yaml
# ============================================================================


class TestSaveYaml:
    """Tests for save_yaml()."""

    def test_saves_valid_data(self, tmp_path):
        """Saves data to YAML file."""
        from src.core.config.loader import load_yaml, save_yaml

        config_file = tmp_path / "output.yaml"
        data = {"key": "value", "count": 42, "items": ["a", "b"]}

        assert save_yaml(config_file, data) is True
        assert config_file.exists()

        # Verify round-trip
        loaded = load_yaml(config_file)
        assert loaded == data

    def test_creates_parent_directories(self, tmp_path):
        """Creates parent directories if they don't exist."""
        from src.core.config.loader import save_yaml

        config_file = tmp_path / "sub" / "dir" / "config.yaml"
        assert save_yaml(config_file, {"test": True}) is True
        assert config_file.exists()

    def test_returns_false_on_error(self):
        """Returns False when save fails (e.g., permission error)."""
        from src.core.config.loader import save_yaml

        # Attempt to write to root (should fail on Windows)
        result = save_yaml(Path("Z:\\nonexistent\\impossible\\config.yaml"), {})
        assert result is False

    def test_overwrites_existing(self, tmp_path):
        """Overwrites existing file."""
        from src.core.config.loader import load_yaml, save_yaml

        config_file = tmp_path / "overwrite.yaml"
        save_yaml(config_file, {"old": True})
        save_yaml(config_file, {"new": True})

        result = load_yaml(config_file)
        assert result == {"new": True}

    def test_preserves_unicode(self, tmp_path):
        """Saves and loads Unicode content correctly."""
        from src.core.config.loader import load_yaml, save_yaml

        config_file = tmp_path / "unicode.yaml"
        data = {"name": "caf\u00e9", "symbol": "\u00a7"}
        save_yaml(config_file, data)

        loaded = load_yaml(config_file)
        assert loaded["name"] == "caf\u00e9"
        assert loaded["symbol"] == "\u00a7"
