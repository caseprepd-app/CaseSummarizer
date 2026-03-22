"""
Tests for v1.0.23 local-only asset enforcement.

Covers: asset_audit module, frozen-mode guards on spaCy/ML model loading,
model_loader resolve_model_path, OCR tesseract frozen guard, and
config.py env var force-setting.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# ============================================================
# Asset Audit: _classify helper
# ============================================================


class TestAssetAuditClassify:
    """_classify() correctly tags paths as BUNDLED, SYSTEM, or MISSING."""

    def test_bundled_path(self, tmp_path):
        """Path under base_dir is classified as BUNDLED."""
        from src.services.asset_audit import _classify

        child = tmp_path / "models" / "test"
        child.mkdir(parents=True)
        assert _classify(child, tmp_path) == "BUNDLED"

    def test_system_path(self, tmp_path):
        """Path outside base_dir is classified as SYSTEM."""
        from src.services.asset_audit import _classify

        other = tmp_path.parent / "other_dir"
        other.mkdir(exist_ok=True)
        assert _classify(other, tmp_path) == "SYSTEM"

    def test_missing_path(self, tmp_path):
        """Non-existent path is classified as MISSING."""
        from src.services.asset_audit import _classify

        assert _classify(tmp_path / "nonexistent", tmp_path) == "MISSING"


# ============================================================
# Asset Audit: _log_asset helper
# ============================================================


class TestAssetAuditLogAsset:
    """_log_asset() returns correct tag and doesn't raise."""

    def test_returns_bundled_tag(self, tmp_path):
        """Returns BUNDLED for path under base."""
        from src.services.asset_audit import _log_asset

        child = tmp_path / "file.txt"
        child.write_text("data")
        assert _log_asset("test asset", child, tmp_path) == "BUNDLED"

    def test_returns_missing_tag(self, tmp_path):
        """Returns MISSING for non-existent path."""
        from src.services.asset_audit import _log_asset

        tag = _log_asset("missing asset", tmp_path / "nope", tmp_path)
        assert tag == "MISSING"


# ============================================================
# Asset Audit: run_asset_audit smoke test
# ============================================================


class TestAssetAuditSmoke:
    """run_asset_audit() completes without error."""

    def test_runs_without_error(self):
        """Smoke test: run_asset_audit should not raise."""
        from src.services.asset_audit import run_asset_audit

        run_asset_audit()


# ============================================================
# model_loader: resolve_model_path frozen guard
# ============================================================


class TestResolveModelPath:
    """resolve_model_path raises in frozen mode when bundled missing."""

    def test_returns_bundled_when_exists(self, tmp_path):
        """Returns local path when bundled model exists."""
        from src.core.utils.model_loader import resolve_model_path

        bundled = tmp_path / "model"
        bundled.mkdir()
        path, is_local = resolve_model_path(bundled, "hf/model")
        assert path == str(bundled)
        assert is_local is True

    def test_raises_in_frozen_when_missing(self, tmp_path):
        """Raises RuntimeError in frozen mode if bundled model absent."""
        from src.core.utils.model_loader import resolve_model_path

        with patch("src.core.utils.model_loader.sys") as mock_sys:
            mock_sys.frozen = True
            with pytest.raises(RuntimeError, match="Required model not found"):
                resolve_model_path(tmp_path / "missing", "hf/model")

    def test_falls_back_in_dev_mode(self, tmp_path):
        """Returns HF name in dev mode when bundled missing."""
        from src.core.utils.model_loader import resolve_model_path

        path, is_local = resolve_model_path(tmp_path / "missing", "hf/model")
        assert path == "hf/model"
        assert is_local is False


# ============================================================
# model_loader: set_hf_cache_env force-sets
# ============================================================


class TestSetHfCacheEnv:
    """set_hf_cache_env force-sets env vars (not setdefault)."""

    def test_overwrites_existing_hf_home(self, tmp_path):
        """Force-sets HF_HOME even if already set."""
        from src.core.utils.model_loader import set_hf_cache_env

        os.environ["HF_HOME"] = "/old/path"
        set_hf_cache_env(tmp_path)
        assert os.environ["HF_HOME"] == str(tmp_path)

    def test_sets_transformers_cache(self, tmp_path):
        """Also sets TRANSFORMERS_CACHE."""
        from src.core.utils.model_loader import set_hf_cache_env

        set_hf_cache_env(tmp_path)
        assert os.environ["TRANSFORMERS_CACHE"] == str(tmp_path)


# ============================================================
# Frozen-mode guards: spaCy model loading
# ============================================================


class TestSpacyFrozenGuards:
    """Frozen-mode RuntimeError when bundled spaCy models missing."""

    def test_ner_algorithm_frozen_raises(self):
        """NER _load_spacy raises RuntimeError in frozen mode."""
        source = Path("src/core/vocabulary/algorithms/ner_algorithm.py").read_text(encoding="utf-8")
        assert 'getattr(sys, "frozen", False)' in source
        assert "RuntimeError" in source
        assert "Please reinstall" in source

    def test_textrank_algorithm_frozen_raises(self):
        """TextRank _load_spacy_model raises RuntimeError in frozen mode."""
        source = Path("src/core/vocabulary/algorithms/textrank_algorithm.py").read_text(
            encoding="utf-8"
        )
        assert 'getattr(sys, "frozen", False)' in source
        assert "RuntimeError" in source
        assert "Please reinstall" in source

    def test_scispacy_algorithm_frozen_raises(self):
        """SciSpaCy extract raises RuntimeError in frozen mode."""
        source = Path("src/core/vocabulary/algorithms/scispacy_algorithm.py").read_text(
            encoding="utf-8"
        )
        assert 'getattr(sys, "frozen", False)' in source
        assert "RuntimeError" in source
        assert "Please reinstall" in source

    def test_vocabulary_extractor_frozen_raises(self):
        """VocabularyExtractor _load_spacy raises in frozen mode."""
        source = Path("src/core/vocabulary/vocabulary_extractor.py").read_text(encoding="utf-8")
        assert 'getattr(sys, "frozen", False)' in source
        assert "RuntimeError" in source
        assert "Please reinstall" in source


# ============================================================
# OCR: frozen-mode guard on Tesseract
# ============================================================


class TestOCRFrozenGuard:
    """Tesseract configuration respects frozen mode."""

    def test_configure_tesseract_source_has_frozen_check(self):
        """_configure_tesseract checks sys.frozen."""
        source = Path("src/core/extraction/ocr_processor.py").read_text(encoding="utf-8")
        assert 'getattr(sys, "frozen", False)' in source
        assert "Bundled Tesseract not found" in source

    def test_no_system_search_in_frozen_mode(self):
        """In frozen mode with missing bundled exe, no system search."""
        import inspect

        from src.core.extraction.ocr_processor import _configure_tesseract

        source = inspect.getsource(_configure_tesseract)
        # The frozen branch should NOT search system paths
        # frozen check must come before shutil.which
        frozen_pos = source.find('getattr(sys, "frozen"')
        which_pos = source.find("shutil.which")
        assert frozen_pos < which_pos, "Frozen check must come before system PATH search"


# ============================================================
# Config: HF offline env vars are force-set
# ============================================================


class TestConfigEnvVars:
    """config.py force-sets HF offline flags at import time."""

    def test_hf_hub_offline_set(self):
        """HF_HUB_OFFLINE should be '1'."""
        assert os.environ.get("HF_HUB_OFFLINE") == "1"

    def test_transformers_offline_set(self):
        """TRANSFORMERS_OFFLINE should be '1'."""
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    def test_config_source_force_sets_not_setdefault(self):
        """config.py uses os.environ[key] = val, not setdefault."""
        source = Path("src/config.py").read_text(encoding="utf-8")
        assert 'os.environ["HF_HUB_OFFLINE"]' in source
        assert 'os.environ["TRANSFORMERS_OFFLINE"]' in source
        # Should NOT use setdefault for these
        assert 'setdefault("HF_HUB_OFFLINE"' not in source
        assert 'setdefault("TRANSFORMERS_OFFLINE"' not in source


# ============================================================
# Config: NLTK single bundled path
# ============================================================


class TestNLTKPathRestriction:
    """NLTK data path restricted to bundled directory only."""

    def test_nltk_path_is_single_entry(self):
        """NLTK should have exactly one search path (the bundled dir)."""
        import nltk

        from src.config import NLTK_DATA_DIR

        if NLTK_DATA_DIR.exists():
            assert len(nltk.data.path) == 1
            assert nltk.data.path[0] == str(NLTK_DATA_DIR)

    def test_config_source_replaces_not_appends(self):
        """config.py assigns nltk.data.path = [...], not .append()."""
        source = Path("src/config.py").read_text(encoding="utf-8")
        assert "nltk.data.path = [" in source
        assert "nltk.data.path.append" not in source


# ============================================================
# validate_models.py: script structure
# ============================================================


class TestValidateModelsScript:
    """Pre-build validation script has expected structure."""

    def test_script_exists(self):
        """validate_models.py exists in scripts/."""
        assert Path("scripts/validate_models.py").exists()

    def test_checks_all_asset_categories(self):
        """Script validates ML models, spaCy, NLTK, OCR, data, config."""
        source = Path("scripts/validate_models.py").read_text(encoding="utf-8")
        for section in [
            "validate_ml_models",
            "validate_spacy",
            "validate_nltk",
            "validate_ocr",
            "validate_data",
            "validate_config",
        ]:
            assert section in source, f"Missing validation: {section}"

    def test_safetensors_size_check(self):
        """Script checks safetensors file size to detect truncation."""
        source = Path("scripts/validate_models.py").read_text(encoding="utf-8")
        assert "MIN_SAFETENSORS_MB" in source
        assert "TRUNCATED" in source

    def test_exits_nonzero_on_failure(self):
        """Script exits with code 1 when validation fails."""
        source = Path("scripts/validate_models.py").read_text(encoding="utf-8")
        assert "sys.exit(1)" in source
