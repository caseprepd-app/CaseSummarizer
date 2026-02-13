"""
Tests for installer path resolution and frozen-mode compatibility.

Covers:
- BUNDLED_BASE_DIR resolution (dev vs frozen)
- BUNDLED_MODELS_DIR and model path constants
- TESSDATA_PREFIX env var registration
- PyInstaller spec file requirements
- All model paths are relative (no hardcoded user paths)
- Download script path resolution
"""

import os
import sys
from pathlib import Path

import pytest

# ============================================================================
# A. BUNDLED_BASE_DIR Resolution
# ============================================================================


class TestBundledBaseDirResolution:
    """Tests that BUNDLED_BASE_DIR resolves correctly in dev and frozen mode."""

    def test_dev_mode_uses_parent_parent(self):
        """In dev mode, BUNDLED_BASE_DIR = Path(__file__).parent.parent."""
        from src.config import BUNDLED_BASE_DIR

        # We're in dev mode (not frozen)
        assert not getattr(sys, "frozen", False)
        expected = Path(__file__).parent.parent
        assert expected == BUNDLED_BASE_DIR

    def test_bundled_base_dir_is_project_root(self):
        """BUNDLED_BASE_DIR should be the project root (contains src/, config/)."""
        from src.config import BUNDLED_BASE_DIR

        assert (BUNDLED_BASE_DIR / "src").is_dir()
        assert (BUNDLED_BASE_DIR / "config").is_dir()

    def test_config_uses_sys_frozen_check(self):
        """config.py source code checks sys.frozen for BUNDLED_BASE_DIR."""
        import src.config as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert 'getattr(sys, "frozen", False)' in source
        assert "sys._MEIPASS" in source

    def test_bundled_models_dir_uses_bundled_base(self):
        """BUNDLED_MODELS_DIR is derived from BUNDLED_BASE_DIR, not hardcoded."""
        from src.config import BUNDLED_BASE_DIR, BUNDLED_MODELS_DIR

        assert BUNDLED_MODELS_DIR == BUNDLED_BASE_DIR / "models"


# ============================================================================
# B. Model Path Constants
# ============================================================================


class TestModelPathConstants:
    """Tests that all model paths are relative and properly structured."""

    def test_all_model_paths_under_bundled_models_dir(self):
        """All model paths are children of BUNDLED_MODELS_DIR."""
        from src.config import (
            BUNDLED_MODELS_DIR,
            COREF_MODEL_LOCAL_PATH,
            EMBEDDING_MODEL_LOCAL_PATH,
            GLINER_MODEL_LOCAL_PATH,
            HALLUCINATION_MODEL_LOCAL_PATH,
            RERANKER_MODEL_LOCAL_PATH,
            SEMANTIC_CHUNKER_MODEL_LOCAL_PATH,
            SPACY_MODELS_DIR,
        )

        for path in [
            HALLUCINATION_MODEL_LOCAL_PATH,
            GLINER_MODEL_LOCAL_PATH,
            EMBEDDING_MODEL_LOCAL_PATH,
            SEMANTIC_CHUNKER_MODEL_LOCAL_PATH,
            RERANKER_MODEL_LOCAL_PATH,
            COREF_MODEL_LOCAL_PATH,
            SPACY_MODELS_DIR,
        ]:
            # All should be under BUNDLED_MODELS_DIR
            try:
                path.relative_to(BUNDLED_MODELS_DIR)
            except ValueError:
                pytest.fail(f"{path} is not under BUNDLED_MODELS_DIR ({BUNDLED_MODELS_DIR})")

    def test_no_hardcoded_user_paths_in_config(self):
        """config.py source has no hardcoded user-specific paths."""
        import src.config as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "C:\\Users\\" not in source
        assert "C:/Users/" not in source
        assert "noahc" not in source
        assert "Dropbox" not in source

    def test_tesseract_paths_under_models(self):
        """Tesseract paths are under BUNDLED_MODELS_DIR."""
        from src.config import BUNDLED_MODELS_DIR, TESSERACT_BUNDLED_DIR, TESSERACT_BUNDLED_EXE

        assert TESSERACT_BUNDLED_DIR == BUNDLED_MODELS_DIR / "tesseract"
        assert TESSERACT_BUNDLED_EXE == TESSERACT_BUNDLED_DIR / "tesseract.exe"

    def test_poppler_path_under_models(self):
        """Poppler path is under BUNDLED_MODELS_DIR."""
        from src.config import BUNDLED_MODELS_DIR, POPPLER_BUNDLED_DIR

        assert POPPLER_BUNDLED_DIR == BUNDLED_MODELS_DIR / "poppler"


# ============================================================================
# C. TESSDATA_PREFIX Registration
# ============================================================================


class TestTessdataPrefix:
    """Tests for TESSDATA_PREFIX environment variable."""

    def test_tessdata_prefix_set_when_dir_exists(self):
        """TESSDATA_PREFIX env var points to bundled tessdata."""
        from src.config import TESSERACT_BUNDLED_DIR

        if TESSERACT_BUNDLED_DIR.exists():
            assert "TESSDATA_PREFIX" in os.environ
            assert os.environ["TESSDATA_PREFIX"] == str(TESSERACT_BUNDLED_DIR / "tessdata")

    def test_tessdata_prefix_path_valid(self):
        """TESSDATA_PREFIX path is syntactically valid."""
        prefix = os.environ.get("TESSDATA_PREFIX", "")
        if prefix:
            assert Path(prefix).is_absolute() or Path(prefix).parts


# ============================================================================
# D. NLTK Path Registration
# ============================================================================


class TestNLTKPathRegistration:
    """Tests for NLTK data path registration."""

    def test_nltk_data_dir_under_models(self):
        """NLTK_DATA_DIR is under BUNDLED_MODELS_DIR."""
        from src.config import BUNDLED_MODELS_DIR, NLTK_DATA_DIR

        assert NLTK_DATA_DIR == BUNDLED_MODELS_DIR / "nltk_data"

    def test_nltk_path_registered(self):
        """When NLTK data exists, it's in nltk.data.path."""
        import nltk

        from src.config import NLTK_DATA_DIR

        if NLTK_DATA_DIR.exists():
            assert str(NLTK_DATA_DIR) in nltk.data.path


# ============================================================================
# E. PyInstaller Spec Requirements
# ============================================================================


class TestSpecFileRequirements:
    """Tests for caseprepd.spec correctness."""

    @pytest.fixture
    def spec_content(self):
        """Read the spec file content."""
        spec_path = Path(__file__).parent.parent / "caseprepd.spec"
        if not spec_path.exists():
            pytest.skip("caseprepd.spec not found")
        return spec_path.read_text(encoding="utf-8")

    def test_spec_bundles_config(self, spec_content):
        """Spec bundles config/ directory."""
        assert '"config"' in spec_content or "'config'" in spec_content

    def test_spec_bundles_models(self, spec_content):
        """Spec includes model bundling loop."""
        assert "models" in spec_content

    def test_spec_skips_hf_cache(self, spec_content):
        """Spec skips .hf_cache directory."""
        assert ".hf_cache" in spec_content

    def test_spec_clears_nltk_path(self, spec_content):
        """Spec clears nltk.data.path before Analysis."""
        assert "nltk.data.path.clear()" in spec_content

    def test_spec_uses_onedir(self, spec_content):
        """Spec uses COLLECT (onedir mode) not onefile."""
        assert "COLLECT" in spec_content

    def test_spec_has_icon(self, spec_content):
        """Spec references the app icon."""
        assert "icon.ico" in spec_content


# ============================================================================
# F. Download Script Path Resolution
# ============================================================================


class TestDownloadScriptPaths:
    """Tests that download_models.py uses relative paths."""

    def test_project_root_is_relative(self):
        """PROJECT_ROOT is computed from __file__, not hardcoded."""
        script_path = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(script_path))
        try:
            import download_models

            # PROJECT_ROOT should be parent of scripts/
            assert Path(__file__).parent.parent == download_models.PROJECT_ROOT
        finally:
            sys.path.pop(0)

    def test_models_dir_under_project_root(self):
        """MODELS_DIR is under PROJECT_ROOT."""
        script_path = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(script_path))
        try:
            import download_models

            assert download_models.MODELS_DIR == download_models.PROJECT_ROOT / "models"
        finally:
            sys.path.pop(0)

    def test_no_hardcoded_paths_in_script(self):
        """Script source has no hardcoded user-specific paths."""
        script_path = Path(__file__).parent.parent / "scripts" / "download_models.py"
        source = script_path.read_text(encoding="utf-8")
        assert "noahc" not in source
        assert "Dropbox" not in source
        # C:\Program Files is OK (Tesseract system install path)


# ============================================================================
# G. Model Fallback Paths
# ============================================================================


class TestModelFallbacks:
    """Tests that models have proper fallback behavior."""

    def test_hallucination_local_only_flag(self):
        """HALLUCINATION_LOCAL_ONLY is True iff bundled model exists."""
        from src.config import HALLUCINATION_LOCAL_ONLY, HALLUCINATION_MODEL_LOCAL_PATH

        assert HALLUCINATION_MODEL_LOCAL_PATH.exists() == HALLUCINATION_LOCAL_ONLY

    def test_hf_cache_dir_under_models(self):
        """HF_CACHE_DIR is under BUNDLED_MODELS_DIR (not system default)."""
        from src.config import BUNDLED_MODELS_DIR, HF_CACHE_DIR

        assert HF_CACHE_DIR == BUNDLED_MODELS_DIR / ".hf_cache"

    def test_appdata_dir_uses_env_variable(self):
        """APPDATA_DIR uses %APPDATA% environment variable."""
        from src.config import APPDATA_DIR

        appdata = os.environ.get("APPDATA", "")
        if appdata:
            assert str(APPDATA_DIR).startswith(appdata)


# ============================================================================
# H. No Absolute Paths in Tests
# ============================================================================


class TestNoAbsolutePathsInTests:
    """Meta-tests: verify test files don't use hardcoded absolute paths."""

    def test_no_hardcoded_paths_in_test_files(self):
        """No test file references specific user directories (except this meta-test)."""
        test_dir = Path(__file__).parent
        bad_patterns = [
            "C:\\\\Users\\\\noahc",
            "C:/Users/noahc",
            "/home/noahc",
            "Dropbox/Not Work",
        ]
        problems = []
        for test_file in test_dir.glob("test_*.py"):
            # Skip this file — it contains the patterns as test data
            if test_file.name == "test_installer_paths.py":
                continue
            content = test_file.read_text(encoding="utf-8")
            for pattern in bad_patterns:
                if pattern in content:
                    problems.append(f"{test_file.name} contains '{pattern}'")
        assert not problems, "Hardcoded paths found:\n" + "\n".join(problems)
