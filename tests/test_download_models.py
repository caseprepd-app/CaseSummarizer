"""
Tests for the model download script (scripts/download_models.py).

Covers:
- Function signatures and structure
- Path resolution logic
- Skip-when-exists behavior
- Error handling (mocked network)
- NLTK zip extraction
- Tesseract copy logic
- Poppler download logic
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def download_module():
    """Import download_models from scripts/."""
    script_path = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(script_path))
    try:
        # Force reimport if cached
        if "download_models" in sys.modules:
            del sys.modules["download_models"]
        import download_models

        yield download_models
    finally:
        sys.path.pop(0)
        if "download_models" in sys.modules:
            del sys.modules["download_models"]


# ============================================================================
# A. Module Structure
# ============================================================================


class TestModuleStructure:
    """Tests for script structure and constants."""

    def test_has_all_download_functions(self, download_module):
        """Script exports all required download functions."""
        assert callable(download_module.download_spacy_models)
        assert callable(download_module.download_nltk_data)
        assert callable(download_module.download_huggingface_models)
        assert callable(download_module.download_tesseract)
        assert callable(download_module.download_poppler)
        assert callable(download_module.main)

    def test_spacy_models_list(self, download_module):
        """SPACY_MODELS contains all required models."""
        required = {"en_core_web_lg", "en_core_web_sm", "en_ner_bc5cdr_md"}
        assert required == set(download_module.SPACY_MODELS)

    def test_nltk_corpora_list(self, download_module):
        """NLTK_CORPORA contains all required corpora."""
        required = {"words", "wordnet", "omw-1.4", "stopwords"}
        assert required == set(download_module.NLTK_CORPORA)

    def test_tiktoken_cache_dir_defined(self, download_module):
        """TIKTOKEN_CACHE_DIR is defined for offline bundling."""
        assert hasattr(download_module, "TIKTOKEN_CACHE_DIR")
        assert download_module.TIKTOKEN_CACHE_DIR == download_module.MODELS_DIR / "tiktoken_cache"

    def test_hf_models_complete(self, download_module):
        """HF_MODELS contains all required HuggingFace models."""
        repo_ids = {entry[0] for entry in download_module.HF_MODELS}
        expected = {
            "nomic-ai/nomic-embed-text-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "Alibaba-NLP/gte-reranker-modernbert-base",
        }
        assert expected == repo_ids


# ============================================================================
# B. Path Resolution
# ============================================================================


class TestPathResolution:
    """Tests for path constants."""

    def test_project_root(self, download_module):
        """PROJECT_ROOT is correct."""
        expected = Path(__file__).parent.parent
        assert expected == download_module.PROJECT_ROOT

    def test_models_dir(self, download_module):
        """MODELS_DIR is under PROJECT_ROOT."""
        assert download_module.MODELS_DIR == download_module.PROJECT_ROOT / "models"

    def test_spacy_dir(self, download_module):
        """SPACY_DIR is under MODELS_DIR."""
        assert download_module.SPACY_DIR == download_module.MODELS_DIR / "spacy"

    def test_nltk_dir(self, download_module):
        """NLTK_DIR is under MODELS_DIR."""
        assert download_module.NLTK_DIR == download_module.MODELS_DIR / "nltk_data"

    def test_tesseract_dir(self, download_module):
        """TESSERACT_DIR is under MODELS_DIR."""
        assert download_module.TESSERACT_DIR == download_module.MODELS_DIR / "tesseract"

    def test_poppler_dir(self, download_module):
        """POPPLER_DIR is under MODELS_DIR."""
        assert download_module.POPPLER_DIR == download_module.MODELS_DIR / "poppler"


# ============================================================================
# C. Download Functions — Skip-When-Exists
# ============================================================================


class TestSkipWhenExists:
    """Tests that download functions skip when target already exists."""

    def test_spacy_skips_existing(self, download_module, tmp_path):
        """download_spacy_models skips models that already exist."""
        # Create a fake existing model
        with patch.object(download_module, "SPACY_DIR", tmp_path):
            for model in download_module.SPACY_MODELS:
                (tmp_path / model).mkdir()

            results = download_module.download_spacy_models()
            assert all(results.values())

    def test_tesseract_skips_existing(self, download_module, tmp_path):
        """download_tesseract skips when tesseract.exe exists."""
        with patch.object(download_module, "TESSERACT_DIR", tmp_path):
            (tmp_path / "tesseract.exe").touch()
            results = download_module.download_tesseract()
            assert results["tesseract"] is True

    def test_poppler_skips_existing(self, download_module, tmp_path):
        """download_poppler skips when pdftoppm.exe exists."""
        with patch.object(download_module, "POPPLER_DIR", tmp_path):
            (tmp_path / "pdftoppm.exe").touch()
            results = download_module.download_poppler()
            assert results["poppler"] is True


# ============================================================================
# D. Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in download functions."""

    def test_tesseract_missing_system_install(self, download_module, tmp_path):
        """download_tesseract reports failure when system install missing."""
        with (
            patch.object(download_module, "TESSERACT_DIR", tmp_path / "tesseract"),
            patch.object(download_module, "TESSERACT_SYSTEM_DIR", tmp_path / "nonexistent"),
        ):
            results = download_module.download_tesseract()
            assert results["tesseract"] is False

    def test_poppler_network_failure(self, download_module, tmp_path):
        """download_poppler handles network failure gracefully."""
        with (
            patch.object(download_module, "POPPLER_DIR", tmp_path / "poppler"),
            patch("builtins.__import__", side_effect=Exception("network error")),
        ):
            # If urlopen fails, should return False
            results = download_module.download_poppler()
            assert results["poppler"] is False

    def test_hf_download_failure(self, download_module):
        """download_huggingface_models handles individual failures."""
        mock_download = MagicMock(side_effect=Exception("download failed"))
        with (
            patch.object(download_module, "MODELS_DIR", Path("/fake/models")),
            patch("huggingface_hub.snapshot_download", mock_download),
        ):
            # Should return False for each model, not crash
            results = download_module.download_huggingface_models()
            assert not any(results.values())


# ============================================================================
# E. NLTK Zip Extraction
# ============================================================================


class TestNLTKZipExtraction:
    """Tests for NLTK zip extraction logic."""

    def test_zip_extraction_logic(self, download_module, tmp_path):
        """Zips without extracted directories get extracted."""
        import zipfile

        with patch.object(download_module, "NLTK_DIR", tmp_path):
            # Create a test zip
            corpora_dir = tmp_path / "corpora"
            corpora_dir.mkdir()
            zip_path = corpora_dir / "testcorpus.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("testcorpus/data.txt", "test data")

            # Mock nltk.download to do nothing
            with patch("nltk.download"):
                download_module.download_nltk_data()

            # The zip should have been extracted
            assert (corpora_dir / "testcorpus").is_dir()
            assert (corpora_dir / "testcorpus" / "data.txt").exists()


# ============================================================================
# F. Tesseract Copy Logic
# ============================================================================


class TestTesseractCopy:
    """Tests for Tesseract binary copying."""

    def test_copies_exe_and_dlls(self, download_module, tmp_path):
        """download_tesseract copies .exe and .dll files."""
        source_dir = tmp_path / "system_tess"
        source_dir.mkdir()
        (source_dir / "tesseract.exe").write_bytes(b"fake")
        (source_dir / "leptonica.dll").write_bytes(b"fake")
        tessdata = source_dir / "tessdata"
        tessdata.mkdir()
        (tessdata / "eng.traineddata").write_bytes(b"fake")

        target_dir = tmp_path / "bundled_tess"

        with (
            patch.object(download_module, "TESSERACT_DIR", target_dir),
            patch.object(download_module, "TESSERACT_SYSTEM_DIR", source_dir),
        ):
            results = download_module.download_tesseract()

        assert results["tesseract"] is True
        assert (target_dir / "tesseract.exe").exists()
        assert (target_dir / "leptonica.dll").exists()
        assert (target_dir / "tessdata" / "eng.traineddata").exists()


# ============================================================================
# G. Embedding Model Ignore Patterns
# ============================================================================


class TestIgnorePatterns:
    """Tests for HuggingFace model download ignore patterns."""

    def test_nomic_skips_onnx(self, download_module):
        """nomic-embed-text-v1.5 skips ONNX variants."""
        for repo_id, _, ignore in download_module.HF_MODELS:
            if repo_id == "nomic-ai/nomic-embed-text-v1.5":
                assert ignore is not None
                assert any("onnx" in p for p in ignore)
                return
        pytest.fail("nomic-embed-text-v1.5 not found")

    def test_minilm_skips_large_files(self, download_module):
        """all-MiniLM-L6-v2 skips ONNX, OpenVINO, and large binaries."""
        for repo_id, _, ignore in download_module.HF_MODELS:
            if repo_id == "sentence-transformers/all-MiniLM-L6-v2":
                assert ignore is not None
                assert any("onnx" in p for p in ignore)
                assert any("pytorch_model.bin" in p for p in ignore)
                return
        pytest.fail("all-MiniLM-L6-v2 not found")
