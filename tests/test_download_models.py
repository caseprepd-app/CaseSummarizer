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

    # test_minilm_removed removed — KeyBERT deprecated, test no longer relevant


# ============================================================================
# H. Nomic Custom Code Bundling (download_models.py)
# ============================================================================


class TestNomicCustomCodeBundling:
    """Tests for _bundle_nomic_custom_code in download_models.py."""

    def test_function_exists(self, download_module):
        """_bundle_nomic_custom_code is defined and callable."""
        assert callable(download_module._bundle_nomic_custom_code)

    def test_copies_custom_code_files(self, download_module, tmp_path):
        """Downloads and copies configuration and modeling .py files."""
        # Create a config.json with remote auto_map
        config = {
            "auto_map": {
                "AutoConfig": "nomic-ai/nomic-bert-2048--configuration_hf_nomic_bert.NomicBertConfig",
                "AutoModel": "nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel",
            }
        }
        (tmp_path / "config.json").write_text(__import__("json").dumps(config), encoding="utf-8")

        fake_py = tmp_path / "cached_file.py"
        fake_py.write_text("# fake", encoding="utf-8")

        with patch("huggingface_hub.hf_hub_download", return_value=str(fake_py)):
            download_module._bundle_nomic_custom_code(tmp_path)

        assert (tmp_path / "configuration_hf_nomic_bert.py").exists()
        assert (tmp_path / "modeling_hf_nomic_bert.py").exists()

    def test_patches_auto_map_to_local(self, download_module, tmp_path):
        """Rewrites auto_map entries to remove remote repo prefix."""
        import json

        config = {
            "auto_map": {
                "AutoConfig": "nomic-ai/nomic-bert-2048--configuration_hf_nomic_bert.NomicBertConfig",
                "AutoModel": "nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel",
            }
        }
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

        fake_py = tmp_path / "cached_file.py"
        fake_py.write_text("# fake", encoding="utf-8")

        with patch("huggingface_hub.hf_hub_download", return_value=str(fake_py)):
            download_module._bundle_nomic_custom_code(tmp_path)

        patched = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        for value in patched["auto_map"].values():
            assert "--" not in value, f"Remote prefix not stripped: {value}"
        assert patched["auto_map"]["AutoConfig"] == "configuration_hf_nomic_bert.NomicBertConfig"
        assert patched["auto_map"]["AutoModel"] == "modeling_hf_nomic_bert.NomicBertModel"

    def test_skips_patch_when_no_auto_map(self, download_module, tmp_path):
        """No crash when config.json has no auto_map key."""
        import json

        config = {"model_type": "bert"}
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

        fake_py = tmp_path / "cached_file.py"
        fake_py.write_text("# fake", encoding="utf-8")

        with patch("huggingface_hub.hf_hub_download", return_value=str(fake_py)):
            download_module._bundle_nomic_custom_code(tmp_path)

        patched = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert "auto_map" not in patched

    def test_leaves_already_local_auto_map_unchanged(self, download_module, tmp_path):
        """auto_map entries without '--' are left as-is."""
        import json

        config = {
            "auto_map": {
                "AutoConfig": "configuration_hf_nomic_bert.NomicBertConfig",
            }
        }
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

        fake_py = tmp_path / "cached_file.py"
        fake_py.write_text("# fake", encoding="utf-8")

        with patch("huggingface_hub.hf_hub_download", return_value=str(fake_py)):
            download_module._bundle_nomic_custom_code(tmp_path)

        patched = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert patched["auto_map"]["AutoConfig"] == "configuration_hf_nomic_bert.NomicBertConfig"

    def test_called_after_nomic_download(self, download_module):
        """_bundle_nomic_custom_code is called after nomic model downloads."""
        call_log = []

        def mock_snapshot(**kwargs):
            """Track snapshot_download calls."""
            call_log.append(("snapshot", kwargs["repo_id"]))

        def mock_bundle(target_dir):
            """Track _bundle_nomic_custom_code calls."""
            call_log.append(("bundle", str(target_dir)))

        with (
            patch.object(download_module, "MODELS_DIR", Path("/fake/models")),
            patch("huggingface_hub.snapshot_download", side_effect=mock_snapshot),
            patch.object(download_module, "_bundle_nomic_custom_code", side_effect=mock_bundle),
        ):
            download_module.download_huggingface_models()

        # Bundle should be called right after nomic snapshot_download
        snapshot_calls = [c for c in call_log if c[0] == "snapshot"]
        bundle_calls = [c for c in call_log if c[0] == "bundle"]
        assert len(bundle_calls) == 1
        assert any("nomic" in c[1] for c in snapshot_calls)

    def test_not_called_for_reranker(self, download_module):
        """_bundle_nomic_custom_code is NOT called for the reranker model."""
        bundle_calls = []

        with (
            patch.object(download_module, "MODELS_DIR", Path("/fake/models")),
            patch("huggingface_hub.snapshot_download"),
            patch.object(
                download_module,
                "_bundle_nomic_custom_code",
                side_effect=lambda d: bundle_calls.append(d),
            ),
        ):
            download_module.download_huggingface_models()

        # Should be called exactly once (for nomic only)
        assert len(bundle_calls) == 1


# ============================================================================
# I. Nomic Custom Code Bundling (download_embedding_model.py)
# ============================================================================


@pytest.fixture
def embedding_download_module():
    """Import download_embedding_model from scripts/."""
    script_path = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(script_path))
    try:
        if "download_embedding_model" in sys.modules:
            del sys.modules["download_embedding_model"]
        import download_embedding_model

        yield download_embedding_model
    finally:
        sys.path.pop(0)
        if "download_embedding_model" in sys.modules:
            del sys.modules["download_embedding_model"]


class TestEmbeddingDownloadCustomCode:
    """Tests for _bundle_custom_code in download_embedding_model.py."""

    def test_function_exists(self, embedding_download_module):
        """_bundle_custom_code is defined and callable."""
        assert callable(embedding_download_module._bundle_custom_code)

    def test_patches_auto_map(self, embedding_download_module, tmp_path):
        """Rewrites auto_map to local module references."""
        import json

        config = {
            "auto_map": {
                "AutoConfig": "nomic-ai/nomic-bert-2048--configuration_hf_nomic_bert.NomicBertConfig",
            }
        }
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

        fake_py = tmp_path / "cached.py"
        fake_py.write_text("# fake", encoding="utf-8")

        with patch("huggingface_hub.hf_hub_download", return_value=str(fake_py)):
            embedding_download_module._bundle_custom_code(tmp_path)

        patched = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert patched["auto_map"]["AutoConfig"] == "configuration_hf_nomic_bert.NomicBertConfig"

    def test_main_calls_bundle_on_success(self, embedding_download_module):
        """main() calls _bundle_custom_code after successful download."""
        with (
            patch.object(embedding_download_module, "download_model", return_value=True),
            patch.object(embedding_download_module, "verify_model", return_value=True),
            patch.object(embedding_download_module, "_bundle_custom_code") as mock_bundle,
            patch.object(embedding_download_module, "models_dir", Path("/fake")),
        ):
            embedding_download_module.main()

        mock_bundle.assert_called_once()

    def test_main_skips_bundle_on_failure(self, embedding_download_module):
        """main() does NOT call _bundle_custom_code when download fails."""
        with (
            patch.object(embedding_download_module, "download_model", return_value=False),
            patch.object(embedding_download_module, "verify_model", return_value=False),
            patch.object(embedding_download_module, "_bundle_custom_code") as mock_bundle,
            pytest.raises(SystemExit),
        ):
            embedding_download_module.main()

        mock_bundle.assert_not_called()
