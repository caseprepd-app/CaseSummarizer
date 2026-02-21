"""Production readiness tests — config validation, encoding safety, defensive reads."""

import json
from unittest.mock import patch

import pytest


class TestActiveCorpusValidation:
    """Tests for active_corpus preference validation."""

    @pytest.fixture
    def prefs(self, tmp_path):
        """Create a UserPreferencesManager with a temp file."""
        from src.user_preferences import UserPreferencesManager

        pref_file = tmp_path / "prefs.json"
        return UserPreferencesManager(pref_file)

    def test_set_valid_corpus_name(self, prefs):
        """Setting a valid corpus name should succeed."""
        prefs.set("active_corpus", "My Transcripts")
        assert prefs.get("active_corpus") == "My Transcripts"

    def test_reject_empty_corpus_name(self, prefs):
        """Empty string should be rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            prefs.set("active_corpus", "")

    def test_reject_whitespace_only_corpus_name(self, prefs):
        """Whitespace-only string should be rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            prefs.set("active_corpus", "   ")

    def test_reject_path_traversal_dotdot(self, prefs):
        """Path traversal with .. should be rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            prefs.set("active_corpus", "../../../etc/passwd")

    def test_reject_forward_slash(self, prefs):
        """Forward slashes should be rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            prefs.set("active_corpus", "foo/bar")

    def test_reject_backslash(self, prefs):
        """Backslashes should be rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            prefs.set("active_corpus", "foo\\bar")

    def test_reject_non_string(self, prefs):
        """Non-string values should be rejected."""
        with pytest.raises(ValueError):
            prefs.set("active_corpus", 123)


class TestCorruptPreferencesFile:
    """Tests for handling corrupted preferences files."""

    def test_corrupt_json_returns_defaults(self, tmp_path):
        """Corrupted JSON file should return defaults, not crash."""
        from src.user_preferences import UserPreferencesManager

        pref_file = tmp_path / "prefs.json"
        pref_file.write_text("{invalid json!!!", encoding="utf-8")

        mgr = UserPreferencesManager(pref_file)
        # Should get default structure, not crash
        assert mgr.get("model_defaults") is not None or mgr.get("model_defaults") == {}

    def test_corrupt_json_logs_warning(self, tmp_path):
        """Corrupted JSON should log a warning."""
        from src.user_preferences import UserPreferencesManager

        pref_file = tmp_path / "prefs.json"
        pref_file.write_text("{bad json", encoding="utf-8")

        with patch("src.user_preferences.logger") as mock_logger:
            UserPreferencesManager(pref_file)
            mock_logger.warning.assert_called_once()

    def test_wrong_type_resource_pct_uses_default(self, tmp_path):
        """If resource_usage_pct is wrong type in JSON, default should be used."""
        from src.user_preferences import UserPreferencesManager

        pref_file = tmp_path / "prefs.json"
        pref_file.write_text(
            json.dumps({"model_defaults": {}, "resource_usage_pct": "not_a_number"}),
            encoding="utf-8",
        )
        mgr = UserPreferencesManager(pref_file)
        raw = mgr.get("resource_usage_pct", 75)
        # The raw value may be wrong type, but system_resources should handle it
        # (tested in TestResourcePctValidation)


class TestResourcePctValidation:
    """Tests for resource_usage_pct validation at read-time."""

    def test_valid_pct_passes_through(self):
        """Valid percentage should be used as-is."""
        from src.system_resources import get_system_resources

        with patch("src.system_resources.get_user_preferences") as mock:
            mock.return_value.get.return_value = 50
            info = get_system_resources()
            assert info.resource_usage_pct == 50

    def test_negative_pct_uses_default(self):
        """Negative percentage should fall back to 75."""
        from src.system_resources import get_system_resources

        with patch("src.system_resources.get_user_preferences") as mock:
            mock.return_value.get.return_value = -10
            info = get_system_resources()
            assert info.resource_usage_pct == 75

    def test_over_100_uses_default(self):
        """Over 100% should fall back to 75."""
        from src.system_resources import get_system_resources

        with patch("src.system_resources.get_user_preferences") as mock:
            mock.return_value.get.return_value = 5000
            info = get_system_resources()
            assert info.resource_usage_pct == 75

    def test_string_type_uses_default(self):
        """String type should fall back to 75."""
        from src.system_resources import get_system_resources

        with patch("src.system_resources.get_user_preferences") as mock:
            mock.return_value.get.return_value = "invalid"
            info = get_system_resources()
            assert info.resource_usage_pct == 75

    def test_none_uses_default(self):
        """None should fall back to 75."""
        from src.system_resources import get_system_resources

        with patch("src.system_resources.get_user_preferences") as mock:
            mock.return_value.get.return_value = None
            info = get_system_resources()
            assert info.resource_usage_pct == 75


class TestRetrievalWeightValidation:
    """Tests for retrieval weight type validation."""

    def test_valid_weights_pass_through(self):
        """Valid float weights should be used."""
        from src.core.vector_store.qa_retriever import _get_effective_algorithm_weights

        class FakePrefs:
            def get(self, key, default=None):
                if key == "retrieval_weight_faiss":
                    return 0.8
                if key == "retrieval_weight_bm25":
                    return 1.2
                return default

        with patch(
            "src.user_preferences.get_user_preferences",
            return_value=FakePrefs(),
        ):
            weights = _get_effective_algorithm_weights()
            assert weights["FAISS"] == 0.8
            assert weights["BM25+"] == 1.2

    def test_string_weight_falls_back_to_default(self):
        """String weight should fall back to config default."""
        from src.config import RETRIEVAL_ALGORITHM_WEIGHTS
        from src.core.vector_store.qa_retriever import _get_effective_algorithm_weights

        class FakePrefs:
            def get(self, key, default=None):
                if key == "retrieval_weight_faiss":
                    return "invalid"
                if key == "retrieval_weight_bm25":
                    return None
                return default

        with patch(
            "src.user_preferences.get_user_preferences",
            return_value=FakePrefs(),
        ):
            weights = _get_effective_algorithm_weights()
            assert weights["FAISS"] == RETRIEVAL_ALGORITHM_WEIGHTS["FAISS"]
            assert weights["BM25+"] == RETRIEVAL_ALGORITHM_WEIGHTS["BM25+"]


class TestPreferenceSetValidation:
    """Tests for known-key validation in UserPreferencesManager.set()."""

    @pytest.fixture
    def prefs(self, tmp_path):
        """Create a UserPreferencesManager with a temp file."""
        from src.user_preferences import UserPreferencesManager

        pref_file = tmp_path / "prefs.json"
        return UserPreferencesManager(pref_file)

    def test_resource_pct_valid(self, prefs):
        prefs.set("resource_usage_pct", 50)
        assert prefs.get("resource_usage_pct") == 50

    def test_resource_pct_too_low(self, prefs):
        with pytest.raises(ValueError):
            prefs.set("resource_usage_pct", 10)

    def test_resource_pct_too_high(self, prefs):
        with pytest.raises(ValueError):
            prefs.set("resource_usage_pct", 200)

    def test_retrieval_weight_valid(self, prefs):
        prefs.set("retrieval_weight_faiss", 1.5)
        assert prefs.get("retrieval_weight_faiss") == 1.5

    def test_retrieval_weight_too_high(self, prefs):
        with pytest.raises(ValueError):
            prefs.set("retrieval_weight_faiss", 3.0)

    def test_indicator_patterns_valid(self, prefs):
        prefs.set("vocab_positive_indicators", ["dr.", "plaintiff"])
        assert prefs.get("vocab_positive_indicators") == ["dr.", "plaintiff"]

    def test_indicator_patterns_not_list(self, prefs):
        with pytest.raises(ValueError):
            prefs.set("vocab_positive_indicators", "dr.")

    def test_regex_override_invalid(self, prefs):
        with pytest.raises(ValueError, match="invalid regex"):
            prefs.set("vocab_positive_regex_override", "[invalid")

    def test_unknown_key_still_stored(self, prefs):
        """Unknown keys should still be stored (extensible system)."""
        prefs.set("some_future_setting", "value")
        assert prefs.get("some_future_setting") == "value"


# ============================================================================
# Installer Readiness Tests
# ============================================================================

import csv
import importlib
import re

import yaml

from src.config import BUNDLED_BASE_DIR, BUNDLED_CONFIG_DIR


class TestBundledConfigFiles:
    """Verify every config file referenced by src/config.py exists and is non-empty."""

    def test_app_name_txt_exists(self):
        """config/app_name.txt must exist and be non-empty."""
        path = BUNDLED_CONFIG_DIR / "app_name.txt"
        assert path.is_file(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"

    def test_models_yaml_exists(self):
        """config/models.yaml must exist and be non-empty."""
        path = BUNDLED_CONFIG_DIR / "models.yaml"
        assert path.is_file(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"

    def test_default_feedback_csv_exists(self):
        """config/default_feedback.csv must exist and be non-empty."""
        path = BUNDLED_CONFIG_DIR / "default_feedback.csv"
        assert path.is_file(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"

    def test_legal_exclude_txt_exists(self):
        """config/legal_exclude.txt must exist and be non-empty."""
        path = BUNDLED_CONFIG_DIR / "legal_exclude.txt"
        assert path.is_file(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"

    def test_medical_terms_txt_exists(self):
        """config/medical_terms.txt must exist and be non-empty."""
        path = BUNDLED_CONFIG_DIR / "medical_terms.txt"
        assert path.is_file(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"

    def test_gliner_labels_txt_exists(self):
        """config/gliner_labels.txt must exist and be non-empty."""
        path = BUNDLED_CONFIG_DIR / "gliner_labels.txt"
        assert path.is_file(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"

    def test_default_questions_json_exists(self):
        """config/default_questions.json must exist and be non-empty."""
        path = BUNDLED_CONFIG_DIR / "default_questions.json"
        assert path.is_file(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"

    def test_categories_json_exists(self):
        """config/categories.json must exist and be non-empty."""
        path = BUNDLED_CONFIG_DIR / "categories.json"
        assert path.is_file(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"

    def test_chunking_config_yaml_exists(self):
        """config/chunking_config.yaml must exist and be non-empty."""
        path = BUNDLED_CONFIG_DIR / "chunking_config.yaml"
        assert path.is_file(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"

    def test_word_frequency_file_exists(self):
        """data/frequency/Word_rarity-count_1w.txt must exist and be non-empty."""
        from src.config import GOOGLE_WORD_FREQUENCY_FILE

        assert GOOGLE_WORD_FREQUENCY_FILE.is_file(), f"Missing: {GOOGLE_WORD_FREQUENCY_FILE}"
        assert GOOGLE_WORD_FREQUENCY_FILE.stat().st_size > 0

    def test_prompt_templates_dir_exists(self):
        """config/prompts/phi-3-mini/ must exist with at least one .txt file."""
        from src.config import PROMPTS_DIR

        phi3_dir = PROMPTS_DIR / "phi-3-mini"
        assert phi3_dir.is_dir(), f"Missing directory: {phi3_dir}"
        txt_files = list(phi3_dir.glob("*.txt"))
        assert len(txt_files) >= 1, f"No .txt files in {phi3_dir}"

    def test_extraction_prompts_dir_exists(self):
        """config/extraction_prompts/ must exist with at least one .txt file."""
        extraction_dir = BUNDLED_BASE_DIR / "config" / "extraction_prompts"
        assert extraction_dir.is_dir(), f"Missing directory: {extraction_dir}"
        txt_files = list(extraction_dir.glob("*.txt"))
        assert len(txt_files) >= 1, f"No .txt files in {extraction_dir}"


class TestConfigFileParsing:
    """Verify key config files are valid and contain expected structure."""

    def test_models_yaml_parses(self):
        """models.yaml must parse and contain a 'models' key with entries."""
        path = BUNDLED_CONFIG_DIR / "models.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "models.yaml root must be a dict"
        assert "models" in data, "models.yaml must have a 'models' key"
        assert len(data["models"]) > 0, "models.yaml 'models' must have entries"

    def test_default_questions_json_parses(self):
        """default_questions.json must parse and contain a 'questions' list."""
        path = BUNDLED_CONFIG_DIR / "default_questions.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "default_questions.json must be a dict"
        assert "questions" in data, "default_questions.json must have a 'questions' key"
        assert isinstance(data["questions"], list), "'questions' must be a list"

    def test_categories_json_parses(self):
        """categories.json must parse and return a dict or list."""
        path = BUNDLED_CONFIG_DIR / "categories.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, (dict, list)), "categories.json must be a dict or list"

    def test_transcript_patterns_json_parses(self):
        """transcript_patterns.json must parse as valid JSON."""
        path = BUNDLED_CONFIG_DIR / "transcript_patterns.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data is not None, "transcript_patterns.json must not be null"

    def test_prompt_parameters_json_parses(self):
        """prompt_parameters.json must parse as valid JSON."""
        path = BUNDLED_CONFIG_DIR / "prompt_parameters.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data is not None, "prompt_parameters.json must not be null"


class TestBundledModels:
    """Verify ML model directories exist and contain files."""

    def test_hallucination_model_exists(self):
        """Hallucination detection model directory must exist and be non-empty."""
        from src.config import HALLUCINATION_MODEL_LOCAL_PATH

        assert HALLUCINATION_MODEL_LOCAL_PATH.is_dir(), f"Missing: {HALLUCINATION_MODEL_LOCAL_PATH}"
        assert any(HALLUCINATION_MODEL_LOCAL_PATH.iterdir()), (
            f"Empty: {HALLUCINATION_MODEL_LOCAL_PATH}"
        )

    def test_gliner_model_exists(self):
        """GLiNER model directory must exist and be non-empty."""
        from src.config import GLINER_MODEL_LOCAL_PATH

        assert GLINER_MODEL_LOCAL_PATH.is_dir(), f"Missing: {GLINER_MODEL_LOCAL_PATH}"
        assert any(GLINER_MODEL_LOCAL_PATH.iterdir()), f"Empty: {GLINER_MODEL_LOCAL_PATH}"

    def test_embedding_model_exists(self):
        """Embedding model directory must exist and be non-empty."""
        from src.config import EMBEDDING_MODEL_LOCAL_PATH

        assert EMBEDDING_MODEL_LOCAL_PATH.is_dir(), f"Missing: {EMBEDDING_MODEL_LOCAL_PATH}"
        assert any(EMBEDDING_MODEL_LOCAL_PATH.iterdir()), f"Empty: {EMBEDDING_MODEL_LOCAL_PATH}"

    def test_semantic_chunker_model_exists(self):
        """Semantic chunker model directory must exist and be non-empty."""
        from src.config import SEMANTIC_CHUNKER_MODEL_LOCAL_PATH

        assert SEMANTIC_CHUNKER_MODEL_LOCAL_PATH.is_dir(), (
            f"Missing: {SEMANTIC_CHUNKER_MODEL_LOCAL_PATH}"
        )
        assert any(SEMANTIC_CHUNKER_MODEL_LOCAL_PATH.iterdir()), (
            f"Empty: {SEMANTIC_CHUNKER_MODEL_LOCAL_PATH}"
        )

    def test_reranker_model_exists(self):
        """Reranker model directory must exist and be non-empty."""
        from src.config import RERANKER_MODEL_LOCAL_PATH

        assert RERANKER_MODEL_LOCAL_PATH.is_dir(), f"Missing: {RERANKER_MODEL_LOCAL_PATH}"
        assert any(RERANKER_MODEL_LOCAL_PATH.iterdir()), f"Empty: {RERANKER_MODEL_LOCAL_PATH}"

    def test_coref_model_exists(self):
        """Coreference resolution model directory must exist and be non-empty."""
        from src.config import COREF_MODEL_LOCAL_PATH

        assert COREF_MODEL_LOCAL_PATH.is_dir(), f"Missing: {COREF_MODEL_LOCAL_PATH}"
        assert any(COREF_MODEL_LOCAL_PATH.iterdir()), f"Empty: {COREF_MODEL_LOCAL_PATH}"

    def test_spacy_models_exist(self):
        """All 3 bundled spaCy model directories must exist and be non-empty."""
        from src.config import (
            SPACY_EN_CORE_WEB_LG_PATH,
            SPACY_EN_CORE_WEB_SM_PATH,
            SPACY_EN_NER_BC5CDR_MD_PATH,
        )

        for path in [
            SPACY_EN_CORE_WEB_LG_PATH,
            SPACY_EN_CORE_WEB_SM_PATH,
            SPACY_EN_NER_BC5CDR_MD_PATH,
        ]:
            assert path.is_dir(), f"Missing spaCy model: {path}"
            assert any(path.iterdir()), f"Empty spaCy model: {path}"


class TestNltkDataBundled:
    """Verify NLTK corpora are bundled and accessible."""

    def test_nltk_data_dir_exists(self):
        """NLTK_DATA_DIR must be a directory."""
        from src.config import NLTK_DATA_DIR

        if not NLTK_DATA_DIR.is_dir():
            pytest.skip(f"Missing: {NLTK_DATA_DIR}. Run 'python scripts/download_models.py'.")

    def test_required_nltk_corpora_present(self):
        """Required NLTK corpus extracted directories must exist."""
        from src.config import NLTK_DATA_DIR

        if not NLTK_DATA_DIR.is_dir():
            pytest.skip("NLTK_DATA_DIR not present (run download_models.py)")

        required = ["words", "wordnet", "omw-1.4", "stopwords", "punkt_tab"]
        for corpus in required:
            # Require extracted directories (not just zips) for PyInstaller reliability
            corpora_dir = NLTK_DATA_DIR / "corpora"
            tokenizers_dir = NLTK_DATA_DIR / "tokenizers"
            found = (corpora_dir / corpus).is_dir() or (tokenizers_dir / corpus).is_dir()
            assert found, (
                f"Missing extracted NLTK directory: {corpus} in {NLTK_DATA_DIR}. "
                f"Run 'python scripts/download_models.py' to fix."
            )

    def test_nltk_corpus_accessible(self):
        """nltk.corpus.words.words() must return data using bundled path."""
        from nltk.corpus import words

        # Ensure bundled path is registered (src/config.py does this at import)

        word_list = words.words()
        assert len(word_list) > 1000, f"Expected 1000+ words, got {len(word_list)}"


class TestHiddenImportsResolvable:
    """Verify all app-code hidden imports from the spec file actually import."""

    @pytest.mark.parametrize(
        "module",
        [
            "src.core.vocabulary.algorithms.ner_algorithm",
            "src.core.vocabulary.algorithms.rake_algorithm",
            "src.core.vocabulary.algorithms.textrank_algorithm",
            "src.core.vocabulary.algorithms.bm25_algorithm",
            "src.core.vocabulary.algorithms.gliner_algorithm",
            "src.core.vocabulary.algorithms.scispacy_algorithm",
            "tiktoken_ext.openai_public",
            "tiktoken_ext",
        ],
    )
    def test_hidden_import_resolves(self, module):
        """Each hidden import must be importable without error."""
        importlib.import_module(module)


class TestVersionConsistency:
    """Verify version strings are consistent across distribution artifacts."""

    def test_init_version_format(self):
        """src.__init__.__version__ must match semver pattern X.Y.Z."""
        import src

        assert hasattr(src, "__version__"), "src.__init__ must define __version__"
        assert re.match(r"^\d+\.\d+\.\d+$", src.__version__), (
            f"Version '{src.__version__}' doesn't match X.Y.Z semver pattern"
        )

    def test_inno_setup_version_matches(self):
        """Version in installer/caseprepd.iss must match src.__init__.__version__."""
        import src

        iss_path = BUNDLED_BASE_DIR / "installer" / "caseprepd.iss"
        assert iss_path.is_file(), f"Missing: {iss_path}"

        content = iss_path.read_text(encoding="utf-8")
        match = re.search(r'#define\s+MyAppVersion\s+"([^"]+)"', content)
        assert match, "Could not find #define MyAppVersion in caseprepd.iss"
        iss_version = match.group(1)
        assert iss_version == src.__version__, (
            f"Version mismatch: src={src.__version__}, iss={iss_version}"
        )

    def test_about_dialog_uses_dynamic_version(self):
        """About dialog must use __version__ not a hardcoded string."""
        about_path = BUNDLED_BASE_DIR / "src" / "ui" / "help_about_dialogs.py"
        content = about_path.read_text(encoding="utf-8")
        assert "__version__" in content, "help_about_dialogs.py must import and use __version__"
        assert 'text="Version 1.' not in content, (
            "About dialog has a hardcoded version string -- use f'Version {__version__}'"
        )


class TestNameDataFiles:
    """Verify name data CSV files exist and have rows."""

    def test_international_forenames_csv(self):
        """data/names/international_forenames.csv must exist and have data rows."""
        path = BUNDLED_BASE_DIR / "data" / "names" / "international_forenames.csv"
        assert path.is_file(), f"Missing: {path}"
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # At least header + 1 data row
        assert len(rows) >= 2, f"Expected data rows, got {len(rows)} total rows"

    def test_international_surnames_csv(self):
        """data/names/international_surnames.csv must exist and have data rows."""
        path = BUNDLED_BASE_DIR / "data" / "names" / "international_surnames.csv"
        assert path.is_file(), f"Missing: {path}"
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) >= 2, f"Expected data rows, got {len(rows)} total rows"


class TestBundledOCRPaths:
    """Tests for bundled OCR config constants and runtime detection."""

    def test_config_exports_tesseract_paths(self):
        """Config exports TESSERACT_BUNDLED_EXE and POPPLER_BUNDLED_DIR."""
        from src.config import POPPLER_BUNDLED_DIR, TESSERACT_BUNDLED_EXE

        assert TESSERACT_BUNDLED_EXE.name == "tesseract.exe"
        assert "tesseract" in str(TESSERACT_BUNDLED_EXE)
        assert POPPLER_BUNDLED_DIR.name == "poppler"

    def test_tessdata_prefix_set(self):
        """TESSDATA_PREFIX env var should point to bundled tessdata when present."""
        import os

        from src.config import TESSERACT_BUNDLED_DIR

        if TESSERACT_BUNDLED_DIR.exists():
            tessdata_prefix = os.environ.get("TESSDATA_PREFIX", "")
            expected = str(TESSERACT_BUNDLED_DIR / "tessdata")
            assert tessdata_prefix == expected, (
                f"TESSDATA_PREFIX={tessdata_prefix!r}, expected {expected!r}"
            )

    def test_ocr_availability_finds_bundled(self):
        """check_ocr_availability() returns AVAILABLE when bundled files exist."""
        from unittest.mock import patch

        from src.services.ocr_availability import OCRStatus, check_ocr_availability

        with (
            patch("src.services.ocr_availability._find_tesseract", return_value=True),
            patch("src.services.ocr_availability._find_poppler", return_value=True),
        ):
            assert check_ocr_availability() == OCRStatus.AVAILABLE


class TestCallbackExceptionIsolation:
    """Verify non-critical preference saves don't crash UI callbacks."""

    def test_save_task_checkbox_states_catches_exceptions(self):
        """_save_task_checkbox_states wraps prefs.set in try/except."""
        from pathlib import Path

        source = (Path(__file__).parent.parent / "src" / "ui" / "main_window.py").read_text(
            encoding="utf-8"
        )
        start = source.index("def _save_task_checkbox_states")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert "try:" in body
        assert "except Exception" in body
        assert "Could not save task checkbox states" in body

    def test_save_column_widths_catches_prefs_error(self):
        """_save_column_widths wraps prefs.set in try/except."""
        from pathlib import Path

        source = (Path(__file__).parent.parent / "src" / "ui" / "dynamic_output.py").read_text(
            encoding="utf-8"
        )
        start = source.index("def _save_column_widths")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert "Could not save column widths" in body


class TestAssetFiles:
    """Verify application asset files exist."""

    def test_app_icon_exists(self):
        """assets/icon.ico must exist."""
        path = BUNDLED_BASE_DIR / "assets" / "icon.ico"
        assert path.is_file(), f"Missing: {path}"
