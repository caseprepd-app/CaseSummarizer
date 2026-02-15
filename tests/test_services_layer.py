"""Tests for the services layer: VocabularyService, AIService, ExportService, etc."""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# VocabularyService
# ---------------------------------------------------------------------------


class TestVocabularyServiceInit:
    """VocabularyService lazy-loads VocabularyExtractor."""

    def test_extractor_is_none_initially(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        assert svc._extractor is None

    def test_extractor_property_creates_instance(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        with patch("src.services.vocabulary_service.VocabularyExtractor") as mock_cls:
            mock_cls.return_value = MagicMock()
            ext = svc.extractor
            assert ext is not None
            mock_cls.assert_called_once()

    def test_extractor_property_cached(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        with patch("src.services.vocabulary_service.VocabularyExtractor") as mock_cls:
            mock_cls.return_value = MagicMock()
            ext1 = svc.extractor
            ext2 = svc.extractor
            assert ext1 is ext2
            mock_cls.assert_called_once()


class TestVocabularyServiceExtract:
    """VocabularyService.extract_vocabulary delegates to extractor."""

    def test_empty_text_returns_empty_list(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        assert svc.extract_vocabulary("") == []
        assert svc.extract_vocabulary("   ") == []

    def test_delegates_to_extractor(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        mock_ext = MagicMock()
        mock_ext.extract.return_value = [{"Term": "plaintiff"}]
        svc._extractor = mock_ext

        result = svc.extract_vocabulary("The plaintiff filed a motion.")
        mock_ext.extract.assert_called_once_with("The plaintiff filed a motion.")
        assert result == [{"Term": "plaintiff"}]

    def test_per_document_empty_returns_empty(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        assert svc.extract_vocabulary_per_document([]) == []

    def test_per_document_delegates(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        mock_ext = MagicMock()
        mock_ext.extract_per_document.return_value = [{"Term": "defendant"}]
        svc._extractor = mock_ext

        docs = [{"text": "doc text", "doc_id": "d1", "confidence": 95}]
        result = svc.extract_vocabulary_per_document(docs)
        assert result == [{"Term": "defendant"}]


class TestVocabularyServiceFeedback:
    """VocabularyService.record_feedback passes through to extractor."""

    def test_positive_feedback(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        mock_ext = MagicMock()
        svc._extractor = mock_ext

        svc.record_feedback("plaintiff", True, "doc1")
        mock_ext.record_feedback.assert_called_once_with("plaintiff", True, "doc1")

    def test_negative_feedback(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        mock_ext = MagicMock()
        svc._extractor = mock_ext

        svc.record_feedback("gibberish", False)
        mock_ext.record_feedback.assert_called_once_with("gibberish", False, None)


class TestVocabularyServiceAlgorithms:
    """Algorithm info and toggle."""

    def test_get_algorithm_info(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        mock_algo = MagicMock()
        mock_algo.name = "NER"
        mock_algo.enabled = True
        mock_algo.description = "Named entity recognition"

        mock_ext = MagicMock()
        mock_ext.algorithms = [mock_algo]
        svc._extractor = mock_ext

        info = svc.get_algorithm_info()
        assert len(info) == 1
        assert info[0]["name"] == "NER"
        assert info[0]["enabled"] is True

    def test_set_algorithm_enabled_found(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        mock_algo = MagicMock()
        mock_algo.name = "RAKE"
        mock_algo.enabled = True

        mock_ext = MagicMock()
        mock_ext.algorithms = [mock_algo]
        svc._extractor = mock_ext

        assert svc.set_algorithm_enabled("RAKE", False) is True
        assert mock_algo.enabled is False

    def test_set_algorithm_enabled_not_found(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        mock_ext = MagicMock()
        mock_ext.algorithms = []
        svc._extractor = mock_ext

        assert svc.set_algorithm_enabled("NONEXISTENT", True) is False


class TestVocabularyServiceCsvExport:
    """VocabularyService.export_to_csv writes CSV with BOM."""

    def test_export_empty_returns_false(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        assert svc.export_to_csv([], "output.csv") is False

    def test_export_writes_csv(self, tmp_path):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        data = [{"Term": "plaintiff", "Score": 0.95}, {"Term": "defendant", "Score": 0.88}]
        out = tmp_path / "vocab.csv"

        assert svc.export_to_csv(data, str(out)) is True
        assert out.exists()

        # Verify BOM for Excel compatibility
        raw = out.read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf"

        # Verify content
        with open(out, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["Term"] == "plaintiff"

    def test_export_handles_write_error(self, tmp_path):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        data = [{"Term": "test"}]
        # Write to nonexistent directory
        bad_path = str(tmp_path / "no" / "such" / "dir" / "out.csv")
        assert svc.export_to_csv(data, bad_path) is False


class TestVocabularyServiceCorpus:
    """Corpus-related methods."""

    def test_get_corpus_doc_count_missing_dir(self, tmp_path):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        assert svc.get_corpus_doc_count(str(tmp_path / "nonexistent")) == 0

    def test_get_corpus_doc_count_counts_supported_files(self, tmp_path):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        (tmp_path / "doc1.pdf").write_text("pdf", encoding="utf-8")
        (tmp_path / "doc2.txt").write_text("txt", encoding="utf-8")
        (tmp_path / "doc3.jpg").write_text("jpg", encoding="utf-8")  # unsupported
        (tmp_path / "doc4_preprocessed.pdf").write_text("skip", encoding="utf-8")  # preprocessed

        count = svc.get_corpus_doc_count(str(tmp_path))
        # Should count pdf + txt but not jpg or preprocessed
        assert count >= 2

    def test_get_max_corpus_docs_returns_int(self):
        from src.services.vocabulary_service import VocabularyService

        svc = VocabularyService()
        result = svc.get_max_corpus_docs()
        assert isinstance(result, int)
        assert result > 0


# ---------------------------------------------------------------------------
# AIService
# ---------------------------------------------------------------------------


class TestAIServiceSingleton:
    """AIService is a singleton."""

    def test_singleton_returns_same_instance(self):
        import src.services.ai_service as mod

        mod._ai_service_instance = None  # Reset singleton

        from src.services.ai_service import AIService

        s1 = AIService()
        s2 = AIService()
        assert s1 is s2

        # Clean up
        mod._ai_service_instance = None

    def test_initialized_flag_prevents_re_init(self):
        import src.services.ai_service as mod

        mod._ai_service_instance = None

        from src.services.ai_service import AIService

        svc = AIService()
        assert svc._initialized is True
        svc._ollama_manager = "sentinel"

        svc2 = AIService()
        assert svc2._ollama_manager == "sentinel"  # Not reset

        mod._ai_service_instance = None


class TestAIServiceOllama:
    """AIService Ollama integration."""

    def test_get_ollama_manager_lazy_loads(self):
        import src.services.ai_service as mod

        mod._ai_service_instance = None

        from src.services.ai_service import AIService

        svc = AIService()
        assert svc._ollama_manager is None

        with patch("src.core.ai.OllamaModelManager") as mock_cls:
            mock_cls.return_value = MagicMock()
            mgr = svc.get_ollama_manager()
            assert mgr is not None
            mock_cls.assert_called_once()

        mod._ai_service_instance = None

    def test_check_ollama_connection(self):
        import src.services.ai_service as mod

        mod._ai_service_instance = None

        from src.services.ai_service import AIService

        svc = AIService()
        mock_mgr = MagicMock()
        mock_mgr.check_connection.return_value = True
        svc._ollama_manager = mock_mgr

        assert svc.check_ollama_connection() is True

        mod._ai_service_instance = None

    def test_get_available_models(self):
        import src.services.ai_service as mod

        mod._ai_service_instance = None

        from src.services.ai_service import AIService

        svc = AIService()
        mock_mgr = MagicMock()
        mock_mgr.get_available_models.return_value = [{"name": "gemma3:1b"}]
        svc._ollama_manager = mock_mgr

        models = svc.get_available_models()
        assert len(models) == 1
        assert models[0]["name"] == "gemma3:1b"

        mod._ai_service_instance = None


class TestAIServiceGPU:
    """AIService GPU detection pass-through."""

    def test_has_dedicated_gpu_delegates(self):
        import src.services.ai_service as mod

        mod._ai_service_instance = None

        from src.services.ai_service import AIService

        svc = AIService()
        with patch("src.core.utils.gpu_detector.has_dedicated_gpu", return_value=False):
            assert svc.has_dedicated_gpu() is False

        mod._ai_service_instance = None

    def test_get_gpu_status_text_returns_string(self):
        import src.services.ai_service as mod

        mod._ai_service_instance = None

        from src.services.ai_service import AIService

        svc = AIService()
        with patch("src.core.utils.gpu_detector.get_gpu_status_text", return_value="No GPU"):
            assert svc.get_gpu_status_text() == "No GPU"

        mod._ai_service_instance = None

    def test_get_optimal_context_size_returns_int(self):
        import src.services.ai_service as mod

        mod._ai_service_instance = None

        from src.services.ai_service import AIService

        svc = AIService()
        with patch("src.core.utils.gpu_detector.get_optimal_context_size", return_value=4096):
            assert svc.get_optimal_context_size() == 4096

        mod._ai_service_instance = None


# ---------------------------------------------------------------------------
# ExportService
# ---------------------------------------------------------------------------


class TestExportServiceRunExport:
    """The _run_export helper handles success, failure, and auto-open."""

    def test_run_export_success(self):
        from src.services.export_service import _run_export

        result = _run_export("test export", "/tmp/test.html", "test", lambda: True)
        assert result is True

    def test_run_export_returns_false_on_exception(self):
        from src.services.export_service import _run_export

        def fail():
            raise RuntimeError("disk full")

        result = _run_export("test", "/tmp/x.html", "test", fail)
        assert result is False

    def test_run_export_none_treated_as_success(self):
        from src.services.export_service import _run_export

        result = _run_export("test", "/tmp/x.html", "test", lambda: None)
        assert result is True


class TestExportServiceVocabulary:
    """ExportService vocabulary export methods."""

    def test_export_vocabulary_to_txt(self, tmp_path):
        from src.services.export_service import ExportService

        svc = ExportService()
        data = [{"Term": "plaintiff"}, {"Term": "defendant"}]
        out = tmp_path / "vocab.txt"

        with patch("src.services.export_service.export_vocabulary_txt") as mock_fn:
            mock_fn.return_value = True
            result = svc.export_vocabulary_to_txt(data, str(out))
            assert result is True
            mock_fn.assert_called_once()

    def test_export_vocabulary_to_word(self, tmp_path):
        from src.services.export_service import ExportService

        svc = ExportService()
        data = [{"Term": "plaintiff"}]
        out = tmp_path / "vocab.docx"

        with patch("src.services.export_service.WordDocumentBuilder") as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder_cls.return_value = mock_builder
            with patch("src.services.export_service.export_vocabulary"):
                result = svc.export_vocabulary_to_word(data, str(out))
                assert result is True
                mock_builder.save.assert_called_once_with(str(out))

    def test_export_vocabulary_to_pdf(self, tmp_path):
        from src.services.export_service import ExportService

        svc = ExportService()
        data = [{"Term": "plaintiff"}]
        out = tmp_path / "vocab.pdf"

        with patch("src.services.export_service.PdfDocumentBuilder") as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder_cls.return_value = mock_builder
            with patch("src.services.export_service.export_vocabulary"):
                result = svc.export_vocabulary_to_pdf(data, str(out))
                assert result is True
                mock_builder.save.assert_called_once_with(str(out))


class TestExportServiceQA:
    """ExportService Q&A export methods."""

    def test_export_qa_to_word(self, tmp_path):
        from src.services.export_service import ExportService

        svc = ExportService()
        results = [MagicMock()]
        out = tmp_path / "qa.docx"

        with patch("src.services.export_service.WordDocumentBuilder") as mock_cls:
            mock_builder = MagicMock()
            mock_cls.return_value = mock_builder
            with patch("src.services.export_service.export_qa_results"):
                result = svc.export_qa_to_word(results, str(out))
                assert result is True

    def test_export_qa_to_html(self, tmp_path):
        from src.services.export_service import ExportService

        svc = ExportService()
        results = [MagicMock()]
        out = tmp_path / "qa.html"

        with patch("src.services.export_service.export_qa_html") as mock_fn:
            mock_fn.return_value = True
            result = svc.export_qa_to_html(results, str(out))
            assert result is True


class TestExportServiceCombined:
    """ExportService combined export methods."""

    def test_export_combined_html(self, tmp_path):
        from src.services.export_service import ExportService

        svc = ExportService()
        out = tmp_path / "combined.html"

        with patch(
            "src.core.export.combined_html_builder.build_combined_html",
            return_value="<html></html>",
        ):
            result = svc.export_combined_html(
                vocab_data=[{"Term": "a"}],
                qa_results=[],
                summary_text="Summary",
                file_path=str(out),
            )
            assert result is True
            assert out.exists()

    def test_export_combined_to_word(self, tmp_path):
        from src.services.export_service import ExportService

        svc = ExportService()
        out = tmp_path / "combined.docx"

        with patch("src.services.export_service.WordDocumentBuilder") as mock_cls:
            mock_builder = MagicMock()
            mock_cls.return_value = mock_builder
            with patch("src.services.export_service.export_combined"):
                result = svc.export_combined_to_word(
                    vocab_data=[{"Term": "a"}],
                    qa_results=[],
                    file_path=str(out),
                )
                assert result is True


class TestExportServiceSingleton:
    """get_export_service returns singleton."""

    def test_singleton(self):
        import src.services.export_service as mod

        mod._export_service = None

        from src.services.export_service import get_export_service

        s1 = get_export_service()
        s2 = get_export_service()
        assert s1 is s2

        mod._export_service = None


class TestAutoOpenFile:
    """_auto_open_file respects preferences."""

    def test_auto_open_disabled(self):
        from src.services.export_service import _auto_open_file

        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value = MagicMock()
            mock_prefs.return_value.get.return_value = False
            # Should not raise, should do nothing
            _auto_open_file("/tmp/test.html")

    @pytest.mark.skipif(
        __import__("sys").platform != "win32", reason="os.startfile only on Windows"
    )
    def test_auto_open_enabled_calls_startfile(self):
        from src.services.export_service import _auto_open_file

        with patch("src.user_preferences.get_user_preferences") as mock_prefs:
            mock_prefs.return_value = MagicMock()
            mock_prefs.return_value.get.return_value = True
            with patch("os.startfile") as mock_start:
                _auto_open_file("/tmp/test.html")
                mock_start.assert_called_once_with("/tmp/test.html")


# ---------------------------------------------------------------------------
# ModelIOService
# ---------------------------------------------------------------------------


class TestModelIOExportUserModel:
    """export_user_model copies .pkl file."""

    def test_export_missing_model_returns_false(self, tmp_path):
        from src.services.model_io_service import export_user_model

        with patch("src.services.model_io_service.VOCAB_MODEL_PATH", tmp_path / "nonexistent.pkl"):
            success, msg = export_user_model(tmp_path / "dest.pkl")
            assert success is False
            assert "No vocabulary model" in msg

    def test_export_existing_model(self, tmp_path):
        from src.services.model_io_service import export_user_model

        model = tmp_path / "model.pkl"
        model.write_bytes(b"model_data")

        dest = tmp_path / "exported.pkl"
        with patch("src.services.model_io_service.VOCAB_MODEL_PATH", model):
            success, msg = export_user_model(dest)
            assert success is True
            assert dest.exists()
            assert dest.read_bytes() == b"model_data"


class TestModelIOImportUserModel:
    """import_user_model validates and restores on failure."""

    def test_import_invalid_model_restores_backup(self, tmp_path):
        from src.services.model_io_service import import_user_model

        existing = tmp_path / "current.pkl"
        existing.write_bytes(b"current_model")

        bad_import = tmp_path / "bad.pkl"
        bad_import.write_bytes(b"bad_model")

        with patch("src.services.model_io_service.VOCAB_MODEL_PATH", existing):
            with patch("src.services.model_io_service.load_model") as mock_load:
                mock_load.return_value = (None, None, None, None, 0, 0, False)
                success, msg = import_user_model(bad_import)
                assert success is False
                assert "incompatible" in msg
                # Original should be restored
                assert existing.read_bytes() == b"current_model"

    def test_import_valid_model(self, tmp_path):
        from src.services.model_io_service import import_user_model

        model_path = tmp_path / "model.pkl"
        good_import = tmp_path / "good.pkl"
        good_import.write_bytes(b"good_data")

        with patch("src.services.model_io_service.VOCAB_MODEL_PATH", model_path):
            with patch("src.services.model_io_service.load_model") as mock_load:
                mock_load.return_value = (
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    100,
                    500,
                    True,
                )
                success, msg = import_user_model(good_import)
                assert success is True
                assert "500 samples" in msg


class TestModelIOFeedbackExport:
    """Feedback CSV export/import."""

    def test_export_no_feedback_returns_false(self, tmp_path):
        from src.services.model_io_service import export_user_feedback

        mock_mgr = MagicMock()
        mock_mgr.user_feedback_file = tmp_path / "nonexistent.csv"

        success, msg = export_user_feedback(tmp_path / "dest.csv", mock_mgr)
        assert success is False
        assert "No feedback" in msg

    def test_validate_csv_columns_valid(self, tmp_path):
        from src.services.model_io_service import FEEDBACK_COLUMNS, _validate_csv_columns

        csv_file = tmp_path / "feedback.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FEEDBACK_COLUMNS)
            writer.writerow(["test_term", "positive", "2026-01-01", "", "", ""])

        valid, msg, headers = _validate_csv_columns(csv_file)
        assert valid is True

    def test_validate_csv_missing_required_columns(self, tmp_path):
        from src.services.model_io_service import _validate_csv_columns

        csv_file = tmp_path / "bad.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["foo", "bar"])

        valid, msg, _ = _validate_csv_columns(csv_file)
        assert valid is False
        assert "missing required" in msg

    def test_validate_csv_empty_file(self, tmp_path):
        from src.services.model_io_service import _validate_csv_columns

        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")

        valid, msg, _ = _validate_csv_columns(csv_file)
        assert valid is False

    def test_validate_csv_unrecognized_columns(self, tmp_path):
        from src.services.model_io_service import _validate_csv_columns

        csv_file = tmp_path / "extra.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["term", "feedback", "timestamp", "EXTRA_COL"])

        valid, msg, _ = _validate_csv_columns(csv_file)
        assert valid is False
        assert "unrecognized" in msg


class TestModelIOFeedbackImport:
    """Import feedback CSV in replace and append modes."""

    def _make_csv(self, path, columns, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_import_replace_mode(self, tmp_path):
        from src.services.model_io_service import FEEDBACK_COLUMNS, import_user_feedback

        src_csv = tmp_path / "import.csv"
        self._make_csv(
            src_csv,
            FEEDBACK_COLUMNS,
            [{"term": "test", "feedback": "positive", "timestamp": "2026-01-01"}],
        )

        mock_mgr = MagicMock()
        mock_mgr.user_feedback_file = tmp_path / "user_feedback.csv"

        success, msg, count = import_user_feedback(src_csv, "replace", mock_mgr)
        assert success is True
        assert count == 1
        assert mock_mgr.user_feedback_file.exists()

    def test_import_unknown_mode(self, tmp_path):
        from src.services.model_io_service import FEEDBACK_COLUMNS, import_user_feedback

        src_csv = tmp_path / "import.csv"
        self._make_csv(
            src_csv,
            FEEDBACK_COLUMNS,
            [{"term": "test", "feedback": "positive", "timestamp": "2026-01-01"}],
        )

        mock_mgr = MagicMock()
        mock_mgr.user_feedback_file = tmp_path / "user_feedback.csv"

        success, msg, count = import_user_feedback(src_csv, "badmode", mock_mgr)
        assert success is False
        assert count == 0


# ---------------------------------------------------------------------------
# DocumentService
# ---------------------------------------------------------------------------


class TestDocumentServiceInit:
    """DocumentService initializes with extractor, sanitizer, preprocessor."""

    def test_has_required_components(self):
        with (
            patch("src.services.document_service.RawTextExtractor") as mock_ext,
            patch("src.services.document_service.CharacterSanitizer") as mock_san,
            patch("src.services.document_service.create_default_pipeline") as mock_pipe,
        ):
            mock_ext.return_value = MagicMock()
            mock_san.return_value = MagicMock()
            mock_pipe.return_value = MagicMock()

            from src.services.document_service import DocumentService

            svc = DocumentService()

            assert svc.extractor is not None
            assert svc.sanitizer is not None
            assert svc.preprocessor is not None


class TestDocumentServiceProcess:
    """DocumentService.process_single_document pipeline."""

    def _make_service(self):
        from src.services.document_service import DocumentService

        with (
            patch("src.services.document_service.RawTextExtractor") as mock_ext,
            patch("src.services.document_service.CharacterSanitizer") as mock_san,
            patch("src.services.document_service.create_default_pipeline") as mock_pipe,
        ):
            svc = DocumentService()
        return svc

    def test_process_single_document_returns_dict(self):
        svc = self._make_service()
        svc.extractor = MagicMock()
        svc.extractor.extract.return_value = {"text": "raw text", "confidence": 90}
        svc.sanitizer = MagicMock()
        svc.sanitizer.sanitize.return_value = ("sanitized text", {"chars_removed": 5})
        svc.preprocessor = MagicMock()
        svc.preprocessor.process.return_value = "final text"

        result = svc.process_single_document("/tmp/test.pdf")
        assert result["file_path"] == "/tmp/test.pdf"
        assert result["text"] == "final text"
        assert result["raw_text"] == "raw text"
        assert result["confidence"] == 90
        assert result["word_count"] == 2

    def test_process_documents_calls_progress(self):
        svc = self._make_service()
        svc.extractor = MagicMock()
        svc.extractor.extract.return_value = {"text": "text", "confidence": 80}
        svc.sanitizer = MagicMock()
        svc.sanitizer.sanitize.return_value = ("text", {})
        svc.preprocessor = MagicMock()
        svc.preprocessor.process.return_value = "text"

        progress_calls = []
        svc.process_documents(
            ["/tmp/a.pdf", "/tmp/b.pdf"],
            progress_callback=lambda c, t: progress_calls.append((c, t)),
        )
        assert (0, 2) in progress_calls
        assert (1, 2) in progress_calls
        assert (2, 2) in progress_calls

    def test_combine_texts(self):
        svc = self._make_service()
        results = [{"text": "doc one"}, {"text": "doc two"}, {"text": ""}]
        combined = svc.combine_texts(results)
        assert "doc one" in combined
        assert "doc two" in combined

    def test_get_total_word_count(self):
        svc = self._make_service()
        results = [{"word_count": 100}, {"word_count": 200}]
        assert svc.get_total_word_count(results) == 300


# ---------------------------------------------------------------------------
# OCRAvailability
# ---------------------------------------------------------------------------


class TestOCRAvailability:
    """OCR availability detection."""

    def test_both_available(self):
        from src.services.ocr_availability import OCRStatus, check_ocr_availability

        with (
            patch("src.services.ocr_availability._find_tesseract", return_value=True),
            patch("src.services.ocr_availability._find_poppler", return_value=True),
        ):
            assert check_ocr_availability() == OCRStatus.AVAILABLE

    def test_both_missing(self):
        from src.services.ocr_availability import OCRStatus, check_ocr_availability

        with (
            patch("src.services.ocr_availability._find_tesseract", return_value=False),
            patch("src.services.ocr_availability._find_poppler", return_value=False),
        ):
            assert check_ocr_availability() == OCRStatus.BOTH_MISSING

    def test_tesseract_missing(self):
        from src.services.ocr_availability import OCRStatus, check_ocr_availability

        with (
            patch("src.services.ocr_availability._find_tesseract", return_value=False),
            patch("src.services.ocr_availability._find_poppler", return_value=True),
        ):
            assert check_ocr_availability() == OCRStatus.TESSERACT_MISSING

    def test_poppler_missing(self):
        from src.services.ocr_availability import OCRStatus, check_ocr_availability

        with (
            patch("src.services.ocr_availability._find_tesseract", return_value=True),
            patch("src.services.ocr_availability._find_poppler", return_value=False),
        ):
            assert check_ocr_availability() == OCRStatus.POPPLER_MISSING

    def test_find_tesseract_bundled(self, tmp_path):
        from src.services.ocr_availability import _find_tesseract

        bundled = tmp_path / "tesseract.exe"
        bundled.write_text("fake", encoding="utf-8")

        with patch("src.config.TESSERACT_BUNDLED_EXE", bundled):
            assert _find_tesseract() is True

    def test_find_tesseract_on_path(self):
        from src.services.ocr_availability import _find_tesseract

        with patch("src.config.TESSERACT_BUNDLED_EXE", Path("/nonexistent")):
            with patch("shutil.which", return_value="/usr/bin/tesseract"):
                assert _find_tesseract() is True

    def test_find_poppler_bundled(self, tmp_path):
        from src.services.ocr_availability import _find_poppler

        poppler_dir = tmp_path / "poppler"
        poppler_dir.mkdir()
        (poppler_dir / "pdftoppm.exe").write_text("fake", encoding="utf-8")

        with patch("src.config.POPPLER_BUNDLED_DIR", poppler_dir):
            assert _find_poppler() is True

    def test_find_poppler_on_path(self):
        from src.services.ocr_availability import _find_poppler

        with patch("src.config.POPPLER_BUNDLED_DIR", Path("/nonexistent")):
            with patch("shutil.which", return_value="/usr/bin/pdftoppm"):
                assert _find_poppler() is True

    def test_find_poppler_missing(self):
        from src.services.ocr_availability import _find_poppler

        with patch("src.config.POPPLER_BUNDLED_DIR", Path("/nonexistent")):
            with patch("shutil.which", return_value=None):
                assert _find_poppler() is False
