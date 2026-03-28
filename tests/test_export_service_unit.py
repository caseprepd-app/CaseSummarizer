"""
Tests for src/services/export_service.py.

Covers the ExportService high-level API and _run_export helper.
Heavy dependencies (Word/PDF builders, file I/O) are mocked.
"""

from unittest.mock import MagicMock, patch

from src.core.vocab_schema import VF


def _make_vocab(n=2):
    """Create sample vocabulary data."""
    return [
        {VF.TERM: f"term_{i}", VF.QUALITY_SCORE: 0.9, VF.IS_PERSON: VF.NO, VF.FOUND_BY: "NER"}
        for i in range(n)
    ]


def _make_result():
    """Create a mock SemanticResult."""
    r = MagicMock()
    r.question = "Who filed?"
    r.citation = "Plaintiff filed."
    r.source_summary = "doc.pdf"
    r.quick_answer = ""
    return r


class TestRunExport:
    """Tests for the _run_export helper."""

    def test_success_returns_true(self):
        """Successful export should return (True, None)."""
        from src.services.export_service import _run_export

        with patch("src.services.export_service._auto_open_file"):
            ok, err = _run_export("test", "/tmp/f.txt", "test", lambda: True)
        assert ok is True
        assert err is None

    def test_exception_returns_false_with_detail(self):
        """Exception during export should return (False, error_detail)."""
        from src.services.export_service import _run_export

        def fail():
            raise ValueError("disk full")

        ok, err = _run_export("test", "/tmp/f.txt", "test", fail)
        assert ok is False
        assert "disk full" in err

    def test_none_return_treated_as_success(self):
        """Export function returning None is treated as success."""
        from src.services.export_service import _run_export

        with patch("src.services.export_service._auto_open_file"):
            ok, err = _run_export("test", "/tmp/f.txt", "test", lambda: None)
        assert ok is True


class TestAutoOpenFile:
    """Tests for _auto_open_file."""

    def test_skips_when_disabled(self):
        """Should not open file when auto_open_exports is False."""
        from src.services.export_service import _auto_open_file

        mock_prefs = MagicMock()
        mock_prefs.get.return_value = False

        with (
            patch("src.user_preferences.get_user_preferences", return_value=mock_prefs),
            patch("os.startfile") as mock_start,
        ):
            _auto_open_file("test.pdf")
            mock_start.assert_not_called()


class TestExportServiceVocab:
    """Tests for ExportService vocabulary exports."""

    @patch("src.services.export_service._auto_open_file")
    @patch("src.services.export_service.WordDocumentBuilder")
    @patch("src.services.export_service.export_vocabulary")
    def test_vocab_to_word_calls_builder(self, mock_export, mock_builder, mock_open):
        """export_vocabulary_to_word should create builder and call export."""
        from src.services.export_service import ExportService

        svc = ExportService()
        ok, err = svc.export_vocabulary_to_word(_make_vocab(), "/tmp/v.docx")

        assert ok is True
        mock_builder.assert_called_once()
        mock_export.assert_called_once()

    @patch("src.services.export_service._auto_open_file")
    @patch("src.services.export_service.PdfDocumentBuilder")
    @patch("src.services.export_service.export_vocabulary")
    def test_vocab_to_pdf_calls_builder(self, mock_export, mock_builder, mock_open):
        """export_vocabulary_to_pdf should create PDF builder and call export."""
        from src.services.export_service import ExportService

        svc = ExportService()
        ok, err = svc.export_vocabulary_to_pdf(_make_vocab(), "/tmp/v.pdf")

        assert ok is True
        mock_builder.assert_called_once()

    @patch("src.services.export_service._auto_open_file")
    @patch("src.services.export_service.export_vocabulary_txt")
    def test_vocab_to_txt(self, mock_txt, mock_open):
        """export_vocabulary_to_txt should delegate to export_vocabulary_txt."""
        from src.services.export_service import ExportService

        mock_txt.return_value = True
        svc = ExportService()
        ok, err = svc.export_vocabulary_to_txt(_make_vocab(), "/tmp/v.txt")

        assert ok is True
        mock_txt.assert_called_once()


class TestExportServiceSemantic:
    """Tests for ExportService semantic exports."""

    @patch("src.services.export_service._auto_open_file")
    @patch("src.services.export_service.WordDocumentBuilder")
    @patch("src.services.export_service.export_semantic_results")
    def test_semantic_to_word(self, mock_export, mock_builder, mock_open):
        """export_semantic_to_word should create builder and export."""
        from src.services.export_service import ExportService

        svc = ExportService()
        ok, err = svc.export_semantic_to_word([_make_result()], "/tmp/s.docx")

        assert ok is True

    @patch("src.services.export_service._auto_open_file")
    @patch("src.services.export_service.PdfDocumentBuilder")
    @patch("src.services.export_service.export_semantic_results")
    def test_semantic_to_pdf(self, mock_export, mock_builder, mock_open):
        """export_semantic_to_pdf should create PDF builder and export."""
        from src.services.export_service import ExportService

        svc = ExportService()
        ok, err = svc.export_semantic_to_pdf([_make_result()], "/tmp/s.pdf")

        assert ok is True


class TestExportServiceCombined:
    """Tests for ExportService combined exports."""

    @patch("src.services.export_service._auto_open_file")
    @patch("src.services.export_service.WordDocumentBuilder")
    @patch("src.services.export_service.export_combined")
    def test_combined_to_word(self, mock_export, mock_builder, mock_open):
        """export_combined_to_word should use WordDocumentBuilder."""
        from src.services.export_service import ExportService

        svc = ExportService()
        ok, err = svc.export_combined_to_word(_make_vocab(), [_make_result()], "/tmp/c.docx")
        assert ok is True

    @patch("src.services.export_service._auto_open_file")
    @patch("src.services.export_service.PdfDocumentBuilder")
    @patch("src.services.export_service.export_combined")
    def test_combined_to_pdf(self, mock_export, mock_builder, mock_open):
        """export_combined_to_pdf should use PdfDocumentBuilder."""
        from src.services.export_service import ExportService

        svc = ExportService()
        ok, err = svc.export_combined_to_pdf(_make_vocab(), [_make_result()], "/tmp/c.pdf")
        assert ok is True


class TestExportServiceSingleton:
    """Tests for singleton management."""

    def test_get_export_service_returns_instance(self):
        """get_export_service should return an ExportService."""
        from src.services.export_service import ExportService, get_export_service

        svc = get_export_service()
        assert isinstance(svc, ExportService)

    def test_reset_export_service(self):
        """reset_export_service should clear the singleton."""
        from src.services.export_service import get_export_service, reset_export_service

        svc1 = get_export_service()
        reset_export_service()
        svc2 = get_export_service()
        # After reset, a new instance should be created
        assert svc1 is not svc2
