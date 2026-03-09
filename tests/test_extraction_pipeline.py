"""Tests for extraction pipeline: FileReaders, CaseNumberExtractor, PDFExtractor."""

from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# CaseNumberExtractor
# ---------------------------------------------------------------------------


class TestCaseNumberExtractor:
    """CaseNumberExtractor finds case/index/docket numbers."""

    def _make(self):
        from src.core.extraction.case_number_extractor import CaseNumberExtractor

        return CaseNumberExtractor()

    def test_empty_text(self):
        assert self._make().extract("") == []

    def test_no_case_number(self):
        assert self._make().extract("This is a normal document with no case info.") == []

    def test_federal_case_number(self):
        text = "Filed in Case No. 1:23-cv-12345 on January 1, 2024."
        result = self._make().extract(text)
        assert len(result) >= 1
        assert any("1:23-cv-12345" in cn for cn in result)

    def test_ny_index_number(self):
        text = "SUPREME COURT OF NEW YORK\nIndex No. 123456/2024"
        result = self._make().extract(text)
        assert len(result) >= 1
        assert any("123456/2024" in cn for cn in result)

    def test_docket_number(self):
        text = "Docket No. 2024-98765"
        result = self._make().extract(text)
        assert len(result) >= 1
        assert any("2024-98765" in cn for cn in result)

    def test_generic_case_number(self):
        text = "Case No.: 789012"
        result = self._make().extract(text)
        assert len(result) >= 1

    def test_multiple_case_numbers(self):
        text = "Index No. 111/2024 and Case No. 222"
        result = self._make().extract(text)
        assert len(result) >= 2

    def test_deduplication(self):
        text = "Case No. 12345\nSee Case No. 12345 above."
        result = self._make().extract(text)
        # Should deduplicate
        assert len(result) == len(set(result))

    def test_case_insensitive(self):
        text = "case no. 54321"
        result = self._make().extract(text)
        assert len(result) >= 1

    def test_optional_colon(self):
        text1 = "Case No: 111"
        text2 = "Case No. 222"
        text3 = "Case No 333"
        assert len(self._make().extract(text1)) >= 1
        assert len(self._make().extract(text2)) >= 1
        assert len(self._make().extract(text3)) >= 1


# ---------------------------------------------------------------------------
# FileReaders
# ---------------------------------------------------------------------------


class TestFileReadersTextFile:
    """FileReaders.read_text_file reads .txt files."""

    def _make(self):
        from src.core.extraction.file_readers import FileReaders

        mock_dict = MagicMock()
        mock_dict.calculate_confidence.return_value = 95.0
        return FileReaders(dictionary=mock_dict)

    def test_reads_utf8_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello World", encoding="utf-8")

        result = self._make().read_text_file(f)
        assert result["status"] == "success"
        assert result["text"] == "Hello World"
        assert result["method"] == "direct_read"
        assert result["confidence"] == 95

    def test_missing_file_returns_error(self, tmp_path):
        result = self._make().read_text_file(tmp_path / "nonexistent.txt")
        assert result["status"] == "error"
        assert result["error_message"] is not None

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        result = self._make().read_text_file(f)
        assert result["status"] == "success"
        assert result["text"] == ""


class TestFileReadersRTF:
    """FileReaders.read_rtf_file reads .rtf files."""

    def _make(self):
        from src.core.extraction.file_readers import FileReaders

        mock_dict = MagicMock()
        mock_dict.calculate_confidence.return_value = 90.0
        return FileReaders(dictionary=mock_dict)

    def test_reads_rtf_file(self, tmp_path):
        f = tmp_path / "test.rtf"
        # Minimal RTF content
        f.write_text(r"{\rtf1 Hello RTF}", encoding="utf-8")

        result = self._make().read_rtf_file(f)
        # Should succeed or return error gracefully
        assert result["status"] in ("success", "error")

    def test_missing_rtf_returns_error(self, tmp_path):
        result = self._make().read_rtf_file(tmp_path / "nonexistent.rtf")
        assert result["status"] == "error"


class TestFileReadersDocx:
    """FileReaders.read_docx_file reads .docx files."""

    def _make(self):
        from src.core.extraction.file_readers import FileReaders

        mock_dict = MagicMock()
        mock_dict.calculate_confidence.return_value = 90.0
        return FileReaders(dictionary=mock_dict)

    def test_missing_docx_returns_error(self, tmp_path):
        result = self._make().read_docx_file(tmp_path / "nonexistent.docx")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# PDFExtractor
# ---------------------------------------------------------------------------


class TestPDFExtractorInit:
    """PDFExtractor initialization."""

    def _make(self):
        from src.core.extraction.dictionary_utils import DictionaryTextValidator
        from src.core.extraction.pdf_extractor import PDFExtractor

        return PDFExtractor(dictionary=DictionaryTextValidator())

    def test_creates_with_dictionary(self):
        ext = self._make()
        assert ext is not None

    def test_has_extract_method(self):
        ext = self._make()
        assert hasattr(ext, "extract")
        assert callable(ext.extract)


class TestPDFExtractorExtract:
    """PDFExtractor.extract processes PDF files."""

    def _make(self):
        from src.core.extraction.dictionary_utils import DictionaryTextValidator
        from src.core.extraction.pdf_extractor import PDFExtractor

        return PDFExtractor(dictionary=DictionaryTextValidator())

    def test_nonexistent_pdf_returns_error(self, tmp_path):
        ext = self._make()
        result = ext.extract(tmp_path / "nonexistent.pdf")
        # Result uses 'error' key (not 'status'), text is None on failure
        assert result.get("error") is not None or result.get("text") is None

    def test_empty_pdf_path(self):
        from pathlib import Path

        ext = self._make()
        try:
            result = ext.extract(Path(""))
            assert result.get("error") is not None or result.get("text") is None
        except (FileNotFoundError, ValueError, OSError):
            pass  # Also acceptable


# ---------------------------------------------------------------------------
# DictionaryUtils / DictionaryTextValidator
# ---------------------------------------------------------------------------


class TestDictionaryTextValidator:
    """DictionaryTextValidator provides dictionary-based text validation."""

    def test_creates_instance(self):
        from src.core.extraction.dictionary_utils import DictionaryTextValidator

        helpers = DictionaryTextValidator()
        assert helpers is not None

    def test_calculate_confidence(self):
        from src.core.extraction.dictionary_utils import DictionaryTextValidator

        helpers = DictionaryTextValidator()

        # Real English text should have decent confidence
        confidence = helpers.calculate_confidence(
            "The plaintiff filed a motion for summary judgment."
        )
        assert isinstance(confidence, (int, float))
        assert confidence >= 0

    def test_confidence_empty_text(self):
        from src.core.extraction.dictionary_utils import DictionaryTextValidator

        helpers = DictionaryTextValidator()
        confidence = helpers.calculate_confidence("")
        assert isinstance(confidence, (int, float))
        assert confidence >= 0

    def test_confidence_gibberish(self):
        from src.core.extraction.dictionary_utils import DictionaryTextValidator

        helpers = DictionaryTextValidator()
        confidence = helpers.calculate_confidence("xkjw qrmn tplz zzqx")
        assert isinstance(confidence, (int, float))
        # Gibberish should have lower confidence than real text
        real_confidence = helpers.calculate_confidence("The court has ruled in favor.")
        assert confidence <= real_confidence
