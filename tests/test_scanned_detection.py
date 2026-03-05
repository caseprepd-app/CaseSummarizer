"""
Tests for scanned page pre-detection, CID marker detection, per-page OCR
routing, and pdfplumber tolerance tuning.

All PDF/OCR dependencies are mocked — no real PDFs or Tesseract needed.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_page(width=612, height=792, images=None, text="", image_rects=None):
    """Create a mock fitz page with configurable image coverage and text."""
    page = MagicMock()
    page.rect = MagicMock()
    page.rect.width = width
    page.rect.height = height

    # Images on the page
    page.get_images.return_value = images or []

    # Image rectangles (coverage areas)
    if image_rects is None and images:
        # Default: one big image covering the whole page
        rect = MagicMock()
        rect.width = width
        rect.height = height
        image_rects = [rect]
    page.get_image_rects.return_value = image_rects or []

    page.get_text.return_value = text
    return page


# ---------------------------------------------------------------------------
# detect_scanned_pages tests
# ---------------------------------------------------------------------------


class TestDetectScannedPages:
    """Tests for PDFExtractor.detect_scanned_pages()."""

    def _make_extractor(self):
        """Create a PDFExtractor with a mocked dictionary."""
        from src.core.extraction.pdf_extractor import PDFExtractor

        dictionary = MagicMock()
        return PDFExtractor(dictionary)

    @patch("src.core.extraction.pdf_extractor.fitz")
    def test_all_digital_pages(self, mock_fitz):
        """Pages with lots of text and no big images → no scanned pages."""
        pages = [
            _make_mock_page(images=[], image_rects=[], text="A" * 200),
            _make_mock_page(images=[], image_rects=[], text="B" * 300),
        ]
        doc = MagicMock()
        doc.__len__ = lambda self: len(pages)
        doc.__iter__ = lambda self: iter(pages)
        doc.__enter__ = lambda self: self
        doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = doc

        extractor = self._make_extractor()
        scanned, all_scanned = extractor.detect_scanned_pages(Path("test.pdf"))

        assert scanned == set()
        assert all_scanned is False

    @patch("src.core.extraction.pdf_extractor.fitz")
    def test_all_scanned_pages(self, mock_fitz):
        """Pages with big image and <50 chars → all scanned."""
        big_img = [("img1",)]
        big_rect = MagicMock()
        big_rect.width = 600
        big_rect.height = 780

        pages = [
            _make_mock_page(images=big_img, image_rects=[big_rect], text=""),
            _make_mock_page(images=big_img, image_rects=[big_rect], text="x" * 10),
        ]
        doc = MagicMock()
        doc.__len__ = lambda self: len(pages)
        doc.__iter__ = lambda self: iter(pages)
        doc.__enter__ = lambda self: self
        doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = doc

        extractor = self._make_extractor()
        scanned, all_scanned = extractor.detect_scanned_pages(Path("test.pdf"))

        assert scanned == {0, 1}
        assert all_scanned is True

    @patch("src.core.extraction.pdf_extractor.fitz")
    def test_mixed_pages(self, mock_fitz):
        """Page 0 digital, page 1 scanned → only page 1 in set."""
        big_img = [("img1",)]
        big_rect = MagicMock()
        big_rect.width = 610
        big_rect.height = 790

        pages = [
            _make_mock_page(images=[], image_rects=[], text="Digital content " * 20),
            _make_mock_page(images=big_img, image_rects=[big_rect], text=""),
        ]
        doc = MagicMock()
        doc.__len__ = lambda self: len(pages)
        doc.__iter__ = lambda self: iter(pages)
        doc.__enter__ = lambda self: self
        doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = doc

        extractor = self._make_extractor()
        scanned, all_scanned = extractor.detect_scanned_pages(Path("test.pdf"))

        assert scanned == {1}
        assert all_scanned is False

    @patch("src.core.extraction.pdf_extractor.fitz")
    def test_empty_pdf(self, mock_fitz):
        """Empty PDF → no scanned pages."""
        doc = MagicMock()
        doc.__len__ = lambda self: 0
        doc.__enter__ = lambda self: self
        doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = doc

        extractor = self._make_extractor()
        scanned, all_scanned = extractor.detect_scanned_pages(Path("test.pdf"))

        assert scanned == set()
        assert all_scanned is False

    @patch("src.core.extraction.pdf_extractor.fitz")
    def test_exception_returns_empty(self, mock_fitz):
        """If fitz.open raises, return empty set gracefully."""
        mock_fitz.open.side_effect = Exception("corrupted")

        extractor = self._make_extractor()
        scanned, all_scanned = extractor.detect_scanned_pages(Path("bad.pdf"))

        assert scanned == set()
        assert all_scanned is False

    @patch("src.core.extraction.pdf_extractor.fitz")
    def test_image_below_threshold(self, mock_fitz):
        """Image covering only 50% of page → not scanned even with little text."""
        half_rect = MagicMock()
        half_rect.width = 306  # half width
        half_rect.height = 792

        pages = [
            _make_mock_page(images=[("img",)], image_rects=[half_rect], text="x" * 10),
        ]
        doc = MagicMock()
        doc.__len__ = lambda self: len(pages)
        doc.__iter__ = lambda self: iter(pages)
        doc.__enter__ = lambda self: self
        doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = doc

        extractor = self._make_extractor()
        scanned, all_scanned = extractor.detect_scanned_pages(Path("test.pdf"))

        assert scanned == set()
        assert all_scanned is False


# ---------------------------------------------------------------------------
# CID marker detection tests
# ---------------------------------------------------------------------------


class TestCIDDetection:
    """Tests for PDFExtractor.has_cid_problem()."""

    def test_clean_text(self):
        from src.core.extraction.pdf_extractor import PDFExtractor

        text = "The plaintiff filed a motion for summary judgment in court."
        assert PDFExtractor.has_cid_problem(text) is False

    def test_text_with_many_cids(self):
        from src.core.extraction.pdf_extractor import PDFExtractor

        # 6 out of 10 words are CID markers → 60% > 5% threshold
        words = ["(cid:72)", "(cid:101)", "(cid:108)", "(cid:108)", "(cid:111)", "(cid:32)"]
        words += ["real", "text", "here", "ok"]
        text = " ".join(words)
        assert PDFExtractor.has_cid_problem(text) is True

    def test_text_with_few_cids(self):
        from src.core.extraction.pdf_extractor import PDFExtractor

        # 1 CID out of 100 words → 1% < 5%
        words = ["word"] * 99 + ["(cid:42)"]
        text = " ".join(words)
        assert PDFExtractor.has_cid_problem(text) is False

    def test_empty_text(self):
        from src.core.extraction.pdf_extractor import PDFExtractor

        assert PDFExtractor.has_cid_problem("") is False
        assert PDFExtractor.has_cid_problem("   ") is False

    def test_custom_threshold(self):
        from src.core.extraction.pdf_extractor import PDFExtractor

        # 2 out of 10 = 20%, above 10% threshold but below default 5%
        words = ["(cid:1)", "(cid:2)"] + ["word"] * 8
        text = " ".join(words)
        assert PDFExtractor.has_cid_problem(text, threshold=0.10) is True
        assert PDFExtractor.has_cid_problem(text, threshold=0.25) is False


# ---------------------------------------------------------------------------
# Per-page OCR routing tests
# ---------------------------------------------------------------------------


class TestPerPageOCR:
    """Tests for OCRProcessor.process_pages()."""

    @patch("src.core.extraction.ocr_processor.OCR_PREPROCESSING_ENABLED", False)
    def test_process_pages_success(self):
        from src.core.extraction.ocr_processor import OCRProcessor

        dictionary = MagicMock()
        dictionary.calculate_confidence.return_value = 75.0
        processor = OCRProcessor(dictionary)

        mock_image = MagicMock()

        with (
            patch("src.core.extraction.ocr_processor._configure_tesseract"),
            patch("src.core.extraction.ocr_processor.OCRProcessor.process_pages") as mock_pp,
        ):
            # Actually call the real method but mock the internals
            mock_pp.side_effect = None

        # Test with direct mocking of imports inside method
        with (
            patch("src.core.extraction.ocr_processor._configure_tesseract"),
            patch.dict("sys.modules", {"pytesseract": MagicMock(), "pdf2image": MagicMock()}),
        ):
            import sys

            mock_pytesseract = sys.modules["pytesseract"]
            mock_pytesseract.image_to_string.return_value = "OCR text page 3"

            mock_pdf2image = sys.modules["pdf2image"]
            mock_pdf2image.convert_from_path.return_value = [mock_image]

            with patch("src.config.POPPLER_BUNDLED_DIR", MagicMock(exists=lambda: False)):
                result = processor.process_pages(Path("test.pdf"), [3], 5)

        assert result["status"] == "success"
        assert 3 in result["pages"]

    @patch("src.core.extraction.ocr_processor.OCR_PREPROCESSING_ENABLED", False)
    def test_process_pages_error_handling(self):
        from src.core.extraction.ocr_processor import OCRProcessor

        dictionary = MagicMock()
        processor = OCRProcessor(dictionary)

        with (
            patch("src.core.extraction.ocr_processor._configure_tesseract"),
            patch.dict("sys.modules", {"pytesseract": MagicMock(), "pdf2image": MagicMock()}),
        ):
            import sys

            sys.modules["pdf2image"].convert_from_path.side_effect = RuntimeError("poppler missing")

            with patch("src.config.POPPLER_BUNDLED_DIR", MagicMock(exists=lambda: False)):
                result = processor.process_pages(Path("test.pdf"), [1], 1)

        assert result["status"] == "error"
        assert "Poppler" in result["error_message"]


# ---------------------------------------------------------------------------
# pdfplumber tolerance tests
# ---------------------------------------------------------------------------


class TestPdfplumberTolerance:
    """Verify x_tolerance_ratio is passed to pdfplumber."""

    @patch("src.core.extraction.pdf_extractor.pdfplumber")
    def test_extract_pdfplumber_uses_tolerance_ratio(self, mock_pdfplumber):
        from src.core.extraction.pdf_extractor import PDFExtractor

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Some text"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = lambda self: self
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        dictionary = MagicMock()
        extractor = PDFExtractor(dictionary)
        extractor._extract_pdfplumber(Path("test.pdf"))

        mock_page.extract_text.assert_called_once_with(x_tolerance_ratio=0.1)


# ---------------------------------------------------------------------------
# Integration: _process_pdf_inner routing tests
# ---------------------------------------------------------------------------


class TestProcessPdfInnerRouting:
    """Tests for _process_pdf_inner pre-scan detection and CID routing."""

    def _make_extractor(self):
        """Create a RawTextExtractor with mocked components."""
        with patch("src.core.extraction.raw_text_extractor.DictionaryTextValidator"):
            with patch("src.core.extraction.raw_text_extractor.TextNormalizer"):
                with patch("src.core.extraction.raw_text_extractor.OCRProcessor"):
                    with patch("src.core.extraction.raw_text_extractor.PDFExtractor"):
                        with patch("src.core.extraction.raw_text_extractor.FileReaders"):
                            with patch(
                                "src.core.extraction.raw_text_extractor.CaseNumberExtractor"
                            ):
                                with patch(
                                    "src.core.extraction.raw_text_extractor.CharacterSanitizer"
                                ):
                                    from src.core.extraction.raw_text_extractor import (
                                        RawTextExtractor,
                                    )

                                    return RawTextExtractor()

    def test_all_scanned_skips_digital_extraction(self):
        """When all pages are scanned, go straight to full OCR."""
        ext = self._make_extractor()
        ext.pdf_extractor.detect_scanned_pages.return_value = ({0, 1, 2}, True)
        ext.ocr_processor.process_pdf.return_value = {
            "text": "OCR text",
            "page_count": 3,
            "method": "ocr",
            "confidence": 70,
            "status": "success",
            "error_message": None,
        }

        result = ext._process_pdf_inner(Path("scan.pdf"))

        assert result["method"] == "ocr"
        ext.ocr_processor.process_pdf.assert_called_once()

    def test_all_scanned_ocr_disabled(self):
        """When all scanned but OCR disabled, return ocr_skipped."""
        ext = self._make_extractor()
        ext.ocr_allowed = False
        ext.pdf_extractor.detect_scanned_pages.return_value = ({0, 1}, True)

        result = ext._process_pdf_inner(Path("scan.pdf"))

        assert result["status"] == "ocr_skipped"

    def test_cid_triggers_ocr(self):
        """CID markers in digital text trigger OCR fallback."""
        ext = self._make_extractor()
        ext.pdf_extractor.detect_scanned_pages.return_value = (set(), False)

        # Mock facade methods
        cid_text = "(cid:72)(cid:101) " * 50
        ext._extract_text_pymupdf = MagicMock(return_value=(cid_text, 2, None))
        ext._extract_pdf_text = MagicMock(return_value=(None, 0, "failed"))
        ext._calculate_dictionary_confidence = MagicMock(return_value=80.0)

        # Mock has_cid_problem to return True
        with patch(
            "src.core.extraction.raw_text_extractor.PDFExtractor.has_cid_problem",
            return_value=True,
        ):
            ext.ocr_processor.process_pdf.return_value = {
                "text": "Real text",
                "page_count": 2,
                "method": "ocr",
                "confidence": 85,
                "status": "success",
                "error_message": None,
            }

            result = ext._process_pdf_inner(Path("cid.pdf"))

        assert result["method"] == "ocr"

    def test_mixed_pages_trigger_partial_ocr(self):
        """Mixed doc: digital extraction + per-page OCR on scanned pages."""
        ext = self._make_extractor()
        ext.pdf_extractor.detect_scanned_pages.return_value = ({2}, False)

        # Mock the facade methods used by _process_pdf_inner
        # Text must be >1000 chars to avoid the len(text) <= 1000 OCR trigger
        digital_text = ("Page1 content " * 100) + "\f" + ("Page2 content " * 100) + "\fBlank"
        ext._extract_text_pymupdf = MagicMock(return_value=(digital_text, 3, None))
        ext._extract_pdf_text = MagicMock(return_value=(None, 0, "failed"))
        ext._calculate_dictionary_confidence = MagicMock(return_value=80.0)

        with patch(
            "src.core.extraction.raw_text_extractor.PDFExtractor.has_cid_problem",
            return_value=False,
        ):
            ext.ocr_processor.process_pages.return_value = {
                "pages": {3: "OCR page 3 text"},
                "method": "ocr_partial",
                "confidence": 75,
                "status": "success",
                "error_message": None,
            }

            result = ext._process_pdf_inner(Path("mixed.pdf"))

        assert result["method"].endswith("+partial_ocr")
        ext.ocr_processor.process_pages.assert_called_once()


# ---------------------------------------------------------------------------
# _splice_ocr_pages tests
# ---------------------------------------------------------------------------


class TestSpliceOcrPages:
    """Tests for RawTextExtractor._splice_ocr_pages()."""

    def test_splice_replaces_correct_page(self):
        from src.core.extraction.raw_text_extractor import RawTextExtractor

        digital = "Page1 text\fPage2 text\fPage3 text"
        ocr_pages = {2: "OCR page 2"}

        result = RawTextExtractor._splice_ocr_pages(digital, ocr_pages)
        pages = result.split("\f")

        assert pages[0] == "Page1 text"
        assert pages[1] == "OCR page 2"
        assert pages[2] == "Page3 text"

    def test_splice_multiple_pages(self):
        from src.core.extraction.raw_text_extractor import RawTextExtractor

        digital = "P1\fP2\fP3\fP4"
        ocr_pages = {1: "OCR1", 3: "OCR3"}

        result = RawTextExtractor._splice_ocr_pages(digital, ocr_pages)
        pages = result.split("\f")

        assert pages[0] == "OCR1"
        assert pages[1] == "P2"
        assert pages[2] == "OCR3"
        assert pages[3] == "P4"

    def test_splice_out_of_range_ignored(self):
        from src.core.extraction.raw_text_extractor import RawTextExtractor

        digital = "P1\fP2"
        ocr_pages = {5: "OCR5"}  # page 5 doesn't exist

        result = RawTextExtractor._splice_ocr_pages(digital, ocr_pages)
        assert result == "P1\fP2"
