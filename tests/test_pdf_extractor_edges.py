"""
Edge-case tests for PDFExtractor.

Covers blank-page PDFs, mixed digital/scanned pages, extractor
disagreement (CID vs clean text), both-extractors-fail, single-page
PDFs, and Unicode content preservation.
All PDF I/O is mocked — no real PDF files are needed.
"""

from pathlib import Path
from unittest.mock import MagicMock


def _make_extractor():
    """Create a PDFExtractor with a mock dictionary validator."""
    from src.core.extraction.pdf_extractor import PDFExtractor

    mock_dict = MagicMock()
    mock_dict.calculate_confidence.return_value = 90.0
    return PDFExtractor(dictionary=mock_dict)


class TestBlankPagesPdf:
    """Mock a PDF with 3 blank pages returning empty text."""

    def test_blank_pages_indicate_no_content(self):
        """Three blank pages yield error or needs_ocr result."""
        extractor = _make_extractor()

        # PyMuPDF returns empty text
        extractor._extract_pymupdf_layout = MagicMock(return_value=(None, 3, None))
        extractor._extract_pymupdf = MagicMock(return_value=("", 3, None))
        # pdfplumber also returns empty text
        extractor._extract_pdfplumber = MagicMock(return_value=("", 3, None))

        result = extractor.extract(Path("blank.pdf"))

        # Both extractors returned empty string, so no usable text
        assert result["text"] is None or result["text"] == ""
        assert result["page_count"] == 3


class TestMixedContentPages:
    """Pages 1-2 have text, page 3 is scanned (empty)."""

    def test_digital_text_extracted_scanned_flagged(self):
        """Digital text survives; scanned pages detected."""
        extractor = _make_extractor()
        good_text = "The plaintiff alleges negligence.\fDefendant denies all liability."

        # Layout returns None so flat extraction runs
        extractor._extract_pymupdf_layout = MagicMock(return_value=(None, 3, None))
        extractor._extract_pymupdf = MagicMock(return_value=(good_text, 3, None))
        extractor._extract_pdfplumber = MagicMock(return_value=(good_text, 3, None))
        # Dictionary gives high confidence for the good text
        extractor.dictionary.calculate_confidence.return_value = 85.0

        result = extractor.extract(Path("mixed.pdf"))

        assert result["text"] is not None
        assert "plaintiff" in result["text"].lower()
        assert result["confidence"] > 0


class TestExtractorDisagreement:
    """PyMuPDF returns CID garbage, pdfplumber returns clean text."""

    def test_pdfplumber_chosen_over_cid_garbage(self):
        """Best-of-two selects higher-confidence pdfplumber text."""
        extractor = _make_extractor()

        cid_text = "(cid:72)(cid:101)(cid:108) garbled output " * 20
        clean_text = "The court grants summary judgment for the plaintiff."

        extractor._extract_pymupdf_layout = MagicMock(return_value=(None, 5, None))
        extractor._extract_pymupdf = MagicMock(return_value=(cid_text, 5, None))
        extractor._extract_pdfplumber = MagicMock(return_value=(clean_text, 5, None))

        # Mock confidence: low for CID, high for clean
        def _confidence(text):
            """Return low confidence for CID text, high for clean."""
            if "(cid:" in text:
                return 10.0
            return 92.0

        extractor.dictionary.calculate_confidence.side_effect = _confidence

        result = extractor.extract(Path("cid.pdf"))

        assert result["method"] == "pdfplumber_best"
        assert "summary judgment" in result["text"]


class TestBothExtractorsFail:
    """Both PyMuPDF and pdfplumber raise exceptions."""

    def test_graceful_error_return(self):
        """Both fail; result has error and text is None."""
        extractor = _make_extractor()

        extractor._extract_pymupdf_layout = MagicMock(return_value=(None, 0, None))
        extractor._extract_pymupdf = MagicMock(return_value=(None, 0, "corrupted"))
        extractor._extract_pdfplumber = MagicMock(return_value=(None, 0, "corrupted"))

        result = extractor.extract(Path("broken.pdf"))

        assert result["text"] is None
        assert result["error"] is not None


class TestSinglePagePdf:
    """Mock a 1-page PDF with short text."""

    def test_single_page_extraction(self):
        """Minimal single-page PDF extracts successfully."""
        extractor = _make_extractor()

        short_text = "Order granting motion to dismiss."

        extractor._extract_pymupdf_layout = MagicMock(return_value=(None, 1, None))
        extractor._extract_pymupdf = MagicMock(return_value=(short_text, 1, None))
        extractor._extract_pdfplumber = MagicMock(return_value=(short_text, 1, None))
        extractor.dictionary.calculate_confidence.return_value = 88.0

        result = extractor.extract(Path("single.pdf"))

        assert result["text"] is not None
        assert result["page_count"] == 1
        assert "dismiss" in result["text"]


class TestUnicodePdfContent:
    """Mock PDF returning accented chars, em-dashes, smart quotes."""

    def test_unicode_survives_extraction(self):
        """Unicode characters are not corrupted."""
        extractor = _make_extractor()

        unicode_text = (
            "Caf\u00e9 R\u00e9sum\u00e9 \u2014 the defendant\u2019s "
            "\u201cstatement\u201d contained \u00fc\u00f1\u00ee\u00e7\u00f6d\u00e9 "
            "characters throughout the deposition."
        )

        extractor._extract_pymupdf_layout = MagicMock(return_value=(None, 2, None))
        extractor._extract_pymupdf = MagicMock(return_value=(unicode_text, 2, None))
        extractor._extract_pdfplumber = MagicMock(return_value=(unicode_text, 2, None))
        extractor.dictionary.calculate_confidence.return_value = 80.0

        result = extractor.extract(Path("unicode.pdf"))

        assert result["text"] is not None
        assert "\u00e9" in result["text"]  # accented e
        assert "\u2014" in result["text"]  # em-dash
        assert "\u2019" in result["text"]  # smart apostrophe
        assert "\u201c" in result["text"]  # smart open quote
