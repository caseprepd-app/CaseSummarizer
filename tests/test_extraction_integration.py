"""
Integration tests for the extraction pipeline.

Verifies that extraction components work together correctly:
best-of-two selection, full pipeline flow, column detection edge
cases, OCR page splicing, and layout safety checks.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor():
    """Create a RawTextExtractor for integration tests."""
    from src.core.extraction.raw_text_extractor import RawTextExtractor

    return RawTextExtractor()


@pytest.fixture
def long_good_text():
    """Text that passes the >1000 char and high-confidence checks."""
    return "The plaintiff filed a motion in the court. " * 40


@pytest.fixture
def short_garbage_text():
    """Text with low dictionary confidence."""
    return "xkcd zqwv plmk rvxt " * 40


# ===========================================================================
# Group 1: Best-of-Two Selection
# ===========================================================================


class TestBestOfTwoSelection:
    """Tests for _pick_best_extraction and the extract() best-of-two logic."""

    def test_pymupdf_wins_higher_confidence(self, extractor, long_good_text, short_garbage_text):
        """High-confidence PyMuPDF + low-confidence pdfplumber → pymupdf_best."""
        text, method = extractor._pick_best_extraction(long_good_text, short_garbage_text)

        assert method == "pymupdf_best"
        assert text == long_good_text

    def test_pdfplumber_wins_higher_confidence(self, extractor, long_good_text, short_garbage_text):
        """Low-confidence PyMuPDF + high-confidence pdfplumber → pdfplumber_best."""
        text, method = extractor._pick_best_extraction(short_garbage_text, long_good_text)

        assert method == "pdfplumber_best"
        assert text == long_good_text

    def test_tied_confidence_pymupdf_wins(self, extractor, long_good_text):
        """Same text from both → PyMuPDF wins (>= preference)."""
        text, method = extractor._pick_best_extraction(long_good_text, long_good_text)

        assert method == "pymupdf_best"

    def test_one_extractor_returns_none(self, extractor, long_good_text):
        """When one extractor returns None, the other is used with *_only method."""
        from pathlib import Path

        extractor._extract_text_pymupdf = MagicMock(return_value=(long_good_text, 5, None))
        extractor._extract_pdf_text = MagicMock(return_value=(None, 0, "corrupted"))
        extractor.pdf_extractor.detect_scanned_pages = MagicMock(return_value=(set(), False))

        result = extractor._process_pdf_inner(Path("test.pdf"))

        assert result["method"] == "pymupdf_only"
        assert result["text"] is not None

    def test_both_extractors_return_none(self, extractor):
        """When both extractors fail, error result returned."""
        from pathlib import Path

        extractor._extract_text_pymupdf = MagicMock(return_value=(None, 0, "corrupted"))
        extractor._extract_pdf_text = MagicMock(return_value=(None, 0, "corrupted"))
        extractor.pdf_extractor.detect_scanned_pages = MagicMock(return_value=(set(), False))

        result = extractor._process_pdf_inner(Path("test.pdf"))

        assert result["status"] == "error"
        assert result["text"] is None
        assert "corrupted" in result["error_message"]


# ===========================================================================
# Group 2: Pipeline Integration
# ===========================================================================


class TestPipelineIntegration:
    """Tests verifying components work together in the full pipeline."""

    def test_process_pdf_inner_flow(self, extractor, long_good_text):
        """Full _process_pdf_inner flow: mock extractors, verify structure."""
        from pathlib import Path

        extractor._extract_text_pymupdf = MagicMock(return_value=(long_good_text, 10, None))
        extractor._extract_pdf_text = MagicMock(return_value=(long_good_text, 10, None))
        extractor.pdf_extractor.detect_scanned_pages = MagicMock(return_value=(set(), False))

        result = extractor._process_pdf_inner(Path("test.pdf"))

        assert result["status"] == "success"
        assert result["method"] in ("pymupdf_best", "pdfplumber_best")
        assert result["confidence"] > 0
        assert result["page_count"] == 10
        assert result["text"] is not None

    def test_normalization_preserves_transcript_text(self):
        """Transcript-style text survives normalization with content intact."""
        from src.core.extraction.text_normalizer import TextNormalizer

        normalizer = TextNormalizer()

        transcript = (
            "Q. Can you state your name for the record?\n"
            "A. My name is John Smith.\n"
            "Q. And where were you on the night of January 15th?\n"
            "A. I was at home with my family.\n"
            "Q. Did you witness the accident?\n"
            "A. Yes, I saw the car run the red light."
        )

        result = normalizer.normalize(transcript)

        assert "John Smith" in result
        assert "January 15th" in result or "January" in result
        assert "accident" in result
        assert "red light" in result

    def test_sanitization_nondestructive_on_clean_text(self):
        """Clean legal text passes through CharacterSanitizer unchanged."""
        from src.core.sanitization import CharacterSanitizer

        sanitizer = CharacterSanitizer()

        clean_text = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "COUNTY OF NEW YORK\n\n"
            "John Smith, Plaintiff,\n"
            "  -against-\n"
            "Jane Doe, Defendant.\n\n"
            "The plaintiff alleges negligence in the performance "
            "of medical duties on or about March 15, 2024."
        )

        result, stats = sanitizer.sanitize(clean_text)

        # All original content should survive
        assert "SUPREME COURT" in result
        assert "John Smith" in result
        assert "negligence" in result
        assert "March 15, 2024" in result


# ===========================================================================
# Group 3: Column Detector Edge Cases
# ===========================================================================


class TestColumnDetectorEdgeCases:
    """Edge cases for multi-column detection logic."""

    def test_three_nonoverlapping_columns(self):
        """Three non-overlapping column clusters → multi-column detected."""
        from src.core.extraction.column_detector import (
            _cluster_x_positions,
            _is_multi_column,
        )

        # Simulate 3 columns: blocks at x=50, x=250, x=450
        # block format: (x0, y0, x1, y1, text, block_no, block_type)
        # MIN_BLOCKS_PER_COLUMN=3, so each column needs ≥3 blocks
        blocks = [
            (40, 100, 150, 120, "Left col line 1", 0, 0),
            (40, 130, 150, 150, "Left col line 2", 1, 0),
            (40, 160, 150, 180, "Left col line 3", 2, 0),
            (240, 100, 350, 120, "Mid col line 1", 3, 0),
            (240, 130, 350, 150, "Mid col line 2", 4, 0),
            (240, 160, 350, 180, "Mid col line 3", 5, 0),
            (440, 100, 550, 120, "Right col line 1", 6, 0),
            (440, 130, 550, 150, "Right col line 2", 7, 0),
            (440, 160, 550, 180, "Right col line 3", 8, 0),
        ]

        columns = _cluster_x_positions(blocks, page_width=600)
        assert len(columns) >= 3
        assert _is_multi_column(columns)

    def test_identical_centers_different_widths_overlap(self):
        """Narrow labels + wide body with same center → overlap → single column."""
        from src.core.extraction.column_detector import (
            _cluster_x_positions,
            _is_multi_column,
        )

        # All blocks centered around x=300 but with varying widths
        # Narrow label (250-350) and wide body (100-500) overlap
        blocks = [
            (250, 100, 350, 120, "Label A", 0, 0),
            (100, 130, 500, 200, "Wide paragraph body text here", 1, 0),
            (250, 210, 350, 230, "Label B", 2, 0),
            (100, 240, 500, 310, "Another wide paragraph body", 3, 0),
        ]

        columns = _cluster_x_positions(blocks, page_width=600)
        # Should NOT be multi-column: blocks overlap in x range
        assert not _is_multi_column(columns)


# ===========================================================================
# Group 4: Splice OCR Pages
# ===========================================================================


class TestSpliceOCRPages:
    """Tests for _splice_ocr_pages static method."""

    def test_correct_page_replacement(self):
        """OCR replaces page 3 of 5 → page 3 updated, others unchanged."""
        from src.core.extraction.raw_text_extractor import RawTextExtractor

        digital = "Page1\fPage2\fPage3_digital\fPage4\fPage5"
        ocr_pages = {3: "Page3_OCR"}

        result = RawTextExtractor._splice_ocr_pages(digital, ocr_pages)

        pages = result.split("\f")
        assert len(pages) == 5
        assert pages[0] == "Page1"
        assert pages[1] == "Page2"
        assert pages[2] == "Page3_OCR"
        assert pages[3] == "Page4"
        assert pages[4] == "Page5"

    def test_ocr_page_beyond_count_ignored(self):
        """Page number exceeding len(pages) is silently ignored."""
        from src.core.extraction.raw_text_extractor import RawTextExtractor

        digital = "Page1\fPage2\fPage3"
        ocr_pages = {10: "Should be ignored"}

        result = RawTextExtractor._splice_ocr_pages(digital, ocr_pages)

        pages = result.split("\f")
        assert len(pages) == 3
        assert pages == ["Page1", "Page2", "Page3"]


# ===========================================================================
# Group 5: Layout Safety Check
# ===========================================================================


class TestLayoutSafetyCheck:
    """Tests for the clipping safety check in _extract_pymupdf_layout."""

    @staticmethod
    def _make_layout_mocks(flat_text):
        """Build mock doc/zone for layout safety check tests.

        The doc is iterated twice (enumerate for clipping, then generator
        for flat text), so __iter__ returns a fresh iterator each call.
        """
        mock_zone = MagicMock()
        mock_zone.left, mock_zone.top = 50, 50
        mock_zone.right, mock_zone.bottom = 550, 750

        mock_page = MagicMock()
        mock_page.get_text.return_value = flat_text

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        # Return fresh iterator each time __iter__ is called
        mock_doc.__iter__ = lambda _self: iter([mock_page])
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)

        return mock_doc, mock_zone

    def test_clipping_keeps_majority_accepted(self):
        """Clipping keeps >70% of text → layout extraction accepted."""
        from src.core.extraction.dictionary_utils import DictionaryTextValidator
        from src.core.extraction.pdf_extractor import PDFExtractor

        extractor = PDFExtractor(DictionaryTextValidator())

        clipped_text = "word " * 100  # 100 words
        flat_text = "word " * 110  # 110 words → 91% kept

        mock_doc, mock_zone = self._make_layout_mocks(flat_text)
        extractor._layout_analyzer.detect_zones = MagicMock(return_value=mock_zone)

        with patch("src.core.extraction.pdf_extractor.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            mock_fitz.Rect.return_value = MagicMock()

            with patch(
                "src.core.extraction.pdf_extractor.extract_page_text",
                return_value=clipped_text,
            ):
                text, page_count, error = extractor._extract_pymupdf_layout(MagicMock())

        assert text is not None
        assert error is None

    def test_clipping_loses_too_much_falls_back(self):
        """Clipping keeps <70% of text → returns None to fall back to flat."""
        from src.core.extraction.dictionary_utils import DictionaryTextValidator
        from src.core.extraction.pdf_extractor import PDFExtractor

        extractor = PDFExtractor(DictionaryTextValidator())

        clipped_text = "word " * 30  # 30 words
        flat_text = "word " * 100  # 100 words → 30% kept

        mock_doc, mock_zone = self._make_layout_mocks(flat_text)
        extractor._layout_analyzer.detect_zones = MagicMock(return_value=mock_zone)

        with patch("src.core.extraction.pdf_extractor.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            mock_fitz.Rect.return_value = MagicMock()

            with patch(
                "src.core.extraction.pdf_extractor.extract_page_text",
                return_value=clipped_text,
            ):
                text, page_count, error = extractor._extract_pymupdf_layout(MagicMock())

        assert text is None
        assert error is None
        assert page_count == 1
