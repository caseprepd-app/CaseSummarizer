"""
Tests for Hybrid PDF Extraction.

Tests the hybrid extraction pipeline that uses both PyMuPDF
and pdfplumber with best-of-two confidence selection.
"""

from unittest.mock import MagicMock

import pytest


class TestIsValidWord:
    """Tests for the _is_valid_word helper."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with test dictionary."""
        from src.core.extraction.raw_text_extractor import RawTextExtractor

        extractor = RawTextExtractor()
        extractor.english_words = {"the", "plaintiff", "court"}
        return extractor

    def test_valid_word(self, extractor):
        """Valid dictionary word returns True."""
        assert extractor._is_valid_word("plaintiff") is True

    def test_invalid_word(self, extractor):
        """Invalid word returns False."""
        assert extractor._is_valid_word("xyzabc") is False

    def test_case_insensitive(self, extractor):
        """Check is case-insensitive."""
        assert extractor._is_valid_word("PLAINTIFF") is True
        assert extractor._is_valid_word("Plaintiff") is True

    def test_strips_punctuation(self, extractor):
        """Punctuation is stripped before checking."""
        assert extractor._is_valid_word("plaintiff,") is True
        assert extractor._is_valid_word("'plaintiff'") is True
        assert extractor._is_valid_word("(court)") is True


class TestTokenizeForVoting:
    """Tests for the _tokenize_for_voting helper."""

    @pytest.fixture
    def extractor(self):
        """Create extractor."""
        from src.core.extraction.raw_text_extractor import RawTextExtractor

        return RawTextExtractor()

    def test_simple_tokenization(self, extractor):
        """Simple sentence tokenizes correctly."""
        result = extractor._tokenize_for_voting("The plaintiff filed")
        assert result == ["The", "plaintiff", "filed"]

    def test_preserves_punctuation(self, extractor):
        """Punctuation stays attached to words."""
        result = extractor._tokenize_for_voting("Hello, world!")
        assert result == ["Hello,", "world!"]


class TestPyMuPDFExtraction:
    """Tests for the PyMuPDF extraction method."""

    def test_import_fitz(self):
        """PyMuPDF (fitz) should be importable."""
        import fitz

        assert hasattr(fitz, "open")

    def test_method_exists(self):
        """The _extract_text_pymupdf method should exist."""
        from src.core.extraction.raw_text_extractor import RawTextExtractor

        extractor = RawTextExtractor()
        assert hasattr(extractor, "_extract_text_pymupdf")


class TestHybridExtractionPipeline:
    """Integration tests for the hybrid extraction pipeline."""

    @pytest.fixture
    def extractor(self):
        """Create extractor."""
        from src.core.extraction.raw_text_extractor import RawTextExtractor

        return RawTextExtractor()

    def test_process_pdf_method_exists(self, extractor):
        """The _process_pdf method should exist."""
        assert hasattr(extractor, "_process_pdf")

    def test_hybrid_method_reported(self, extractor):
        """When both extractors succeed, method should be best-of-two."""
        # Create long enough text to pass the >1000 char check
        long_text = "The plaintiff filed a motion. " * 50  # ~1500 chars

        # Mock both extraction methods
        extractor._extract_text_pymupdf = MagicMock(return_value=(long_text, 1, None))
        extractor._extract_pdf_text = MagicMock(return_value=(long_text, 1, None))

        from pathlib import Path

        result = extractor._process_pdf(Path("test.pdf"))

        assert result["method"] in ("pymupdf_best", "pdfplumber_best")
        assert result["status"] == "success"
        assert result["confidence"] > 0
