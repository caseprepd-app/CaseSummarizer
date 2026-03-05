"""
Tests for Hybrid PDF Extraction with Word-Level Voting.

Session 79: Tests the hybrid extraction pipeline that uses both PyMuPDF
and pdfplumber, reconciling differences with word-level voting.
"""

from unittest.mock import MagicMock

import pytest


class TestWordLevelVoting:
    """Tests for the word-level voting reconciliation logic."""

    @pytest.fixture
    def extractor(self):
        """Create a RawTextExtractor with mocked dictionary."""
        from src.core.extraction.raw_text_extractor import RawTextExtractor

        extractor = RawTextExtractor()
        # Use a small test dictionary
        extractor.english_words = {
            "the",
            "and",
            "is",
            "of",
            "in",
            "to",
            "for",
            "with",
            "on",
            "at",
            "plaintiff",
            "defendant",
            "court",
            "case",
            "motion",
            "order",
            "john",
            "smith",
            "hospital",
            "surgery",
            "patient",
            "doctor",
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "medical",
            "injury",
            "negligence",
            "damages",
            "evidence",
        }
        return extractor

    def test_matching_words_kept(self, extractor):
        """When words match, they should be kept."""
        primary = "The plaintiff filed a motion"
        secondary = "The plaintiff filed a motion"

        result = extractor._reconcile_extractions(primary, secondary)
        assert result == "The plaintiff filed a motion"

    def test_dictionary_word_wins(self, extractor):
        """Dictionary word should win over non-dictionary word."""
        # "teh" is a common OCR error for "the"
        primary = "teh plaintiff filed a motion"
        secondary = "the plaintiff filed a motion"

        result = extractor._reconcile_extractions(primary, secondary)
        assert "the plaintiff" in result

    def test_primary_wins_on_tie(self, extractor):
        """When both words are valid (or both invalid), primary wins."""
        # Both "plaintiff" and "claimant" are valid words
        # But our test dict only has "plaintiff"
        primary = "The plaintiff filed a motion"
        secondary = "The claimant filed a motion"

        result = extractor._reconcile_extractions(primary, secondary)
        # "plaintiff" is in our dict, "claimant" is not
        assert "plaintiff" in result

    def test_ocr_error_correction(self, extractor):
        """OCR errors should be corrected when one extractor gets it right."""
        # Primary has OCR error, secondary is correct
        primary = "Dr. Srnith performed the surgery"
        secondary = "Dr. Smith performed the surgery"

        result = extractor._reconcile_extractions(primary, secondary)
        # "smith" is in our dict, "srnith" is not
        assert "Smith" in result or "smith" in result.lower()

    def test_handles_different_word_counts(self, extractor):
        """Should handle when extractors have different word counts."""
        primary = "The plaintiff filed motion"  # Missing "a"
        secondary = "The plaintiff filed a motion"

        result = extractor._reconcile_extractions(primary, secondary)
        # Should include content from both
        assert "plaintiff" in result
        assert "motion" in result

    def test_empty_primary_returns_secondary(self, extractor):
        """If primary is empty, return secondary."""
        result = extractor._reconcile_extractions("", "The plaintiff filed")
        assert result == "The plaintiff filed"

    def test_empty_secondary_returns_primary(self, extractor):
        """If secondary is empty, return primary."""
        result = extractor._reconcile_extractions("The plaintiff filed", "")
        assert result == "The plaintiff filed"

    def test_both_empty_returns_empty(self, extractor):
        """If both are empty, return empty string."""
        result = extractor._reconcile_extractions("", "")
        assert result == ""

    def test_punctuation_preserved(self, extractor):
        """Punctuation should be preserved in output."""
        primary = "The plaintiff, John Smith, filed."
        secondary = "The plaintiff, John Smith, filed."

        result = extractor._reconcile_extractions(primary, secondary)
        assert "," in result
        assert "." in result


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

    def test_reconcile_extractions_method_exists(self, extractor):
        """The _reconcile_extractions method should exist."""
        assert hasattr(extractor, "_reconcile_extractions")

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
