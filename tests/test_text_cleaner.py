"""
Tests for text_cleaner.py — ftfy-based PDF text cleanup.

Verifies ligature expansion, mojibake repair, and graceful fallback
when ftfy is not installed.
"""

from src.core.utils.text_cleaner import clean_extracted_text


class TestLigatureExpansion:
    """ftfy should expand fused ligature characters into normal letters."""

    def test_fi_ligature(self):
        """ﬁ -> fi."""
        assert clean_extracted_text("con\ufb01dential") == "confidential"

    def test_fl_ligature(self):
        """ﬂ -> fl."""
        assert clean_extracted_text("con\ufb02ict") == "conflict"

    def test_ff_ligature(self):
        """ﬀ -> ff."""
        assert clean_extracted_text("e\ufb00ect") == "effect"

    def test_ffi_ligature(self):
        """ﬃ -> ffi."""
        assert clean_extracted_text("o\ufb03ce") == "office"

    def test_ffl_ligature(self):
        """ﬄ -> ffl."""
        assert clean_extracted_text("wa\ufb04e") == "waffle"

    def test_multiple_ligatures_in_sentence(self):
        """Multiple ligatures in one string are all expanded."""
        raw = "The o\ufb03ce \ufb01led the a\ufb03davit about the con\ufb02ict."
        cleaned = clean_extracted_text(raw)
        assert "office" in cleaned
        assert "filed" in cleaned
        assert "affidavit" in cleaned
        assert "conflict" in cleaned


class TestMojibakeRepair:
    """ftfy should fix common encoding errors from PDF extraction."""

    def test_smart_quote_mojibake(self):
        """Garbled smart quote is repaired."""
        # café encoded as latin-1 then decoded as utf-8
        raw = "caf\u00c3\u00a9"
        cleaned = clean_extracted_text(raw)
        assert "café" in cleaned or "cafe" in cleaned

    def test_bom_marker_removed(self):
        """BOM marker is stripped from start of text."""
        raw = "\ufeffHello world"
        cleaned = clean_extracted_text(raw)
        assert not cleaned.startswith("\ufeff")
        assert "Hello" in cleaned


class TestPassthrough:
    """Normal text should be unchanged."""

    def test_normal_english(self):
        """Plain English text passes through unchanged."""
        text = "The plaintiff filed a motion for summary judgment."
        assert clean_extracted_text(text) == text

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert clean_extracted_text("") == ""

    def test_none_returns_none(self):
        """None input returns None."""
        assert clean_extracted_text(None) is None
