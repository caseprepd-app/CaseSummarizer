import pytest

from src.core.extraction.text_normalizer import TextNormalizer


@pytest.fixture
def normalizer():
    return TextNormalizer()


# De-hyphenation
def test_dehyphenation(normalizer):
    result = normalizer.normalize("The plain-\ntiff filed.")
    assert "plaintiff" in result


# Page number patterns
def test_page_number_patterns(normalizer):
    assert normalizer._is_page_number("Page 5")
    assert normalizer._is_page_number("Page 1 of 10")
    assert normalizer._is_page_number("- 12 -")
    assert normalizer._is_page_number("42")
    assert normalizer._is_page_number("P. 3")
    assert normalizer._is_page_number("Pg. 7")
    assert normalizer._is_page_number("3/10")
    assert not normalizer._is_page_number("The plaintiff")
    assert not normalizer._is_page_number("12345")  # too many digits
    assert not normalizer._is_page_number("1.")  # list item


# Legal header detection
def test_legal_header(normalizer):
    assert normalizer._is_legal_header("SUPREME COURT OF NEW YORK")
    assert not normalizer._is_legal_header("The plaintiff filed")
    assert not normalizer._is_legal_header("A" * 60 + " COURT")  # too long


# Line filtering
def test_should_keep_line(normalizer):
    assert normalizer._should_keep_line("The plaintiff filed a motion.")
    assert normalizer._should_keep_line("SUPREME COURT")  # legal header
    assert not normalizer._should_keep_line("##")  # too short, not legal


# Whitespace normalization
def test_whitespace_normalization(normalizer):
    # Use lines long enough to pass line filtering (MIN_LINE_LENGTH=15)
    result = normalizer.normalize(
        "The plaintiff filed a motion.\n\n\n\n\nThe defendant responded promptly."
    )
    assert "\n\n\n" not in result
    assert "plaintiff" in result
    assert "defendant" in result


# Full pipeline
def test_full_pipeline(normalizer):
    text = "The plain-\ntiff filed a motion in court.\nPage 1\n\n\nThe defendant responded to the motion."
    result = normalizer.normalize(text)
    assert "plaintiff" in result
    assert "Page 1" not in result
