"""
Tests for document_paginator.split_into_sections().

Pure function tests — no GUI mocking needed.
"""

from src.ui.document_paginator import split_into_sections


class TestSplitIntoSections:
    """Tests for split_into_sections()."""

    def test_short_text_single_section(self):
        """Text under 300 words returns a single section."""
        text = "Hello world. " * 50  # 100 words
        result = split_into_sections(text)
        assert len(result) == 1
        assert result[0] == text

    def test_splits_at_paragraph_boundaries(self):
        """Breaks happen at newlines, not mid-paragraph."""
        para1 = " ".join(["word"] * 200)
        para2 = " ".join(["other"] * 200)
        text = para1 + "\n" + para2

        result = split_into_sections(text, words_per_section=300)

        assert len(result) == 2
        assert "word" in result[0]
        assert "other" in result[1]

    def test_long_single_paragraph_force_splits(self):
        """A 1000-word paragraph with no newlines is force-split."""
        text = " ".join(["term"] * 1000)
        result = split_into_sections(text, words_per_section=300)

        assert len(result) >= 3
        for section in result:
            assert len(section.split()) <= 300

    def test_empty_text(self):
        """Empty string returns ['']."""
        assert split_into_sections("") == [""]

    def test_exact_boundary(self):
        """Exactly 300 words returns a single section."""
        text = " ".join(["word"] * 300)
        result = split_into_sections(text, words_per_section=300)
        assert len(result) == 1

    def test_word_count_roughly_correct(self):
        """Each section has at most ~words_per_section + one paragraph of slack."""
        paras = [" ".join(["w"] * 80) for _ in range(10)]  # 800 words total
        text = "\n".join(paras)

        result = split_into_sections(text, words_per_section=300)

        assert len(result) >= 2
        for section in result:
            assert len(section.split()) <= 350  # allow ~1 paragraph slack
