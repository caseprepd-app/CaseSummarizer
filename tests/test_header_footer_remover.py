"""
Tests for src/core/preprocessing/header_footer_remover.py.

Covers frequency-based header/footer detection and removal.
"""

from unittest.mock import patch


def _make_remover(**pref_overrides):
    """Create a HeaderFooterRemover with mocked preferences."""
    prefs = {
        "header_footer_min_occurrences": 3,
        "header_footer_short_line_detection": True,
        "custom_header_footer_patterns": "",
    }
    prefs.update(pref_overrides)

    mock_prefs = type("MockPrefs", (), {"get": lambda self, k, d=None: prefs.get(k, d)})()
    with patch("src.user_preferences.get_user_preferences", return_value=mock_prefs):
        from src.core.preprocessing.header_footer_remover import HeaderFooterRemover

        return HeaderFooterRemover()


class TestNormalizeLine:
    """Tests for _normalize_line()."""

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        remover = _make_remover()
        assert remover._normalize_line("  hello  ") == "hello"

    def test_removes_trailing_page_numbers(self):
        """Should strip trailing page numbers like '- 12 -'."""
        remover = _make_remover()
        result = remover._normalize_line("SMITH DEPOSITION - 12 -")
        assert "12" not in result

    def test_lowercases(self):
        """Should lowercase the result."""
        remover = _make_remover()
        assert remover._normalize_line("HELLO WORLD") == "hello world"

    def test_collapses_whitespace(self):
        """Should collapse multiple spaces to one."""
        remover = _make_remover()
        assert remover._normalize_line("hello   world") == "hello world"


class TestIsHeaderFooterCandidate:
    """Tests for _is_header_footer_candidate()."""

    def test_page_number_only(self):
        """Bare page number should be a candidate."""
        remover = _make_remover()
        assert remover._is_header_footer_candidate("  12  ") is True

    def test_page_with_text(self):
        """'Page 5 of 10' should be a candidate."""
        remover = _make_remover()
        assert remover._is_header_footer_candidate("Page 5 of 10") is True

    def test_case_caption_pattern(self):
        """'Plaintiff v. Defendant' should match."""
        remover = _make_remover()
        assert remover._is_header_footer_candidate("PLAINTIFF v. DEFENDANT") is True

    def test_confidentiality_notice(self):
        """'CONFIDENTIAL' should match."""
        remover = _make_remover()
        assert remover._is_header_footer_candidate("CONFIDENTIAL") is True

    def test_empty_line_not_candidate(self):
        """Empty line should not be a candidate."""
        remover = _make_remover()
        assert remover._is_header_footer_candidate("") is False

    def test_long_prose_not_candidate(self):
        """Long prose lines should not be candidates."""
        remover = _make_remover()
        long_line = "The defendant testified that on the morning of March 15, " * 5
        assert remover._is_header_footer_candidate(long_line) is False

    def test_custom_pattern_matches(self):
        """Custom user patterns should be detected."""
        remover = _make_remover(custom_header_footer_patterns="Smith & Jones LLP")
        assert remover._is_header_footer_candidate("Smith & Jones LLP") is True

    def test_short_line_keyword_plaintiff(self):
        """Short line with 'PLAINTIFF' should match when short-line detection on."""
        remover = _make_remover(header_footer_short_line_detection=True)
        assert remover._is_header_footer_candidate("PLAINTIFF") is True

    def test_short_line_keyword_disabled(self):
        """Short-line keyword should NOT match when detection is disabled."""
        remover = _make_remover(header_footer_short_line_detection=False)
        # "PLAINTIFF" alone doesn't match standard patterns
        # (only matches SHORT_LINE_KEYWORDS, which are disabled)
        # But it might match "plaintiff.*defendant" if combined.
        # Standalone "PLAINTIFF" only matches short-line keywords.
        result = remover._is_header_footer_candidate("PLAINTIFF")
        # With short-line disabled, standalone PLAINTIFF only matches short-line keywords
        # which are disabled, so should be False
        assert result is False

    def test_firm_name_pattern(self):
        """Law firm names ending in LLP should match."""
        remover = _make_remover()
        assert remover._is_header_footer_candidate("Johnson & Associates, LLP") is True


class TestProcess:
    """Tests for the process() method."""

    def test_empty_text_returns_unchanged(self):
        """Empty text should return with 0 changes."""
        remover = _make_remover()
        result = remover.process("")
        assert result.text == ""
        assert result.changes_made == 0

    def test_removes_repeated_page_numbers(self):
        """Lines that are just page numbers, repeated 3+ times, should be removed."""
        remover = _make_remover(header_footer_min_occurrences=3)
        lines = []
        for i in range(5):
            lines.append(f"Content on page {i + 1}.")
            lines.append("Page 1 of 5")  # Repeated header
        text = "\n".join(lines)

        result = remover.process(text)
        assert result.changes_made > 0
        # Content lines should be preserved
        assert "Content on page 1." in result.text

    def test_preserves_unique_content(self):
        """Lines that appear only once should never be removed."""
        remover = _make_remover()
        text = "This is unique content.\nAnother unique line.\nA third line."
        result = remover.process(text)
        assert result.changes_made == 0
        assert result.text == text

    def test_requires_both_frequency_and_pattern(self):
        """A line must both repeat AND match a pattern to be removed."""
        remover = _make_remover(header_footer_min_occurrences=2)
        # "Hello world" repeats but doesn't match any pattern
        text = "Hello world\nHello world\nHello world\nOther content."
        result = remover.process(text)
        assert "Hello world" in result.text

    def test_metadata_tracks_removals(self):
        """Result metadata should track removal counts."""
        remover = _make_remover(header_footer_min_occurrences=2)
        # Create text with repeated confidential notice
        lines = ["Content here."] + ["CONFIDENTIAL"] * 5
        text = "\n".join(lines)

        result = remover.process(text)
        assert result.metadata["total_lines_removed"] > 0
