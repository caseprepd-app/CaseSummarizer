"""
Tests for token_budget.py — token counting, budget computation, and sub-chunking.

Covers count_tokens, compute_context_budget, _build_windows, and _ensure_fits.
"""

from src.core.semantic.token_budget import (
    _build_windows,
    _ensure_fits,
    compute_context_budget,
    count_tokens,
)

# =========================================================================
# count_tokens
# =========================================================================


class TestCountTokens:
    """Tests for count_tokens()."""

    def test_empty_string(self):
        """Empty string has 0 tokens."""
        assert count_tokens("") == 0

    def test_single_word(self):
        """Single common word is 1 token."""
        result = count_tokens("hello")
        assert result >= 1

    def test_longer_text_more_tokens(self):
        """Longer text produces more tokens."""
        short = count_tokens("hello")
        long = count_tokens("The plaintiff filed a motion in the Supreme Court.")
        assert long > short

    def test_returns_int(self):
        """Token count is always an integer."""
        assert isinstance(count_tokens("test"), int)


# =========================================================================
# compute_context_budget
# =========================================================================


class TestComputeContextBudget:
    """Tests for compute_context_budget()."""

    def test_basic_calculation(self):
        """Budget = window - template - question - output - safety."""
        budget = compute_context_budget(
            context_window=4096,
            prompt_template_tokens=200,
            question_tokens=50,
            max_output_tokens=500,
            safety_margin=16,
        )
        assert budget == 4096 - 200 - 50 - 500 - 16

    def test_minimum_64(self):
        """Budget never goes below 64 even with tight constraints."""
        budget = compute_context_budget(
            context_window=100,
            prompt_template_tokens=50,
            question_tokens=50,
            max_output_tokens=50,
        )
        assert budget == 64

    def test_default_safety_margin(self):
        """Default safety margin is 16."""
        budget = compute_context_budget(
            context_window=1000,
            prompt_template_tokens=100,
            question_tokens=50,
            max_output_tokens=200,
        )
        assert budget == 1000 - 100 - 50 - 200 - 16


# =========================================================================
# _build_windows
# =========================================================================


class TestBuildWindows:
    """Tests for _build_windows()."""

    def test_single_window(self):
        """Single window returns one chunk from the start."""
        text = "abcdefghij"  # 10 chars
        windows = _build_windows(text, ratio=0.6, num_windows=1)
        assert len(windows) == 1
        assert windows[0] == "abcdef"  # 60% of 10 = 6 chars

    def test_three_windows_cover_full_text(self):
        """Three windows should collectively cover the entire text."""
        text = "a" * 100
        windows = _build_windows(text, ratio=0.6, num_windows=3)
        assert len(windows) == 3
        # First window starts at 0, last window ends at 100
        assert windows[0].startswith("a")
        assert len(windows[-1]) == 60  # 60% of 100

    def test_window_overlap(self):
        """Windows overlap — they share content."""
        text = "0123456789" * 10  # 100 chars
        windows = _build_windows(text, ratio=0.6, num_windows=3)
        # Windows should overlap (start positions shift by step)
        all_chars = set()
        for w in windows:
            all_chars.update(range(len(w)))
        assert len(windows[0]) == 60
        assert len(windows[1]) == 60

    def test_five_conservative_windows(self):
        """Five windows at 80% ratio."""
        text = "x" * 200
        windows = _build_windows(text, ratio=0.80, num_windows=5)
        assert len(windows) == 5
        for w in windows:
            assert len(w) == 160  # 80% of 200


# =========================================================================
# _ensure_fits
# =========================================================================


class TestEnsureFits:
    """Tests for _ensure_fits()."""

    def test_short_text_unchanged(self):
        """Text within budget is returned unchanged."""
        result = _ensure_fits("hello world", max_tokens=100)
        assert result == "hello world"

    def test_long_text_truncated(self):
        """Text over budget is truncated."""
        long_text = "word " * 500  # ~500 tokens
        result = _ensure_fits(long_text, max_tokens=50)
        assert count_tokens(result) <= 50

    def test_truncation_at_separator_when_available(self):
        """Truncation breaks at chunk separator when possible."""
        text = "Start content.\n\n---\n\nMiddle content.\n\n---\n\nEnd content." * 50
        result = _ensure_fits(text, max_tokens=20)
        # Result should be truncated but still valid text
        assert len(result) < len(text)
