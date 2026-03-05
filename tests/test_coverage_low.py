"""
Low-priority coverage gap tests.

Covers two modules:
1. src.core.utils.tokenizer — tokenize(), tokenize_simple(), TokenizerConfig
2. src.logging_config — refresh_log_filter() (update handler level),
   _CategoryFilter.filter() (pass/block records)
"""

import logging
from unittest.mock import MagicMock, patch

from src.core.utils.tokenizer import (
    RETRIEVAL_CONFIG,
    TokenizerConfig,
    tokenize,
    tokenize_simple,
)
from src.logging_config import (
    _CategoryFilter,
    refresh_log_filter,
)

# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


class TestTokenize:
    """Tests for the tokenize() function with default and custom configs."""

    def test_splits_text_and_filters_stopwords(self):
        """tokenize() with default config removes stopwords and short tokens."""
        result = tokenize("The plaintiff's medical records show evidence")
        # "the" is a stopword, "show" is a stopword
        assert "plaintiff's" in result
        assert "medical" in result
        assert "records" not in result  # "records" is in STOPWORDS
        assert "evidence" in result
        assert "the" not in result
        # All tokens are lowercase
        for token in result:
            assert token == token.lower()

    def test_default_config_filters_short_tokens(self):
        """Tokens shorter than min_length=3 are excluded by default config."""
        result = tokenize("I am ok in an MRI scan")
        # "i", "am", "in", "an" are < 3 chars; "ok" is 2 chars
        assert "mri" in result
        assert "scan" in result  # 4 chars, not a stopword — kept
        # No token shorter than min_length=3 should survive
        for token in result:
            assert len(token) >= 3

    def test_custom_config_no_stopword_filtering(self):
        """TokenizerConfig with filter_stopwords=False preserves all words."""
        config = TokenizerConfig(filter_stopwords=False, min_length=1)
        result = tokenize("The cat is on the mat", config)
        assert "the" in result
        assert "is" in result
        assert "on" in result
        assert "cat" in result
        assert "mat" in result

    def test_custom_config_min_length(self):
        """TokenizerConfig with higher min_length filters shorter words."""
        config = TokenizerConfig(filter_stopwords=False, min_length=5)
        result = tokenize("The plaintiff suffered cervical injuries", config)
        assert "plaintiff" in result
        assert "suffered" in result
        assert "cervical" in result
        assert "injuries" in result
        # "the" is 3 chars, should be excluded by min_length=5
        assert "the" not in result

    def test_empty_input_returns_empty_list(self):
        """tokenize() on empty or whitespace-only text returns an empty list."""
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestTokenizeSimple:
    """Tests for the tokenize_simple() convenience function."""

    def test_preserves_all_words(self):
        """tokenize_simple() keeps stopwords and short tokens."""
        result = tokenize_simple("Who is the plaintiff")
        assert "who" in result
        assert "is" in result
        assert "the" in result
        assert "plaintiff" in result

    def test_uses_retrieval_config(self):
        """tokenize_simple() uses RETRIEVAL_CONFIG (min_length=1, no filtering)."""
        assert RETRIEVAL_CONFIG.min_length == 1
        assert RETRIEVAL_CONFIG.filter_stopwords is False
        # Single-letter words preserved
        result = tokenize_simple("I object")
        assert "i" in result

    def test_lowercases_all_tokens(self):
        """tokenize_simple() lowercases all tokens."""
        result = tokenize_simple("MEDICAL RECORDS FROM DR SMITH")
        for token in result:
            assert token == token.lower()


# ---------------------------------------------------------------------------
# Logging config tests
# ---------------------------------------------------------------------------


class TestRefreshLogFilter:
    """Tests for refresh_log_filter() which updates file handler level."""

    def test_updates_handler_level_to_warning_for_off(self):
        """refresh_log_filter() sets file handler to WARNING when level is 'off'."""
        mock_handler = MagicMock(spec=logging.Handler)
        with (
            patch("src.logging_config._file_handler", mock_handler),
            patch("src.logging_config._get_logging_level", return_value="off"),
            patch("src.logging_config._custom_enabled_prefixes", "something"),
        ):
            refresh_log_filter()
            mock_handler.setLevel.assert_called_once_with(logging.WARNING)

    def test_updates_handler_level_to_debug_for_comprehensive(self):
        """refresh_log_filter() sets file handler to DEBUG when level is 'comprehensive'."""
        mock_handler = MagicMock(spec=logging.Handler)
        with (
            patch("src.logging_config._file_handler", mock_handler),
            patch("src.logging_config._get_logging_level", return_value="comprehensive"),
            patch("src.logging_config._custom_enabled_prefixes", "something"),
        ):
            refresh_log_filter()
            mock_handler.setLevel.assert_called_once_with(logging.DEBUG)

    def test_invalidates_custom_prefix_cache(self):
        """refresh_log_filter() resets _custom_enabled_prefixes to None."""
        with (
            patch("src.logging_config._file_handler", None),
            patch("src.logging_config._custom_enabled_prefixes", ("src.core",)),
        ):
            refresh_log_filter()
            import src.logging_config as lc

            assert lc._custom_enabled_prefixes is None


class TestCategoryFilter:
    """Tests for _CategoryFilter.filter() pass/block behavior."""

    def _make_record(self, name: str, level: int) -> logging.LogRecord:
        """Create a LogRecord with given logger name and level.

        Args:
            name: Logger name (e.g. 'src.core.qa.service').
            level: Logging level integer.

        Returns:
            A LogRecord instance.
        """
        return logging.LogRecord(
            name=name,
            level=level,
            pathname="",
            lineno=0,
            msg="test message",
            args=None,
            exc_info=None,
        )

    def test_warning_always_passes(self):
        """WARNING+ records always pass the filter regardless of mode."""
        filt = _CategoryFilter()
        record = self._make_record("src.unknown.module", logging.WARNING)
        assert filt.filter(record) is True

    def test_non_custom_mode_passes_all(self):
        """In non-custom modes (e.g. 'brief'), all records pass the filter."""
        filt = _CategoryFilter()
        record = self._make_record("src.some.random.module", logging.DEBUG)
        with patch("src.logging_config._get_logging_level", return_value="brief"):
            assert filt.filter(record) is True

    def test_custom_mode_blocks_unmatched_prefix(self):
        """In custom mode, DEBUG records from non-enabled categories are blocked."""
        filt = _CategoryFilter()
        record = self._make_record("src.ui.main_window", logging.DEBUG)
        with (
            patch("src.logging_config._get_logging_level", return_value="custom"),
            patch(
                "src.logging_config._custom_enabled_prefixes",
                ("src.core.qa",),
            ),
        ):
            assert filt.filter(record) is False

    def test_custom_mode_passes_matched_prefix(self):
        """In custom mode, DEBUG records from enabled categories pass through."""
        filt = _CategoryFilter()
        record = self._make_record("src.core.qa.retrieval", logging.DEBUG)
        with (
            patch("src.logging_config._get_logging_level", return_value="custom"),
            patch(
                "src.logging_config._custom_enabled_prefixes",
                ("src.core.qa",),
            ),
        ):
            assert filt.filter(record) is True
