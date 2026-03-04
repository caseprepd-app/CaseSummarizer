"""Tests for session stats display including word count."""

from unittest.mock import MagicMock

from src.ui.main_window import MainWindow


def _make_stub():
    """Create a MainWindow stub for _update_session_stats."""
    stub = MagicMock()
    stub.stats_label = MagicMock()
    stub.processing_results = []
    return stub


def _result(filename="doc.pdf", text="", pages=5, size=1024):
    """Build a processing result dict with optional extracted text."""
    return {
        "filename": filename,
        "status": "success",
        "page_count": pages,
        "file_size": size,
        "extracted_text": text,
    }


class TestSessionStatsWordCount:
    """Tests for word count in _update_session_stats."""

    def test_word_count_shown_when_text_present(self):
        """Word count should appear in stats when documents have text."""
        stub = _make_stub()
        stub.processing_results = [
            _result(text="The quick brown fox jumps over the lazy dog"),  # 9 words
        ]
        MainWindow._update_session_stats(stub)
        text = stub.stats_label.configure.call_args[1]["text"]
        assert "9 words" in text

    def test_word_count_sums_across_documents(self):
        """Word count should sum across all documents."""
        stub = _make_stub()
        stub.processing_results = [
            _result(filename="a.pdf", text="one two three"),  # 3 words
            _result(filename="b.pdf", text="four five six seven"),  # 4 words
        ]
        MainWindow._update_session_stats(stub)
        text = stub.stats_label.configure.call_args[1]["text"]
        assert "7 words" in text

    def test_word_count_uses_preprocessed_text_when_available(self):
        """Should prefer preprocessed_text over extracted_text."""
        stub = _make_stub()
        result = _result(text="raw long text with many extra words here")
        result["preprocessed_text"] = "cleaned short"  # 2 words
        stub.processing_results = [result]
        MainWindow._update_session_stats(stub)
        text = stub.stats_label.configure.call_args[1]["text"]
        assert "2 words" in text

    def test_word_count_comma_formatted_for_large_counts(self):
        """Word counts >= 1000 should use comma formatting."""
        stub = _make_stub()
        # Generate text with 1500 words
        stub.processing_results = [
            _result(text=" ".join(["word"] * 1500)),
        ]
        MainWindow._update_session_stats(stub)
        text = stub.stats_label.configure.call_args[1]["text"]
        assert "1,500 words" in text

    def test_word_count_omitted_when_zero(self):
        """Word count should not appear when no text extracted."""
        stub = _make_stub()
        stub.processing_results = [_result(text="")]
        MainWindow._update_session_stats(stub)
        text = stub.stats_label.configure.call_args[1]["text"]
        assert "words" not in text

    def test_empty_results_clears_label(self):
        """Empty processing_results should clear the stats label."""
        stub = _make_stub()
        stub.processing_results = []
        MainWindow._update_session_stats(stub)
        stub.stats_label.configure.assert_called_with(text="")

    def test_stats_include_files_pages_size_and_words(self):
        """Stats line should contain all four components."""
        stub = _make_stub()
        stub.processing_results = [
            _result(text="hello world", pages=10, size=2 * 1024 * 1024),
        ]
        MainWindow._update_session_stats(stub)
        text = stub.stats_label.configure.call_args[1]["text"]
        assert "1/1 files" in text
        assert "10 pages" in text
        assert "MB" in text
        assert "2 words" in text
