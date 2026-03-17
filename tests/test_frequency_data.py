"""
Tests for the Google word frequency data loader.

Covers load_raw_frequency_data() from src.core.vocabulary.frequency_data.
"""

import src.core.vocabulary.frequency_data as freq_module
from src.core.vocabulary.frequency_data import load_raw_frequency_data


class TestLoadRawFrequencyData:
    """Tests for load_raw_frequency_data()."""

    def setup_method(self):
        """Reset the module-level cache before each test."""
        freq_module._raw_frequencies = None

    def teardown_method(self):
        """Reset cache after each test to avoid polluting other test files."""
        freq_module._raw_frequencies = None

    def test_returns_dict(self):
        """Returns a dict."""
        result = load_raw_frequency_data()
        assert isinstance(result, dict)

    def test_values_are_ints(self):
        """All values in the dict should be integers."""
        result = load_raw_frequency_data()
        if result:
            for word, count in list(result.items())[:100]:
                assert isinstance(count, int), f"{word}: {count} is not int"

    def test_keys_are_lowercase(self):
        """All keys should be lowercase strings."""
        result = load_raw_frequency_data()
        if result:
            for word in list(result.keys())[:100]:
                assert word == word.lower(), f"Key '{word}' is not lowercase"

    def test_common_words_present(self):
        """Common English words should be in the dataset."""
        result = load_raw_frequency_data()
        if not result:
            return  # File not available in test env
        for word in ["the", "of", "and", "to", "in"]:
            assert word in result, f"Common word '{word}' missing"

    def test_caching_returns_same_object(self):
        """Second call returns the cached object (same identity)."""
        first = load_raw_frequency_data()
        second = load_raw_frequency_data()
        assert first is second

    def test_file_not_found_returns_empty_dict(self, tmp_path, monkeypatch):
        """Returns empty dict when frequency file doesn't exist."""
        monkeypatch.setattr(
            "src.core.vocabulary.frequency_data.GOOGLE_WORD_FREQUENCY_FILE",
            tmp_path / "nonexistent.tsv",
        )
        result = load_raw_frequency_data()
        assert result == {}

    def test_bad_lines_skipped(self, tmp_path, monkeypatch):
        """Lines with bad format are skipped without error."""
        freq_file = tmp_path / "freq.tsv"
        freq_file.write_text("hello\t100\nbadline\nworld\t200\noops\tnot_a_number\n")
        monkeypatch.setattr(
            "src.core.vocabulary.frequency_data.GOOGLE_WORD_FREQUENCY_FILE",
            freq_file,
        )
        result = load_raw_frequency_data()
        assert result == {"hello": 100, "world": 200}

    def test_thread_safe_loading(self, tmp_path, monkeypatch):
        """Concurrent loads produce the same result without errors."""
        import threading

        freq_file = tmp_path / "freq.tsv"
        freq_file.write_text("test\t42\n")
        monkeypatch.setattr(
            "src.core.vocabulary.frequency_data.GOOGLE_WORD_FREQUENCY_FILE",
            freq_file,
        )

        results = []
        errors = []

        def load():
            """Load frequency data in a thread."""
            try:
                results.append(load_raw_frequency_data())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r == {"test": 42} for r in results)
