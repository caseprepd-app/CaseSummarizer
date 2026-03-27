"""
Tests for the Markov chain gibberish detection layer.

Verifies that the character bigram model correctly identifies obvious
gibberish without needing the slower spell-checker fallback.
"""

from src.core.utils.gibberish_filter import GibberishFilter, _build_markov_detector


class TestMarkovDetector:
    """Tests for the Markov chain gibberish scorer."""

    def test_builds_successfully(self):
        """Detector builds from NLTK words without error."""
        detector = _build_markov_detector()
        assert detector is not None

    def test_random_chars_flagged(self):
        """Random character strings score above threshold."""
        detector = _build_markov_detector()
        assert detector.is_gibberish("xkjwqr")
        assert detector.is_gibberish("zxcvbn")
        assert detector.is_gibberish("qqqqq")

    def test_real_words_pass(self):
        """Real English words score below threshold."""
        detector = _build_markov_detector()
        assert not detector.is_gibberish("finding")
        assert not detector.is_gibberish("plaintiff")
        assert not detector.is_gibberish("cervical")
        assert not detector.is_gibberish("hemorrhage")

    def test_legal_terms_pass(self):
        """Legal terminology is not flagged."""
        detector = _build_markov_detector()
        assert not detector.is_gibberish("deposition")
        assert not detector.is_gibberish("subpoena")
        assert not detector.is_gibberish("affidavit")


class TestMarkovIntegration:
    """Tests that Markov layer integrates with the full GibberishFilter."""

    def test_markov_loaded_in_filter(self):
        """GibberishFilter initializes with Markov detector."""
        gf = GibberishFilter()
        assert gf._markov is not None

    def test_obvious_gibberish_caught_by_markov(self):
        """Obvious gibberish is caught (Markov layer handles it)."""
        gf = GibberishFilter()
        assert gf.is_gibberish("xkjwqrzm")
        assert gf.is_gibberish("zqxwvpts")

    def test_real_words_still_pass(self):
        """Real words still pass with Markov layer active."""
        gf = GibberishFilter()
        assert not gf.is_gibberish("plaintiff")
        assert not gf.is_gibberish("deposition")
        assert not gf.is_gibberish("hemorrhage")
