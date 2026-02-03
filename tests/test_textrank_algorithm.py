"""
Tests for TextRank keyword extraction algorithm.

Tests cover:
- Shared spaCy model (nlp parameter)
- Fallback when no nlp provided
- Textrank pipe added only once
"""

from unittest.mock import MagicMock


class TestSharedNlp:
    """Test that TextRank can share an NLP instance."""

    def test_shared_nlp_adds_textrank_pipe(self):
        """TextRank adds textrank pipe to shared model if not present."""
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ["ner", "parser"]

        from src.core.vocabulary.algorithms.textrank_algorithm import TextRankAlgorithm

        alg = TextRankAlgorithm(nlp=mock_nlp)

        assert alg._nlp is mock_nlp
        mock_nlp.add_pipe.assert_called_once_with("textrank")

    def test_shared_nlp_skips_if_already_present(self):
        """TextRank does not add textrank pipe if already present."""
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ["ner", "parser", "textrank"]

        from src.core.vocabulary.algorithms.textrank_algorithm import TextRankAlgorithm

        alg = TextRankAlgorithm(nlp=mock_nlp)

        assert alg._nlp is mock_nlp
        mock_nlp.add_pipe.assert_not_called()

    def test_no_nlp_defers_loading(self):
        """Without nlp param, model is not loaded until extract() is called."""
        from src.core.vocabulary.algorithms.textrank_algorithm import TextRankAlgorithm

        alg = TextRankAlgorithm()

        assert alg._nlp is None
