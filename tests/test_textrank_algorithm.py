"""
Tests for TextRank keyword extraction algorithm.

Tests cover:
- Shared spaCy model (nlp parameter)
- Fallback when no nlp provided
- Textrank pipe added only once
- extract() returns correct CandidateTerms with original casing
"""

from unittest.mock import MagicMock, patch


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


class TestTextRankExtract:
    """Tests for TextRank extract() method with mocked spaCy model."""

    def _make_mock_phrase(self, text, rank=0.5, count=2):
        """Create a mock pytextrank phrase object.

        Args:
            text: Phrase text
            rank: PageRank score (0-1)
            count: Number of occurrences

        Returns:
            MagicMock mimicking a pytextrank phrase
        """
        phrase = MagicMock()
        phrase.text = text
        phrase.rank = rank
        phrase.count = count
        return phrase

    def _make_algo_with_mock(self, phrases):
        """Create a TextRankAlgorithm with a mocked NLP model.

        Args:
            phrases: List of mock phrase objects for doc._.phrases

        Returns:
            TextRankAlgorithm with injected mock
        """
        from src.core.vocabulary.algorithms.textrank_algorithm import TextRankAlgorithm

        algo = TextRankAlgorithm()
        mock_doc = MagicMock()
        mock_doc._.phrases = phrases
        algo._nlp = MagicMock(return_value=mock_doc)
        return algo

    def test_extract_returns_algorithm_result(self):
        """extract() should return an AlgorithmResult."""
        from src.core.vocabulary.algorithms.base import AlgorithmResult

        algo = self._make_algo_with_mock(
            [
                self._make_mock_phrase("summary judgment"),
            ]
        )
        result = algo.extract("The court granted summary judgment.")
        assert isinstance(result, AlgorithmResult)
        assert result.processing_time_ms >= 0

    def test_extract_preserves_original_casing(self):
        """Terms should preserve the original casing from spaCy, not title-case."""
        algo = self._make_algo_with_mock(
            [
                self._make_mock_phrase("motion in limine", rank=0.8),
                self._make_mock_phrase("summary judgment", rank=0.6),
                self._make_mock_phrase("DNA evidence", rank=0.5),
            ]
        )
        result = algo.extract("Filed motion in limine regarding DNA evidence.")
        terms = [c.term for c in result.candidates]
        assert "motion in limine" in terms, "Should preserve lowercase"
        assert "summary judgment" in terms, "Should preserve lowercase"
        assert "DNA evidence" in terms, "Should preserve mixed case"

    def test_extract_does_not_title_case(self):
        """Verify .title() is NOT applied — lowercase input stays lowercase."""
        algo = self._make_algo_with_mock(
            [
                self._make_mock_phrase("ibuprofen", rank=0.7),
            ]
        )
        result = algo.extract("The patient took ibuprofen.")
        assert result.candidates[0].term == "ibuprofen"
        assert result.candidates[0].term != "Ibuprofen"

    def test_extract_candidate_fields(self):
        """Candidates should have correct algorithm name and bounded confidence."""
        algo = self._make_algo_with_mock(
            [
                self._make_mock_phrase("spinal fusion", rank=0.9, count=3),
            ]
        )
        result = algo.extract("Spinal fusion was performed.")
        c = result.candidates[0]
        assert c.source_algorithm == "TextRank"
        assert c.suggested_type == "Technical"
        assert 0 <= c.confidence <= 1.0
        assert c.frequency >= 1

    def test_extract_skips_single_char(self):
        """Single-character phrases should be filtered out."""
        algo = self._make_algo_with_mock(
            [
                self._make_mock_phrase("x", rank=0.5),
                self._make_mock_phrase("valid term", rank=0.5),
            ]
        )
        result = algo.extract("Some text.")
        terms = [c.term for c in result.candidates]
        assert "x" not in terms
        assert "valid term" in terms

    def test_extract_skips_pure_numbers(self):
        """Pure numeric phrases should be filtered out."""
        algo = self._make_algo_with_mock(
            [
                self._make_mock_phrase("123 456", rank=0.5),
            ]
        )
        result = algo.extract("Numbers only.")
        assert len(result.candidates) == 0

    def test_extract_skips_stopwords(self):
        """Single-word stopwords should be filtered out."""
        algo = self._make_algo_with_mock(
            [
                self._make_mock_phrase("the", rank=0.5),
                self._make_mock_phrase("also", rank=0.5),
            ]
        )
        result = algo.extract("Some text.")
        terms = [c.term for c in result.candidates]
        assert "the" not in terms

    def test_extract_deduplicates(self):
        """Duplicate phrases (case-insensitive) should be deduplicated."""
        algo = self._make_algo_with_mock(
            [
                self._make_mock_phrase("Summary Judgment", rank=0.8),
                self._make_mock_phrase("summary judgment", rank=0.6),
            ]
        )
        result = algo.extract("Summary judgment motion.")
        assert len(result.candidates) == 1

    def test_max_candidates_respected(self):
        """Should not return more candidates than max_candidates."""
        phrases = [self._make_mock_phrase(f"term {i}", rank=0.5) for i in range(20)]
        algo = self._make_algo_with_mock(phrases)
        algo.max_candidates = 5
        result = algo.extract("Lots of terms.")
        assert len(result.candidates) <= 5

    def test_extract_graceful_when_model_missing(self):
        """Should return empty result when spaCy model is not installed."""
        from src.core.vocabulary.algorithms.textrank_algorithm import TextRankAlgorithm

        algo = TextRankAlgorithm()
        with patch("spacy.load", side_effect=OSError("Model not found")):
            result = algo.extract("Some text.")
            assert result.candidates == []
            assert result.metadata.get("skipped") is True

    def test_metadata_fields(self):
        """Result metadata should contain expected keys."""
        algo = self._make_algo_with_mock(
            [
                self._make_mock_phrase("legal term", rank=0.5),
            ]
        )
        result = algo.extract("Legal term appeared.")
        assert "total_phrases_found" in result.metadata
        assert "filtered_candidates" in result.metadata
        assert "text_truncated" in result.metadata
