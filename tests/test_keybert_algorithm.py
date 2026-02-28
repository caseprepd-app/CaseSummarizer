"""
Tests for KeyBERT keyword extraction algorithm.

Tests cover:
- Registration and metadata (name, weight)
- extract() returns AlgorithmResult with CandidateTerms
- Metadata fields (keybert_score, word_count)
- Empty text returns empty candidates
- Deduplication and filtering
- Graceful fallback when model unavailable
"""

from unittest.mock import MagicMock, patch


class TestKeyBERTRegistration:
    """Test algorithm registration and configuration."""

    def test_registered_name(self):
        """KeyBERT should be registered under 'KeyBERT'."""
        from src.core.vocabulary.algorithms import get_available_algorithms
        from src.core.vocabulary.algorithms.keybert_algorithm import KeyBERTAlgorithm  # noqa: F401

        assert "KeyBERT" in get_available_algorithms()

    def test_algorithm_name(self):
        """Algorithm name property should be 'KeyBERT'."""
        from src.core.vocabulary.algorithms.keybert_algorithm import KeyBERTAlgorithm

        algo = KeyBERTAlgorithm()
        assert algo.name == "KeyBERT"

    def test_default_weight(self):
        """Default weight should be 0.65."""
        from src.core.vocabulary.algorithms.keybert_algorithm import KeyBERTAlgorithm

        algo = KeyBERTAlgorithm()
        assert algo.weight == 0.65

    def test_get_config(self):
        """get_config() should return algorithm settings."""
        from src.core.vocabulary.algorithms.keybert_algorithm import KeyBERTAlgorithm

        algo = KeyBERTAlgorithm()
        config = algo.get_config()
        assert config["top_n"] == 150
        assert config["ngram_range"] == (1, 3)
        assert config["diversity"] == 0.5


class TestKeyBERTExtract:
    """Tests for KeyBERT extract() method."""

    def _make_algo_with_mock(self, keywords):
        """Create KeyBERTAlgorithm with a mocked model.

        Args:
            keywords: List of (keyphrase, score) tuples

        Returns:
            KeyBERTAlgorithm with injected mock model
        """
        from src.core.vocabulary.algorithms.keybert_algorithm import KeyBERTAlgorithm

        algo = KeyBERTAlgorithm()
        mock_model = MagicMock()
        mock_model.extract_keywords.return_value = keywords
        algo._model = mock_model
        return algo

    def test_extract_returns_algorithm_result(self):
        """extract() should return an AlgorithmResult."""
        from src.core.vocabulary.algorithms.base import AlgorithmResult

        algo = self._make_algo_with_mock([("summary judgment", 0.85)])
        result = algo.extract("The court granted summary judgment.")
        assert isinstance(result, AlgorithmResult)
        assert result.processing_time_ms >= 0

    def test_confidence_is_cosine_similarity(self):
        """KeyBERT score (cosine similarity) should be used as confidence."""
        algo = self._make_algo_with_mock(
            [
                ("high relevance", 0.95),
                ("low relevance", 0.20),
            ]
        )
        result = algo.extract("Some text about relevance.")
        candidates = {c.term: c.confidence for c in result.candidates}

        assert candidates["high relevance"] == 0.95
        assert candidates["low relevance"] == 0.20

    def test_confidence_clamped(self):
        """Confidence should be clamped to [0, 1]."""
        algo = self._make_algo_with_mock(
            [
                ("over one", 1.5),
                ("negative", -0.1),
            ]
        )
        result = algo.extract("Some text.")
        candidates = {c.term: c.confidence for c in result.candidates}

        assert candidates["over one"] == 1.0
        assert candidates["negative"] == 0.0

    def test_metadata_fields(self):
        """Candidates should have keybert_score and word_count in metadata."""
        algo = self._make_algo_with_mock([("spinal fusion", 0.82)])
        result = algo.extract("Spinal fusion was performed.")
        c = result.candidates[0]
        assert "keybert_score" in c.metadata
        assert c.metadata["keybert_score"] == 0.82
        assert c.metadata["word_count"] == 2

    def test_candidate_fields(self):
        """Candidates should have correct source_algorithm and type."""
        algo = self._make_algo_with_mock([("legal term", 0.7)])
        result = algo.extract("A legal term appeared.")
        c = result.candidates[0]
        assert c.source_algorithm == "KeyBERT"
        assert c.suggested_type == "Technical"

    def test_empty_text_returns_empty(self):
        """Empty text should return empty candidates."""
        from src.core.vocabulary.algorithms.keybert_algorithm import KeyBERTAlgorithm

        algo = KeyBERTAlgorithm()
        result = algo.extract("")
        assert result.candidates == []
        assert result.metadata.get("skipped") is True

    def test_whitespace_text_returns_empty(self):
        """Whitespace-only text should return empty candidates."""
        from src.core.vocabulary.algorithms.keybert_algorithm import KeyBERTAlgorithm

        algo = KeyBERTAlgorithm()
        result = algo.extract("   \n\t  ")
        assert result.candidates == []

    def test_deduplicates_case_insensitive(self):
        """Duplicate phrases (case-insensitive) should be deduplicated."""
        algo = self._make_algo_with_mock(
            [
                ("Summary Judgment", 0.85),
                ("summary judgment", 0.80),
            ]
        )
        result = algo.extract("Summary judgment motion.")
        assert len(result.candidates) == 1

    def test_skips_pure_numbers(self):
        """Pure numeric keyphrases should be filtered out."""
        algo = self._make_algo_with_mock(
            [
                ("123", 0.5),
                ("valid term", 0.8),
            ]
        )
        result = algo.extract("Some text.")
        terms = [c.term for c in result.candidates]
        assert "123" not in terms
        assert "valid term" in terms

    def test_skips_single_char(self):
        """Single-character keyphrases should be filtered out."""
        algo = self._make_algo_with_mock(
            [
                ("x", 0.5),
                ("valid term", 0.8),
            ]
        )
        result = algo.extract("Some text.")
        terms = [c.term for c in result.candidates]
        assert "x" not in terms

    def test_graceful_when_model_unavailable(self):
        """Should return empty result when model fails to load."""
        from src.core.vocabulary.algorithms.keybert_algorithm import KeyBERTAlgorithm

        algo = KeyBERTAlgorithm()
        with patch.object(algo, "_load_model", side_effect=RuntimeError("No model")):
            result = algo.extract("Some text.")
            assert result.candidates == []
            assert result.metadata.get("skipped") is True

    def test_result_metadata(self):
        """Result metadata should contain expected keys."""
        algo = self._make_algo_with_mock([("test", 0.8)])
        result = algo.extract("Test text.")
        assert "raw_keywords_found" in result.metadata
        assert "filtered_candidates" in result.metadata
        assert "text_truncated" in result.metadata

    def test_mmr_params_passed(self):
        """extract_keywords should be called with MMR parameters."""
        algo = self._make_algo_with_mock([])
        algo.extract("Test text.")
        call_kwargs = algo._model.extract_keywords.call_args[1]
        assert call_kwargs["use_mmr"] is True
        assert call_kwargs["diversity"] == 0.5
        assert call_kwargs["keyphrase_ngram_range"] == (1, 3)
