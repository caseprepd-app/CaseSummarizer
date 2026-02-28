"""
Tests for YAKE keyword extraction algorithm.

Tests cover:
- Registration and metadata (name, weight)
- extract() returns AlgorithmResult with CandidateTerms
- Score inversion (confidence = 1 / (1 + yake_score))
- Metadata fields (yake_score, word_count)
- Empty text returns empty candidates
- Deduplication and filtering
"""

from unittest.mock import MagicMock


class TestYAKERegistration:
    """Test algorithm registration and configuration."""

    def test_registered_name(self):
        """YAKE should be registered under 'YAKE'."""
        from src.core.vocabulary.algorithms import get_available_algorithms
        from src.core.vocabulary.algorithms.yake_algorithm import YAKEAlgorithm  # noqa: F401

        assert "YAKE" in get_available_algorithms()

    def test_algorithm_name(self):
        """Algorithm name property should be 'YAKE'."""
        from src.core.vocabulary.algorithms.yake_algorithm import YAKEAlgorithm

        algo = YAKEAlgorithm()
        assert algo.name == "YAKE"

    def test_default_weight(self):
        """Default weight should be 0.55."""
        from src.core.vocabulary.algorithms.yake_algorithm import YAKEAlgorithm

        algo = YAKEAlgorithm()
        assert algo.weight == 0.55

    def test_get_config(self):
        """get_config() should return algorithm settings."""
        from src.core.vocabulary.algorithms.yake_algorithm import YAKEAlgorithm

        algo = YAKEAlgorithm()
        config = algo.get_config()
        assert config["max_ngram_size"] == 3
        assert config["max_candidates"] == 150


class TestYAKEExtract:
    """Tests for YAKE extract() method."""

    def _make_algo_with_mock(self, keywords):
        """Create YAKEAlgorithm with a mocked extractor.

        Args:
            keywords: List of (keyphrase, score) tuples

        Returns:
            YAKEAlgorithm with injected mock extractor
        """
        from src.core.vocabulary.algorithms.yake_algorithm import YAKEAlgorithm

        algo = YAKEAlgorithm()
        mock_extractor = MagicMock()
        mock_extractor.extract_keywords.return_value = keywords
        algo._extractor = mock_extractor
        return algo

    def test_extract_returns_algorithm_result(self):
        """extract() should return an AlgorithmResult."""
        from src.core.vocabulary.algorithms.base import AlgorithmResult

        algo = self._make_algo_with_mock([("summary judgment", 0.05)])
        result = algo.extract("The court granted summary judgment.")
        assert isinstance(result, AlgorithmResult)
        assert result.processing_time_ms >= 0

    def test_score_inversion(self):
        """YAKE score 0.0 should map to confidence 1.0, large scores to near 0."""
        algo = self._make_algo_with_mock(
            [
                ("perfect keyword", 0.0),
                ("good keyword", 1.0),
                ("weak keyword", 9.0),
            ]
        )
        result = algo.extract("Some text with keywords.")
        candidates = {c.term: c.confidence for c in result.candidates}

        # confidence = 1 / (1 + score)
        assert candidates["perfect keyword"] == 1.0  # 1/(1+0) = 1.0
        assert candidates["good keyword"] == 0.5  # 1/(1+1) = 0.5
        assert candidates["weak keyword"] == 0.1  # 1/(1+9) = 0.1

    def test_metadata_fields(self):
        """Candidates should have yake_score and word_count in metadata."""
        algo = self._make_algo_with_mock([("spinal fusion", 0.02)])
        result = algo.extract("Spinal fusion was performed.")
        c = result.candidates[0]
        assert "yake_score" in c.metadata
        assert c.metadata["yake_score"] == 0.02
        assert c.metadata["word_count"] == 2

    def test_candidate_fields(self):
        """Candidates should have correct source_algorithm and type."""
        algo = self._make_algo_with_mock([("legal term", 0.1)])
        result = algo.extract("A legal term appeared.")
        c = result.candidates[0]
        assert c.source_algorithm == "YAKE"
        assert c.suggested_type == "Technical"

    def test_empty_text_returns_empty(self):
        """Empty text should return empty candidates."""
        from src.core.vocabulary.algorithms.yake_algorithm import YAKEAlgorithm

        algo = YAKEAlgorithm()
        result = algo.extract("")
        assert result.candidates == []
        assert result.metadata.get("skipped") is True

    def test_whitespace_text_returns_empty(self):
        """Whitespace-only text should return empty candidates."""
        from src.core.vocabulary.algorithms.yake_algorithm import YAKEAlgorithm

        algo = YAKEAlgorithm()
        result = algo.extract("   \n\t  ")
        assert result.candidates == []

    def test_deduplicates_case_insensitive(self):
        """Duplicate phrases (case-insensitive) should be deduplicated."""
        algo = self._make_algo_with_mock(
            [
                ("Summary Judgment", 0.05),
                ("summary judgment", 0.08),
            ]
        )
        result = algo.extract("Summary judgment motion.")
        assert len(result.candidates) == 1

    def test_skips_pure_numbers(self):
        """Pure numeric keyphrases should be filtered out."""
        algo = self._make_algo_with_mock(
            [
                ("123", 0.5),
                ("valid term", 0.1),
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
                ("valid term", 0.1),
            ]
        )
        result = algo.extract("Some text.")
        terms = [c.term for c in result.candidates]
        assert "x" not in terms

    def test_max_candidates_respected(self):
        """Should not return more candidates than max_candidates."""
        keywords = [(f"term {i}", 0.1 * i) for i in range(20)]
        algo = self._make_algo_with_mock(keywords)
        algo.max_candidates = 5
        result = algo.extract("Lots of terms.")
        assert len(result.candidates) <= 5

    def test_result_metadata(self):
        """Result metadata should contain expected keys."""
        algo = self._make_algo_with_mock([("test", 0.1)])
        result = algo.extract("Test text.")
        assert "raw_keywords_found" in result.metadata
        assert "filtered_candidates" in result.metadata
        assert "text_truncated" in result.metadata
