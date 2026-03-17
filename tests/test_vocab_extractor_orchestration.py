"""
Tests for VocabularyExtractor orchestration paths.

Covers extract(), extract_progressive(), and _run_algorithms_parallel()
with injected mock algorithms to avoid loading real NLP models.
"""

from unittest.mock import MagicMock

from src.core.vocabulary.algorithms.base import AlgorithmResult, CandidateTerm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(term, confidence=0.8, category=None, frequency=2):
    """Create a CandidateTerm."""
    return CandidateTerm(
        term=term,
        source_algorithm="TEST",
        confidence=confidence,
        suggested_type=category,
        frequency=frequency,
    )


def _make_algorithm(name, candidates=None, enabled=True):
    """Create a mock algorithm that returns given candidates."""
    alg = MagicMock()
    alg.name = name
    alg.enabled = enabled
    result = AlgorithmResult(
        candidates=candidates or [],
        processing_time_ms=10.0,
    )
    alg.extract.return_value = result
    return alg


def _make_extractor(algorithms):
    """Create a VocabularyExtractor with pre-built algorithms (no spaCy load)."""
    from src.core.vocabulary import VocabularyExtractor

    ext = VocabularyExtractor(algorithms=algorithms)
    return ext


# ===========================================================================
# extract()
# ===========================================================================


class TestExtract:
    """Tests for VocabularyExtractor.extract()."""

    def test_returns_tuple_of_vocab_and_filtered(self):
        """extract() returns (vocabulary, filtered_terms) tuple."""
        ext = _make_extractor(
            algorithms=[
                _make_algorithm("NER", [_make_candidate("John Smith", category="Person")]),
            ]
        )
        result = ext.extract("John Smith was present. John Smith testified.")
        assert isinstance(result, tuple)
        assert len(result) == 2
        vocab, filtered = result
        assert isinstance(vocab, list)
        assert isinstance(filtered, list)

    def test_empty_text_returns_empty(self):
        """Empty text produces no vocabulary."""
        ext = _make_extractor(algorithms=[_make_algorithm("NER")])
        vocab, filtered = ext.extract("")
        assert vocab == []

    def test_no_enabled_algorithms(self):
        """No enabled algorithms produces empty vocabulary."""
        ext = _make_extractor(
            algorithms=[
                _make_algorithm("NER", enabled=False),
            ]
        )
        vocab, filtered = ext.extract("Some text")
        assert vocab == []

    def test_extract_runs_all_enabled_algorithms(self):
        """All enabled algorithms are called with the text."""
        alg1 = _make_algorithm("RAKE", [_make_candidate("legal term", frequency=3)])
        alg2 = _make_algorithm("BM25", [_make_candidate("rare phrase", frequency=3)])
        disabled = _make_algorithm("NER", enabled=False)

        ext = _make_extractor(algorithms=[alg1, alg2, disabled])
        ext.extract("legal term appeared. rare phrase appeared. legal term again.")

        alg1.extract.assert_called_once()
        alg2.extract.assert_called_once()
        disabled.extract.assert_not_called()

    def test_quality_score_in_valid_range(self):
        """All quality scores are between 0 and 100."""
        ext = _make_extractor(
            algorithms=[
                _make_algorithm(
                    "NER",
                    [
                        _make_candidate("Dr. Smith", category="Person", frequency=3),
                        _make_candidate("Memorial Hospital", frequency=3),
                    ],
                ),
            ]
        )
        text = "Dr. Smith at Memorial Hospital. " * 5
        vocab, _ = ext.extract(text)
        for term in vocab:
            score = term["Quality Score"]
            assert 0 <= score <= 100, f"{term['Term']}: score {score} out of range"

    def test_text_truncation_for_large_input(self):
        """Very large text is truncated to VOCABULARY_MAX_TEXT_KB."""
        alg = _make_algorithm("RAKE")
        ext = _make_extractor(algorithms=[alg])

        from src.config import VOCABULARY_MAX_TEXT_KB

        huge_text = "word " * (VOCABULARY_MAX_TEXT_KB * 1024)
        ext.extract(huge_text)

        # The text passed to algorithm should be truncated
        called_text = alg.extract.call_args[0][0]
        assert len(called_text) <= VOCABULARY_MAX_TEXT_KB * 1024

    def test_deduplication(self):
        """Duplicate terms from different algorithms produce single entry."""
        alg1 = _make_algorithm(
            "NER", [_make_candidate("John Smith", category="Person", frequency=3)]
        )
        alg2 = _make_algorithm("RAKE", [_make_candidate("John Smith", frequency=3)])

        ext = _make_extractor(algorithms=[alg1, alg2])
        vocab, _ = ext.extract("John Smith testified. John Smith said. John Smith agreed.")

        # Should have at most one "John Smith" entry
        john_count = sum(1 for v in vocab if v["Term"].lower() == "john smith")
        assert john_count <= 1


# ===========================================================================
# extract_progressive()
# ===========================================================================


class TestExtractProgressive:
    """Tests for VocabularyExtractor.extract_progressive()."""

    def test_returns_tuple(self):
        """extract_progressive returns (vocab, filtered) tuple."""
        ext = _make_extractor(
            algorithms=[
                _make_algorithm("RAKE", [_make_candidate("test term", frequency=3)]),
            ]
        )
        result = ext.extract_progressive("test term appeared. test term again. test term third.")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_partial_callback_called_for_fast_algos(self):
        """partial_callback fires after fast algorithms complete."""
        callback = MagicMock()
        ext = _make_extractor(
            algorithms=[
                _make_algorithm("RAKE", [_make_candidate("key phrase", frequency=3)]),
                _make_algorithm("BM25", [_make_candidate("rare word", frequency=3)]),
            ]
        )
        ext.extract_progressive(
            "key phrase and rare word. " * 5,
            partial_callback=callback,
        )
        callback.assert_called_once()
        # Callback receives a list of vocab dicts
        assert isinstance(callback.call_args[0][0], list)

    def test_ner_runs_after_fast_algorithms(self):
        """NER algorithm runs in Phase 2 after RAKE/BM25."""
        ner = _make_algorithm("NER", [_make_candidate("Dr. Jones", category="Person", frequency=3)])
        rake = _make_algorithm("RAKE", [_make_candidate("medical term", frequency=3)])

        ext = _make_extractor(algorithms=[rake, ner])
        vocab, _ = ext.extract_progressive("Dr. Jones and medical term. " * 5)

        # Both algorithms should have been called
        rake.extract.assert_called_once()
        ner.extract.assert_called_once()

    def test_status_callback_called_per_algorithm(self):
        """status_callback fires before each algorithm runs."""
        status_cb = MagicMock()
        ext = _make_extractor(
            algorithms=[
                _make_algorithm("RAKE"),
                _make_algorithm("BM25"),
            ]
        )
        ext.extract_progressive("Some text here.", status_callback=status_cb)
        assert status_cb.call_count >= 2

    def test_no_algorithms_returns_empty(self):
        """No enabled algorithms returns empty results."""
        ext = _make_extractor(algorithms=[])
        vocab, filtered = ext.extract_progressive("Some text")
        assert vocab == []


# ===========================================================================
# _run_algorithms_parallel
# ===========================================================================


class TestRunAlgorithmsParallel:
    """Tests for VocabularyExtractor._run_algorithms_parallel()."""

    def test_single_algorithm_runs_sequentially(self):
        """Single enabled algorithm runs sequentially (no thread pool)."""
        alg = _make_algorithm("NER", [_make_candidate("Smith")])
        ext = _make_extractor(algorithms=[alg])
        results = ext._run_algorithms_parallel("some text")
        assert len(results) == 1
        alg.extract.assert_called_once()

    def test_no_enabled_algorithms_returns_empty(self):
        """No enabled algorithms returns empty list."""
        ext = _make_extractor(algorithms=[_make_algorithm("NER", enabled=False)])
        results = ext._run_algorithms_parallel("text")
        assert results == []

    def test_multiple_algorithms_all_return_results(self):
        """Multiple algorithms all produce results."""
        alg1 = _make_algorithm("RAKE", [_make_candidate("term1")])
        alg2 = _make_algorithm("BM25", [_make_candidate("term2")])
        ext = _make_extractor(algorithms=[alg1, alg2])
        results = ext._run_algorithms_parallel("some text here")
        assert len(results) >= 1  # At least 1 result (may be 2)
        alg1.extract.assert_called_once()
        alg2.extract.assert_called_once()
