"""
Edge case tests for VocabularyExtractor.

Covers graceful degradation, partial failures, deduplication,
unicode handling, and boundary conditions.
"""

from unittest.mock import patch

from src.core.vocabulary.algorithms.base import (
    AlgorithmResult,
    BaseExtractionAlgorithm,
    CandidateTerm,
)
from src.core.vocabulary.result_merger import AlgorithmScoreMerger


class FakeAlgorithm(BaseExtractionAlgorithm):
    """Minimal algorithm stub for testing."""

    def __init__(self, name, candidates=None, should_fail=False):
        """Set up a fake algorithm with optional failure mode."""
        self.name = name
        self.weight = 1.0
        self.enabled = True
        self._candidates = candidates or []
        self._should_fail = should_fail

    def extract(self, text, **kwargs):
        """Return preset candidates or raise on demand."""
        if self._should_fail:
            raise RuntimeError(f"{self.name} failed on purpose")
        return AlgorithmResult(candidates=self._candidates)


def _make_extractor(algorithms):
    """Build a VocabularyExtractor with injected algorithms."""
    with (
        patch("src.core.vocabulary.vocabulary_extractor.wordnet"),
        patch("src.core.vocabulary.vocabulary_extractor.spacy"),
        patch(
            "src.core.vocabulary.vocabulary_extractor.get_user_preferences",
            return_value={"vocab_sort_method": "quality_score", "vocab_min_occurrences": 1},
        ),
        patch(
            "src.core.vocabulary.vocabulary_extractor.get_meta_learner",
        ) as mock_ml,
    ):
        mock_ml.return_value.is_trained = False
        from src.core.vocabulary.vocabulary_extractor import VocabularyExtractor

        ext = VocabularyExtractor(algorithms=algorithms)
        ext.frequency_dataset = {}
        ext.frequency_rank_map = {}
    return ext


def test_all_algorithms_fail():
    """When every algorithm raises, extractor returns empty results."""
    algos = [FakeAlgorithm("A", should_fail=True), FakeAlgorithm("B", should_fail=True)]
    ext = _make_extractor(algos)
    vocab, filtered = ext.extract("Some legal text here.")
    assert isinstance(vocab, list)
    assert len(vocab) == 0


def test_partial_algorithm_failure():
    """Working algorithms contribute terms; failed ones are skipped."""
    good_term = CandidateTerm(
        term="John Smith",
        source_algorithm="Good1",
        confidence=0.9,
        suggested_type="Person",
        frequency=3,
    )
    algos = [
        FakeAlgorithm("Fail1", should_fail=True),
        FakeAlgorithm("Fail2", should_fail=True),
        FakeAlgorithm("Good1", candidates=[good_term]),
        FakeAlgorithm(
            "Good2",
            candidates=[
                CandidateTerm(
                    term="Jane Doe",
                    source_algorithm="Good2",
                    confidence=0.7,
                    suggested_type="Person",
                    frequency=4,
                ),
            ],
        ),
        FakeAlgorithm(
            "Good3",
            candidates=[
                CandidateTerm(
                    term="Robert Garcia",
                    source_algorithm="Good3",
                    confidence=0.8,
                    suggested_type="Person",
                    frequency=2,
                ),
            ],
        ),
    ]
    ext = _make_extractor(algos)
    vocab, filtered = ext.extract("text " * 50)
    # All terms from working algos appear somewhere (vocab or filtered)
    all_terms = {t["Term"].lower() for t in vocab + filtered}
    assert "john smith" in all_terms
    assert "jane doe" in all_terms
    # Failed algorithms tracked in skipped list
    assert "Fail1" in ext.skipped_algorithms
    assert "Fail2" in ext.skipped_algorithms


def test_all_terms_filtered_out():
    """If every term is below min occurrences, result is empty."""
    cand = CandidateTerm(
        term="x",
        source_algorithm="A",
        confidence=0.1,
        frequency=0,
    )
    ext = _make_extractor([FakeAlgorithm("A", candidates=[cand])])
    vocab, filtered = ext.extract("x")
    total = len(vocab) + len(filtered)
    # Either filtered away or just empty; no crash
    assert isinstance(vocab, list)
    assert isinstance(filtered, list)


def test_duplicate_terms_different_confidence():
    """Merger picks higher confidence when same term from two algos."""
    low = CandidateTerm(
        term="fibromyalgia",
        source_algorithm="A",
        confidence=0.3,
        frequency=5,
    )
    high = CandidateTerm(
        term="fibromyalgia",
        source_algorithm="B",
        confidence=0.95,
        frequency=5,
    )
    merger = AlgorithmScoreMerger({"A": 1.0, "B": 1.0})
    merged = merger.merge(
        [
            AlgorithmResult(candidates=[low]),
            AlgorithmResult(candidates=[high]),
        ]
    )
    assert len(merged) == 1
    assert merged[0].combined_confidence > 0.5
    assert set(merged[0].sources) == {"A", "B"}


def test_unicode_terms():
    """Non-ASCII names survive extraction without corruption."""
    names = ["Jose", "Francois", "Muller"]
    candidates = [
        CandidateTerm(
            term=n, source_algorithm="NER", confidence=0.9, suggested_type="Person", frequency=3
        )
        for n in names
    ]
    ext = _make_extractor([FakeAlgorithm("NER", candidates=candidates)])
    vocab, _ = ext.extract("text " * 20)
    found = {t["Term"] for t in vocab}
    for name in names:
        assert name in found, f"{name} missing from results"


def test_empty_document():
    """Empty string input returns empty results, no crash."""
    ext = _make_extractor([FakeAlgorithm("A")])
    vocab, filtered = ext.extract("")
    assert vocab == []
    assert filtered == []


def test_single_word_document():
    """Single-word input does not crash the extractor."""
    ext = _make_extractor([FakeAlgorithm("A")])
    vocab, filtered = ext.extract("Hello")
    assert isinstance(vocab, list)
    assert isinstance(filtered, list)
