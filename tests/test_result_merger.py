from src.core.vocabulary.algorithms.base import AlgorithmResult, CandidateTerm
from src.core.vocabulary.result_merger import AlgorithmScoreMerger


def test_merge_single_algorithm():
    candidates = [
        CandidateTerm(
            term="radiculopathy",
            source_algorithm="NER",
            confidence=0.9,
            suggested_type="Medical",
            frequency=3,
        )
    ]
    result = AlgorithmResult(candidates=candidates)
    merger = AlgorithmScoreMerger()
    merged = merger.merge([result])
    assert len(merged) == 1
    assert merged[0].term == "radiculopathy"
    assert merged[0].sources == ["NER"]
    assert merged[0].frequency == 3


def test_merge_same_term_multiple_algorithms():
    ner = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="John Smith",
                source_algorithm="NER",
                confidence=0.9,
                suggested_type="Person",
                frequency=5,
            )
        ]
    )
    rake = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="john smith",
                source_algorithm="RAKE",
                confidence=0.7,
                suggested_type=None,
                frequency=3,
            )
        ]
    )
    merger = AlgorithmScoreMerger()
    merged = merger.merge([ner, rake])
    assert len(merged) == 1
    assert set(merged[0].sources) == {"NER", "RAKE"}
    assert merged[0].frequency == 8


def test_type_resolution_ner_wins():
    ner = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="Smith",
                source_algorithm="NER",
                confidence=0.9,
                suggested_type="Person",
                frequency=1,
            )
        ]
    )
    rake = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="smith",
                source_algorithm="RAKE",
                confidence=0.7,
                suggested_type="Technical",
                frequency=1,
            )
        ]
    )
    merger = AlgorithmScoreMerger()
    merged = merger.merge([ner, rake])
    assert merged[0].final_type == "Person"


def test_weighted_confidence():
    ner = AlgorithmResult(
        candidates=[CandidateTerm(term="test", source_algorithm="NER", confidence=0.9, frequency=1)]
    )
    rake = AlgorithmResult(
        candidates=[
            CandidateTerm(term="test", source_algorithm="RAKE", confidence=0.5, frequency=1)
        ]
    )
    merger = AlgorithmScoreMerger(algorithm_weights={"NER": 1.0, "RAKE": 0.5})
    merged = merger.merge([ner, rake])
    # weighted: (0.9*1.0 + 0.5*0.5) / (1.0 + 0.5) = 1.15/1.5 ~ 0.767
    assert abs(merged[0].combined_confidence - 0.767) < 0.01


def test_empty_results():
    merger = AlgorithmScoreMerger()
    merged = merger.merge([])
    assert merged == []


def test_canonical_casing_person_title_case():
    # Person entities should be title cased
    ner = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="JAMES LUCAS",
                source_algorithm="NER",
                confidence=0.9,
                suggested_type="Person",
                frequency=2,
            )
        ]
    )
    merger = AlgorithmScoreMerger()
    merged = merger.merge([ner])
    assert merged[0].term == "James Lucas"  # Title cased for Person


# === Possessive normalization tests (Session 82) ===


def test_normalize_for_merge_strips_possessive_s():
    """_normalize_for_merge should strip 's suffix."""
    from src.core.vocabulary.result_merger import _normalize_for_merge

    assert _normalize_for_merge("Sturman's") == "sturman"
    assert _normalize_for_merge("sturman's") == "sturman"
    assert _normalize_for_merge("Jones's") == "jones"


def test_normalize_for_merge_strips_trailing_apostrophe():
    """_normalize_for_merge should strip trailing ' after s (e.g., Morales')."""
    from src.core.vocabulary.result_merger import _normalize_for_merge

    assert _normalize_for_merge("Morales'") == "morales"
    assert _normalize_for_merge("Jones'") == "jones"


def test_normalize_for_merge_preserves_non_possessive():
    """_normalize_for_merge should not strip non-possessive apostrophes."""
    from src.core.vocabulary.result_merger import _normalize_for_merge

    assert _normalize_for_merge("O'Brien") == "o'brien"
    assert _normalize_for_merge("normal") == "normal"


def test_merge_possessive_with_base_form():
    """Merger should group 'Sturman's' with 'Sturman' as the same term."""
    ner = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="Sturman",
                source_algorithm="NER",
                confidence=0.9,
                suggested_type="Person",
                frequency=5,
            )
        ]
    )
    bm25 = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="Sturman's",
                source_algorithm="BM25",
                confidence=0.7,
                suggested_type="Technical",
                frequency=3,
            )
        ]
    )
    merger = AlgorithmScoreMerger()
    merged = merger.merge([ner, bm25])

    # Should merge into a single term
    assert len(merged) == 1
    # Frequency should be combined
    assert merged[0].frequency == 8
    # Both sources should be tracked
    assert set(merged[0].sources) == {"NER", "BM25"}


# === Algorithm score promotion tests ===


def test_algorithm_scores_promoted_to_top_level():
    """Merger should promote algorithm scores from source_details to top-level metadata."""
    topicrank = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="radiculopathy",
                source_algorithm="TopicRank",
                confidence=0.8,
                frequency=3,
                metadata={"topicrank_score": 0.45},
            )
        ]
    )
    yake = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="radiculopathy",
                source_algorithm="YAKE",
                confidence=0.7,
                frequency=2,
                metadata={"yake_score": 0.12},
            )
        ]
    )
    merger = AlgorithmScoreMerger()
    merged = merger.merge([topicrank, yake])
    assert len(merged) == 1
    # Scores should be at top level, not buried in source_details
    assert merged[0].metadata.get("topicrank_score") == 0.45
    assert merged[0].metadata.get("yake_score") == 0.12


def test_best_score_wins_when_multiple_candidates():
    """When multiple candidates have the same score key, max should win."""
    rake1 = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="legal term",
                source_algorithm="RAKE",
                confidence=0.6,
                frequency=2,
                metadata={"rake_score": 5.0},
            )
        ]
    )
    rake2 = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="Legal Term",
                source_algorithm="RAKE",
                confidence=0.8,
                frequency=3,
                metadata={"rake_score": 8.5},
            )
        ]
    )
    merger = AlgorithmScoreMerger()
    merged = merger.merge([rake1, rake2])
    assert len(merged) == 1
    # Should pick the best (max) rake score
    assert merged[0].metadata.get("rake_score") == 8.5


def test_all_four_score_keys_promoted():
    """All 4 algorithm score keys should be promoted when present."""
    candidates = [
        CandidateTerm(
            term="test",
            source_algorithm="TopicRank",
            confidence=0.9,
            frequency=1,
            metadata={"topicrank_score": 0.3},
        ),
        CandidateTerm(
            term="test",
            source_algorithm="YAKE",
            confidence=0.8,
            frequency=1,
            metadata={"yake_score": 0.05},
        ),
        CandidateTerm(
            term="test",
            source_algorithm="RAKE",
            confidence=0.6,
            frequency=1,
            metadata={"rake_score": 7.2},
        ),
        CandidateTerm(
            term="test",
            source_algorithm="BM25",
            confidence=0.5,
            frequency=1,
            metadata={"bm25_score": 12.1},
        ),
    ]
    results = [AlgorithmResult(candidates=[c]) for c in candidates]
    merger = AlgorithmScoreMerger()
    merged = merger.merge(results)
    assert len(merged) == 1
    m = merged[0].metadata
    assert m["topicrank_score"] == 0.3
    assert m["yake_score"] == 0.05
    assert m["rake_score"] == 7.2
    assert m["bm25_score"] == 12.1


def test_missing_scores_not_in_metadata():
    """Algorithms without scores should not add zero entries to metadata."""
    ner = AlgorithmResult(
        candidates=[
            CandidateTerm(
                term="Smith",
                source_algorithm="NER",
                confidence=0.9,
                frequency=1,
                metadata={},  # NER has no score
            )
        ]
    )
    merger = AlgorithmScoreMerger()
    merged = merger.merge([ner])
    assert len(merged) == 1
    # No score keys should be present
    assert "topicrank_score" not in merged[0].metadata
    assert "yake_score" not in merged[0].metadata
