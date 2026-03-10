"""
Tests for the VocabularyExtractor class.

Tests cover:
- Word list loading (exclude list, medical terms)
- Category assignment (via extraction pipeline)
- Definition lookup via WordNet
- Full extraction pipeline with deduplication and relevance scoring
- Multi-algorithm result merging (Session 25)

Note: Internal methods like _is_unusual and _get_category have been refactored
into the algorithm classes (Session 25). Tests focus on public API behavior.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.vocabulary import VocabularyExtractor  # noqa: E402

# Define paths for test resources
TEST_DIR = Path(__file__).parent
EXCLUDE_LIST_PATH = TEST_DIR / "test_legal_exclude.txt"
MEDICAL_TERMS_PATH = TEST_DIR / "test_medical_terms.txt"


# Create dummy exclude and medical terms files for testing
@pytest.fixture(scope="module", autouse=True)
def setup_test_files():
    # Create a dummy legal exclude list
    with open(EXCLUDE_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("verdict\n")
        f.write("plaintiff\n")
        f.write("defendant\n")
        f.write("court\n")

    # Create a dummy medical terms list
    with open(MEDICAL_TERMS_PATH, "w", encoding="utf-8") as f:
        f.write("cardiomyopathy\n")
        f.write("nephrology\n")
        f.write("endoscopy\n")

    yield

    # Teardown: Remove dummy files
    os.remove(EXCLUDE_LIST_PATH)
    os.remove(MEDICAL_TERMS_PATH)


@pytest.fixture
def extractor():
    """Create VocabularyExtractor with test configuration."""
    return VocabularyExtractor(
        exclude_list_path=EXCLUDE_LIST_PATH, medical_terms_path=MEDICAL_TERMS_PATH
    )


def test_load_word_list(extractor):
    """Test that word lists are loaded correctly."""
    assert "verdict" in extractor.exclude_list
    assert "plaintiff" in extractor.exclude_list
    assert "cardiomyopathy" in extractor.medical_terms
    assert "nephrology" in extractor.medical_terms


def test_algorithms_initialized(extractor):
    """Test that algorithms are properly initialized (Session 25)."""
    assert len(extractor.algorithms) >= 2
    algorithm_names = [alg.name for alg in extractor.algorithms]
    assert "NER" in algorithm_names
    assert "RAKE" in algorithm_names


# test_get_definition removed: WordNet definitions no longer generated.
# _get_definition() method was commented out in vocabulary_extractor.py.


def test_extract(extractor):
    """Test full extraction pipeline."""
    # All terms (including PERSON) must appear at least twice to pass min_occurrences=2 filter.
    # Session 57: Removed cardiomyopathy expectation - RAKE min_score=2.0 filters single words
    # (single words score 1.0 in RAKE), and NER only extracts named entities, not medical terms
    test_text = "The plaintiff, Mr. John Doe, presented with cardiomyopathy. The cardiomyopathy was severe. He visited Dr. Jane Smith at Mayo Clinic for treatment. Jane Smith referred him to Mayo Clinic's cardiology department. John Doe was satisfied with the care. The court delivered its verdict."

    vocabulary, filtered_terms = extractor.extract(test_text)

    # Session 52: Changed from Type to Is Person (binary flag)
    # Expected terms with Is Person flag and Role/Relevance
    # Note: Only NER-extractable entities (Person, Place, Org) are found
    expected_terms = {
        "john doe": {"Is Person": "Yes", "Role/Relevance": "Person in case"},
        "jane smith": {"Is Person": "Yes", "Role/Relevance": "Medical professional"},
    }

    found_terms = {item["Term"].lower(): item for item in vocabulary}

    for term, expected_data in expected_terms.items():
        assert term in found_terms, f"Term '{term}' not found in extracted vocabulary"
        assert found_terms[term]["Is Person"] == expected_data["Is Person"]
        # Support multiple acceptable Role/Relevance values for flexible NER classification
        expected_roles = expected_data["Role/Relevance"]
        if isinstance(expected_roles, list):
            assert found_terms[term]["Role/Relevance"] in expected_roles, (
                f"Role/Relevance '{found_terms[term]['Role/Relevance']}' not in expected {expected_roles}"
            )
        else:
            assert found_terms[term]["Role/Relevance"] == expected_roles
        # Definition column removed: WordNet definitions no longer generated
        # Session 23: Verify new confidence columns exist
        assert "Quality Score" in found_terms[term], "Missing Quality Score column"
        assert "Occurrences" in found_terms[term], "Missing Occurrences column"
        assert "Google Rarity Rank" in found_terms[term], "Missing Google Rarity Rank column"
        # Session 25: Verify Sources column exists
        assert "Sources" in found_terms[term], "Missing Sources column"

    # Ensure excluded terms are not present
    assert "plaintiff" not in found_terms
    assert "court" not in found_terms
    assert "verdict" not in found_terms


def test_extract_deduplication(extractor):
    """Test that duplicates are handled correctly."""
    # Session 57: Updated test to use named entities (which NER extracts) instead of
    # single-word medical terms (which RAKE filters due to min_score=2.0 threshold)
    test_text_dup = "Dr. John Smith examined the patient. John Smith recommended surgery. Later, John Smith reviewed the results."
    vocabulary_dup, _filtered = extractor.extract(test_text_dup)

    # Expected: "John Smith" should appear only once despite multiple mentions
    found_john_smith = next(
        (item for item in vocabulary_dup if item["Term"].lower() == "john smith"), None
    )
    assert found_john_smith is not None, "John Smith not found in extracted vocabulary"
    assert found_john_smith["Is Person"] == "Yes"  # Named entity, is a person
    # Definition column removed: WordNet definitions no longer generated
    # Verify deduplication: only one entry for "John Smith"
    john_smith_count = sum(1 for item in vocabulary_dup if item["Term"].lower() == "john smith")
    assert john_smith_count == 1, f"Expected 1 entry for 'John Smith', found {john_smith_count}"


def test_quality_score_range(extractor):
    """Test that quality scores are in valid range."""
    test_text = "Dr. John Smith diagnosed the patient with cardiomyopathy at Memorial Hospital. The cardiomyopathy was severe. John Smith recommended follow-up."
    vocabulary, _filtered = extractor.extract(test_text)

    for term in vocabulary:
        score = term["Quality Score"]
        assert 0.0 <= score <= 100.0, f"Quality score {score} out of range for '{term['Term']}'"


def test_sources_column(extractor):
    """Test that Sources column tracks algorithm provenance (Session 25)."""
    test_text = "Dr. Smith and Dr. Jones consulted on the adenocarcinoma diagnosis. The adenocarcinoma was confirmed. Dr. Smith and Dr. Jones agreed on the treatment plan."
    vocabulary, _filtered = extractor.extract(test_text)

    for term in vocabulary:
        sources = term.get("Sources", "")
        # Sources should be comma-separated list of algorithm names
        assert sources, f"Empty Sources for '{term['Term']}'"
        # Should contain at least one known algorithm
        assert any(alg in sources for alg in ["NER", "RAKE"]), (
            f"Unknown algorithms in Sources: {sources}"
        )
