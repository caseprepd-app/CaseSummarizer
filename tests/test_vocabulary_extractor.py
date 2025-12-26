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

from src.vocabulary import VocabularyExtractor  # noqa: E402

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
        exclude_list_path=EXCLUDE_LIST_PATH,
        medical_terms_path=MEDICAL_TERMS_PATH
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


def test_get_definition(extractor):
    """Test WordNet definition lookup."""
    # Session 52: API changed to use is_person instead of category

    # WordNet definition for non-person term
    definition = extractor._get_definition("cat", is_person=False)
    assert "feline" in definition.lower()  # Check for part of the definition

    # No WordNet definition
    definition_no_def = extractor._get_definition("asdfghjkl", is_person=False)
    assert definition_no_def == "—"

    # Person - no definition needed
    definition_person = extractor._get_definition("John Smith", is_person=True)
    assert definition_person == "—"


def test_extract(extractor):
    """Test full extraction pipeline."""
    # Session 23: Test text updated to have terms appear twice (except PERSON)
    # Minimum occurrence filter requires ≥2 for non-PERSON terms (Medical, Place, Technical)
    # PERSON entities are exempt from this filter - they can appear just once
    test_text = "The plaintiff, Mr. John Doe, presented with cardiomyopathy. The cardiomyopathy was severe. He visited Dr. Jane Smith at Mayo Clinic for treatment. She referred him to Mayo Clinic's cardiology department. The court delivered its verdict."

    vocabulary = extractor.extract(test_text)

    # Session 52: Changed from Type to Is Person (binary flag)
    # Expected terms with Is Person flag and Role/Relevance
    expected_terms_single = {
        "john doe": {"Is Person": "Yes", "Role/Relevance": "Person in case"},
        "cardiomyopathy": {"Is Person": "No", "Role/Relevance": "Vocabulary term"},
        "jane smith": {"Is Person": "Yes", "Role/Relevance": "Medical professional"},
    }

    found_terms = {item["Term"].lower(): item for item in vocabulary}

    for term, expected_data in expected_terms_single.items():
        assert term in found_terms, f"Term '{term}' not found in extracted vocabulary"
        assert found_terms[term]["Is Person"] == expected_data["Is Person"]
        # Support multiple acceptable Role/Relevance values for flexible NER classification
        expected_roles = expected_data["Role/Relevance"]
        if isinstance(expected_roles, list):
            assert found_terms[term]["Role/Relevance"] in expected_roles, \
                f"Role/Relevance '{found_terms[term]['Role/Relevance']}' not in expected {expected_roles}"
        else:
            assert found_terms[term]["Role/Relevance"] == expected_roles
        # Definition check: Person should be "—", non-Person may have definition or "—"
        if expected_data["Is Person"] == "Yes":
            assert found_terms[term]["Definition"] == "—"
        # Session 23: Verify new confidence columns exist
        assert "Quality Score" in found_terms[term], "Missing Quality Score column"
        assert "In-Case Freq" in found_terms[term], "Missing In-Case Freq column"
        assert "Freq Rank" in found_terms[term], "Missing Freq Rank column"
        # Session 25: Verify Sources column exists
        assert "Sources" in found_terms[term], "Missing Sources column"

    # Ensure excluded terms are not present
    assert "plaintiff" not in found_terms
    assert "court" not in found_terms
    assert "verdict" not in found_terms


def test_extract_deduplication(extractor):
    """Test that duplicates are handled correctly."""
    test_text_dup = "Cardiomyopathy is a serious condition. The patient had cardiomyopathy. Also, cardiomyopathy can be genetic."
    vocabulary_dup = extractor.extract(test_text_dup)

    # Expected for duplicated term: cardiomyopathy
    # Session 52: Changed from Type to Is Person
    found_cardiomyopathy = next((item for item in vocabulary_dup if item["Term"].lower() == "cardiomyopathy"), None)
    assert found_cardiomyopathy is not None
    assert found_cardiomyopathy["Is Person"] == "No"  # Medical term, not a person
    assert found_cardiomyopathy["Role/Relevance"] == "Vocabulary term"
    assert found_cardiomyopathy["Definition"] != "—"  # Should have a definition
    assert sum(1 for item in vocabulary_dup if item["Term"].lower() == "cardiomyopathy") == 1  # Only one entry


def test_quality_score_range(extractor):
    """Test that quality scores are in valid range."""
    test_text = "Dr. John Smith diagnosed the patient with cardiomyopathy at Memorial Hospital. The cardiomyopathy was severe."
    vocabulary = extractor.extract(test_text)

    for term in vocabulary:
        score = term["Quality Score"]
        assert 0.0 <= score <= 100.0, f"Quality score {score} out of range for '{term['Term']}'"


def test_sources_column(extractor):
    """Test that Sources column tracks algorithm provenance (Session 25)."""
    test_text = "Dr. Smith and Dr. Jones consulted on the adenocarcinoma diagnosis. The adenocarcinoma was confirmed."
    vocabulary = extractor.extract(test_text)

    for term in vocabulary:
        sources = term.get("Sources", "")
        # Sources should be comma-separated list of algorithm names
        assert sources, f"Empty Sources for '{term['Term']}'"
        # Should contain at least one known algorithm
        assert any(alg in sources for alg in ["NER", "RAKE"]), f"Unknown algorithms in Sources: {sources}"
