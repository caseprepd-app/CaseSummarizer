"""
Person Detection Utilities for Vocabulary Extraction.

Centralized person detection to handle inconsistent formats.

The vocabulary pipeline has two sources that mark person names:
1. VocabularyExtractor.extract(): Sets {"Is Person": "Yes"}
2. LLMVocabExtractor: Sets {"Type": "Person"}

This module provides a single function to check both formats,
ensuring consistent behavior across all filtering and processing.
"""

from src.core.vocab_schema import VF


def is_person_entry(term_data: dict) -> bool:
    """
    Check if a vocabulary term represents a person name.

    Handles multiple formats from different extraction sources:
    - VocabularyExtractor: {"Is Person": "Yes"} or {"Is Person": "No"}
    - LLMVocabExtractor: {"Type": "Person"}
    - ML features: {"is_person": 1} or {"is_person": 0}

    Args:
        term_data: Dictionary containing term metadata

    Returns:
        True if the term is identified as a person name

    Examples:
        >>> is_person_entry({"Term": "John Smith", "Is Person": "Yes"})
        True
        >>> is_person_entry({"Term": "radiculopathy", "Type": "Medical"})
        False
        >>> is_person_entry({"Term": "Jane Doe", "Type": "Person"})
        True
    """
    # Check "Is Person" format (VocabularyExtractor)
    is_person_val = term_data.get(VF.IS_PERSON, "")
    if str(is_person_val).lower() in ("yes", "true", "1"):
        return True

    # Check "Type" format (LLMVocabExtractor)
    type_val = term_data.get(VF.TYPE, "")
    if str(type_val).lower() == "person":
        return True

    # Check ML feature format
    ml_person = term_data.get("is_person", 0)
    return bool(ml_person == 1 or str(ml_person).lower() in ("1", "true", "yes"))


def count_persons(vocab_data: list[dict]) -> int:
    """
    Count how many terms in vocab_data are marked as persons.

    Args:
        vocab_data: List of vocabulary term dictionaries

    Returns:
        Number of person entries
    """
    return sum(1 for v in vocab_data if v.get(VF.IS_PERSON) == VF.YES)


def vocab_summary_counts(vocab_data: list[dict]) -> tuple[int, int, int]:
    """
    Get summary counts for vocabulary data.

    Args:
        vocab_data: List of vocabulary term dictionaries

    Returns:
        Tuple of (total_count, person_count, term_count)
    """
    total = len(vocab_data)
    persons = count_persons(vocab_data)
    return total, persons, total - persons
