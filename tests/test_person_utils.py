from src.core.vocabulary.person_utils import (
    is_person_entry,
)


def test_is_person_format():
    assert is_person_entry({"Term": "John", "Is Person": "Yes"})
    assert not is_person_entry({"Term": "x", "Is Person": "No"})
    assert is_person_entry({"Term": "x", "Is Person": "true"})
    assert is_person_entry({"Term": "x", "Is Person": "1"})


def test_type_format():
    assert is_person_entry({"Term": "Jane", "Type": "Person"})
    assert not is_person_entry({"Term": "x", "Type": "Medical"})


def test_ml_format():
    assert is_person_entry({"Term": "x", "is_person": 1})
    assert not is_person_entry({"Term": "x", "is_person": 0})
    assert is_person_entry({"Term": "x", "is_person": "yes"})


def test_missing_keys():
    assert not is_person_entry({})
    assert not is_person_entry({"Term": "hello"})


def test_empty_values():
    assert not is_person_entry({"Is Person": ""})
    assert not is_person_entry({"Type": ""})


# ---------------------------------------------------------------------------
# count_persons / vocab_summary_counts
# ---------------------------------------------------------------------------


def test_count_persons_all_formats():
    """count_persons must detect persons from all 3 extraction sources."""
    from src.core.vocabulary.person_utils import count_persons

    data = [
        {"Term": "Alice", "Is Person": "Yes"},  # VocabularyExtractor format
        {"Term": "Bob", "Type": "Person"},  # Type-based detection format
        {"Term": "Carol", "is_person": 1},  # ML feature format
        {"Term": "radiculopathy", "Type": "Medical"},  # Not a person
        {"Term": "stenosis"},  # Not a person
    ]
    assert count_persons(data) == 3


def test_count_persons_empty():
    """count_persons returns 0 for empty list."""
    from src.core.vocabulary.person_utils import count_persons

    assert count_persons([]) == 0


def test_vocab_summary_counts():
    """vocab_summary_counts returns (total, persons, terms)."""
    from src.core.vocabulary.person_utils import vocab_summary_counts

    data = [
        {"Term": "Alice", "Is Person": "Yes"},
        {"Term": "Bob", "Type": "Person"},
        {"Term": "stenosis", "Type": "Medical"},
    ]
    total, persons, terms = vocab_summary_counts(data)
    assert total == 3
    assert persons == 2
    assert terms == 1
