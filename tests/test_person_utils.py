from src.core.vocabulary.person_utils import is_person_entry


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
