from src.core.vocabulary.string_utils import edit_distance, get_term_words, normalize_term


def test_edit_distance_identical():
    assert edit_distance("hello", "hello") == 0


def test_edit_distance_one_sub():
    assert edit_distance("Smith", "Smitb") == 1


def test_edit_distance_two():
    assert edit_distance("Jenkins", "Jenidns") == 2


def test_edit_distance_empty():
    assert edit_distance("", "") == 0
    assert edit_distance("abc", "") == 3
    assert edit_distance("", "abc") == 3


def test_edit_distance_classic():
    assert edit_distance("kitten", "sitting") == 3


def test_normalize_term():
    assert normalize_term("  John Smith  ") == "john smith"
    assert normalize_term("RADICULOPATHY") == "radiculopathy"
    assert normalize_term("") == ""


def test_get_term_words():
    assert get_term_words("John Smith") == ["john", "smith"]
    assert get_term_words("  Di Leo  ") == ["di", "leo"]
    assert get_term_words("single") == ["single"]
