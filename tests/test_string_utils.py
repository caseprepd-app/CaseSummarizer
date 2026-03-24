from src.core.vocabulary.string_utils import (
    edit_distance,
    fuzzy_match,
    get_term_words,
    normalize_term,
)

# ---------------------------------------------------------------------------
# edit_distance
# ---------------------------------------------------------------------------


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


def test_edit_distance_insertion_and_deletion():
    """Verify insertions and deletions are counted correctly."""
    assert edit_distance("abc", "abcd") == 1  # one insertion
    assert edit_distance("abcd", "abc") == 1  # one deletion
    assert edit_distance("abc", "axbyc") == 2  # two insertions


def test_edit_distance_symmetric():
    """edit_distance(a, b) == edit_distance(b, a) for all inputs."""
    pairs = [("cat", "car"), ("", "xyz"), ("ab", "ba"), ("foo", "foo")]
    for a, b in pairs:
        assert edit_distance(a, b) == edit_distance(b, a)


def test_edit_distance_unicode():
    """Handles unicode characters correctly."""
    assert edit_distance("café", "cafe") == 1
    assert edit_distance("naïve", "naive") == 1


# ---------------------------------------------------------------------------
# normalize_term
# ---------------------------------------------------------------------------


def test_normalize_term():
    assert normalize_term("  John Smith  ") == "john smith"
    assert normalize_term("RADICULOPATHY") == "radiculopathy"
    assert normalize_term("") == ""


def test_normalize_term_tabs_and_newlines():
    """Strips only leading/trailing whitespace, not internal."""
    assert normalize_term("\tHello World\n") == "hello world"


def test_normalize_term_unicode():
    """Unicode characters are lowercased correctly."""
    assert normalize_term("MÜNCHEN") == "münchen"


# ---------------------------------------------------------------------------
# get_term_words
# ---------------------------------------------------------------------------


def test_get_term_words():
    assert get_term_words("John Smith") == ["john", "smith"]
    assert get_term_words("  Di Leo  ") == ["di", "leo"]
    assert get_term_words("single") == ["single"]


def test_get_term_words_empty():
    """Empty input returns empty list."""
    assert get_term_words("") == []
    assert get_term_words("   ") == []


def test_get_term_words_multiple_spaces():
    """Multiple spaces between words don't create empty entries."""
    assert get_term_words("John    Smith") == ["john", "smith"]


# ---------------------------------------------------------------------------
# fuzzy_match
# ---------------------------------------------------------------------------


def test_fuzzy_match_identical():
    """Identical strings have ratio 1.0 and always match."""
    is_match, ratio = fuzzy_match("Smith", "Smith")
    assert is_match is True
    assert ratio == 1.0


def test_fuzzy_match_similar():
    """One-character difference still matches at default 0.8 threshold."""
    is_match, ratio = fuzzy_match("Smith", "Smitb")
    assert is_match is True
    assert ratio >= 0.8


def test_fuzzy_match_dissimilar():
    """Completely different strings fail to match."""
    is_match, ratio = fuzzy_match("hello", "world")
    assert is_match is False
    assert ratio < 0.5


def test_fuzzy_match_custom_threshold():
    """Custom threshold changes match behavior."""
    is_match_strict, _ = fuzzy_match("cat", "car", threshold=0.9)
    is_match_loose, _ = fuzzy_match("cat", "car", threshold=0.5)
    assert is_match_strict is False
    assert is_match_loose is True


def test_fuzzy_match_empty_strings():
    """Two empty strings are identical."""
    is_match, ratio = fuzzy_match("", "")
    assert is_match is True
    assert ratio == 1.0


def test_fuzzy_match_one_empty():
    """Empty vs non-empty has ratio 0.0."""
    is_match, ratio = fuzzy_match("", "hello")
    assert is_match is False
    assert ratio == 0.0
