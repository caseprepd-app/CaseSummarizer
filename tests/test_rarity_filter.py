"""Tests for rarity_filter.py"""

from unittest.mock import patch

import pytest

# Mock frequencies: score = rank/total_words (0.0 = most common, 1.0 = rarest)
# is_common_word checks: score < top_n / total_words
# With 9 entries and top_n=5, threshold = 5/9 ≈ 0.556
# So words with score < 0.556 are "common", score >= 0.556 are "rare"
MOCK_FREQUENCIES = {
    "the": 0.0,
    "same": 0.1,
    "age": 0.15,
    "body": 0.2,
    "left": 0.25,
    "side": 0.3,
    "cervical": 0.6,
    "spine": 0.7,
    "radiculopathy": 0.8,
}
# Use top_n=5 with 9 total words → threshold ≈ 0.556
# Common: the(0.0), same(0.1), age(0.15), body(0.2), left(0.25), side(0.3) → all < 0.556
# Rare: cervical(0.6), spine(0.7), radiculopathy(0.8) → all >= 0.556
MOCK_TOP_N = 5


def _make_term(term, is_person=False, freq=5):
    return {"Term": term, "Is Person": is_person, "Occurrences": freq}


@pytest.fixture(autouse=True)
def mock_frequencies():
    with patch(
        "src.core.vocabulary.rarity_filter._load_scaled_frequencies",
        return_value=MOCK_FREQUENCIES,
    ):
        # Also need to reset the cached module-level variable
        import src.core.vocabulary.rarity_filter as rf

        rf._scaled_frequencies = None
        yield
        rf._scaled_frequencies = None


class TestIsCommonWord:
    def test_common_word(self):
        from src.core.vocabulary.rarity_filter import is_common_word

        assert is_common_word("the", MOCK_TOP_N) is True

    def test_rare_word(self):
        from src.core.vocabulary.rarity_filter import is_common_word

        assert is_common_word("radiculopathy", MOCK_TOP_N) is False

    def test_unknown_word(self):
        from src.core.vocabulary.rarity_filter import is_common_word

        # Word not in frequency list should not be common
        assert is_common_word("xyzzyplugh", MOCK_TOP_N) is False


class TestShouldFilterPhrase:
    def test_all_common_words_filtered(self):
        from src.core.vocabulary.rarity_filter import should_filter_phrase

        assert should_filter_phrase("the same", is_person=False) is True

    def test_rare_words_kept(self):
        from src.core.vocabulary.rarity_filter import should_filter_phrase

        assert should_filter_phrase("cervical spine", is_person=False) is False

    def test_person_exempt(self):
        from src.core.vocabulary.rarity_filter import should_filter_phrase

        # Person names should not be filtered even if words look common
        assert should_filter_phrase("John Smith", is_person=True) is False


class TestFilterCommonPhrases:
    def test_filters_common_keeps_rare(self):
        from src.core.vocabulary.rarity_filter import filter_common_phrases

        vocab = [
            _make_term("the same", is_person=False, freq=10),
            _make_term("cervical spine", is_person=False, freq=8),
            _make_term("left side", is_person=False, freq=6),
        ]
        result = filter_common_phrases(vocab, "Term")
        result_terms = [t["Term"] for t in result]
        assert "cervical spine" in result_terms
        # Common phrases should be filtered
        assert "the same" not in result_terms

    def test_empty_vocabulary(self):
        from src.core.vocabulary.rarity_filter import filter_common_phrases

        assert filter_common_phrases([], "Term") == []


class TestGetPhraseRarityScores:
    def test_returns_tuple(self):
        from src.core.vocabulary.rarity_filter import get_phrase_rarity_scores

        result = get_phrase_rarity_scores("cervical spine")
        assert isinstance(result, tuple)
        assert len(result) == 3
        max_rarity, mean_rarity, word_count = result
        assert word_count == 2
        assert max_rarity >= mean_rarity

    def test_single_word(self):
        from src.core.vocabulary.rarity_filter import get_phrase_rarity_scores

        max_rarity, mean_rarity, word_count = get_phrase_rarity_scores("radiculopathy")
        assert word_count == 1
        assert max_rarity == mean_rarity
