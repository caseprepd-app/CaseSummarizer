"""Tests for key excerpt interestingness scoring."""

import pytest

from src.core.summarization.excerpt_scorer import (
    PERSON_HIT_POINTS,
    RAREST_WORD_BONUS,
    REJECTED_TERM_PENALTY,
    VOCAB_HIT_POINTS,
    score_excerpt,
)


@pytest.fixture
def vocab_terms():
    """Sample vocab terms with quality scores."""
    return {
        "toxicology": 85,
        "deposition": 75,
        "proceedings": 60,  # Below 70 threshold
        "martinez": 90,
    }


@pytest.fixture
def person_terms():
    """Sample person entity terms."""
    return {"martinez", "officer davis"}


@pytest.fixture
def rejected_terms():
    """Sample rejected terms from feedback CSV."""
    return {"proceedings", "stipulated", "admonished"}


@pytest.fixture
def rank_map():
    """Frequency rank map (lower rank = more common)."""
    # Simulate a 100-word dataset so rarity = rank / 100
    words = {f"word{i}": i for i in range(100)}
    # Override specific words at known ranks
    words["the"] = 0
    words["is"] = 1
    words["and"] = 2
    words["was"] = 3
    words["court"] = 10
    words["jury"] = 15
    words["witness"] = 20
    words["session"] = 12
    words["present"] = 14
    words["deposition"] = 50
    words["toxicology"] = 85
    words["martinez"] = 90
    words["spondylosis"] = 115
    words["doctor"] = 25
    words["noted"] = 30
    words["patient"] = 35
    words["records"] = 40
    return words


class TestVocabHits:
    """Tests for vocab term hit scoring."""

    def test_scores_high_quality_terms(self, vocab_terms):
        """Vocab terms with score >= 70 earn points."""
        text = "The toxicology report was discussed at the deposition."
        score = score_excerpt(text, vocab_terms, set(), set(), {})
        # toxicology (85 >= 70) + deposition (75 >= 70) = 2 hits
        assert score >= 2 * VOCAB_HIT_POINTS

    def test_ignores_low_quality_terms(self, vocab_terms):
        """Vocab terms below 70 quality don't count."""
        text = "The proceedings continued normally."
        score = score_excerpt(text, vocab_terms, set(), set(), {})
        # proceedings (60 < 70) = 0 hits from vocab
        assert score < VOCAB_HIT_POINTS

    def test_no_vocab_hits(self):
        """Chunks with no vocab terms score zero for this feature."""
        text = "The court will now recess until Monday morning."
        score = score_excerpt(text, {}, set(), set(), {})
        assert score == 0.0


class TestPersonHits:
    """Tests for person entity scoring."""

    def test_person_names_score_high(self, person_terms):
        """Person entities earn 5 points each."""
        text = "Martinez testified that Officer Davis arrived at the scene."
        score = score_excerpt(text, {}, person_terms, set(), {})
        assert score >= 2 * PERSON_HIT_POINTS

    def test_no_persons(self, person_terms):
        """Chunks without person names get no person points."""
        text = "The court will recess for lunch."
        score = score_excerpt(text, {}, person_terms, set(), {})
        assert score == 0.0


class TestRejectedTerms:
    """Tests for rejected term penalty."""

    def test_rejected_terms_subtract_points(self, rejected_terms):
        """Each rejected term costs -1 point."""
        text = "The proceedings were stipulated and the jury was admonished."
        score = score_excerpt(text, {}, set(), rejected_terms, {})
        # All 3 rejected terms appear
        assert score <= 3 * REJECTED_TERM_PENALTY

    def test_no_rejected_terms(self, rejected_terms):
        """Chunks without rejected terms get no penalty."""
        text = "The witness described the collision in detail."
        score = score_excerpt(text, {}, set(), rejected_terms, {})
        assert score >= 0.0


class TestRarityScoring:
    """Tests for word rarity features."""

    def test_rare_word_gets_bonus(self, rank_map):
        """A very rare word triggers the rarest word bonus."""
        # spondylosis has rank 95/100 = 0.95 rarity, above 0.85 threshold
        text = "The doctor noted spondylosis in the patient records."
        score = score_excerpt(text, {}, set(), set(), rank_map)
        assert score >= RAREST_WORD_BONUS

    def test_common_words_lower_than_rare(self, rank_map):
        """Common-word chunks score lower than rare-word chunks."""
        common_text = "The court was in session and the jury was present."
        rare_text = "The doctor noted spondylosis in the patient records."
        common_score = score_excerpt(common_text, {}, set(), set(), rank_map)
        rare_score = score_excerpt(rare_text, {}, set(), set(), rank_map)
        assert rare_score > common_score

    def test_mean_rarity_contributes(self, rank_map):
        """Average rarity of non-stopwords contributes 0-10 points."""
        text = "The toxicology deposition discussed spondylosis evidence."
        score = score_excerpt(text, {}, set(), set(), rank_map)
        # Should get mean rarity points plus rare word bonus
        assert score > 0.0


class TestCombinedScoring:
    """Tests for combined feature interactions."""

    def test_interesting_vs_routine(
        self,
        vocab_terms,
        person_terms,
        rejected_terms,
        rank_map,
    ):
        """An interesting excerpt should outscore a routine one."""
        interesting = (
            "Martinez described how the toxicology report showed "
            "elevated levels during the deposition testimony."
        )
        routine = (
            "The proceedings were stipulated and the court admonished "
            "the jury before the lunch recess was taken."
        )
        interesting_score = score_excerpt(
            interesting,
            vocab_terms,
            person_terms,
            rejected_terms,
            rank_map,
        )
        routine_score = score_excerpt(
            routine,
            vocab_terms,
            person_terms,
            rejected_terms,
            rank_map,
        )
        assert interesting_score > routine_score

    def test_empty_text_scores_zero(self):
        """Empty text gets zero score."""
        assert score_excerpt("", {}, set(), set(), {}) == 0.0

    def test_short_text_scores_zero(self):
        """Text with no 3+ char words scores zero."""
        assert score_excerpt("a b", {}, set(), set(), {}) == 0.0
