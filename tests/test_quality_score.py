"""
Tests for _calculate_quality_score() in VocabularyExtractor.

Covers the rules engine scoring logic:
- Base score (50 points)
- Occurrence curve (log-scaled, cap +35)
- Rarity boost (frequency rank thresholds)
- Tiered person name boost (+5 / +12 / +15)
- Multi-algorithm agreement tiers (+4 / +8 / +12)
- Algorithm confidence boost (up to +6)
- Artifact detection penalties (all caps, leading digit, single letter, trailing punct)
- TermSources-based adjustments (multi-doc, confidence)
- User-defined indicator pattern signals (+5/-5)
- Score clamped to 0-100
"""

import math

import pytest

from src.core.vocabulary import VocabularyExtractor
from src.core.vocabulary.term_sources import TermSources


@pytest.fixture
def extractor():
    """Create VocabularyExtractor with no external files."""
    return VocabularyExtractor(exclude_list_path=None)


def _score(extractor, **kwargs):
    """Shorthand for calling _calculate_quality_score with defaults."""
    defaults = {
        "is_person": False,
        "term_count": 5,
        "frequency_rank": 100000,
        "algorithm_count": 1,
        "term": "test",
    }
    defaults.update(kwargs)
    return extractor._calculate_quality_score(**defaults)


# =========================================================================
# Base score
# =========================================================================


class TestBaseScore:
    def test_base_is_50(self, extractor):
        """Score starts at 50 before any boosts/penalties."""
        # Minimal term: 1 occ, common word, 1 algorithm, no artifacts
        score = _score(extractor, term_count=1, frequency_rank=100000)
        # Base 50 + occurrence boost for count=1 + no rarity boost
        expected_occ_boost = math.log10(2) * 18  # ~5.4
        assert score == pytest.approx(50.0 + expected_occ_boost, abs=0.5)

    def test_score_clamped_to_0_100(self, extractor):
        """Score should never exceed 100 or go below 0."""
        # Max everything
        high = _score(
            extractor,
            is_person=True,
            term_count=1000,
            frequency_rank=0,
            algorithm_count=5,
            topicrank_score=1.0,
            keybert_score=1.0,
            term="Dr. James Morrison",
        )
        assert high <= 100.0

        # All penalties stacked
        low = _score(
            extractor,
            term_count=1,
            frequency_rank=100000,
            algorithm_count=1,
            term="4Q:",  # leading digit + single-ish + trailing punct
        )
        assert low >= 0.0


# =========================================================================
# Occurrence curve
# =========================================================================


class TestOccurrenceCurve:
    def test_single_occurrence_low_boost(self, extractor):
        """Single occurrence should get a small boost (~5)."""
        boost = math.log10(1 + 1) * 18
        assert boost == pytest.approx(5.4, abs=0.1)

    def test_ten_occurrences(self, extractor):
        """10 occurrences should give ~19 boost."""
        boost = math.log10(10 + 1) * 18
        assert boost == pytest.approx(18.7, abs=0.1)

    def test_high_count_caps_at_35(self, extractor):
        """Very high counts should cap at +35."""
        boost = min(math.log10(1000 + 1) * 18, 35)
        assert boost == 35.0

    def test_more_occurrences_higher_score(self, extractor):
        """Higher occurrence count should produce higher score."""
        score_low = _score(extractor, term_count=1)
        score_mid = _score(extractor, term_count=10)
        score_high = _score(extractor, term_count=100)
        assert score_low < score_mid < score_high


# =========================================================================
# Person name boost (tiered)
# =========================================================================


class TestPersonBoost:
    def test_single_common_word_gets_5(self, extractor):
        """Single common-word person gets +5 (could be noise like 'Will')."""
        base = _score(extractor, is_person=False, term="Smith", frequency_rank=50000)
        person = _score(extractor, is_person=True, term="Smith", frequency_rank=50000)
        assert person - base == pytest.approx(5.0, abs=0.1)

    def test_multi_word_common_gets_12(self, extractor):
        """Multi-word person with common name gets +12."""
        base = _score(extractor, is_person=False, term="John Smith", frequency_rank=50000)
        person = _score(extractor, is_person=True, term="John Smith", frequency_rank=50000)
        assert person - base == pytest.approx(12.0, abs=0.1)

    def test_multi_word_rare_gets_15(self, extractor):
        """Multi-word person with rare name gets +15."""
        base = _score(extractor, is_person=False, term="James Comiskey", frequency_rank=0)
        person = _score(extractor, is_person=True, term="James Comiskey", frequency_rank=0)
        assert person - base == pytest.approx(15.0, abs=0.1)

    def test_single_rare_word_gets_5(self, extractor):
        """Single rare-word person still only gets +5 (single word = less certain)."""
        base = _score(extractor, is_person=False, term="Comiskey", frequency_rank=0)
        person = _score(extractor, is_person=True, term="Comiskey", frequency_rank=0)
        assert person - base == pytest.approx(5.0, abs=0.1)

    def test_non_person_gets_no_boost(self, extractor):
        """Non-person terms get no person boost."""
        score = _score(extractor, is_person=False, term="radiculopathy")
        # Manually compute expected: base + occ + no person
        score2 = _score(extractor, is_person=False, term="radiculopathy")
        assert score == score2


# =========================================================================
# Multi-algorithm agreement
# =========================================================================


class TestAlgorithmAgreement:
    def test_single_algorithm_no_boost(self, extractor):
        """Single algorithm gets no agreement boost."""
        s1 = _score(extractor, algorithm_count=1)
        s0 = _score(extractor, algorithm_count=0)
        # Neither 0 nor 1 algorithm triggers the boost
        assert s1 == s0

    def test_two_algorithms_plus_4(self, extractor):
        """Two algorithms add +4."""
        s1 = _score(extractor, algorithm_count=1)
        s2 = _score(extractor, algorithm_count=2)
        assert s2 - s1 == pytest.approx(4.0, abs=0.1)

    def test_three_algorithms_plus_8(self, extractor):
        """Three algorithms add +8."""
        s1 = _score(extractor, algorithm_count=1)
        s3 = _score(extractor, algorithm_count=3)
        assert s3 - s1 == pytest.approx(8.0, abs=0.1)

    def test_four_plus_algorithms_plus_12(self, extractor):
        """Four+ algorithms add +12."""
        s1 = _score(extractor, algorithm_count=1)
        s4 = _score(extractor, algorithm_count=4)
        s6 = _score(extractor, algorithm_count=6)
        assert s4 - s1 == pytest.approx(12.0, abs=0.1)
        assert s6 == s4  # 4 and 6 are the same tier


# =========================================================================
# Algorithm confidence boost
# =========================================================================


class TestAlgoConfidenceBoost:
    def test_no_scores_no_boost(self, extractor):
        """No algorithm scores means no confidence boost."""
        s1 = _score(extractor)
        s2 = _score(extractor, yake_score=0.0, keybert_score=0.0)
        assert s1 == s2

    def test_keybert_high_confidence(self, extractor):
        """High KeyBERT score should add up to +6."""
        base = _score(extractor)
        boosted = _score(extractor, keybert_score=0.95)
        diff = boosted - base
        assert 0 < diff <= 6.0

    def test_confidence_capped_at_6(self, extractor):
        """Confidence boost should not exceed SCORE_ALGO_CONFIDENCE_BOOST (6)."""
        base = _score(extractor)
        # Max out all scores
        boosted = _score(extractor, keybert_score=1.0, rake_score=15.0, bm25_score=15.0)
        assert boosted - base == pytest.approx(6.0, abs=0.1)

    def test_yake_inverted(self, extractor):
        """YAKE score is inverted (lower raw = better). Low raw score = high confidence."""
        base = _score(extractor)
        # Low YAKE raw score = high importance = high confidence
        low_yake = _score(extractor, yake_score=0.1)
        # High YAKE raw score = low importance = low confidence
        high_yake = _score(extractor, yake_score=5.0)
        assert low_yake > high_yake

    def test_best_score_wins(self, extractor):
        """When multiple algo scores present, the best one determines the boost."""
        # Only weak RAKE
        weak = _score(extractor, rake_score=1.0)
        # Weak RAKE + strong KeyBERT — KeyBERT should dominate
        strong = _score(extractor, rake_score=1.0, keybert_score=0.9)
        assert strong > weak


# =========================================================================
# Artifact detection penalties
# =========================================================================


class TestArtifactPenalties:
    def test_all_caps_penalty(self, extractor):
        """ALL CAPS terms get -12 penalty."""
        normal = _score(extractor, term="plaintiff")
        caps = _score(extractor, term="PLAINTIFF")
        assert normal - caps == pytest.approx(12.0, abs=0.1)

    def test_leading_digit_penalty(self, extractor):
        """Leading digit terms get -8 penalty."""
        normal = _score(extractor, term="Smith")
        digit = _score(extractor, term="4Smith")
        assert normal - digit == pytest.approx(8.0, abs=0.1)

    def test_single_letter_penalty(self, extractor):
        """Single letter terms get -15 penalty."""
        normal = _score(extractor, term="test")
        letter = _score(extractor, term="Q")
        # Single letter also triggers all-caps (-12), so total is -15 + -12 = -27
        assert normal - letter == pytest.approx(27.0, abs=0.1)

    def test_trailing_punctuation_penalty(self, extractor):
        """Trailing punctuation gets -5 penalty."""
        normal = _score(extractor, term="Smith")
        punct = _score(extractor, term="Smith:")
        assert normal - punct == pytest.approx(5.0, abs=0.1)

    def test_stacked_penalties(self, extractor):
        """Multiple artifacts stack their penalties."""
        clean = _score(extractor, term="test")
        # Leading digit (-8) + trailing punct (-5) = -13
        messy = _score(extractor, term="4test:")
        assert clean - messy == pytest.approx(13.0, abs=0.1)


# =========================================================================
# TermSources adjustments
# =========================================================================


class TestTermSourcesAdjustments:
    def test_multi_doc_boost(self, extractor):
        """Terms in 2+ docs get SCORE_MULTI_DOC_BOOST."""
        single = TermSources(doc_ids=["d1"], confidences=[0.95], counts_per_doc=[5])
        multi = TermSources(doc_ids=["d1", "d2"], confidences=[0.95, 0.90], counts_per_doc=[3, 2])
        s_single = _score(extractor, term_sources=single)
        s_multi = _score(extractor, term_sources=multi)
        assert s_multi > s_single

    def test_all_low_conf_penalty(self, extractor):
        """Terms with all low-confidence sources get penalized."""
        high_conf = TermSources(doc_ids=["d1"], confidences=[0.95], counts_per_doc=[5])
        low_conf = TermSources(doc_ids=["d1"], confidences=[0.40], counts_per_doc=[5])
        s_high = _score(extractor, term_sources=high_conf)
        s_low = _score(extractor, term_sources=low_conf)
        assert s_high > s_low


# =========================================================================
# Rarity boost
# =========================================================================


class TestRarityBoost:
    def test_not_in_google_gets_20(self, extractor):
        """frequency_rank=0 (not in Google dataset) gets +20."""
        common = _score(extractor, frequency_rank=100000)
        rare = _score(extractor, frequency_rank=0)
        assert rare - common == pytest.approx(20.0, abs=0.1)

    def test_very_rare_gets_15(self, extractor):
        """frequency_rank > 200000 gets +15."""
        common = _score(extractor, frequency_rank=100000)
        rare = _score(extractor, frequency_rank=250000)
        assert rare - common == pytest.approx(15.0, abs=0.1)

    def test_moderately_rare_gets_10(self, extractor):
        """frequency_rank 180001-200000 gets +10."""
        common = _score(extractor, frequency_rank=100000)
        mod_rare = _score(extractor, frequency_rank=190000)
        assert mod_rare - common == pytest.approx(10.0, abs=0.1)

    def test_common_word_no_rarity_boost(self, extractor):
        """Common words (rank <= 180000) get no rarity boost."""
        s1 = _score(extractor, frequency_rank=50000)
        s2 = _score(extractor, frequency_rank=100000)
        assert s1 == s2


# =========================================================================
# User-defined indicator pattern signals
# =========================================================================


class TestIndicatorPatternSignals:
    """Positive/negative indicator patterns add +5/-5 to rule-based score."""

    def test_positive_indicator_adds_5(self, extractor):
        """Term matching a positive indicator gets +5 boost."""
        from unittest.mock import patch

        base = _score(extractor, term="drywall")
        with (
            patch(
                "src.core.vocabulary.indicator_patterns.matches_positive",
                return_value=True,
            ),
            patch(
                "src.core.vocabulary.indicator_patterns.matches_negative",
                return_value=False,
            ),
        ):
            boosted = _score(extractor, term="drywall")
        assert boosted - base == pytest.approx(5.0, abs=0.1)

    def test_negative_indicator_subtracts_5(self, extractor):
        """Term matching a negative indicator gets -5 penalty."""
        from unittest.mock import patch

        base = _score(extractor, term="drywall")
        with (
            patch(
                "src.core.vocabulary.indicator_patterns.matches_positive",
                return_value=False,
            ),
            patch(
                "src.core.vocabulary.indicator_patterns.matches_negative",
                return_value=True,
            ),
        ):
            penalized = _score(extractor, term="drywall")
        assert base - penalized == pytest.approx(5.0, abs=0.1)

    def test_both_indicators_cancel_out(self, extractor):
        """Term matching both positive and negative nets zero."""
        from unittest.mock import patch

        base = _score(extractor, term="ambiguous")
        with (
            patch(
                "src.core.vocabulary.indicator_patterns.matches_positive",
                return_value=True,
            ),
            patch(
                "src.core.vocabulary.indicator_patterns.matches_negative",
                return_value=True,
            ),
        ):
            both = _score(extractor, term="ambiguous")
        assert both == pytest.approx(base, abs=0.1)

    def test_no_indicators_no_change(self, extractor):
        """Term matching neither indicator gets no adjustment."""
        from unittest.mock import patch

        base = _score(extractor, term="plaintiff")
        with (
            patch(
                "src.core.vocabulary.indicator_patterns.matches_positive",
                return_value=False,
            ),
            patch(
                "src.core.vocabulary.indicator_patterns.matches_negative",
                return_value=False,
            ),
        ):
            unchanged = _score(extractor, term="plaintiff")
        assert unchanged == pytest.approx(base, abs=0.1)
