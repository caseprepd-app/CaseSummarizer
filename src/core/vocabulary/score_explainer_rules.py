"""
Rules-Based Score Explanation

Evaluates which rules-based quality rules fired for a given term
and returns their individual contributions. Used by the score
explanation system to show rules-based factors alongside ML factors.

These rules mirror the logic in vocabulary_extractor._calculate_quality_score().
If the scoring rules change there, update the corresponding rule here too.
"""

import logging
import math
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


class RuleContribution(NamedTuple):
    """A single rule's contribution to the quality score.

    Attributes:
        feature_key: Dedup key matching ML feature names where applicable
        label: Human-readable description including point value
        points: Signed point value (positive = toward keep)
    """

    feature_key: str
    label: str
    points: float


def evaluate_rules(term_data: dict[str, Any]) -> list[RuleContribution]:
    """
    Evaluate rules-based quality rules for a term.

    Returns all rules that fired (non-zero contribution),
    sorted by absolute point value descending.

    Args:
        term_data: Term data dictionary from vocab table

    Returns:
        List of RuleContribution, sorted by |points| descending
    """
    contributions: list[RuleContribution] = []

    # --- Extract values from term_data ---
    term = str(term_data.get("Term", "") or term_data.get("term", "") or "")
    occurrences = int(float(term_data.get("occurrences", 1) or 1))
    frequency_rank = int(
        float(term_data.get("rarity_rank", 0) or term_data.get("Google Rarity Rank", 0) or 0)
    )

    # Algorithm count from comma/space-separated string
    algorithms = str(term_data.get("algorithms", "")).lower()
    algo_set = {a.strip() for a in algorithms.replace(",", " ").split() if a.strip()}
    algorithm_count = len(algo_set)

    # Person detection (handles int, float, or string values)
    is_person_val = term_data.get("is_person", 0)
    is_person = (isinstance(is_person_val, (int, float)) and is_person_val > 0) or str(
        is_person_val
    ).lower() in ("1", "yes", "true")

    # Algorithm scores
    topicrank_score = float(term_data.get("topicrank_score", 0) or 0)
    yake_score = float(term_data.get("yake_score", 0) or 0)
    rake_score = float(term_data.get("rake_score", 0) or 0)
    bm25_score = float(term_data.get("bm25_score", 0) or 0)

    # --- Positive rules ---
    _eval_occurrence_boost(contributions, occurrences)
    _eval_rarity_boost(contributions, frequency_rank)
    _eval_person_boost(contributions, is_person, term, frequency_rank)
    _eval_multi_algo_boost(contributions, algorithm_count)
    _eval_algo_score_boosts(contributions, topicrank_score, yake_score, rake_score, bm25_score)
    _eval_term_sources(contributions, term_data)
    _eval_user_indicators(contributions, term)

    # --- Penalty rules ---
    _eval_artifact_penalties(contributions, term)

    # Sort by absolute point value, biggest impact first
    contributions.sort(key=lambda c: abs(c.points), reverse=True)
    return contributions


def _eval_occurrence_boost(contributions: list[RuleContribution], occurrences: int) -> None:
    """Occurrence frequency boost: log curve, max +35 points."""
    if occurrences > 0:
        boost = min(math.log10(occurrences + 1) * 18, 35)
        if boost >= 1.0:
            contributions.append(
                RuleContribution("log_count", f"Appears {occurrences} times (+{boost:.0f})", boost)
            )


def _eval_rarity_boost(contributions: list[RuleContribution], frequency_rank: int) -> None:
    """Rare word boost based on Google 333K frequency rank."""
    if frequency_rank == 0:
        contributions.append(
            RuleContribution("word_log_rarity_score", "Not in Google 333K dataset (+20)", 20.0)
        )
    elif frequency_rank > 200000:
        contributions.append(
            RuleContribution(
                "word_log_rarity_score", f"Rare word, rank {frequency_rank:,} (+15)", 15.0
            )
        )
    elif frequency_rank > 180000:
        contributions.append(
            RuleContribution(
                "word_log_rarity_score", f"Uncommon word, rank {frequency_rank:,} (+10)", 10.0
            )
        )


def _eval_person_boost(
    contributions: list[RuleContribution], is_person: bool, term: str, frequency_rank: int
) -> None:
    """Tiered person name boost: multi-word rare > multi-word > single word."""
    if not is_person:
        return
    term_words = term.split() if term else []
    is_rare = frequency_rank == 0 or frequency_rank > 180000
    is_multi = len(term_words) >= 2
    if is_multi and is_rare:
        contributions.append(
            RuleContribution("is_person", "Multi-word rare person name (+15)", 15.0)
        )
    elif is_multi:
        contributions.append(RuleContribution("is_person", "Multi-word person name (+12)", 12.0))
    else:
        contributions.append(RuleContribution("is_person", "Person name detected (+5)", 5.0))


def _eval_multi_algo_boost(contributions: list[RuleContribution], algorithm_count: int) -> None:
    """Multi-algorithm agreement boost: non-linear tiers."""
    if algorithm_count == 2:
        contributions.append(RuleContribution("_rule_multi_algo", "Found by 2 algorithms (+4)", 4))
    elif algorithm_count == 3:
        contributions.append(RuleContribution("_rule_multi_algo", "Found by 3 algorithms (+8)", 8))
    elif algorithm_count >= 4:
        contributions.append(
            RuleContribution("_rule_multi_algo", f"Found by {algorithm_count} algorithms (+12)", 12)
        )


def _eval_algo_score_boosts(
    contributions: list[RuleContribution],
    topicrank_score: float,
    yake_score: float,
    rake_score: float,
    bm25_score: float,
) -> None:
    """TopicRank centrality and algorithm confidence boosts."""
    from src.config import SCORE_ALGO_CONFIDENCE_BOOST, SCORE_TOPICRANK_CENTRALITY_BOOST

    # TopicRank centrality (capped by config)
    if topicrank_score > 0:
        tr_boost = min(topicrank_score * 10, SCORE_TOPICRANK_CENTRALITY_BOOST)
        if tr_boost >= 0.5:
            contributions.append(
                RuleContribution(
                    "topicrank_score",
                    f"TopicRank centrality {topicrank_score:.2f} (+{tr_boost:.0f})",
                    tr_boost,
                )
            )

    # Best algorithm confidence score (YAKE inverted, RAKE/BM25 normalized)
    algo_confs = []
    if yake_score > 0:
        algo_confs.append(1.0 / (1.0 + yake_score))
    if rake_score > 0:
        algo_confs.append(min(rake_score / 15.0, 1.0))
    if bm25_score > 0:
        algo_confs.append(min(bm25_score / 15.0, 1.0))
    if algo_confs:
        best_conf = max(algo_confs)
        conf_boost = min(best_conf * 8, SCORE_ALGO_CONFIDENCE_BOOST)
        if conf_boost >= 0.5:
            contributions.append(
                RuleContribution(
                    "_rule_algo_confidence",
                    f"High algorithm confidence (+{conf_boost:.0f})",
                    conf_boost,
                )
            )


def _eval_term_sources(contributions: list[RuleContribution], term_data: dict) -> None:
    """TermSources-based adjustments: multi-doc, confidence, single-source."""
    from src.core.vocabulary.term_sources import TermSources

    sources = term_data.get("sources")
    if not isinstance(sources, TermSources) or sources.num_documents == 0:
        return

    from src.config import (
        SCORE_ALL_LOW_CONF_PENALTY,
        SCORE_HIGH_CONF_BOOST,
        SCORE_MULTI_DOC_BOOST,
        SCORE_SINGLE_SOURCE_CONF_THRESHOLD,
        SCORE_SINGLE_SOURCE_MIN_DOCS,
        SCORE_SINGLE_SOURCE_PENALTY,
    )

    total_docs = int(float(term_data.get("total_docs_in_session", 1) or 1))

    if sources.num_documents >= 2:
        contributions.append(
            RuleContribution(
                "doc_diversity_ratio",
                f"Found in {sources.num_documents} documents (+{SCORE_MULTI_DOC_BOOST})",
                float(SCORE_MULTI_DOC_BOOST),
            )
        )
    if sources.high_conf_doc_ratio > 0.8:
        contributions.append(
            RuleContribution(
                "high_conf_doc_ratio",
                f"High-quality scan sources (+{SCORE_HIGH_CONF_BOOST})",
                float(SCORE_HIGH_CONF_BOOST),
            )
        )
    if sources.all_low_conf:
        contributions.append(
            RuleContribution(
                "all_low_conf",
                f"All sources low quality ({SCORE_ALL_LOW_CONF_PENALTY})",
                float(SCORE_ALL_LOW_CONF_PENALTY),
            )
        )
    if (
        total_docs >= SCORE_SINGLE_SOURCE_MIN_DOCS
        and sources.num_documents == 1
        and sources.mean_confidence < SCORE_SINGLE_SOURCE_CONF_THRESHOLD
    ):
        contributions.append(
            RuleContribution(
                "_rule_single_source",
                f"Single low-confidence source ({SCORE_SINGLE_SOURCE_PENALTY})",
                float(SCORE_SINGLE_SOURCE_PENALTY),
            )
        )


def _eval_user_indicators(contributions: list[RuleContribution], term: str) -> None:
    """User-defined positive/negative indicator pattern matches."""
    if not term:
        return
    from src.core.vocabulary.indicator_patterns import matches_negative, matches_positive

    if matches_negative(term):
        contributions.append(
            RuleContribution("matches_negative_indicator", "Matches your negative pattern (-5)", -5)
        )
    elif matches_positive(term):
        contributions.append(
            RuleContribution("matches_positive_indicator", "Matches your positive pattern (+5)", 5)
        )


def _eval_artifact_penalties(contributions: list[RuleContribution], term: str) -> None:
    """Artifact detection penalties: all-caps, digits, single letter, punctuation."""
    if not term:
        return

    alpha_chars = [c for c in term if c.isalpha()]
    if alpha_chars and all(c.isupper() for c in alpha_chars):
        contributions.append(
            RuleContribution("is_all_caps", "All uppercase, likely header (-12)", -12.0)
        )

    if term[0].isdigit():
        contributions.append(
            RuleContribution("has_leading_digit", "Starts with digit, likely line number (-8)", -8)
        )

    stripped = term.strip()
    if len(stripped) == 1 and stripped.isalpha():
        contributions.append(
            RuleContribution("is_single_letter", "Single letter artifact (-15)", -15.0)
        )

    if term[-1] in ":;.,!?":
        contributions.append(
            RuleContribution("has_trailing_punctuation", "Trailing punctuation (-5)", -5.0)
        )
