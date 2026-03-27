"""
Key Excerpt Interestingness Scorer

Scores key excerpt candidates so the most useful ones sort to the top.
Uses vocabulary hits, named entities, word rarity, rejected terms from
feedback CSV, and gibberish detection.

Same excerpts are shown — this only changes their ordering.

TODO: Option 3 — collect thumbs-up/down feedback on key excerpts in the
UI and train a gradient-boosted tree on real user judgments once enough
labeled data exists (200+ ratings).
"""

import logging
import re

from src.core.utils.tokenizer import STOPWORDS

logger = logging.getLogger(__name__)

# Scoring weights
VOCAB_HIT_POINTS = 3  # Per vocab term (score >= 70) found in chunk
PERSON_HIT_POINTS = 5  # Per named entity (Is Person = Yes) found
RAREST_WORD_BONUS = 10  # If chunk has a very rare word
MEAN_RARITY_MAX_POINTS = 10  # Max points from average word rarity
REJECTED_TERM_PENALTY = -1  # Per rejected term (feedback = -1) found
GIBBERISH_WORD_PENALTY = -3  # Per gibberish word detected

# Rarity thresholds (scaled frequency: 0 = most common, 1 = rarest)
RAREST_WORD_THRESHOLD = 0.85  # Word must be this rare for bonus
WORD_PATTERN = re.compile(r"[a-zA-Z]{3,}")


def score_excerpt(
    text: str,
    vocab_terms: dict,
    person_terms: set[str],
    rejected_terms: set[str],
    frequency_rank_map: dict[str, int],
) -> float:
    """
    Score a key excerpt candidate by interestingness.

    Higher scores = more interesting to court reporters.
    Uses vocab overlap, NER hits, word rarity, rejected terms,
    and gibberish detection.

    Args:
        text: The excerpt chunk text
        vocab_terms: {lowercase_term: quality_score} from vocab extraction
        person_terms: Set of lowercase terms flagged as persons
        rejected_terms: Set of lowercase terms with feedback = -1
        frequency_rank_map: Google {word: rank} for rarity lookup

    Returns:
        Float score (higher = more interesting)
    """
    words = _extract_words(text)
    if not words:
        return 0.0

    text_lower = text.lower()
    score = 0.0

    # Feature 1: Vocab term hits (quality >= 70)
    score += _score_vocab_hits(text_lower, vocab_terms)

    # Feature 2: Person entity hits
    score += _score_person_hits(text_lower, person_terms)

    # Feature 3: Rarest word bonus
    score += _score_rarest_word(words, frequency_rank_map)

    # Feature 4: Mean word rarity (sans stopwords)
    score += _score_mean_rarity(words, frequency_rank_map)

    # Feature 5: Rejected term penalty
    score += _score_rejected_terms(text_lower, rejected_terms)

    # Feature 6: Gibberish penalty
    score += _score_gibberish(words)

    return score


def _extract_words(text: str) -> list[str]:
    """Extract lowercase alpha words (3+ chars) from text."""
    return [w.lower() for w in WORD_PATTERN.findall(text)]


def _score_vocab_hits(text_lower: str, vocab_terms: dict) -> float:
    """Score +3 per vocab term (score >= 70) found in the chunk."""
    hits = 0
    for term, quality in vocab_terms.items():
        if quality >= 70 and term in text_lower:
            hits += 1
    return hits * VOCAB_HIT_POINTS


def _score_person_hits(text_lower: str, person_terms: set[str]) -> float:
    """Score +5 per person entity found in the chunk."""
    hits = sum(1 for term in person_terms if term in text_lower)
    return hits * PERSON_HIT_POINTS


def _score_rarest_word(words: list[str], rank_map: dict[str, int]) -> float:
    """Score +10 if the chunk contains a very rare word."""
    if not rank_map:
        return 0.0

    total_words = len(rank_map)
    for word in words:
        if word in STOPWORDS:
            continue
        rank = rank_map.get(word)
        if rank is None:
            # Unknown word = potentially very rare (but could be gibberish)
            continue
        rarity = rank / total_words  # 0 = most common, 1 = rarest
        if rarity >= RAREST_WORD_THRESHOLD:
            return RAREST_WORD_BONUS
    return 0.0


def _score_mean_rarity(words: list[str], rank_map: dict[str, int]) -> float:
    """Score 0-10 based on average word rarity (stopwords excluded)."""
    if not rank_map:
        return 0.0

    total_words = len(rank_map)
    rarity_scores = []
    for word in words:
        if word in STOPWORDS:
            continue
        rank = rank_map.get(word)
        if rank is not None:
            rarity_scores.append(rank / total_words)

    if not rarity_scores:
        return 0.0

    mean_rarity = sum(rarity_scores) / len(rarity_scores)
    # Scale: 0.0 rarity → 0 points, 1.0 rarity → 10 points
    return min(mean_rarity * MEAN_RARITY_MAX_POINTS, MEAN_RARITY_MAX_POINTS)


def _score_rejected_terms(
    text_lower: str,
    rejected_terms: set[str],
) -> float:
    """Penalize -1 per rejected term found in the chunk."""
    if not rejected_terms:
        return 0.0
    hits = sum(1 for term in rejected_terms if term in text_lower)
    return hits * REJECTED_TERM_PENALTY


def _score_gibberish(words: list[str]) -> float:
    """Penalize -3 per gibberish word detected in the chunk."""
    try:
        from src.core.utils.gibberish_filter import GibberishFilter

        gf = GibberishFilter.get_instance()
    except Exception:
        return 0.0

    hits = 0
    for word in words:
        if len(word) >= 4 and gf.is_gibberish(word):
            hits += 1
    return hits * GIBBERISH_WORD_PENALTY
