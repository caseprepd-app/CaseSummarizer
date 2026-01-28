"""
Rarity-Based Vocabulary Filtering

This module filters vocabulary terms based on the rarity of their component words.
It handles both SINGLE WORDS and MULTI-WORD PHRASES.

WHY THIS EXISTS:
----------------
RAKE and other algorithms sometimes extract terms that aren't valuable vocabulary:

1. SINGLE WORDS: Common words like "age", "body", "side" that aren't in STOPWORDS
   but are too common to be specialized vocabulary worth learning.

2. MULTI-WORD PHRASES: Phrases where the individual words are extremely common
   (e.g., "the same", "left side", "read copy"). These score well algorithmically
   because they appear together frequently, but provide no vocabulary prep value.

The solution: Examine each word's rarity in general English usage. If words are
too common (in the top X% of English vocabulary), filter them out.

RANK-BASED SCORING (Session 58):
--------------------------------
Score = rank / total_words (percentile position in vocabulary)
- 0.0 = most common word ("the", rank 1)
- 0.5 = median word (top 50%)
- 1.0 = rarest word in dataset

This directly answers: "What percentage of English words are more common than this?"
Court reporters know common English; they need only specialized terms.

HOW IT WORKS:
-------------
1. Load Google word frequency data and convert to rank-based percentile scores

2. For SINGLE words:
   - Filter if score < SINGLE_WORD_COMMONALITY_THRESHOLD (word is in top X%)
   - Example: "age" (score 0.0017) < 0.50 threshold -> FILTERED (top 0.17%)

3. For MULTI-word phrases:
   - Calculate min_score (highest-scoring = rarest word) and mean_score
   - Filter if min < PHRASE_MAX_COMMONALITY_THRESHOLD (all words in top X%)
   - Filter if mean < PHRASE_MEAN_COMMONALITY_THRESHOLD

4. PERSON names are always exempt (names like "Lee" or "John Smith" use common words)

WHY RANK-BASED INSTEAD OF LOG-BASED:
------------------------------------
Log scaling compresses the top end: "age" (0.784) looks similar to "the" (1.0)
even though "the" is 175x more common. This makes threshold tuning unintuitive.

Rank-based scoring is more intuitive:
- "age" at rank 579 = 0.0017 (top 0.17% of English)
- Threshold of 0.50 means "filter the most common half of English vocabulary"
- A court reporter filtering top 50% keeps only specialized terms
"""

import logging
import threading
from functools import lru_cache

from src.config import (
    GOOGLE_WORD_FREQUENCY_FILE,
    NON_NER_PHRASE_COMMON_WORD_FLOOR,
    NON_NER_PHRASE_MAX_PASSTHROUGH_THRESHOLD,
    NON_NER_PHRASE_MEAN_PASSTHROUGH_THRESHOLD,
    NON_NER_SINGLE_PASSTHROUGH_THRESHOLD,
    NON_NER_UNKNOWN_WORD_RARITY,
    PHRASE_MAX_COMMONALITY_THRESHOLD,
    PHRASE_MEAN_COMMONALITY_THRESHOLD,
    SINGLE_WORD_COMMONALITY_THRESHOLD,
)
from src.core.vocabulary.adjusted_mean import compute_adjusted_mean
from src.core.vocabulary.person_utils import is_person_entry
from src.user_preferences import get_user_preferences

logger = logging.getLogger(__name__)

# Module-level cache for scaled frequencies (loaded once) with thread-safe initialization
_scaled_frequencies: dict[str, float] | None = None
_max_frequency: int = 0
_freq_lock = threading.Lock()


def _load_scaled_frequencies() -> dict[str, float]:
    """
    Load Google word frequencies and convert to rank-based percentile scores.

    Thread-safe with double-check locking pattern.

    Returns:
        Dictionary mapping lowercase words to their rarity score.
        Score = rank / total_words (percentile position)
        - 0.0 = most common word ("the", rank 1)
        - 1.0 = rarest word in dataset

    RANK-BASED SCORING (Session 58):
    This is more intuitive than log scaling because:
    - "age" at rank 579 = 0.0017 (top 0.17% of English words)
    - Threshold of 0.50 means "filter the most common half of English"
    - Court reporters know common English; they need only specialized terms

    The score directly answers: "What percentage of English words are
    more common than this one?"
    """
    global _scaled_frequencies, _max_frequency

    # Fast path: already loaded
    if _scaled_frequencies is not None:
        return _scaled_frequencies

    # Slow path: need to load (with lock)
    with _freq_lock:
        # Double-check after acquiring lock
        if _scaled_frequencies is not None:
            return _scaled_frequencies

        if not GOOGLE_WORD_FREQUENCY_FILE.exists():
            logger.debug("Frequency file not found: %s", GOOGLE_WORD_FREQUENCY_FILE)
            _scaled_frequencies = {}
            return _scaled_frequencies

        # Load raw frequencies
        raw_frequencies: dict[str, int] = {}

        try:
            with open(GOOGLE_WORD_FREQUENCY_FILE, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        word, count_str = parts
                        try:
                            count = int(count_str)
                            raw_frequencies[word.lower()] = count
                        except ValueError:
                            continue

            if not raw_frequencies:
                logger.debug("No valid entries found in frequency file")
                _scaled_frequencies = {}
                return _scaled_frequencies

            # Sort by frequency (descending) to get ranks
            # Most common word = rank 0, rarest = rank (n-1)
            sorted_words = sorted(raw_frequencies.items(), key=lambda x: -x[1])
            total_words = len(sorted_words)
            _max_frequency = sorted_words[0][1] if sorted_words else 0

            # Convert to rank-based percentile (0.0 = common, 1.0 = rare)
            # rank / total gives the percentile position
            _scaled_frequencies = {
                word: rank / total_words for rank, (word, count) in enumerate(sorted_words)
            }

            logger.debug(
                "Loaded %s words (rank-based), max freq: %s",
                len(_scaled_frequencies),
                f"{_max_frequency:,}",
            )

        except Exception as e:
            logger.debug("Error loading frequencies: %s", e)
            _scaled_frequencies = {}

        return _scaled_frequencies


def get_word_commonality(word: str) -> float:
    """
    Get the commonality score for a single word.

    Args:
        word: The word to look up

    Returns:
        Commonality score from 0.0 (rare/unknown) to 1.0 (extremely common)
    """
    scaled = _load_scaled_frequencies()
    return scaled.get(word.lower().strip(), 0.0)


def is_common_word(word: str, top_n: int = 200000) -> bool:
    """
    Check if a word is in the top N most common English words.

    Uses the Google word frequency dataset. Words in the top N are considered
    "common" and unlikely to be part of a person's actual name.

    Args:
        word: The word to check
        top_n: Number of top words to consider "common" (default 200K of 333K)

    Returns:
        True if word is in top N most common words (or unknown), False otherwise

    Example:
        is_common_word("patient")  # True - common word
        is_common_word("napolitano")  # False - uncommon (likely a name)
    """
    scaled = _load_scaled_frequencies()
    if not scaled:
        return False  # No data - don't filter

    word_lower = word.lower().strip()
    if not word_lower:
        return True  # Empty = treat as common (filter)

    score = scaled.get(word_lower)
    if score is None:
        return False  # Unknown word = not common (might be a name)

    # Score = rank / total_words (lower = more common)
    # Check if rank < top_n, i.e., score < top_n / total_words
    total_words = len(scaled)
    threshold = top_n / total_words

    return score < threshold


@lru_cache(maxsize=2048)
def calculate_phrase_component_scores(phrase: str, floor: float = 0.0) -> tuple[float, float, int]:
    """
    Calculate rarity metrics for a multi-word phrase based on its component words.

    This is the core logic for determining if a phrase contains words that are
    too common to be valuable vocabulary. Court reporters don't need to learn
    phrases like "the same" or "left side" - they need specialized terminology.

    KEY INSIGHT: A phrase with even ONE rare word might be worth keeping.
    We only want to filter phrases where ALL words are common.

    Session 70: Added LRU cache since same phrases are often checked multiple times
    during vocabulary extraction.

    Args:
        phrase: The phrase to analyze (e.g., "cervical spine" or "the same")
        floor: Minimum score to include in the adjusted mean (e.g., 0.10).
            Words below this are treated as common filler and excluded.
            Default 0.0 means no filtering (backward compatible).

    Returns:
        Tuple of (min_commonality, adjusted_mean_commonality, word_count):
        - min_commonality: LOWEST commonality score (the rarest word)
          If this is high, even the rarest word is common -> filter
          If this is low, at least one word is rare -> keep
        - adjusted_mean_commonality: Mean commonality excluding filler words
          (if this is high, the phrase is generally common words)
        - word_count: Number of words in the phrase

    Examples:
        "the same" -> (0.90, 0.95, 2)  # Rarest word still common -> filter
        "cervical spine" -> (0.62, 0.64, 2)  # Has rare-ish words -> might keep
        "lumbar radiculopathy" -> (0.45, 0.52, 2)  # Has rare word -> keep
    """
    scaled = _load_scaled_frequencies()

    # Split phrase into words, filtering out empty strings
    words = [w.strip() for w in phrase.lower().split() if w.strip()]

    if len(words) == 0:
        return (0.0, 0.0, 0)

    if len(words) == 1:
        # Single words are handled by other filters (NER rarity check, stopwords)
        score = scaled.get(words[0], 0.0)
        return (score, score, 1)

    # Get commonality score for each word
    scores = [scaled.get(w, 0.0) for w in words]

    min_score = min(scores)
    mean_score = compute_adjusted_mean(scores, floor)

    return (min_score, mean_score, len(words))


def get_phrase_rarity_scores(phrase: str) -> tuple[float, float, int]:
    """
    Calculate rarity scores for a phrase, treating unknown words as rare.

    Unlike calculate_phrase_component_scores which gives 0.0 to unknown words
    (treating them as maximally common), this function assigns them a
    configurable rarity score (default 0.85) since words absent from the
    Google frequency dataset are likely proper nouns or specialized terms.

    The mean is "adjusted": common filler words (below the floor threshold)
    are excluded from the mean calculation to prevent words like "of", "the"
    from dragging down the average rarity of phrases with genuinely rare words.

    Args:
        phrase: The phrase to analyze

    Returns:
        Tuple of (max_rarity, adjusted_mean_rarity, word_count):
        - max_rarity: Highest rarity score (the rarest word)
        - adjusted_mean_rarity: Mean rarity excluding common filler words
        - word_count: Number of words in the phrase
    """
    scaled = _load_scaled_frequencies()
    prefs = get_user_preferences()
    unknown_score = prefs.get("non_ner_unknown_word_rarity", NON_NER_UNKNOWN_WORD_RARITY)

    words = [w.strip() for w in phrase.lower().split() if w.strip()]
    if not words:
        return (0.0, 0.0, 0)

    scores = [scaled.get(w, unknown_score) for w in words]
    max_rarity = max(scores)

    floor = prefs.get("non_ner_phrase_common_word_floor", NON_NER_PHRASE_COMMON_WORD_FLOOR)
    adjusted_mean_rarity = compute_adjusted_mean(scores, floor)

    return (max_rarity, adjusted_mean_rarity, len(words))


def should_passthrough_non_ner_term(term: str, term_data: dict) -> bool:
    """
    Determine if a non-Person RAKE/BM25 term should pass through rarity filtering.

    RAKE/BM25 find terms typed as "Technical" (not "Person"). The rarity filter
    treats words not in the Google frequency dataset as score 0.0 (maximally common),
    which incorrectly filters out proper nouns and rare names found by these algorithms.

    This passthrough check runs BEFORE the rarity filter removes a term. If the term
    was found by RAKE or BM25 and its rarity exceeds thresholds, it is kept.

    Args:
        term: The term text
        term_data: Full term dictionary with algorithm flags

    Returns:
        True if the term should be KEPT (passed through rarity filtering)
        False if normal rarity filtering should apply
    """
    # Only passthrough non-Person terms found by RAKE or BM25
    if term_data.get("Is Person", "No") == "Yes":
        return False
    if term_data.get("RAKE") != "Yes" and term_data.get("BM25") != "Yes":
        return False

    prefs = get_user_preferences()
    single_threshold = prefs.get(
        "non_ner_single_passthrough_threshold", NON_NER_SINGLE_PASSTHROUGH_THRESHOLD
    )
    phrase_max_threshold = prefs.get(
        "non_ner_phrase_max_passthrough_threshold", NON_NER_PHRASE_MAX_PASSTHROUGH_THRESHOLD
    )
    phrase_mean_threshold = prefs.get(
        "non_ner_phrase_mean_passthrough_threshold", NON_NER_PHRASE_MEAN_PASSTHROUGH_THRESHOLD
    )

    max_rarity, adjusted_mean_rarity, word_count = get_phrase_rarity_scores(term)

    if word_count == 0:
        return False

    if word_count == 1:
        passed = max_rarity >= single_threshold
    else:
        passed = (
            max_rarity >= phrase_max_threshold and adjusted_mean_rarity >= phrase_mean_threshold
        )

    if passed:
        logger.debug(
            "Passthrough non-NER term '%s': max=%.3f, adj_mean=%.3f, words=%s",
            term,
            max_rarity,
            adjusted_mean_rarity,
            word_count,
        )

    return passed


def should_filter_phrase(phrase: str, is_person: bool = False) -> bool:
    """
    Determine if a term should be filtered out due to common component words.

    RANK-BASED SCORING (Session 58):
    Score = rank / total_words (0.0 = most common, 1.0 = rarest)
    Lower score = more common word = should be filtered

    For SINGLE words:
        - Filter if score < threshold (word is too common)
        - Example: "age" (score 0.0017) < 0.50 threshold -> FILTERED

    For MULTI-word phrases:
        - Filter only if ALL words are common (even the rarest word is in top X%)
        - Example: "the same" - even rarest word is very common -> FILTERED
        - Example: "radiculopathy syndrome" - one rare word -> KEPT

    Args:
        phrase: The phrase or word to evaluate
        is_person: If True, skip filtering (person names are always kept)

    Returns:
        True if the term should be FILTERED OUT (removed)
        False if the term should be KEPT

    Decision logic:
        - Person names: Never filter (names like "John Smith" or "Lee")
        - Single words: Filter if score < threshold (in top X% of vocabulary)
        - Multi-word phrases: Filter if even the rarest word is in top X%

    Session 59: Thresholds now read from user preferences for GUI configurability.
    """
    # Person names are exempt - "John Smith" or "Lee" uses common words but is valuable
    if is_person:
        return False

    # Get thresholds from user preferences (Session 59)
    # Fall back to config.py values if not set
    prefs = get_user_preferences()
    single_threshold = prefs.get("single_word_rarity_threshold", SINGLE_WORD_COMMONALITY_THRESHOLD)
    phrase_threshold = prefs.get("phrase_rarity_threshold", PHRASE_MAX_COMMONALITY_THRESHOLD)
    phrase_mean_threshold = prefs.get(
        "phrase_mean_rarity_threshold", PHRASE_MEAN_COMMONALITY_THRESHOLD
    )

    floor = prefs.get("non_ner_phrase_common_word_floor", NON_NER_PHRASE_COMMON_WORD_FLOOR)
    min_common, mean_common, word_count = calculate_phrase_component_scores(phrase, floor)

    # Empty or invalid - don't filter (let other validation handle)
    if word_count == 0:
        return False

    # === SINGLE WORD FILTERING (Session 58 - rank-based) ===
    # Filter common single words that aren't valuable vocabulary
    # Lower score = more common (rank-based: 0.0 = most common)
    # Filter if word is in the top X% (score < threshold)
    if word_count == 1:
        if min_common < single_threshold:
            logger.debug(
                "Filtering single word '%s': rank_pct=%.4f < %s (top %.1f%%)",
                phrase,
                min_common,
                single_threshold,
                min_common * 100,
            )
            return True
        return False

    # === MULTI-WORD PHRASE FILTERING (rank-based) ===
    # Filter if even the RAREST word is too common (in top X%)
    # min_common = score of the RAREST word in the phrase
    # If this is < threshold, ALL words are in the top X% -> filter
    if min_common < phrase_threshold:
        logger.debug(
            "Filtering '%s': min_rank_pct=%.4f < %s (all words in top %.1f%%)",
            phrase,
            min_common,
            phrase_threshold,
            min_common * 100,
        )
        return True

    # Filter if the adjusted mean word rarity is too common
    if mean_common < phrase_mean_threshold:
        logger.debug(
            "Filtering '%s': adj_mean_rank_pct=%.4f < %s",
            phrase,
            mean_common,
            phrase_mean_threshold,
        )
        return True

    return False


def filter_common_phrases(
    vocabulary: list[dict],
    term_key: str = "Term",
) -> list[dict]:
    """
    Filter vocabulary list to remove common terms (single words AND phrases).

    This is the main entry point for rarity-based filtering. It should be called
    after all algorithms have contributed their terms, but before displaying to
    the user.

    FILTERING SCOPE:
    - Single words: Filters words with commonality > SINGLE_WORD_COMMONALITY_THRESHOLD
    - Multi-word phrases: Filters phrases where ALL component words are too common

    WHY CENTRALIZED FILTERING:
    We filter here (after merging) rather than in each algorithm because:
    1. Single point of control for tuning thresholds
    2. Consistent behavior across NER, RAKE, BM25, and LLM sources
    3. Catches common single words that individual algorithms miss (like RAKE)

    Session 70: Now uses centralized is_person_entry() for consistent person detection.

    Args:
        vocabulary: List of term dictionaries from vocabulary extraction
        term_key: Key for the term text in each dictionary

    Returns:
        Filtered vocabulary list with common terms removed
    """
    if not vocabulary:
        return vocabulary

    # Ensure frequency data is loaded
    _load_scaled_frequencies()

    original_count = len(vocabulary)
    filtered = []
    removed_count = 0

    for term_data in vocabulary:
        term = term_data.get(term_key, "")
        # Session 70: Use centralized person detection
        is_person = is_person_entry(term_data)

        if should_filter_phrase(term, is_person) and not should_passthrough_non_ner_term(
            term, term_data
        ):
            removed_count += 1
            continue

        filtered.append(term_data)

    if removed_count > 0:
        logger.debug(
            "Filtered %s/%s phrases with common component words",
            removed_count,
            original_count,
        )

    return filtered
