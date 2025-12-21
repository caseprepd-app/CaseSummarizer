"""
Rarity-Based Phrase Filtering

This module filters vocabulary terms based on the rarity of their component words.

WHY THIS EXISTS:
----------------
RAKE and other algorithms sometimes extract multi-word phrases where the individual
words are extremely common (e.g., "the same", "left side", "read copy"). These phrases
score well algorithmically because they appear together frequently in the document,
but they provide no value for court reporter vocabulary prep.

The solution: For multi-word phrases, examine each component word's rarity in general
English usage. If the words are too common (high frequency in Google's word corpus),
the phrase is likely not specialized vocabulary worth learning.

HOW IT WORKS:
-------------
1. Load Google word frequency data and scale to 0.0-1.0 range
   - 0.0 = extremely rare (not in dataset or very low frequency)
   - 1.0 = extremely common (like "the", "and", "is")

2. For multi-word phrases, calculate:
   - max_commonality: The most common word in the phrase
   - mean_commonality: Average commonality across all words

3. Filter out phrases where:
   - Any word is too common (max exceeds threshold), OR
   - The average word is too common (mean exceeds threshold)

SCALING APPROACH:
-----------------
We use logarithmic scaling rather than rank-based ordering because:
- Log scaling preserves RELATIVE frequency differences
- Rank only tells you position, not magnitude ("the" vs "a" are ranks 1 and 2,
  but "the" appears 2x as often - rank hides this)
- Log compression handles the extreme skew in word frequencies
  (top words appear millions of times, rare words appear once)

Formula: scaled = log(count + 1) / log(max_count + 1)
"""

import math
from pathlib import Path

from src.config import (
    GOOGLE_WORD_FREQUENCY_FILE,
    PHRASE_MAX_COMMONALITY_THRESHOLD,
    PHRASE_MEAN_COMMONALITY_THRESHOLD,
)
from src.logging_config import debug_log


# Module-level cache for scaled frequencies (loaded once)
_scaled_frequencies: dict[str, float] | None = None
_max_frequency: int = 0


def _load_scaled_frequencies() -> dict[str, float]:
    """
    Load Google word frequencies and scale to 0.0-1.0 range.

    Returns:
        Dictionary mapping lowercase words to their commonality score.
        Score of 1.0 = most common word in dataset ("the")
        Score of 0.0 = not in dataset (extremely rare)

    The scaling uses log transformation to preserve relative frequency
    differences while compressing the extreme range of raw counts.
    """
    global _scaled_frequencies, _max_frequency

    if _scaled_frequencies is not None:
        return _scaled_frequencies

    if not GOOGLE_WORD_FREQUENCY_FILE.exists():
        debug_log(f"[RARITY] Frequency file not found: {GOOGLE_WORD_FREQUENCY_FILE}")
        _scaled_frequencies = {}
        return _scaled_frequencies

    # First pass: load raw frequencies and find maximum
    raw_frequencies: dict[str, int] = {}

    try:
        with open(GOOGLE_WORD_FREQUENCY_FILE, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word, count_str = parts
                    try:
                        count = int(count_str)
                        raw_frequencies[word.lower()] = count
                    except ValueError:
                        continue

        if not raw_frequencies:
            debug_log("[RARITY] No valid entries found in frequency file")
            _scaled_frequencies = {}
            return _scaled_frequencies

        # Find maximum frequency for scaling
        _max_frequency = max(raw_frequencies.values())
        log_max = math.log(_max_frequency + 1)

        # Second pass: scale all frequencies using log transformation
        # log(count + 1) / log(max + 1) gives us 0.0 to 1.0 range
        _scaled_frequencies = {
            word: math.log(count + 1) / log_max
            for word, count in raw_frequencies.items()
        }

        debug_log(f"[RARITY] Loaded {len(_scaled_frequencies)} words, "
                  f"max freq: {_max_frequency:,}")

    except Exception as e:
        debug_log(f"[RARITY] Error loading frequencies: {e}")
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


def calculate_phrase_component_scores(phrase: str) -> tuple[float, float, int]:
    """
    Calculate rarity metrics for a multi-word phrase based on its component words.

    This is the core logic for determining if a phrase contains words that are
    too common to be valuable vocabulary. Court reporters don't need to learn
    phrases like "the same" or "left side" - they need specialized terminology.

    KEY INSIGHT: A phrase with even ONE rare word might be worth keeping.
    We only want to filter phrases where ALL words are common.

    Args:
        phrase: The phrase to analyze (e.g., "cervical spine" or "the same")

    Returns:
        Tuple of (min_commonality, mean_commonality, word_count):
        - min_commonality: LOWEST commonality score (the rarest word)
          If this is high, even the rarest word is common -> filter
          If this is low, at least one word is rare -> keep
        - mean_commonality: Average commonality across all words
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

    min_score = min(scores)  # The rarest word - this is what matters
    mean_score = sum(scores) / len(scores)

    return (min_score, mean_score, len(words))


def should_filter_phrase(phrase: str, is_person: bool = False) -> bool:
    """
    Determine if a phrase should be filtered out due to common component words.

    This function implements the filtering logic based on component word rarity.
    A phrase is filtered only if ALL its words are common - if even one word
    is rare, the phrase might be specialized vocabulary worth keeping.

    Args:
        phrase: The phrase to evaluate
        is_person: If True, skip filtering (person names are always kept)

    Returns:
        True if the phrase should be FILTERED OUT (removed)
        False if the phrase should be KEPT

    Decision logic:
        - Single words: Don't filter here (handled elsewhere)
        - Person names: Never filter (names like "John Smith" use common words)
        - Multi-word phrases: Filter only if the RAREST word is still too common
    """
    # Person names are exempt - "John Smith" uses common words but is valuable
    if is_person:
        return False

    min_common, mean_common, word_count = calculate_phrase_component_scores(phrase)

    # Single words are handled by algorithm-level filtering (NER rarity, stopwords)
    if word_count <= 1:
        return False

    # Filter if even the RAREST word is too common
    # This means ALL words in the phrase are common -> no vocabulary value
    # Example: "the same" - rarest word "same" is still very common -> filter
    # Example: "cervical spine" - "cervical" is uncommon -> keep
    if min_common > PHRASE_MAX_COMMONALITY_THRESHOLD:
        debug_log(f"[RARITY] Filtering '{phrase}': min_common={min_common:.2f} "
                  f"> {PHRASE_MAX_COMMONALITY_THRESHOLD} (all words common)")
        return True

    # Filter if the average commonality exceeds threshold
    # This catches phrases where words are generally common
    if mean_common > PHRASE_MEAN_COMMONALITY_THRESHOLD:
        debug_log(f"[RARITY] Filtering '{phrase}': mean_common={mean_common:.2f} "
                  f"> {PHRASE_MEAN_COMMONALITY_THRESHOLD}")
        return True

    return False


def filter_common_phrases(
    vocabulary: list[dict],
    term_key: str = "Term",
    is_person_key: str = "Is Person"
) -> list[dict]:
    """
    Filter vocabulary list to remove phrases with overly common component words.

    This is the main entry point for phrase rarity filtering. It should be called
    after all algorithms have contributed their terms, but before displaying to
    the user.

    WHY CENTRALIZED FILTERING:
    We filter here (after merging) rather than in each algorithm because:
    1. Single point of control for tuning thresholds
    2. Consistent behavior across NER, RAKE, BM25, and LLM sources
    3. Can consider cross-algorithm signals (e.g., term found by multiple algos)

    Args:
        vocabulary: List of term dictionaries from vocabulary extraction
        term_key: Key for the term text in each dictionary
        is_person_key: Key for the person flag in each dictionary

    Returns:
        Filtered vocabulary list with common phrases removed
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
        is_person = term_data.get(is_person_key, "No") == "Yes"

        if should_filter_phrase(term, is_person):
            removed_count += 1
            continue

        filtered.append(term_data)

    if removed_count > 0:
        debug_log(f"[RARITY] Filtered {removed_count}/{original_count} phrases "
                  f"with common component words")

    return filtered
