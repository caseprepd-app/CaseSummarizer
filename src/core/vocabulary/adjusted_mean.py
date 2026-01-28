"""
Adjusted Mean Rarity Calculator.

Computes the mean rarity of word scores while excluding common filler words
(those below a configurable floor threshold). This prevents words like "of",
"the", "and" from dragging down the mean rarity of phrases that contain
genuinely rare or specialized terms.

Used by:
- rarity_filter.py: get_phrase_rarity_scores (RAKE/BM25 passthrough)
- rarity_filter.py: calculate_phrase_component_scores (NER phrase filtering)
- preference_learner_features.py: word_log_rarity_score (ML feature)
"""


def compute_adjusted_mean(
    scores: list[float],
    floor: float,
    filter_scores: list[float] | None = None,
) -> float:
    """
    Compute mean of scores, excluding entries where filter values are below floor.

    Words with filter values below the floor are considered common filler
    (e.g., "of", "the", "and") and excluded from the mean calculation.
    If ALL entries are below the floor, falls back to the full mean.

    The optional filter_scores parameter allows filtering based on one set
    of scores (e.g., linear rarity) while averaging a different set
    (e.g., log-transformed rarity). When not provided, scores are used
    for both filtering and averaging.

    Args:
        scores: Values to average (may be transformed, e.g., log-scaled)
        floor: Minimum filter value to include in the mean (e.g., 0.10)
        filter_scores: Scores to compare against floor. If None, uses scores.
            Useful when scores have been transformed (e.g., log-scaled) but
            the floor applies to the original linear scale.

    Returns:
        Adjusted mean score. Falls back to full mean if all entries are
        below floor, or 0.0 if scores is empty.

    Examples:
        # Simple case: filter and average same scores
        >>> compute_adjusted_mean([0.75, 0.00003, 0.000003, 0.05], 0.10)
        0.75

        # Transform case: filter by linear scores, average log scores
        >>> linear = [0.75, 0.00003, 0.000003]
        >>> log = [0.90, 0.25, 0.10]
        >>> compute_adjusted_mean(log, 0.10, filter_scores=linear)
        0.90
    """
    if not scores:
        return 0.0

    check = filter_scores if filter_scores is not None else scores
    adjusted = [s for s, f in zip(scores, check) if f >= floor]
    if adjusted:
        return sum(adjusted) / len(adjusted)

    # All below floor - fall back to full mean
    return sum(scores) / len(scores)
