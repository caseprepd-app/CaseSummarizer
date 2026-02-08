"""
Human-Readable Alternative Metadata for Person Names.

Builds reason strings explaining why a canonical name was chosen
and why each rejected variant was rejected.

Used by name_deduplicator.py and name_regularizer.py to attach
_alternatives and _canonical_reason metadata to term dicts.
"""

import logging

from src.core.utils.ocr_patterns import has_ocr_artifacts

logger = logging.getLogger(__name__)


def build_alternatives_from_scorer(
    variants: list[dict],
    canonical_term: str,
    branch: str,
    scored_list: list[tuple[str, float]] | None = None,
) -> tuple[list[dict], str]:
    """
    Build alternative metadata after CanonicalScorer picks a winner.

    Args:
        variants: All variant dicts with 'Term' and 'Occurrences' keys
        canonical_term: The winning term string
        branch: Selection branch ('single_known', 'none_known', 'multiple_known')
        scored_list: List of (term_str, score) tuples from _select_by_score,
                     or None if single_known branch was used

    Returns:
        Tuple of (alternatives_list, canonical_reason) where:
        - alternatives_list: List of dicts with 'variant', 'reason', 'frequency'
        - canonical_reason: Human-readable reason for the winner
    """
    alternatives = []
    canonical_reason = _format_canonical_reason(branch, canonical_term, scored_list)

    # Build winning score lookup for comparison
    score_map = {}
    if scored_list:
        score_map = {term: score for term, score in scored_list}

    winning_score = score_map.get(canonical_term, 0.0)

    for v in variants:
        term = v.get("Term", "")
        if term.lower() == canonical_term.lower():
            continue

        freq = v.get("Occurrences", 0)
        reason = _format_rejection_reason(term, branch, score_map, winning_score)
        alternatives.append({"variant": term, "reason": reason, "frequency": freq})

    return alternatives, canonical_reason


def build_alternatives_from_legacy(
    sorted_group: list[dict],
    freq_key: str,
) -> tuple[list[dict], str]:
    """
    Build alternative metadata for the legacy heuristic path.

    Args:
        sorted_group: Group sorted by score descending (winner is first)
        freq_key: Key for frequency field

    Returns:
        Tuple of (alternatives_list, canonical_reason)
    """
    if not sorted_group:
        return [], "No variants"

    canonical_reason = "Highest heuristic score (word validity + casing + frequency)"
    alternatives = []

    for entry in sorted_group[1:]:
        original = entry["original"]
        term = original.get("Term", "")
        freq = original.get(freq_key, 0)
        reason = "Lower heuristic score"

        # Add specifics
        cleaned = entry.get("cleaned", term)
        if cleaned == cleaned.upper() and len(cleaned) > 1:
            reason += "; ALL CAPS formatting penalty"
        if has_ocr_artifacts(cleaned):
            reason += "; OCR artifact penalty applied"

        alternatives.append({"variant": term, "reason": reason, "frequency": freq})

    return alternatives, canonical_reason


def build_single_word_alternative(
    single_entry: dict,
    target_term: str,
) -> dict:
    """
    Build an alternative entry for a single-word name absorbed into a full name.

    Args:
        single_entry: The absorbed single-word entry
        target_term: The full name that absorbed it

    Returns:
        Alternative dict with 'variant', 'reason', 'frequency'
    """
    term = single_entry.get("Term", "")
    freq = single_entry.get("Occurrences", 0)
    return {
        "variant": term,
        "reason": f"Single-word fragment of '{target_term}'",
        "frequency": freq,
    }


def build_titled_alternative(
    titled_entry: dict,
    title: str,
) -> dict:
    """
    Build an alternative entry for a title-prefixed variant merged into a full name.

    Args:
        titled_entry: The merged titled entry (e.g. "Dr. Jones")
        title: The title string (e.g. "dr.")

    Returns:
        Alternative dict with 'variant', 'reason', 'frequency'
    """
    term = titled_entry.get("Term", "")
    freq = titled_entry.get("Occurrences", 0)
    return {
        "variant": term,
        "reason": "Title-prefixed variant merged into full name",
        "frequency": freq,
    }


def build_fragment_alternative(
    fragment_term: str,
    fragment_freq: int,
    canonical_term: str,
) -> dict:
    """
    Build an alternative entry for a name fragment removed by regularization.

    Args:
        fragment_term: The fragment that was removed
        fragment_freq: Frequency of the fragment
        canonical_term: The canonical term it was a fragment of

    Returns:
        Alternative dict with 'variant', 'reason', 'frequency'
    """
    return {
        "variant": fragment_term,
        "reason": f"Name fragment of '{canonical_term}'",
        "frequency": fragment_freq,
    }


def build_typo_alternative(
    typo_term: str,
    typo_freq: int,
    branch: str,
    scored_list: list[tuple[str, float]] | None = None,
    winning_score: float = 0.0,
) -> dict:
    """
    Build an alternative entry for a typo variant removed by regularization.

    Args:
        typo_term: The typo term that was removed
        typo_freq: Frequency of the typo
        branch: Selection branch from CanonicalScorer
        scored_list: Score list for reason formatting
        winning_score: The winning term's score

    Returns:
        Alternative dict with 'variant', 'reason', 'frequency'
    """
    score_map = {}
    if scored_list:
        score_map = {term: score for term, score in scored_list}

    reason = _format_rejection_reason(typo_term, branch, score_map, winning_score)
    return {
        "variant": typo_term,
        "reason": reason,
        "frequency": typo_freq,
    }


def _format_canonical_reason(
    branch: str,
    term: str,
    scored_list: list[tuple[str, float]] | None = None,
) -> str:
    """
    Generate the winner's reason string.

    Args:
        branch: Selection branch ('single_known', 'none_known', 'multiple_known')
        term: The winning term
        scored_list: Optional score list for score display

    Returns:
        Human-readable reason string
    """
    if branch == "single_known":
        return "Only variant found in names dictionary"

    winning_score = 0.0
    if scored_list:
        for t, s in scored_list:
            if t.lower() == term.lower():
                winning_score = s
                break

    if branch == "none_known":
        return f"Highest confidence-weighted score ({winning_score:.1f})"

    if branch == "multiple_known":
        return f"Highest confidence-weighted score ({winning_score:.1f}) among known variants"

    return "Selected as canonical variant"


def _format_rejection_reason(
    term: str,
    branch: str,
    score_map: dict[str, float],
    winning_score: float,
) -> str:
    """
    Generate a human-readable rejection reason for a variant.

    Args:
        term: The rejected variant
        branch: Selection branch
        score_map: Mapping of term -> score
        winning_score: The winning term's score

    Returns:
        Human-readable rejection reason
    """
    reasons = []

    if branch == "single_known":
        reasons.append("Not found in names dictionary")
    elif branch in ("none_known", "multiple_known"):
        term_score = score_map.get(term, 0.0)
        reasons.append(f"Lower confidence-weighted score ({term_score:.1f})")

    if has_ocr_artifacts(term):
        reasons.append("OCR artifact penalty applied")

    if term == term.upper() and len(term) > 1:
        reasons.append("ALL CAPS formatting penalty")

    return "; ".join(reasons) if reasons else "Lower-ranked variant"
