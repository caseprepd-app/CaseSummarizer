"""
Hallucination Verification Configuration.

Thresholds for classifying spans based on hallucination probability
and computing display styles. Used by HallucinationVerifier and QAPanel.

Note: All threshold values are imported from src/config.py (single source of truth).
This module provides convenience functions for threshold lookups.

Score Interpretation:
    - LettuceDetect returns "probability of hallucination" (0.0 to 1.0)
    - LOWER scores = more reliable (supported by context)
    - HIGHER scores = less reliable (likely hallucinated)
"""

from src.config import (
    ANSWER_REJECTION_THRESHOLD,
    HALLUCINATION_MODEL,
    HALLUCINATION_MODEL_FAST,
    HALLUCINATION_MODEL_FASTEST,
    HALLUCINATION_REJECTION_MESSAGE,
    HALLUCINATION_THRESHOLDS,
)

# Re-export for backward compatibility
VERIFIER_MODEL_PATH = HALLUCINATION_MODEL
VERIFIER_MODEL_PATH_FAST = HALLUCINATION_MODEL_FAST
VERIFIER_MODEL_PATH_FASTEST = HALLUCINATION_MODEL_FASTEST
REJECTION_MESSAGE = HALLUCINATION_REJECTION_MESSAGE


def get_span_category(hallucination_prob: float) -> str:
    """
    Map hallucination probability to display category.

    Args:
        hallucination_prob: Probability that the span is hallucinated (0.0-1.0)

    Returns:
        Category name: "verified", "uncertain", "suspicious", "unreliable", or "hallucinated"
    """
    if hallucination_prob < HALLUCINATION_THRESHOLDS["verified"]:
        return "verified"
    elif hallucination_prob < HALLUCINATION_THRESHOLDS["uncertain"]:
        return "uncertain"
    elif hallucination_prob < HALLUCINATION_THRESHOLDS["suspicious"]:
        return "suspicious"
    elif hallucination_prob < HALLUCINATION_THRESHOLDS["unreliable"]:
        return "unreliable"
    else:
        return "hallucinated"


def get_reliability_level(overall_reliability: float) -> str:
    """
    Get display level for overall reliability score.

    Args:
        overall_reliability: Overall answer reliability (0.0-1.0, higher = better)

    Returns:
        Level name: "high", "medium", or "low"
    """
    if overall_reliability >= 0.80:
        return "high"
    elif overall_reliability >= ANSWER_REJECTION_THRESHOLD:
        return "medium"
    else:
        return "low"
