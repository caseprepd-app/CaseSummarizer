"""
Hallucination Verification Configuration.

Thresholds for classifying spans based on hallucination probability
and computing display styles. Used by HallucinationVerifier and QAPanel.

Score Interpretation:
    - LettuceDetect returns "probability of hallucination" (0.0 to 1.0)
    - LOWER scores = more reliable (supported by context)
    - HIGHER scores = less reliable (likely hallucinated)
"""

# Span classification thresholds
# These define the boundaries for color-coding answer text
HALLUCINATION_THRESHOLDS = {
    "verified": 0.30,  # < 0.30 = green (verified, strongly supported)
    "uncertain": 0.50,  # 0.30 - 0.50 = yellow (uncertain, borderline)
    "suspicious": 0.70,  # 0.50 - 0.70 = orange (suspicious, likely unsupported)
    "unreliable": 0.85,  # 0.70 - 0.85 = red (unreliable, probably hallucinated)
    # >= 0.85 = strikethrough (hallucinated, very high confidence)
}

# Overall answer rejection threshold
# If overall reliability falls below this, we don't show the answer
ANSWER_REJECTION_THRESHOLD = 0.50  # Reject if reliability < 50%

# Model configuration
VERIFIER_MODEL_PATH = "KRLabsOrg/lettucedect-base-modernbert-en-v1"

# Rejection message shown when answer confidence is too low
REJECTION_MESSAGE = (
    "Confidence in answer too low after verification step, " "declining to show answer..."
)


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
