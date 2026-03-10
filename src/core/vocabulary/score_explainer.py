"""
Score Explanation for Vocabulary Terms.

Computes per-feature contributions to the ML score using logistic regression
coefficients. Returns human-readable explanations for the top factors
influencing a term's score.

Used by the "Why this score?" right-click menu item in the vocabulary table.
"""

import logging
from typing import Any

from src.core.vocabulary.preference_learner import get_meta_learner
from src.core.vocabulary.preference_learner_features import FEATURE_NAMES, extract_features

logger = logging.getLogger(__name__)

# Human-readable labels for features, grouped for clarity.
# Positive phrasing = pushes toward "keep", negative = pushes toward "skip".
_FEATURE_LABELS: dict[str, tuple[str, str]] = {
    # (positive_reason, negative_reason)
    # Count/frequency
    "count_bin_1": ("Appears only once", "Appears only once"),
    "count_bin_2_3": ("Appears 2-3 times", "Appears 2-3 times"),
    "count_bin_4_6": ("Appears 4-6 times", "Appears 4-6 times"),
    "count_bin_7_20": ("Appears 7-20 times", "Appears 7-20 times"),
    "count_bin_21_plus": ("Appears 21+ times", "Appears 21+ times"),
    "log_count": ("Appears frequently", "Appears infrequently"),
    "freq_per_1k_words": ("High density in document", "Low density in document"),
    # Algorithms
    "has_ner": ("Found by Named Entity Recognition", "Not found by NER"),
    "has_rake": ("Found by RAKE keyword extractor", "Not found by RAKE"),
    "has_bm25": ("Found by BM25 relevance scorer", "Not found by BM25"),
    "has_topicrank": ("Found by TopicRank", "Not found by TopicRank"),
    "has_medical_ner": ("Found by Medical NER", "Not found by Medical NER"),
    "has_yake": ("Found by YAKE keyword extractor", "Not found by YAKE"),
    "topicrank_score": ("High TopicRank score", "Low TopicRank score"),
    "yake_score": ("High YAKE importance", "Low YAKE importance"),
    "rake_score": ("High RAKE score", "Low RAKE score"),
    "bm25_score": ("High BM25 relevance", "Low BM25 relevance"),
    # Type
    "is_person": ("Recognized as a person name", "Not a person name"),
    # Artifacts
    "has_trailing_punctuation": ("Has trailing punctuation", "Has trailing punctuation"),
    "has_leading_digit": ("Starts with a digit", "Starts with a digit"),
    "has_trailing_digit": ("Ends with a digit", "Ends with a digit"),
    "word_count": ("Multi-word phrase", "Single word"),
    "is_all_caps": ("All uppercase (likely header)", "All uppercase (likely header)"),
    "is_title_case": ("Title case (proper noun)", "Not title case"),
    # Quality
    "source_doc_confidence": ("From high-quality scan", "From low-quality scan"),
    "corpus_common_term": ("Common across documents", "Common across documents"),
    # Word-level
    "freq_dict_word_ratio": ("Contains common English words", "Contains uncommon words"),
    "word_log_rarity_score": ("Rare/specialized term", "Common/everyday word"),
    "term_length": ("Longer term", "Shorter term"),
    "vowel_ratio": ("Normal vowel pattern", "Unusual vowel pattern"),
    "is_single_letter": ("Single letter", "Single letter"),
    "has_internal_digits": ("Has digits inside", "Has digits inside"),
    "has_medical_suffix": ("Has medical suffix", "Has medical suffix"),
    "has_repeated_chars": ("Has repeated characters", "Has repeated characters"),
    "contains_hyphen": ("Hyphenated term", "Hyphenated term"),
    # TermSources
    "mean_count_per_doc": ("Concentrated in documents", "Spread thin across docs"),
    "doc_diversity_ratio": ("Found in many documents", "Found in few documents"),
    "median_doc_confidence": ("Source docs are high quality", "Source docs are low quality"),
    "confidence_std_dev": ("Inconsistent doc quality", "Consistent doc quality"),
    "high_conf_doc_ratio": ("Mostly from good scans", "Mostly from poor scans"),
    "all_low_conf": ("All sources low quality", "All sources low quality"),
    # Names
    "is_in_names_dataset": ("Matches known name database", "Not in name database"),
    "names_word_ratio": ("Words match known names", "Words don't match names"),
    "has_forename_and_surname": ("Has first and last name", "Missing first or last name"),
    "name_country_spread": ("Name found across cultures", "Name not widely recognized"),
    "has_legal_suffix": ("Has legal suffix (-ant, -ee)", "Has legal suffix"),
    "has_title_prefix": ("Has title (Dr., Judge, etc.)", "Has title prefix"),
    "has_professional_suffix": ("Has credential (M.D., Esq.)", "Has credential suffix"),
    "max_consonant_run": ("Long consonant run (gibberish?)", "Normal letter pattern"),
    # Stop words
    "starts_with_stop_word": ("Starts with common word", "Starts with common word"),
    "ends_with_stop_word": ("Ends with common word", "Ends with common word"),
    # User patterns
    "matches_positive_indicator": ("Matches your positive pattern", "Matches positive pattern"),
    "matches_negative_indicator": ("Matches your negative pattern", "Matches negative pattern"),
}


def explain_score(term_data: dict[str, Any], max_reasons: int = 5) -> dict[str, Any] | None:
    """
    Explain why a term received its ML score.

    Computes per-feature contributions by multiplying the logistic regression
    coefficients by the scaled feature values. Returns the top positive and
    negative contributors in plain English.

    Args:
        term_data: Term data dictionary (same format as vocab table rows)
        max_reasons: Maximum number of reasons to return (default 5)

    Returns:
        Dict with keys:
            - score: float, the ML probability (0-1)
            - direction: str, "keep" or "skip"
            - reasons: list of (label: str, contribution: float) tuples,
              sorted by absolute contribution descending
            - model_status: str, "lr" or "ensemble"
        Returns None if the model is not trained.
    """
    learner = get_meta_learner()
    if not learner.is_trained:
        return None

    lr_model = learner._lr_model
    scaler = learner._scaler
    if lr_model is None or scaler is None:
        return None

    try:
        features = extract_features(term_data)
        X = features.reshape(1, -1)
        X_scaled = scaler.transform(X)

        # LR coefficients: positive = pushes toward class 1 (keep)
        coefficients = lr_model.coef_[0]
        intercept = lr_model.intercept_[0]

        # Per-feature contribution = coefficient * scaled_value
        contributions = coefficients * X_scaled[0]

        # Build labeled contributions
        labeled = []
        for i, (name, contrib) in enumerate(zip(FEATURE_NAMES, contributions, strict=True)):
            if abs(contrib) < 0.01:
                continue  # Skip negligible contributions
            labels = _FEATURE_LABELS.get(name, (name, name))
            if contrib > 0:
                label = labels[0]  # Positive reason
            else:
                label = labels[1]  # Negative reason
            labeled.append((label, float(contrib)))

        # Sort by absolute contribution (most impactful first)
        labeled.sort(key=lambda x: abs(x[1]), reverse=True)
        top_reasons = labeled[:max_reasons]

        # Get the actual prediction
        prob = learner.predict_preference(term_data)
        direction = "keep" if prob >= 0.5 else "skip"

        model_status = "ensemble" if learner.is_ensemble else "lr"

        return {
            "score": prob,
            "direction": direction,
            "reasons": top_reasons,
            "model_status": model_status,
        }
    except Exception:
        logger.error("Failed to explain score", exc_info=True)
        return None
