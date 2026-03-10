"""
Score Explanation for Vocabulary Terms.

Combines explanations from all three scoring components:
1. Rules-based quality scorer — which rules fired and their point impact
2. Logistic Regression — per-feature coefficient x scaled value contributions
3. Random Forest (when ensemble active) — globally important features
   weighted by per-term relevance

Returns a deduplicated list of the top contributing factors from all models,
interleaved so each model gets fair representation. Because the ML models
retrain on new user feedback, the top factors can shift between sessions.
"""

import logging
from typing import Any, NamedTuple

import numpy as np

from src.core.vocabulary.preference_learner import get_meta_learner
from src.core.vocabulary.preference_learner_features import FEATURE_NAMES, extract_features
from src.core.vocabulary.score_explainer_rules import evaluate_rules

logger = logging.getLogger(__name__)


class Contribution(NamedTuple):
    """A single factor's contribution to the term's score.

    Attributes:
        feature_key: Dedup key matching ML feature names where applicable
        label: Human-readable explanation
        value: Signed contribution (positive = toward keep)
        source: Model component: "LR", "RF", or "Rules"
    """

    feature_key: str
    label: str
    value: float
    source: str


# Human-readable labels for ML features.
# (positive_reason, negative_reason) — used by both LR and RF explanations.
_FEATURE_LABELS: dict[str, tuple[str, str]] = {
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


def _label_for_feature(name: str, direction_positive: bool) -> str:
    """Get the human-readable label for a feature based on contribution direction."""
    labels = _FEATURE_LABELS.get(name, (name, name))
    return labels[0] if direction_positive else labels[1]


# ---------------------------------------------------------------------------
# Per-model contribution extractors
# ---------------------------------------------------------------------------


def _get_lr_contributions(lr_model: Any, X_scaled: np.ndarray) -> list[Contribution]:
    """
    Per-feature contributions from Logistic Regression.

    Each contribution = LR coefficient x scaled feature value.
    Positive = pushes toward "keep", negative = pushes toward "skip".

    Args:
        lr_model: Trained LogisticRegression model
        X_scaled: Scaled feature vector (1, n_features)

    Returns:
        List of Contribution sorted by |value| descending
    """
    coefficients = lr_model.coef_[0]
    contributions = coefficients * X_scaled[0]

    result = []
    for name, contrib in zip(FEATURE_NAMES, contributions, strict=True):
        if abs(contrib) < 0.01:
            continue  # Skip negligible contributions
        label = _label_for_feature(name, contrib > 0)
        result.append(Contribution(name, label, float(contrib), "LR"))

    result.sort(key=lambda c: abs(c.value), reverse=True)
    return result


def _get_rf_contributions(
    rf_model: Any, X_scaled: np.ndarray, lr_coefs: np.ndarray
) -> list[Contribution]:
    """
    Per-feature contributions from Random Forest (approximation).

    RF doesn't have per-instance coefficients like LR. We approximate
    per-term relevance as: feature_importance x |scaled_value|.
    Direction (sign) is inferred from LR coefficient x scaled value,
    since LR and RF generally agree on feature direction.

    Args:
        rf_model: Trained RandomForestClassifier model
        X_scaled: Scaled feature vector (1, n_features)
        lr_coefs: LR coefficients for direction inference

    Returns:
        List of Contribution sorted by |value| descending
    """
    importances = rf_model.feature_importances_
    scaled_values = X_scaled[0]

    # Per-term relevance: global importance x how extreme this term's value is
    relevance = importances * np.abs(scaled_values)

    # Direction: use sign of (LR coefficient x scaled value) as proxy
    signs = np.sign(lr_coefs * scaled_values)

    result = []
    for i, name in enumerate(FEATURE_NAMES):
        if relevance[i] < 0.001:
            continue  # Skip negligible relevance
        direction_positive = signs[i] >= 0
        label = _label_for_feature(name, direction_positive)
        signed_value = float(relevance[i]) * (1.0 if direction_positive else -1.0)
        result.append(Contribution(name, label, signed_value, "RF"))

    result.sort(key=lambda c: abs(c.value), reverse=True)
    return result


# ---------------------------------------------------------------------------
# Merge and deduplicate
# ---------------------------------------------------------------------------

_PER_SOURCE = 2  # Top N contributions from each model component


def _merge_contributions(
    lr: list[Contribution],
    rf: list[Contribution],
    rules: list[Contribution],
) -> list[Contribution]:
    """
    Merge top contributions from each model, deduplicating by feature_key.

    Interleaves contributions round-robin (LR#1, RF#1, Rules#1, LR#2, ...)
    so each model gets fair representation. Skips any feature_key already
    seen from an earlier model. Returns 4-6 unique contributions.

    Args:
        lr: LR contributions sorted by |value|
        rf: RF contributions sorted by |value| (empty if no ensemble)
        rules: Rules contributions sorted by |points|

    Returns:
        Deduplicated list of Contribution, interleaved by source
    """
    seen_keys: set[str] = set()
    merged: list[Contribution] = []

    # Interleave: take rank-0 from each source, then rank-1, etc.
    sources = [lr[:_PER_SOURCE], rf[:_PER_SOURCE], rules[:_PER_SOURCE]]
    for rank in range(_PER_SOURCE):
        for source_list in sources:
            if rank < len(source_list):
                contrib = source_list[rank]
                if contrib.feature_key not in seen_keys:
                    seen_keys.add(contrib.feature_key)
                    merged.append(contrib)

    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def explain_score(term_data: dict[str, Any], max_reasons: int = 6) -> dict[str, Any] | None:
    """
    Explain why a term received its score.

    Collects the top 2 contributing factors from each model component
    (LR, RF, Rules), deduplicates by feature key, and returns a merged
    list. The factors can shift between sessions as the ML models retrain.

    Args:
        term_data: Term data dictionary (same format as vocab table rows)
        max_reasons: Maximum number of reasons to return (default 6)

    Returns:
        Dict with keys:
            - score: float, the ML probability (0-1)
            - direction: str, "keep" or "skip"
            - reasons: list of (label, value, source) tuples
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
        # Extract and scale features for this term
        features = extract_features(term_data)
        X = features.reshape(1, -1)
        X_scaled = scaler.transform(X)

        # --- Collect contributions from each model component ---

        # 1. Logistic Regression: coefficient x scaled value (always available)
        lr_contribs = _get_lr_contributions(lr_model, X_scaled)

        # 2. Random Forest: importance x |scaled value| (only when ensemble active)
        rf_contribs: list[Contribution] = []
        if learner.is_ensemble and learner._rf_model is not None:
            rf_contribs = _get_rf_contributions(learner._rf_model, X_scaled, lr_model.coef_[0])

        # 3. Rules: which rules fired for this term's data
        rule_results = evaluate_rules(term_data)
        rules_contribs = [
            Contribution(r.feature_key, r.label, r.points, "Rules") for r in rule_results
        ]

        # --- Merge top 2 from each, deduplicated ---
        merged = _merge_contributions(lr_contribs, rf_contribs, rules_contribs)

        # Build reasons list: (label, value, source)
        reasons = [(c.label, c.value, c.source) for c in merged[:max_reasons]]

        # Get overall prediction
        prob = learner.predict_preference(term_data)
        direction = "keep" if prob >= 0.5 else "skip"
        model_status = "ensemble" if learner.is_ensemble else "lr"

        return {
            "score": prob,
            "direction": direction,
            "reasons": reasons,
            "model_status": model_status,
        }
    except Exception:
        logger.error("Failed to explain score", exc_info=True)
        return None
