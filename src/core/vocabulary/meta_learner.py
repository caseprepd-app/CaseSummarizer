"""
Vocabulary Meta-Learner

Learns user preferences from feedback (thumbs up/down) to predict
which vocabulary terms a user would likely find useful.

Uses logistic regression for simplicity and interpretability.
The model learns from features like:
- Quality score from extraction algorithms
- Term frequency in document
- Google word frequency rank (rarity)
- Which algorithms detected the term
- Term type (Person, Medical, Technical, etc.)

Training occurs automatically when enough feedback accumulates.
The trained model persists to disk for use across sessions.

Training Configuration (from config.py):
- ML_MIN_SAMPLES: Minimum feedback entries before first training (30)
- ML_RETRAIN_THRESHOLD: New feedback to trigger retraining (10)
- ML_DECAY_HALF_LIFE_DAYS: Time for sample weight to decay to 50% (~1270 days / 3.5 years)
- ML_DECAY_WEIGHT_FLOOR: Minimum weight for old samples (0.55)

Time Decay Rationale:
Most early feedback flags universal false positives (common words incorrectly
identified as vocabulary). This feedback should persist strongly. Career changes
that affect preferences (new courthouse, new case types) are infrequent (~years).
"""

import math
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import (
    ML_DECAY_HALF_LIFE_DAYS,
    ML_DECAY_WEIGHT_FLOOR,
    ML_ENSEMBLE_MIN_SAMPLES,
    ML_MIN_SAMPLES,
    ML_RETRAIN_THRESHOLD,
    ML_SOURCE_WEIGHTS,
    ML_WEIGHT_THRESHOLDS,
    VOCAB_MODEL_PATH,
)
from src.logging_config import debug_log
from src.core.vocabulary.feedback_manager import FeedbackManager, get_feedback_manager
from src.core.vocabulary.rarity_filter import _load_scaled_frequencies
from src.core.vocabulary.term_sources import TermSources

# Medical suffixes for domain-specific feature (Session 76)
MEDICAL_SUFFIXES = (
    "itis",
    "osis",
    "ectomy",
    "pathy",
    "algia",
    "emia",
    "plasty",
    "scopy",
    "otomy",
    "ology",
    "oma",
    "esis",
    "iasis",
    "trophy",
    "megaly",
    "rrhea",
    "rrhage",
    "plegia",
)


def confidence_weighted_blend(prob_lr: float, prob_rf: float) -> float:
    """
    Blend two model predictions using confidence-weighted averaging.

    Confidence is measured as distance from 0.5 (uncertainty).
    Models with higher confidence contribute more to the final prediction.

    Args:
        prob_lr: Probability from logistic regression (0-1)
        prob_rf: Probability from random forest (0-1)

    Returns:
        Blended probability (0-1)

    Example:
        LR=0.8 (conf=0.3), RF=0.4 (conf=0.1)
        weights: LR=0.75, RF=0.25
        result: 0.8*0.75 + 0.4*0.25 = 0.70
    """
    conf_lr = abs(prob_lr - 0.5)
    conf_rf = abs(prob_rf - 0.5)
    total_conf = conf_lr + conf_rf

    # If both models are completely uncertain, return neutral
    if total_conf == 0:
        return 0.5

    weight_lr = conf_lr / total_conf
    weight_rf = conf_rf / total_conf

    return prob_lr * weight_lr + prob_rf * weight_rf


# Feature indices for interpretability
# Session 52: Removed unreliable type features (is_medical, is_technical, is_place, is_unknown)
# Only is_person is reliable (NER person detection)
# Session 54: Added source_doc_confidence to weight terms by OCR quality
# Session 68: Added corpus_familiarity_score and is_title_case
# Session 78: Added 7 TermSources features for per-document confidence tracking
FEATURE_NAMES = [
    # Frequency features (kept for backward compat, used by Session 76)
    "log_count",  # Replaces in_case_freq - better low-count discrimination
    "occurrence_ratio",  # Document-relative frequency (count / total_unique_terms)
    # Algorithm features
    "has_ner",
    "has_rake",
    "has_bm25",  # Added in Session 47 for per-algorithm tracking
    "is_person",  # NER person detection - the only reliable type info
    # Character/format features for artifact detection
    "has_trailing_punctuation",  # "Smith:", "Di Leo." - likely artifacts
    "has_leading_digit",  # "4 Ms. Di Leo", "17 SMITH" - line numbers
    "has_trailing_digit",  # "Smith 17", "Di Leo 2" - page/line number suffixes
    "word_count",  # 1-3 words = good, 4+ = suspicious over-extraction
    "is_all_caps",  # "PLAINTIFF'S EXHIBIT" - headers, not vocabulary
    "is_title_case",  # 1.0 if term.istitle() (proper nouns), 0.0 otherwise
    # Document quality feature (Session 54)
    "source_doc_confidence",  # OCR/extraction confidence (0-100) - lower = more OCR errors
    # Corpus familiarity features (Session 68)
    "corpus_familiarity_score",  # 0.0-1.0, proportion of corpus docs containing term
    # Session 76 features (9 total)
    "freq_dict_word_ratio",
    "all_words_in_freq_dict",
    "term_length",
    "vowel_ratio",
    "is_single_letter",
    "has_internal_digits",
    "has_medical_suffix",
    "has_repeated_chars",
    "contains_hyphen",
    # Session 78: TermSources-based per-document confidence features (7 total)
    # These provide richer signals about term reliability across source documents
    "num_source_documents",  # How many docs contain this term (more = more reliable)
    "doc_diversity_ratio",  # num_docs / total_docs (0-1, spread across corpus)
    "mean_doc_confidence",  # Count-weighted mean OCR confidence (0-1)
    "median_doc_confidence",  # Median confidence - robust to single bad scan (0-1)
    "confidence_std_dev",  # Consistency of confidence across docs (0-0.5)
    "high_conf_doc_ratio",  # % of source docs with confidence > 0.80 (0-1)
    "all_low_conf",  # 1 if ALL source docs have conf < 0.60 (red flag)
]


class VocabularyPreferenceLearner:
    """
    Learns user preferences for vocabulary terms using an ensemble approach.

    Uses a graduated training strategy:
    - 30+ samples: Logistic Regression only (interpretable, works with small data)
    - 200+ samples: Ensemble of LR + Random Forest (confidence-weighted blend)

    The model outputs a probability score [0, 1] where:
    - > 0.5: User would likely approve (thumbs up)
    - < 0.5: User would likely reject (thumbs down)

    Example:
        learner = VocabularyPreferenceLearner()
        if learner.is_trained:
            score = learner.predict_preference(term_data)
            print(f"User preference probability: {score:.2f}")
    """

    def __init__(self, model_path: Path | None = None):
        """
        Initialize the preference learner.

        Args:
            model_path: Path to save/load the trained model.
                       Defaults to VOCAB_MODEL_PATH from config.
        """
        self.model_path = Path(model_path) if model_path else VOCAB_MODEL_PATH

        # Models - LR always used, RF added when enough data
        self._lr_model: LogisticRegression | None = None
        self._rf_model: RandomForestClassifier | None = None
        self._scaler: StandardScaler | None = None

        # Training state
        self._is_trained = False
        self._ensemble_enabled = False

        # Sample counts for graduated ML weight (Session 55)
        self._user_sample_count = 0
        self._total_sample_count = 0

        # Load existing model if available
        self._load_model()

    @property
    def is_trained(self) -> bool:
        """Check if at least the LR model has been trained."""
        return self._is_trained and self._lr_model is not None

    @property
    def is_ensemble(self) -> bool:
        """Check if ensemble mode is active (both LR and RF trained)."""
        return self._ensemble_enabled and self._rf_model is not None

    @property
    def user_sample_count(self) -> int:
        """Get the number of user samples the model was trained on."""
        return self._user_sample_count

    def get_ml_weight(self) -> float:
        """
        Get the ML weight based on user sample count.

        The ML weight determines how much influence the ML prediction has
        on the final score vs the rule-based base score.

        Returns:
            Weight between 0.0 and 1.0
        """
        if not self.is_trained:
            return 0.0

        # Find the appropriate weight based on user sample count
        for threshold, weight in ML_WEIGHT_THRESHOLDS:
            if self._user_sample_count < threshold:
                return weight

        # Fallback (shouldn't reach here due to inf threshold)
        return ML_WEIGHT_THRESHOLDS[-1][1]

    def _extract_features(self, term_data: dict[str, Any]) -> np.ndarray:
        """
        Extract feature vector from term data.

        Session 76: Major feature overhaul
        - Removed: quality_score (circular), num_algorithms (redundant), freq_rank_normalized
        - Added: 9 new features including word-level frequency, vowel ratio, medical suffix

        Session 78: Added 7 TermSources-based per-document features:
        - num_source_documents, doc_diversity_ratio, mean_doc_confidence,
          median_doc_confidence, confidence_std_dev, high_conf_doc_ratio, all_low_conf
        - These provide richer signals about term reliability across source documents
        - Falls back to sensible defaults when TermSources not available (legacy data)

        Args:
            term_data: Dictionary with term information (from feedback CSV or extractor)
                      May include "sources" (TermSources) and "total_docs_in_session"

        Returns:
            numpy array of 30 features
        """
        # Get the term text
        term = str(term_data.get("Term", "") or term_data.get("term", "") or "")
        term_lower = term.lower()

        # === FREQUENCY FEATURES ===
        in_case_freq = float(term_data.get("in_case_freq", 1) or 1)

        # Log-transformed count - better discrimination at low counts
        log_count = math.log(max(in_case_freq, 1))

        # Document-relative frequency - normalizes for document size
        total_unique_terms = float(term_data.get("total_unique_terms", 0) or 0)
        if total_unique_terms > 0:
            occurrence_ratio = in_case_freq / total_unique_terms
        else:
            occurrence_ratio = in_case_freq / 100.0  # Fallback for legacy data

        # === ALGORITHM SOURCE FEATURES ===
        algorithms = str(term_data.get("algorithms", "")).lower()
        has_ner = 1.0 if "ner" in algorithms else 0.0
        has_rake = 1.0 if "rake" in algorithms else 0.0
        has_bm25 = 1.0 if "bm25" in algorithms else 0.0

        # === PERSON DETECTION ===
        is_person_val = term_data.get("is_person", 0)
        is_person = (
            float(is_person_val)
            if isinstance(is_person_val, (int, float))
            else (1.0 if str(is_person_val).lower() in ("1", "yes", "true") else 0.0)
        )

        # === CHARACTER/FORMAT FEATURES ===
        # Trailing punctuation (":Smith", "Di Leo.") - likely artifacts
        trailing_punct = ":;.,!?"
        has_trailing_punctuation = 1.0 if term and term[-1] in trailing_punct else 0.0

        # Leading/trailing digits
        has_leading_digit = 1.0 if term and term[0].isdigit() else 0.0
        has_trailing_digit = 1.0 if term and term[-1].isdigit() else 0.0

        # Word count
        words = term.split() if term else []
        word_count = float(len(words)) if words else 1.0

        # All caps detection
        alpha_chars = [c for c in term if c.isalpha()]
        is_all_caps = 1.0 if alpha_chars and all(c.isupper() for c in alpha_chars) else 0.0

        # Source document confidence (Session 54)
        source_doc_confidence_raw = float(term_data.get("source_doc_confidence", 100) or 100)
        source_doc_confidence = source_doc_confidence_raw / 100.0

        # Corpus familiarity (Session 68)
        corpus_familiarity_score = float(term_data.get("corpus_familiarity_score", 0) or 0)

        # Title case detection (Session 68)
        is_title_case = 1.0 if term and term.istitle() else 0.0

        # === NEW SESSION 76 FEATURES ===

        # Word-level frequency dictionary features
        freq_dict = _load_scaled_frequencies()
        words_lower = [w.lower() for w in words] if words else []
        if words_lower:
            words_in_dict = [1.0 if w in freq_dict else 0.0 for w in words_lower]
            freq_dict_word_ratio = sum(words_in_dict) / len(words_in_dict)
            all_words_in_freq_dict = 1.0 if all(w in freq_dict for w in words_lower) else 0.0
        else:
            freq_dict_word_ratio = 0.0
            all_words_in_freq_dict = 0.0

        # Term length (character count)
        term_length = float(len(term))

        # Vowel ratio - gibberish detector
        # Real words ~40% vowels, gibberish often 0% or very low
        vowels = set("aeiouAEIOU")
        if alpha_chars:
            vowel_count = sum(1 for c in alpha_chars if c in vowels)
            vowel_ratio = vowel_count / len(alpha_chars)
        else:
            vowel_ratio = 0.0

        # Single letter detection ("Q", "A" - transcript artifacts)
        is_single_letter = 1.0 if len(term.strip()) == 1 and term.strip().isalpha() else 0.0

        # Internal digits (digits not at start or end)
        # "Smith17" has trailing, "17Smith" has leading, "Sm1th" has internal
        if len(term) > 2:
            internal_chars = term[1:-1]
            has_internal_digits = 1.0 if any(c.isdigit() for c in internal_chars) else 0.0
        else:
            has_internal_digits = 0.0

        # Medical suffix detection - strong signal for legitimate vocabulary
        has_medical_suffix = (
            1.0 if any(term_lower.endswith(suffix) for suffix in MEDICAL_SUFFIXES) else 0.0
        )

        # Repeated characters (3+ in a row) - artifact detection
        # Catches "aaaa", ".....", "---"
        has_repeated_chars = 0.0
        if len(term) >= 3:
            for i in range(len(term) - 2):
                if term[i] == term[i + 1] == term[i + 2]:
                    has_repeated_chars = 1.0
                    break

        # Contains hyphen - often legitimate compound terms
        contains_hyphen = 1.0 if "-" in term else 0.0

        # === SESSION 78: TermSources-based per-document features ===
        # These features provide richer signals about term reliability by
        # tracking which source documents contributed each occurrence.
        #
        # When TermSources is available:
        # - Multiple high-confidence documents → more reliable term
        # - Single low-confidence document → potentially OCR error
        # - High confidence_std_dev → inconsistent quality (red flag)
        #
        # NOTE: An ML model trained on user feedback could learn to weight
        # these features based on the user's document types and preferences.

        # Check if TermSources is available
        sources = term_data.get("sources")
        total_docs_in_session = float(term_data.get("total_docs_in_session", 1) or 1)

        if isinstance(sources, TermSources) and sources.num_documents > 0:
            # Extract features from actual TermSources
            num_source_documents = float(sources.num_documents)
            doc_diversity_ratio = sources.doc_diversity_ratio(int(total_docs_in_session))
            mean_doc_confidence = sources.mean_confidence
            median_doc_confidence = sources.median_confidence
            confidence_std_dev = sources.confidence_std_dev
            high_conf_doc_ratio = sources.high_conf_doc_ratio
            all_low_conf = 1.0 if sources.all_low_conf else 0.0
        else:
            # Legacy fallback: no TermSources available
            # Use sensible defaults based on available data
            num_source_documents = 1.0
            doc_diversity_ratio = 1.0 / total_docs_in_session
            # Use source_doc_confidence as a single-doc approximation
            mean_doc_confidence = source_doc_confidence
            median_doc_confidence = source_doc_confidence
            confidence_std_dev = 0.0  # No variance with single source
            high_conf_doc_ratio = 1.0 if source_doc_confidence > 0.80 else 0.0
            all_low_conf = 1.0 if source_doc_confidence < 0.60 else 0.0

        return np.array(
            [
                # Frequency features (2)
                log_count,
                occurrence_ratio,
                # Algorithm features (3)
                has_ner,
                has_rake,
                has_bm25,
                # Type feature (1)
                is_person,
                # Original artifact features (6)
                has_trailing_punctuation,
                has_leading_digit,
                has_trailing_digit,
                word_count,
                is_all_caps,
                is_title_case,
                # Quality features (2)
                source_doc_confidence,
                corpus_familiarity_score,
                # Session 76 features (9)
                freq_dict_word_ratio,
                all_words_in_freq_dict,
                term_length,
                vowel_ratio,
                is_single_letter,
                has_internal_digits,
                has_medical_suffix,
                has_repeated_chars,
                contains_hyphen,
                # Session 78: TermSources features (7)
                num_source_documents,
                doc_diversity_ratio,
                mean_doc_confidence,
                median_doc_confidence,
                confidence_std_dev,
                high_conf_doc_ratio,
                all_low_conf,
            ]
        )

    def _calculate_sample_weight(
        self, timestamp_str: str, source: str = "user", user_sample_count: int = 0
    ) -> float:
        """
        Calculate combined weight for a feedback sample.

        Combines time-decay with source-based weighting (Session 55).
        User feedback is weighted higher than default feedback once
        the user has enough samples.

        Formula: weight = time_decay * source_weight

        Args:
            timestamp_str: ISO8601 timestamp from feedback record
            source: "user" or "default" - where the feedback came from
            user_sample_count: Number of user samples (for source weight lookup)

        Returns:
            Combined weight (time_decay * source_weight)
        """
        # Calculate time decay
        try:
            feedback_time = datetime.fromisoformat(timestamp_str)
            days_old = (datetime.now() - feedback_time).days

            # Exponential decay: halves every half_life days
            decay = 0.5 ** (days_old / ML_DECAY_HALF_LIFE_DAYS)

            # Apply floor - old feedback still matters
            time_weight = max(decay, ML_DECAY_WEIGHT_FLOOR)

        except (ValueError, TypeError):
            # Malformed timestamp - use moderate weight
            debug_log(f"[META-LEARNER] Invalid timestamp '{timestamp_str}', using default weight")
            time_weight = 0.75

        # Calculate source weight based on user sample count
        source_weight = 1.0
        for threshold, default_w, user_w in ML_SOURCE_WEIGHTS:
            if user_sample_count < threshold:
                source_weight = user_w if source == "user" else default_w
                break

        return time_weight * source_weight

    def train(self, feedback_manager: FeedbackManager | None = None) -> bool:
        """
        Train the model on accumulated feedback data.

        Args:
            feedback_manager: FeedbackManager instance. Uses global singleton if not provided.

        Returns:
            True if training succeeded, False if insufficient data
        """
        if feedback_manager is None:
            feedback_manager = get_feedback_manager()

        # Get all feedback records
        feedback_records = feedback_manager.export_training_data()

        # Filter to only records with +1 or -1 feedback (ignore cleared ratings)
        labeled_records = [
            r for r in feedback_records if r.get("feedback") in ("+1", "-1", "1", "-1", 1, -1)
        ]

        if len(labeled_records) < ML_MIN_SAMPLES:
            debug_log(
                f"[META-LEARNER] Insufficient training data: {len(labeled_records)} < {ML_MIN_SAMPLES}"
            )
            return False

        # Count user samples for source weighting (Session 55)
        user_sample_count = sum(1 for r in labeled_records if r.get("source") == "user")
        default_sample_count = len(labeled_records) - user_sample_count
        debug_log(
            f"[META-LEARNER] Training on {len(labeled_records)} feedback samples "
            f"({user_sample_count} user, {default_sample_count} default)"
        )

        # Store for graduated ML weight calculation
        self._user_sample_count = user_sample_count
        self._total_sample_count = len(labeled_records)

        # Extract features, labels, and combined weights (time-decay + source)
        X = []
        y = []
        sample_weights = []

        for record in labeled_records:
            features = self._extract_features(record)
            X.append(features)

            # Convert feedback to binary label
            feedback = str(record.get("feedback", "0"))
            label = 1 if feedback in ("+1", "1") else 0
            y.append(label)

            # Calculate combined weight (time-decay + source weighting)
            timestamp = record.get("timestamp", "")
            source = record.get("source", "user")  # Default to user if not specified
            weight = self._calculate_sample_weight(timestamp, source, user_sample_count)
            sample_weights.append(weight)

        X = np.array(X)
        y = np.array(y)
        sample_weights = np.array(sample_weights)

        # Log weight distribution
        debug_log(
            f"[META-LEARNER] Sample weights - min: {sample_weights.min():.2f}, "
            f"max: {sample_weights.max():.2f}, mean: {sample_weights.mean():.2f}"
        )

        # Check for class balance
        pos_count = np.sum(y)
        neg_count = len(y) - pos_count
        debug_log(f"[META-LEARNER] Class distribution: {pos_count} positive, {neg_count} negative")

        if pos_count < 3 or neg_count < 3:
            debug_log("[META-LEARNER] Insufficient class diversity for training")
            return False

        # Scale features for better convergence (shared by both models)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Always train Logistic Regression (works well with small data)
        debug_log("[PREF-LEARNER] Training Logistic Regression...")
        self._lr_model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs"
        )
        self._lr_model.fit(X_scaled, y, sample_weight=sample_weights)
        self._is_trained = True

        # Log LR feature importances
        if hasattr(self._lr_model, "coef_"):
            coefs = self._lr_model.coef_[0]
            importance = list(zip(FEATURE_NAMES, coefs))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            debug_log("[PREF-LEARNER] LR feature importance (top 5):")
            for name, coef in importance[:5]:
                debug_log(f"  {name}: {coef:.3f}")

        # Train Random Forest if enough data for ensemble
        n_samples = len(labeled_records)
        if n_samples >= ML_ENSEMBLE_MIN_SAMPLES:
            debug_log(
                f"[PREF-LEARNER] Training Random Forest (ensemble mode, {n_samples} samples)..."
            )
            self._rf_model = RandomForestClassifier(
                n_estimators=23,  # Few trees for speed; 200 samples doesn't need more
                max_depth=10,  # Prevent overfitting
                min_samples_leaf=5,  # Require 5 samples per leaf
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,  # Use all CPU cores
            )
            self._rf_model.fit(X_scaled, y, sample_weight=sample_weights)
            self._ensemble_enabled = True

            # Log RF feature importances
            rf_importance = list(zip(FEATURE_NAMES, self._rf_model.feature_importances_))
            rf_importance.sort(key=lambda x: x[1], reverse=True)
            debug_log("[PREF-LEARNER] RF feature importance (top 5):")
            for name, imp in rf_importance[:5]:
                debug_log(f"  {name}: {imp:.3f}")
        else:
            self._rf_model = None
            self._ensemble_enabled = False
            debug_log(
                f"[PREF-LEARNER] RF not trained (need {ML_ENSEMBLE_MIN_SAMPLES} samples, have {n_samples})"
            )

        # Save the trained model(s)
        self._save_model()

        # Reset pending count in feedback manager
        feedback_manager.reset_pending_count()

        return True

    def predict_preference(self, term_data: dict[str, Any]) -> float:
        """
        Predict user preference probability for a term.

        Uses LR-only when ensemble is not enabled, otherwise blends
        LR and RF predictions using confidence-weighted averaging.

        Args:
            term_data: Dictionary with term information

        Returns:
            Probability [0, 1] that user would approve this term.
            Returns 0.5 (neutral) if model is not trained.
        """
        if not self.is_trained:
            return 0.5

        features = self._extract_features(term_data)
        X = features.reshape(1, -1)

        # Scale features using trained scaler
        X_scaled = self._scaler.transform(X)

        # Get LR probability
        prob_lr = self._lr_model.predict_proba(X_scaled)[0][1]

        # If ensemble enabled, blend with RF
        if self.is_ensemble:
            prob_rf = self._rf_model.predict_proba(X_scaled)[0][1]
            return confidence_weighted_blend(prob_lr, prob_rf)

        return float(prob_lr)

    def predict_batch(self, terms_data: list[dict[str, Any]]) -> list[float]:
        """
        Predict user preference for multiple terms at once.

        Args:
            terms_data: List of term data dictionaries

        Returns:
            List of preference probabilities
        """
        if not self.is_trained or not terms_data:
            return [0.5] * len(terms_data)

        # Extract features for all terms
        X = np.array([self._extract_features(t) for t in terms_data])

        # Scale features
        X_scaled = self._scaler.transform(X)

        # Get LR probabilities
        probs_lr = self._lr_model.predict_proba(X_scaled)[:, 1]

        # If ensemble enabled, blend with RF
        if self.is_ensemble:
            probs_rf = self._rf_model.predict_proba(X_scaled)[:, 1]
            # Apply confidence-weighted blend to each pair
            blended = [confidence_weighted_blend(lr, rf) for lr, rf in zip(probs_lr, probs_rf)]
            return blended

        return probs_lr.tolist()

    def should_retrain(self, feedback_manager: FeedbackManager | None = None) -> bool:
        """
        Check if model should be retrained based on new feedback.

        Args:
            feedback_manager: FeedbackManager instance

        Returns:
            True if retraining is recommended
        """
        if feedback_manager is None:
            feedback_manager = get_feedback_manager()

        return feedback_manager.should_retrain(ML_MIN_SAMPLES, ML_RETRAIN_THRESHOLD)

    def _save_model(self) -> bool:
        """
        Save the trained model(s) and scaler to disk.

        Returns:
            True if save succeeded
        """
        if not self.is_trained:
            return False

        try:
            # Ensure directory exists
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

            # Save both models, scaler, state, and sample counts
            model_data = {
                "lr_model": self._lr_model,
                "rf_model": self._rf_model,  # May be None if not enough data
                "scaler": self._scaler,
                "ensemble_enabled": self._ensemble_enabled,
                "feature_names": FEATURE_NAMES,
                # Session 55: Sample counts for graduated ML weight
                "user_sample_count": self._user_sample_count,
                "total_sample_count": self._total_sample_count,
            }

            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f)

            mode = "ensemble" if self._ensemble_enabled else "LR-only"
            debug_log(f"[PREF-LEARNER] Model saved ({mode}) to {self.model_path}")
            return True

        except Exception as e:
            debug_log(f"[PREF-LEARNER] Failed to save model: {e}")
            return False

    def _load_model(self) -> bool:
        """
        Load previously trained model(s) from disk.

        Returns:
            True if load succeeded
        """
        if not self.model_path.exists():
            debug_log("[PREF-LEARNER] No existing model found")
            return False

        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)

            # Handle both old format (single model) and new format (ensemble)
            if "lr_model" in model_data:
                # New ensemble format
                self._lr_model = model_data.get("lr_model")
                self._rf_model = model_data.get("rf_model")
                self._ensemble_enabled = model_data.get("ensemble_enabled", False)
            else:
                # Old format - single model was LR
                self._lr_model = model_data.get("model")
                self._rf_model = None
                self._ensemble_enabled = False

            self._scaler = model_data.get("scaler")
            saved_feature_names = model_data.get("feature_names", [])

            # Session 55: Load sample counts (default to 0 for old models)
            self._user_sample_count = model_data.get("user_sample_count", 0)
            self._total_sample_count = model_data.get("total_sample_count", 0)

            # Check for feature count mismatch (model trained with different features)
            if len(saved_feature_names) != len(FEATURE_NAMES):
                debug_log(
                    f"[PREF-LEARNER] Feature count mismatch: saved model has "
                    f"{len(saved_feature_names)} features, current expects {len(FEATURE_NAMES)}. "
                    f"Model invalidated - will retrain with new features."
                )
                self._lr_model = None
                self._rf_model = None
                self._scaler = None
                self._is_trained = False
                self._ensemble_enabled = False
                return False

            if self._lr_model is not None and self._scaler is not None:
                self._is_trained = True
                mode = "ensemble" if self._ensemble_enabled else "LR-only"
                ml_weight = self.get_ml_weight()
                debug_log(
                    f"[PREF-LEARNER] Model loaded ({mode}) from {self.model_path} "
                    f"({self._user_sample_count} user samples, ML weight: {ml_weight:.0%})"
                )
                return True
            else:
                debug_log("[PREF-LEARNER] Invalid model data in file")
                return False

        except Exception as e:
            debug_log(f"[META-LEARNER] Failed to load model: {e}")
            return False

    def reset_to_default(self) -> bool:
        """
        Reset the model to the default (shipped) version.

        Copies the bundled default model over the user's personalized model.
        If no default model exists, deletes the user's model to start fresh.

        Returns:
            True if reset succeeded
        """
        from src.config import DEFAULT_VOCAB_MODEL_PATH

        try:
            # Clear current model state
            self._lr_model = None
            self._rf_model = None
            self._scaler = None
            self._is_trained = False
            self._ensemble_enabled = False
            self._user_sample_count = 0
            self._total_sample_count = 0

            # Check if default model exists (bundled with app)
            if DEFAULT_VOCAB_MODEL_PATH.exists():
                # Copy default model to user's model path
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(DEFAULT_VOCAB_MODEL_PATH, self.model_path)
                debug_log(f"[PREF-LEARNER] Reset to default model from {DEFAULT_VOCAB_MODEL_PATH}")

                # Reload the default model
                if self._load_model():
                    debug_log("[PREF-LEARNER] Default model loaded successfully")
                    return True
                else:
                    debug_log("[PREF-LEARNER] Warning: Default model exists but failed to load")
                    return False
            else:
                # No default model - just delete user's model to start fresh
                if self.model_path.exists():
                    self.model_path.unlink()
                    debug_log(f"[PREF-LEARNER] Deleted user model (no default available)")
                else:
                    debug_log("[PREF-LEARNER] No model to reset (already clean)")
                return True

        except Exception as e:
            debug_log(f"[PREF-LEARNER] Failed to reset model: {e}")
            return False


# Backward compatibility alias
VocabularyMetaLearner = VocabularyPreferenceLearner

# Global singleton instance
_preference_learner: VocabularyPreferenceLearner | None = None


def get_meta_learner() -> VocabularyPreferenceLearner:
    """
    Get the global VocabularyPreferenceLearner singleton.

    Returns:
        VocabularyPreferenceLearner instance
    """
    global _preference_learner
    if _preference_learner is None:
        _preference_learner = VocabularyPreferenceLearner()
    return _preference_learner
