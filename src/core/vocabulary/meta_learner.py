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

import threading
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import (
    ML_MIN_SAMPLES,
    ML_RETRAIN_THRESHOLD,
    ML_RF_WEIGHT_THRESHOLDS,
    ML_WEIGHT_THRESHOLDS,
    VOCAB_MODEL_PATH,
)
from src.core.vocabulary.feedback_manager import FeedbackManager, get_feedback_manager
from src.core.vocabulary.meta_learner_features import FEATURE_NAMES, extract_features
from src.core.vocabulary.meta_learner_training import (
    confidence_weighted_blend,
    load_model,
    reset_to_default,
    save_model,
    train_models,
)
from src.logging_config import debug_log


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

        # Session 86: Auto-train if model doesn't exist but sufficient data is available
        # This ensures ML kicks in as soon as we have enough feedback, without needing
        # a manual retrain trigger
        if not self._is_trained:
            debug_log(
                "[META-LEARNER] No trained model - checking for sufficient data to auto-train"
            )
            self.train()  # train() internally checks for min samples

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

        # Find the appropriate weight based on total sample count
        # Session 86: Changed from _user_sample_count to _total_sample_count
        # so default samples contribute to ML weight (user feedback is upweighted anyway)
        for threshold, weight in ML_WEIGHT_THRESHOLDS:
            if self._total_sample_count < threshold:
                return weight

        # Fallback (shouldn't reach here due to inf threshold)
        return ML_WEIGHT_THRESHOLDS[-1][1]

    def _get_rf_blend_weight(self) -> float | None:
        """
        Get RF weight for ensemble blending based on sample count.

        Below 200 samples: returns fixed weight (0.0 to 0.4) based on thresholds.
        At 200+ samples: returns None to signal confidence-weighted blend.

        Returns:
            RF weight (0.0-0.4) for fixed blend, or None for confidence-weighted.
        """
        if self._total_sample_count >= 200:
            return None  # Use confidence-weighted blend at 200+

        for threshold, weight in ML_RF_WEIGHT_THRESHOLDS:
            if self._total_sample_count < threshold:
                return weight

        return None  # Fallback to confidence-weighted

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

        # Train the models
        lr_model, rf_model, scaler, ensemble_enabled, user_count, total_count = train_models(
            feedback_records
        )

        if lr_model is None:
            return False

        # Store the trained models
        self._lr_model = lr_model
        self._rf_model = rf_model
        self._scaler = scaler
        self._is_trained = True
        self._ensemble_enabled = ensemble_enabled
        self._user_sample_count = user_count
        self._total_sample_count = total_count

        # Save the trained model(s)
        save_model(
            self.model_path,
            lr_model,
            rf_model,
            scaler,
            ensemble_enabled,
            user_count,
            total_count,
        )

        # Reset pending count in feedback manager
        feedback_manager.reset_pending_count()

        return True

    def predict_preference(self, term_data: dict[str, Any]) -> float:
        """
        Predict user preference probability for a term.

        Uses graduated blending strategy:
        - LR-only when ensemble not enabled (< 40 samples)
        - Fixed-weight blend (10-40% RF) for 40-199 samples
        - Confidence-weighted blend at 200+ samples

        Args:
            term_data: Dictionary with term information

        Returns:
            Probability [0, 1] that user would approve this term.
            Returns 0.5 (neutral) if model is not trained.
        """
        if not self.is_trained:
            return 0.5

        features = extract_features(term_data)
        X = features.reshape(1, -1)

        # Scale features using trained scaler
        X_scaled = self._scaler.transform(X)

        # Get LR probability
        prob_lr = self._lr_model.predict_proba(X_scaled)[0][1]

        # If ensemble enabled, blend with RF using graduated weight
        if self.is_ensemble:
            prob_rf = self._rf_model.predict_proba(X_scaled)[0][1]

            # Get RF blend weight based on sample count
            rf_weight = self._get_rf_blend_weight()

            if rf_weight is None:
                # 200+ samples: use confidence-weighted blend
                return confidence_weighted_blend(prob_lr, prob_rf)
            elif rf_weight > 0:
                # 40-199 samples: use fixed-weight blend
                return prob_lr * (1 - rf_weight) + prob_rf * rf_weight
            # rf_weight == 0: fall through to return LR only

        return float(prob_lr)

    def predict_batch(self, terms_data: list[dict[str, Any]]) -> list[float]:
        """
        Predict user preference for multiple terms at once.

        Uses graduated blending strategy (same as predict_preference).

        Args:
            terms_data: List of term data dictionaries

        Returns:
            List of preference probabilities
        """
        if not self.is_trained or not terms_data:
            return [0.5] * len(terms_data)

        # Extract features for all terms
        X = np.array([extract_features(t) for t in terms_data])

        # Scale features
        X_scaled = self._scaler.transform(X)

        # Get LR probabilities
        probs_lr = self._lr_model.predict_proba(X_scaled)[:, 1]

        # If ensemble enabled, blend with RF using graduated weight
        if self.is_ensemble:
            probs_rf = self._rf_model.predict_proba(X_scaled)[:, 1]

            # Get RF blend weight based on sample count
            rf_weight = self._get_rf_blend_weight()

            if rf_weight is None:
                # 200+ samples: use confidence-weighted blend
                blended = [
                    confidence_weighted_blend(lr, rf)
                    for lr, rf in zip(probs_lr, probs_rf, strict=False)
                ]
                return blended
            elif rf_weight > 0:
                # 40-199 samples: use fixed-weight blend
                return [
                    lr * (1 - rf_weight) + rf * rf_weight
                    for lr, rf in zip(probs_lr, probs_rf, strict=False)
                ]
            # rf_weight == 0: fall through to return LR only

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

    def _load_model(self) -> bool:
        """
        Load previously trained model(s) from disk.

        Returns:
            True if load succeeded
        """
        (
            lr_model,
            rf_model,
            scaler,
            ensemble_enabled,
            user_count,
            total_count,
            success,
        ) = load_model(self.model_path)

        if success:
            self._lr_model = lr_model
            self._rf_model = rf_model
            self._scaler = scaler
            self._is_trained = True
            self._ensemble_enabled = ensemble_enabled
            self._user_sample_count = user_count
            self._total_sample_count = total_count

            ml_weight = self.get_ml_weight()
            mode = "ensemble" if ensemble_enabled else "LR-only"
            debug_log(
                f"[PREF-LEARNER] Model loaded ({mode}), "
                f"{user_count} user samples, ML weight: {ml_weight:.0%}"
            )

        return success

    def reset_to_default(self) -> bool:
        """
        Reset the model to the default (shipped) version.

        Copies the bundled default model over the user's personalized model.
        If no default model exists, deletes the user's model to start fresh.

        Returns:
            True if reset succeeded
        """
        # Clear current model state
        self._lr_model = None
        self._rf_model = None
        self._scaler = None
        self._is_trained = False
        self._ensemble_enabled = False
        self._user_sample_count = 0
        self._total_sample_count = 0

        # Perform the reset
        success = reset_to_default(self.model_path)

        # Reload the default model if reset succeeded
        if success:
            self._load_model()

        return success


# Backward compatibility alias
VocabularyMetaLearner = VocabularyPreferenceLearner

# Global singleton instance with thread-safe initialization
_preference_learner: VocabularyPreferenceLearner | None = None
_learner_lock = threading.Lock()


def get_meta_learner() -> VocabularyPreferenceLearner:
    """
    Get the global VocabularyPreferenceLearner singleton.

    Thread-safe with double-check locking pattern.

    Returns:
        VocabularyPreferenceLearner instance
    """
    global _preference_learner

    # Fast path: already initialized
    if _preference_learner is not None:
        return _preference_learner

    # Slow path: need to initialize (with lock)
    with _learner_lock:
        # Double-check after acquiring lock
        if _preference_learner is None:
            _preference_learner = VocabularyPreferenceLearner()
    return _preference_learner


# Re-export for backward compatibility
__all__ = [
    "FEATURE_NAMES",
    "VocabularyMetaLearner",
    "VocabularyPreferenceLearner",
    "confidence_weighted_blend",
    "extract_features",
    "get_meta_learner",
]
