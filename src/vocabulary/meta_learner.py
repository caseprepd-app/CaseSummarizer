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
- ML_DECAY_HALF_LIFE_DAYS: Time for sample weight to decay to 50% (365 days)
- ML_DECAY_WEIGHT_FLOOR: Minimum weight for old samples (0.55)

Time Decay Rationale:
Most early feedback flags universal false positives (common words incorrectly
identified as vocabulary). This feedback should persist strongly. Career changes
that affect preferences (new courthouse, new case types) are infrequent (~years).
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import (
    ML_DECAY_HALF_LIFE_DAYS,
    ML_DECAY_WEIGHT_FLOOR,
    ML_MIN_SAMPLES,
    ML_RETRAIN_THRESHOLD,
    VOCAB_MODEL_PATH,
)
from src.logging_config import debug_log
from src.vocabulary.feedback_manager import FeedbackManager, get_feedback_manager


# Feature indices for interpretability
FEATURE_NAMES = [
    "quality_score",
    "in_case_freq",
    "freq_rank_normalized",
    "num_algorithms",
    "has_ner",
    "has_rake",
    "has_bm25",  # Added in Session 47 for per-algorithm tracking
    "is_person",
    "is_medical",
    "is_technical",
    "is_place",
    "is_unknown",
]


class VocabularyMetaLearner:
    """
    Meta-learner that predicts user preference for vocabulary terms.

    Uses logistic regression to learn from user feedback (thumbs up/down)
    and predict which terms a user would likely approve in the future.

    The model outputs a probability score [0, 1] where:
    - > 0.5: User would likely approve (thumbs up)
    - < 0.5: User would likely reject (thumbs down)

    Example:
        learner = VocabularyMetaLearner()
        if learner.is_trained:
            score = learner.predict_preference(term_data)
            print(f"User preference probability: {score:.2f}")
    """

    def __init__(self, model_path: Path | None = None):
        """
        Initialize the meta-learner.

        Args:
            model_path: Path to save/load the trained model.
                       Defaults to VOCAB_MODEL_PATH from config.
        """
        self.model_path = Path(model_path) if model_path else VOCAB_MODEL_PATH
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._is_trained = False

        # Load existing model if available
        self._load_model()

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained and self._model is not None

    def _extract_features(self, term_data: dict[str, Any]) -> np.ndarray:
        """
        Extract feature vector from term data.

        Args:
            term_data: Dictionary with term information (from feedback CSV or extractor)
                      Expected keys: quality_score, in_case_freq, freq_rank, algorithms, type

        Returns:
            numpy array of features
        """
        # Numeric features (with defaults for missing data)
        quality_score = float(term_data.get("quality_score", 0) or 0)
        in_case_freq = float(term_data.get("in_case_freq", 1) or 1)

        # Normalize freq_rank (0 = not in dataset = rarest, higher = more common)
        freq_rank = float(term_data.get("freq_rank", 0) or 0)
        # Normalize to [0, 1] where 0 = very rare, 1 = very common
        freq_rank_normalized = min(freq_rank / 300000, 1.0) if freq_rank > 0 else 0.0

        # Algorithm source features (binary)
        algorithms = str(term_data.get("algorithms", "")).lower()
        has_ner = 1.0 if "ner" in algorithms else 0.0
        has_rake = 1.0 if "rake" in algorithms else 0.0
        has_bm25 = 1.0 if "bm25" in algorithms else 0.0
        num_algorithms = has_ner + has_rake + has_bm25  # Count of algorithms that found term

        # Type features (one-hot encoding)
        term_type = str(term_data.get("type", "Unknown")).lower()
        is_person = 1.0 if term_type == "person" else 0.0
        is_medical = 1.0 if term_type == "medical" else 0.0
        is_technical = 1.0 if term_type == "technical" else 0.0
        is_place = 1.0 if term_type == "place" else 0.0
        is_unknown = 1.0 if term_type == "unknown" else 0.0

        return np.array([
            quality_score,
            in_case_freq,
            freq_rank_normalized,
            num_algorithms,
            has_ner,
            has_rake,
            has_bm25,
            is_person,
            is_medical,
            is_technical,
            is_place,
            is_unknown,
        ])

    def _calculate_sample_weight(self, timestamp_str: str) -> float:
        """
        Calculate time-decay weight for a feedback sample.

        Uses exponential decay with a floor to ensure old feedback
        still contributes meaningfully. Most early feedback flags
        universal false positives that should persist.

        Formula: weight = max(floor, 0.5^(days_old / half_life))

        Args:
            timestamp_str: ISO8601 timestamp from feedback record

        Returns:
            Weight between ML_DECAY_WEIGHT_FLOOR and 1.0
        """
        try:
            # Parse ISO8601 timestamp
            feedback_time = datetime.fromisoformat(timestamp_str)
            days_old = (datetime.now() - feedback_time).days

            # Exponential decay: halves every half_life days
            decay = 0.5 ** (days_old / ML_DECAY_HALF_LIFE_DAYS)

            # Apply floor - old feedback still matters
            weight = max(decay, ML_DECAY_WEIGHT_FLOOR)

            return weight

        except (ValueError, TypeError):
            # Malformed timestamp - use moderate weight
            debug_log(f"[META-LEARNER] Invalid timestamp '{timestamp_str}', using default weight")
            return 0.75

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
            r for r in feedback_records
            if r.get("feedback") in ("+1", "-1", "1", "-1", 1, -1)
        ]

        if len(labeled_records) < ML_MIN_SAMPLES:
            debug_log(f"[META-LEARNER] Insufficient training data: {len(labeled_records)} < {ML_MIN_SAMPLES}")
            return False

        debug_log(f"[META-LEARNER] Training on {len(labeled_records)} feedback samples")

        # Extract features, labels, and time-decay weights
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

            # Calculate time-decay weight for this sample
            timestamp = record.get("timestamp", "")
            weight = self._calculate_sample_weight(timestamp)
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

        # Scale features for better convergence
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Train logistic regression with balanced class weights and time-decay sample weights
        self._model = LogisticRegression(
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        self._model.fit(X_scaled, y, sample_weight=sample_weights)

        self._is_trained = True

        # Log feature importances (coefficients)
        if hasattr(self._model, 'coef_'):
            coefs = self._model.coef_[0]
            importance = list(zip(FEATURE_NAMES, coefs))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            debug_log("[META-LEARNER] Feature importance (top 5):")
            for name, coef in importance[:5]:
                debug_log(f"  {name}: {coef:.3f}")

        # Save the trained model
        self._save_model()

        # Reset pending count in feedback manager
        feedback_manager.reset_pending_count()

        return True

    def predict_preference(self, term_data: dict[str, Any]) -> float:
        """
        Predict user preference probability for a term.

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

        # Get probability of positive class (thumbs up)
        prob = self._model.predict_proba(X_scaled)[0][1]

        return float(prob)

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

        # Scale and predict
        X_scaled = self._scaler.transform(X)
        probs = self._model.predict_proba(X_scaled)[:, 1]

        return probs.tolist()

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
        Save the trained model and scaler to disk.

        Returns:
            True if save succeeded
        """
        if not self.is_trained:
            return False

        try:
            # Ensure directory exists
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model and scaler together
            model_data = {
                'model': self._model,
                'scaler': self._scaler,
                'feature_names': FEATURE_NAMES,
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            debug_log(f"[META-LEARNER] Model saved to {self.model_path}")
            return True

        except Exception as e:
            debug_log(f"[META-LEARNER] Failed to save model: {e}")
            return False

    def _load_model(self) -> bool:
        """
        Load a previously trained model from disk.

        Returns:
            True if load succeeded
        """
        if not self.model_path.exists():
            debug_log("[META-LEARNER] No existing model found")
            return False

        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self._model = model_data.get('model')
            self._scaler = model_data.get('scaler')
            saved_feature_names = model_data.get('feature_names', [])

            # Check for feature count mismatch (model trained with different features)
            if len(saved_feature_names) != len(FEATURE_NAMES):
                debug_log(
                    f"[META-LEARNER] Feature count mismatch: saved model has "
                    f"{len(saved_feature_names)} features, current expects {len(FEATURE_NAMES)}. "
                    f"Model invalidated - will retrain with new features."
                )
                self._model = None
                self._scaler = None
                self._is_trained = False
                return False

            if self._model is not None and self._scaler is not None:
                self._is_trained = True
                debug_log(f"[META-LEARNER] Model loaded from {self.model_path}")
                return True
            else:
                debug_log("[META-LEARNER] Invalid model data in file")
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
        import shutil
        from src.config import DEFAULT_VOCAB_MODEL_PATH

        try:
            # Clear current model state
            self._model = None
            self._scaler = None
            self._is_trained = False

            # Check if default model exists (bundled with app)
            if DEFAULT_VOCAB_MODEL_PATH.exists():
                # Copy default model to user's model path
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(DEFAULT_VOCAB_MODEL_PATH, self.model_path)
                debug_log(f"[META-LEARNER] Reset to default model from {DEFAULT_VOCAB_MODEL_PATH}")

                # Reload the default model
                if self._load_model():
                    debug_log("[META-LEARNER] Default model loaded successfully")
                    return True
                else:
                    debug_log("[META-LEARNER] Warning: Default model exists but failed to load")
                    return False
            else:
                # No default model - just delete user's model to start fresh
                if self.model_path.exists():
                    self.model_path.unlink()
                    debug_log(f"[META-LEARNER] Deleted user model (no default available)")
                else:
                    debug_log("[META-LEARNER] No model to reset (already clean)")
                return True

        except Exception as e:
            debug_log(f"[META-LEARNER] Failed to reset model: {e}")
            return False


# Global singleton instance
_meta_learner: VocabularyMetaLearner | None = None


def get_meta_learner() -> VocabularyMetaLearner:
    """
    Get the global VocabularyMetaLearner singleton.

    Returns:
        VocabularyMetaLearner instance
    """
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = VocabularyMetaLearner()
    return _meta_learner
