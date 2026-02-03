"""
Model Training and Persistence for Vocabulary Meta-Learner

Handles:
- Training logistic regression and random forest models
- Saving/loading trained models to disk
- Time-decay and source-based sample weighting
- Confidence-weighted ensemble blending
- Model reset functionality
"""

import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import (
    DEFAULT_VOCAB_MODEL_PATH,
    ML_DECAY_HALF_LIFE_DAYS,
    ML_DECAY_WEIGHT_FLOOR,
    ML_ENSEMBLE_MIN_SAMPLES,
    ML_MIN_SAMPLES,
    ML_SOURCE_WEIGHTS,
    VOCAB_MODEL_PATH,
)
from src.core.vocabulary.preference_learner_features import FEATURE_NAMES, extract_features

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


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


def calculate_sample_weight(
    timestamp_str: str, source: str = "user", user_sample_count: int = 0
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
        logger.debug("Invalid timestamp '%s', using default weight", timestamp_str)
        time_weight = 0.75

    # Calculate source weight based on user sample count
    source_weight = 1.0
    for threshold, default_w, user_w in ML_SOURCE_WEIGHTS:
        if user_sample_count < threshold:
            source_weight = user_w if source == "user" else default_w
            break

    return time_weight * source_weight


def train_models(
    feedback_records: list[dict],
) -> tuple[
    LogisticRegression | None,
    RandomForestClassifier | None,
    StandardScaler | None,
    bool,
    int,
    int,
]:
    """
    Train the ML models on feedback data.

    Args:
        feedback_records: List of feedback records with labels

    Returns:
        Tuple of (lr_model, rf_model, scaler, ensemble_enabled, user_sample_count, total_sample_count)
        Returns (None, None, None, False, 0, 0) if insufficient data
    """
    # Filter to only records with +1 or -1 feedback (ignore cleared ratings)
    labeled_records = [r for r in feedback_records if r.get("feedback") in ("+1", "-1", "1", 1, -1)]

    if len(labeled_records) < ML_MIN_SAMPLES:
        logger.debug("Insufficient training data: %d < %d", len(labeled_records), ML_MIN_SAMPLES)
        return None, None, None, False, 0, 0

    # Count user samples for source weighting (Session 55)
    user_sample_count = sum(1 for r in labeled_records if r.get("source") == "user")
    default_sample_count = len(labeled_records) - user_sample_count
    total_sample_count = len(labeled_records)

    logger.debug(
        "Training on %d feedback samples (%d user, %d default)",
        total_sample_count,
        user_sample_count,
        default_sample_count,
    )

    # Extract features, labels, and combined weights (time-decay + source)
    X = []
    y = []
    sample_weights = []

    for record in labeled_records:
        features = extract_features(record)
        X.append(features)

        # Convert feedback to binary label
        feedback = str(record.get("feedback", "0"))
        label = 1 if feedback in ("+1", "1") else 0
        y.append(label)

        # Calculate combined weight (time-decay + source weighting)
        timestamp = record.get("timestamp", "")
        source = record.get("source", "user")  # Default to user if not specified
        weight = calculate_sample_weight(timestamp, source, user_sample_count)
        sample_weights.append(weight)

    X = np.array(X)
    y = np.array(y)
    sample_weights = np.array(sample_weights)

    # Log weight distribution
    logger.debug(
        "Sample weights - min: %.2f, max: %.2f, mean: %.2f",
        sample_weights.min(),
        sample_weights.max(),
        sample_weights.mean(),
    )

    # Check for class balance
    pos_count = np.sum(y)
    neg_count = len(y) - pos_count
    logger.debug("Class distribution: %d positive, %d negative", pos_count, neg_count)

    if pos_count < 3 or neg_count < 3:
        logger.debug("Insufficient class diversity for training")
        return None, None, None, False, 0, 0

    # Scale features for better convergence (shared by both models)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Always train Logistic Regression (works well with small data)
    logger.debug("Training Logistic Regression...")
    lr_model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs"
    )
    lr_model.fit(X_scaled, y, sample_weight=sample_weights)

    # Log LR feature importances
    if hasattr(lr_model, "coef_"):
        coefs = lr_model.coef_[0]
        importance = list(zip(FEATURE_NAMES, coefs, strict=False))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)
        logger.debug("LR feature importance (top 5):")
        for name, coef in importance[:5]:
            logger.debug("  %s: %.3f", name, coef)

    # Train Random Forest if enough data for ensemble
    rf_model = None
    ensemble_enabled = False

    if total_sample_count >= ML_ENSEMBLE_MIN_SAMPLES:
        logger.debug("Training Random Forest (ensemble mode, %d samples)...", total_sample_count)
        rf_model = RandomForestClassifier(
            n_estimators=23,  # Few trees for speed; 200 samples doesn't need more
            max_depth=10,  # Prevent overfitting
            min_samples_leaf=5,  # Require 5 samples per leaf
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
        )
        rf_model.fit(X_scaled, y, sample_weight=sample_weights)
        ensemble_enabled = True

        # Log RF feature importances
        rf_importance = list(zip(FEATURE_NAMES, rf_model.feature_importances_, strict=False))
        rf_importance.sort(key=lambda x: x[1], reverse=True)
        logger.debug("RF feature importance (top 5):")
        for name, imp in rf_importance[:5]:
            logger.debug("  %s: %.3f", name, imp)
    else:
        logger.debug(
            "RF not trained (need %d samples, have %d)", ML_ENSEMBLE_MIN_SAMPLES, total_sample_count
        )

    return lr_model, rf_model, scaler, ensemble_enabled, user_sample_count, total_sample_count


def save_model(
    model_path: Path,
    lr_model: LogisticRegression,
    rf_model: RandomForestClassifier | None,
    scaler: StandardScaler,
    ensemble_enabled: bool,
    user_sample_count: int,
    total_sample_count: int,
) -> bool:
    """
    Save the trained model(s) and scaler to disk.

    Args:
        model_path: Path to save the model
        lr_model: Trained logistic regression model
        rf_model: Trained random forest model (may be None)
        scaler: Fitted StandardScaler
        ensemble_enabled: Whether ensemble mode is active
        user_sample_count: Number of user samples used in training
        total_sample_count: Total samples used in training

    Returns:
        True if save succeeded
    """
    try:
        # Ensure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save both models, scaler, state, and sample counts
        model_data = {
            "lr_model": lr_model,
            "rf_model": rf_model,  # May be None if not enough data
            "scaler": scaler,
            "ensemble_enabled": ensemble_enabled,
            "feature_names": FEATURE_NAMES,
            # Session 55: Sample counts for graduated ML weight
            "user_sample_count": user_sample_count,
            "total_sample_count": total_sample_count,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        mode = "ensemble" if ensemble_enabled else "LR-only"
        logger.debug("Model saved (%s) to %s", mode, model_path)
        return True

    except Exception as e:
        logger.debug("Failed to save model: %s", e)
        return False


def load_model(
    model_path: Path,
) -> tuple[
    LogisticRegression | None,
    RandomForestClassifier | None,
    StandardScaler | None,
    bool,
    int,
    int,
    bool,
]:
    """
    Load previously trained model(s) from disk.

    Args:
        model_path: Path to the saved model

    Returns:
        Tuple of (lr_model, rf_model, scaler, ensemble_enabled, user_sample_count, total_sample_count, success)
    """
    if not model_path.exists():
        logger.debug("No existing model found")
        return None, None, None, False, 0, 0, False

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Handle both old format (single model) and new format (ensemble)
        if "lr_model" in model_data:
            # New ensemble format
            lr_model = model_data.get("lr_model")
            rf_model = model_data.get("rf_model")
            ensemble_enabled = model_data.get("ensemble_enabled", False)
        else:
            # Old format - single model was LR
            lr_model = model_data.get("model")
            rf_model = None
            ensemble_enabled = False

        scaler = model_data.get("scaler")
        saved_feature_names = model_data.get("feature_names", [])

        # Session 55: Load sample counts (default to 0 for old models)
        user_sample_count = model_data.get("user_sample_count", 0)
        total_sample_count = model_data.get("total_sample_count", 0)

        # Check for feature mismatch (model trained with different features)
        if saved_feature_names != FEATURE_NAMES:
            logger.debug(
                "Feature mismatch: saved model features differ from current. "
                "Model invalidated - will retrain with new features. "
                "(saved %d features, current %d features)",
                len(saved_feature_names),
                len(FEATURE_NAMES),
            )
            return None, None, None, False, 0, 0, False

        if lr_model is not None and scaler is not None:
            mode = "ensemble" if ensemble_enabled else "LR-only"
            logger.debug(
                "Model loaded (%s) from %s (%d user samples)",
                mode,
                model_path,
                user_sample_count,
            )
            return (
                lr_model,
                rf_model,
                scaler,
                ensemble_enabled,
                user_sample_count,
                total_sample_count,
                True,
            )

        logger.debug("Invalid model data in file")
        return None, None, None, False, 0, 0, False

    except Exception as e:
        logger.debug("Failed to load model: %s", e)
        return None, None, None, False, 0, 0, False


def reset_to_default(model_path: Path = VOCAB_MODEL_PATH) -> bool:
    """
    Reset the model to the default (shipped) version.

    Copies the bundled default model over the user's personalized model.
    If no default model exists, deletes the user's model to start fresh.

    Args:
        model_path: Path to the user's model file

    Returns:
        True if reset succeeded
    """
    try:
        # Check if default model exists (bundled with app)
        if DEFAULT_VOCAB_MODEL_PATH.exists():
            # Copy default model to user's model path
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(DEFAULT_VOCAB_MODEL_PATH, model_path)
            logger.debug("Reset to default model from %s", DEFAULT_VOCAB_MODEL_PATH)
            return True

        # No default model - just delete user's model to start fresh
        if model_path.exists():
            model_path.unlink()
            logger.debug("Deleted user model (no default available)")
        else:
            logger.debug("No model to reset (already clean)")
        return True

    except Exception as e:
        logger.debug("Failed to reset model: %s", e)
        return False
