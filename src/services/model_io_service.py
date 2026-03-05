"""
Model Import/Export Service

Provides export and import of user vocabulary models (.pkl)
and feedback history (.csv). Never exposes default/developer
files — only user-specific assets.

Exported/Imported paths:
- Model: %APPDATA%/CasePrepd/data/models/vocab_meta_learner.pkl
- Feedback: %APPDATA%/CasePrepd/data/feedback/user_feedback.csv
"""

import csv
import logging
import shutil
from datetime import datetime
from pathlib import Path

from src.config import VOCAB_MODEL_PATH
from src.core.vocabulary.feedback_manager import FEEDBACK_COLUMNS, FeedbackManager
from src.core.vocabulary.preference_learner_features import FEATURE_NAMES
from src.core.vocabulary.preference_learner_training import load_model

logger = logging.getLogger(__name__)


def _backup_path(original: Path) -> Path:
    """Generate a timestamped backup path for a file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return original.with_name(f"{original.stem}_backup_{ts}{original.suffix}")


def export_user_model(dest_path: Path) -> tuple[bool, str]:
    """
    Export the user's trained vocabulary model to a chosen path.

    Args:
        dest_path: Destination file path chosen by user

    Returns:
        (success, user-friendly message)
    """
    if not VOCAB_MODEL_PATH.exists():
        return (
            False,
            "No vocabulary model to export yet. Process some documents and provide feedback first.",
        )

    try:
        shutil.copy2(VOCAB_MODEL_PATH, dest_path)
        logger.info("Exported user model to %s", dest_path)
        return True, f"Model exported to {dest_path.name}"
    except Exception as e:
        logger.error("Failed to export model: %s", e, exc_info=True)
        return False, f"Export failed: {e}"


def import_user_model(src_path: Path) -> tuple[bool, str]:
    """
    Import a vocabulary model, validating feature compatibility.

    Creates a backup of the current model before import. On validation
    failure, restores the backup automatically.

    Args:
        src_path: Path to the .pkl file to import

    Returns:
        (success, user-friendly message)
    """
    had_existing = VOCAB_MODEL_PATH.exists()
    backup = None

    try:
        # Backup current model if it exists
        if had_existing:
            backup = _backup_path(VOCAB_MODEL_PATH)
            shutil.copy2(VOCAB_MODEL_PATH, backup)
            logger.info("Backed up current model to %s", backup)

        # Copy the new file in
        VOCAB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, VOCAB_MODEL_PATH)

        # Validate by loading (reuses existing feature_names check)
        lr, rf, scaler, ensemble, user_ct, total_ct, success = load_model(VOCAB_MODEL_PATH)

        if not success:
            # Restore backup
            if backup and backup.exists():
                shutil.copy2(backup, VOCAB_MODEL_PATH)
                backup.unlink()
            elif not had_existing:
                VOCAB_MODEL_PATH.unlink(missing_ok=True)

            return False, (
                f"This model was trained with incompatible features "
                f"(expected {len(FEATURE_NAMES)} features). "
                f"Import cancelled. Your previous model has been restored."
            )

        # Success — clean up backup
        if backup and backup.exists():
            backup.unlink()

        logger.info("Imported model from %s (trained on %d samples)", src_path, total_ct)
        return True, f"Model imported successfully (trained on {total_ct} samples)."

    except Exception as e:
        # Restore backup on any failure
        if backup and backup.exists():
            shutil.copy2(backup, VOCAB_MODEL_PATH)
            backup.unlink()
        elif not had_existing:
            VOCAB_MODEL_PATH.unlink(missing_ok=True)

        logger.error("Model import failed: %s", e, exc_info=True)
        return False, f"Import failed: {e}"


def export_user_feedback(dest_path: Path, feedback_mgr: FeedbackManager) -> tuple[bool, str]:
    """
    Export the user's feedback history CSV.

    Args:
        dest_path: Destination file path chosen by user
        feedback_mgr: FeedbackManager instance

    Returns:
        (success, user-friendly message)
    """
    src = feedback_mgr.user_feedback_file
    if not src.exists():
        return False, "No feedback history to export yet. Rate some vocabulary terms first."

    try:
        shutil.copy2(src, dest_path)
        # Count rows
        records = feedback_mgr.get_all_user_feedback()
        logger.info("Exported %d feedback records to %s", len(records), dest_path)
        return True, f"Exported {len(records)} feedback records to {dest_path.name}"
    except Exception as e:
        logger.error("Failed to export feedback: %s", e, exc_info=True)
        return False, f"Export failed: {e}"


def _validate_csv_columns(src_path: Path) -> tuple[bool, str, list[str]]:
    """
    Validate that a CSV has compatible feedback columns.

    Args:
        src_path: Path to CSV to validate

    Returns:
        (valid, message, actual_columns)
    """
    try:
        with open(src_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
    except Exception as e:
        return False, f"Could not read CSV: {e}", []

    if not headers:
        return False, "CSV file is empty or has no header row.", []

    headers_set = set(headers)
    required = {"term", "feedback", "timestamp"}

    if not required.issubset(headers_set):
        missing = required - headers_set
        return False, f"CSV is missing required columns: {', '.join(sorted(missing))}", headers

    expected_set = set(FEEDBACK_COLUMNS)
    extra = headers_set - expected_set
    missing_optional = expected_set - headers_set

    warning = ""
    if extra:
        # Tolerate columns from older versions (e.g. GLiNER_detection,
        # KeyBERT_detection, keybert_score) — they'll be silently dropped
        # by DictWriter(extrasaction="ignore") during write-back.
        logger.debug("Ignoring extra CSV columns from older version: %s", sorted(extra))
        warning += f" (dropped {len(extra)} unrecognized column(s): {', '.join(sorted(extra))})"

    if missing_optional:
        warning += f" (missing optional columns: {', '.join(sorted(missing_optional))} — defaults will be used during training)"

    return True, warning, headers


def import_user_feedback(
    src_path: Path, mode: str, feedback_mgr: FeedbackManager
) -> tuple[bool, str, int]:
    """
    Import feedback CSV in replace or append mode.

    Args:
        src_path: Path to the CSV file to import
        mode: "replace" or "append"
        feedback_mgr: FeedbackManager instance

    Returns:
        (success, user-friendly message, imported_row_count)
    """
    # Validate columns
    valid, col_msg, headers = _validate_csv_columns(src_path)
    if not valid:
        return False, col_msg, 0

    user_file = feedback_mgr.user_feedback_file

    try:
        # Read incoming records
        with open(src_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            new_records = list(reader)

        imported_count = len(new_records)

        if mode == "replace":
            # Backup current file
            if user_file.exists():
                backup = _backup_path(user_file)
                shutil.copy2(user_file, backup)
                logger.info("Backed up feedback to %s", backup)

            # Write new records
            user_file.parent.mkdir(parents=True, exist_ok=True)
            with open(user_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(new_records)

        elif mode == "append":
            # Read existing records
            existing = []
            if user_file.exists():
                with open(user_file, encoding="utf-8", newline="") as f:
                    existing = list(csv.DictReader(f))

            combined = existing + new_records
            user_file.parent.mkdir(parents=True, exist_ok=True)
            with open(user_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(combined)
        else:
            return False, f"Unknown mode: {mode}", 0

        action = "Replaced with" if mode == "replace" else "Appended"
        msg = f"{action} {imported_count} feedback records.{col_msg}"
        logger.info("Imported feedback (%s): %d records from %s", mode, imported_count, src_path)
        return True, msg, imported_count

    except Exception as e:
        logger.error("Feedback import failed: %s", e, exc_info=True)
        return False, f"Import failed: {e}", 0
