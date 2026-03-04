"""
Vocabulary Feedback Manager

Manages user feedback (thumbs up/down) on vocabulary terms.
Stores feedback in CSV format for ML training.

Two-File System:
- Default feedback: Ships with app (developer's training data)
- User feedback: Collected during normal use

The feedback data is used to train a meta-learner that adapts
to user preferences over time. User feedback is weighted higher
than default feedback once the user has enough samples.

CSV Schema:
- timestamp: ISO8601 datetime when feedback was recorded
- document_id: Hash/ID of the document being processed
- term: The vocabulary term
- feedback: +1 (thumbs up) or -1 (thumbs down)
- is_person: 1 if NER detected as person name, 0 otherwise
- algorithms: Comma-separated list of algorithms that detected the term
- NER_detection: Boolean - whether NER algorithm detected this term
- RAKE_detection: Boolean - whether RAKE algorithm detected this term
- BM25_detection: Boolean - whether BM25 algorithm detected this term
- algo_count: Number of algorithms that detected this term (sum of detection booleans)
- quality_score: Quality score at time of feedback
- occurrences: Term occurrence count
- rarity_rank: Google frequency rank

TermSources-based per-document features:
- num_source_documents: How many docs contain this term
- doc_diversity_ratio: num_docs / total_docs (0-1)
- mean_doc_confidence: Count-weighted mean OCR confidence (0-1)
- median_doc_confidence: Median confidence - robust to outliers (0-1)
- confidence_std_dev: Standard deviation of confidences (0-0.5)
- high_conf_doc_ratio: % of source docs with confidence > 0.80 (0-1)
- all_low_conf: 1 if ALL source docs have conf < 0.60, else 0
- total_docs_in_session: Total docs in extraction session (for diversity ratio)
"""

import csv
import hashlib
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import (
    DEBUG_MODE,
    DEFAULT_FEEDBACK_CSV,
    FEEDBACK_DIR,
    ML_MIN_SAMPLES,
    ML_RETRAIN_THRESHOLD,
    get_count_bin,
)
from src.core.vocabulary.term_sources import TermSources

logger = logging.getLogger(__name__)

# Required fields for valid feedback records
REQUIRED_FEEDBACK_FIELDS = {"term", "feedback", "timestamp"}


def _validate_record(record: dict) -> bool:
    """
    Check if feedback record has required fields.

    Args:
        record: Feedback record dictionary

    Returns:
        True if record has all required fields with non-empty values
    """
    for field in REQUIRED_FEEDBACK_FIELDS:
        value = record.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            return False
    return True


# CSV columns
# Uses "is_person" binary flag instead of unreliable "type" string.
# Includes TermSources-based per-document features (8 columns).
FEEDBACK_COLUMNS = [
    "timestamp",
    "document_id",
    "term",
    "feedback",
    "is_person",  # 1 if NER detected as person, 0 otherwise
    "algorithms",
    # Per-algorithm detection flags (all 8 algorithms)
    "NER_detection",
    "RAKE_detection",
    "BM25_detection",
    "TopicRank_detection",
    "MedicalNER_detection",
    "GLiNER_detection",
    "YAKE_detection",
    "KeyBERT_detection",
    "algo_count",
    "quality_score",
    "occurrences",
    "rarity_rank",
    # Per-algorithm numeric scores (0.0 if algorithm didn't find term)
    "topicrank_score",
    "yake_score",
    "keybert_score",
    "rake_score",
    "bm25_score",
    # TermSources-based per-document features
    "num_source_documents",  # How many docs contain this term
    "doc_diversity_ratio",  # num_docs / total_docs (0-1)
    "mean_doc_confidence",  # Count-weighted mean OCR confidence (0-1)
    "median_doc_confidence",  # Median confidence - robust to outliers (0-1)
    "confidence_std_dev",  # Standard deviation of confidences (0-0.5)
    "high_conf_doc_ratio",  # % of source docs with confidence > 0.80 (0-1)
    "all_low_conf",  # 1 if ALL source docs have conf < 0.60, else 0
    "total_docs_in_session",  # Total docs in extraction session
    "total_word_count",  # Total words in document(s) for scale-independent features
]


class FeedbackManager:
    """
    Manages user feedback on vocabulary terms.

    Two-File System:
    - Default feedback: Ships with app (developer's training data)
      Location: config/default_feedback.csv
    - User feedback: Collected during normal use
      Location: %APPDATA%/CasePrepd/feedback/user_feedback.csv

    Routing:
    - DEBUG_MODE=True: Feedback writes to default_feedback.csv (for development)
    - DEBUG_MODE=False: Feedback writes to user_feedback.csv (for end users)

    Provides:
    - Recording feedback (thumbs up/down) for terms
    - Persisting feedback to appropriate CSV file based on DEBUG_MODE
    - Loading combined feedback history for ML training (with source tags)
    - Caching feedback state for UI display

    The feedback is keyed by normalized term (lowercase) for
    consistent lookups across sessions.

    Example:
        manager = FeedbackManager()
        manager.record_feedback(term_data, +1, "doc123")
        rating = manager.get_rating("spondylosis")  # Returns +1, -1, or 0
    """

    def __init__(self, feedback_dir: Path | None = None, default_feedback_file: Path | None = None):
        """
        Initialize feedback manager.

        Args:
            feedback_dir: Directory to store user feedback files.
                         Defaults to %APPDATA%/CasePrepd/feedback/
            default_feedback_file: Path to default feedback CSV (for testing).
                         Defaults to config/default_feedback.csv
        """
        self.feedback_dir = Path(feedback_dir) if feedback_dir else FEEDBACK_DIR

        # Two-file system: default (shipped) + user (collected)
        # Default file: shipped data, or override for testing
        self.default_feedback_file = (
            Path(default_feedback_file) if default_feedback_file else DEFAULT_FEEDBACK_CSV
        )
        # User file goes in the feedback_dir (which may be overridden for testing)
        self.user_feedback_file = self.feedback_dir / "user_feedback.csv"

        # In-memory cache: normalized_term -> rating (+1, -1, or 0)
        # Only tracks USER feedback for display purposes
        self._cache: dict[str, int] = {}

        # Track terms rated in current GUI session (vs loaded from file)
        # Used to distinguish user clicks (darker green) from dataset entries (lighter green)
        self._session_rated: set[str] = set()

        # Track pending feedback count (for retraining threshold)
        self._pending_count = 0

        # Current document ID for feedback context
        self._current_doc_id: str = ""

        # Lock for thread-safe file operations (prevents race condition on rapid clicks)
        self._file_lock = threading.Lock()

        # Ensure directory exists
        self._ensure_directory()

        # Load existing user feedback into cache
        self._load_cache()

    def _ensure_directory(self):
        """Create feedback directory if it doesn't exist."""
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

    def _load_cache(self):
        """
        Load existing feedback from CSV into cache.

        Routing:
        - DEBUG_MODE=True: Load from default_feedback.csv (developer data)
        - DEBUG_MODE=False: Load from user_feedback.csv (user data)

        This ensures the GUI reflects the dataset that will be modified.
        """
        # Choose which file to load based on DEBUG_MODE
        target_file = self.default_feedback_file if DEBUG_MODE else self.user_feedback_file
        target_type = "default" if DEBUG_MODE else "user"

        try:
            with open(target_file, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    term = row.get("term", "").lower().strip()
                    feedback_str = row.get("feedback", "0")
                    try:
                        feedback = int(feedback_str)
                        self._cache[term] = feedback
                    except ValueError:
                        continue

            logger.debug(
                "Loaded %d %s feedback entries (DEBUG_MODE=%s)",
                len(self._cache),
                target_type,
                DEBUG_MODE,
            )
        except FileNotFoundError:
            logger.debug("No existing %s feedback file, starting fresh", target_type)
        except Exception as e:
            logger.debug("Error loading %s feedback: %s", target_type, e)

    def set_document_id(self, doc_id: str):
        """
        Set the current document ID for feedback context.

        Should be called when processing starts to associate
        feedback with the documents being processed.

        Args:
            doc_id: Unique identifier for the document(s) being processed
        """
        self._current_doc_id = doc_id

    def generate_document_id(self, text: str) -> str:
        """
        Generate a document ID from text content.

        Uses first 1000 chars to create a hash for consistent ID
        across sessions processing the same document.

        Args:
            text: Document text

        Returns:
            Hash-based document ID
        """
        # Use first 1000 chars for hash (performance + consistency)
        sample = text[:1000] if len(text) > 1000 else text
        hash_obj = hashlib.md5(sample.encode("utf-8"))
        return f"doc_{hash_obj.hexdigest()[:12]}"

    def _delete_feedback_from_csv(self, lower_term: str, term_data: dict[str, Any]) -> bool:
        """
        Delete a feedback entry from CSV when user clears their rating.

        Instead of recording feedback=0, deletes the row entirely.
        Matches by (term, count_bin) key.

        Args:
            lower_term: Lowercase term to delete
            term_data: Term data dict (used to get count for bin calculation)

        Returns:
            True if deletion succeeded (or no matching row existed)
        """
        target_file = self.default_feedback_file if DEBUG_MODE else self.user_feedback_file

        # Use lock to prevent race condition on rapid clicks
        with self._file_lock:
            try:
                # Get count bin for the term being deleted
                try:
                    count = int(term_data.get("Occurrences") or 1)
                except (ValueError, TypeError):
                    count = 1
                delete_key = (lower_term, get_count_bin(count))

                # Read all rows, filter out matching ones
                kept_records: list[dict] = []
                deleted = False

                with open(target_file, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_term = row.get("term", "").lower().strip()
                        try:
                            row_count = int(row.get("occurrences") or 1)
                        except (ValueError, TypeError):
                            row_count = 1
                        row_key = (row_term, get_count_bin(row_count))

                        if row_key == delete_key:
                            deleted = True  # Skip this row (delete it)
                        else:
                            kept_records.append(row)

                # Write remaining records back (atomic via temp file)
                import os
                import tempfile

                fd, temp_path = tempfile.mkstemp(
                    dir=target_file.parent, suffix=".tmp", prefix=".feedback_"
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
                        writer.writeheader()
                        writer.writerows(kept_records)
                    from src.file_utils import safe_replace

                    safe_replace(temp_path, target_file)
                except Exception:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    raise

                target_type = "default" if DEBUG_MODE else "user"
                if deleted:
                    logger.debug("Deleted '%s' from %s feedback", lower_term, target_type)
                else:
                    logger.debug("No matching entry to delete for '%s'", lower_term)

                return True

            except FileNotFoundError:
                return True  # Nothing to delete
            except Exception as e:
                logger.debug("Error deleting feedback: %s", e)
                return False

    def record_feedback(
        self, term_data: dict[str, Any], feedback: int, doc_id: str | None = None
    ) -> bool:
        """
        Record user feedback for a term.

        Args:
            term_data: Dictionary with term info (from vocabulary extractor)
                      Expected keys: Term, Type, Sources, Quality Score, Occurrences, Google Rarity Rank
            feedback: +1 for thumbs up, -1 for thumbs down, 0 to clear
            doc_id: Optional document ID (uses current_doc_id if not provided)

        Returns:
            True if feedback was recorded successfully
        """
        term = term_data.get("Term", "")
        if not term:
            return False

        lower_term = term.lower().strip()
        doc_id = doc_id or self._current_doc_id or "unknown"

        # Update cache
        if feedback == 0:
            self._cache.pop(lower_term, None)
            self._session_rated.discard(lower_term)
            # Delete from CSV instead of recording feedback=0
            return self._delete_feedback_from_csv(lower_term, term_data)
        # Build feedback record
        # Parse algorithms string to create boolean detection columns
        algorithms_str = term_data.get("Sources", "")
        algorithms_upper = [a.strip().upper() for a in algorithms_str.split(",") if a.strip()]

        # Per-algorithm detection flags (all 8 algorithms)
        ner_detected = "NER" in algorithms_upper
        rake_detected = "RAKE" in algorithms_upper
        bm25_detected = "BM25" in algorithms_upper
        topicrank_detected = "TOPICRANK" in algorithms_upper
        medical_ner_detected = "MEDICALNER" in algorithms_upper
        gliner_detected = "GLINER" in algorithms_upper
        yake_detected = "YAKE" in algorithms_upper
        keybert_detected = "KEYBERT" in algorithms_upper
        algo_count = sum(
            [
                ner_detected,
                rake_detected,
                bm25_detected,
                topicrank_detected,
                medical_ner_detected,
                gliner_detected,
                yake_detected,
                keybert_detected,
            ]
        )

        # Use is_person (binary) instead of unreliable type string
        is_person_val = term_data.get("Is Person", "No")
        is_person = 1 if str(is_person_val).lower() in ("yes", "1", "true") else 0

        # Extract TermSources-based features if available
        sources = term_data.get("sources")
        total_docs_in_session = term_data.get("total_docs_in_session", 1)

        if isinstance(sources, TermSources) and sources.num_documents > 0:
            num_source_documents = sources.num_documents
            doc_diversity_ratio = sources.doc_diversity_ratio(int(total_docs_in_session))
            mean_doc_confidence = sources.mean_confidence
            median_doc_confidence = sources.median_confidence
            confidence_std_dev = sources.confidence_std_dev
            high_conf_doc_ratio = sources.high_conf_doc_ratio
            all_low_conf = 1 if sources.all_low_conf else 0
        else:
            # Legacy fallback: no TermSources available
            source_doc_confidence = (term_data.get("source_doc_confidence") or 100) / 100.0
            num_source_documents = 1
            doc_diversity_ratio = 1.0 / max(total_docs_in_session, 1)
            mean_doc_confidence = source_doc_confidence
            median_doc_confidence = source_doc_confidence
            confidence_std_dev = 0.0
            high_conf_doc_ratio = 1.0 if source_doc_confidence > 0.80 else 0.0
            all_low_conf = 1 if source_doc_confidence < 0.60 else 0

        record = {
            "timestamp": datetime.now().isoformat(),
            "document_id": doc_id,
            "term": term,
            "feedback": feedback,
            "is_person": is_person,
            "algorithms": algorithms_str,
            # Per-algorithm detection flags (all 8 algorithms)
            "NER_detection": ner_detected,
            "RAKE_detection": rake_detected,
            "BM25_detection": bm25_detected,
            "TopicRank_detection": topicrank_detected,
            "MedicalNER_detection": medical_ner_detected,
            "GLiNER_detection": gliner_detected,
            "YAKE_detection": yake_detected,
            "KeyBERT_detection": keybert_detected,
            "algo_count": algo_count,
            "quality_score": term_data.get("Quality Score", 0),
            "occurrences": term_data.get("Occurrences", 1),
            "rarity_rank": term_data.get("Google Rarity Rank", 0),
            # Per-algorithm numeric scores
            "topicrank_score": term_data.get("topicrank_score", 0.0),
            "yake_score": term_data.get("yake_score", 0.0),
            "keybert_score": term_data.get("keybert_score", 0.0),
            "rake_score": term_data.get("rake_score", 0.0),
            "bm25_score": term_data.get("bm25_score", 0.0),
            # TermSources-based per-document features
            "num_source_documents": num_source_documents,
            "doc_diversity_ratio": doc_diversity_ratio,
            "mean_doc_confidence": mean_doc_confidence,
            "median_doc_confidence": median_doc_confidence,
            "confidence_std_dev": confidence_std_dev,
            "high_conf_doc_ratio": high_conf_doc_ratio,
            "all_low_conf": all_low_conf,
            "total_docs_in_session": total_docs_in_session,
            "total_word_count": term_data.get("total_word_count", 0),
        }

        # Route feedback based on DEBUG_MODE
        # - DEBUG_MODE=True: Write to default_feedback.csv (development data)
        # - DEBUG_MODE=False: Write to user_feedback.csv (user data)
        target_file = self.default_feedback_file if DEBUG_MODE else self.user_feedback_file

        # Use lock to prevent race condition on rapid clicks
        with self._file_lock:
            try:
                # Ensure parent directory exists (needed for default_feedback.csv in config/)
                target_file.parent.mkdir(parents=True, exist_ok=True)

                # Deduplicate by (term, count_bin) at write time
                # If same (term, count_bin) exists, replace it; otherwise append
                new_count_bin = get_count_bin(int(record.get("occurrences") or 1))
                new_key = (lower_term, new_count_bin)

                existing_records: list[dict] = []
                replaced = False

                try:
                    with open(target_file, encoding="utf-8", newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            row_term = row.get("term", "").lower().strip()
                            try:
                                row_count = int(row.get("occurrences") or 1)
                            except (ValueError, TypeError):
                                row_count = 1
                            row_key = (row_term, get_count_bin(row_count))

                            if row_key == new_key:
                                # Replace this row with new record
                                existing_records.append(record)
                                replaced = True
                            else:
                                existing_records.append(row)
                except FileNotFoundError:
                    pass  # New file, no existing records to load

                if not replaced:
                    existing_records.append(record)

                # Write all records back to file (atomic via temp file)
                import os
                import tempfile

                fd, temp_path = tempfile.mkstemp(
                    dir=target_file.parent, suffix=".tmp", prefix=".feedback_"
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
                        writer.writeheader()
                        writer.writerows(existing_records)
                    from src.file_utils import safe_replace

                    safe_replace(temp_path, target_file)
                except Exception:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    raise

                # Update cache AFTER successful write so cache stays consistent
                # if the write fails
                self._cache[lower_term] = feedback
                self._session_rated.add(lower_term)

                self._pending_count += 1
                target_type = "default" if DEBUG_MODE else "user"
                action = "replaced" if replaced else "added"
                logger.debug(
                    "%s %s for '%s' (%s)",
                    action.capitalize(),
                    "+" if feedback > 0 else "-",
                    term,
                    target_type,
                )
                return True

            except Exception as e:
                logger.debug("Error recording feedback: %s", e)
                return False

    def get_rating(self, term: str) -> int:
        """
        Get the current rating for a term.

        Args:
            term: The vocabulary term (case-insensitive)

        Returns:
            +1 (thumbs up), -1 (thumbs down), or 0 (unrated)
        """
        return self._cache.get(term.lower().strip(), 0)

    def has_rating(self, term: str) -> bool:
        """Check if a term has been rated."""
        return term.lower().strip() in self._cache

    def get_rating_source(self, term: str) -> str | None:
        """
        Get the source of a term's rating.

        Distinguishes colors in GUI:
        - "session": User clicked Keep/Skip in current session (darker green/red)
        - "loaded": Rating from loaded dataset file (lighter green/red)
        - None: No rating exists

        Args:
            term: The vocabulary term (case-insensitive)

        Returns:
            "session", "loaded", or None
        """
        lower_term = term.lower().strip()
        if lower_term not in self._cache:
            return None
        if lower_term in self._session_rated:
            return "session"
        return "loaded"

    def clear_rating(self, term: str) -> bool:
        """
        Clear ALL ratings for a term (all count bins).

        Deletes all entries for this term from CSV, regardless of count bin.
        Use record_feedback(term_data, 0) to clear a specific (term, count_bin) entry.

        Args:
            term: The vocabulary term

        Returns:
            True if a rating was cleared, False if term was unrated
        """
        lower_term = term.lower().strip()
        if lower_term not in self._cache:
            return False

        # Delete ALL entries for this term from CSV (any count bin)
        target_file = self.default_feedback_file if DEBUG_MODE else self.user_feedback_file

        with self._file_lock:
            try:
                kept_records: list[dict] = []
                with open(target_file, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_term = row.get("term", "").lower().strip()
                        if row_term != lower_term:
                            kept_records.append(row)

                # Write remaining records back (atomic via temp file)
                import os
                import tempfile

                fd, temp_path = tempfile.mkstemp(
                    dir=target_file.parent, suffix=".tmp", prefix=".feedback_"
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
                        writer.writeheader()
                        writer.writerows(kept_records)
                    from src.file_utils import safe_replace

                    safe_replace(temp_path, target_file)
                except Exception:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    raise

                # Update cache AFTER successful file write (matches record_feedback pattern)
                del self._cache[lower_term]
                self._session_rated.discard(lower_term)
                logger.debug("Cleared all ratings for '%s'", term)
                return True

            except FileNotFoundError:
                # File doesn't exist — clear cache since there's nothing on disk
                del self._cache[lower_term]
                self._session_rated.discard(lower_term)
                return True
            except Exception as e:
                logger.debug("Error clearing rating for '%s': %s", term, e)
                return False  # File write failed, keep cache consistent with disk

    def _load_feedback_from_file(self, filepath: Path) -> list[dict]:
        """
        Load feedback records from a CSV file.

        Args:
            filepath: Path to feedback CSV file

        Returns:
            List of feedback records as dictionaries
        """
        with self._file_lock:
            try:
                with open(filepath, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            except FileNotFoundError:
                return []
            except Exception as e:
                logger.debug("Error loading feedback from %s: %s", filepath, e)
                return []

    def get_all_user_feedback(self) -> list[dict]:
        """
        Load all user feedback from CSV.

        Returns:
            List of user feedback records as dictionaries
        """
        return self._load_feedback_from_file(self.user_feedback_file)

    def get_all_default_feedback(self) -> list[dict]:
        """
        Load all default (shipped) feedback from CSV.

        Returns:
            List of default feedback records as dictionaries
        """
        return self._load_feedback_from_file(self.default_feedback_file)

    def get_feedback_count(self) -> int:
        """Get total number of user feedback entries in cache."""
        return len(self._cache)

    def get_user_feedback_count(self) -> int:
        """
        Get count of user feedback entries.

        This is used for ML weight calculations - determines how much
        to weight user feedback vs default feedback during training.

        Returns:
            Number of unique user feedback entries
        """
        return len(self._cache)

    def get_pending_count(self) -> int:
        """Get number of feedback entries since last training."""
        return self._pending_count

    def reset_pending_count(self):
        """Reset pending count after training."""
        self._pending_count = 0

    def should_retrain(
        self, min_samples: int = ML_MIN_SAMPLES, retrain_threshold: int = ML_RETRAIN_THRESHOLD
    ) -> bool:
        """
        Check if model should be retrained based on feedback count.

        Args:
            min_samples: Minimum total samples needed before training
            retrain_threshold: New feedback needed to trigger retraining

        Returns:
            True if retraining is recommended
        """
        total = self.get_feedback_count()
        pending = self.get_pending_count()

        if total < min_samples:
            return False

        return pending >= retrain_threshold

    def get_rated_terms(self, rating_filter: int | None = None) -> list[str]:
        """
        Get list of terms with feedback.

        Args:
            rating_filter: If specified, only return terms with this rating (+1 or -1)

        Returns:
            List of term strings
        """
        if rating_filter is None:
            return list(self._cache.keys())
        return [term for term, rating in self._cache.items() if rating == rating_filter]

    def clear_all_feedback(self) -> bool:
        """
        Clear all user feedback data (cache and CSV file).

        Used when user wants to start fresh with vocabulary preferences.
        This only clears USER feedback - default (shipped) feedback is preserved.

        Returns:
            True if clear succeeded
        """
        with self._file_lock:
            try:
                # Clear in-memory cache
                self._cache.clear()
                self._pending_count = 0

                # Delete the user feedback CSV file (not the default file)
                if self.user_feedback_file.exists():
                    self.user_feedback_file.unlink()
                    logger.debug("Deleted user feedback file: %s", self.user_feedback_file)
                else:
                    logger.debug("No user feedback file to delete")

                logger.debug("User feedback cleared successfully")
                return True

            except Exception as e:
                logger.debug("Error clearing feedback: %s", e)
                return False

    def export_training_data(self) -> list[dict]:
        """
        Export combined feedback data formatted for ML training.

        Combines default (shipped) and user feedback with source tags.
        Deduplicates by (term, count_bin) - keeps most recent feedback.

        Deduplicates by (term, count_bin) rather than term alone.
        Rationale: Same term at count=1 (possible OCR error) is semantically
        different from count=50 (definitely real). But same term+bin with
        multiple entries is a duplicate that would over-weight that observation.

        Returns:
            List of training records with features, labels, and source tags
        """
        # Load default feedback (shipped with app)
        default_feedback = self.get_all_default_feedback()
        for record in default_feedback:
            record["source"] = "default"

        # Load user feedback
        user_feedback = self.get_all_user_feedback()
        for record in user_feedback:
            record["source"] = "user"

        # Combine all feedback, then deduplicate by (term, count_bin)
        # Later entries (by timestamp) win over earlier ones
        # User feedback files are loaded after default, so user wins over default
        all_feedback = default_feedback + user_feedback

        # Sort by timestamp so most recent entry wins when we deduplicate
        def get_timestamp(record: dict) -> str:
            return record.get("timestamp", "")

        all_feedback.sort(key=get_timestamp)

        # Deduplicate by (term, count_bin) - last entry wins
        # Also validate each record has required fields
        term_bin_feedback: dict[tuple[str, str], dict] = {}
        skipped_invalid = 0
        for record in all_feedback:
            # Validate required fields
            if not _validate_record(record):
                skipped_invalid += 1
                continue

            term = record.get("term", "").lower().strip()
            if not term:
                continue

            # Get count bin for deduplication key
            try:
                count = int(record.get("occurrences") or 1)
            except (ValueError, TypeError):
                count = 1
            count_bin = get_count_bin(count)

            key = (term, count_bin)
            term_bin_feedback[key] = record

        result = list(term_bin_feedback.values())
        default_count = sum(1 for r in result if r.get("source") == "default")
        user_count = sum(1 for r in result if r.get("source") == "user")
        total_raw = len(default_feedback) + len(user_feedback)
        deduped = total_raw - len(result) - skipped_invalid
        if skipped_invalid:
            logger.debug(
                "Exported training data: %d records (%d default, %d user, %d duplicates removed, %d invalid)",
                len(result),
                default_count,
                user_count,
                deduped,
                skipped_invalid,
            )
        else:
            logger.debug(
                "Exported training data: %d records (%d default, %d user, %d duplicates removed)",
                len(result),
                default_count,
                user_count,
                deduped,
            )

        return result


# Global singleton instance with thread-safe initialization
_feedback_manager: FeedbackManager | None = None
_feedback_lock = threading.Lock()


def get_feedback_manager() -> FeedbackManager:
    """
    Get the global FeedbackManager singleton.

    Thread-safe with double-check locking pattern.

    Returns:
        FeedbackManager instance
    """
    global _feedback_manager

    # Fast path: already initialized
    if _feedback_manager is not None:
        return _feedback_manager

    # Slow path: need to initialize (with lock)
    with _feedback_lock:
        # Double-check after acquiring lock
        if _feedback_manager is None:
            _feedback_manager = FeedbackManager()
    return _feedback_manager
