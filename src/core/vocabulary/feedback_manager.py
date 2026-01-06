"""
Vocabulary Feedback Manager

Manages user feedback (thumbs up/down) on vocabulary terms.
Stores feedback in CSV format for ML training.

Two-File System (Session 55):
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
- is_person: 1 if NER detected as person name, 0 otherwise (Session 52)
- algorithms: Comma-separated list of algorithms that detected the term
- NER_detection: Boolean - whether NER algorithm detected this term
- RAKE_detection: Boolean - whether RAKE algorithm detected this term
- BM25_detection: Boolean - whether BM25 algorithm detected this term
- algo_count: Number of algorithms that detected this term (sum of detection booleans)
- quality_score: Quality score at time of feedback
- in_case_freq: Term occurrence count
- freq_rank: Google frequency rank

Session 78: Added TermSources-based per-document features:
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
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import (
    DEBUG_MODE,
    DEFAULT_FEEDBACK_CSV,
    FEEDBACK_DIR,
    ML_MIN_SAMPLES,
    ML_RETRAIN_THRESHOLD,
)
from src.core.vocabulary.term_sources import TermSources
from src.logging_config import debug_log

# CSV columns
# Session 52: Replaced "type" with "is_person" (binary flag, more reliable)
# Session 78: Added TermSources-based per-document features (8 columns)
FEEDBACK_COLUMNS = [
    "timestamp",
    "document_id",
    "term",
    "feedback",
    "is_person",  # 1 if NER detected as person, 0 otherwise
    "algorithms",
    "NER_detection",
    "RAKE_detection",
    "BM25_detection",
    "algo_count",
    "quality_score",
    "in_case_freq",
    "freq_rank",
    # Session 78: TermSources-based features
    "num_source_documents",  # How many docs contain this term
    "doc_diversity_ratio",  # num_docs / total_docs (0-1)
    "mean_doc_confidence",  # Count-weighted mean OCR confidence (0-1)
    "median_doc_confidence",  # Median confidence - robust to outliers (0-1)
    "confidence_std_dev",  # Standard deviation of confidences (0-0.5)
    "high_conf_doc_ratio",  # % of source docs with confidence > 0.80 (0-1)
    "all_low_conf",  # 1 if ALL source docs have conf < 0.60, else 0
    "total_docs_in_session",  # Total docs in extraction session
]


class FeedbackManager:
    """
    Manages user feedback on vocabulary terms.

    Two-File System (Session 55, updated Session 76):
    - Default feedback: Ships with app (developer's training data)
      Location: config/default_feedback.csv
    - User feedback: Collected during normal use
      Location: %APPDATA%/LocalScribe/feedback/user_feedback.csv

    Routing (Session 76):
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
                         Defaults to %APPDATA%/LocalScribe/feedback/
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

        # Track pending feedback count (for retraining threshold)
        self._pending_count = 0

        # Current document ID for feedback context
        self._current_doc_id: str = ""

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

        Routing (Session 78):
        - DEBUG_MODE=True: Load from default_feedback.csv (developer data)
        - DEBUG_MODE=False: Load from user_feedback.csv (user data)

        This ensures the GUI reflects the dataset that will be modified.
        """
        # Choose which file to load based on DEBUG_MODE
        target_file = self.default_feedback_file if DEBUG_MODE else self.user_feedback_file
        target_type = "default" if DEBUG_MODE else "user"

        if not target_file.exists():
            debug_log(f"[FEEDBACK] No existing {target_type} feedback file, starting fresh")
            return

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

            debug_log(
                f"[FEEDBACK] Loaded {len(self._cache)} {target_type} feedback entries (DEBUG_MODE={DEBUG_MODE})"
            )
        except Exception as e:
            debug_log(f"[FEEDBACK] Error loading {target_type} feedback: {e}")

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

    def record_feedback(
        self, term_data: dict[str, Any], feedback: int, doc_id: str | None = None
    ) -> bool:
        """
        Record user feedback for a term.

        Args:
            term_data: Dictionary with term info (from vocabulary extractor)
                      Expected keys: Term, Type, Sources, Quality Score, In-Case Freq, Freq Rank
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
        else:
            self._cache[lower_term] = feedback

        # Build feedback record
        # Parse algorithms string to create boolean detection columns
        algorithms_str = term_data.get("Sources", "")
        algorithms_upper = [a.strip().upper() for a in algorithms_str.split(",") if a.strip()]

        # Calculate algo_count (sum of detection booleans)
        ner_detected = "NER" in algorithms_upper
        rake_detected = "RAKE" in algorithms_upper
        bm25_detected = "BM25" in algorithms_upper
        algo_count = sum([ner_detected, rake_detected, bm25_detected])

        # Session 52: Use is_person (binary) instead of unreliable type
        is_person_val = term_data.get("Is Person", "No")
        is_person = 1 if str(is_person_val).lower() in ("yes", "1", "true") else 0

        # Session 78: Extract TermSources-based features if available
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
            source_doc_confidence = term_data.get("source_doc_confidence", 100) / 100.0
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
            "NER_detection": ner_detected,
            "RAKE_detection": rake_detected,
            "BM25_detection": bm25_detected,
            "algo_count": algo_count,
            "quality_score": term_data.get("Quality Score", 0),
            "in_case_freq": term_data.get("In-Case Freq", 1),
            "freq_rank": term_data.get("Freq Rank", 0),
            # Session 78: TermSources-based features
            "num_source_documents": num_source_documents,
            "doc_diversity_ratio": doc_diversity_ratio,
            "mean_doc_confidence": mean_doc_confidence,
            "median_doc_confidence": median_doc_confidence,
            "confidence_std_dev": confidence_std_dev,
            "high_conf_doc_ratio": high_conf_doc_ratio,
            "all_low_conf": all_low_conf,
            "total_docs_in_session": total_docs_in_session,
        }

        # Session 76: Route feedback based on DEBUG_MODE
        # - DEBUG_MODE=True: Write to default_feedback.csv (development data)
        # - DEBUG_MODE=False: Write to user_feedback.csv (user data)
        target_file = self.default_feedback_file if DEBUG_MODE else self.user_feedback_file

        try:
            file_exists = target_file.exists()

            # Ensure parent directory exists (needed for default_feedback.csv in config/)
            target_file.parent.mkdir(parents=True, exist_ok=True)

            with open(target_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)

            self._pending_count += 1
            target_type = "default" if DEBUG_MODE else "user"
            debug_log(
                f"[FEEDBACK] Recorded {'+' if feedback > 0 else '-'} for '{term}' ({target_type})"
            )
            return True

        except Exception as e:
            debug_log(f"[FEEDBACK] Error recording feedback: {e}")
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

    def clear_rating(self, term: str) -> bool:
        """
        Clear the rating for a term.

        Args:
            term: The vocabulary term

        Returns:
            True if a rating was cleared, False if term was unrated
        """
        lower_term = term.lower().strip()
        if lower_term in self._cache:
            del self._cache[lower_term]
            # Record the clear as feedback=0
            self.record_feedback({"Term": term}, 0)
            return True
        return False

    def _load_feedback_from_file(self, filepath: Path) -> list[dict]:
        """
        Load feedback records from a CSV file.

        Args:
            filepath: Path to feedback CSV file

        Returns:
            List of feedback records as dictionaries
        """
        if not filepath.exists():
            return []

        try:
            with open(filepath, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            debug_log(f"[FEEDBACK] Error loading feedback from {filepath}: {e}")
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
        try:
            # Clear in-memory cache
            self._cache.clear()
            self._pending_count = 0

            # Delete the user feedback CSV file (not the default file)
            if self.user_feedback_file.exists():
                self.user_feedback_file.unlink()
                debug_log(f"[FEEDBACK] Deleted user feedback file: {self.user_feedback_file}")
            else:
                debug_log("[FEEDBACK] No user feedback file to delete")

            debug_log("[FEEDBACK] User feedback cleared successfully")
            return True

        except Exception as e:
            debug_log(f"[FEEDBACK] Error clearing feedback: {e}")
            return False

    def export_training_data(self) -> list[dict]:
        """
        Export combined feedback data formatted for ML training.

        Combines default (shipped) and user feedback with source tags.
        Aggregates feedback by term (most recent user feedback wins over default).

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

        # Combine with user feedback taking precedence
        # (user feedback for same term overwrites default)
        term_feedback: dict[str, dict] = {}

        # First, add all default feedback
        for record in default_feedback:
            term = record.get("term", "").lower().strip()
            if term:
                term_feedback[term] = record

        # Then, add/overwrite with user feedback
        for record in user_feedback:
            term = record.get("term", "").lower().strip()
            if term:
                term_feedback[term] = record

        result = list(term_feedback.values())
        default_count = sum(1 for r in result if r.get("source") == "default")
        user_count = sum(1 for r in result if r.get("source") == "user")
        debug_log(f"[FEEDBACK] Exported training data: {default_count} default, {user_count} user")

        return result


# Global singleton instance
_feedback_manager: FeedbackManager | None = None


def get_feedback_manager() -> FeedbackManager:
    """
    Get the global FeedbackManager singleton.

    Returns:
        FeedbackManager instance
    """
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = FeedbackManager()
    return _feedback_manager
