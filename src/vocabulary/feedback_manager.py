"""
Vocabulary Feedback Manager

Manages user feedback (thumbs up/down) on vocabulary terms.
Stores feedback in CSV format for ML training.

The feedback data is used to train a meta-learner that adapts
to user preferences over time.

CSV Schema:
- timestamp: ISO8601 datetime when feedback was recorded
- document_id: Hash/ID of the document being processed
- term: The vocabulary term
- feedback: +1 (thumbs up) or -1 (thumbs down)
- type: Category (Person/Place/Medical/Technical/Unknown)
- algorithms: Comma-separated list of algorithms that detected the term
- NER_detection: Boolean - whether NER algorithm detected this term
- RAKE_detection: Boolean - whether RAKE algorithm detected this term
- BM25_detection: Boolean - whether BM25 algorithm detected this term
- algo_count: Number of algorithms that detected this term (sum of detection booleans)
- quality_score: Quality score at time of feedback
- in_case_freq: Term occurrence count
- freq_rank: Google frequency rank
"""

import csv
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import FEEDBACK_DIR, ML_MIN_SAMPLES, ML_RETRAIN_THRESHOLD, VOCAB_FEEDBACK_CSV
from src.logging_config import debug_log

# CSV columns
FEEDBACK_COLUMNS = [
    "timestamp",
    "document_id",
    "term",
    "feedback",
    "type",
    "algorithms",
    "NER_detection",
    "RAKE_detection",
    "BM25_detection",
    "algo_count",
    "quality_score",
    "in_case_freq",
    "freq_rank",
]


class FeedbackManager:
    """
    Manages user feedback on vocabulary terms.

    Provides:
    - Recording feedback (thumbs up/down) for terms
    - Persisting feedback to CSV file
    - Loading feedback history for ML training
    - Caching feedback state for UI display

    The feedback is keyed by normalized term (lowercase) for
    consistent lookups across sessions.

    Example:
        manager = FeedbackManager()
        manager.record_feedback(term_data, +1, "doc123")
        rating = manager.get_rating("spondylosis")  # Returns +1, -1, or 0
    """

    def __init__(self, feedback_dir: Path | None = None):
        """
        Initialize feedback manager.

        Args:
            feedback_dir: Directory to store feedback files.
                         Defaults to %APPDATA%/LocalScribe/feedback/
        """
        self.feedback_dir = Path(feedback_dir) if feedback_dir else FEEDBACK_DIR
        self.feedback_file = self.feedback_dir / "vocab_feedback.csv"

        # In-memory cache: normalized_term -> rating (+1, -1, or 0)
        self._cache: dict[str, int] = {}

        # Track pending feedback count (for retraining threshold)
        self._pending_count = 0

        # Current document ID for feedback context
        self._current_doc_id: str = ""

        # Ensure directory exists
        self._ensure_directory()

        # Load existing feedback into cache
        self._load_cache()

    def _ensure_directory(self):
        """Create feedback directory if it doesn't exist."""
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

    def _load_cache(self):
        """Load existing feedback from CSV into cache."""
        if not self.feedback_file.exists():
            debug_log("[FEEDBACK] No existing feedback file, starting fresh")
            return

        try:
            with open(self.feedback_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    term = row.get("term", "").lower().strip()
                    feedback_str = row.get("feedback", "0")
                    try:
                        feedback = int(feedback_str)
                        self._cache[term] = feedback
                    except ValueError:
                        continue

            debug_log(f"[FEEDBACK] Loaded {len(self._cache)} feedback entries from {self.feedback_file}")
        except Exception as e:
            debug_log(f"[FEEDBACK] Error loading feedback: {e}")

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
        hash_obj = hashlib.md5(sample.encode('utf-8'))
        return f"doc_{hash_obj.hexdigest()[:12]}"

    def record_feedback(
        self,
        term_data: dict[str, Any],
        feedback: int,
        doc_id: str | None = None
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

        record = {
            "timestamp": datetime.now().isoformat(),
            "document_id": doc_id,
            "term": term,
            "feedback": feedback,
            "type": term_data.get("Type", "Unknown"),
            "algorithms": algorithms_str,
            "NER_detection": ner_detected,
            "RAKE_detection": rake_detected,
            "BM25_detection": bm25_detected,
            "algo_count": algo_count,
            "quality_score": term_data.get("Quality Score", 0),
            "in_case_freq": term_data.get("In-Case Freq", 1),
            "freq_rank": term_data.get("Freq Rank", 0),
        }

        # Append to CSV
        try:
            file_exists = self.feedback_file.exists()

            with open(self.feedback_file, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)

            self._pending_count += 1
            debug_log(f"[FEEDBACK] Recorded {'+' if feedback > 0 else '-'} for '{term}'")
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

    def get_all_feedback(self) -> list[dict]:
        """
        Load all feedback from CSV for ML training.

        Returns:
            List of feedback records as dictionaries
        """
        if not self.feedback_file.exists():
            return []

        try:
            with open(self.feedback_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            debug_log(f"[FEEDBACK] Error loading all feedback: {e}")
            return []

    def get_feedback_count(self) -> int:
        """Get total number of feedback entries in cache."""
        return len(self._cache)

    def get_pending_count(self) -> int:
        """Get number of feedback entries since last training."""
        return self._pending_count

    def reset_pending_count(self):
        """Reset pending count after training."""
        self._pending_count = 0

    def should_retrain(self, min_samples: int = ML_MIN_SAMPLES, retrain_threshold: int = ML_RETRAIN_THRESHOLD) -> bool:
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
        Clear all feedback data (cache and CSV file).

        Used when user wants to start fresh with vocabulary preferences.
        This is a destructive operation - all feedback history is lost.

        Returns:
            True if clear succeeded
        """
        try:
            # Clear in-memory cache
            self._cache.clear()
            self._pending_count = 0

            # Delete the feedback CSV file
            if self.feedback_file.exists():
                self.feedback_file.unlink()
                debug_log(f"[FEEDBACK] Deleted feedback file: {self.feedback_file}")
            else:
                debug_log("[FEEDBACK] No feedback file to delete")

            debug_log("[FEEDBACK] All feedback cleared successfully")
            return True

        except Exception as e:
            debug_log(f"[FEEDBACK] Error clearing feedback: {e}")
            return False

    def export_training_data(self) -> list[dict]:
        """
        Export feedback data formatted for ML training.

        Aggregates feedback by term (uses most recent feedback for duplicates).

        Returns:
            List of training records with features and labels
        """
        # Load all feedback records
        all_feedback = self.get_all_feedback()

        # Aggregate by term (most recent wins)
        term_feedback: dict[str, dict] = {}
        for record in all_feedback:
            term = record.get("term", "").lower().strip()
            if term:
                term_feedback[term] = record

        return list(term_feedback.values())


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
