"""
Vocabulary Service for LocalScribe.

Provides a clean interface for vocabulary extraction operations.
Wraps the VocabularyExtractor and related components.

Usage:
    from src.services import VocabularyService

    service = VocabularyService()
    vocab_data = service.extract_vocabulary(text)
"""

from collections.abc import Callable

from src.config import DEBUG_MODE
from src.core.vocabulary import VocabularyExtractor
from src.logging_config import debug_log


class VocabularyService:
    """
    Service layer for vocabulary extraction.

    Coordinates vocabulary extraction from document text.
    Provides a simplified interface for the UI layer.
    """

    def __init__(self):
        """Initialize the vocabulary service."""
        self._extractor: VocabularyExtractor | None = None

    @property
    def extractor(self) -> VocabularyExtractor:
        """Lazy-load the vocabulary extractor."""
        if self._extractor is None:
            self._extractor = VocabularyExtractor()
        return self._extractor

    def extract_vocabulary(
        self,
        text: str,
        document_id: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """
        Extract vocabulary from document text.

        Args:
            text: Combined document text
            document_id: Optional ID for feedback tracking
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of vocabulary dicts with Term, Is Person, Quality Score, etc.
        """
        if not text.strip():
            if DEBUG_MODE:
                debug_log("[VocabularyService] Empty text provided, returning empty list")
            return []

        result = self.extractor.extract(text)

        if DEBUG_MODE:
            debug_log(f"[VocabularyService] Extracted {len(result)} terms")

        return result

    def extract_vocabulary_per_document(
        self, documents: list[dict], progress_callback: Callable[[int, int], None] | None = None
    ) -> list[dict]:
        """
        Extract vocabulary with per-document tracking (Session 78).

        This method processes each document individually and tracks which
        documents contributed each term occurrence. This enables confidence-
        weighted canonical selection for better handling of OCR variants.

        Args:
            documents: List of dicts with keys:
                      - 'text': Document text
                      - 'doc_id': Unique identifier (e.g., file hash)
                      - 'confidence': OCR confidence (0-100)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of vocabulary dicts with TermSources attached via 'sources' key.
            The TermSources enables confidence-weighted canonical selection.
        """
        if not documents:
            if DEBUG_MODE:
                debug_log("[VocabularyService] No documents provided, returning empty list")
            return []

        result = self.extractor.extract_per_document(documents, progress_callback=progress_callback)

        if DEBUG_MODE:
            debug_log(
                f"[VocabularyService] Extracted {len(result)} terms from {len(documents)} documents"
            )

        return result

    def record_feedback(self, term: str, is_positive: bool, document_id: str | None = None) -> None:
        """
        Record user feedback on a vocabulary term.

        Args:
            term: The vocabulary term
            is_positive: True for positive feedback, False for negative
            document_id: Optional document identifier
        """
        self.extractor.record_feedback(term, is_positive, document_id)

        if DEBUG_MODE:
            feedback_type = "positive" if is_positive else "negative"
            debug_log(f"[VocabularyService] Recorded {feedback_type} feedback for '{term}'")

    def get_algorithm_info(self) -> list[dict]:
        """
        Get information about enabled extraction algorithms.

        Returns:
            List of dicts with algorithm name, enabled status, and description
        """
        return [
            {
                "name": algo.name,
                "enabled": algo.enabled,
                "description": algo.description,
            }
            for algo in self.extractor.algorithms
        ]

    def set_algorithm_enabled(self, name: str, enabled: bool) -> bool:
        """
        Enable or disable an extraction algorithm.

        Args:
            name: Algorithm name (e.g., "NER", "RAKE", "BM25")
            enabled: Whether to enable the algorithm

        Returns:
            True if algorithm was found and updated, False otherwise
        """
        for algo in self.extractor.algorithms:
            if algo.name == name:
                algo.enabled = enabled
                if DEBUG_MODE:
                    status = "enabled" if enabled else "disabled"
                    debug_log(f"[VocabularyService] Algorithm {name} {status}")
                return True
        return False

    def export_to_csv(self, vocab_data: list[dict], file_path: str) -> bool:
        """
        Export vocabulary data to CSV file.

        Args:
            vocab_data: List of vocabulary dicts
            file_path: Path to save CSV

        Returns:
            True if successful, False otherwise
        """
        import csv

        try:
            if not vocab_data:
                return False

            # Get headers from first item
            headers = list(vocab_data[0].keys())

            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(vocab_data)

            if DEBUG_MODE:
                debug_log(f"[VocabularyService] Exported {len(vocab_data)} terms to {file_path}")

            return True

        except Exception as e:
            debug_log(f"[VocabularyService] Export failed: {e}")
            return False
