"""
Vocabulary Service for CasePrepd.

Provides a clean interface for vocabulary extraction operations.
Wraps the VocabularyExtractor and related components.

Usage:
    from src.services import VocabularyService

    service = VocabularyService()
    vocab_data = service.extract_vocabulary(text)
"""

import logging
from collections.abc import Callable

from src.core.vocabulary import VocabularyExtractor

logger = logging.getLogger(__name__)


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
        logger.info("Starting vocabulary extraction (%d chars)", len(text))
        if not text.strip():
            logger.debug("Empty text provided, returning empty list")
            return []

        result, _filtered = self.extractor.extract(text)

        logger.debug("Extracted %s terms", len(result))

        return result

    def extract_vocabulary_per_document(
        self, documents: list[dict], progress_callback: Callable[[int, int], None] | None = None
    ) -> list[dict]:
        """
        Extract vocabulary with per-document tracking.

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
        logger.info("Starting per-document vocabulary extraction (%d documents)", len(documents))
        if not documents:
            logger.debug("No documents provided, returning empty list")
            return []

        result = self.extractor.extract_per_document(documents, progress_callback=progress_callback)

        logger.debug("Extracted %s terms from %s documents", len(result), len(documents))

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

        feedback_type = "positive" if is_positive else "negative"
        logger.debug("Recorded %s feedback for '%s'", feedback_type, term)

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
                status = "enabled" if enabled else "disabled"
                logger.debug("Algorithm %s %s", name, status)
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

            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(vocab_data)

            logger.debug("Exported %s terms to %s", len(vocab_data), file_path)

            return True

        except Exception as e:
            logger.error("Export failed: %s", e, exc_info=True)
            return False

    def get_corpus_manager(self):
        """
        Get the corpus manager instance.

        Returns:
            CorpusManager singleton for corpus operations.
        """
        from src.core.vocabulary.corpus_manager import get_corpus_manager

        return get_corpus_manager()

    def get_corpus_registry(self):
        """
        Get the corpus registry for managing multiple corpora.

        Returns:
            CorpusRegistry singleton.
        """
        from src.core.vocabulary import get_corpus_registry

        return get_corpus_registry()

    def get_corpus_files_with_status(self, corpus_path):
        """
        Get files in a corpus with their preprocessing status.

        Args:
            corpus_path: Path to the corpus directory.

        Returns:
            List of CorpusFileInfo objects with name, is_preprocessed, modified_at.
        """
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=corpus_path)
        return manager.get_corpus_files_with_status()

    def preprocess_corpus_file(self, corpus_path, file_path):
        """
        Preprocess a single file in a corpus.

        Args:
            corpus_path: Path to the corpus directory.
            file_path: Path to the file to preprocess.
        """
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=corpus_path)
        manager.preprocess_file(file_path)

    def preprocess_corpus_pending(self, corpus_path):
        """
        Preprocess all pending files in a corpus.

        Args:
            corpus_path: Path to the corpus directory.

        Returns:
            Number of files preprocessed.
        """
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=corpus_path)
        return manager.preprocess_pending()

    def get_feedback_manager(self):
        """
        Get the feedback manager instance.

        Returns:
            FeedbackManager singleton for ML feedback operations.
        """
        from src.core.vocabulary.feedback_manager import get_feedback_manager

        return get_feedback_manager()

    def get_meta_learner(self):
        """
        Get the meta learner instance.

        Returns:
            MetaLearner singleton for ML model operations.
        """
        from src.core.vocabulary.preference_learner import get_meta_learner

        return get_meta_learner()

    def get_max_corpus_docs(self) -> int:
        """
        Get the maximum number of documents allowed in a corpus.

        Returns:
            Maximum document count (currently 25).
        """
        from src.core.vocabulary.corpus_manager import MAX_CORPUS_DOCS

        return MAX_CORPUS_DOCS

    def get_corpus_doc_count(self, corpus_path) -> int:
        """
        Count documents in a corpus folder.

        Args:
            corpus_path: Path to the corpus directory.

        Returns:
            Number of supported documents (PDF, TXT, RTF) in the folder.
        """
        from pathlib import Path

        from src.core.vocabulary.corpus_manager import SUPPORTED_EXTENSIONS

        path = Path(corpus_path)
        if not path.exists():
            return 0

        # Count supported files (excluding preprocessed files)
        count = 0
        for ext in SUPPORTED_EXTENSIONS:
            for f in path.glob(f"*{ext}"):
                if "_preprocessed" not in f.stem:
                    count += 1
            for f in path.glob(f"*{ext.upper()}"):
                if "_preprocessed" not in f.stem:
                    count += 1

        return count

    def is_corpus_disabled(self) -> bool:
        """
        Check if the active corpus is disabled due to exceeding document limit.

        Returns:
            True if corpus has >25 documents and is disabled.
        """
        manager = self.get_corpus_manager()
        return manager.is_corpus_disabled()

    def explain_term_score(self, term_data: dict, max_reasons: int = 6) -> dict | None:
        """
        Explain why the scoring system rated a vocabulary term the way it did.

        Combines top factors from Rules, LR, and RF (when active),
        deduplicated by feature. Returns up to 6 unique reasons.

        Args:
            term_data: Term data dictionary (same format as vocab table rows)
            max_reasons: Maximum number of reasons to return

        Returns:
            Dict with score, direction, reasons, model_status.
            None if the model is not trained.
        """
        from src.core.vocabulary.score_explainer import explain_score

        return explain_score(term_data, max_reasons=max_reasons)

    def get_corpus_disabled_reason(self) -> str | None:
        """
        Get the reason why the corpus is disabled.

        Returns:
            Error message if disabled, None if not disabled.
        """
        manager = self.get_corpus_manager()
        return manager.get_disabled_reason()
