"""
Q&A Service for CasePrepd.

Provides a clean interface for question-answering operations.
Wraps the QAOrchestrator, vector store, and retrieval components.

Usage:
    from src.services import QAService

    service = QAService()
    service.build_index(text)
    results = service.run_default_questions()
    answer = service.ask_question("Who is the plaintiff?")
"""

import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def get_embeddings_model():
    """
    Get the shared embedding model instance (GPU-aware, prefix-configured).

    Re-exports from faiss_semantic for use by UI layer (which cannot
    import from src.core.* directly).

    Returns:
        HuggingFaceEmbeddings instance
    """
    from src.core.retrieval.algorithms.faiss_semantic import get_embeddings_model as _get

    return _get()


class QAService:
    """
    Service layer for Q&A operations.

    Coordinates vector store building, retrieval, and answer generation.
    Provides a simplified interface for the UI layer.
    """

    def __init__(self, vector_store_path: Path | None = None):
        """
        Initialize the Q&A service.

        Args:
            vector_store_path: Path to store/load vector index.
                               If None, uses a temp directory.
        """
        self._vector_store_path = vector_store_path
        self._temp_dir: Path | None = None  # Tracks temp dir we created (for cleanup)
        self._embeddings = None
        self._orchestrator = None
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """Check if Q&A service is ready (index built)."""
        return self._is_ready

    def build_index(
        self, text: str, progress_callback: Callable[[str], None] | None = None
    ) -> bool:
        """
        Build the vector index for Q&A.

        Args:
            text: Document text to index
            progress_callback: Optional callback(status_message) for updates

        Returns:
            True if successful, False otherwise
        """
        try:
            if progress_callback:
                progress_callback("Loading embeddings model...")

            # Lazy-load embeddings (shared instance, GPU-aware)
            if self._embeddings is None:
                from src.core.retrieval.algorithms.faiss_semantic import get_embeddings_model

                self._embeddings = get_embeddings_model()

            if progress_callback:
                progress_callback("Chunking document...")

            # Chunk the text
            from src.core.chunking import create_unified_chunker

            chunker = create_unified_chunker()
            chunks = chunker.chunk_text(text)

            if progress_callback:
                progress_callback(f"Indexing {len(chunks)} chunks...")

            # Build vector store
            from src.core.vector_store import VectorStoreBuilder

            if self._vector_store_path is None:
                import atexit
                import shutil
                import tempfile

                self._temp_dir = Path(tempfile.mkdtemp())
                self._vector_store_path = self._temp_dir / "qa_index"

                # atexit backstop: clean up even if destroy() never runs
                temp_dir = self._temp_dir  # capture for lambda
                atexit.register(lambda p=temp_dir: shutil.rmtree(p, ignore_errors=True))

            builder = VectorStoreBuilder()
            builder.create_from_unified_chunks(
                chunks=chunks, embeddings=self._embeddings, persist_dir=self._vector_store_path
            )

            # Initialize orchestrator
            from src.core.qa import QAOrchestrator

            self._orchestrator = QAOrchestrator(
                vector_store_path=self._vector_store_path,
                embeddings=self._embeddings,
                answer_mode="ollama",
            )

            self._is_ready = True

            logger.debug("Index built with %s chunks", len(chunks))

            if progress_callback:
                progress_callback("Q&A ready")

            return True

        except Exception as e:
            logger.error("build_index failed: %s", e)
            self._is_ready = False
            return False

    def run_default_questions(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> list:
        """
        Run all default questions against the document.

        Args:
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of QAResult objects
        """
        if not self._is_ready or self._orchestrator is None:
            logger.warning("Cannot run questions - index not ready")
            return []

        return self._orchestrator.run_default_questions(progress_callback)

    def ask_question(self, question: str) -> dict | None:
        """
        Ask a follow-up question.

        Args:
            question: The question to ask

        Returns:
            QAResult object or None if not ready
        """
        if not self._is_ready or self._orchestrator is None:
            logger.warning("Cannot ask question - index not ready")
            return None

        return self._orchestrator.ask_followup(question)

    def get_default_questions(self) -> list[str]:
        """
        Get the list of default question texts.

        Returns:
            List of question strings
        """
        if self._orchestrator is None:
            # Load questions without full orchestrator
            from src.core.config import load_yaml_with_fallback
            from src.core.qa.qa_orchestrator import DEFAULT_QUESTIONS_PATH

            config = load_yaml_with_fallback(
                DEFAULT_QUESTIONS_PATH, fallback={}, log_prefix="[QAService]"
            )

            return [q.get("text", "") for q in config.get("questions", [])]

        return self._orchestrator.get_default_questions()

    def toggle_export(self, index: int) -> bool:
        """
        Toggle include_in_export for a result by index.

        Args:
            index: Index of the result to toggle

        Returns:
            New value of include_in_export
        """
        if self._orchestrator is None:
            return False

        return self._orchestrator.toggle_export(index)

    def get_results(self) -> list:
        """Get all Q&A results."""
        if self._orchestrator is None:
            return []
        return self._orchestrator.results

    def get_exportable_results(self) -> list:
        """Get only results marked for export."""
        if self._orchestrator is None:
            return []
        return self._orchestrator.get_exportable_results()

    def export_to_text(self) -> str:
        """Format exportable results as plain text."""
        if self._orchestrator is None:
            return ""
        return self._orchestrator.export_to_text()

    def export_to_csv(self) -> str:
        """Format exportable results as CSV."""
        if self._orchestrator is None:
            return ""
        return self._orchestrator.export_to_csv()

    def clear(self) -> None:
        """Clear all results and reset state."""
        if self._orchestrator:
            self._orchestrator.clear_results()
        self._is_ready = False

        logger.debug("Cleared")

    def cleanup(self) -> None:
        """Delete temp vector store directory if we created one."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)
            logger.debug("Cleaned up temp dir: %s", self._temp_dir)
            self._temp_dir = None
            self._vector_store_path = None

    def get_default_questions_manager(self):
        """
        Get the default questions manager.

        Returns:
            DefaultQuestionsManager for managing Q&A questions.
        """
        from src.core.qa.default_questions_manager import get_default_questions_manager

        return get_default_questions_manager()

    def get_reliability_level(self, probability: float) -> str:
        """
        Get reliability level string for a probability value.

        Used for displaying verification results with color coding.

        Args:
            probability: Hallucination probability (0-1).

        Returns:
            Reliability level string (e.g., "verified", "uncertain").
        """
        from src.core.qa.verification_config import get_reliability_level

        return get_reliability_level(probability)

    def get_span_category(self, span_dict: dict) -> str:
        """
        Get span category for color coding verification results.

        Args:
            span_dict: Span dictionary with hallucination probability.

        Returns:
            Category string for CSS styling.
        """
        from src.core.qa.verification_config import get_span_category

        return get_span_category(span_dict)

    def create_orchestrator(self, vector_store_path=None, embeddings=None, answer_mode="ollama"):
        """
        Create a new QAOrchestrator instance.

        Used by workers that need direct access to the orchestrator.

        Args:
            vector_store_path: Path to vector store directory.
            embeddings: Embeddings model instance.
            answer_mode: "ollama" or "retrieval_only".

        Returns:
            QAOrchestrator instance.
        """
        from src.core.qa import QAOrchestrator

        return QAOrchestrator(
            vector_store_path=vector_store_path,
            embeddings=embeddings,
            answer_mode=answer_mode,
        )

    def get_qa_result_class(self):
        """
        Get the QAResult class for type checking.

        Returns:
            QAResult class from qa_orchestrator.
        """
        from src.core.qa.qa_orchestrator import QAResult

        return QAResult

    def get_vector_store_builder(self):
        """
        Get a VectorStoreBuilder instance.

        Used by UI components that need to create vector stores directly.

        Returns:
            VectorStoreBuilder instance.
        """
        from src.core.vector_store import VectorStoreBuilder

        return VectorStoreBuilder()
