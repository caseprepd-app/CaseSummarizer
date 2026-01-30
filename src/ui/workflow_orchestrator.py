"""
Workflow Orchestrator Module

Manages the document processing workflow state machine, coordinating between:
- Document extraction (Step 1-2.5)
- Vocabulary extraction (Step 2.5b)
- AI summary generation (Step 3+)

This module separates workflow orchestration logic from UI updates,
improving testability and maintainability.

Workflow State Machine (SEQUENTIAL - Session 14 fix):
    IDLE -> EXTRACTING -> VOCAB_EXTRACTION -> AI_GENERATION -> COMPLETE

Key Design Decision:
    Vocabulary extraction completes BEFORE AI generation starts.
    This ensures the user can view/download the vocab list while AI runs.

The orchestrator decides WHAT to do next; the QueueMessageHandler decides
HOW to update the UI in response.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import LEGAL_EXCLUDE_LIST_PATH, MEDICAL_TERMS_LIST_PATH, USER_VOCAB_EXCLUDE_PATH
from src.ui.queue_messages import QueueMessage

logger = logging.getLogger(__name__)


@dataclass
class WorkflowState:
    """
    Represents the current state of a document processing workflow.

    Attributes:
        extracted_documents: List of documents that have been extracted
        pending_ai_params: AI generation parameters (model, length, options)
        output_options: Dictionary of requested output types
        vocab_complete: Whether vocabulary extraction has finished
        ai_complete: Whether AI generation has finished
        is_complete: Whether the entire workflow has finished
        vector_store_path: Path to the created vector store (for Q&A)
        vector_store_ready: Whether vector store is ready for Q&A
    """

    extracted_documents: list[dict] = None
    pending_ai_params: dict | None = None
    output_options: dict | None = None
    vocab_complete: bool = False
    ai_complete: bool = False
    is_complete: bool = False
    vector_store_path: Path | None = None
    vector_store_ready: bool = False

    def __post_init__(self):
        if self.extracted_documents is None:
            self.extracted_documents = []


class WorkflowOrchestrator:
    """
    Orchestrates the document processing workflow.

    This class is responsible for:
    1. Deciding what workflow steps to execute next
    2. Managing workflow state transitions
    3. Coordinating parallel execution of vocabulary extraction and AI generation

    It does NOT directly update the UI - that's the QueueMessageHandler's job.

    Example:
        orchestrator = WorkflowOrchestrator(main_window)
        orchestrator.on_extraction_complete(documents, ai_params)
    """

    def __init__(self, main_window):
        """
        Initialize the workflow orchestrator.

        Args:
            main_window: Reference to MainWindow instance (for spawning workers
                        and accessing state). The orchestrator reads from but
                        doesn't directly modify UI widgets.
        """
        self.main_window = main_window
        self.state = WorkflowState()
        self.vocab_worker = None  # Track vocabulary worker for cancellation

    def get_output_options(self) -> dict[str, bool]:
        """
        Read current output options from UI checkboxes.

        Returns:
            Dictionary with keys: individual_summaries, meta_summary, vocab_csv, qa_questions
        """
        return {
            "individual_summaries": self.main_window.output_options.individual_summaries_check.get(),
            "meta_summary": self.main_window.output_options.meta_summary_check.get(),
            "vocab_csv": self.main_window.output_options.vocab_csv_check.get(),
            "qa_questions": self.main_window.output_options.qa_questions_check.get(),
        }

    def on_extraction_complete(
        self, extracted_documents: list[dict], ai_params: dict | None
    ) -> dict[str, Any]:
        """
        Handle completion of document extraction phase.

        This is the main orchestration method. It decides what workflow steps
        to execute next based on the current state and user options.

        SEQUENTIAL WORKFLOW (Session 14):
        1. If vocab requested → start vocab extraction, wait for completion
        2. When vocab completes → start AI generation
        3. This ensures user can view/export vocab while AI runs

        Args:
            extracted_documents: List of extracted document result dictionaries
            ai_params: AI generation parameters, or None if AI not requested

        Returns:
            Dictionary describing actions taken:
            {
                'vocab_extraction_started': bool,
                'ai_generation_started': bool,
                'workflow_complete': bool,
                'combined_text': str (if vocab extraction started)
            }
        """
        logger.info("Extraction complete. %s documents.", len(extracted_documents))

        # Update state
        self.state.extracted_documents = extracted_documents
        self.state.pending_ai_params = ai_params
        self.state.output_options = self.get_output_options()
        self.state.vocab_complete = False
        self.state.ai_complete = False
        self.state.vector_store_ready = False
        self.state.vector_store_path = None

        actions_taken = {
            "vocab_extraction_started": False,
            "ai_generation_started": False,
            "vector_store_started": False,
            "workflow_complete": False,
            "combined_text": None,
        }

        # Always start vector store creation in background (for Q&A feature)
        # This runs in parallel with vocabulary extraction and AI generation
        if extracted_documents:
            self._create_vector_store_async(extracted_documents)
            actions_taken["vector_store_started"] = True
            logger.debug("Started vector store creation (background).")

        # If no AI generation requested and no vocab requested, workflow is done
        # Note: Vector store creation continues in background even if workflow is "complete"
        if not ai_params and not self.state.output_options.get("vocab_csv", False):
            logger.debug("No outputs requested. Workflow complete.")
            self.state.is_complete = True
            actions_taken["workflow_complete"] = True
            return actions_taken

        # SEQUENTIAL: Start vocabulary extraction FIRST if requested
        # AI generation will start AFTER vocab completes (in on_vocab_complete)
        if self.state.output_options.get("vocab_csv", False):
            combined_text, doc_count, doc_confidence = self._get_combined_text(extracted_documents)
            actions_taken["combined_text"] = combined_text
            actions_taken["vocab_extraction_started"] = True
            self._start_vocab_extraction(
                combined_text,
                doc_count,
                doc_confidence,
                extracted_documents=extracted_documents,
            )
            logger.debug("Started vocabulary extraction (AI will start after).")
            # Do NOT start AI here - wait for vocab to complete
            return actions_taken

        # If no vocab requested but AI is, start AI directly
        if ai_params:
            self._start_ai_generation(extracted_documents, ai_params)
            actions_taken["ai_generation_started"] = True
            logger.debug("Started AI summary generation (no vocab requested).")

        return actions_taken

    def _get_combined_text(self, extracted_documents: list[dict]) -> tuple:
        """
        Get combined text from all documents for vocabulary extraction.

        Uses the shared utility function from src/utils/text_utils.
        Text is already preprocessed by ProcessingWorker (stored as 'preprocessed_text'),
        so this just combines the documents.

        Args:
            extracted_documents: List of document result dictionaries

        Returns:
            Tuple of (combined_text, doc_count, doc_confidence)
        """
        # Text is already preprocessed by ProcessingWorker
        # combine_document_texts prefers 'preprocessed_text' over 'extracted_text'
        from src.services import DocumentService

        doc_service = DocumentService()
        combined = doc_service.combine_document_texts(extracted_documents)
        doc_count = sum(
            1 for d in extracted_documents if d.get("preprocessed_text") or d.get("extracted_text")
        )

        # Calculate minimum confidence (Session 54)
        confidences = [
            d.get("confidence", 100) for d in extracted_documents if d.get("confidence") is not None
        ]
        doc_confidence = min(confidences) if confidences else 100.0

        logger.debug(
            "Combined %s documents (%s characters total, confidence=%.1f%%).",
            doc_count,
            len(combined),
            doc_confidence,
        )
        return combined, doc_count, doc_confidence

    def _start_vocab_extraction(
        self,
        combined_text: str,
        doc_count: int = 1,
        doc_confidence: float = 100.0,
        extracted_documents: list[dict] | None = None,
    ):
        """
        Start vocabulary extraction worker thread.

        Args:
            combined_text: Combined text from all documents
            doc_count: Number of documents being processed (for frequency filtering)
            doc_confidence: Aggregate OCR confidence (0-100) for ML feature (Session 54)
            extracted_documents: Optional list of per-document dicts for parallel extraction
        """
        # Import here to avoid circular imports
        from src.services.workers import VocabularyWorker

        # Session 131: Build per-document list for parallel extraction
        documents = None
        if extracted_documents and len(extracted_documents) > 1:
            documents = [
                {
                    "text": d.get("preprocessed_text") or d.get("extracted_text", ""),
                    "doc_id": d.get("filename", f"doc_{i}"),
                    "confidence": d.get("confidence", 100),
                }
                for i, d in enumerate(extracted_documents)
                if d.get("preprocessed_text") or d.get("extracted_text")
            ]

        self.vocab_worker = VocabularyWorker(
            combined_text=combined_text,
            ui_queue=self.main_window.ui_queue,
            exclude_list_path=str(LEGAL_EXCLUDE_LIST_PATH),
            medical_terms_path=str(MEDICAL_TERMS_LIST_PATH),
            user_exclude_path=str(USER_VOCAB_EXCLUDE_PATH),
            doc_count=doc_count,
            doc_confidence=doc_confidence,
            documents=documents,
        )
        self.vocab_worker.start()
        logger.debug(
            "VocabularyWorker thread started (doc_count=%s, confidence=%.1f%%, per_doc=%s).",
            doc_count,
            doc_confidence,
            documents is not None,
        )

    def _start_ai_generation(self, extracted_documents: list[dict], ai_params: dict):
        """
        Start AI summary generation - routes to single or multi-doc mode.

        Single document: Uses existing Ollama worker for direct summarization.
        Multiple documents: Uses MultiDocSummaryWorker for hierarchical
        map-reduce summarization (each doc chunked, then meta-summarized).

        Args:
            extracted_documents: List of extracted document dictionaries
            ai_params: AI generation parameters (model, length, options)
        """
        # Count documents with actual text
        valid_docs = [d for d in extracted_documents if d.get("extracted_text")]
        doc_count = len(valid_docs)

        logger.debug("Starting AI generation for %s document(s)", doc_count)

        if doc_count <= 1:
            # Single document: Use existing fast path via main window
            logger.debug("Single document mode - using direct Ollama summarization")
            self.main_window._start_ai_generation(extracted_documents, ai_params)
        else:
            # Multiple documents: Use hierarchical map-reduce via MultiDocSummaryWorker
            logger.debug("Multi-document mode - using hierarchical summarization")
            self._start_multi_doc_generation(valid_docs, ai_params)

    def _start_multi_doc_generation(self, documents: list[dict], ai_params: dict):
        """
        Start multi-document hierarchical summarization.

        Uses MultiDocSummaryWorker which:
        1. Summarizes each document in parallel via progressive chunking
        2. Combines individual summaries into a coherent meta-summary

        Args:
            documents: List of documents with 'filename' and 'extracted_text'
            ai_params: AI parameters (model_name, summary_length, etc.)
        """
        from src.services.workers import MultiDocSummaryWorker

        # Track worker for cancellation
        self.multi_doc_worker = MultiDocSummaryWorker(
            documents=documents, ui_queue=self.main_window.ui_queue, ai_params=ai_params
        )
        self.multi_doc_worker.start()
        logger.debug("MultiDocSummaryWorker started for %s documents", len(documents))

    def on_summary_complete(self):
        """Handle completion of AI summary generation."""
        logger.info("AI summary generation complete.")
        self.state.ai_complete = True
        self._check_workflow_complete()

    def on_vocab_complete(self):
        """
        Handle completion of vocabulary extraction.

        SEQUENTIAL WORKFLOW: Now that vocab is done and visible to user,
        start AI generation if it was requested.
        """
        logger.info("Vocabulary extraction complete.")
        self.state.vocab_complete = True

        # Now start AI generation if it was requested
        if self.state.pending_ai_params and self.state.output_options.get("meta_summary", False):
            logger.debug("Vocab complete, now starting AI generation...")
            self._start_ai_generation(self.state.extracted_documents, self.state.pending_ai_params)
        else:
            # No AI requested, workflow is complete
            logger.debug("No AI generation requested. Workflow complete.")
            self._check_workflow_complete()

    def _check_workflow_complete(self):
        """Check if all requested workflow steps are complete."""
        vocab_needed = (
            self.state.output_options.get("vocab_csv", False)
            if self.state.output_options
            else False
        )
        ai_needed = (
            self.state.output_options.get("meta_summary", False)
            if self.state.output_options
            else False
        )

        vocab_done = self.state.vocab_complete or not vocab_needed
        ai_done = self.state.ai_complete or not ai_needed

        if vocab_done and ai_done:
            self.state.is_complete = True
            logger.info("All workflow steps complete.")

    def reset(self):
        """Reset the orchestrator state for a new workflow."""
        self.state = WorkflowState()
        logger.debug("State reset for new workflow.")

    # =========================================================================
    # Vector Store Methods (Session 24 - Q&A Feature)
    # =========================================================================

    def _create_vector_store_async(self, extracted_documents: list[dict]):
        """
        Create vector store in background thread.

        Builds a FAISS index from extracted documents for Q&A retrieval.
        Runs in parallel with vocabulary extraction and AI generation.

        Args:
            extracted_documents: List of document dicts with 'extracted_text'
        """

        def build_store():
            try:
                logger.debug("Vector store thread started.")

                # Update status - embedding model load can take 15-30s on first run
                if self.state.output_options and self.state.output_options.get(
                    "qa_questions", False
                ):
                    self.main_window.ui_queue.put(
                        QueueMessage.progress(65, "Loading AI embedding model...")
                    )

                # Import and initialize embeddings
                # NOTE: This import triggers torch loading which can take 10-15 seconds

                from src.services import QAService

                # Initialize embeddings (shared instance, GPU-aware)
                from src.services.qa_service import get_embeddings_model

                embeddings = get_embeddings_model()

                # Update status now that model is loaded
                if self.state.output_options and self.state.output_options.get(
                    "qa_questions", False
                ):
                    self.main_window.ui_queue.put(
                        QueueMessage.progress(75, "Building Q&A search index...")
                    )

                # Build vector store via QAService
                qa_service = QAService()
                builder = qa_service.get_vector_store_builder()
                result = builder.create_from_documents(
                    documents=extracted_documents, embeddings=embeddings
                )

                # Notify UI that vector store is ready
                self.main_window.ui_queue.put(
                    QueueMessage.vector_store_ready(
                        path=result.persist_dir,
                        case_id=result.case_id,
                        chunk_count=result.chunk_count,
                        creation_time_ms=result.creation_time_ms,
                    )
                )

                logger.info(
                    "Vector store ready: %s (%s chunks, %.0fms)",
                    result.case_id,
                    result.chunk_count,
                    result.creation_time_ms,
                )

            except Exception as e:
                logger.error("Vector store creation failed: %s", e)
                self.main_window.ui_queue.put(QueueMessage.vector_store_error(str(e)))

        thread = threading.Thread(target=build_store, daemon=True, name="VectorStoreBuilder")
        thread.start()

    def on_vector_store_complete(self, result: dict):
        """
        Handle completion of vector store creation.

        Updates workflow state with vector store path for Q&A retrieval.
        If Q&A is requested, starts the Q&A worker.

        Args:
            result: Dictionary with 'path', 'case_id', 'chunk_count'
        """
        self.state.vector_store_path = result.get("path")
        self.state.vector_store_ready = True
        logger.debug("Vector store complete: %s", result.get("case_id"))

        # Start Q&A processing if requested
        if self.state.output_options and self.state.output_options.get("qa_questions", False):
            self._start_qa_processing(result)

    def _start_qa_processing(self, vector_store_result: dict):
        """
        Start Q&A worker to process default questions against documents.

        Args:
            vector_store_result: Dictionary with 'path', 'case_id', 'chunk_count'
        """

        from src.services.workers import QAWorker
        from src.user_preferences import get_user_preferences

        logger.debug("Starting Q&A processing...")

        # Get user's preferred answer mode from settings
        prefs = get_user_preferences()
        answer_mode = prefs.get("qa_answer_mode", "ollama")

        # Initialize embeddings (shared instance, GPU-aware)
        from src.services.qa_service import get_embeddings_model

        embeddings = get_embeddings_model()

        self.qa_worker = QAWorker(
            vector_store_path=Path(vector_store_result["path"]),
            embeddings=embeddings,
            ui_queue=self.main_window.ui_queue,
            answer_mode=answer_mode,
        )
        self.qa_worker.start()
        logger.debug("QAWorker started (mode=%s)", answer_mode)

    def ask_followup_question(self, question: str):
        """
        Ask a follow-up question against the current vector store.

        This runs in a background thread and sends the result to the UI queue.

        Args:
            question: The question text to ask
        """
        if not self.state.vector_store_ready or not self.state.vector_store_path:
            logger.warning("Cannot ask follow-up: vector store not ready")
            self.main_window.ui_queue.put(
                QueueMessage.qa_error("Vector store not ready. Process documents first.")
            )
            return

        def ask_question():
            try:
                from langchain_huggingface import HuggingFaceEmbeddings

                from src.services import QAService
                from src.user_preferences import get_user_preferences

                logger.debug("Asking follow-up: %s...", question[:50])

                # Get answer mode from settings
                prefs = get_user_preferences()
                answer_mode = prefs.get("qa_answer_mode", "ollama")

                # Initialize embeddings
                # IMPORTANT: Explicitly set device='cpu' to avoid "meta tensor" errors
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
                )

                # Create orchestrator with vector store path via QAService
                qa_service = QAService()
                qa_orchestrator = qa_service.create_orchestrator(
                    vector_store_path=self.state.vector_store_path,
                    embeddings=embeddings,
                    answer_mode=answer_mode,
                )

                # Ask the follow-up question
                result = qa_orchestrator.ask_followup(question)

                # Send result to UI
                self.main_window.ui_queue.put(QueueMessage.qa_followup_result(result))
                logger.debug(
                    "Follow-up answered: %s...",
                    result.answer[:50] if result.answer else "No answer",
                )

            except Exception as e:
                logger.error("Follow-up error: %s", e)
                self.main_window.ui_queue.put(QueueMessage.qa_error(str(e)))

        thread = threading.Thread(target=ask_question, daemon=True, name="FollowupQuestion")
        thread.start()
