"""
Queue Message Factory Module

Provides type-safe message construction for inter-thread communication.
All worker threads should use QueueMessage factory methods instead of
raw tuple construction to ensure consistency and catch typos at development time.

Usage:
    from src.ui.queue_messages import QueueMessage, MessageType

    # Instead of: ui_queue.put(('progress', (50, "Processing...")))
    ui_queue.put(QueueMessage.progress(50, "Processing..."))

    # Instead of: ui_queue.put(('error', error_msg))
    ui_queue.put(QueueMessage.error(error_msg))
"""

from typing import Any


class MessageType:
    """
    Constants for all queue message types.

    Use these constants in queue_message_handler.py dispatch dictionary
    to ensure consistency between senders and receivers.
    """

    # Core messages
    PROGRESS = "progress"
    ERROR = "error"
    STATUS_ERROR = "status_error"  # Non-fatal error displayed in status bar (orange)

    # Document processing
    FILE_PROCESSED = "file_processed"
    PROCESSING_FINISHED = "processing_finished"

    # Vocabulary extraction
    VOCAB_CSV_GENERATED = "vocab_csv_generated"

    # Summarization
    SUMMARY_RESULT = "summary_result"
    MULTI_DOC_RESULT = "multi_doc_result"
    META_SUMMARY_GENERATED = "meta_summary_generated"

    # Vector Store (Session 24)
    VECTOR_STORE_READY = "vector_store_ready"
    VECTOR_STORE_ERROR = "vector_store_error"

    # Q&A (Session 28)
    QA_PROGRESS = "qa_progress"
    QA_RESULT = "qa_result"
    QA_COMPLETE = "qa_complete"
    QA_FOLLOWUP_RESULT = "qa_followup_result"
    QA_ERROR = "qa_error"
    TRIGGER_DEFAULT_QA = "trigger_default_qa"

    # Progressive Extraction (Session 48)
    NER_COMPLETE = "ner_complete"
    QA_READY = "qa_ready"
    LLM_PROGRESS = "llm_progress"
    LLM_COMPLETE = "llm_complete"

    # Progressive Vocabulary Loading (Session 85)
    PARTIAL_VOCAB_COMPLETE = "partial_vocab_complete"  # BM25 + RAKE results before NER
    NER_PROGRESS = "ner_progress"  # NER chunk progress update
    EXTRACTION_STARTED = "extraction_started"  # Signals extraction has begun (dim buttons)
    EXTRACTION_COMPLETE = "extraction_complete"  # Signals all extraction done (enable buttons)


class QueueMessage:
    """
    Factory for type-safe queue messages.

    Each method returns a tuple (message_type, payload) ready for ui_queue.put().
    Using these factory methods instead of raw tuples provides:
    - IDE autocomplete for message types
    - Type hints for payloads
    - Single source of truth for message structure
    - Compile-time detection of typos
    """

    # =========================================================================
    # Core Messages
    # =========================================================================

    @staticmethod
    def progress(percentage: int, message: str) -> tuple[str, tuple[int, str]]:
        """
        Create progress update message.

        Args:
            percentage: Progress 0-100
            message: Status message to display
        """
        return (MessageType.PROGRESS, (percentage, message))

    @staticmethod
    def error(message: str) -> tuple[str, str]:
        """
        Create error message (shows blocking modal dialog).

        Args:
            message: Human-readable error description
        """
        return (MessageType.ERROR, message)

    @staticmethod
    def status_error(message: str) -> tuple[str, str]:
        """
        Create non-fatal status bar error (orange text, no modal).

        Use this for errors that shouldn't block the user, such as
        a single document failing in a batch or an optional feature failing.

        Args:
            message: Human-readable error description
        """
        return (MessageType.STATUS_ERROR, message)

    # =========================================================================
    # Document Processing
    # =========================================================================

    @staticmethod
    def file_processed(result: dict) -> tuple[str, dict]:
        """
        Create file processed message.

        Args:
            result: Dict with filename, status, confidence, summary, page_count, etc.
        """
        return (MessageType.FILE_PROCESSED, result)

    @staticmethod
    def processing_finished(results: list[dict]) -> tuple[str, list[dict]]:
        """
        Create processing finished message.

        Args:
            results: List of all extraction result dictionaries
        """
        return (MessageType.PROCESSING_FINISHED, results)

    # =========================================================================
    # Vocabulary Extraction
    # =========================================================================

    @staticmethod
    def vocab_csv_generated(vocab_data: list[dict]) -> tuple[str, list[dict]]:
        """
        Create vocabulary CSV generated message.

        Args:
            vocab_data: List of vocabulary term dictionaries
        """
        return (MessageType.VOCAB_CSV_GENERATED, vocab_data)

    # =========================================================================
    # Summarization
    # =========================================================================

    @staticmethod
    def summary_result(summary: str) -> tuple[str, dict]:
        """
        Create single-document summary result message.

        Args:
            summary: Generated summary text
        """
        return (MessageType.SUMMARY_RESULT, {"summary": summary})

    @staticmethod
    def multi_doc_result(result: Any) -> tuple[str, Any]:
        """
        Create multi-document summary result message.

        Args:
            result: MultiDocumentSummaryResult object
        """
        return (MessageType.MULTI_DOC_RESULT, result)

    @staticmethod
    def meta_summary_generated(summary: str) -> tuple[str, str]:
        """
        Create meta-summary generated message.

        Args:
            summary: Meta-summary text
        """
        return (MessageType.META_SUMMARY_GENERATED, summary)

    # =========================================================================
    # Vector Store (Session 24)
    # =========================================================================

    @staticmethod
    def vector_store_ready(
        path: str, case_id: str, chunk_count: int, creation_time_ms: float
    ) -> tuple[str, dict]:
        """
        Create vector store ready message.

        Args:
            path: Path to persisted vector store
            case_id: Case identifier
            chunk_count: Number of chunks indexed
            creation_time_ms: Time to create in milliseconds
        """
        return (
            MessageType.VECTOR_STORE_READY,
            {
                "path": path,
                "case_id": case_id,
                "chunk_count": chunk_count,
                "creation_time_ms": creation_time_ms,
            },
        )

    @staticmethod
    def vector_store_error(error: str) -> tuple[str, dict]:
        """
        Create vector store error message.

        Args:
            error: Error message
        """
        return (MessageType.VECTOR_STORE_ERROR, {"error": error})

    # =========================================================================
    # Q&A (Session 28)
    # =========================================================================

    @staticmethod
    def qa_progress(current: int, total: int, question: str) -> tuple[str, tuple[int, int, str]]:
        """
        Create Q&A progress message.

        Args:
            current: Current question number (1-indexed)
            total: Total number of questions
            question: Current question text
        """
        return (MessageType.QA_PROGRESS, (current, total, question))

    @staticmethod
    def qa_result(result: Any) -> tuple[str, Any]:
        """
        Create single Q&A result message.

        Args:
            result: QAResult object
        """
        return (MessageType.QA_RESULT, result)

    @staticmethod
    def qa_complete(results: list) -> tuple[str, list]:
        """
        Create Q&A complete message.

        Args:
            results: List of all QAResult objects
        """
        return (MessageType.QA_COMPLETE, results)

    @staticmethod
    def qa_followup_result(result: Any) -> tuple[str, Any]:
        """
        Create Q&A follow-up result message.

        Args:
            result: QAResult object for follow-up question
        """
        return (MessageType.QA_FOLLOWUP_RESULT, result)

    @staticmethod
    def qa_error(error: str) -> tuple[str, dict]:
        """
        Create Q&A error message.

        Args:
            error: Error message
        """
        return (MessageType.QA_ERROR, {"error": error})

    @staticmethod
    def trigger_default_qa(vector_store_path: str, embeddings: Any) -> tuple[str, dict]:
        """
        Signal to trigger default Q&A questions.

        Args:
            vector_store_path: Path to vector store
            embeddings: HuggingFaceEmbeddings instance
        """
        return (
            MessageType.TRIGGER_DEFAULT_QA,
            {
                "vector_store_path": vector_store_path,
                "embeddings": embeddings,
            },
        )

    # =========================================================================
    # Progressive Extraction (Session 48)
    # =========================================================================

    @staticmethod
    def ner_complete(vocab_data: list[dict]) -> tuple[str, list[dict]]:
        """
        Create local algorithm extraction complete message (Phase 1).

        Phase 1 runs NER, RAKE, and BM25 (if corpus available) - all local,
        fast algorithms without LLM calls.

        Args:
            vocab_data: List of vocabulary terms from local algorithms
        """
        return (MessageType.NER_COMPLETE, vocab_data)

    @staticmethod
    def qa_ready(vector_store_path: str, embeddings: Any, chunk_count: int) -> tuple[str, dict]:
        """
        Create Q&A ready message (Phase 2).

        Args:
            vector_store_path: Path to vector store
            embeddings: HuggingFaceEmbeddings instance
            chunk_count: Number of chunks indexed
        """
        return (
            MessageType.QA_READY,
            {
                "vector_store_path": vector_store_path,
                "embeddings": embeddings,
                "chunk_count": chunk_count,
            },
        )

    @staticmethod
    def llm_progress(current: int, total: int) -> tuple[str, tuple[int, int]]:
        """
        Create LLM extraction progress message (Phase 3).

        Args:
            current: Current chunk number
            total: Total chunks to process
        """
        return (MessageType.LLM_PROGRESS, (current, total))

    @staticmethod
    def llm_complete(vocab_data: list[dict]) -> tuple[str, list[dict]]:
        """
        Create LLM extraction complete message (Phase 3).

        Args:
            vocab_data: Reconciled NER + LLM vocabulary terms
        """
        return (MessageType.LLM_COMPLETE, vocab_data)

    # =========================================================================
    # Progressive Vocabulary Loading (Session 85)
    # =========================================================================

    @staticmethod
    def partial_vocab_complete(vocab_data: list[dict]) -> tuple[str, list[dict]]:
        """
        Create partial vocabulary complete message (BM25 + RAKE results).

        Sent before NER completes to show initial results quickly.

        Args:
            vocab_data: List of vocabulary terms from BM25 and RAKE only
        """
        return (MessageType.PARTIAL_VOCAB_COMPLETE, vocab_data)

    @staticmethod
    def ner_progress(vocab_data: list[dict], chunk_num: int, total_chunks: int) -> tuple[str, dict]:
        """
        Create NER chunk progress message.

        Sent after each NER chunk completes with new terms found.

        Args:
            vocab_data: New vocabulary terms found in this chunk
            chunk_num: Current chunk number (1-indexed)
            total_chunks: Total number of chunks to process
        """
        return (
            MessageType.NER_PROGRESS,
            {
                "vocab_data": vocab_data,
                "chunk_num": chunk_num,
                "total_chunks": total_chunks,
            },
        )

    @staticmethod
    def extraction_started() -> tuple[str, None]:
        """
        Signal that vocabulary extraction has started.

        Used to dim feedback buttons in the UI.
        """
        return (MessageType.EXTRACTION_STARTED, None)

    @staticmethod
    def extraction_complete() -> tuple[str, None]:
        """
        Signal that vocabulary extraction is complete.

        Used to re-enable feedback buttons in the UI.
        """
        return (MessageType.EXTRACTION_COMPLETE, None)
