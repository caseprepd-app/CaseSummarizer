"""
Background Workers Module

Contains threading and multiprocessing workers for document processing:
- ProcessingWorker: Document extraction thread (with parallel processing)
- VocabularyWorker: Vocabulary extraction thread
- OllamaAIWorkerManager: AI generation process manager

Performance Optimizations:
- Session 14: Non-blocking termination for AI worker
- Session 18: Parallel document extraction via Strategy Pattern
- Session 49: DRY refactor - BaseWorker eliminates boilerplate

Moved to services layer in Session 83 to enforce pipeline architecture.
Workers are orchestration, not UI display.
"""

import gc
import logging
import multiprocessing
import threading
from pathlib import Path
from queue import Empty, Queue

from src.config import PARALLEL_MAX_WORKERS
from src.core.extraction import RawTextExtractor
from src.core.parallel import (
    ExecutorStrategy,
    ParallelTaskRunner,
    ProgressAggregator,
    ThreadPoolStrategy,
)
from src.core.vocabulary import VocabularyExtractor
from src.ui.base_worker import BaseWorker, CleanupWorker
from src.ui.ollama_worker import ollama_generation_worker_process
from src.ui.queue_messages import QueueMessage

logger = logging.getLogger(__name__)


class ProcessingWorker(BaseWorker):
    """
    Background worker for parallel document extraction and normalization.

    Uses the Strategy Pattern for parallel execution, enabling:
    - Production: ThreadPoolStrategy for concurrent document processing
    - Testing: SequentialStrategy for deterministic, debuggable tests

    The worker processes multiple documents concurrently (up to PARALLEL_MAX_WORKERS)
    while maintaining responsive UI updates via ProgressAggregator.

    Session 49: Refactored to use BaseWorker.

    Attributes:
        file_paths: List of document paths to process.
        ui_queue: Queue for communication with the main UI thread.
        jurisdiction: Legal jurisdiction for document parsing (default "ny").
        strategy: ExecutorStrategy for parallel execution (injectable for testing).
        processed_results: List of extraction results after processing.

    Example:
        # Standard usage (parallel)
        worker = ProcessingWorker(file_paths, ui_queue)
        worker.start()

        # Testing (sequential, deterministic)
        from src.core.parallel import SequentialStrategy
        worker = ProcessingWorker(
            file_paths, ui_queue,
            strategy=SequentialStrategy()
        )
    """

    def __init__(
        self,
        file_paths: list,
        ui_queue: Queue,
        jurisdiction: str = "ny",
        strategy: ExecutorStrategy = None,
    ):
        """
        Initialize the processing worker.

        Args:
            file_paths: List of document file paths to process.
            ui_queue: Queue for UI communication.
            jurisdiction: Legal jurisdiction for parsing (default "ny").
            strategy: ExecutorStrategy for execution. Defaults to ThreadPoolStrategy
                     with PARALLEL_MAX_WORKERS from config.
        """
        super().__init__(ui_queue)
        self.file_paths = file_paths
        self.jurisdiction = jurisdiction

        # Dependency injection: use provided strategy or default ThreadPool
        self.strategy = strategy or ThreadPoolStrategy(max_workers=PARALLEL_MAX_WORKERS)

        # RawTextExtractor is thread-safe (stateless after init)
        self.extractor = RawTextExtractor(jurisdiction=self.jurisdiction)

        self.processed_results = []
        self._runner = None  # Track runner for cancellation

    def stop(self):
        """
        Signals the worker to stop processing.

        Cancels any pending tasks and shuts down the executor.
        Tasks in progress may complete before shutdown.
        """
        super().stop()
        if self._runner:
            self._runner.cancel()
        self.strategy.shutdown(wait=False, cancel_futures=True)

    def execute(self):
        """
        Execute parallel document extraction.

        Processes documents concurrently using the configured strategy.
        Results are collected in completion order and sent to the UI
        as they finish.
        """
        total_files = len(self.file_paths)
        self.processed_results = []

        if total_files == 0:
            self.ui_queue.put(QueueMessage.processing_finished([]))
            return

        logger.debug(
            "Starting parallel extraction of %s documents (max_workers=%s)",
            total_files,
            self.strategy.max_workers,
        )

        # Set up progress aggregation
        aggregator = ProgressAggregator(self.ui_queue, throttle_ms=100)
        aggregator.set_total(total_files)

        def process_single_doc(file_path: str) -> dict:
            """
            Process a single document (called in thread pool).

            Args:
                file_path: Path to the document file.

            Returns:
                dict: Extraction result from RawTextExtractor.

            Raises:
                InterruptedError: If stop signal received during processing.
            """
            if self.is_stopped:
                raise InterruptedError("Processing cancelled")

            filename = Path(file_path).name
            aggregator.update(file_path, f"Extracting {filename}...")

            # Progress callback that checks for cancellation
            def progress_callback(msg, pct=0):
                if self.is_stopped:
                    raise InterruptedError("Processing stopped by user.")
                # Update aggregator with detailed message
                aggregator.update(file_path, msg)

            result = self.extractor.process_document(file_path, progress_callback=progress_callback)

            aggregator.complete(file_path)
            return result

        def on_task_complete(task_id: str, result: dict):
            """Callback when a document finishes processing."""
            self.ui_queue.put(QueueMessage.file_processed(result))

        # Create and run the task runner
        self._runner = ParallelTaskRunner(strategy=self.strategy, on_task_complete=on_task_complete)

        # Prepare tasks: (task_id, payload) tuples
        items = [(fp, fp) for fp in self.file_paths]

        # Execute parallel processing
        results = self._runner.run(process_single_doc, items)

        # Collect successful results
        for task_result in results:
            if task_result.success:
                self.processed_results.append(task_result.result)
            else:
                # Log errors but continue with other documents
                logger.debug("Document failed: %s - %s", task_result.task_id, task_result.error)

        # Apply preprocessing to all results (removes line numbers, headers, etc.)
        # Store as "preprocessed_text" so downstream consumers don't need to preprocess again
        if self.processed_results:
            from src.core.preprocessing import create_default_pipeline

            preprocessor = create_default_pipeline()
            for result in self.processed_results:
                extracted = result.get("extracted_text", "")
                if extracted:
                    result["preprocessed_text"] = preprocessor.process(extracted)
            logger.debug("Preprocessing applied to %s documents", len(self.processed_results))

        # Send completion message if not cancelled
        if not self.is_stopped:
            self.ui_queue.put(QueueMessage.processing_finished(self.processed_results))
            self.send_progress(
                100, f"Processed {len(self.processed_results)}/{total_files} documents"
            )
            logger.debug("Completed: %s/%s documents", len(self.processed_results), total_files)
        else:
            logger.debug("Processing cancelled by user.")
            self.ui_queue.put(QueueMessage.error("Document processing cancelled."))

    def _cleanup(self):
        """Clean up strategy on exit."""
        # Note: shutdown already called in stop() - no action needed here
        pass


class VocabularyWorker(BaseWorker):
    """
    Background worker for vocabulary extraction (Step 2.5).
    Extracts unusual terms from combined document text asynchronously.

    Session 43: Added use_llm parameter for NER+LLM reconciled extraction.
    Session 49: Refactored to use BaseWorker.
    Session 54: Added doc_confidence for ML feature (OCR quality signal).
    Session 78: Added documents parameter for per-document extraction with TermSources.
    """

    def __init__(
        self,
        combined_text,
        ui_queue,
        exclude_list_path=None,
        medical_terms_path=None,
        user_exclude_path=None,
        doc_count=1,
        use_llm=True,
        doc_confidence=100.0,
        documents=None,  # Session 78: Per-document extraction
    ):
        super().__init__(ui_queue)
        self.combined_text = combined_text
        self.exclude_list_path = exclude_list_path or "config/legal_exclude.txt"
        self.medical_terms_path = medical_terms_path or "config/medical_terms.txt"
        self.user_exclude_path = user_exclude_path  # User's personal exclusion list
        self.doc_count = doc_count  # Number of documents (for frequency filtering)
        self.use_llm = use_llm  # Whether to use LLM extraction (Session 43)
        self.doc_confidence = doc_confidence  # Session 54: Aggregate OCR confidence for ML
        # Session 78: Per-document extraction with TermSources tracking
        # documents is a list of dicts with 'text', 'doc_id', 'confidence' keys
        self.documents = documents

    def execute(self):
        """Execute vocabulary extraction in background thread."""
        self.check_cancelled("Vocabulary extraction cancelled.")

        # Show text size to set user expectations
        text_len = len(self.combined_text)
        text_kb = text_len // 1024
        self.send_progress(30, f"Analyzing {text_kb}KB of text...")

        # Create extractor with graceful fallback for missing files
        try:
            extractor = VocabularyExtractor(
                exclude_list_path=self.exclude_list_path,
                medical_terms_path=self.medical_terms_path,
                user_exclude_path=self.user_exclude_path,
            )
        except FileNotFoundError as e:
            # Graceful fallback: create extractor with empty exclude lists
            logger.debug("Config file missing: %s. Using empty exclude lists.", e)
            extractor = VocabularyExtractor(
                exclude_list_path=None,  # Will use empty list
                medical_terms_path=None,  # Will use empty list
                user_exclude_path=self.user_exclude_path,  # Still try user list
            )

        # Check for cancellation before heavy processing
        self.check_cancelled()

        # Session 78: Use per-document extraction if documents are provided
        if self.documents and not self.use_llm:
            # Per-document extraction with TermSources tracking
            self.send_progress(40, f"Extracting vocabulary from {len(self.documents)} documents...")

            def doc_progress(current, total):
                pct = 40 + int((current / total) * 25)  # 40-65% range
                self.send_progress(pct, f"Processing document {current}/{total}...")

            vocab_data = extractor.extract_per_document(
                self.documents, progress_callback=doc_progress
            )
        elif self.use_llm:
            # Update progress - NLP/LLM processing is the slow part
            self.send_progress(40, "Running local + LLM extraction (this may take a while)...")

            # Progress callback for LLM chunk processing
            def llm_progress(current, total):
                pct = 40 + int((current / total) * 25)  # 40-65% range
                self.send_progress(pct, f"LLM analyzing chunk {current}/{total}...")

            vocab_data = extractor.extract_with_llm(
                self.combined_text,
                doc_count=self.doc_count,
                include_llm=True,
                progress_callback=llm_progress,
            )
        else:
            # Legacy NER-only extraction (combined text mode)
            self.send_progress(40, "Running local extraction (NER, RAKE)...")
            vocab_data = extractor.extract(
                self.combined_text, doc_count=self.doc_count, doc_confidence=self.doc_confidence
            )

        # Check for cancellation after extraction
        self.check_cancelled()

        term_count = len(vocab_data) if vocab_data else 0
        self.send_progress(70, f"Found {term_count} vocabulary terms")

        # Send results to GUI
        self.ui_queue.put(QueueMessage.vocab_csv_generated(vocab_data))
        logger.info("Vocabulary extraction completed: %s terms.", term_count)


class QAWorker(BaseWorker):
    """
    Background worker for Q&A document querying.

    Runs default questions against the document using FAISS vector search
    and generates answers via extraction or Ollama.

    Signals sent to ui_queue:
    - ('qa_progress', (current, total, question)) - Question being processed
    - ('qa_result', QAResult) - Single result ready
    - ('qa_complete', list[QAResult]) - All questions processed
    - ('error', str) - Error occurred

    Session 49: Refactored to use BaseWorker.

    Example:
        worker = QAWorker(
            vector_store_path=Path("./vector_stores/case_123"),
            embeddings=embeddings_model,
            ui_queue=ui_queue,
            answer_mode="extraction"
        )
        worker.start()
    """

    def __init__(
        self,
        vector_store_path: Path,
        embeddings,
        ui_queue: Queue,
        answer_mode: str = "ollama",
        questions: list[str] | None = None,
        use_default_questions: bool = False,
    ):
        """
        Initialize Q&A worker.

        Args:
            vector_store_path: Path to FAISS index directory
            embeddings: HuggingFaceEmbeddings model
            ui_queue: Queue for UI communication
            answer_mode: "extraction" or "ollama"
            questions: Custom questions to ask (None = use defaults from YAML)
            use_default_questions: If True, load questions from qa_default_questions.txt
        """
        super().__init__(ui_queue)
        self.vector_store_path = Path(vector_store_path)
        self.embeddings = embeddings
        self.answer_mode = answer_mode
        self.custom_questions = questions
        self.use_default_questions = use_default_questions
        self.results: list = []

    def execute(self):
        """Execute Q&A in background thread."""
        from src.core.qa import QAOrchestrator

        logger.debug("Starting Q&A with mode: %s", self.answer_mode)

        # Initialize orchestrator
        orchestrator = QAOrchestrator(
            vector_store_path=self.vector_store_path,
            embeddings=self.embeddings,
            answer_mode=self.answer_mode,
        )

        # Determine which questions to ask and whether they are default questions
        if self.custom_questions:
            # User provided specific custom questions
            questions = self.custom_questions
            is_default = False
            logger.debug("Using %s custom questions", len(questions))
        else:
            # Use enabled questions from DefaultQuestionsManager (respects user toggles)
            questions = orchestrator.load_default_questions_from_txt()
            is_default = True
            logger.debug("Using %s enabled default questions", len(questions))

        total = len(questions)
        if total == 0:
            logger.debug("No questions to process")
            self.ui_queue.put(QueueMessage.qa_complete([]))
            return

        logger.debug("Processing %s questions", total)

        # Process questions (parallel when beneficial - Session 69)
        self.results = self._process_questions_parallel(orchestrator, questions, is_default, total)

        # Send completion signal with all results
        self.ui_queue.put(QueueMessage.qa_complete(self.results))
        logger.info("All %s questions processed successfully", total)

    def _process_questions_parallel(
        self, orchestrator, questions: list[str], is_default: bool, total: int
    ) -> list:
        """
        Process Q&A questions, using parallelization when beneficial (Session 69).

        Uses ThreadPoolStrategy to process 2-4 questions concurrently.
        Falls back to sequential execution when:
        - Only 1 question to ask
        - Ollama is unavailable (answer_mode != "ollama")

        Results are streamed to UI as they complete via qa_result messages.

        Args:
            orchestrator: QAOrchestrator instance
            questions: List of questions to ask
            is_default: Whether these are default questions
            total: Total number of questions for progress tracking

        Returns:
            List of QAResult objects in original question order
        """
        import os
        import threading

        from src.core.parallel.executor_strategy import ThreadPoolStrategy
        from src.core.parallel.task_runner import ParallelTaskRunner
        from src.system_resources import get_optimal_workers

        # Decide whether to parallelize
        # Skip for single question or non-Ollama mode (extraction is fast enough)
        cpu_count = os.cpu_count() or 1
        use_parallel = (
            len(questions) > 1
            and cpu_count > 1
            and self.answer_mode == "ollama"  # Ollama benefits from parallelization
        )

        if not use_parallel:
            # Sequential fallback
            logger.debug("Processing %s question(s) sequentially", len(questions))
            results = []
            for i, question in enumerate(questions):
                self.check_cancelled()

                # Report progress
                truncated_q = question[:50] + "..." if len(question) > 50 else question
                self.ui_queue.put(QueueMessage.qa_progress(i, total, truncated_q))

                result = orchestrator._ask_single_question(
                    question, is_followup=False, is_default=is_default
                )
                results.append(result)
                self.ui_queue.put(QueueMessage.qa_result(result))
                logger.debug("Q%s/%s complete: %s chars", i + 1, total, len(result.answer))

            return results

        # Parallel execution
        # Limit workers to avoid overloading Ollama (typically 2-4)
        workers = min(
            len(questions), get_optimal_workers(task_ram_gb=2.0, max_workers=4, min_workers=2)
        )
        logger.debug("Processing %s questions in parallel (%s workers)", len(questions), workers)

        # Track completion for progress reporting
        completed_count = [0]  # Using list for mutable in closure
        count_lock = threading.Lock()
        results_dict = {}  # Store results by index for ordering
        results_lock = threading.Lock()  # Session 70: Thread-safe dict access

        def ask_question(args):
            """Worker function to ask a single question."""
            idx, question = args

            # Check cancellation
            if self._stop_event.is_set():
                return (idx, None)

            result = orchestrator._ask_single_question(
                question, is_followup=False, is_default=is_default
            )
            return (idx, result)

        strategy = ThreadPoolStrategy(max_workers=workers)

        try:
            # Build items: (task_id, (index, question))
            items = [(f"Q{i + 1}", (i, q)) for i, q in enumerate(questions)]

            def on_complete(task_id: str, result):
                """Callback when question completes - stream to UI."""
                idx, qa_result = result
                if qa_result is None:
                    return  # Cancelled

                # Thread-safe completion tracking
                with count_lock:
                    completed_count[0] += 1
                    count = completed_count[0]

                # Store result for ordered return (Session 70: thread-safe)
                with results_lock:
                    results_dict[idx] = qa_result

                # Stream result to UI
                self.ui_queue.put(QueueMessage.qa_result(qa_result))

                # Progress update
                truncated_q = (
                    questions[idx][:50] + "..." if len(questions[idx]) > 50 else questions[idx]
                )
                self.ui_queue.put(QueueMessage.qa_progress(count - 1, total, truncated_q))

                logger.debug("Q%s/%s complete: %s chars", idx + 1, total, len(qa_result.answer))

            runner = ParallelTaskRunner(strategy=strategy, on_task_complete=on_complete)
            task_results = runner.run(ask_question, items)

            # Handle any failures
            for task_result in task_results:
                if not task_result.success:
                    logger.debug("Question %s failed: %s", task_result.task_id, task_result.error)

            # Return results in original order (Session 70: thread-safe read)
            with results_lock:
                ordered_results = [results_dict.get(i) for i in range(len(questions))]
            final_results = [r for r in ordered_results if r is not None]

            # LOG-009: Log if any questions failed/were cancelled
            failed_count = len(ordered_results) - len(final_results)
            if failed_count > 0:
                logger.debug("%s/%s questions returned no result", failed_count, len(questions))

            return final_results

        finally:
            strategy.shutdown(wait=True)


class OllamaAIWorkerManager:
    """
    Manages the multiprocessing worker for Ollama AI generation.
    Handles starting, stopping, and communication with the worker process.
    """

    def __init__(self, ui_queue: Queue):
        self.ui_queue = ui_queue
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.process = None
        self.is_running = False

    @staticmethod
    def _clear_queue(queue):
        """Safely clear all messages from a queue."""
        # LOG-010: Track how many items cleared for debugging
        cleared_count = 0
        while not queue.empty():
            try:
                queue.get_nowait()
                cleared_count += 1
            except Empty:
                break
        if cleared_count > 0:
            logger.debug("Cleared %s items from queue", cleared_count)

    def start_worker(self):
        """Starts the Ollama AI worker process."""
        if self.is_running and self.process and self.process.is_alive():
            logger.debug("Worker already running.")
            return

        # Ensure queues are empty from previous runs
        self._clear_queue(self.input_queue)
        self._clear_queue(self.output_queue)

        logger.debug("Starting Ollama AI worker process.")
        self.process = multiprocessing.Process(
            target=ollama_generation_worker_process,
            args=(self.input_queue, self.output_queue),
            daemon=True,  # Daemon process allows main process to exit even if worker is alive
        )
        self.process.start()
        self.is_running = True
        logger.debug("Worker process started with PID: %s", self.process.pid)

    def stop_worker(self, blocking=False):
        """
        Sends a termination signal and stops the Ollama AI worker process.

        Args:
            blocking: If True, wait for process to terminate. If False (default),
                     terminate immediately without blocking the main thread.
        """
        if self.is_running and self.process and self.process.is_alive():
            logger.debug("Sending TERMINATE signal to worker.")
            try:
                self.input_queue.put_nowait("TERMINATE")  # Non-blocking put

                if blocking:
                    # Wait briefly for graceful shutdown
                    self.process.join(timeout=2)
                else:
                    # Non-blocking: check if it's already dead, but don't wait
                    self.process.join(timeout=0.1)

            except Exception as e:
                logger.debug("Error sending terminate signal: %s", e)

            # Force terminate if still alive
            if self.process and self.process.is_alive():
                logger.debug("Worker did not terminate gracefully, forcing shutdown.")
                try:
                    self.process.terminate()
                    self.process.join(timeout=0.5)  # Brief wait for terminate
                except Exception as e:
                    logger.debug("Error during force terminate: %s", e)

            # Clean up queues to prevent memory leaks
            self._clear_queue(self.input_queue)
            self._clear_queue(self.output_queue)

            self.process = None
            self.is_running = False

            # Force garbage collection
            gc.collect()

            logger.debug("Ollama AI worker process stopped, memory cleaned.")
        elif self.is_running:
            logger.debug("Worker process already stopped or not alive.")
            self.process = None
            self.is_running = False

    def send_task(self, task_type: str, payload: dict):
        """Sends a task to the worker process."""
        if not (self.is_running and self.process and self.process.is_alive()):
            self.start_worker()  # Ensure worker is running before sending task

        logger.debug("Sending task '%s' to worker.", task_type)
        self.input_queue.put((task_type, payload))

    def check_for_messages(self):
        """Checks the output queue for messages from the worker process."""
        messages = []
        while not self.output_queue.empty():
            try:
                messages.append(self.output_queue.get_nowait())
            except Empty:
                break
        return messages

    def is_worker_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()


class MultiDocSummaryWorker(CleanupWorker):
    """
    Background worker for multi-document hierarchical summarization.

    Uses MultiDocumentOrchestrator to process multiple documents in parallel
    with a map-reduce approach:
    1. Map: Each document summarized via ProgressiveDocumentSummarizer
    2. Reduce: Individual summaries combined into meta-summary

    This worker runs in a background thread to keep the UI responsive
    during potentially long summarization operations.

    Session 49: Refactored to use CleanupWorker (BaseWorker with gc.collect).

    Attributes:
        documents: List of document dicts with 'filename' and 'extracted_text'.
        ui_queue: Queue for communication with the main UI thread.
        ai_params: AI parameters (summary_length, meta_length, etc.).
        strategy: ExecutorStrategy for parallel processing.
    """

    def __init__(
        self,
        documents: list[dict],
        ui_queue: Queue,
        ai_params: dict,
        strategy: ExecutorStrategy = None,
    ):
        """
        Initialize the multi-document summary worker.

        Args:
            documents: List of document dicts with 'filename' and 'extracted_text'.
            ui_queue: Queue for UI communication.
            ai_params: Dict with 'summary_length', 'meta_length', 'model_name', etc.
            strategy: ExecutorStrategy for parallel execution. Defaults to
                     ThreadPoolStrategy with PARALLEL_MAX_WORKERS.
        """
        super().__init__(ui_queue)
        self.documents = documents
        self.ai_params = ai_params
        self.strategy = strategy or ThreadPoolStrategy(max_workers=PARALLEL_MAX_WORKERS)
        self._orchestrator = None

    def stop(self):
        """Signal the worker to stop processing."""
        super().stop()
        if self._orchestrator:
            self._orchestrator.stop()
        self.strategy.shutdown(wait=False, cancel_futures=True)

    def execute(self):
        """Execute multi-document summarization in background thread."""
        doc_count = len(self.documents)
        logger.debug("Starting summarization of %s documents", doc_count)

        # Import here to avoid circular imports
        from src.core.ai import OllamaModelManager
        from src.core.summarization import (
            MultiDocumentOrchestrator,
            ProgressiveDocumentSummarizer,
        )

        # Initialize components
        model_manager = OllamaModelManager()

        # Load specified model if provided
        model_name = self.ai_params.get("model_name")
        if model_name:
            model_manager.load_model(model_name)

        # Extract preset_id from ai_params (set by main_window from user selection)
        preset_id = self.ai_params.get("preset_id", "factual-summary")
        logger.debug("Using preset_id: %s", preset_id)

        # Create prompt builder for thread-through focus areas
        # This builder extracts focus from the user's template and threads
        # it through all stages of the summarization pipeline
        from src.core.prompting import MultiDocStagePromptBuilder

        prompt_adapter = MultiDocStagePromptBuilder(
            template_manager=model_manager.prompt_template_manager, model_manager=model_manager
        )

        doc_summarizer = ProgressiveDocumentSummarizer(
            model_manager, prompt_adapter=prompt_adapter, preset_id=preset_id
        )

        self._orchestrator = MultiDocumentOrchestrator(
            document_summarizer=doc_summarizer,
            model_manager=model_manager,
            strategy=self.strategy,
            prompt_adapter=prompt_adapter,
            preset_id=preset_id,
        )

        # Progress callback to UI
        def on_progress(percent: int, message: str):
            self.send_progress(percent, message)

        # Get parameters
        summary_length = self.ai_params.get("summary_length", 200)
        meta_length = self.ai_params.get("meta_length", 500)

        # Execute summarization
        result = self._orchestrator.summarize_documents(
            documents=self.documents,
            max_words_per_document=summary_length,
            max_meta_summary_words=meta_length,
            progress_callback=on_progress,
            ui_queue=self.ui_queue,
        )

        # Send result to UI
        if not self.is_stopped:
            self.ui_queue.put(QueueMessage.multi_doc_result(result))
            logger.info(
                "Completed: %s documents, %s failed, %.1fs",
                result.documents_processed,
                result.documents_failed,
                result.total_processing_time_seconds,
            )
        else:
            logger.debug("Processing cancelled by user.")
            self.ui_queue.put(QueueMessage.error("Multi-document summarization cancelled."))

    def _cleanup(self):
        """Clean up strategy and memory."""
        self.strategy.shutdown(wait=False)
        super()._cleanup()  # Calls gc.collect()


class ProgressiveExtractionWorker(BaseWorker):
    """
    Progressive three-phase extraction worker (Session 45).

    Implements the vocabulary-first architecture with progressive output:
    - Phase 1 (NER): Fast extraction, returns results in ~5 seconds
    - Phase 2 (Q&A): Builds vector store for Q&A (parallel with Phase 3)
    - Phase 3 (LLM): Slow LLM extraction, updates progressively

    Session 49: Refactored to use BaseWorker.

    Signals sent to ui_queue:
    - ('ner_complete', vocab_data) - Phase 1 complete, display immediately
    - ('qa_ready', vector_store_path) - Phase 2 complete, enable Q&A
    - ('llm_progress', (current, total, new_terms)) - LLM chunk processed
    - ('llm_complete', reconciled_data) - Phase 3 complete, final results
    - ('error', str) - Error occurred

    Example:
        worker = ProgressiveExtractionWorker(
            documents=processed_docs,
            combined_text=full_text,
            ui_queue=ui_queue,
            embeddings=embeddings_model,
        )
        worker.start()
    """

    def __init__(
        self,
        documents: list[dict],
        combined_text: str,
        ui_queue: Queue,
        embeddings=None,  # HuggingFaceEmbeddings, lazy-load if None
        exclude_list_path: str | None = None,
        medical_terms_path: str | None = None,
        user_exclude_path: str | None = None,
        doc_confidence: float = 100.0,
        use_llm: bool = True,  # Session 62b: Respect LLM preference
    ):
        """
        Initialize progressive extraction worker.

        Args:
            documents: List of document dicts with 'filename' and 'extracted_text'
            combined_text: Combined text from all documents
            ui_queue: Queue for UI communication
            embeddings: HuggingFaceEmbeddings model (lazy-loads if None)
            exclude_list_path: Path to legal exclusion list
            medical_terms_path: Path to medical terms list
            user_exclude_path: Path to user exclusion list
            doc_confidence: Aggregate OCR confidence (0-100) for ML feature (Session 54)
            use_llm: Whether to run Phase 3 LLM extraction (Session 62b)
        """
        super().__init__(ui_queue)
        self.documents = documents
        self.combined_text = combined_text
        self.embeddings = embeddings
        self.exclude_list_path = exclude_list_path
        self.medical_terms_path = medical_terms_path
        self.user_exclude_path = user_exclude_path
        self.doc_confidence = doc_confidence  # Session 54: OCR quality for ML
        self.use_llm = use_llm  # Session 62b: Whether to run LLM phase

    def execute(self):
        """Execute three-phase progressive extraction."""
        logger.debug("Starting progressive extraction")

        # Session 85: Signal extraction started (dims feedback buttons)
        self.ui_queue.put(QueueMessage.extraction_started())

        # ===== PHASE 1: Local Algorithms (Progressive - Session 85) =====
        # Runs BM25 + RAKE first (fast), then NER with chunk progress
        logger.debug("Phase 1: Local algorithm extraction starting...")

        # Check which algorithms will run for accurate status message
        from src.config import CORPUS_MIN_DOCUMENTS
        from src.core.vocabulary.corpus_manager import get_corpus_manager

        corpus_manager = get_corpus_manager()
        bm25_active = corpus_manager.is_corpus_ready(min_docs=CORPUS_MIN_DOCUMENTS)

        algo_list = "NER, RAKE, BM25" if bm25_active else "NER, RAKE"
        self.send_progress(10, f"Phase 1: Running local extraction ({algo_list})...")

        from src.core.vocabulary import VocabularyExtractor

        extractor = VocabularyExtractor(
            exclude_list_path=self.exclude_list_path,
            medical_terms_path=self.medical_terms_path,
            user_exclude_path=self.user_exclude_path,
        )

        # Session 85: Progressive extraction with callbacks for fast UX
        def on_partial_complete(partial_vocab):
            """Called when BM25 + RAKE complete (before NER)."""
            logger.debug("Partial results: %s terms", len(partial_vocab))
            self.ui_queue.put(QueueMessage.partial_vocab_complete(partial_vocab))

        def on_ner_progress(chunk_candidates, chunk_num, total_chunks):
            """Called after each NER chunk completes."""
            pct = int((chunk_num / total_chunks) * 100)
            self.send_progress(
                10 + int((chunk_num / total_chunks) * 20),  # 10-30% range for NER
                f"NER: {pct}% complete (chunk {chunk_num}/{total_chunks})...",
            )
            self.ui_queue.put(QueueMessage.ner_progress(chunk_candidates, chunk_num, total_chunks))

        # Use progressive extraction: BM25+RAKE first, then NER with progress
        ner_results = extractor.extract_progressive(
            self.combined_text,
            doc_count=len(self.documents),
            doc_confidence=self.doc_confidence,
            partial_callback=on_partial_complete,
            ner_progress_callback=on_ner_progress,
        )

        logger.info("Phase 1 complete: %s terms from local algorithms", len(ner_results))
        self.ui_queue.put(QueueMessage.ner_complete(ner_results))

        # Session 85: Signal extraction complete (re-enables feedback buttons)
        self.ui_queue.put(QueueMessage.extraction_complete())

        self.check_cancelled()

        # ===== PHASE 2: Q&A Indexing (Fast - ~10-30 seconds) =====
        # Run in parallel thread while Phase 3 starts
        qa_thread = threading.Thread(target=self._build_vector_store, daemon=True)
        qa_thread.start()

        # ===== PHASE 3: LLM Enhancement (Slow - minutes) =====
        # Session 62b: Only run LLM phase if enabled (GPU auto-detect or user override)
        if self.use_llm:
            logger.debug("Phase 3: LLM extraction starting...")
            self.send_progress(30, "Phase 3: Starting LLM enhancement...")

            from src.core.chunking import create_unified_chunker
            from src.core.extraction import LLMVocabExtractor

            # Get NER candidates for reconciliation
            ner_candidates = []
            for algorithm in extractor.algorithms:
                if algorithm.name == "NER" and algorithm.enabled:
                    result = algorithm.extract(self.combined_text)
                    ner_candidates = result.candidates
                    break

            # Create unified chunks
            chunker = create_unified_chunker()
            chunks = chunker.chunk_text(self.combined_text)
            logger.debug("Created %s unified chunks", len(chunks))

            # Extract with LLM progressively
            llm_extractor = LLMVocabExtractor()

            def llm_progress(current, total):
                if not self.is_stopped:
                    pct = 30 + int((current / total) * 60)  # 30-90% range
                    self.send_progress(pct, f"LLM analyzing chunk {current}/{total}...")
                    self.ui_queue.put(QueueMessage.llm_progress(current, total))

            llm_result = llm_extractor.extract_from_unified_chunks(
                chunks,
                progress_callback=llm_progress,
            )

            self.check_cancelled()

            # Deduplicate NER + LLM results
            from src.core.vocabulary.reconciler import VocabularyDeduplicator

            logger.debug("Deduplicating NER + LLM results...")
            reconciler = VocabularyDeduplicator()

            # Reconcile people
            ner_people = [c for c in ner_candidates if getattr(c, "suggested_type", "") == "Person"]
            reconciled_people = reconciler.reconcile_people(ner_people, llm_result.people)

            # Reconcile vocabulary terms
            reconciled_terms = reconciler.reconcile(ner_candidates, llm_result.terms)

            # Convert to unified CSV format
            final_data = reconciler.combined_to_csv_data(reconciled_people, reconciled_terms)

            logger.info("Phase 3 complete: %s reconciled terms", len(final_data))
            self.ui_queue.put(QueueMessage.llm_complete(final_data))
            self.send_progress(100, f"Complete: {len(final_data)} names & terms found")
        else:
            # Session 62b: Skip LLM phase - NER results are already sent in Phase 1
            logger.debug("Phase 3: Skipped (LLM disabled by user preference or no GPU)")
            self.send_progress(90, "Phase 3: Skipped (LLM disabled)")
            # Signal LLM complete with empty list - UI will show NER-only results
            self.ui_queue.put(QueueMessage.llm_complete([]))
            self.send_progress(100, f"Complete: {len(ner_results)} terms (NER only)")

        # Wait for Q&A thread to finish
        # Session 80: Increased timeout from 60s to 180s - large documents can take longer
        # to chunk and build vector store, especially on first run when loading embeddings
        qa_thread.join(timeout=180)

    def _build_vector_store(self):
        """Build vector store for Q&A (Phase 2) - runs in parallel thread."""
        try:
            logger.debug("Phase 2: Building vector store...")
            self.ui_queue.put(QueueMessage.progress(20, "Phase 2: Building Q&A index..."))

            from src.core.chunking import create_unified_chunker
            from src.core.vector_store import VectorStoreBuilder

            # Lazy-load embeddings if not provided
            if self.embeddings is None:
                logger.debug("Loading embeddings model...")
                from langchain_huggingface import HuggingFaceEmbeddings

                self.embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
                )

            # Create unified chunks from each document with source attribution
            # This ensures each chunk knows which document it came from
            # Use preprocessed_text (already cleaned by ProcessingWorker) to avoid redundant work
            chunker = create_unified_chunker()
            all_chunks = []
            for doc in self.documents:
                filename = doc.get("filename", "unknown")
                # Prefer preprocessed_text (already cleaned) over extracted_text (raw)
                text = doc.get("preprocessed_text") or doc.get("extracted_text", "")
                if text.strip():
                    doc_chunks = chunker.chunk_text(text, source_file=filename)
                    all_chunks.extend(doc_chunks)

            # Build vector store
            builder = VectorStoreBuilder()
            result = builder.create_from_unified_chunks(
                chunks=all_chunks,
                embeddings=self.embeddings,
            )

            logger.info("Phase 2 complete: %s chunks indexed", result.chunk_count)
            self.ui_queue.put(
                QueueMessage.qa_ready(
                    vector_store_path=result.persist_dir,
                    embeddings=self.embeddings,
                    chunk_count=result.chunk_count,
                )
            )

            # Trigger default questions if enabled
            logger.debug("Triggering default Q&A check")
            self.ui_queue.put(
                QueueMessage.trigger_default_qa(
                    vector_store_path=result.persist_dir,
                    embeddings=self.embeddings,
                )
            )

        except Exception as e:
            logger.error("Q&A indexing failed: %s", e)
            self.ui_queue.put(QueueMessage.qa_error(str(e)))
