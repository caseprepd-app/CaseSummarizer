"""
Background Workers Module

Contains threading and multiprocessing workers for document processing:
- ProcessingWorker: Document extraction thread (with parallel processing)
- VocabularyWorker: Vocabulary extraction thread
- OllamaAIWorkerManager: AI generation process manager

Performance Optimizations:
- Session 14: Non-blocking termination for AI worker
- Session 18: Parallel document extraction via Strategy Pattern
"""

import gc
import multiprocessing
import os
from pathlib import Path
import threading
import traceback
from queue import Empty, Queue

from src.config import PARALLEL_MAX_WORKERS
from src.extraction import RawTextExtractor
from src.logging_config import debug_log
from src.parallel import (
    ExecutorStrategy,
    ParallelTaskRunner,
    ProgressAggregator,
    ThreadPoolStrategy,
)
from src.ui.ollama_worker import ollama_generation_worker_process
from src.vocabulary import VocabularyExtractor


class ProcessingWorker(threading.Thread):
    """
    Background worker for parallel document extraction and normalization.

    Uses the Strategy Pattern for parallel execution, enabling:
    - Production: ThreadPoolStrategy for concurrent document processing
    - Testing: SequentialStrategy for deterministic, debuggable tests

    The worker processes multiple documents concurrently (up to PARALLEL_MAX_WORKERS)
    while maintaining responsive UI updates via ProgressAggregator.

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
        from src.parallel import SequentialStrategy
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
        strategy: ExecutorStrategy = None
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
        super().__init__(daemon=True)
        self.file_paths = file_paths
        self.ui_queue = ui_queue
        self.jurisdiction = jurisdiction

        # Dependency injection: use provided strategy or default ThreadPool
        self.strategy = strategy or ThreadPoolStrategy(max_workers=PARALLEL_MAX_WORKERS)

        # RawTextExtractor is thread-safe (stateless after init)
        self.extractor = RawTextExtractor(jurisdiction=self.jurisdiction)

        self.processed_results = []
        self._stop_event = threading.Event()
        self._runner = None  # Track runner for cancellation

    def stop(self):
        """
        Signals the worker to stop processing.

        Cancels any pending tasks and shuts down the executor.
        Tasks in progress may complete before shutdown.
        """
        debug_log("[PROCESSING WORKER] Stop signal received.")
        self._stop_event.set()
        if self._runner:
            self._runner.cancel()
        self.strategy.shutdown(wait=False, cancel_futures=True)

    def run(self):
        """
        Execute parallel document extraction.

        Processes documents concurrently using the configured strategy.
        Results are collected in completion order and sent to the UI
        as they finish.
        """
        try:
            total_files = len(self.file_paths)
            self.processed_results = []

            if total_files == 0:
                self.ui_queue.put(('processing_finished', []))
                return

            debug_log(f"[PROCESSING WORKER] Starting parallel extraction of {total_files} documents "
                     f"(max_workers={self.strategy.max_workers})")

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
                if self._stop_event.is_set():
                    raise InterruptedError("Processing cancelled")

                filename = Path(file_path).name
                aggregator.update(file_path, f"Extracting {filename}...")

                # Progress callback that checks for cancellation
                def progress_callback(msg, pct=0):
                    if self._stop_event.is_set():
                        raise InterruptedError("Processing stopped by user.")
                    # Update aggregator with detailed message
                    aggregator.update(file_path, msg)

                result = self.extractor.process_document(
                    file_path,
                    progress_callback=progress_callback
                )

                aggregator.complete(file_path)
                return result

            def on_task_complete(task_id: str, result: dict):
                """Callback when a document finishes processing."""
                self.ui_queue.put(('file_processed', result))

            # Create and run the task runner
            self._runner = ParallelTaskRunner(
                strategy=self.strategy,
                on_task_complete=on_task_complete
            )

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
                    debug_log(f"[PROCESSING WORKER] Document failed: {task_result.task_id} - {task_result.error}")

            # Send completion message if not cancelled
            if not self._stop_event.is_set():
                self.ui_queue.put(('processing_finished', self.processed_results))
                self.ui_queue.put(('progress', (100, f"Processed {len(self.processed_results)}/{total_files} documents")))
                debug_log(f"[PROCESSING WORKER] Completed: {len(self.processed_results)}/{total_files} documents")
            else:
                debug_log("[PROCESSING WORKER] Processing cancelled by user.")
                self.ui_queue.put(('error', "Document processing cancelled."))

        except Exception as e:
            debug_log(f"ProcessingWorker encountered a critical error: {e}\n{traceback.format_exc()}")
            self.ui_queue.put(('error', f"Critical document processing error: {str(e)}"))
        finally:
            # Ensure cleanup
            self.strategy.shutdown(wait=False)


class VocabularyWorker(threading.Thread):
    """
    Background worker for vocabulary extraction (Step 2.5).
    Extracts unusual terms from combined document text asynchronously.

    Session 43: Added use_llm parameter for NER+LLM reconciled extraction.
    """
    def __init__(self, combined_text, ui_queue, exclude_list_path=None, medical_terms_path=None, user_exclude_path=None, doc_count=1, use_llm=True):
        super().__init__(daemon=True)
        self.combined_text = combined_text
        self.ui_queue = ui_queue
        self.exclude_list_path = exclude_list_path or "config/legal_exclude.txt"
        self.medical_terms_path = medical_terms_path or "config/medical_terms.txt"
        self.user_exclude_path = user_exclude_path  # User's personal exclusion list
        self.doc_count = doc_count  # Number of documents (for frequency filtering)
        self.use_llm = use_llm  # Whether to use LLM extraction (Session 43)
        self._stop_event = threading.Event()  # Event for graceful stopping

    def stop(self):
        """Signals the worker to stop processing."""
        self._stop_event.set()

    def run(self):
        """Execute vocabulary extraction in background thread."""
        try:
            if self._stop_event.is_set():
                debug_log("[VOCAB WORKER] Stop signal received before starting. Exiting.")
                self.ui_queue.put(('error', "Vocabulary extraction cancelled."))
                return

            # Show text size to set user expectations
            text_len = len(self.combined_text)
            text_kb = text_len // 1024
            self.ui_queue.put(('progress', (30, f"Analyzing {text_kb}KB of text...")))

            # Create extractor with graceful fallback for missing files
            try:
                extractor = VocabularyExtractor(
                    exclude_list_path=self.exclude_list_path,
                    medical_terms_path=self.medical_terms_path,
                    user_exclude_path=self.user_exclude_path
                )
            except FileNotFoundError as e:
                # Graceful fallback: create extractor with empty exclude lists
                debug_log(f"[VOCAB WORKER] Config file missing: {e}. Using empty exclude lists.")
                extractor = VocabularyExtractor(
                    exclude_list_path=None,  # Will use empty list
                    medical_terms_path=None,  # Will use empty list
                    user_exclude_path=self.user_exclude_path  # Still try user list
                )

            # Check for cancellation before heavy processing
            if self._stop_event.is_set():
                debug_log("[VOCAB WORKER] Cancelled before extraction.")
                return

            # Update progress - NLP/LLM processing is the slow part
            if self.use_llm:
                self.ui_queue.put(('progress', (40, "Running NER + LLM extraction (this may take a while)...")))
            else:
                self.ui_queue.put(('progress', (40, "Running NLP analysis (this may take a while)...")))

            # Extract vocabulary - this is the slow part
            # Session 43: Use extract_with_llm for reconciled NER+LLM output
            if self.use_llm:
                # Progress callback for LLM chunk processing
                def llm_progress(current, total):
                    pct = 40 + int((current / total) * 25)  # 40-65% range
                    self.ui_queue.put(('progress', (pct, f"LLM analyzing chunk {current}/{total}...")))

                vocab_data = extractor.extract_with_llm(
                    self.combined_text,
                    doc_count=self.doc_count,
                    include_llm=True,
                    progress_callback=llm_progress
                )
            else:
                # Legacy NER-only extraction
                vocab_data = extractor.extract(self.combined_text, doc_count=self.doc_count)

            # Check for cancellation after extraction
            if self._stop_event.is_set():
                debug_log("[VOCAB WORKER] Cancelled after extraction.")
                return

            term_count = len(vocab_data) if vocab_data else 0
            self.ui_queue.put(('progress', (70, f"Found {term_count} vocabulary terms")))

            # Send results to GUI
            self.ui_queue.put(('vocab_csv_generated', vocab_data))
            debug_log(f"[VOCAB WORKER] Vocabulary extraction completed: {term_count} terms.")

        except Exception as e:
            error_msg = f"Vocabulary extraction failed: {str(e)}"
            debug_log(f"[VOCAB WORKER] {error_msg}\n{traceback.format_exc()}")
            self.ui_queue.put(('error', error_msg))


class QAWorker(threading.Thread):
    """
    Background worker for Q&A document querying.

    Runs default questions against the document using FAISS vector search
    and generates answers via extraction or Ollama.

    Signals sent to ui_queue:
    - ('qa_progress', (current, total, question)) - Question being processed
    - ('qa_result', QAResult) - Single result ready
    - ('qa_complete', list[QAResult]) - All questions processed
    - ('error', str) - Error occurred

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
        answer_mode: str = "extraction",
        questions: list[str] | None = None
    ):
        """
        Initialize Q&A worker.

        Args:
            vector_store_path: Path to FAISS index directory
            embeddings: HuggingFaceEmbeddings model
            ui_queue: Queue for UI communication
            answer_mode: "extraction" or "ollama"
            questions: Custom questions to ask (None = use defaults from YAML)
        """
        super().__init__(daemon=True)
        self.vector_store_path = Path(vector_store_path)
        self.embeddings = embeddings
        self.ui_queue = ui_queue
        self.answer_mode = answer_mode
        self.custom_questions = questions
        self._stop_event = threading.Event()
        self.results: list = []

    def stop(self):
        """Signal the worker to stop processing."""
        debug_log("[QA WORKER] Stop signal received.")
        self._stop_event.set()

    def run(self):
        """Execute Q&A in background thread."""
        try:
            from src.qa import QAOrchestrator

            debug_log(f"[QA WORKER] Starting Q&A with mode: {self.answer_mode}")

            # Initialize orchestrator
            orchestrator = QAOrchestrator(
                vector_store_path=self.vector_store_path,
                embeddings=self.embeddings,
                answer_mode=self.answer_mode
            )

            # Get questions to ask
            if self.custom_questions:
                questions = self.custom_questions
            else:
                questions = orchestrator.get_default_questions()

            total = len(questions)
            if total == 0:
                debug_log("[QA WORKER] No questions to process")
                self.ui_queue.put(('qa_complete', []))
                return

            debug_log(f"[QA WORKER] Processing {total} questions")

            # Process each question
            self.results = []
            for i, question in enumerate(questions):
                if self._stop_event.is_set():
                    debug_log("[QA WORKER] Cancelled during processing")
                    return

                # Report progress
                self.ui_queue.put(('qa_progress', (i, total, question[:50] + "..." if len(question) > 50 else question)))

                # Ask the question
                result = orchestrator._ask_single_question(question, is_followup=False)
                self.results.append(result)

                # Send individual result
                self.ui_queue.put(('qa_result', result))

                debug_log(f"[QA WORKER] Q{i + 1}/{total} complete: {len(result.answer)} chars")

            # Send completion signal with all results
            self.ui_queue.put(('qa_complete', self.results))
            debug_log(f"[QA WORKER] All {total} questions processed successfully")

        except FileNotFoundError as e:
            error_msg = f"Vector store not found: {e}"
            debug_log(f"[QA WORKER] {error_msg}")
            self.ui_queue.put(('error', error_msg))

        except Exception as e:
            error_msg = f"Q&A processing failed: {str(e)}"
            debug_log(f"[QA WORKER] {error_msg}\n{traceback.format_exc()}")
            self.ui_queue.put(('error', error_msg))


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
        while not queue.empty():
            try:
                queue.get_nowait()
            except Empty:
                break

    def start_worker(self):
        """Starts the Ollama AI worker process."""
        if self.is_running and self.process and self.process.is_alive():
            debug_log("[OLLAMA MANAGER] Worker already running.")
            return

        # Ensure queues are empty from previous runs
        self._clear_queue(self.input_queue)
        self._clear_queue(self.output_queue)

        debug_log("[OLLAMA MANAGER] Starting Ollama AI worker process.")
        self.process = multiprocessing.Process(
            target=ollama_generation_worker_process,
            args=(self.input_queue, self.output_queue),
            daemon=True # Daemon process allows main process to exit even if worker is alive
        )
        self.process.start()
        self.is_running = True
        debug_log(f"[OLLAMA MANAGER] Worker process started with PID: {self.process.pid}")

    def stop_worker(self, blocking=False):
        """
        Sends a termination signal and stops the Ollama AI worker process.

        Args:
            blocking: If True, wait for process to terminate. If False (default),
                     terminate immediately without blocking the main thread.
        """
        if self.is_running and self.process and self.process.is_alive():
            debug_log("[OLLAMA MANAGER] Sending TERMINATE signal to worker.")
            try:
                self.input_queue.put_nowait("TERMINATE")  # Non-blocking put

                if blocking:
                    # Wait briefly for graceful shutdown
                    self.process.join(timeout=2)
                else:
                    # Non-blocking: check if it's already dead, but don't wait
                    self.process.join(timeout=0.1)

            except Exception as e:
                debug_log(f"[OLLAMA MANAGER] Error sending terminate signal: {e}")

            # Force terminate if still alive
            if self.process and self.process.is_alive():
                debug_log("[OLLAMA MANAGER] Worker did not terminate gracefully, forcing shutdown.")
                try:
                    self.process.terminate()
                    self.process.join(timeout=0.5)  # Brief wait for terminate
                except Exception as e:
                    debug_log(f"[OLLAMA MANAGER] Error during force terminate: {e}")

            # Clean up queues to prevent memory leaks
            self._clear_queue(self.input_queue)
            self._clear_queue(self.output_queue)

            self.process = None
            self.is_running = False

            # Force garbage collection
            gc.collect()

            debug_log("[OLLAMA MANAGER] Ollama AI worker process stopped, memory cleaned.")
        elif self.is_running:
            debug_log("[OLLAMA MANAGER] Worker process already stopped or not alive.")
            self.process = None
            self.is_running = False

    def send_task(self, task_type: str, payload: dict):
        """Sends a task to the worker process."""
        if not (self.is_running and self.process and self.process.is_alive()):
            self.start_worker() # Ensure worker is running before sending task

        debug_log(f"[OLLAMA MANAGER] Sending task '{task_type}' to worker.")
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


class MultiDocSummaryWorker(threading.Thread):
    """
    Background worker for multi-document hierarchical summarization.

    Uses MultiDocumentOrchestrator to process multiple documents in parallel
    with a map-reduce approach:
    1. Map: Each document summarized via ProgressiveDocumentSummarizer
    2. Reduce: Individual summaries combined into meta-summary

    This worker runs in a background thread to keep the UI responsive
    during potentially long summarization operations.

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
        strategy: ExecutorStrategy = None
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
        super().__init__(daemon=True)
        self.documents = documents
        self.ui_queue = ui_queue
        self.ai_params = ai_params
        self.strategy = strategy or ThreadPoolStrategy(max_workers=PARALLEL_MAX_WORKERS)
        self._stop_event = threading.Event()
        self._orchestrator = None

    def stop(self):
        """Signal the worker to stop processing."""
        debug_log("[MULTI-DOC WORKER] Stop signal received.")
        self._stop_event.set()
        if self._orchestrator:
            self._orchestrator.stop()
        self.strategy.shutdown(wait=False, cancel_futures=True)

    def run(self):
        """Execute multi-document summarization in background thread."""
        try:
            doc_count = len(self.documents)
            debug_log(f"[MULTI-DOC WORKER] Starting summarization of {doc_count} documents")

            # Import here to avoid circular imports
            from src.ai import OllamaModelManager
            from src.prompting import MultiDocPromptAdapter
            from src.summarization import (
                MultiDocumentOrchestrator,
                ProgressiveDocumentSummarizer,
            )

            # Initialize components
            model_manager = OllamaModelManager()

            # Load specified model if provided
            model_name = self.ai_params.get('model_name')
            if model_name:
                model_manager.load_model(model_name)

            # Extract preset_id from ai_params (set by main_window from user selection)
            preset_id = self.ai_params.get('preset_id', 'factual-summary')
            debug_log(f"[MULTI-DOC WORKER] Using preset_id: {preset_id}")

            # Create prompt adapter for thread-through focus areas
            # This adapter extracts focus from the user's template and threads
            # it through all stages of the summarization pipeline
            prompt_adapter = MultiDocPromptAdapter(
                template_manager=model_manager.prompt_template_manager,
                model_manager=model_manager
            )

            doc_summarizer = ProgressiveDocumentSummarizer(
                model_manager,
                prompt_adapter=prompt_adapter,
                preset_id=preset_id
            )

            self._orchestrator = MultiDocumentOrchestrator(
                document_summarizer=doc_summarizer,
                model_manager=model_manager,
                strategy=self.strategy,
                prompt_adapter=prompt_adapter,
                preset_id=preset_id
            )

            # Progress callback to UI
            def on_progress(percent: int, message: str):
                if not self._stop_event.is_set():
                    self.ui_queue.put(('progress', (percent, message)))

            # Get parameters
            summary_length = self.ai_params.get('summary_length', 200)
            meta_length = self.ai_params.get('meta_length', 500)

            # Execute summarization
            result = self._orchestrator.summarize_documents(
                documents=self.documents,
                max_words_per_document=summary_length,
                max_meta_summary_words=meta_length,
                progress_callback=on_progress,
                ui_queue=self.ui_queue
            )

            # Send result to UI
            if not self._stop_event.is_set():
                self.ui_queue.put(('multi_doc_result', result))
                debug_log(f"[MULTI-DOC WORKER] Completed: {result.documents_processed} documents, "
                         f"{result.documents_failed} failed, {result.total_processing_time_seconds:.1f}s")
            else:
                debug_log("[MULTI-DOC WORKER] Processing cancelled by user.")
                self.ui_queue.put(('error', "Multi-document summarization cancelled."))

        except Exception as e:
            error_msg = f"Multi-document summarization failed: {str(e)}"
            debug_log(f"[MULTI-DOC WORKER] {error_msg}\n{traceback.format_exc()}")
            self.ui_queue.put(('error', error_msg))

        finally:
            # Cleanup
            self.strategy.shutdown(wait=False)
            gc.collect()


class ProgressiveExtractionWorker(threading.Thread):
    """
    Progressive three-phase extraction worker (Session 45).

    Implements the vocabulary-first architecture with progressive output:
    - Phase 1 (NER): Fast extraction, returns results in ~5 seconds
    - Phase 2 (Q&A): Builds vector store for Q&A (parallel with Phase 3)
    - Phase 3 (LLM): Slow LLM extraction, updates progressively

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
        """
        super().__init__(daemon=True)
        self.documents = documents
        self.combined_text = combined_text
        self.ui_queue = ui_queue
        self.embeddings = embeddings
        self.exclude_list_path = exclude_list_path
        self.medical_terms_path = medical_terms_path
        self.user_exclude_path = user_exclude_path
        self._stop_event = threading.Event()

    def stop(self):
        """Signal the worker to stop processing."""
        debug_log("[PROGRESSIVE WORKER] Stop signal received.")
        self._stop_event.set()

    def run(self):
        """Execute three-phase progressive extraction."""
        try:
            debug_log("[PROGRESSIVE WORKER] Starting progressive extraction")

            # ===== PHASE 1: NER (Fast - ~5 seconds) =====
            debug_log("[PROGRESSIVE WORKER] Phase 1: NER extraction starting...")
            self.ui_queue.put(('progress', (10, "Phase 1: Running NER extraction...")))

            from src.vocabulary import VocabularyExtractor

            extractor = VocabularyExtractor(
                exclude_list_path=self.exclude_list_path,
                medical_terms_path=self.medical_terms_path,
                user_exclude_path=self.user_exclude_path,
            )

            # NER-only extraction (fast)
            ner_results = extractor.extract_with_llm(
                self.combined_text,
                doc_count=len(self.documents),
                include_llm=False,  # NER only - fast!
            )

            debug_log(f"[PROGRESSIVE WORKER] Phase 1 complete: {len(ner_results)} NER terms")
            self.ui_queue.put(('ner_complete', ner_results))

            if self._stop_event.is_set():
                return

            # ===== PHASE 2: Q&A Indexing (Fast - ~10-30 seconds) =====
            # Run in parallel thread while Phase 3 starts
            qa_thread = threading.Thread(
                target=self._build_vector_store,
                daemon=True
            )
            qa_thread.start()

            # ===== PHASE 3: LLM Enhancement (Slow - minutes) =====
            debug_log("[PROGRESSIVE WORKER] Phase 3: LLM extraction starting...")
            self.ui_queue.put(('progress', (30, "Phase 3: Starting LLM enhancement...")))

            from src.chunking import create_unified_chunker
            from src.extraction import LLMVocabExtractor
            from src.vocabulary.reconciler import VocabularyReconciler

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
            debug_log(f"[PROGRESSIVE WORKER] Created {len(chunks)} unified chunks")

            # Extract with LLM progressively
            llm_extractor = LLMVocabExtractor()

            def llm_progress(current, total):
                if not self._stop_event.is_set():
                    pct = 30 + int((current / total) * 60)  # 30-90% range
                    self.ui_queue.put(('progress', (pct, f"LLM analyzing chunk {current}/{total}...")))
                    self.ui_queue.put(('llm_progress', (current, total)))

            llm_result = llm_extractor.extract_from_unified_chunks(
                chunks,
                progress_callback=llm_progress,
            )

            if self._stop_event.is_set():
                return

            # Reconcile NER + LLM results
            debug_log("[PROGRESSIVE WORKER] Reconciling NER + LLM results...")
            reconciler = VocabularyReconciler()

            # Reconcile people
            ner_people = [c for c in ner_candidates if getattr(c, 'suggested_type', '') == 'Person']
            reconciled_people = reconciler.reconcile_people(ner_people, llm_result.people)

            # Reconcile vocabulary terms
            reconciled_terms = reconciler.reconcile(ner_candidates, llm_result.terms)

            # Convert to unified CSV format
            final_data = reconciler.combined_to_csv_data(reconciled_people, reconciled_terms)

            debug_log(f"[PROGRESSIVE WORKER] Phase 3 complete: {len(final_data)} reconciled terms")
            self.ui_queue.put(('llm_complete', final_data))
            self.ui_queue.put(('progress', (100, f"Complete: {len(final_data)} names & terms found")))

            # Wait for Q&A thread to finish
            qa_thread.join(timeout=60)

        except Exception as e:
            error_msg = f"Progressive extraction failed: {str(e)}"
            debug_log(f"[PROGRESSIVE WORKER] {error_msg}\n{traceback.format_exc()}")
            self.ui_queue.put(('error', error_msg))

    def _build_vector_store(self):
        """Build vector store for Q&A (Phase 2) - runs in parallel thread."""
        try:
            debug_log("[PROGRESSIVE WORKER] Phase 2: Building vector store...")
            self.ui_queue.put(('progress', (20, "Phase 2: Building Q&A index...")))

            from src.chunking import create_unified_chunker
            from src.vector_store import VectorStoreBuilder

            # Lazy-load embeddings if not provided
            if self.embeddings is None:
                debug_log("[PROGRESSIVE WORKER] Loading embeddings model...")
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )

            # Create unified chunks
            chunker = create_unified_chunker()
            chunks = chunker.chunk_text(self.combined_text)

            # Build vector store
            builder = VectorStoreBuilder()
            result = builder.create_from_unified_chunks(
                chunks=chunks,
                embeddings=self.embeddings,
            )

            debug_log(f"[PROGRESSIVE WORKER] Phase 2 complete: {result.chunk_count} chunks indexed")
            self.ui_queue.put(('qa_ready', {
                'vector_store_path': result.persist_dir,
                'embeddings': self.embeddings,
                'chunk_count': result.chunk_count,
            }))

        except Exception as e:
            debug_log(f"[PROGRESSIVE WORKER] Q&A indexing failed: {e}")
            self.ui_queue.put(('qa_error', str(e)))


class BriefingWorker(threading.Thread):
    """
    Background worker for Case Briefing Sheet generation.

    Uses the BriefingOrchestrator to process documents through the
    Map-Reduce pipeline (chunk → extract → aggregate → synthesize → format).

    Signals sent to ui_queue:
    - ('briefing_progress', (phase, current, total, message)) - Phase progress
    - ('briefing_complete', BriefingResult) - Generation complete
    - ('error', str) - Error occurred

    Example:
        worker = BriefingWorker(
            documents=[{"filename": "...", "text": "..."}],
            ui_queue=ui_queue
        )
        worker.start()
    """

    def __init__(
        self,
        documents: list[dict],
        ui_queue: Queue,
    ):
        """
        Initialize briefing worker.

        Args:
            documents: List of document dicts with 'filename' and 'extracted_text'
            ui_queue: Queue for UI communication
        """
        super().__init__(daemon=True)
        self.documents = documents
        self.ui_queue = ui_queue
        self._stop_event = threading.Event()
        self._orchestrator = None

    def stop(self):
        """Signal the worker to stop processing."""
        debug_log("[BRIEFING WORKER] Stop signal received.")
        self._stop_event.set()

    def run(self):
        """Execute briefing generation in background thread."""
        try:
            debug_log(f"[BRIEFING WORKER] Starting briefing for {len(self.documents)} documents")

            # Import briefing components
            from src.briefing import BriefingOrchestrator, BriefingFormatter

            # Initialize orchestrator
            self._orchestrator = BriefingOrchestrator()

            # Check if ready
            if not self._orchestrator.is_ready():
                self.ui_queue.put(('error', "Ollama is not available. Please start Ollama and try again."))
                return

            # Prepare documents for briefing (rename key)
            briefing_docs = []
            for doc in self.documents:
                if doc.get('status') != 'success':
                    continue
                briefing_docs.append({
                    'filename': doc.get('filename', 'unknown'),
                    'text': doc.get('extracted_text', ''),
                })

            if not briefing_docs:
                self.ui_queue.put(('error', "No valid documents to process."))
                return

            debug_log(f"[BRIEFING WORKER] Prepared {len(briefing_docs)} documents for briefing")

            # Progress callback
            def progress_callback(phase: str, current: int, total: int, message: str):
                if not self._stop_event.is_set():
                    self.ui_queue.put(('briefing_progress', (phase, current, total, message)))

            # Run the briefing pipeline
            result = self._orchestrator.generate_briefing(
                documents=briefing_docs,
                progress_callback=progress_callback
            )

            # Check for cancellation
            if self._stop_event.is_set():
                debug_log("[BRIEFING WORKER] Cancelled during processing")
                self.ui_queue.put(('error', "Briefing generation cancelled."))
                return

            # Format the result
            formatter = BriefingFormatter(include_metadata=True)
            formatted = formatter.format(result)

            # Send completion with both result and formatted output
            self.ui_queue.put(('briefing_complete', {
                'result': result,
                'formatted': formatted,
            }))

            debug_log(f"[BRIEFING WORKER] Complete: {result.total_time_seconds:.1f}s, "
                     f"success={result.success}")

        except Exception as e:
            error_msg = f"Briefing generation failed: {str(e)}"
            debug_log(f"[BRIEFING WORKER] {error_msg}\n{traceback.format_exc()}")
            self.ui_queue.put(('error', error_msg))

        finally:
            gc.collect()
