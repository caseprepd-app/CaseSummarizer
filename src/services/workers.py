"""
Background Workers Module

Contains threading and multiprocessing workers for document processing:
- ProcessingWorker: Document extraction thread (with parallel processing)
- MultiDocSummaryWorker: Hierarchical summarization worker
- ProgressiveExtractionWorker: NER + LLM + Q&A extraction worker

Performance Optimizations:
- Parallel document extraction via Strategy Pattern
- BaseWorker base class eliminates boilerplate

Lives in services layer to enforce pipeline architecture.
Workers are orchestration, not UI display.
"""

import logging
import threading
from pathlib import Path
from queue import Queue

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
        ocr_allowed: bool = True,
    ):
        """
        Initialize the processing worker.

        Args:
            file_paths: List of document file paths to process.
            ui_queue: Queue for UI communication.
            jurisdiction: Legal jurisdiction for parsing (default "ny").
            strategy: ExecutorStrategy for execution. Defaults to ThreadPoolStrategy
                     with PARALLEL_MAX_WORKERS from config.
            ocr_allowed: Whether OCR is permitted (False when Tesseract missing).
        """
        super().__init__(ui_queue)
        self.file_paths = file_paths
        self.jurisdiction = jurisdiction

        # Dependency injection: use provided strategy or default ThreadPool
        self.strategy = strategy or ThreadPoolStrategy(max_workers=PARALLEL_MAX_WORKERS)

        # RawTextExtractor is thread-safe (stateless after init)
        self.extractor = RawTextExtractor(jurisdiction=self.jurisdiction, ocr_allowed=ocr_allowed)

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
                # Log errors and show in status bar (non-blocking)
                logger.debug("Document failed: %s - %s", task_result.task_id, task_result.error)
                filename = Path(task_result.task_id).name
                self.send_status_error(f"Failed to extract {filename}")

        # Apply preprocessing to all results (removes line numbers, headers, etc.)
        # Store as "preprocessed_text" so downstream consumers don't need to preprocess again
        if self.processed_results:
            from src.core.preprocessing import create_default_pipeline
            from src.services.document_service import DocumentService
            from src.ui.silly_messages import get_silly_message

            self.send_progress(80, "Cleaning up headers and footers...")
            preprocessor = create_default_pipeline(DocumentService._get_preprocessing_settings())
            for result in self.processed_results:
                extracted = result.get("extracted_text", "")
                if extracted:
                    result["preprocessed_text"] = preprocessor.process(extracted)
            self.send_progress(90, get_silly_message())
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

        # Process questions (parallel when beneficial)
        self.results = self._process_questions_parallel(orchestrator, questions, is_default, total)

        # Send completion signal with all results
        self.ui_queue.put(QueueMessage.qa_complete(self.results))
        logger.info("All %s questions processed successfully", total)

    def _process_questions_parallel(
        self, orchestrator, questions: list[str], is_default: bool, total: int
    ) -> list:
        """
        Process Q&A questions, using parallelization when beneficial.

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
        results_lock = threading.Lock()  # Thread-safe dict access

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

                # Store result for ordered return (thread-safe)
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

            # Return results in original order (thread-safe read)
            with results_lock:
                ordered_results = [results_dict.get(i) for i in range(len(questions))]
            final_results = [r for r in ordered_results if r is not None]

            # Log if any questions failed/were cancelled
            failed_count = len(ordered_results) - len(final_results)
            if failed_count > 0:
                logger.debug("%s/%s questions returned no result", failed_count, len(questions))

            return final_results

        finally:
            strategy.shutdown(wait=True)


class MultiDocSummaryWorker(CleanupWorker):
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

        # Resolve enhanced mode and chunk scores
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        enhanced_mode = prefs.is_enhanced_summary_enabled()
        chunk_scores = self.ai_params.get("chunk_scores")

        if enhanced_mode:
            logger.debug("Enhanced summary mode enabled (two-pass extraction)")

        doc_summarizer = ProgressiveDocumentSummarizer(
            model_manager,
            prompt_adapter=prompt_adapter,
            preset_id=preset_id,
            chunk_scores=chunk_scores,
            enhanced_mode=enhanced_mode,
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
            logger.debug("Processing cancelled by user")
            self.ui_queue.put(QueueMessage.error("Multi-document summarization cancelled."))

    def _cleanup(self):
        """Clean up strategy and memory."""
        self.strategy.shutdown(wait=False)
        super()._cleanup()  # Calls gc.collect()


class ProgressiveExtractionWorker(BaseWorker):
    """
    Progressive three-phase extraction worker.

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
        doc_confidence: float = 100.0,
        use_llm: bool = True,
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
            doc_confidence: Aggregate OCR confidence (0-100) for ML feature
            use_llm: Whether to run Phase 3 LLM extraction
        """
        super().__init__(ui_queue)
        self.documents = documents
        self.combined_text = combined_text
        self.embeddings = embeddings
        self.exclude_list_path = exclude_list_path
        self.medical_terms_path = medical_terms_path
        self.user_exclude_path = user_exclude_path
        self.doc_confidence = doc_confidence  # OCR quality for ML feature
        self.use_llm = use_llm  # Whether to run LLM phase
        # Cross-thread state for Q&A phase (Phase 2)
        self._qa_succeeded = threading.Event()
        self._qa_error_lock = threading.Lock()
        self._qa_error_msg: str | None = None

    def execute(self):
        """Execute three-phase progressive extraction."""
        logger.debug("Starting progressive extraction")

        # Signal extraction started (dims feedback buttons)
        self.ui_queue.put(QueueMessage.extraction_started())

        # ===== PHASE 1: Local Algorithms (Progressive) =====
        # Runs BM25 + RAKE first (fast), then NER with chunk progress
        logger.debug("Phase 1: Local algorithm extraction starting...")

        extractor = VocabularyExtractor(
            exclude_list_path=self.exclude_list_path,
            medical_terms_path=self.medical_terms_path,
            user_exclude_path=self.user_exclude_path,
        )

        # Build algo list dynamically from extractor's enabled algorithms
        algo_list = ", ".join(alg.name for alg in extractor.algorithms if alg.enabled)
        self.send_progress(5, "Scanning for names and entities...")
        self.send_progress(10, f"Phase 1: Running local extraction ({algo_list})...")

        # Per-document parallel extraction when 2+ documents.
        # Each doc runs the full pipeline independently, then results are merged
        # with real TermSources so # Docs reflects actual document counts.
        if len(self.documents) > 1:
            doc_list = [
                {
                    "text": d.get("preprocessed_text") or d.get("extracted_text", ""),
                    "doc_id": d.get("filename", f"doc_{i}"),
                    "confidence": d.get("confidence", 100),
                }
                for i, d in enumerate(self.documents)
                if d.get("preprocessed_text") or d.get("extracted_text")
            ]

            def doc_progress(current, total, doc_id):
                pct = 10 + int((current / total) * 20)
                self.send_progress(pct, f"Doc {current}/{total}: extraction complete ({doc_id})")

            ner_results = extractor.extract_documents(
                doc_list,
                use_llm=self.use_llm,
                progress_callback=doc_progress,
            )
        else:
            # Single document: use progressive extraction for fast UX
            def on_partial_complete(partial_vocab):
                """Called when BM25 + RAKE complete (before NER)."""
                logger.debug("Partial results: %s terms", len(partial_vocab))
                self.ui_queue.put(QueueMessage.partial_vocab_complete(partial_vocab))

            def on_ner_progress(chunk_candidates, chunk_num, total_chunks):
                """Called after each NER chunk completes."""
                pct = int((chunk_num / total_chunks) * 100)
                self.send_progress(
                    10 + int((chunk_num / total_chunks) * 20),
                    f"NER: {pct}% complete (chunk {chunk_num}/{total_chunks})...",
                )
                self.ui_queue.put(
                    QueueMessage.ner_progress(chunk_candidates, chunk_num, total_chunks)
                )

            def on_algo_status(message):
                """Called before each algorithm runs."""
                self.send_progress(10, message)

            ner_results = extractor.extract_progressive(
                self.combined_text,
                doc_count=len(self.documents),
                doc_confidence=self.doc_confidence,
                partial_callback=on_partial_complete,
                ner_progress_callback=on_ner_progress,
                status_callback=on_algo_status,
            )

        logger.info("Phase 1 complete: %s terms from local algorithms", len(ner_results))
        self.ui_queue.put(QueueMessage.ner_complete(ner_results))

        # Signal extraction complete (re-enables feedback buttons)
        self.ui_queue.put(QueueMessage.extraction_complete())

        self.check_cancelled()

        # ===== PHASE 2: Q&A Indexing (Fast - ~10-30 seconds) =====
        # Run in parallel thread while Phase 3 starts
        # Reset Q&A phase state for this execution
        self._qa_succeeded.clear()
        with self._qa_error_lock:
            self._qa_error_msg = None
        qa_thread = threading.Thread(target=self._build_vector_store, daemon=False)
        qa_thread.start()

        # ===== PHASE 3: LLM Enhancement (Slow - minutes) =====
        # Only runs if enabled (GPU auto-detect or user override)
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

            # CHUNKING SITE 2 of 3: LLM vocab extraction (combined text, no source attribution)
            #
            # Unlike Q&A (Site 1) which chunks per-document, this chunks all documents
            # concatenated together. The LLM extracts terms/names from each chunk.
            # Coreference resolution runs on the full combined text inside chunk_text(),
            # helping the LLM understand pronoun references across document boundaries.
            #
            # Note: NER (Phase 1) does NOT use the chunker -- it runs directly on
            # preprocessed_text with its own internal chunking, so NER sees original
            # pronouns and identifies entities without coreference interference.
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
            from src.core.vocabulary.reconciler import VocabularyReconciler
            from src.ui.silly_messages import get_silly_message

            self.send_progress(92, get_silly_message())
            logger.debug("Deduplicating NER + LLM results...")
            reconciler = VocabularyReconciler()

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
            # Skip LLM phase - NER results are already sent in Phase 1
            logger.debug("Phase 3: Skipped (LLM disabled by user preference or no GPU)")
            self.send_progress(90, "Phase 3: Skipped (LLM disabled)")
            # Signal LLM complete with empty list - UI will show NER-only results
            self.ui_queue.put(QueueMessage.llm_complete([]))
            self.send_progress(100, f"Complete: {len(ner_results)} terms (local algorithms only)")

        # Wait for Q&A thread with periodic status updates (CPU embeddings can take 5+ minutes)
        # Using timeout loop instead of indefinite join to keep UI responsive with status updates
        QA_JOIN_TIMEOUT_SECONDS = 30  # Check every 30 seconds
        QA_MAX_WAIT_MINUTES = 900  # Give up after 15 hours
        wait_count = 0
        max_waits = (QA_MAX_WAIT_MINUTES * 60) // QA_JOIN_TIMEOUT_SECONDS

        while qa_thread.is_alive() and wait_count < max_waits:
            wait_minutes = (wait_count * QA_JOIN_TIMEOUT_SECONDS) // 60
            if wait_count == 0:
                logger.debug("Vocabulary done, waiting for Q&A index to finish...")
                self.send_progress(100, "Vocabulary complete. Building Q&A search index...")
            else:
                self.send_progress(
                    100, f"Q&A index still building ({wait_minutes}m elapsed, please wait)..."
                )
            qa_thread.join(timeout=QA_JOIN_TIMEOUT_SECONDS)
            wait_count += 1

        if qa_thread.is_alive():
            logger.warning(
                "Q&A thread still running after %d minutes, continuing without waiting",
                QA_MAX_WAIT_MINUTES,
            )
            self.send_progress(100, "Q&A index taking too long, proceeding without it...")
            self.ui_queue.put(
                QueueMessage.status_error(
                    "Q&A search index timed out. Q&A tab may not work for this session."
                )
            )
            self.ui_queue.put(QueueMessage.qa_error("Q&A index timed out"))
        elif not self._qa_succeeded.is_set():
            # Thread exited but didn't signal success -- it crashed
            with self._qa_error_lock:
                error_detail = self._qa_error_msg or "unknown error"
            logger.error("Q&A thread failed: %s", error_detail)
            self.send_progress(100, "Q&A indexing failed, vocabulary results still available.")
            self.ui_queue.put(QueueMessage.qa_error(f"Q&A thread failed: {error_detail}"))

    def _build_vector_store(self):
        """Build vector store for Q&A (Phase 2) - runs in parallel thread."""
        try:
            logger.debug("Phase 2: Building vector store...")
            self.ui_queue.put(QueueMessage.progress(20, "Phase 2: Building Q&A index..."))

            from src.core.chunking import create_unified_chunker
            from src.core.vector_store import VectorStoreBuilder

            # Lazy-load embeddings if not provided (shared instance, GPU-aware)
            if self.embeddings is None:
                logger.debug("Loading embeddings model...")
                self.ui_queue.put(
                    QueueMessage.progress(
                        22, "Loading AI language model (first time may be slow)..."
                    )
                )
                from src.core.retrieval.algorithms.faiss_semantic import get_embeddings_model

                self.embeddings = get_embeddings_model()

            # CHUNKING SITE 1 of 3: Q&A vector store (per-document, with source attribution)
            #
            # Chunks each document separately so each chunk retains its source_file for
            # citation in Q&A answers. Coreference resolution (pronoun -> name replacement)
            # runs inside chunk_text() on the full document text BEFORE splitting, so each
            # chunk is self-contained (e.g., "He testified" becomes "Dr. Smith testified").
            #
            # This same text gets chunked again by the summarizer (Site 3) if summary is
            # enabled. That duplication is intentional and acceptable -- summary is rarely
            # enabled, and the bottleneck is LLM calls (minutes), not chunking (seconds).
            # Sharing chunks across phases would require cross-thread plumbing for minimal gain.
            self.ui_queue.put(
                QueueMessage.progress(24, "Splitting documents into searchable passages...")
            )
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
            total_chunks = len(all_chunks)
            self.ui_queue.put(
                QueueMessage.progress(26, f"Building search index (0/{total_chunks} passages)...")
            )

            def on_index_progress(current, total):
                self.ui_queue.put(
                    QueueMessage.progress(
                        26, f"Building search index ({current}/{total} passages)..."
                    )
                )

            builder = VectorStoreBuilder()
            result = builder.create_from_unified_chunks(
                chunks=all_chunks,
                embeddings=self.embeddings,
                progress_callback=on_index_progress,
            )

            logger.info("Phase 2 complete: %s chunks indexed", result.chunk_count)
            self.ui_queue.put(
                QueueMessage.progress(
                    28, f"Search index ready! ({result.chunk_count} passages indexed)"
                )
            )
            # Run redundancy detection on embeddings for summarization
            chunk_scores = None
            if result.chunk_embeddings:
                try:
                    from src.core.utils.chunk_scoring import detect_redundant_chunks

                    chunk_scores = detect_redundant_chunks(result.chunk_embeddings)
                    skip_count = sum(1 for s in chunk_scores.skip if s)
                    logger.debug(
                        "Redundancy detection: %d/%d chunks flagged",
                        skip_count,
                        len(result.chunk_embeddings),
                    )
                except Exception as e:
                    logger.warning("Redundancy detection failed: %s", e)

            # Guard: skip sending messages if extraction was cancelled
            if self.is_stopped:
                logger.debug("Phase 2: Cancelled — discarding qa_ready/trigger_default_qa")
                return

            self.ui_queue.put(
                QueueMessage.qa_ready(
                    vector_store_path=result.persist_dir,
                    embeddings=self.embeddings,
                    chunk_count=result.chunk_count,
                    chunk_scores=chunk_scores,
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

            # Signal success so the main thread knows we didn't crash
            self._qa_succeeded.set()

        except Exception as e:
            logger.error("Q&A indexing failed: %s", e, exc_info=True)
            with self._qa_error_lock:
                self._qa_error_msg = str(e)
            self.ui_queue.put(QueueMessage.status_error("Q&A indexing failed"))
            self.ui_queue.put(QueueMessage.qa_error(str(e)))
