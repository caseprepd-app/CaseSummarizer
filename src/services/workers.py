"""
Background Workers Module

Contains threading and multiprocessing workers for document processing:
- ProcessingWorker: Document extraction thread (with parallel processing)
- ProgressiveExtractionWorker: NER + semantic search extraction worker

Performance Optimizations:
- Parallel document extraction via Strategy Pattern
- BaseWorker base class eliminates boilerplate

Lives in services layer to enforce pipeline architecture.
Workers are orchestration, not UI display.
"""

import logging
import random
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
from src.ui.base_worker import BaseWorker
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
                logger.warning("Document failed: %s - %s", task_result.task_id, task_result.error)
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


class SemanticWorker(BaseWorker):
    """
    Background worker for semantic search document querying.

    Runs default questions against the document using FAISS vector search
    and retrieval-based extraction.

    Signals sent to ui_queue:
    - ('semantic_progress', (current, total, question)) - Question being processed
    - ('semantic_result', SemanticResult) - Single result ready
    - ('semantic_complete', list[SemanticResult]) - All questions processed
    - ('error', str) - Error occurred

    Example:
        worker = SemanticWorker(
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
        questions: list[str] | None = None,
        use_default_questions: bool = False,
    ):
        """
        Initialize semantic search worker.

        Args:
            vector_store_path: Path to FAISS index directory
            embeddings: HuggingFaceEmbeddings model
            ui_queue: Queue for UI communication
            answer_mode: Ignored (kept for backward compat). Always uses extraction.
            questions: Custom questions to ask (None = use defaults from YAML)
            use_default_questions: If True, load questions from semantic_default_questions.txt
        """
        super().__init__(ui_queue)
        self.vector_store_path = Path(vector_store_path)
        self.embeddings = embeddings
        self.answer_mode = "extraction"
        self.custom_questions = questions
        self.use_default_questions = use_default_questions
        self.results: list = []

    def execute(self):
        """Execute semantic search in background thread."""
        from src.core.semantic import SemanticOrchestrator

        logger.debug("Starting semantic search")

        # Initialize orchestrator
        orchestrator = SemanticOrchestrator(
            vector_store_path=self.vector_store_path,
            embeddings=self.embeddings,
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
            self.ui_queue.put(QueueMessage.semantic_complete([]))
            return

        logger.debug("Processing %s questions", total)

        # Process questions (parallel when beneficial)
        self.results = self._process_questions_parallel(orchestrator, questions, is_default, total)

        # Send completion signal with all results
        self.ui_queue.put(QueueMessage.semantic_complete(self.results))
        logger.info("All %s questions processed successfully", total)

    def _process_questions_parallel(
        self, orchestrator, questions: list[str], is_default: bool, total: int
    ) -> list:
        """
        Process search questions sequentially.

        Extraction mode is fast enough that parallelization is unnecessary.
        Results are streamed to UI as they complete via semantic_result messages.

        Args:
            orchestrator: SemanticOrchestrator instance
            questions: List of questions to ask
            is_default: Whether these are default questions
            total: Total number of questions for progress tracking

        Returns:
            List of SemanticResult objects in original question order
        """
        logger.debug("Processing %s question(s) sequentially", len(questions))
        results = []
        for i, question in enumerate(questions):
            self.check_cancelled()

            # Report progress
            truncated_q = question[:50] + "..." if len(question) > 50 else question
            self.ui_queue.put(QueueMessage.semantic_progress(i, total, truncated_q))

            result = orchestrator._ask_single_question(
                question, is_followup=False, is_default=is_default
            )
            results.append(result)
            self.ui_queue.put(QueueMessage.semantic_result(result))
            logger.debug("Q%s/%s complete: %s chars", i + 1, total, len(result.answer))

        return results


class ProgressiveExtractionWorker(BaseWorker):
    """
    Progressive two-phase extraction worker.

    Implements the vocabulary-first architecture with progressive output:
    - Phase 1 (NER): Fast extraction via local algorithms (NER, RAKE, BM25)
    - Phase 2 (Search): Builds vector store for semantic search (parallel with Phase 1)

    Signals sent to ui_queue:
    - ('ner_complete', vocab_data) - Phase 1 complete, display immediately
    - ('semantic_ready', vector_store_path) - Phase 2 complete, enable semantic search
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
        """
        super().__init__(ui_queue)
        self.documents = documents
        self.combined_text = combined_text
        self.embeddings = embeddings
        self.exclude_list_path = exclude_list_path
        self.medical_terms_path = medical_terms_path
        self.user_exclude_path = user_exclude_path
        self.doc_confidence = doc_confidence  # OCR quality for ML feature
        # Cross-thread state for search indexing phase (Phase 2)
        self._search_succeeded = threading.Event()
        self._search_error_lock = threading.Lock()
        self._search_error_msg: str | None = None

    def execute(self):
        """Execute two-phase progressive extraction."""
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
        from src.config import DEBUG_MODE

        if DEBUG_MODE:
            self.send_progress(5, "Scanning for names and entities...")
            self.send_progress(10, f"Phase 1: Running local extraction ({algo_list})...")
        else:
            self.send_progress(5, "Scanning your documents...")
            self.send_progress(10, "Extracting vocabulary and names...")

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
                pct = int((chunk_num / max(total_chunks, 1)) * 100)
                if DEBUG_MODE:
                    msg = f"NER: {pct}% complete (chunk {chunk_num}/{total_chunks})..."
                else:
                    msg = f"Extracting names... {pct}% complete"
                self.send_progress(10 + int((chunk_num / max(total_chunks, 1)) * 20), msg)
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

        ner_results, filtered_terms = ner_results
        logger.info(
            "Phase 1 complete: %s terms from local algorithms, %s filtered",
            len(ner_results),
            len(filtered_terms),
        )
        self.ui_queue.put(QueueMessage.ner_complete(ner_results, filtered_terms))

        # Signal extraction complete (re-enables feedback buttons)
        self.ui_queue.put(QueueMessage.extraction_complete())

        self.check_cancelled()

        self.send_progress(90, f"Complete: {len(ner_results)} terms extracted")

        # ===== PHASE 2: Search Indexing (CPU-only can take 15+ minutes) =====
        # Run in parallel thread
        # Reset search indexing phase state for this execution
        self._search_succeeded.clear()
        with self._search_error_lock:
            self._search_error_msg = None
        semantic_thread = threading.Thread(target=self._build_vector_store, daemon=False)
        semantic_thread.start()

        # Wait for search index thread with periodic status updates (CPU embeddings can take 5+ minutes)
        # Using timeout loop instead of indefinite join to keep UI responsive with status updates
        SEMANTIC_JOIN_TIMEOUT_SECONDS = 30  # Check every 30 seconds
        SEMANTIC_MAX_WAIT_MINUTES = 60  # Give up after 1 hour
        wait_count = 0
        max_waits = (SEMANTIC_MAX_WAIT_MINUTES * 60) // SEMANTIC_JOIN_TIMEOUT_SECONDS

        while semantic_thread.is_alive() and wait_count < max_waits and not self.is_stopped:
            wait_minutes = (wait_count * SEMANTIC_JOIN_TIMEOUT_SECONDS) // 60
            if wait_count == 0:
                logger.debug("Vocabulary done, waiting for semantic search index to finish...")
                self.send_progress(100, "Vocabulary complete. Building semantic search index...")
            else:
                self.send_progress(
                    100,
                    f"Semantic search index still building ({wait_minutes}m elapsed)...",
                )
            semantic_thread.join(timeout=SEMANTIC_JOIN_TIMEOUT_SECONDS)
            wait_count += 1

        if semantic_thread.is_alive():
            logger.warning(
                "Semantic search thread still running after %d minutes, continuing without waiting",
                SEMANTIC_MAX_WAIT_MINUTES,
            )
            self.send_progress(100, "Search index taking too long, proceeding without it...")
            self.ui_queue.put(
                QueueMessage.status_error(
                    "Search index timed out. Search tab may not work for this session."
                )
            )
            self.ui_queue.put(QueueMessage.semantic_error("Search index timed out"))
        elif not self._search_succeeded.is_set():
            # Thread exited but didn't signal success -- it crashed
            with self._search_error_lock:
                error_detail = self._search_error_msg or "unknown error"
            logger.error("Search index thread failed: %s", error_detail)
            self.send_progress(100, "Search indexing failed, vocabulary results still available.")
            self.ui_queue.put(
                QueueMessage.semantic_error(f"Semantic search thread failed: {error_detail}")
            )

    def _build_vector_store(self):
        """Build vector store for search (Phase 2) - runs in parallel thread."""
        try:
            logger.debug("Phase 2: Building vector store...")
            self.ui_queue.put(
                QueueMessage.progress(20, "Phase 2: Building semantic search index...")
            )

            from src.core.chunking import create_unified_chunker
            from src.core.vector_store import VectorStoreBuilder

            # Early exit if cancelled before we start heavy work
            if self.is_stopped:
                logger.debug("Phase 2: Cancelled before start")
                return

            # Lazy-load embeddings if not provided (shared instance, GPU-aware)
            if self.embeddings is None:
                logger.debug("Loading embeddings model...")
                self.ui_queue.put(
                    QueueMessage.progress(
                        22, "Loading semantic search model (first time may be slow)..."
                    )
                )
                from src.core.retrieval.algorithms.faiss_semantic import get_embeddings_model

                self.embeddings = get_embeddings_model()

            # CHUNKING SITE 1 of 2: Search vector store (per-document, with source attribution)
            #
            # Chunks each document separately so each chunk retains its source_file for
            # citation in search results. Coreference resolution (pronoun -> name replacement)
            # runs inside chunk_text() on the full document text BEFORE splitting, so each
            # chunk is self-contained (e.g., "He testified" becomes "Dr. Smith testified").
            #
            # Single chunking pass: chunks are reused for search indexing, vocabulary
            # extraction, and key excerpts (via FAISS embeddings).
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

            if self.is_stopped:
                logger.debug("Phase 2: Cancelled before vector store build")
                return

            # Build vector store
            total_chunks = len(all_chunks)
            self.ui_queue.put(
                QueueMessage.progress(26, f"Building search index (0/{total_chunks} passages)...")
            )

            def on_index_progress(current, total):
                from src.ui.silly_messages import get_silly_message

                msg = f"Building search index ({current}/{total} passages)..."
                # Higher silly message rate for incremental progress (15% vs 4%)
                if random.randint(1, 7) == 1:
                    msg = get_silly_message()
                self.ui_queue.put(QueueMessage.progress(26, msg))

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
                    logger.warning("Redundancy detection failed: %s", e, exc_info=True)

            # Guard: skip sending messages if extraction was cancelled
            if self.is_stopped:
                logger.debug(
                    "Phase 2: Cancelled — discarding semantic_ready/trigger_default_semantic"
                )
                return

            # Extract chunk texts and metadata for key excerpts (reuses embeddings)
            chunk_texts = [c.text for c in all_chunks]
            chunk_metadata = [
                {"source_file": c.source_file, "chunk_num": c.chunk_num} for c in all_chunks
            ]

            self.ui_queue.put(
                QueueMessage.semantic_ready(
                    vector_store_path=result.persist_dir,
                    embeddings=self.embeddings,
                    chunk_count=result.chunk_count,
                    chunk_scores=chunk_scores,
                    chunk_texts=chunk_texts,
                    chunk_metadata=chunk_metadata,
                    chunk_embeddings=result.chunk_embeddings,
                )
            )

            # Trigger default questions if enabled
            logger.debug("Triggering default semantic search check")
            self.ui_queue.put(
                QueueMessage.trigger_default_semantic(
                    vector_store_path=result.persist_dir,
                    embeddings=self.embeddings,
                )
            )

            # Signal success so the main thread knows we didn't crash
            self._search_succeeded.set()

        except Exception as e:
            logger.error("Semantic search indexing failed: %s", e, exc_info=True)
            with self._search_error_lock:
                self._search_error_msg = str(e)
            self.ui_queue.put(QueueMessage.status_error("Semantic search indexing failed"))
            self.ui_queue.put(QueueMessage.semantic_error(str(e)))
