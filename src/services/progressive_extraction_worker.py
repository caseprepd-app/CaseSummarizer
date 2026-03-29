"""
Progressive Extraction Worker.

Two-phase extraction worker that runs vocabulary extraction (Phase 1)
and search indexing (Phase 2) with progressive UI updates.
"""

import logging
import random
import threading
from queue import Queue

from src.core.vocabulary import VocabularyExtractor
from src.services.base_worker import BaseWorker
from src.services.queue_messages import QueueMessage

logger = logging.getLogger(__name__)


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
        self._index_progress: tuple[int, int] = (0, 0)  # (current, total) chunks

    def execute(self):
        """Execute two-phase progressive extraction."""
        logger.debug("Starting progressive extraction")

        # Signal extraction started (dims feedback buttons)
        self.ui_queue.put(QueueMessage.extraction_started())

        # ===== PHASE 1: Local Algorithms (Progressive) =====
        self._run_vocabulary_extraction()

        self.check_cancelled()

        # ===== PHASE 2: Search Indexing (CPU-only can take 15+ minutes) =====
        self._run_search_indexing()

    def _run_vocabulary_extraction(self):
        """
        Phase 1: Run local vocabulary extraction algorithms.

        Runs BM25 + RAKE first (fast), then NER with chunk progress.
        Supports both single-doc progressive mode and multi-doc parallel mode.
        """
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
        if len(self.documents) > 1:
            ner_results = self._extract_multi_document(extractor)
        else:
            ner_results = self._extract_single_document(extractor)

        ner_results, filtered_terms = ner_results
        skipped = getattr(extractor, "skipped_algorithms", [])
        logger.info(
            "Phase 1 complete: %s terms from local algorithms, %s filtered, %s skipped",
            len(ner_results),
            len(filtered_terms),
            skipped or "none",
        )
        self.ui_queue.put(QueueMessage.ner_complete(ner_results, filtered_terms, skipped))

        # Signal extraction complete (re-enables feedback buttons)
        self.ui_queue.put(QueueMessage.extraction_complete())
        self.send_progress(90, f"Complete: {len(ner_results)} terms extracted")

    def _extract_multi_document(self, extractor):
        """
        Extract vocabulary from multiple documents in parallel.

        Each doc runs the full pipeline independently, then results are merged
        with real TermSources so # Docs reflects actual document counts.

        Args:
            extractor: VocabularyExtractor instance

        Returns:
            Tuple of (ner_results, filtered_terms)
        """
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
            """Report per-document progress."""
            pct = 10 + int((current / max(total, 1)) * 20)
            self.send_progress(pct, f"Doc {current}/{total}: extraction complete ({doc_id})")

        return extractor.extract_documents(doc_list, progress_callback=doc_progress)

    def _extract_single_document(self, extractor):
        """
        Extract vocabulary from a single document with progressive UI.

        Uses progressive extraction for fast UX: BM25 + RAKE results display
        immediately, then NER runs with chunk-by-chunk progress updates.

        Args:
            extractor: VocabularyExtractor instance

        Returns:
            Tuple of (ner_results, filtered_terms)
        """
        from src.config import DEBUG_MODE

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
            self.ui_queue.put(QueueMessage.ner_progress(chunk_candidates, chunk_num, total_chunks))

        def on_algo_status(message):
            """Called before each algorithm runs."""
            self.send_progress(10, message)

        return extractor.extract_progressive(
            self.combined_text,
            doc_count=len(self.documents),
            doc_confidence=self.doc_confidence,
            partial_callback=on_partial_complete,
            ner_progress_callback=on_ner_progress,
            status_callback=on_algo_status,
        )

    def _run_search_indexing(self):
        """
        Phase 2: Build vector store for semantic search.

        Runs in a parallel thread with periodic status updates.
        CPU-only embedding can take 15+ minutes on large documents.
        """
        self._search_succeeded.clear()
        with self._search_error_lock:
            self._search_error_msg = None
        semantic_thread = threading.Thread(target=self._build_vector_store, daemon=False)
        semantic_thread.start()

        self._wait_for_search_indexing(semantic_thread)

    def _wait_for_search_indexing(self, semantic_thread):
        """
        Wait for search index thread with periodic status updates.

        Uses timeout loop instead of indefinite join to keep UI responsive.

        Args:
            semantic_thread: Thread running _build_vector_store
        """
        SEMANTIC_JOIN_TIMEOUT_SECONDS = 30  # Check every 30 seconds
        SEMANTIC_MAX_WAIT_MINUTES = 60  # Give up after 1 hour
        wait_count = 0
        max_waits = (SEMANTIC_MAX_WAIT_MINUTES * 60) // SEMANTIC_JOIN_TIMEOUT_SECONDS

        while semantic_thread.is_alive() and wait_count < max_waits and not self.is_stopped:
            wait_seconds = wait_count * SEMANTIC_JOIN_TIMEOUT_SECONDS
            wait_minutes = wait_seconds // 60
            current, total = self._index_progress
            if wait_count == 0:
                logger.debug("Vocabulary done, waiting for semantic search index to finish...")
                self.send_progress(100, "Vocabulary complete. Building semantic search index...")
            elif total > 0:
                pct = int((current / max(total, 1)) * 100)
                elapsed = f"{wait_minutes}m" if wait_minutes >= 1 else f"{wait_seconds}s"
                self.send_progress(
                    100,
                    f"Indexing passages ({current}/{total}, {pct}%) — {elapsed} elapsed...",
                )
            else:
                elapsed = f"{wait_minutes}m" if wait_minutes >= 1 else f"{wait_seconds}s"
                self.send_progress(
                    100,
                    f"Preparing search index ({elapsed} elapsed)...",
                )
            semantic_thread.join(timeout=SEMANTIC_JOIN_TIMEOUT_SECONDS)
            wait_count += 1

        if semantic_thread.is_alive() and self.is_stopped:
            logger.info("Cancellation requested, waiting up to 5s for semantic thread to finish...")
            semantic_thread.join(timeout=5)

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
            with self._search_error_lock:
                error_detail = self._search_error_msg or "unknown error"
            logger.error("Search index thread failed: %s", error_detail)
            self.send_progress(100, "Search indexing failed, vocabulary results still available.")

    def _get_search_title_remover(self):
        """
        Return a TitlePageRemover for search chunking if the setting calls for it.

        For "vocab_only" mode, title pages are excluded from search/key excerpts
        but were kept in preprocessed_text (used for vocabulary extraction).

        Returns:
            TitlePageRemover instance if vocab_only mode, else None
        """
        try:
            from src.user_preferences import get_user_preferences

            prefs = get_user_preferences()
            handling = prefs.get("title_page_handling", "vocab_only")
        except Exception:
            handling = "vocab_only"

        if handling == "vocab_only":
            from src.core.preprocessing.title_page_remover import TitlePageRemover

            return TitlePageRemover()
        return None

    def _chunk_documents_for_search(self, chunker, title_page_remover):
        """
        Chunk all documents for search indexing, optionally removing title pages.

        Args:
            chunker: UnifiedChunker instance
            title_page_remover: TitlePageRemover or None

        Returns:
            List of chunks across all documents
        """
        all_chunks = []
        for doc in self.documents:
            filename = doc.get("filename", "unknown")
            # Prefer preprocessed_text (already cleaned) over extracted_text (raw)
            text = doc.get("preprocessed_text") or doc.get("extracted_text", "")
            if not text.strip():
                continue
            if title_page_remover is not None:
                text = title_page_remover.process(text).text
            doc_chunks = chunker.chunk_text(text, source_file=filename)
            all_chunks.extend(doc_chunks)
        return all_chunks

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

            # Chunk each document separately for source attribution in search results
            self.ui_queue.put(
                QueueMessage.progress(24, "Splitting documents into searchable passages...")
            )
            chunker = create_unified_chunker()
            title_page_remover = self._get_search_title_remover()
            all_chunks = self._chunk_documents_for_search(chunker, title_page_remover)

            if self.is_stopped:
                logger.debug("Phase 2: Cancelled before vector store build")
                return

            # Build vector store
            total_chunks = len(all_chunks)
            self.ui_queue.put(
                QueueMessage.progress(26, f"Indexing passages (0/{total_chunks}, 0%)...")
            )

            def on_index_progress(current, total):
                """Report chunk embedding progress to UI and parent thread."""
                from src.services.silly_messages import get_silly_message

                self._index_progress = (current, total)
                pct = int((current / max(total, 1)) * 100)
                msg = f"Indexing passages ({current}/{total}, {pct}%)..."
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
