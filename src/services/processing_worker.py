"""
Document Processing Worker.

Background worker for parallel document extraction and normalization.
Uses the Strategy Pattern for execution (ThreadPool in production,
Sequential for testing).
"""

import logging
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
from src.services.base_worker import BaseWorker
from src.services.queue_messages import QueueMessage

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
            from src.services.silly_messages import get_silly_message

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
