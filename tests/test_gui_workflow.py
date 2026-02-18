"""
GUI Workflow Integration Tests

Tests the full document processing workflow that simulates user actions:
1. Load sample PDFs
2. Preprocess documents (extraction)
3. Click "Process Documents" (vocabulary + Q&A)
4. Verify all phases complete

Two testing modes:
- Headless: Tests workers directly without GUI (fast, reliable for debugging)
- GUI Simulation: Tests with actual MainWindow (slower, full integration)

Usage:
    # Run all tests
    pytest tests/test_gui_workflow.py -v

    # Run just headless tests (faster)
    pytest tests/test_gui_workflow.py -v -k "headless"

    # Run with timeout debugging
    pytest tests/test_gui_workflow.py -v -s --timeout=120
"""

import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logger = logging.getLogger(__name__)
from src.ui.queue_messages import MessageType, QueueMessage

# =============================================================================
# Test Configuration
# =============================================================================

SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "sampleDocuments"
SAMPLE_PDFS = list(SAMPLE_DOCS_DIR.glob("*.pdf"))

# Timeouts (seconds)
PREPROCESSING_TIMEOUT = 120  # 2 minutes per file
PHASE1_TIMEOUT = 600  # 10 minutes for NER/RAKE/GLiNER/BM25 on CPU
PHASE2_TIMEOUT = 600  # 10 minutes for Q&A indexing (CPU embedding can be very slow)
PHASE3_TIMEOUT = 600  # 10 minutes for LLM (if enabled)
TOTAL_TIMEOUT = 900  # 15 minutes total


@dataclass
class WorkflowProgress:
    """Tracks progress through the workflow phases."""

    files_processed: int = 0
    preprocessing_complete: bool = False
    ner_complete: bool = False
    qa_ready: bool = False
    llm_complete: bool = False
    errors: list = field(default_factory=list)
    messages: list = field(default_factory=list)

    def add_message(self, msg_type: str, data):
        """Record a message for debugging."""
        self.messages.append((time.time(), msg_type, str(data)[:200]))

    def is_complete(self, require_llm: bool = False) -> bool:
        """Check if workflow completed all expected phases."""
        base_complete = self.preprocessing_complete and self.ner_complete and self.qa_ready
        if require_llm:
            return base_complete and self.llm_complete
        return base_complete


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_pdfs():
    """Get list of sample PDF paths."""
    pdfs = list(SAMPLE_DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"No sample PDFs found in {SAMPLE_DOCS_DIR}")
    return [str(p) for p in pdfs]


@pytest.fixture
def ui_queue():
    """Create a fresh queue for worker communication."""
    return Queue()


@pytest.fixture
def progress_tracker():
    """Create a progress tracker for monitoring workflow."""
    return WorkflowProgress()


# =============================================================================
# Queue Message Collector
# =============================================================================


class QueueCollector:
    """
    Collects and categorizes messages from worker threads.

    Provides blocking wait methods with timeout to detect stuck workers.
    """

    def __init__(self, queue: Queue, progress: WorkflowProgress):
        self.queue = queue
        self.progress = progress
        self._stop_event = threading.Event()

    def stop(self):
        """Signal collector to stop."""
        self._stop_event.set()

    def collect_until(
        self, condition: Callable[[], bool], timeout: float, poll_interval: float = 0.1
    ) -> bool:
        """
        Collect messages until condition is met or timeout.

        Args:
            condition: Callable that returns True when done
            timeout: Maximum seconds to wait
            poll_interval: How often to check condition

        Returns:
            True if condition met, False if timeout
        """
        start_time = time.time()

        while not self._stop_event.is_set():
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.debug(f"[COLLECTOR] Timeout after {elapsed:.1f}s")
                return False

            # Check condition
            if condition():
                logger.debug(f"[COLLECTOR] Condition met after {elapsed:.1f}s")
                return True

            # Collect any pending messages
            self._drain_queue()

            # Brief pause
            time.sleep(poll_interval)

        return False

    def _drain_queue(self):
        """Process all pending messages in the queue."""
        while True:
            try:
                msg = self.queue.get_nowait()
                self._handle_message(msg)
            except Empty:
                break

    def _handle_message(self, msg):
        """Handle a single queue message."""
        if not isinstance(msg, tuple) or len(msg) != 2:
            logger.debug(f"[COLLECTOR] Invalid message format: {msg}")
            return

        msg_type, data = msg
        self.progress.add_message(msg_type, data)

        # Update progress based on message type
        if msg_type == MessageType.FILE_PROCESSED:
            self.progress.files_processed += 1
            logger.debug(f"[COLLECTOR] File processed: {self.progress.files_processed}")

        elif msg_type == MessageType.PROCESSING_FINISHED:
            self.progress.preprocessing_complete = True
            logger.debug(f"[COLLECTOR] Preprocessing complete: {len(data)} files")

        elif msg_type == MessageType.NER_COMPLETE:
            self.progress.ner_complete = True
            term_count = len(data) if data else 0
            logger.debug(f"[COLLECTOR] NER complete: {term_count} terms")

        elif msg_type == MessageType.QA_READY:
            self.progress.qa_ready = True
            chunk_count = data.get("chunk_count", 0) if isinstance(data, dict) else 0
            logger.debug(f"[COLLECTOR] Q&A ready: {chunk_count} chunks")

        elif msg_type == MessageType.LLM_COMPLETE:
            self.progress.llm_complete = True
            term_count = len(data) if data else 0
            logger.debug(f"[COLLECTOR] LLM complete: {term_count} terms")

        elif msg_type == MessageType.ERROR:
            self.progress.errors.append(str(data))
            logger.debug(f"[COLLECTOR] Error: {data}")

        elif msg_type == MessageType.PROGRESS:
            pct, msg = data if isinstance(data, tuple) else (0, str(data))
            logger.debug(f"[COLLECTOR] Progress {pct}%: {msg}")


# =============================================================================
# HEADLESS TESTS - Test workers directly without GUI
# =============================================================================


class TestHeadlessPreprocessing:
    """Test document preprocessing (extraction) without GUI."""

    def test_preprocessing_worker_completes(self, sample_pdfs, ui_queue, progress_tracker):
        """Test that ProcessingWorker extracts all sample PDFs."""
        from src.services.workers import ProcessingWorker

        # Start worker
        worker = ProcessingWorker(file_paths=sample_pdfs, ui_queue=ui_queue)
        worker.start()

        # Collect messages until preprocessing complete
        collector = QueueCollector(ui_queue, progress_tracker)
        success = collector.collect_until(
            condition=lambda: progress_tracker.preprocessing_complete,
            timeout=PREPROCESSING_TIMEOUT * len(sample_pdfs),
        )

        # Wait for worker to finish
        worker.join(timeout=5)

        # Assertions
        assert success, f"Preprocessing timed out. Messages: {progress_tracker.messages[-10:]}"
        assert progress_tracker.files_processed == len(sample_pdfs), (
            f"Expected {len(sample_pdfs)} files, got {progress_tracker.files_processed}"
        )
        assert not progress_tracker.errors, f"Errors: {progress_tracker.errors}"

    def test_preprocessing_results_have_text(self, sample_pdfs, ui_queue, progress_tracker):
        """Test that extracted documents contain text."""
        from src.services.workers import ProcessingWorker

        worker = ProcessingWorker(
            file_paths=sample_pdfs[:1],  # Just first file for speed
            ui_queue=ui_queue,
        )
        worker.start()

        collector = QueueCollector(ui_queue, progress_tracker)
        collector.collect_until(
            condition=lambda: progress_tracker.preprocessing_complete, timeout=PREPROCESSING_TIMEOUT
        )
        worker.join(timeout=5)

        # Check that we captured file_processed with text
        file_messages = [m for m in progress_tracker.messages if m[1] == MessageType.FILE_PROCESSED]
        assert file_messages, "No file_processed messages received"


class TestHeadlessVocabulary:
    """Test vocabulary extraction without GUI."""

    @pytest.fixture
    def extracted_text(self, sample_pdfs, ui_queue, progress_tracker):
        """Pre-extract text from one sample PDF for vocabulary tests."""
        from src.services.workers import ProcessingWorker

        # Use only the first (smallest) PDF to keep extraction time reasonable on CPU
        one_pdf = sample_pdfs[:1]
        worker = ProcessingWorker(file_paths=one_pdf, ui_queue=ui_queue)
        worker.start()

        collector = QueueCollector(ui_queue, progress_tracker)
        collector.collect_until(
            condition=lambda: progress_tracker.preprocessing_complete,
            timeout=PREPROCESSING_TIMEOUT,
        )
        worker.join(timeout=5)

        return worker.processed_results

    def test_ner_extraction_completes(self, extracted_text, ui_queue, progress_tracker):
        """Test that NER extraction completes within timeout."""
        from src.core.utils.text_utils import combine_document_texts
        from src.services.workers import VocabularyWorker

        if not extracted_text:
            pytest.skip("No extracted text available")

        combined_text = combine_document_texts(extracted_text)

        worker = VocabularyWorker(
            combined_text=combined_text,
            ui_queue=ui_queue,
            doc_count=len(extracted_text),
            use_llm=False,  # NER only for speed
        )
        worker.start()

        # Wait for vocab result
        start_time = time.time()
        result = None
        while time.time() - start_time < PHASE1_TIMEOUT:
            try:
                msg = ui_queue.get(timeout=1)
                if msg[0] == MessageType.VOCAB_CSV_GENERATED:
                    result = msg[1]
                    break
            except Empty:
                continue

        worker.join(timeout=5)

        assert result is not None, f"Vocabulary extraction timed out after {PHASE1_TIMEOUT}s"
        assert len(result) > 0, "No vocabulary terms extracted"
        logger.debug(f"[TEST] Extracted {len(result)} vocabulary terms")


class TestHeadlessProgressiveExtraction:
    """Test the full progressive extraction pipeline without GUI."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_progressive_extraction_phases(self, sample_pdfs, ui_queue, progress_tracker):
        """
        Test that ProgressiveExtractionWorker completes all phases.

        This is the core test that reproduces the "stuck" issue.
        """
        from src.core.utils.text_utils import combine_document_texts
        from src.services.workers import ProcessingWorker, ProgressiveExtractionWorker

        # Phase 0: Preprocessing
        logger.debug("[TEST] Starting preprocessing...")
        preprocess_worker = ProcessingWorker(file_paths=sample_pdfs, ui_queue=ui_queue)
        preprocess_worker.start()

        collector = QueueCollector(ui_queue, progress_tracker)
        success = collector.collect_until(
            condition=lambda: progress_tracker.preprocessing_complete,
            timeout=PREPROCESSING_TIMEOUT * len(sample_pdfs),
        )
        preprocess_worker.join(timeout=5)

        assert success, "Preprocessing timed out"
        documents = preprocess_worker.processed_results
        assert documents, "No documents extracted"

        # Check confidence scores
        for doc in documents:
            conf = doc.get("confidence", 0)
            logger.debug(f"[TEST] {doc.get('filename', '?')}: {conf}% confidence")
            # Your typical range is 80-90%
            assert conf > 50, f"Low confidence: {conf}%"

        # Phase 1-3: Progressive Extraction
        logger.debug("[TEST] Starting progressive extraction...")
        combined_text = combine_document_texts(documents)

        # Reset progress tracker for extraction phases
        progress_tracker.ner_complete = False
        progress_tracker.qa_ready = False
        progress_tracker.llm_complete = False

        # Fresh queue for extraction
        extract_queue = Queue()
        extract_progress = WorkflowProgress()

        extract_worker = ProgressiveExtractionWorker(
            documents=documents,
            combined_text=combined_text,
            ui_queue=extract_queue,
            use_llm=False,  # Skip LLM for faster testing
        )
        extract_worker.start()

        # Collect extraction messages
        extract_collector = QueueCollector(extract_queue, extract_progress)

        # Wait for NER (Phase 1)
        logger.debug("[TEST] Waiting for Phase 1 (NER)...")
        ner_success = extract_collector.collect_until(
            condition=lambda: extract_progress.ner_complete, timeout=PHASE1_TIMEOUT
        )
        assert ner_success, f"Phase 1 (NER) timed out. Messages: {extract_progress.messages[-10:]}"
        logger.debug("[TEST] Phase 1 complete!")

        # Wait for Q&A Ready (Phase 2)
        logger.debug("[TEST] Waiting for Phase 2 (Q&A indexing)...")
        qa_success = extract_collector.collect_until(
            condition=lambda: extract_progress.qa_ready, timeout=PHASE2_TIMEOUT
        )
        assert qa_success, f"Phase 2 (Q&A) timed out. Messages: {extract_progress.messages[-10:]}"
        logger.debug("[TEST] Phase 2 complete!")

        # Wait for LLM Complete (Phase 3) - even with use_llm=False, we get llm_complete with empty list
        logger.debug("[TEST] Waiting for Phase 3 (LLM/finalization)...")
        llm_success = extract_collector.collect_until(
            condition=lambda: extract_progress.llm_complete,
            timeout=60,  # Should be fast since LLM is disabled
        )
        assert llm_success, f"Phase 3 timed out. Messages: {extract_progress.messages[-10:]}"
        logger.debug("[TEST] Phase 3 complete!")

        # Cleanup
        extract_worker.join(timeout=10)

        # Final assertions
        assert not extract_progress.errors, f"Errors during extraction: {extract_progress.errors}"
        logger.debug("[TEST] All phases completed successfully!")


# =============================================================================
# GUI SIMULATION TESTS - Test with actual MainWindow
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestGUISimulation:
    """
    Test the full GUI workflow by simulating user actions.

    These tests are slower and require a display (or Xvfb on Linux CI).
    Skip if no display available.
    """

    @pytest.fixture
    def can_run_gui(self):
        """Check if GUI tests can run.

        Tests both basic tkinter AND customtkinter/MainWindow compatibility.
        The Microsoft Store Python installation can have incomplete Tcl/Tk files
        that cause failures only with customtkinter.
        """
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            root.destroy()
            return True
        except Exception:
            return False

    def test_full_workflow_simulation(self, sample_pdfs, can_run_gui):
        """
        Simulate the full user workflow:
        1. Load files
        2. Wait for preprocessing
        3. Click Process Documents
        4. Wait for completion
        """
        if not can_run_gui:
            pytest.skip("No display available for GUI tests")

        # Additional check: try importing MainWindow early to catch Tcl/Tk issues
        try:
            from src.ui.main_window import MainWindow
        except Exception as e:
            if "tk.tcl" in str(e) or "TclError" in str(type(e).__name__):
                pytest.skip(f"Tcl/Tk installation incomplete: {e}")
            raise

        from src.ui.main_window import MainWindow

        # Track progress
        workflow_complete = threading.Event()
        errors = []

        def run_gui():
            """Run GUI in main thread with simulated actions."""
            try:
                app = MainWindow()

                # Inject sample files
                app.selected_files = sample_pdfs

                # Start preprocessing
                app._start_preprocessing()

                # Poll until preprocessing complete or timeout
                start_time = time.time()
                while time.time() - start_time < PREPROCESSING_TIMEOUT * len(sample_pdfs):
                    app.update()
                    if app.processing_results and len(app.processing_results) == len(sample_pdfs):
                        break
                    time.sleep(0.1)

                # Verify preprocessing succeeded
                if not app.processing_results:
                    errors.append("Preprocessing produced no results")
                    app.destroy()
                    return

                logger.debug(
                    f"[GUI TEST] Preprocessing complete: {len(app.processing_results)} files"
                )

                # Simulate clicking "Process Documents"
                # Set checkboxes
                app.vocab_check.select()
                app.qa_check.select()
                app.vocab_llm_check.deselect()  # Skip LLM for speed

                # Click process
                app._perform_tasks()

                # Poll until extraction complete or timeout
                start_time = time.time()
                while time.time() - start_time < TOTAL_TIMEOUT:
                    app.update()

                    # Check if tasks completed
                    if hasattr(app, "_completed_tasks") and "vocab" in app._completed_tasks:
                        logger.debug("[GUI TEST] Vocabulary task complete")
                    if app._qa_ready:
                        logger.debug("[GUI TEST] Q&A ready")

                    # Check for stuck state
                    if (
                        app._qa_ready
                        and hasattr(app, "_completed_tasks")
                        and "vocab" in app._completed_tasks
                    ):
                        workflow_complete.set()
                        break

                    time.sleep(0.1)

                if not workflow_complete.is_set():
                    errors.append(f"Workflow did not complete within {TOTAL_TIMEOUT}s")
                    # Capture state for debugging
                    errors.append(f"_qa_ready: {app._qa_ready}")
                    errors.append(f"_completed_tasks: {getattr(app, '_completed_tasks', 'N/A')}")

                app.destroy()

            except Exception as e:
                errors.append(f"GUI error: {e}")
                import traceback

                errors.append(traceback.format_exc())

        # Run in main thread (required for tkinter)
        run_gui()

        # Check for Tcl/Tk installation issues (should skip, not fail)
        if errors:
            error_text = "\n".join(errors)
            if "tk.tcl" in error_text or "TclError" in error_text:
                pytest.skip("Tcl/Tk installation incomplete - skipping GUI test")

        # Assertions
        assert not errors, "GUI test errors:\n" + "\n".join(errors)
        assert workflow_complete.is_set(), "Workflow did not complete"


# =============================================================================
# Diagnostic Tests - Help identify stuck points
# =============================================================================


class TestDiagnostics:
    """Diagnostic tests to help identify where processing gets stuck."""

    def test_worker_thread_health(self, sample_pdfs, ui_queue):
        """Test that worker threads start and stop properly."""
        from src.services.workers import ProcessingWorker

        worker = ProcessingWorker(
            file_paths=sample_pdfs[:1],  # Just one file
            ui_queue=ui_queue,
        )

        # Thread should not be alive before start
        assert not worker.is_alive()

        worker.start()

        # Thread should be alive after start
        assert worker.is_alive()

        # Wait for completion with timeout
        worker.join(timeout=PREPROCESSING_TIMEOUT)

        # Thread should be dead after completion
        assert not worker.is_alive(), "Worker thread did not terminate"

    def test_queue_message_delivery(self, ui_queue):
        """Test that queue messages are delivered correctly."""
        # Put a message
        ui_queue.put(QueueMessage.progress(50, "Test message"))

        # Get the message
        msg = ui_queue.get(timeout=1)

        assert msg[0] == MessageType.PROGRESS
        assert msg[1] == (50, "Test message")

    def test_embeddings_load_time(self):
        """Test how long embeddings take to load (common stuck point)."""
        start_time = time.time()

        from langchain_huggingface import HuggingFaceEmbeddings

        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

        load_time = time.time() - start_time
        logger.debug(f"[DIAGNOSTIC] Embeddings loaded in {load_time:.1f}s")

        # Should load within reasonable time
        assert load_time < 60, f"Embeddings took too long to load: {load_time}s"

    def test_ollama_connection(self):
        """Test that Ollama is available (required for LLM phases)."""
        from src.core.ai import OllamaModelManager

        manager = OllamaModelManager()

        if not manager.is_connected:
            pytest.skip("Ollama not running - LLM tests will be skipped")

        logger.debug(f"[DIAGNOSTIC] Ollama connected, model: {manager.model_name}")


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    # Enable debug logging
    os.environ["DEBUG"] = "true"

    # Run specific tests
    pytest.main(
        [
            __file__,
            "-v",
            "-s",  # Show print output
            "--tb=short",
            "-k",
            "TestHeadlessProgressiveExtraction",  # Run the main workflow test
        ]
    )
