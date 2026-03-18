"""
Base Worker Module

Provides a base class for all background workers with common boilerplate:
- Thread setup with daemon mode
- Stop event handling
- Error handling and logging
- Progress reporting

Usage:
    from src.services.base_worker import BaseWorker
    from src.services.queue_messages import QueueMessage

    class MyWorker(BaseWorker):
        def __init__(self, data, ui_queue):
            super().__init__(ui_queue)
            self.data = data

        def execute(self):
            # Main work happens here
            self.check_cancelled()
            self.send_progress(50, "Working...")
            result = process(self.data)
            self.ui_queue.put(QueueMessage.my_result(result))

        def _cleanup(self):
            # Optional cleanup (called in finally block)
            pass
"""

import gc
import logging
import threading
from queue import Queue

from src.services.queue_messages import QueueMessage
from src.services.status_reporter import StatusReporter

logger = logging.getLogger(__name__)


class BaseWorker(threading.Thread):
    """
    Base class for all background workers.

    Provides:
    - Daemon thread setup
    - Stop event handling
    - Standard error handling with debug logging
    - Progress reporting utilities via StatusReporter

    Subclasses should:
    1. Call super().__init__(ui_queue) in their __init__
    2. Override execute() with their main work
    3. Optionally override _cleanup() for custom cleanup
    4. Optionally override _operation_name for better error messages
    """

    def __init__(self, ui_queue: Queue):
        """
        Initialize base worker.

        Args:
            ui_queue: Queue for communication with the main UI thread.
        """
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.ui_queue = ui_queue
        self.status = StatusReporter(ui_queue)

    def stop(self) -> None:
        """
        Signal the worker to stop processing.

        Subclasses can override this to add custom stop logic,
        but should call super().stop() first.
        """
        logger.debug("Stop signal received for %s", self._worker_name)
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_event.is_set()

    def check_cancelled(self, message: str = "Cancelled") -> None:
        """
        Raise InterruptedError if stop has been requested.

        Call this at safe cancellation points in execute().

        Args:
            message: Message for the InterruptedError

        Raises:
            InterruptedError: If stop has been requested
        """
        if self.is_stopped:
            raise InterruptedError(message)

    def send_progress(self, percentage: int, message: str) -> None:
        """
        Send progress update if not stopped.

        Args:
            percentage: Progress 0-100
            message: Status message to display
        """
        if not self.is_stopped:
            self.status.update(percentage, message)

    def send_status_error(self, message: str) -> None:
        """
        Display a non-fatal error in the status bar (orange text).

        Unlike send_error() which shows a blocking modal dialog,
        this displays briefly in the status bar without interrupting.

        Args:
            message: Human-readable error description
        """
        self.status.error(message)

    def send_error(self, operation: str, error: Exception) -> None:
        """
        Log and send error message.

        Args:
            operation: What was being attempted (for error message)
            error: The exception that occurred
        """
        error_msg = f"{operation} failed: {error!s}"
        logger.error("%s: %s", self._worker_name, error_msg, exc_info=True)
        self.ui_queue.put(QueueMessage.error(error_msg))

    def run(self) -> None:
        """
        Execute the worker's main task with error handling.

        Calls execute() with try/except/finally wrapping.
        Do not override this - override execute() instead.
        """
        try:
            self.execute()
        except InterruptedError:
            logger.debug("%s cancelled by user", self._worker_name)
        except Exception as e:
            self.send_error(self._operation_name, e)
        finally:
            self._cleanup()

    def execute(self) -> None:
        """
        Main work implementation.

        Override this in subclasses. This is where the worker's
        main processing happens.

        Raises:
            NotImplementedError: If not overridden
        """
        raise NotImplementedError("Subclasses must implement execute()")

    @property
    def _worker_name(self) -> str:
        """Worker name for logging (derived from class name)."""
        return self.__class__.__name__.upper().replace("WORKER", " WORKER")

    @property
    def _operation_name(self) -> str:
        """
        Human-readable operation name for error messages.

        Override this for more descriptive error messages.
        Default extracts name from class name.
        """
        name = self.__class__.__name__
        # Remove 'Worker' suffix and add spaces before capitals
        name = name.replace("Worker", "")
        # Add spaces before capitals (CamelCase -> words)
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append(" ")
            result.append(char)
        return "".join(result) + " processing"

    def _cleanup(self) -> None:
        """
        Optional cleanup called in finally block.

        Override this to add custom cleanup such as:
        - Shutting down strategies/executors
        - Garbage collection
        - Releasing resources
        """
        pass


class CleanupWorker(BaseWorker):
    """
    BaseWorker with automatic garbage collection on cleanup.

    Use this for workers that may allocate significant memory
    (e.g., loading models, processing large documents).
    """

    def _cleanup(self) -> None:
        """Force garbage collection on cleanup."""
        gc.collect()
