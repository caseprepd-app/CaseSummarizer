"""
Progress aggregation for parallel tasks.

Collects progress updates from multiple concurrent tasks and sends
throttled, unified updates to the UI queue. Thread-safe for use with
parallel document processing.

Design Principles:
- Thread Safety: All public methods use locking for concurrent access.
- Throttling: Prevents UI flooding from rapid parallel updates.
- Aggregation: Combines multiple task statuses into single UI message.

The problem this solves:
When processing 4 documents in parallel, each sending progress updates,
the UI queue could be flooded with 40+ messages per second. This causes:
- UI lag from processing too many queue messages
- Flickering/unreadable progress text
- Wasted CPU cycles

Solution: Aggregate updates and throttle to max 10/second by default.

Usage:
    aggregator = ProgressAggregator(ui_queue, throttle_ms=100)
    aggregator.set_total(5)

    # In parallel tasks:
    aggregator.update("doc1.pdf", "Extracting text...")
    aggregator.update("doc2.pdf", "Running OCR...")
    aggregator.complete("doc1.pdf")

    # UI receives throttled messages like:
    # "Extracting text... | Running OCR..." (25%)
    # "Processed 1/5 documents" (20%)
"""

from dataclasses import dataclass, field
from queue import Queue
import time
import threading


@dataclass
class ProgressState:
    """
    Tracks progress across multiple parallel tasks.

    Mutable state container for the aggregator. Not thread-safe on its own;
    ProgressAggregator provides the locking.

    Attributes:
        total_tasks: Total number of tasks to process.
        completed_tasks: Number of tasks that have finished.
        task_messages: Map of task_id -> current status message.
    """
    total_tasks: int
    completed_tasks: int = 0
    task_messages: dict = field(default_factory=dict)

    @property
    def percentage(self) -> int:
        """
        Calculate overall completion percentage.

        Returns:
            Integer percentage (0-100) of completed tasks.
        """
        if self.total_tasks == 0:
            return 0
        return int((self.completed_tasks / self.total_tasks) * 100)


class ProgressAggregator:
    """
    Aggregates progress from parallel tasks into unified UI updates.

    Thread-safe aggregator that collects status messages from multiple
    concurrent tasks and sends throttled updates to the UI queue. This
    prevents UI flooding while maintaining responsive progress feedback.

    Features:
    - Thread-safe: All methods can be called from multiple threads.
    - Throttled: Updates sent at most once per throttle_ms milliseconds.
    - Aggregated: Multiple task messages combined into single UI update.
    - Completion priority: Task completions always trigger immediate update.

    Args:
        ui_queue: Queue for sending progress updates to UI thread.
                  Messages are tuples: ('progress', (percentage, message))
        throttle_ms: Minimum milliseconds between updates (default 100 = 10/sec).

    Example:
        aggregator = ProgressAggregator(ui_queue)
        aggregator.set_total(len(file_paths))

        def process_doc(path):
            aggregator.update(path, f"Processing {Path(path).name}...")
            result = extractor.process_document(path)
            aggregator.complete(path)
            return result

        # Process documents in parallel - aggregator handles UI updates
        with ThreadPoolStrategy() as strategy:
            list(strategy.map(process_doc, file_paths))
    """

    def __init__(self, ui_queue: Queue, throttle_ms: int = 100):
        """
        Initialize the progress aggregator.

        Args:
            ui_queue: Queue for UI progress messages.
            throttle_ms: Minimum ms between updates (default 100).
        """
        self.ui_queue = ui_queue
        self.throttle_ms = throttle_ms
        self._state = ProgressState(total_tasks=0)
        self._last_update = 0.0
        self._lock = threading.Lock()

    def set_total(self, count: int) -> None:
        """
        Set total number of tasks and reset state.

        Should be called before starting parallel processing to initialize
        the progress tracking state.

        Args:
            count: Total number of tasks to be processed.
        """
        with self._lock:
            self._state = ProgressState(total_tasks=count)

    def update(self, task_id: str, message: str) -> None:
        """
        Update progress message for a task (throttled).

        Records the current status message for a task. The UI is updated
        only if enough time has passed since the last update (throttling).

        Thread-safe: Can be called from multiple worker threads.

        Args:
            task_id: Unique identifier for the task (e.g., file path).
            message: Current status message (e.g., "Extracting text...").
        """
        with self._lock:
            self._state.task_messages[task_id] = message
            self._maybe_send_update()

    def complete(self, task_id: str) -> None:
        """
        Mark a task as complete (always sends update).

        Increments the completed count and removes the task's status message.
        Unlike update(), this always sends an immediate UI update to ensure
        the user sees completion progress.

        Thread-safe: Can be called from multiple worker threads.

        Args:
            task_id: Unique identifier for the completed task.
        """
        with self._lock:
            self._state.completed_tasks += 1
            self._state.task_messages.pop(task_id, None)
            self._send_update()

    def _maybe_send_update(self) -> None:
        """
        Send update if throttle time has passed.

        Internal method - must be called while holding _lock.
        """
        now = time.time() * 1000
        if now - self._last_update >= self.throttle_ms:
            self._send_update()

    def _send_update(self) -> None:
        """
        Send aggregated progress to UI queue.

        Combines status messages from active tasks into a single string.
        Shows up to 3 concurrent task messages to avoid overwhelming the UI.

        Internal method - must be called while holding _lock.
        """
        messages = list(self._state.task_messages.values())

        if messages:
            # Show up to 3 concurrent task messages
            combined = " | ".join(messages[:3])
            if len(messages) > 3:
                combined += f" (+{len(messages) - 3} more)"
        else:
            # No active tasks - show overall completion status
            combined = f"Processed {self._state.completed_tasks}/{self._state.total_tasks} documents"

        self.ui_queue.put(('progress', (self._state.percentage, combined)))
        self._last_update = time.time() * 1000

    @property
    def completed(self) -> int:
        """Get number of completed tasks (thread-safe)."""
        with self._lock:
            return self._state.completed_tasks

    @property
    def total(self) -> int:
        """Get total number of tasks (thread-safe)."""
        with self._lock:
            return self._state.total_tasks
