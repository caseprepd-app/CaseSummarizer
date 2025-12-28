"""
Task runner for parallel document processing.

Provides a high-level interface for running tasks with progress
reporting and cancellation support. Decouples task execution logic
from the specific parallelization strategy.

Design Principles:
- Single Responsibility: TaskResult holds result data, ParallelTaskRunner
  handles execution orchestration.
- Open/Closed: New execution strategies can be added without modifying
  the runner (via ExecutorStrategy injection).
- Dependency Inversion: Runner depends on ExecutorStrategy abstraction,
  not concrete implementations.

Usage:
    runner = ParallelTaskRunner(
        strategy=ThreadPoolStrategy(max_workers=2),
        on_task_complete=lambda task_id, result: print(f"{task_id} done")
    )

    items = [("doc1", "path/to/doc1.pdf"), ("doc2", "path/to/doc2.pdf")]
    results = runner.run(process_document, items)

    for result in results:
        if result.success:
            print(f"{result.task_id}: {result.result}")
        else:
            print(f"{result.task_id} failed: {result.error}")
"""

from dataclasses import dataclass
from typing import Callable, Any
from concurrent.futures import as_completed
import threading

from .executor_strategy import ExecutorStrategy


@dataclass
class TaskResult:
    """
    Result of a parallel task execution.

    Encapsulates the outcome of a single task, including success/failure
    status and either the result value or exception.

    Attributes:
        task_id: Unique identifier for the task (e.g., file path).
        success: True if task completed without exception.
        result: Return value from the task function (if success=True).
        error: Exception raised by the task (if success=False).

    Example:
        if result.success:
            process_output(result.result)
        else:
            log_error(f"Task {result.task_id} failed: {result.error}")
    """
    task_id: str
    success: bool
    result: Any = None
    error: Exception = None


class ParallelTaskRunner:
    """
    Runs tasks using a configurable ExecutorStrategy.

    Provides a unified interface for parallel task execution with:
    - Completion callbacks for progress tracking
    - Cancellation support via threading.Event
    - Results returned in completion order (not submission order)
    - Exception handling per-task (doesn't abort other tasks)

    The runner is decoupled from the execution strategy, allowing the same
    code to run with ThreadPoolStrategy (production) or SequentialStrategy
    (testing) without modification.

    Args:
        strategy: ExecutorStrategy implementation to use for execution.
        on_task_complete: Optional callback invoked when each task completes
                         successfully. Signature: (task_id: str, result: Any) -> None

    Attributes:
        strategy: The execution strategy being used.
        on_task_complete: Callback for successful task completions.

    Example:
        # With progress callback
        def on_complete(task_id, result):
            ui_queue.put(('file_processed', result))

        runner = ParallelTaskRunner(
            strategy=ThreadPoolStrategy(),
            on_task_complete=on_complete
        )

        items = [(path, path) for path in file_paths]
        results = runner.run(extractor.process_document, items)
    """

    def __init__(
        self,
        strategy: ExecutorStrategy,
        on_task_complete: Callable[[str, Any], None] = None
    ):
        """
        Initialize the task runner.

        Args:
            strategy: ExecutorStrategy for task execution.
            on_task_complete: Optional callback for successful completions.
        """
        self.strategy = strategy
        self.on_task_complete = on_task_complete
        self._cancel_event = threading.Event()

    def run(
        self,
        fn: Callable[[Any], Any],
        items: list[tuple[str, Any]]
    ) -> list[TaskResult]:
        """
        Run function over items in parallel.

        Submits all tasks to the executor strategy and collects results
        as they complete. Results are returned in completion order, not
        submission order. Each task's success/failure is tracked individually.

        Args:
            fn: Function to execute for each item. Should accept a single
                argument (the payload from items tuple).
            items: List of (task_id, payload) tuples. task_id is used for
                  tracking and callbacks; payload is passed to fn.

        Returns:
            List of TaskResult objects in completion order. Each result
            contains task_id, success status, and either result or error.

        Example:
            def process_doc(path: str) -> dict:
                return {"path": path, "pages": 10}

            items = [
                ("doc1.pdf", "/path/to/doc1.pdf"),
                ("doc2.pdf", "/path/to/doc2.pdf"),
            ]
            results = runner.run(process_doc, items)
        """
        if not items:
            return []

        results = []
        futures = {}

        # Submit all tasks to the executor
        for task_id, payload in items:
            if self._cancel_event.is_set():
                break
            future = self.strategy.submit(fn, payload)
            futures[future] = task_id

        # Collect results as they complete (completion order)
        for future in as_completed(futures):
            if self._cancel_event.is_set():
                break

            task_id = futures[future]
            try:
                result = future.result()
                task_result = TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result
                )
                # Invoke callback for successful completion
                if self.on_task_complete:
                    self.on_task_complete(task_id, result)
            except Exception as e:
                task_result = TaskResult(
                    task_id=task_id,
                    success=False,
                    error=e
                )

            results.append(task_result)

        return results

    def cancel(self):
        """
        Signal cancellation to running tasks.

        Sets the cancel event and attempts to shutdown the executor.
        Tasks that check is_cancelled can exit early. Note that tasks
        already in progress may continue until their next cancellation
        check point.
        """
        self._cancel_event.set()
        self.strategy.shutdown(wait=False, cancel_futures=True)

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_event.is_set()
