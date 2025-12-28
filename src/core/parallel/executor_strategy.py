"""
Parallel execution strategies for LocalScribe.

Implements Strategy Pattern to separate "what to parallelize" from
"how to parallelize". Enables dependency injection for testing.

Design Principles:
- Strategy Pattern: Different execution strategies (ThreadPool, Sequential)
  implement the same interface, allowing runtime selection.
- Dependency Injection: Workers accept strategy as parameter, enabling
  deterministic testing with SequentialStrategy.
- Liskov Substitution: SequentialStrategy can replace ThreadPoolStrategy
  without changing behavior (same interface, same semantics).

Usage:
    # Production (parallel execution)
    strategy = ThreadPoolStrategy(max_workers=4)

    # Testing (deterministic, sequential execution)
    strategy = SequentialStrategy()

    # Both work identically:
    results = list(strategy.map(process_func, items))
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, TypeVar, Iterator
import os

T = TypeVar('T')
R = TypeVar('R')


class ExecutorStrategy(ABC):
    """
    Abstract strategy for parallel execution.

    Defines the interface that all execution strategies must implement.
    Implementations can use ThreadPool, ProcessPool, or sequential
    execution (for testing).

    Attributes:
        max_workers: Number of concurrent workers (1 for sequential).
    """

    max_workers: int

    @abstractmethod
    def submit(self, fn: Callable[[T], R], item: T) -> Future:
        """
        Submit a single task for execution.

        Args:
            fn: Function to execute.
            item: Argument to pass to the function.

        Returns:
            Future object that will contain the result.
        """
        pass

    @abstractmethod
    def map(self, fn: Callable[[T], R], items: list[T]) -> Iterator[R]:
        """
        Map function over items, returning results.

        Note: Unlike concurrent.futures.Executor.map(), results may be
        returned in completion order (not submission order) depending
        on the implementation.

        Args:
            fn: Function to execute for each item.
            items: List of arguments to process.

        Returns:
            Iterator of results.
        """
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """
        Shutdown the executor and release resources.

        Args:
            wait: If True, wait for pending tasks to complete.
            cancel_futures: If True, cancel pending futures (Python 3.9+).
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.shutdown(wait=True)
        return False


class ThreadPoolStrategy(ExecutorStrategy):
    """
    Thread-based parallel execution strategy.

    Optimal for I/O-bound tasks (file reading, network) and tasks that
    release the GIL (PDF parsing via pdfplumber, OCR via pytesseract).

    The GIL (Global Interpreter Lock) is released by:
    - pdfplumber (C library calls)
    - pytesseract (subprocess to Tesseract)
    - pdf2image (C library calls)

    This makes threading effective for document extraction despite
    Python's GIL limitations.

    Args:
        max_workers: Maximum concurrent threads. Defaults to min(cpu_count, 4)
                    to balance performance with memory safety.

    Example:
        with ThreadPoolStrategy(max_workers=2) as strategy:
            results = list(strategy.map(process_doc, file_paths))
    """

    def __init__(self, max_workers: int = None):
        """
        Initialize thread pool strategy.

        Args:
            max_workers: Max threads. Defaults to min(cpu_count, 4) for
                        memory safety on typical business laptops.
        """
        if max_workers is None:
            # Default: CPU count capped at 4 for memory safety
            # Each document can use 200-500MB during processing
            max_workers = min(os.cpu_count() or 4, 4)

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers

    def submit(self, fn: Callable[[T], R], item: T) -> Future:
        """Submit a single task to the thread pool."""
        return self._executor.submit(fn, item)

    def map(self, fn: Callable[[T], R], items: list[T]) -> Iterator[R]:
        """
        Map function over items using thread pool.

        Note: Results are returned in submission order (not completion order).
        For completion-order results, use ParallelTaskRunner.run().
        """
        return self._executor.map(fn, items)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """
        Shutdown the thread pool.

        Args:
            wait: If True, wait for pending tasks to complete.
            cancel_futures: If True, attempt to cancel pending futures.
        """
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)


class SequentialStrategy(ExecutorStrategy):
    """
    Sequential execution strategy for testing and debugging.

    Allows the same code path to run single-threaded, making tests
    deterministic and easier to debug. Implements Liskov Substitution
    Principle - can replace ThreadPoolStrategy without changing behavior.

    Benefits for testing:
    - Deterministic execution order
    - Easier debugging (no thread interleaving)
    - No race conditions
    - Predictable timing

    Example:
        # In tests:
        worker = ProcessingWorker(
            file_paths=["test.pdf"],
            ui_queue=Queue(),
            strategy=SequentialStrategy()  # Deterministic for tests
        )
    """

    def __init__(self):
        """Initialize sequential strategy with max_workers=1."""
        self.max_workers = 1

    def submit(self, fn: Callable[[T], R], item: T) -> Future:
        """
        Execute function synchronously and return completed Future.

        The function is executed immediately (not deferred), and the
        result is wrapped in a Future for interface compatibility.
        """
        future: Future = Future()
        try:
            result = fn(item)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def map(self, fn: Callable[[T], R], items: list[T]) -> Iterator[R]:
        """
        Map function over items sequentially.

        Yields results one at a time as they complete (immediately).
        """
        for item in items:
            yield fn(item)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """No-op for sequential strategy (no resources to release)."""
        pass
