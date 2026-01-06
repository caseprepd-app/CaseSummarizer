"""
Parallel processing utilities for LocalScribe.

This module provides Strategy Pattern-based parallel execution with support
for progress reporting, cancellation, and testability. It is designed for
document processing workflows where multiple files can be processed
concurrently.

Architecture:
    The module uses the Strategy Pattern to separate "what to parallelize"
    from "how to parallelize". This enables:

    1. Production use: ThreadPoolStrategy for actual parallel execution
    2. Testing: SequentialStrategy for deterministic, debuggable tests
    3. Future extension: Easy to add ProcessPoolStrategy if needed

Components:
    ExecutorStrategy - Abstract base class defining the execution interface
    ThreadPoolStrategy - Thread-based parallel execution (production)
    SequentialStrategy - Sequential execution (testing/debugging)
    ParallelTaskRunner - High-level task orchestration with callbacks
    TaskResult - Dataclass for task execution results
    ProgressAggregator - Thread-safe progress aggregation with throttling

Usage Example:
    from src.core.parallel import (
        ThreadPoolStrategy,
        ParallelTaskRunner,
        ProgressAggregator
    )

    # Set up progress reporting
    aggregator = ProgressAggregator(ui_queue)
    aggregator.set_total(len(file_paths))

    # Define the task function
    def process_doc(file_path):
        aggregator.update(file_path, f"Processing {file_path}...")
        result = extractor.process_document(file_path)
        aggregator.complete(file_path)
        return result

    # Run tasks in parallel
    runner = ParallelTaskRunner(
        strategy=ThreadPoolStrategy(max_workers=4),
        on_task_complete=lambda task_id, result: handle_result(result)
    )

    items = [(path, path) for path in file_paths]
    results = runner.run(process_doc, items)

Testing Example:
    from src.core.parallel import SequentialStrategy, ParallelTaskRunner

    # Use SequentialStrategy for deterministic tests
    runner = ParallelTaskRunner(
        strategy=SequentialStrategy(),
        on_task_complete=mock_callback
    )

    # Same code as production, but runs sequentially
    results = runner.run(process_doc, items)
    assert len(results) == len(items)

Performance Notes:
    - ThreadPoolStrategy defaults to min(cpu_count, 4) workers
    - Worker limit prevents memory exhaustion on typical laptops
    - ProgressAggregator throttles to max 10 updates/second
    - Document extraction releases GIL (pdfplumber, tesseract)
      making threading effective despite Python's GIL
"""

from .executor_strategy import (
    ExecutorStrategy,
    ThreadPoolStrategy,
    SequentialStrategy,
)
from .task_runner import ParallelTaskRunner, TaskResult
from .progress_aggregator import ProgressAggregator, ProgressState

__all__ = [
    # Strategies
    "ExecutorStrategy",
    "ThreadPoolStrategy",
    "SequentialStrategy",
    # Task runner
    "ParallelTaskRunner",
    "TaskResult",
    # Progress tracking
    "ProgressAggregator",
    "ProgressState",
]
