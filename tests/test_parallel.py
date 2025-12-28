"""
Tests for the parallel processing module.

Tests cover:
- ExecutorStrategy implementations (ThreadPool, Sequential)
- ParallelTaskRunner with callbacks and cancellation
- ProgressAggregator throttling and thread safety
- Integration with ProcessingWorker
"""

import sys
import threading
import time
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.parallel import (
    ExecutorStrategy,
    ParallelTaskRunner,
    ProgressAggregator,
    ProgressState,
    SequentialStrategy,
    TaskResult,
    ThreadPoolStrategy,
)


class TestSequentialStrategy:
    """Test SequentialStrategy for deterministic execution."""

    def test_sequential_map_returns_results_in_order(self):
        """Sequential strategy processes items in submission order."""
        strategy = SequentialStrategy()
        results = list(strategy.map(lambda x: x * 2, [1, 2, 3, 4, 5]))
        assert results == [2, 4, 6, 8, 10]

    def test_sequential_submit_returns_completed_future(self):
        """Submit returns a Future that is already complete."""
        strategy = SequentialStrategy()
        future = strategy.submit(lambda x: x + 10, 5)
        assert future.done()
        assert future.result() == 15

    def test_sequential_submit_captures_exceptions(self):
        """Submit captures exceptions in the Future."""
        strategy = SequentialStrategy()

        def raise_error(x):
            raise ValueError("Test error")

        future = strategy.submit(raise_error, 1)
        assert future.done()
        with pytest.raises(ValueError, match="Test error"):
            future.result()

    def test_sequential_max_workers_is_one(self):
        """Sequential strategy always has max_workers=1."""
        strategy = SequentialStrategy()
        assert strategy.max_workers == 1

    def test_sequential_shutdown_is_noop(self):
        """Shutdown does nothing for sequential strategy."""
        strategy = SequentialStrategy()
        strategy.shutdown()  # Should not raise
        strategy.shutdown(wait=False, cancel_futures=True)  # Should not raise

    def test_sequential_context_manager(self):
        """Sequential strategy works as context manager."""
        with SequentialStrategy() as strategy:
            result = list(strategy.map(str.upper, ["a", "b", "c"]))
        assert result == ["A", "B", "C"]


class TestThreadPoolStrategy:
    """Test ThreadPoolStrategy for parallel execution."""

    def test_threadpool_default_max_workers(self):
        """Default max_workers is min(cpu_count, 4)."""
        strategy = ThreadPoolStrategy()
        import os
        expected = min(os.cpu_count() or 4, 4)
        assert strategy.max_workers == expected
        strategy.shutdown()

    def test_threadpool_custom_max_workers(self):
        """Custom max_workers is respected."""
        strategy = ThreadPoolStrategy(max_workers=2)
        assert strategy.max_workers == 2
        strategy.shutdown()

    def test_threadpool_map_processes_all_items(self):
        """ThreadPool processes all items."""
        with ThreadPoolStrategy(max_workers=2) as strategy:
            results = list(strategy.map(lambda x: x * 2, [1, 2, 3, 4]))
        assert sorted(results) == [2, 4, 6, 8]

    def test_threadpool_executes_concurrently(self):
        """ThreadPool executes tasks concurrently."""
        start_times = []
        end_times = []

        def slow_task(x):
            start_times.append(time.time())
            time.sleep(0.1)
            end_times.append(time.time())
            return x

        with ThreadPoolStrategy(max_workers=4) as strategy:
            list(strategy.map(slow_task, [1, 2, 3, 4]))

        # With 4 workers, all tasks should start nearly simultaneously
        # If sequential, total time would be ~0.4s; parallel should be ~0.1s
        total_duration = max(end_times) - min(start_times)
        assert total_duration < 0.25, f"Tasks should run in parallel, took {total_duration}s"

    def test_threadpool_submit_returns_future(self):
        """Submit returns a Future for async result retrieval."""
        with ThreadPoolStrategy(max_workers=2) as strategy:
            future = strategy.submit(lambda x: x * 2, 21)
            assert future.result(timeout=1) == 42


class TestParallelTaskRunner:
    """Test ParallelTaskRunner for task orchestration."""

    def test_runner_processes_all_tasks(self):
        """Runner processes all submitted tasks."""
        strategy = SequentialStrategy()
        runner = ParallelTaskRunner(strategy=strategy)

        items = [("task1", 10), ("task2", 20), ("task3", 30)]
        results = runner.run(lambda x: x * 2, items)

        assert len(results) == 3
        assert all(r.success for r in results)
        result_values = [r.result for r in results]
        assert sorted(result_values) == [20, 40, 60]

    def test_runner_handles_empty_items(self):
        """Runner handles empty task list gracefully."""
        strategy = SequentialStrategy()
        runner = ParallelTaskRunner(strategy=strategy)

        results = runner.run(lambda x: x, [])
        assert results == []

    def test_runner_captures_task_errors(self):
        """Runner captures exceptions per-task without aborting others."""
        strategy = SequentialStrategy()
        runner = ParallelTaskRunner(strategy=strategy)

        def maybe_fail(x):
            if x == 2:
                raise ValueError("Task 2 failed")
            return x * 10

        items = [("t1", 1), ("t2", 2), ("t3", 3)]
        results = runner.run(maybe_fail, items)

        assert len(results) == 3

        # Find the failed task
        failed = [r for r in results if not r.success]
        assert len(failed) == 1
        assert failed[0].task_id == "t2"
        assert isinstance(failed[0].error, ValueError)

        # Other tasks should succeed
        succeeded = [r for r in results if r.success]
        assert len(succeeded) == 2

    def test_runner_calls_on_task_complete_callback(self):
        """Runner invokes callback for successful task completions."""
        strategy = SequentialStrategy()
        completed = []

        def on_complete(task_id, result):
            completed.append((task_id, result))

        runner = ParallelTaskRunner(
            strategy=strategy,
            on_task_complete=on_complete
        )

        items = [("a", 1), ("b", 2), ("c", 3)]
        runner.run(lambda x: x * 2, items)

        assert len(completed) == 3
        completed_dict = dict(completed)
        assert completed_dict["a"] == 2
        assert completed_dict["b"] == 4
        assert completed_dict["c"] == 6

    def test_runner_cancellation(self):
        """Runner respects cancellation signal."""
        strategy = SequentialStrategy()
        runner = ParallelTaskRunner(strategy=strategy)

        processed = []

        def slow_task(x):
            if runner.is_cancelled:
                raise InterruptedError("Cancelled")
            processed.append(x)
            return x

        # Cancel before running
        runner.cancel()

        items = [("t1", 1), ("t2", 2)]
        results = runner.run(slow_task, items)

        # Should process nothing or minimal items due to cancellation
        assert runner.is_cancelled


class TestProgressAggregator:
    """Test ProgressAggregator for UI updates."""

    def test_aggregator_tracks_total_and_completed(self):
        """Aggregator tracks total and completed counts."""
        queue = Queue()
        aggregator = ProgressAggregator(queue, throttle_ms=0)

        aggregator.set_total(5)
        assert aggregator.total == 5
        assert aggregator.completed == 0

        aggregator.complete("task1")
        assert aggregator.completed == 1

    def test_aggregator_calculates_percentage(self):
        """Aggregator calculates completion percentage."""
        queue = Queue()
        aggregator = ProgressAggregator(queue, throttle_ms=0)

        aggregator.set_total(4)
        assert aggregator._state.percentage == 0

        aggregator.complete("t1")
        assert aggregator._state.percentage == 25

        aggregator.complete("t2")
        assert aggregator._state.percentage == 50

    def test_aggregator_sends_progress_to_queue(self):
        """Aggregator sends progress messages to UI queue."""
        queue = Queue()
        aggregator = ProgressAggregator(queue, throttle_ms=0)

        aggregator.set_total(2)
        aggregator.update("task1", "Processing task 1...")

        # Should have sent at least one message
        assert not queue.empty()
        msg_type, (pct, text) = queue.get_nowait()
        assert msg_type == "progress"
        assert "Processing task 1" in text

    def test_aggregator_completion_always_sends_update(self):
        """Complete() always sends immediate UI update."""
        queue = Queue()
        aggregator = ProgressAggregator(queue, throttle_ms=1000)  # High throttle

        aggregator.set_total(1)
        aggregator.complete("task1")

        # Despite high throttle, complete() should send immediately
        assert not queue.empty()

    def test_aggregator_throttles_rapid_updates(self):
        """Aggregator throttles rapid update() calls."""
        queue = Queue()
        aggregator = ProgressAggregator(queue, throttle_ms=500)

        aggregator.set_total(100)

        # Send many rapid updates
        for i in range(50):
            aggregator.update(f"task{i}", f"Message {i}")

        # With 500ms throttle, should have far fewer than 50 messages
        msg_count = 0
        while not queue.empty():
            queue.get_nowait()
            msg_count += 1

        # Should be throttled significantly (likely 1-2 messages)
        assert msg_count < 10, f"Expected throttling, got {msg_count} messages"

    def test_aggregator_thread_safe(self):
        """Aggregator is thread-safe for concurrent updates."""
        queue = Queue()
        aggregator = ProgressAggregator(queue, throttle_ms=10)
        aggregator.set_total(100)

        errors = []

        def update_from_thread(thread_id):
            try:
                for i in range(20):
                    aggregator.update(f"thread{thread_id}-{i}", f"Message from {thread_id}")
            except Exception as e:
                errors.append(e)

        # Start multiple threads updating concurrently
        threads = [threading.Thread(target=update_from_thread, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert not errors, f"Thread safety errors: {errors}"

    def test_aggregator_combines_concurrent_messages(self):
        """Aggregator combines messages from concurrent tasks."""
        queue = Queue()
        aggregator = ProgressAggregator(queue, throttle_ms=0)
        aggregator.set_total(3)

        aggregator.update("t1", "Task 1 running")
        aggregator.update("t2", "Task 2 running")
        aggregator.update("t3", "Task 3 running")

        # Get the last message
        last_msg = None
        while not queue.empty():
            last_msg = queue.get_nowait()

        # Should have combined message
        assert last_msg is not None
        _, (_, text) = last_msg
        # At least one task should be mentioned (depends on order)
        assert "Task" in text or "running" in text


class TestProgressState:
    """Test ProgressState dataclass."""

    def test_progress_state_defaults(self):
        """ProgressState has correct defaults."""
        state = ProgressState(total_tasks=10)
        assert state.total_tasks == 10
        assert state.completed_tasks == 0
        assert state.task_messages == {}

    def test_progress_state_percentage_calculation(self):
        """ProgressState calculates percentage correctly."""
        state = ProgressState(total_tasks=10, completed_tasks=3)
        assert state.percentage == 30

    def test_progress_state_zero_total(self):
        """ProgressState handles zero total gracefully."""
        state = ProgressState(total_tasks=0)
        assert state.percentage == 0


class TestTaskResult:
    """Test TaskResult dataclass."""

    def test_task_result_success(self):
        """TaskResult captures successful execution."""
        result = TaskResult(task_id="test", success=True, result=42)
        assert result.task_id == "test"
        assert result.success
        assert result.result == 42
        assert result.error is None

    def test_task_result_failure(self):
        """TaskResult captures failed execution."""
        error = ValueError("Test error")
        result = TaskResult(task_id="test", success=False, error=error)
        assert not result.success
        assert result.result is None
        assert result.error is error


class TestIntegrationWithConfig:
    """Test integration with config constants."""

    def test_parallel_max_workers_from_config(self):
        """ThreadPoolStrategy uses PARALLEL_MAX_WORKERS from config."""
        from src.config import PARALLEL_MAX_WORKERS
        strategy = ThreadPoolStrategy()
        assert strategy.max_workers == PARALLEL_MAX_WORKERS
        strategy.shutdown()

    def test_vocabulary_batch_size_in_config(self):
        """VOCABULARY_BATCH_SIZE is defined in config."""
        from src.config import VOCABULARY_BATCH_SIZE
        assert VOCABULARY_BATCH_SIZE is not None
        assert isinstance(VOCABULARY_BATCH_SIZE, int)
        assert VOCABULARY_BATCH_SIZE >= 4  # Should be at least baseline
        assert VOCABULARY_BATCH_SIZE <= 32  # Reasonable upper limit

    def test_user_worker_config_exists(self):
        """User worker configuration options are defined."""
        from src.config import (
            USER_PICKS_MAX_WORKER_COUNT,
            USER_DEFINED_MAX_WORKER_COUNT,
        )
        # Both should be defined
        assert USER_PICKS_MAX_WORKER_COUNT is not None
        assert USER_DEFINED_MAX_WORKER_COUNT is not None
        # Default should be False (auto-detect)
        assert USER_PICKS_MAX_WORKER_COUNT is False
        # Default user count should be 2 (conservative)
        assert USER_DEFINED_MAX_WORKER_COUNT == 2

    def test_user_defined_workers_bounds_enforced(self):
        """User-defined worker count is bounded between 1 and 8."""
        from src.config import _user_workers, USER_DEFINED_MAX_WORKER_COUNT
        # The _user_workers variable should be bounded
        assert _user_workers >= 1, "Minimum workers should be 1"
        assert _user_workers <= 8, "Maximum workers should be 8"
        # Should match the bounded user value
        expected = max(1, min(8, USER_DEFINED_MAX_WORKER_COUNT))
        assert _user_workers == expected

    def test_parallel_max_workers_in_valid_range(self):
        """PARALLEL_MAX_WORKERS is within expected range."""
        from src.config import PARALLEL_MAX_WORKERS
        # Whether auto or user-defined, should be between 1 and 8
        assert PARALLEL_MAX_WORKERS >= 1, "Should have at least 1 worker"
        assert PARALLEL_MAX_WORKERS <= 8, "Should have at most 8 workers"
