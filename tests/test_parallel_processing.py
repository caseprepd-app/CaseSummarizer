"""Tests for parallel execution: ExecutorStrategy, ParallelTaskRunner, ProgressAggregator."""

import threading
import time
from concurrent.futures import Future
from queue import Queue

import pytest

# ---------------------------------------------------------------------------
# ExecutorStrategy & implementations
# ---------------------------------------------------------------------------


class TestSequentialStrategy:
    """SequentialStrategy executes tasks synchronously."""

    def test_max_workers_is_one(self):
        from src.core.parallel.executor_strategy import SequentialStrategy

        s = SequentialStrategy()
        assert s.max_workers == 1

    def test_map_returns_results_in_order(self):
        from src.core.parallel.executor_strategy import SequentialStrategy

        s = SequentialStrategy()
        results = list(s.map(lambda x: x * 2, [1, 2, 3]))
        assert results == [2, 4, 6]

    def test_map_empty_list(self):
        from src.core.parallel.executor_strategy import SequentialStrategy

        s = SequentialStrategy()
        results = list(s.map(lambda x: x, []))
        assert results == []

    def test_submit_returns_completed_future(self):
        from src.core.parallel.executor_strategy import SequentialStrategy

        s = SequentialStrategy()
        future = s.submit(lambda x: x + 10, 5)
        assert isinstance(future, Future)
        assert future.done()
        assert future.result() == 15

    def test_submit_captures_exception(self):
        from src.core.parallel.executor_strategy import SequentialStrategy

        s = SequentialStrategy()

        def fail(x):
            raise ValueError("bad input")

        future = s.submit(fail, 42)
        assert future.done()
        with pytest.raises(ValueError, match="bad input"):
            future.result()

    def test_shutdown_is_noop(self):
        from src.core.parallel.executor_strategy import SequentialStrategy

        s = SequentialStrategy()
        s.shutdown()  # Should not raise
        s.shutdown(wait=False, cancel_futures=True)  # Also fine

    def test_context_manager(self):
        from src.core.parallel.executor_strategy import SequentialStrategy

        with SequentialStrategy() as s:
            results = list(s.map(str.upper, ["a", "b"]))
        assert results == ["A", "B"]


class TestThreadPoolStrategy:
    """ThreadPoolStrategy uses real threads."""

    def test_default_max_workers(self):
        import os

        from src.core.parallel.executor_strategy import ThreadPoolStrategy

        with ThreadPoolStrategy() as s:
            assert s.max_workers == min(os.cpu_count() or 4, 4)

    def test_custom_max_workers(self):
        from src.core.parallel.executor_strategy import ThreadPoolStrategy

        with ThreadPoolStrategy(max_workers=2) as s:
            assert s.max_workers == 2

    def test_map_produces_correct_results(self):
        from src.core.parallel.executor_strategy import ThreadPoolStrategy

        with ThreadPoolStrategy(max_workers=2) as s:
            results = list(s.map(lambda x: x**2, [1, 2, 3, 4]))
        assert sorted(results) == [1, 4, 9, 16]

    def test_submit_returns_future(self):
        from src.core.parallel.executor_strategy import ThreadPoolStrategy

        with ThreadPoolStrategy(max_workers=1) as s:
            future = s.submit(lambda x: x + 1, 41)
            assert future.result(timeout=5) == 42

    def test_submit_exception_propagates(self):
        from src.core.parallel.executor_strategy import ThreadPoolStrategy

        with ThreadPoolStrategy(max_workers=1) as s:

            def fail(x):
                raise RuntimeError("oops")

            future = s.submit(fail, 0)
            with pytest.raises(RuntimeError, match="oops"):
                future.result(timeout=5)

    def test_context_manager_shuts_down(self):
        from src.core.parallel.executor_strategy import ThreadPoolStrategy

        s = ThreadPoolStrategy(max_workers=1)
        s.__enter__()
        s.__exit__(None, None, None)
        # Submitting after shutdown should raise
        with pytest.raises(RuntimeError):
            s.submit(lambda x: x, 1)

    def test_parallel_execution_actually_parallel(self):
        """Verify tasks run concurrently, not sequentially."""
        from src.core.parallel.executor_strategy import ThreadPoolStrategy

        results = []
        lock = threading.Lock()

        def slow_task(x):
            time.sleep(0.1)
            with lock:
                results.append(x)
            return x

        start = time.time()
        with ThreadPoolStrategy(max_workers=4) as s:
            list(s.map(slow_task, [1, 2, 3, 4]))
        elapsed = time.time() - start

        assert len(results) == 4
        # 4 tasks at 0.1s each should take ~0.1s parallel, not 0.4s
        assert elapsed < 0.35


# ---------------------------------------------------------------------------
# ParallelTaskRunner
# ---------------------------------------------------------------------------


class TestParallelTaskRunner:
    """ParallelTaskRunner.run() with SequentialStrategy for determinism."""

    def test_empty_items_returns_empty(self):
        from src.core.parallel.executor_strategy import SequentialStrategy
        from src.core.parallel.task_runner import ParallelTaskRunner

        runner = ParallelTaskRunner(strategy=SequentialStrategy())
        results = runner.run(lambda x: x, [])
        assert results == []

    def test_run_success(self):
        from src.core.parallel.executor_strategy import SequentialStrategy
        from src.core.parallel.task_runner import ParallelTaskRunner

        runner = ParallelTaskRunner(strategy=SequentialStrategy())
        items = [("a", 1), ("b", 2), ("c", 3)]
        results = runner.run(lambda x: x * 10, items)

        assert len(results) == 3
        by_id = {r.task_id: r for r in results}
        assert by_id["a"].success is True
        assert by_id["a"].result == 10
        assert by_id["b"].result == 20
        assert by_id["c"].result == 30

    def test_run_handles_per_task_failure(self):
        from src.core.parallel.executor_strategy import SequentialStrategy
        from src.core.parallel.task_runner import ParallelTaskRunner

        def sometimes_fail(x):
            if x == 2:
                raise ValueError("bad value")
            return x

        runner = ParallelTaskRunner(strategy=SequentialStrategy())
        items = [("a", 1), ("b", 2), ("c", 3)]
        results = runner.run(sometimes_fail, items)

        assert len(results) == 3
        by_id = {r.task_id: r for r in results}
        assert by_id["a"].success is True
        assert by_id["b"].success is False
        assert isinstance(by_id["b"].error, ValueError)
        assert by_id["c"].success is True

    def test_on_task_complete_callback(self):
        from src.core.parallel.executor_strategy import SequentialStrategy
        from src.core.parallel.task_runner import ParallelTaskRunner

        completed = []
        runner = ParallelTaskRunner(
            strategy=SequentialStrategy(),
            on_task_complete=lambda tid, r: completed.append((tid, r)),
        )
        runner.run(lambda x: x + 1, [("a", 10), ("b", 20)])

        assert ("a", 11) in completed
        assert ("b", 21) in completed

    def test_callback_not_called_on_failure(self):
        from src.core.parallel.executor_strategy import SequentialStrategy
        from src.core.parallel.task_runner import ParallelTaskRunner

        completed = []

        def fail(x):
            raise RuntimeError("fail")

        runner = ParallelTaskRunner(
            strategy=SequentialStrategy(),
            on_task_complete=lambda tid, r: completed.append(tid),
        )
        runner.run(fail, [("a", 1)])
        assert completed == []

    def test_cancel_sets_flag(self):
        from src.core.parallel.executor_strategy import SequentialStrategy
        from src.core.parallel.task_runner import ParallelTaskRunner

        runner = ParallelTaskRunner(strategy=SequentialStrategy())
        assert runner.is_cancelled is False
        runner.cancel()
        assert runner.is_cancelled is True

    def test_cancel_stops_submission(self):
        from src.core.parallel.executor_strategy import SequentialStrategy
        from src.core.parallel.task_runner import ParallelTaskRunner

        runner = ParallelTaskRunner(strategy=SequentialStrategy())
        runner._cancel_event.set()  # Pre-cancel

        results = runner.run(lambda x: x, [("a", 1), ("b", 2)])
        assert len(results) == 0


class TestParallelTaskRunnerThreaded:
    """TaskRunner with ThreadPoolStrategy for real parallelism."""

    def test_run_with_threads(self):
        from src.core.parallel.executor_strategy import ThreadPoolStrategy
        from src.core.parallel.task_runner import ParallelTaskRunner

        runner = ParallelTaskRunner(strategy=ThreadPoolStrategy(max_workers=2))
        items = [("a", 1), ("b", 2), ("c", 3)]
        results = runner.run(lambda x: x * 5, items)

        assert len(results) == 3
        assert all(r.success for r in results)
        result_values = sorted(r.result for r in results)
        assert result_values == [5, 10, 15]


class TestTaskResult:
    """TaskResult dataclass."""

    def test_success_result(self):
        from src.core.parallel.task_runner import TaskResult

        r = TaskResult(task_id="doc1", success=True, result={"pages": 10})
        assert r.task_id == "doc1"
        assert r.success is True
        assert r.result == {"pages": 10}
        assert r.error is None

    def test_failure_result(self):
        from src.core.parallel.task_runner import TaskResult

        err = ValueError("bad")
        r = TaskResult(task_id="doc2", success=False, error=err)
        assert r.success is False
        assert r.error is err
        assert r.result is None


# ---------------------------------------------------------------------------
# ProgressAggregator
# ---------------------------------------------------------------------------


class TestProgressState:
    """ProgressState percentage calculation."""

    def test_zero_total_returns_zero(self):
        from src.core.parallel.progress_aggregator import ProgressState

        state = ProgressState(total_tasks=0)
        assert state.percentage == 0

    def test_percentage_calculation(self):
        from src.core.parallel.progress_aggregator import ProgressState

        state = ProgressState(total_tasks=4, completed_tasks=1)
        assert state.percentage == 25

    def test_full_completion(self):
        from src.core.parallel.progress_aggregator import ProgressState

        state = ProgressState(total_tasks=3, completed_tasks=3)
        assert state.percentage == 100


class TestProgressAggregator:
    """ProgressAggregator throttles and aggregates UI updates."""

    def test_set_total_resets_state(self):
        from src.core.parallel.progress_aggregator import ProgressAggregator

        agg = ProgressAggregator(Queue())
        agg.set_total(5)
        assert agg.total == 5
        assert agg.completed == 0

    def test_complete_increments_counter(self):
        from src.core.parallel.progress_aggregator import ProgressAggregator

        q = Queue()
        agg = ProgressAggregator(q)
        agg.set_total(3)

        agg.complete("doc1")
        assert agg.completed == 1

        agg.complete("doc2")
        assert agg.completed == 2

    def test_complete_sends_immediate_update(self):
        from src.core.parallel.progress_aggregator import ProgressAggregator

        q = Queue()
        agg = ProgressAggregator(q, throttle_ms=10000)  # Long throttle
        agg.set_total(2)

        agg.complete("doc1")
        assert not q.empty()
        msg_type, (pct, text) = q.get_nowait()
        assert msg_type == "progress"
        assert pct == 50
        assert "1/2" in text

    def test_update_sends_task_message(self):
        from src.core.parallel.progress_aggregator import ProgressAggregator

        q = Queue()
        agg = ProgressAggregator(q, throttle_ms=0)  # No throttle
        agg.set_total(2)

        agg.update("doc1", "Extracting text...")
        assert not q.empty()
        msg_type, (pct, text) = q.get_nowait()
        assert "Extracting text" in text

    def test_throttle_suppresses_rapid_updates(self):
        from src.core.parallel.progress_aggregator import ProgressAggregator

        q = Queue()
        agg = ProgressAggregator(q, throttle_ms=5000)  # 5s throttle
        agg.set_total(10)

        # First update goes through
        agg.update("doc1", "msg1")
        count_after_first = q.qsize()

        # Rapid subsequent updates should be throttled
        for i in range(20):
            agg.update(f"doc{i}", f"msg{i}")

        # Should not have 20+ messages due to throttling
        assert q.qsize() < 5

    def test_aggregates_multiple_task_messages(self):
        from src.core.parallel.progress_aggregator import ProgressAggregator

        q = Queue()
        agg = ProgressAggregator(q, throttle_ms=0)
        agg.set_total(4)

        agg.update("doc1", "Processing doc1")
        agg.update("doc2", "Processing doc2")

        # Get last message
        last_msg = None
        while not q.empty():
            last_msg = q.get_nowait()

        _, (pct, text) = last_msg
        assert "Processing doc1" in text or "Processing doc2" in text

    def test_more_than_3_messages_shows_overflow(self):
        from src.core.parallel.progress_aggregator import ProgressAggregator

        q = Queue()
        agg = ProgressAggregator(q, throttle_ms=0)
        agg.set_total(10)

        for i in range(5):
            agg.update(f"doc{i}", f"Task {i}")

        # Get last message
        last_msg = None
        while not q.empty():
            last_msg = q.get_nowait()

        _, (pct, text) = last_msg
        assert "+2 more" in text  # 5 - 3 = 2 overflow

    def test_complete_removes_task_message(self):
        from src.core.parallel.progress_aggregator import ProgressAggregator

        q = Queue()
        agg = ProgressAggregator(q, throttle_ms=0)
        agg.set_total(2)

        agg.update("doc1", "Working...")
        agg.complete("doc1")

        # The completed task's message should be gone from state
        with agg._lock:
            assert "doc1" not in agg._state.task_messages


class TestProgressAggregatorThreadSafety:
    """Thread-safety of ProgressAggregator."""

    def test_concurrent_updates(self):
        from src.core.parallel.progress_aggregator import ProgressAggregator

        q = Queue()
        agg = ProgressAggregator(q, throttle_ms=0)
        n = 100
        agg.set_total(n)

        threads = []
        for i in range(n):
            t = threading.Thread(target=agg.complete, args=(f"task{i}",))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert agg.completed == n
        assert agg.total == n

    def test_concurrent_update_and_complete(self):
        from src.core.parallel.progress_aggregator import ProgressAggregator

        q = Queue()
        agg = ProgressAggregator(q, throttle_ms=0)
        agg.set_total(50)

        def worker(task_id):
            agg.update(task_id, f"Working on {task_id}")
            time.sleep(0.001)
            agg.complete(task_id)

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert agg.completed == 50
