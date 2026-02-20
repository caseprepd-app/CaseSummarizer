"""
Tests for concurrency audit fixes.

Validates thread-safe singletons, metrics locking, queue drain patterns,
daemon thread settings, and cross-thread error message protection.
"""

import threading
from queue import Empty, Queue
from unittest.mock import MagicMock, patch

import pytest


# =========================================================================
# 1. AIService thread-safe singleton
# =========================================================================


class TestAIServiceSingleton:
    """Verify AIService uses double-checked locking."""

    def test_lock_exists(self):
        """Module-level lock must exist for thread safety."""
        from src.services import ai_service

        assert hasattr(ai_service, "_ai_service_lock")
        assert isinstance(ai_service._ai_service_lock, type(threading.Lock()))

    def test_concurrent_creation_returns_same_instance(self):
        """10 threads creating AIService simultaneously must get the same instance."""
        from src.services.ai_service import AIService

        AIService.reset_singleton()
        instances = [None] * 10
        barrier = threading.Barrier(10)

        def create(idx):
            barrier.wait()
            instances[idx] = AIService()

        threads = [threading.Thread(target=create, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All instances must be the same object
        assert all(inst is instances[0] for inst in instances)
        AIService.reset_singleton()

    def test_reset_under_contention(self):
        """Alternating creation and reset from multiple threads must not raise."""
        from src.services.ai_service import AIService

        AIService.reset_singleton()
        errors = []

        def worker():
            try:
                for _ in range(20):
                    _ = AIService()
                    AIService.reset_singleton()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Errors during contention: {errors}"
        AIService.reset_singleton()


# =========================================================================
# 2. UserPreferencesManager thread-safe singleton
# =========================================================================


class TestUserPrefsSingleton:
    """Verify get_user_preferences uses double-checked locking."""

    def test_lock_exists(self):
        """Module-level lock must exist for thread safety."""
        from src import user_preferences

        assert hasattr(user_preferences, "_prefs_lock")
        assert isinstance(user_preferences._prefs_lock, type(threading.Lock()))

    def test_concurrent_creation_returns_same_instance(self, tmp_path):
        """10 threads calling get_user_preferences must get the same instance."""
        from src.user_preferences import get_user_preferences, reset_singleton

        reset_singleton()
        prefs_file = tmp_path / "test_prefs.json"
        instances = [None] * 10
        barrier = threading.Barrier(10)

        def create(idx):
            barrier.wait()
            instances[idx] = get_user_preferences(prefs_file)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert all(inst is instances[0] for inst in instances)
        reset_singleton()

    def test_reset_under_contention(self, tmp_path):
        """Alternating creation and reset must not raise."""
        from src.user_preferences import get_user_preferences, reset_singleton

        reset_singleton()
        prefs_file = tmp_path / "test_prefs.json"
        errors = []

        def worker():
            try:
                for _ in range(20):
                    _ = get_user_preferences(prefs_file)
                    reset_singleton()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Errors during contention: {errors}"
        reset_singleton()


# =========================================================================
# 3. SystemMonitor metrics lock + thread cleanup
# =========================================================================


class TestSystemMonitorConcurrency:
    """Verify metrics lock and thread join in stop_monitoring."""

    def test_metrics_lock_exists(self):
        """SystemMonitor must have a _metrics_lock attribute."""
        from src.ui.system_monitor import SystemMonitor

        # Check the __init__ code references _metrics_lock
        import inspect

        source = inspect.getsource(SystemMonitor.__init__)
        assert "_metrics_lock" in source

    def test_monitor_thread_stored(self):
        """start_monitoring must store the thread reference."""
        import inspect

        from src.ui.system_monitor import SystemMonitor

        source = inspect.getsource(SystemMonitor.start_monitoring)
        assert "_monitor_thread" in source

    def test_stop_monitoring_joins_thread(self):
        """stop_monitoring must join the thread with a timeout."""
        import inspect

        from src.ui.system_monitor import SystemMonitor

        source = inspect.getsource(SystemMonitor.stop_monitoring)
        assert ".join(" in source
        assert "timeout" in source

    def test_collect_metrics_uses_lock(self):
        """_collect_metrics must write under _metrics_lock."""
        import inspect

        from src.ui.system_monitor import SystemMonitor

        source = inspect.getsource(SystemMonitor._collect_metrics)
        assert "_metrics_lock" in source

    def test_schedule_update_uses_lock(self):
        """_schedule_main_thread_update must read under _metrics_lock."""
        import inspect

        from src.ui.system_monitor import SystemMonitor

        source = inspect.getsource(SystemMonitor._schedule_main_thread_update)
        assert "_metrics_lock" in source


# =========================================================================
# 4a/4b. Queue drain without TOCTOU
# =========================================================================


class TestQueueDrainPattern:
    """Verify _clear_queue and check_for_messages don't use queue.empty()."""

    def test_clear_queue_no_empty_check(self):
        """_clear_queue must not call queue.empty() (TOCTOU race)."""
        import inspect

        from src.services.workers import OllamaAIWorkerManager

        source = inspect.getsource(OllamaAIWorkerManager._clear_queue)
        assert "queue.empty()" not in source
        assert ".empty()" not in source

    def test_clear_queue_drains_all(self):
        """_clear_queue must drain all items from the queue."""
        from src.services.workers import OllamaAIWorkerManager

        q = Queue()
        for i in range(5):
            q.put(i)

        OllamaAIWorkerManager._clear_queue(q)
        assert q.empty()

    def test_clear_queue_handles_empty(self):
        """_clear_queue on empty queue must not raise."""
        from src.services.workers import OllamaAIWorkerManager

        q = Queue()
        OllamaAIWorkerManager._clear_queue(q)  # Should not raise

    def test_check_for_messages_no_empty_check(self):
        """check_for_messages must not call queue.empty() (TOCTOU race)."""
        import inspect

        from src.services.workers import OllamaAIWorkerManager

        source = inspect.getsource(OllamaAIWorkerManager.check_for_messages)
        assert ".empty()" not in source

    def test_check_for_messages_returns_all(self):
        """check_for_messages must return all queued items."""
        import time

        from src.services.workers import OllamaAIWorkerManager

        manager = OllamaAIWorkerManager(Queue())
        for i in range(3):
            manager.output_queue.put(f"msg_{i}")

        # multiprocessing.Queue needs a brief moment for items to be available
        time.sleep(0.1)

        messages = manager.check_for_messages()
        assert messages == ["msg_0", "msg_1", "msg_2"]
        assert manager.output_queue.empty()


# =========================================================================
# 4c. Q&A thread daemon=False
# =========================================================================


class TestQAThreadDaemon:
    """Verify Q&A indexing thread uses daemon=False."""

    def test_qa_thread_not_daemon(self):
        """The Q&A vector store thread must be non-daemon for clean shutdown."""
        import inspect

        from src.services.workers import ProgressiveExtractionWorker

        source = inspect.getsource(ProgressiveExtractionWorker.execute)
        # Should have daemon=False for the Q&A thread
        assert "daemon=False" in source


# =========================================================================
# 4d. _qa_error_msg protected by lock
# =========================================================================


class TestQAErrorLock:
    """Verify _qa_error_msg is protected by _qa_error_lock."""

    def test_error_lock_created(self):
        """ProgressiveExtractionWorker.__init__ must create _qa_error_lock."""
        import inspect

        from src.services.workers import ProgressiveExtractionWorker

        source = inspect.getsource(ProgressiveExtractionWorker.__init__)
        assert "_qa_error_lock" in source
        assert "threading.Lock()" in source

    def test_error_msg_read_under_lock(self):
        """Reading _qa_error_msg must be done under _qa_error_lock."""
        import inspect

        from src.services.workers import ProgressiveExtractionWorker

        source = inspect.getsource(ProgressiveExtractionWorker.execute)
        # The read should be inside a `with self._qa_error_lock:` block
        assert "with self._qa_error_lock:" in source

    def test_error_msg_write_under_lock(self):
        """Writing _qa_error_msg must be done under _qa_error_lock."""
        import inspect

        from src.services.workers import ProgressiveExtractionWorker

        source = inspect.getsource(ProgressiveExtractionWorker._build_vector_store)
        assert "with self._qa_error_lock:" in source
