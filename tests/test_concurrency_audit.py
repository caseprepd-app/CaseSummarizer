"""
Tests for concurrency audit fixes.

Validates thread-safe singletons, metrics locking, queue drain patterns,
daemon thread settings, and cross-thread error message protection.
"""

import threading

# =========================================================================
# 1. UserPreferencesManager thread-safe singleton
# =========================================================================


class TestUserPrefsSingleton:
    """Verify get_user_preferences uses thread-safe SingletonHolder."""

    def test_lock_exists(self):
        """SingletonHolder must have a lock for thread safety."""
        from src import user_preferences

        assert hasattr(user_preferences, "_prefs_holder")
        assert isinstance(user_preferences._prefs_holder._lock, type(threading.Lock()))

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
            assert not t.is_alive(), f"Thread {t.name} did not finish within timeout"

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
# 4c. Q&A thread daemon=False
# =========================================================================


class TestQAThreadDaemon:
    """Verify Q&A indexing thread uses daemon=False."""

    def test_qa_thread_not_daemon(self):
        """The Q&A vector store thread must be non-daemon for clean shutdown."""
        import inspect

        from src.services.workers import ProgressiveExtractionWorker

        source = inspect.getsource(ProgressiveExtractionWorker._run_search_indexing)
        # Should have daemon=False for the Q&A thread
        assert "daemon=False" in source


# =========================================================================
# 4d. _search_error_msg protected by lock
# =========================================================================


class TestQAErrorLock:
    """Verify _search_error_msg is protected by _search_error_lock."""

    def test_error_lock_created(self):
        """ProgressiveExtractionWorker.__init__ must create _search_error_lock."""
        import inspect

        from src.services.workers import ProgressiveExtractionWorker

        source = inspect.getsource(ProgressiveExtractionWorker.__init__)
        assert "_search_error_lock" in source
        assert "threading.Lock()" in source

    def test_error_msg_read_under_lock(self):
        """Reading _search_error_msg must be done under _search_error_lock."""
        import inspect

        from src.services.workers import ProgressiveExtractionWorker

        source = inspect.getsource(ProgressiveExtractionWorker._wait_for_search_indexing)
        # The read should be inside a `with self._search_error_lock:` block
        assert "with self._search_error_lock:" in source

    def test_error_msg_write_under_lock(self):
        """Writing _search_error_msg must be done under _search_error_lock."""
        import inspect

        from src.services.workers import ProgressiveExtractionWorker

        source = inspect.getsource(ProgressiveExtractionWorker._build_vector_store)
        assert "with self._search_error_lock:" in source
