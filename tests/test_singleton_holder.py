"""Tests for src/singleton.py — thread-safe SingletonHolder.

SingletonHolder consolidates the double-checked-locking pattern previously
duplicated across user_preferences and ExportService. Behavior under test:
- Lazy creation: factory runs only on first get()
- Caching: subsequent get() returns the same instance
- Factory arguments: forwarded on first call, ignored afterwards
- reset(): clears cache so next get() creates a fresh instance
- Thread safety: under concurrent access the factory runs exactly once
"""

import threading

from src.singleton import SingletonHolder


class TestLazyCreation:
    """The factory is invoked only when get() is first called."""

    def test_factory_not_called_before_get(self):
        """Instantiating SingletonHolder does not call the factory."""
        calls: list[int] = []

        def factory():
            calls.append(1)
            return object()

        SingletonHolder(factory)
        assert calls == []

    def test_factory_called_once_on_first_get(self):
        """First call to get() invokes the factory exactly once."""
        calls: list[int] = []

        def factory():
            calls.append(1)
            return object()

        holder = SingletonHolder(factory)
        holder.get()
        assert calls == [1]


class TestCaching:
    """get() returns the same instance on every call after the first."""

    def test_same_instance_on_repeated_get(self):
        """Two get() calls return the same cached instance."""
        holder = SingletonHolder(object)
        first = holder.get()
        second = holder.get()
        assert first is second

    def test_factory_called_only_once_across_multiple_gets(self):
        """Factory runs exactly once regardless of how many times get() is invoked."""
        calls: list[int] = []

        def factory():
            calls.append(1)
            return object()

        holder = SingletonHolder(factory)
        for _ in range(10):
            holder.get()
        assert len(calls) == 1


class TestFactoryArguments:
    """Arguments passed to get() are forwarded to the factory on first call only."""

    def test_args_forwarded_on_first_get(self):
        """Positional and keyword args reach the factory on initial creation."""
        captured = {}

        def factory(a, b, key=None):
            captured["a"] = a
            captured["b"] = b
            captured["key"] = key
            return object()

        holder = SingletonHolder(factory)
        holder.get(1, 2, key="abc")
        assert captured == {"a": 1, "b": 2, "key": "abc"}

    def test_args_ignored_on_subsequent_gets(self):
        """Args after first call are ignored (factory does not run again)."""
        calls: list[tuple] = []

        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            return object()

        holder = SingletonHolder(factory)
        first = holder.get(1, 2)
        second = holder.get(99, 99, extra="ignored")
        assert first is second
        assert len(calls) == 1
        # The captured args are from the FIRST call, not the second
        assert calls[0] == ((1, 2), {})


class TestReset:
    """reset() discards the cached instance so the factory runs again."""

    def test_reset_clears_cached_instance(self):
        """After reset(), get() calls the factory once more."""
        calls: list[int] = []

        def factory():
            calls.append(1)
            return object()

        holder = SingletonHolder(factory)
        holder.get()
        holder.reset()
        holder.get()
        assert len(calls) == 2

    def test_reset_produces_new_instance(self):
        """The instance returned after reset() differs from the one before."""
        holder = SingletonHolder(object)
        before = holder.get()
        holder.reset()
        after = holder.get()
        assert before is not after

    def test_reset_on_fresh_holder_does_not_raise(self):
        """Calling reset() before any get() is a no-op and does not raise."""
        holder = SingletonHolder(object)
        # Should not raise
        holder.reset()
        # Subsequent get still works
        assert holder.get() is not None


class TestThreadSafety:
    """Concurrent get() calls must still produce exactly one factory invocation."""

    def test_factory_runs_once_under_contention(self):
        """Under N concurrent get() calls, factory executes exactly once.

        Threads wait at a barrier before the get() call itself so they
        all race into the holder simultaneously. Only one thread should
        win the initialization race; the rest should observe the cached
        instance.
        """
        call_count = 0
        count_lock = threading.Lock()

        def slow_factory():
            nonlocal call_count
            # Sleep to widen the window for concurrent access
            import time

            time.sleep(0.05)
            with count_lock:
                call_count += 1
            return object()

        num_threads = 10
        start_barrier = threading.Barrier(num_threads)
        holder = SingletonHolder(slow_factory)
        results: list[object] = []
        results_lock = threading.Lock()

        def worker():
            # Ensure all threads reach holder.get() at roughly the same moment.
            start_barrier.wait()
            instance = holder.get()
            with results_lock:
                results.append(instance)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count == 1
        # All threads got the same instance
        assert len({id(r) for r in results}) == 1
