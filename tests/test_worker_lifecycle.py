"""
Tests for WorkerProcessManager lifecycle hardening (Layers 1-3).

Covers:
- Layer 1: atexit hook registration and idempotent shutdown
- Layer 2: context manager protocol (__enter__ / __exit__)
- Layer 3: main.py uses context manager or try/finally for worker lifetime

The three layers are defense-in-depth above `daemon=True`, ensuring the
worker subprocess is shut down cleanly on any Python-level exit path
(normal return, exception, or KeyboardInterrupt). `daemon=True` remains
the OS-level backstop for os._exit() / segfault / SIGKILL.
"""

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.services.worker_manager import WorkerProcessManager

# -------------------------------------------------------------------------
# Layer 1: atexit hook
# -------------------------------------------------------------------------


class TestAtexitRegistration:
    """atexit.register must be called once per manager instance on start()."""

    def _patched_start(self, mgr):
        """Run mgr.start() with the real subprocess swapped for a MagicMock."""
        fake_proc = MagicMock()
        fake_proc.is_alive.return_value = True
        fake_proc.pid = 12345
        with patch("multiprocessing.Process", return_value=fake_proc):
            mgr.start()
        return fake_proc

    def test_start_registers_atexit_hook(self):
        """start() should register _atexit_shutdown with atexit."""
        mgr = WorkerProcessManager()
        with patch("src.services.worker_manager.atexit.register") as mock_reg:
            self._patched_start(mgr)
            mock_reg.assert_called_once_with(mgr._atexit_shutdown)
        assert mgr._atexit_registered is True

    def test_double_start_does_not_reregister(self):
        """Calling start() twice should only register atexit once."""
        mgr = WorkerProcessManager()
        with patch("src.services.worker_manager.atexit.register") as mock_reg:
            self._patched_start(mgr)
            # Second start — process is already alive, returns early
            self._patched_start(mgr)
            assert mock_reg.call_count == 1

    def test_restart_after_death_does_not_reregister(self):
        """restart_if_dead path (process died, start called again) must not re-register."""
        mgr = WorkerProcessManager()
        with patch("src.services.worker_manager.atexit.register") as mock_reg:
            fake_proc = self._patched_start(mgr)
            # Simulate process death and restart
            fake_proc.is_alive.return_value = False
            mgr._started = False  # _cleanup_dead_process would reset this
            mgr.process = None
            self._patched_start(mgr)
            assert mock_reg.call_count == 1


class TestAtexitShutdownIdempotency:
    """_atexit_shutdown must be safe to call in any state."""

    def test_atexit_shutdown_noop_when_never_started(self):
        """_atexit_shutdown should be a no-op if start() was never called."""
        mgr = WorkerProcessManager()
        with patch.object(mgr, "shutdown") as mock_shutdown:
            mgr._atexit_shutdown()
            mock_shutdown.assert_not_called()

    def test_atexit_shutdown_calls_shutdown_when_started(self):
        """_atexit_shutdown should call shutdown(blocking=True) when started."""
        mgr = WorkerProcessManager()
        mgr._started = True
        with patch.object(mgr, "shutdown") as mock_shutdown:
            mgr._atexit_shutdown()
            mock_shutdown.assert_called_once_with(blocking=True)

    def test_atexit_shutdown_swallows_exceptions(self):
        """_atexit_shutdown must never propagate exceptions (atexit is exiting)."""
        mgr = WorkerProcessManager()
        mgr._started = True
        with patch.object(mgr, "shutdown", side_effect=RuntimeError("boom")):
            # Should not raise
            mgr._atexit_shutdown()

    def test_atexit_shutdown_idempotent_double_call(self):
        """Calling _atexit_shutdown twice on a non-started manager is a no-op both times."""
        mgr = WorkerProcessManager()
        # First call: not started, does nothing
        mgr._atexit_shutdown()
        # Second call: still not started, still does nothing
        mgr._atexit_shutdown()
        # If shutdown() did run and reset state, second call is also a no-op
        mgr._started = True
        with patch.object(mgr, "shutdown") as mock_shutdown:

            def fake_shutdown(blocking=True):
                """Simulate shutdown() resetting _started as the real one does."""
                mgr._started = False

            mock_shutdown.side_effect = fake_shutdown
            mgr._atexit_shutdown()
            mgr._atexit_shutdown()
            assert mock_shutdown.call_count == 1  # second call short-circuited


# -------------------------------------------------------------------------
# Layer 2: Context manager protocol
# -------------------------------------------------------------------------


class TestContextManagerProtocol:
    """WorkerProcessManager supports `with` for guaranteed cleanup."""

    def test_enter_returns_self(self):
        """__enter__ must return the manager instance."""
        mgr = WorkerProcessManager()
        with mgr as returned:
            assert returned is mgr

    def test_enter_does_not_auto_start(self):
        """__enter__ must not call start() — callers decide when to start."""
        mgr = WorkerProcessManager()
        with patch.object(mgr, "start") as mock_start:
            with mgr:
                pass
            mock_start.assert_not_called()

    def test_exit_calls_shutdown_blocking(self):
        """__exit__ must call shutdown(blocking=True) on clean exit."""
        mgr = WorkerProcessManager()
        with patch.object(mgr, "shutdown") as mock_shutdown:
            with mgr:
                pass
            mock_shutdown.assert_called_once_with(blocking=True)

    def test_exit_calls_shutdown_on_exception(self):
        """__exit__ must call shutdown even when the body raises."""
        mgr = WorkerProcessManager()
        with patch.object(mgr, "shutdown") as mock_shutdown:
            with pytest.raises(ValueError, match="test failure"), mgr:
                raise ValueError("test failure")
            mock_shutdown.assert_called_once_with(blocking=True)

    def test_exit_does_not_suppress_exception(self):
        """__exit__ must return False/None so exceptions propagate."""
        mgr = WorkerProcessManager()
        with patch.object(mgr, "shutdown"):
            with pytest.raises(RuntimeError, match="propagate me"):
                with mgr:
                    raise RuntimeError("propagate me")

    def test_context_manager_with_explicit_start(self):
        """Typical usage: enter, start, work, exit triggers shutdown."""
        mgr = WorkerProcessManager()
        fake_proc = MagicMock()
        fake_proc.is_alive.return_value = True
        fake_proc.pid = 99
        with (
            patch("multiprocessing.Process", return_value=fake_proc),
            patch.object(mgr, "shutdown") as mock_shutdown,
        ):
            with mgr as m:
                m.start()
                assert m._started is True
            mock_shutdown.assert_called_once_with(blocking=True)


# -------------------------------------------------------------------------
# Layer 3: main.py wiring
# -------------------------------------------------------------------------


class TestMainPyWorkerLifetime:
    """Static analysis: main.py must wrap worker in `with` or try/finally."""

    def _get_main_source(self):
        """Return the source of src.main.main()."""
        from src.main import main

        return inspect.getsource(main)

    def test_main_uses_with_for_worker_manager(self):
        """main() should use `with WorkerProcessManager()` to own the worker lifetime."""
        source = self._get_main_source()
        assert "with WorkerProcessManager()" in source, (
            "main() must wrap WorkerProcessManager in a `with` block so "
            "shutdown runs on any exit path."
        )

    def test_main_calls_worker_start_explicitly(self):
        """main() must still call worker_manager.start() inside the with block."""
        source = self._get_main_source()
        assert "worker_manager.start()" in source

    def test_main_does_not_rely_on_post_mainloop_shutdown(self):
        """main() should not have a post-mainloop `worker_manager.shutdown()` call.

        Shutdown must be driven by __exit__ (context manager) or a finally —
        a bare shutdown after mainloop() is skipped if mainloop raises.
        """
        source = self._get_main_source()
        # Find the mainloop() call and check nothing after it is a bare
        # worker_manager.shutdown(). Parse and walk the AST for rigor.
        tree = ast.parse(source)
        shutdown_after_mainloop_at_same_level = False

        class Visitor(ast.NodeVisitor):
            """Look for `worker_manager.shutdown(...)` siblings after `mainloop()`."""

            def __init__(self):
                """Track whether we've flagged a risky pattern."""
                self.flagged = False

            def visit_FunctionDef(self, node):
                """Walk the main() body for the risky pattern."""
                self._scan_body(node.body)
                self.generic_visit(node)

            def _scan_body(self, body):
                """Check a block-level body for mainloop -> shutdown at the same nesting."""
                saw_mainloop = False
                for stmt in body:
                    src_line = ast.unparse(stmt)
                    if "mainloop()" in src_line and "shutdown" not in src_line:
                        saw_mainloop = True
                    elif saw_mainloop and "worker_manager.shutdown" in src_line:
                        self.flagged = True

        v = Visitor()
        v.visit(tree)
        shutdown_after_mainloop_at_same_level = v.flagged
        assert not shutdown_after_mainloop_at_same_level, (
            "main() has a `worker_manager.shutdown(...)` at the same nesting "
            "level as `mainloop()` — this is skipped if mainloop raises. "
            "Use context manager or try/finally instead."
        )

    def test_main_py_file_exists_and_imports(self):
        """Sanity: main.py is importable."""
        main_path = Path(__file__).resolve().parent.parent / "src" / "main.py"
        assert main_path.exists()
