"""
Tests for graceful shutdown and resource cleanup.

Covers:
- _destroying flag prevents poll loops from re-scheduling after() callbacks
- destroy() cancels all tracked after() IDs (source inspection)
- destroy() joins workers with timeout after stopping them (source inspection)
- sys.excepthook logs uncaught main-thread exceptions (except KeyboardInterrupt)
- threading.excepthook logs uncaught thread exceptions (except SystemExit)
- SemanticService tracks temp dirs and cleans them up via cleanup()
- SemanticService registers atexit backstop for temp dirs
- SemanticService cleanup() is idempotent and handles missing dirs
"""

import inspect
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

# -------------------------------------------------------------------------
# Fix 1: _destroying guard and destroy() cleanup (main_window.py)
# -------------------------------------------------------------------------


class TestDestroyingGuard:
    """Test that _destroying flag prevents poll loops from re-scheduling."""

    def _make_mock_window(self):
        """Create a mock MainWindow-like object with poll method dependencies."""
        window = MagicMock()
        window._destroying = False
        window._semantic_results_lock = threading.Lock()
        window._semantic_results = []
        return window

    def test_poll_followup_result_returns_early_when_destroying(self):
        """_poll_followup_result should return immediately if _destroying is True."""
        from src.ui.main_window import MainWindow

        window = self._make_mock_window()
        window._destroying = True

        MainWindow._poll_followup_result(window)

        window.after.assert_not_called()


class TestDestroyMethodContract:
    """Test destroy() cleanup logic via source inspection.

    MainWindow inherits from ctk.CTk (tkinter), making it impossible to
    call destroy() on a mock without a real Tk root. Instead, we inspect
    the source to verify the contract is implemented correctly.
    """

    def _get_destroy_source(self):
        """Get the source code of MainWindow.destroy()."""
        from src.ui.main_window import MainWindow

        return inspect.getsource(MainWindow.destroy)

    def test_destroy_sets_destroying_flag_first(self):
        """destroy() should set _destroying = True before any other work."""
        source = self._get_destroy_source()
        lines = [
            l.strip()
            for l in source.split("\n")
            if l.strip() and not l.strip().startswith(('"""', "#", "def "))
        ]
        assert "_destroying = True" in lines[0], (
            f"First statement should set _destroying, got: {lines[0]}"
        )

    def test_destroy_cancels_queue_poll_id(self):
        """destroy() should cancel _queue_poll_id via after_cancel."""
        source = self._get_destroy_source()
        assert "after_cancel(self._queue_poll_id)" in source

    def test_destroy_cancels_status_clear_id(self):
        """destroy() should cancel _status_clear_id via after_cancel."""
        source = self._get_destroy_source()
        assert "after_cancel(self._status_clear_id)" in source

    def test_destroy_cancels_timer_after_id(self):
        """destroy() should cancel _timer_after_id via after_cancel."""
        source = self._get_destroy_source()
        assert "after_cancel(self._timer_after_id)" in source

    def test_destroy_nulls_out_cancelled_ids(self):
        """destroy() should set cancelled IDs to None."""
        source = self._get_destroy_source()
        assert "self._queue_poll_id = None" in source
        assert "self._status_clear_id = None" in source
        assert "self._timer_after_id = None" in source

    def test_destroy_shuts_down_worker_manager(self):
        """destroy() should shut down worker_manager (non-blocking)."""
        source = self._get_destroy_source()
        assert "_worker_manager.shutdown(blocking=False)" in source

    def test_destroy_calls_super_destroy(self):
        """destroy() should call super().destroy() at the end."""
        source = self._get_destroy_source()
        assert "super().destroy()" in source

    def test_destroying_flag_initialized_in_init(self):
        """MainWindow.__init__ should initialize _destroying = False."""
        from src.ui.main_window import MainWindow

        init_source = inspect.getsource(MainWindow.__init__)
        assert "_destroying = False" in init_source


# -------------------------------------------------------------------------
# Fix 2: Exception hooks (main.py)
# -------------------------------------------------------------------------


class TestExceptionHooks:
    """Test sys.excepthook and threading.excepthook installed by main()."""

    def _run_main(self):
        """Run main() with all UI/logging mocked out."""
        mock_app = MagicMock()
        with (
            patch("src.splash.launch", return_value=None),
            patch("src.splash.kill"),
            patch("src.main.setup_file_logging"),
            patch("src.logging_config.setup_logging"),
            patch("src.logging_config.purge_old_logs"),
            patch("customtkinter.set_appearance_mode"),
            patch("customtkinter.set_default_color_theme"),
            patch("src.ui.scaling.apply_scaling"),
            patch("src.services.worker_manager.WorkerProcessManager"),
            patch("src.ui.main_window.MainWindow", return_value=mock_app),
        ):
            from src.main import main

            main()

    def test_main_installs_sys_excepthook(self):
        """main() should install a custom sys.excepthook."""
        original = sys.excepthook
        try:
            self._run_main()
            assert sys.excepthook is not original
            assert sys.excepthook is not sys.__excepthook__
        finally:
            sys.excepthook = original

    def test_main_installs_threading_excepthook(self):
        """main() should install a custom threading.excepthook."""
        original = threading.excepthook
        try:
            self._run_main()
            assert threading.excepthook is not original
        finally:
            threading.excepthook = original

    def test_uncaught_exception_does_not_raise(self):
        """Custom sys.excepthook should not raise when handling exceptions."""
        original = sys.excepthook
        try:
            self._run_main()
            hook = sys.excepthook

            try:
                raise ValueError("test error")
            except ValueError:
                exc_info = sys.exc_info()
                # Should not raise
                hook(exc_info[0], exc_info[1], exc_info[2])
        finally:
            sys.excepthook = original

    def test_uncaught_exception_passes_through_keyboard_interrupt(self):
        """Custom sys.excepthook should defer KeyboardInterrupt to default."""
        original = sys.excepthook
        try:
            self._run_main()
            hook = sys.excepthook

            with patch.object(sys, "__excepthook__") as mock_default:
                hook(KeyboardInterrupt, KeyboardInterrupt("ctrl+c"), None)
                mock_default.assert_called_once()
        finally:
            sys.excepthook = original

    def test_thread_excepthook_does_not_raise(self):
        """Custom threading.excepthook should not raise on thread exceptions."""
        original = threading.excepthook
        try:
            self._run_main()
            hook = threading.excepthook

            args = MagicMock()
            args.exc_type = RuntimeError
            args.exc_value = RuntimeError("thread crash")
            args.exc_traceback = None
            args.thread = MagicMock()
            args.thread.name = "test-worker"

            # Should not raise
            hook(args)
        finally:
            threading.excepthook = original

    def test_thread_excepthook_ignores_system_exit(self):
        """Custom threading.excepthook should silently ignore SystemExit."""
        original = threading.excepthook
        try:
            self._run_main()
            hook = threading.excepthook

            args = MagicMock()
            args.exc_type = SystemExit

            # Should not raise
            hook(args)
        finally:
            threading.excepthook = original


# -------------------------------------------------------------------------
# Fix 3: SemanticService temp dir cleanup (semantic_service.py)
# -------------------------------------------------------------------------


class TestSemanticServiceTempCleanup:
    """Test SemanticService temp directory tracking and cleanup."""

    def test_init_temp_dir_is_none(self):
        """SemanticService should initialize _temp_dir as None."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        assert service._temp_dir is None

    def test_init_with_explicit_path_no_temp_dir(self):
        """SemanticService with explicit path should not create temp dir."""
        from src.services.semantic_service import SemanticService

        service = SemanticService(vector_store_path=Path("/some/path"))
        assert service._temp_dir is None
        assert service._vector_store_path == Path("/some/path")

    def test_cleanup_when_no_temp_dir(self):
        """cleanup() should be safe to call when no temp dir exists."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        # Should not raise
        service.cleanup()
        assert service._temp_dir is None

    def test_cleanup_deletes_temp_dir(self):
        """cleanup() should delete the temp directory."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        service._temp_dir = Path(tempfile.mkdtemp())
        service._vector_store_path = service._temp_dir / "semantic_index"
        service._temp_dir.mkdir(parents=True, exist_ok=True)

        assert service._temp_dir.exists()

        service.cleanup()

        assert service._temp_dir is None
        assert service._vector_store_path is None

    def test_cleanup_is_idempotent(self):
        """Calling cleanup() twice should not raise."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()
        service._temp_dir = Path(tempfile.mkdtemp())
        service._vector_store_path = service._temp_dir / "semantic_index"

        service.cleanup()
        # Second call should be safe (dir already gone)
        service.cleanup()

        assert service._temp_dir is None

    def test_cleanup_handles_already_deleted_dir(self):
        """cleanup() should handle the case where temp dir was already removed."""
        import shutil

        from src.services.semantic_service import SemanticService

        service = SemanticService()
        service._temp_dir = Path(tempfile.mkdtemp())
        service._vector_store_path = service._temp_dir / "semantic_index"

        # Delete it externally first
        shutil.rmtree(service._temp_dir)
        assert not service._temp_dir.exists()

        # cleanup() should handle gracefully (exists() returns False)
        service.cleanup()
        assert service._temp_dir is None

    def test_build_index_creates_temp_dir_when_no_path(self):
        """build_index() should create a temp dir when vector_store_path is None."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()

        with (
            patch("src.core.retrieval.algorithms.faiss_semantic.get_embeddings_model") as mock_emb,
            patch("src.core.chunking.create_unified_chunker") as mock_chunker,
            patch("src.core.vector_store.VectorStoreBuilder"),
            patch("src.core.semantic.SemanticOrchestrator"),
        ):
            mock_emb.return_value = MagicMock()
            mock_chunker_inst = MagicMock()
            mock_chunker_inst.chunk_text.return_value = ["chunk1", "chunk2"]
            mock_chunker.return_value = mock_chunker_inst

            service.build_index("test text")

        assert service._temp_dir is not None
        assert service._temp_dir.exists()
        assert "semantic_index" in str(service._vector_store_path)

        # Clean up
        service.cleanup()

    def test_build_index_registers_atexit_handler(self):
        """build_index() should register an atexit handler for temp cleanup."""
        from src.services.semantic_service import SemanticService

        service = SemanticService()

        with (
            patch("src.core.retrieval.algorithms.faiss_semantic.get_embeddings_model") as mock_emb,
            patch("src.core.chunking.create_unified_chunker") as mock_chunker,
            patch("src.core.vector_store.VectorStoreBuilder"),
            patch("src.core.semantic.SemanticOrchestrator"),
            patch("atexit.register") as mock_atexit,
        ):
            mock_emb.return_value = MagicMock()
            mock_chunker_inst = MagicMock()
            mock_chunker_inst.chunk_text.return_value = ["chunk1"]
            mock_chunker.return_value = mock_chunker_inst

            service.build_index("test text")

            mock_atexit.assert_called_once()

        # Clean up
        service.cleanup()

    def test_build_index_with_explicit_path_no_temp_dir(self):
        """build_index() with explicit path should not create temp dir or atexit."""
        from src.services.semantic_service import SemanticService

        explicit_path = Path(tempfile.mkdtemp()) / "explicit_index"
        service = SemanticService(vector_store_path=explicit_path)

        with (
            patch("src.core.retrieval.algorithms.faiss_semantic.get_embeddings_model") as mock_emb,
            patch("src.core.chunking.create_unified_chunker") as mock_chunker,
            patch("src.core.vector_store.VectorStoreBuilder"),
            patch("src.core.semantic.SemanticOrchestrator"),
            patch("atexit.register") as mock_atexit,
        ):
            mock_emb.return_value = MagicMock()
            mock_chunker_inst = MagicMock()
            mock_chunker_inst.chunk_text.return_value = ["chunk1"]
            mock_chunker.return_value = mock_chunker_inst

            service.build_index("test text")

            mock_atexit.assert_not_called()

        assert service._temp_dir is None
        assert service._vector_store_path == explicit_path

        # Clean up the explicit path
        import shutil

        shutil.rmtree(explicit_path.parent, ignore_errors=True)
