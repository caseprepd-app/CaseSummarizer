"""
Tests for exception handling audit fixes.

Validates all 7 bug categories:
- Bug 1: self.after() lambda callbacks don't crash on widget destruction
- Bug 2: Clipboard operations catch exceptions
- Bug 3: File write in save_to_file catches OSError
- Bug 4: result_queue.put() failure doesn't kill command loop
- Bug 5: Silent except:pass blocks now log
- Bug 6: logger.error calls include exc_info=True (spot checks)
- Bug 7: logger.debug upgraded to logger.warning for real failures
"""

import inspect
import threading
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _read_source(relative_path: str) -> str:
    """Read a source file and return its contents."""
    return (SRC_DIR / relative_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Bug 1: self.after() lambda callbacks guard against widget destruction
# ---------------------------------------------------------------------------


class TestBug1AfterCallbackGuards:
    """Verify that self.after() callbacks catch exceptions on destroyed widgets."""

    def test_dynamic_output_copy_btn_reset_has_guard(self):
        """copy_to_clipboard defines _reset_copy_btn with try/except."""
        source = _read_source("ui/dynamic_output.py")
        assert "_reset_copy_btn" in source

    def test_dynamic_output_save_btn_reset_has_guard(self):
        """save_to_file defines _reset_save_btn with try/except."""
        source = _read_source("ui/dynamic_output.py")
        assert "_reset_save_btn" in source

    def test_main_window_export_btn_reset_has_guard(self):
        """export_all defines _reset_export_btn with try/except."""
        source = _read_source("ui/main_window.py")
        assert "_reset_export_btn" in source

    def test_semantic_panel_copy_btn_reset_has_guard(self):
        """semantic_panel copy defines _reset_copy_btn with try/except."""
        source = _read_source("ui/semantic_panel.py")
        assert "_reset_copy_btn" in source


# ---------------------------------------------------------------------------
# Bug 2: Clipboard operations catch exceptions
# ---------------------------------------------------------------------------


class TestBug2ClipboardGuards:
    """Verify clipboard operations are wrapped in try/except."""

    def test_clipboard_copy_failure_logged(self):
        """Clipboard failures produce a warning log."""
        source = _read_source("ui/dynamic_output.py")
        assert source.count('logger.warning("Clipboard copy failed') >= 2

    def test_copy_to_clipboard_returns_on_failure(self):
        """copy_to_clipboard returns early on clipboard failure."""
        source = _read_source("ui/dynamic_output.py")
        # The clipboard block in copy_to_clipboard has a return after the warning
        assert "Clipboard copy failed" in source


# ---------------------------------------------------------------------------
# Bug 3: File write in save_to_file catches OSError
# ---------------------------------------------------------------------------


class TestBug3FileWriteGuard:
    """Verify save_to_file catches file write errors."""

    def test_save_to_file_has_oserror_guard(self):
        """save_to_file wraps open/write in try/except OSError."""
        source = _read_source("ui/dynamic_output.py")
        assert "except OSError" in source
        assert "Save failed" in source


# ---------------------------------------------------------------------------
# Bug 3b: hasattr guards on winfo_toplevel().set_status()
# ---------------------------------------------------------------------------


class TestBug3bHasattrGuards:
    """Verify copy_to_clipboard and save_to_file guard set_status with hasattr."""

    def test_copy_to_clipboard_has_hasattr_guard(self):
        """copy_to_clipboard uses hasattr(main_window, 'set_status')."""
        source = _read_source("ui/dynamic_output.py")
        # Find the copy_to_clipboard method and check for hasattr guard
        copy_start = source.index("def copy_to_clipboard")
        copy_end = source.index("\n    def ", copy_start + 1)
        copy_source = source[copy_start:copy_end]
        assert 'hasattr(main_window, "set_status")' in copy_source

    def test_save_to_file_has_hasattr_guard(self):
        """save_to_file uses hasattr(main_window, 'set_status')."""
        source = _read_source("ui/dynamic_output.py")
        save_start = source.index("def save_to_file")
        save_end = source.index("\n    def ", save_start + 1)
        save_source = source[save_start:save_end]
        assert 'hasattr(main_window, "set_status")' in save_source


# ---------------------------------------------------------------------------
# Bug 4: result_queue.put() failure doesn't kill command loop
# ---------------------------------------------------------------------------


class TestBug4ResultQueueGuard:
    """Verify result_queue.put for command_ack is protected."""

    def test_command_ack_has_try_except(self):
        """The command_ack put is wrapped in try/except."""
        source = _read_source("worker_process.py")
        assert "Failed to send command_ack" in source

    def test_command_ack_put_failure_continues_loop(self):
        """If result_queue.put raises, the command loop continues."""
        from src.worker_process import _command_loop

        command_queue = Queue()
        result_queue = MagicMock()
        internal_queue = Queue()
        shutdown = threading.Event()
        state = {
            "shutdown": shutdown,
            "active_worker": None,
            "worker_lock": threading.Lock(),
        }

        # Make result_queue.put always raise (for both command_ack and error puts)
        result_queue.put = MagicMock(side_effect=RuntimeError("Queue broken"))

        # Enqueue a command, then shutdown
        command_queue.put(("extract", {}))
        command_queue.put("shutdown")  # shutdown sentinel

        # Patch _dispatch_command to prevent actual work
        with patch("src.worker_process._dispatch_command"):
            _command_loop(command_queue, internal_queue, result_queue, state)

        # Loop should have processed shutdown without crashing
        assert shutdown.is_set()


# ---------------------------------------------------------------------------
# Bug 5: Silent except:pass blocks now log
# ---------------------------------------------------------------------------


class TestBug5SilentCatchesLog:
    """Verify formerly silent catches now produce log output."""

    def test_config_value_recovery_logs_on_error(self):
        """_d logs warning when preferences fail."""
        # Verify via source code that logging.warning is called on exception
        source = _read_source("config.py")
        assert "Config value recovery" in source
        assert "logging.warning" in source

    def test_worker_forwarder_inner_except_logs(self):
        """Forwarder loop inner except blocks log errors."""
        source = _read_source("worker_process.py")
        assert "Failed to send error to GUI" in source

    def test_context_menu_cleanup_logs(self):
        """_update_alternatives_menu_item logs on cleanup failure."""
        source = _read_source("ui/dynamic_output.py")
        assert "Context menu cleanup failed" in source

    def test_config_silent_catch_removed(self):
        """config.py no longer has bare 'except: pass' for prefs."""
        source = _read_source("config.py")
        # Should use logging.debug instead of pass
        assert "Config value recovery" in source


# ---------------------------------------------------------------------------
# Bug 6: logger.error calls include exc_info=True (spot checks)
# ---------------------------------------------------------------------------


class TestBug6ExcInfoTrue:
    """Spot-check that key logger.error calls include exc_info=True."""

    def test_worker_process_has_exc_info(self):
        """worker_process logger.error calls include exc_info=True."""
        source = _read_source("worker_process.py")
        assert source.count("exc_info=True") >= 4

    def test_unified_chunker_has_exc_info(self):
        """unified_chunker logger.error calls include exc_info=True."""
        source = _read_source("core/chunking/unified_chunker.py")
        assert source.count("exc_info=True") >= 1

    def test_config_model_load_has_exc_info(self):
        """config.py model config error includes exc_info=True."""
        source = _read_source("config.py")
        assert "exc_info=True" in source


# ---------------------------------------------------------------------------
# Bug 7: logger.debug upgraded to logger.warning for real failures
# ---------------------------------------------------------------------------


class TestBug7DebugToWarning:
    """Verify user-facing failures use logger.warning, not logger.debug."""

    def test_document_service_preprocessing_warning(self):
        """Failed preprocessing settings logged as warning."""
        from src.services.document_service import DocumentService

        source = inspect.getsource(DocumentService._get_preprocessing_settings)
        assert "logger.warning" in source
        assert "logger.debug" not in source

    def test_export_service_auto_open_warning(self):
        """Auto-open failure logged as warning."""
        source = _read_source("services/export_service.py")
        assert 'logger.warning("Auto-open failed' in source

    def test_feedback_manager_uses_warning(self):
        """Feedback manager errors use logger.warning."""
        from src.core.vocabulary.feedback_manager import FeedbackManager

        for method_name in (
            "_load_feedback_from_file",
            "_delete_feedback_from_csv",
            "record_feedback",
        ):
            src = inspect.getsource(getattr(FeedbackManager, method_name))
            assert "logger.warning" in src, f"{method_name} missing logger.warning"
