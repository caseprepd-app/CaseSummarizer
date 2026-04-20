"""Tests for utility functions in src/logging_config.py.

Covers previously-untested helpers:
- Timer: context manager that measures and auto-logs elapsed time
- get_log_file_path: resolves caseprepd.log under LOGS_DIR
- get_log_file_size_mb: returns file size in MB (0.0 when missing)
- clear_log_file: truncates the log and reinstates the rotating handler
"""

import logging
import time
from pathlib import Path
from unittest.mock import patch


class TestTimerContextManager:
    """Timer measures elapsed time when used as a context manager."""

    def test_enter_returns_timer_instance(self):
        """`with Timer(...)` yields the Timer object itself."""
        from src.logging_config import Timer

        with Timer("op", auto_log=False) as t:
            assert isinstance(t, Timer)

    def test_duration_measured_after_exit(self):
        """duration_ms is populated on exit with a non-negative number."""
        from src.logging_config import Timer

        with Timer("op", auto_log=False) as t:
            pass
        assert t.duration_ms is not None
        assert t.duration_ms >= 0

    def test_duration_reflects_sleep(self):
        """Measured duration should be at least the slept interval."""
        from src.logging_config import Timer

        sleep_ms = 30
        with Timer("op", auto_log=False) as t:
            time.sleep(sleep_ms / 1000.0)
        # Allow wide tolerance for Windows scheduler jitter
        assert t.duration_ms is not None
        assert t.duration_ms >= sleep_ms * 0.5

    def test_get_duration_ms_raises_before_exit(self):
        """Calling get_duration_ms() before __exit__ raises ValueError."""
        from src.logging_config import Timer

        t = Timer("op", auto_log=False)
        try:
            t.get_duration_ms()
        except ValueError:
            return
        raise AssertionError("Expected ValueError before timer completes")

    def test_get_duration_ms_returns_duration_after_exit(self):
        """After exit, get_duration_ms() returns the recorded duration."""
        from src.logging_config import Timer

        with Timer("op", auto_log=False) as t:
            pass
        assert t.get_duration_ms() == t.duration_ms

    def test_exception_does_not_suppress(self):
        """Timer does not swallow exceptions from the with-block."""
        from src.logging_config import Timer

        class Sentinel(RuntimeError):
            pass

        try:
            with Timer("op", auto_log=False):
                raise Sentinel("boom")
        except Sentinel:
            return
        raise AssertionError("Timer should re-raise the original exception")

    def test_operation_name_stored(self):
        """Timer remembers the operation name it was constructed with."""
        from src.logging_config import Timer

        t = Timer("parsing-pdfs", auto_log=False)
        assert t.operation_name == "parsing-pdfs"

    def test_auto_log_emits_debug_messages(self):
        """With auto_log=True, start and end messages are logged at DEBUG."""
        from src.logging_config import Timer

        logger = logging.getLogger("src.logging_config")
        with patch.object(logger, "debug") as mock_debug, Timer("op", auto_log=True):
            pass
        # Expect start + end log calls
        assert mock_debug.call_count >= 2

    def test_auto_log_false_skips_logs(self):
        """With auto_log=False, Timer emits no debug calls."""
        from src.logging_config import Timer

        logger = logging.getLogger("src.logging_config")
        with patch.object(logger, "debug") as mock_debug, Timer("op", auto_log=False):
            pass
        assert mock_debug.call_count == 0


class TestGetLogFilePath:
    """get_log_file_path returns the expected filename under LOGS_DIR."""

    def test_filename_is_caseprepd_log(self):
        """Log path ends with the expected filename."""
        from src.logging_config import get_log_file_path

        assert get_log_file_path().name == "caseprepd.log"

    def test_returns_path_object(self):
        """Function returns a pathlib.Path."""
        from src.logging_config import get_log_file_path

        assert isinstance(get_log_file_path(), Path)

    def test_respects_monkeypatched_logs_dir(self, tmp_path, monkeypatch):
        """When LOGS_DIR is patched, the returned path lives under it."""
        monkeypatch.setattr("src.config.LOGS_DIR", tmp_path)
        from src.logging_config import get_log_file_path

        assert get_log_file_path() == tmp_path / "caseprepd.log"


class TestGetLogFileSizeMb:
    """get_log_file_size_mb reports file size correctly."""

    def test_returns_zero_when_file_missing(self, tmp_path, monkeypatch):
        """Missing log file yields 0.0 MB."""
        monkeypatch.setattr("src.config.LOGS_DIR", tmp_path)
        from src.logging_config import get_log_file_size_mb

        # Ensure no file exists
        (tmp_path / "caseprepd.log").unlink(missing_ok=True)
        assert get_log_file_size_mb() == 0.0

    def test_returns_size_in_megabytes(self, tmp_path, monkeypatch):
        """File of N bytes reports N / (1024**2) MB."""
        monkeypatch.setattr("src.config.LOGS_DIR", tmp_path)
        log_path = tmp_path / "caseprepd.log"
        # Write exactly 2 MiB of ASCII
        size_bytes = 2 * 1024 * 1024
        log_path.write_bytes(b"a" * size_bytes)

        from src.logging_config import get_log_file_size_mb

        result = get_log_file_size_mb()
        # Compare via reasoned math, not by re-running the same division
        expected = size_bytes / (1024 * 1024)
        assert result == expected

    def test_returns_float_type(self, tmp_path, monkeypatch):
        """Return type is always float."""
        monkeypatch.setattr("src.config.LOGS_DIR", tmp_path)
        (tmp_path / "caseprepd.log").write_text("hi", encoding="utf-8")

        from src.logging_config import get_log_file_size_mb

        assert isinstance(get_log_file_size_mb(), float)


class TestClearLogFile:
    """clear_log_file truncates the file and returns success flag."""

    def test_truncates_existing_file(self, tmp_path, monkeypatch):
        """After clear, the log file exists but is empty."""
        # Patch LOGS_DIR in config module so logging_config reads it
        monkeypatch.setattr("src.config.LOGS_DIR", tmp_path)

        log_path = tmp_path / "caseprepd.log"
        log_path.write_text("old content that should be gone", encoding="utf-8")

        # Clear module-level handler so clear_log_file creates a fresh one
        import src.logging_config as lc

        monkeypatch.setattr(lc, "_file_handler", None)

        success = lc.clear_log_file()

        assert success is True
        assert log_path.exists()
        assert log_path.read_text(encoding="utf-8") == ""

        # Cleanup: remove the handler this test created so it doesn't keep
        # the file open and interfere with other tests
        import logging

        root = logging.getLogger("src")
        if lc._file_handler is not None:
            root.removeHandler(lc._file_handler)
            lc._file_handler.close()
            lc._file_handler = None

    def test_returns_bool(self, tmp_path, monkeypatch):
        """clear_log_file returns a boolean result."""
        monkeypatch.setattr("src.config.LOGS_DIR", tmp_path)
        (tmp_path / "caseprepd.log").write_text("x", encoding="utf-8")

        import src.logging_config as lc

        monkeypatch.setattr(lc, "_file_handler", None)

        result = lc.clear_log_file()
        assert isinstance(result, bool)

        # Cleanup
        import logging

        root = logging.getLogger("src")
        if lc._file_handler is not None:
            root.removeHandler(lc._file_handler)
            lc._file_handler.close()
            lc._file_handler = None
