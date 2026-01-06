"""
Unified Logging Configuration for LocalScribe

This module provides a centralized logging system that combines:
- Console output with timestamps
- File output to debug_flow.txt (for debugging sessions)
- Optional file output to logs/processing.log (for production)
- Performance timing via Timer context manager

All modules should import logging functions from this module:
    from src.logging_config import debug_log, info, warning, error, Timer

The module respects DEBUG_MODE from config:
- DEBUG_MODE=True: All messages shown on console, verbose timing
- DEBUG_MODE=False: Only warnings/errors shown on console

Log Levels:
- debug_log(): Always writes to file; console only in DEBUG_MODE
- info(): Standard information messages
- warning(): Warning messages (always shown)
- error(): Error messages with optional exception info
- critical(): Critical errors (always shown with traceback)
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import DEBUG_MODE, LOG_DATE_FORMAT, LOG_FILE, LOG_FORMAT

# =============================================================================
# File Logger Setup (debug_flow.txt for debugging sessions)
# =============================================================================


class _DebugFileLogger:
    """
    Manages the debug_flow.txt file for detailed debugging output.

    This singleton writes all debug messages to a file regardless of DEBUG_MODE,
    providing a complete audit trail for troubleshooting.
    """

    _instance = None
    _log_file = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize_log_file()
        return cls._instance

    @classmethod
    def _initialize_log_file(cls):
        """Create and initialize the debug log file."""
        log_path = Path(__file__).parent.parent / "debug_flow.txt"
        cls._log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        cls._log_file.write("=== LocalScribe Debug Log ===\n")
        cls._log_file.write(f"Started: {datetime.now().isoformat()}\n")
        cls._log_file.write(f"DEBUG_MODE: {DEBUG_MODE}\n")
        cls._log_file.write("=" * 60 + "\n\n")
        cls._log_file.flush()

    def write(self, message: str):
        """Write message to the debug log file."""
        if self._log_file:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            formatted = f"[{timestamp}] {message}"
            self._log_file.write(formatted + "\n")
            self._log_file.flush()

    def close(self):
        """Close the debug log file gracefully."""
        if self._log_file:
            self._log_file.write(f"\n{'=' * 60}\n")
            self._log_file.write(f"Ended: {datetime.now().isoformat()}\n")
            self._log_file.close()
            self._log_file = None


# Global debug file logger instance
_debug_file_logger = _DebugFileLogger()


# =============================================================================
# Standard Python Logging Setup
# =============================================================================


def _setup_standard_logging() -> logging.Logger:
    """
    Configure the standard Python logging framework.

    Returns:
        Configured logger instance for LocalScribe
    """
    # Create logger
    logger = logging.getLogger("LocalScribe")
    logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # File handler (always active for production logs)
    try:
        # SEC-002: Create log directory if it doesn't exist
        log_dir = LOG_FILE.parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(file_handler)
    except Exception as e:
        # SEC-002: Log error to stderr instead of silent pass
        print(f"[WARNING] Could not set up file logging: {e}", file=sys.stderr)
        print("[WARNING] Falling back to console-only logging", file=sys.stderr)

    # Console handler (respects DEBUG_MODE)
    if DEBUG_MODE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(console_handler)

    return logger


# Global standard logger instance
_logger = _setup_standard_logging()


# =============================================================================
# Timer Context Manager
# =============================================================================


class Timer:
    """
    Context manager for timing code blocks with automatic logging.

    Implements performance timing as required by CLAUDE.md debug mode guidelines.
    Timing results are always logged to debug_flow.txt, with console output
    controlled by DEBUG_MODE.

    Usage:
        with Timer("FileParsing"):
            # code to time
            pass

    Output (DEBUG_MODE=True):
        [DEBUG 14:32:01] Starting FileParsing...
        [DEBUG 14:32:01] FileParsing took 842 ms

    Attributes:
        operation_name: Name of the operation being timed
        duration_ms: Duration in milliseconds (available after exit)
    """

    def __init__(self, operation_name: str, auto_log: bool = True):
        """
        Initialize the timer.

        Args:
            operation_name: Descriptive name for the operation
            auto_log: If True, automatically log start/end (respects DEBUG_MODE for console)
        """
        self.operation_name = operation_name
        self.auto_log = auto_log
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.duration_ms: float | None = None

    def __enter__(self):
        if self.auto_log:
            debug_log(f"Starting {self.operation_name}...")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

        if self.auto_log:
            # Format duration in human-readable units
            if self.duration_ms < 1000:
                duration_str = f"{self.duration_ms:.0f} ms"
            else:
                duration_str = f"{self.duration_ms / 1000:.1f} seconds"

            debug_log(f"{self.operation_name} took {duration_str}")

        return False  # Don't suppress exceptions

    def get_duration_ms(self) -> float:
        """
        Get the measured duration in milliseconds.

        Returns:
            Duration in milliseconds

        Raises:
            ValueError: If timer has not completed yet
        """
        if self.duration_ms is None:
            raise ValueError("Timer has not been completed yet")
        return self.duration_ms


# =============================================================================
# Public Logging Functions
# =============================================================================


def debug_log(message: str):
    """
    Log a debug message to both file and console (if DEBUG_MODE).

    This is the primary debug function for LocalScribe. It always writes to
    debug_flow.txt for troubleshooting, and optionally to console based on
    DEBUG_MODE setting.

    Args:
        message: The message to log (prefix with [MODULE] for clarity)

    Example:
        debug_log("[VOCAB] Loading spaCy model...")
        debug_log("[PROCESSOR] Processing 5 documents with max 4 concurrent")
    """
    # Always write to debug file
    _debug_file_logger.write(message)

    # Write to console only in DEBUG_MODE
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {message}"
        try:
            print(formatted)
            sys.stdout.flush()
        except UnicodeEncodeError:
            # Handle Windows console encoding issues
            try:
                sys.stdout.buffer.write((formatted + "\n").encode("utf-8", errors="replace"))
                sys.stdout.buffer.flush()
            except Exception:
                pass  # Skip console if all else fails


def debug(message: str):
    """
    Log a debug-level message (alias for debug_log).

    Provided for backward compatibility with code using utils/logger.debug().

    Args:
        message: The message to log
    """
    debug_log(message)


def info(message: str):
    """
    Log an informational message.

    Info messages are written to the log file and console (when DEBUG_MODE=True).

    Args:
        message: The message to log
    """
    _debug_file_logger.write(f"[INFO] {message}")
    _logger.info(message)


def warning(message: str):
    """
    Log a warning message.

    Warnings are always written to both file and console regardless of DEBUG_MODE.

    Args:
        message: The warning message to log
    """
    _debug_file_logger.write(f"[WARNING] {message}")
    _logger.warning(message)


def error(message: str, exc_info: bool = False):
    """
    Log an error message with optional exception traceback.

    Errors are always written to both file and console.

    Args:
        message: The error message to log
        exc_info: If True, include exception traceback (only in DEBUG_MODE)
    """
    _debug_file_logger.write(f"[ERROR] {message}")
    _logger.error(message, exc_info=exc_info and DEBUG_MODE)


def critical(message: str, exc_info: bool = True):
    """
    Log a critical error with exception traceback.

    Critical errors are always written with full traceback.

    Args:
        message: The critical error message
        exc_info: If True, include exception traceback
    """
    _debug_file_logger.write(f"[CRITICAL] {message}")
    _logger.critical(message, exc_info=exc_info and DEBUG_MODE)


def debug_timing(operation: str, elapsed_seconds: float):
    """
    Log operation timing information in human-readable format.

    This is a convenience function for logging elapsed time from manual timing.
    For automatic timing with start/end logging, use the Timer context manager.

    Args:
        operation: Description of the operation that was timed
        elapsed_seconds: Elapsed time in seconds (float)

    Example:
        start = time.time()
        # ... do work ...
        debug_timing("PDF chunking", time.time() - start)
        # Output: "[14:32:01] PDF chunking took 2.34s"
    """
    if elapsed_seconds < 1:
        time_str = f"{elapsed_seconds * 1000:.0f} ms"
    elif elapsed_seconds < 60:
        time_str = f"{elapsed_seconds:.2f}s"
    else:
        time_str = f"{elapsed_seconds / 60:.1f}m"
    debug_log(f"{operation} took {time_str}")


def close_debug_log():
    """
    Close the debug log file gracefully.

    Call this at application shutdown to ensure all logs are flushed.
    """
    _debug_file_logger.close()


# =============================================================================
# Backward Compatibility Exports
# =============================================================================

# These allow existing code to continue working without changes
__all__ = [
    "DEBUG_MODE",
    "Timer",
    "close_debug_log",
    "critical",
    "debug",
    "debug_log",
    "debug_timing",
    "error",
    "info",
    "warning",
]
