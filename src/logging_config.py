"""
Unified Logging Configuration for CasePrepd

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
        """Create and initialize the debug log file (append mode)."""
        log_path = Path(__file__).parent.parent / "debug_flow.txt"
        cls._log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
        cls._log_file.write("\n" + "=" * 60 + "\n")
        cls._log_file.write("=== CasePrepd Debug Log ===\n")
        cls._log_file.write(f"Started: {datetime.now().isoformat()}\n")
        cls._log_file.write(f"DEBUG_MODE: {DEBUG_MODE}\n")
        cls._log_file.write("=" * 60 + "\n\n")
        cls._log_file.flush()

    def write(self, message: str, force: bool = False):
        """
        Write message to the debug log file based on logging level.

        Args:
            message: The message to log
            force: If True, bypasses level check (for errors/warnings)
        """
        if self.__class__._log_file and _should_log_message(message, force):
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            formatted = f"[{timestamp}] {message}"
            self.__class__._log_file.write(formatted + "\n")
            self.__class__._log_file.flush()

    def close(self):
        """Close the debug log file gracefully."""
        if self.__class__._log_file:
            self.__class__._log_file.write(f"\n{'=' * 60}\n")
            self.__class__._log_file.write(f"Ended: {datetime.now().isoformat()}\n")
            self.__class__._log_file.close()
            self.__class__._log_file = None


# Global debug file logger instance
_debug_file_logger = _DebugFileLogger()


# =============================================================================
# Log Level Configuration
# =============================================================================

# =============================================================================
# Custom Log Categories — maps category names to prefix tuples
# =============================================================================

LOG_CATEGORIES: dict[str, tuple[str, ...]] = {
    "Session & Startup": (
        "=== CasePrepd",
        "Started:",
        "Ended:",
        "[Config]",
        "[Resources]",
    ),
    "Document Processing": (
        "[ORCHESTRATOR]",
        "[QUEUE HANDLER]",
        "[PREPROCESSING]",
        "[UnifiedChunker]",
        "[Index Page Remover]",
        "[TranscriptCleaner]",
        "[PROCESSING WORKER]",
        "[TEXT UTILS]",
    ),
    "Vocabulary Extraction": (
        "[VOCAB",
        "[NER",
        "[RAKE",
        "[VocabularyService]",
        "[PROGRESSIVE WORKER]",
    ),
    "ML & Learning": (
        "[META-LEARNER]",
        "[PREF-LEARNER]",
        "[ML]",
        "[CANONICAL]",
        "[FEEDBACK]",
    ),
    "LLM / Ollama": (
        "[OLLAMA",
        "[PROMPT CONFIG]",
        "[PROMPT ADAPTER]",
        "[FOCUS]",
        "[LLMVocabExtractor]",
        "[AIService]",
    ),
    "Q&A & Retrieval": (
        "[QARetriever]",
        "[QAService]",
        "[FAISS]",
        "[BM25",
        "[HybridRetriever]",
        "[Reranker]",
        "[HallucinationVerifier]",
        "[QueryTransformer]",
        "[AnswerGenerator]",
        "[DefaultQuestions]",
        "[QA WORKER]",
        "[VectorStore]",
    ),
    "Summarization": (
        "[DOC SUMMARIZER]",
        "[CONDENSE]",
        "[LENGTH ENFORCE]",
        "[MULTI-DOC WORKER]",
    ),
    "Export": (
        "[VocabExport]",
        "[COMBINED EXPORT]",
        "Export",
        "export",
        "Saved",
    ),
    "Timing & Performance": (
        "[TIMER]",
        "[SystemMonitor]",
        "Starting ",
        " took ",
    ),
    "UI Events": (
        "[MainWindow]",
        "[QAPanel]",
        "[Settings]",
        "[QAQuestionEditor]",
    ),
    "Corpus": (
        "[Corpus",
        "[CorpusRegistry]",
        "[CorpusFamiliarity]",
    ),
    "Name Processing": (
        "[NAME-REG]",
        "[DEDUP]",
        "[ARTIFACT-FILTER]",
        "[REGEX-FILTER]",
        "[FILTER-CHAIN]",
        "[RARITY]",
    ),
}

# =============================================================================
# Custom Log Filter Cache
# =============================================================================

_custom_enabled_prefixes: tuple[str, ...] | None = None


def _rebuild_custom_prefixes() -> tuple[str, ...]:
    """
    Build tuple of enabled prefixes from user preferences.

    Returns:
        tuple[str, ...]: All prefixes from enabled categories
    """
    try:
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        states = prefs.get_custom_log_categories()
        prefixes = []
        for cat_name, enabled in states.items():
            if enabled and cat_name in LOG_CATEGORIES:
                prefixes.extend(LOG_CATEGORIES[cat_name])
        return tuple(prefixes)
    except Exception:
        # Fallback: enable everything
        all_prefixes = []
        for prefixes in LOG_CATEGORIES.values():
            all_prefixes.extend(prefixes)
        return tuple(all_prefixes)


def refresh_custom_log_filter() -> None:
    """
    Invalidate prefix cache. Call after changing custom categories.

    Forces _should_log_message() to rebuild the prefix list on next call.
    """
    global _custom_enabled_prefixes
    _custom_enabled_prefixes = None


# Brief mode keywords: Only log messages containing these patterns
# Brief logs: session markers, processing milestones, errors, exports
BRIEF_MODE_PATTERNS = (
    "=== CasePrepd",  # Session markers
    "Started:",
    "Ended:",
    "DEBUG_MODE:",
    "Processing document",  # Document processing
    "Completed:",
    "Processing complete",
    "Document processing",
    "Extracted",  # Results milestones
    "Summary:",
    "terms found",
    "vocabulary",
    "[ERROR]",  # All errors/warnings
    "[WARNING]",
    "[CRITICAL]",
    "Error:",
    "Failed:",
    "Exception:",
    "Export",  # Export operations
    "Saved",
    "export",
)


def _get_logging_level() -> str:
    """
    Get current logging level from user preferences.

    Uses lazy import to avoid circular imports.

    Returns:
        str: "off", "brief", or "comprehensive"
    """
    try:
        from src.user_preferences import get_user_preferences

        return get_user_preferences().get_logging_level()
    except Exception:
        return "brief"  # Safe default


def _should_log_message(message: str, force: bool = False) -> bool:
    """
    Check if a message should be logged based on current level.

    Args:
        message: The log message to check
        force: If True, bypasses level check (for errors/warnings)

    Returns:
        bool: True if message should be logged
    """
    if force:
        return True

    level = _get_logging_level()

    if level == "off":
        return False
    if level == "comprehensive":
        return True
    if level == "custom":
        global _custom_enabled_prefixes
        if _custom_enabled_prefixes is None:
            _custom_enabled_prefixes = _rebuild_custom_prefixes()
        return any(p in message for p in _custom_enabled_prefixes)
    # "brief" mode - check if message matches any brief pattern
    return any(pattern in message for pattern in BRIEF_MODE_PATTERNS)


def get_log_file_path() -> Path:
    """
    Get the path to the debug log file.

    Returns:
        Path: Path to debug_flow.txt
    """
    return Path(__file__).parent.parent / "debug_flow.txt"


def get_log_file_size_mb() -> float:
    """
    Get the current debug log file size in megabytes.

    Returns:
        float: File size in MB, or 0.0 if file doesn't exist
    """
    log_path = get_log_file_path()
    if log_path.exists():
        return log_path.stat().st_size / (1024 * 1024)
    return 0.0


def clear_debug_log() -> bool:
    """
    Clear the debug log file and reinitialize with a fresh header.

    Returns:
        bool: True if successful, False on error
    """
    try:
        # Close current file handle
        _DebugFileLogger._instance.close()

        # Truncate the log file
        log_path = get_log_file_path()
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("")  # Clear contents

        # Reinitialize the logger (will write fresh header)
        # Use class attribute directly since it's a singleton
        _DebugFileLogger._log_file = None
        _DebugFileLogger._initialize_log_file()

        return True
    except Exception:
        return False


# =============================================================================
# Standard Python Logging Setup
# =============================================================================


def _setup_standard_logging() -> logging.Logger:
    """
    Configure the standard Python logging framework.

    Returns:
        Configured logger instance for CasePrepd
    """
    # Create logger
    logger = logging.getLogger("CasePrepd")
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

    This is the primary debug function for CasePrepd. It always writes to
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
    _debug_file_logger.write(f"[WARNING] {message}", force=True)
    _logger.warning(message)


def error(message: str, exc_info: bool = False):
    """
    Log an error message with optional exception traceback.

    Errors are always written to both file and console.

    Args:
        message: The error message to log
        exc_info: If True, include exception traceback (only in DEBUG_MODE)
    """
    _debug_file_logger.write(f"[ERROR] {message}", force=True)
    _logger.error(message, exc_info=exc_info and DEBUG_MODE)


def critical(message: str, exc_info: bool = True):
    """
    Log a critical error with exception traceback.

    Critical errors are always written with full traceback.

    Args:
        message: The critical error message
        exc_info: If True, include exception traceback
    """
    _debug_file_logger.write(f"[CRITICAL] {message}", force=True)
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
    "LOG_CATEGORIES",
    "Timer",
    "clear_debug_log",
    "close_debug_log",
    "critical",
    "debug",
    "debug_log",
    "debug_timing",
    "error",
    "get_log_file_path",
    "get_log_file_size_mb",
    "info",
    "refresh_custom_log_filter",
    "warning",
]
