"""
Unified Logging Configuration for CasePrepd

Single logging system using Python's stdlib logging module.
All modules use: logger = logging.getLogger(__name__)

Log file: %APPDATA%/CasePrepd/logs/caseprepd.log
- RotatingFileHandler (5 MB max, 3 backups)
- Console handler (WARNING+ only)

User preferences control file handler level:
- Off:           WARNING+ only
- Brief:         INFO+ (default)
- Comprehensive: DEBUG+ (everything)
- Custom:        Per-category filter using logger name prefixes
"""

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Category definitions — maps display name to logger-name prefixes
# Used by Custom log level and the LoggingDialog UI
# ---------------------------------------------------------------------------

LOG_CATEGORIES: dict[str, tuple[str, ...]] = {
    "Session & Startup": (
        "src.config",
        "src.main",
    ),
    "Document Processing": (
        "src.core.chunking",
        "src.core.preprocessing",
        "src.services.document_service",
    ),
    "Vocabulary Extraction": (
        "src.core.vocabulary.algorithms",
        "src.core.vocabulary.vocabulary_extractor",
        "src.core.vocabulary.reconciler",
        "src.services.vocabulary_service",
    ),
    "ML & Learning": (
        "src.core.vocabulary.preference_learner",
        "src.core.vocabulary.feedback_manager",
        "src.core.vocabulary.canonical_scorer",
    ),
    "LLM / Ollama": (
        "src.core.ai",
        "src.core.prompting",
        "src.core.extraction.llm_extractor",
        "src.services.ai_service",
    ),
    "Q&A & Retrieval": (
        "src.core.qa",
        "src.core.retrieval",
        "src.core.vector_store",
        "src.services.qa_service",
    ),
    "Summarization": ("src.core.summarization",),
    "Export": (
        "src.services.export_service",
        "src.core.export",
    ),
    "Timing & Performance": (
        "src.logging_config",
        "src.system_resources",
    ),
    "UI Events": ("src.ui",),
    "Corpus": (
        "src.core.vocabulary.corpus_manager",
        "src.core.vocabulary.corpus_registry",
        "src.core.vocabulary.corpus_familiarity",
    ),
    "Name Processing": (
        "src.core.vocabulary.name_",
        "src.core.vocabulary.artifact_filter",
        "src.core.vocabulary.filters",
        "src.core.vocabulary.rarity_filter",
    ),
}

# ---------------------------------------------------------------------------
# Category filter — used by the file handler when level is "custom"
# ---------------------------------------------------------------------------

_custom_enabled_prefixes: tuple[str, ...] | None = None


class _CategoryFilter(logging.Filter):
    """
    Filters log records by logger name prefix when custom mode is active.

    In non-custom modes the file handler level alone decides what is written.
    In custom mode, only records whose logger name starts with an enabled
    category prefix are passed through.  WARNING+ always passes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Allow record if its level >= WARNING or its logger matches an enabled category.

        Args:
            record: The log record to evaluate.

        Returns:
            True if the record should be written.
        """
        if record.levelno >= logging.WARNING:
            return True

        level = _get_logging_level()
        if level != "custom":
            return True

        global _custom_enabled_prefixes
        if _custom_enabled_prefixes is None:
            _custom_enabled_prefixes = _rebuild_custom_prefixes()

        return any(record.name.startswith(p) for p in _custom_enabled_prefixes)


def _rebuild_custom_prefixes() -> tuple[str, ...]:
    """
    Build tuple of enabled logger-name prefixes from user preferences.

    Returns:
        tuple[str, ...]: Prefixes from enabled categories.
    """
    try:
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        states = prefs.get_custom_log_categories()
        prefixes: list[str] = []
        for cat_name, enabled in states.items():
            if enabled and cat_name in LOG_CATEGORIES:
                prefixes.extend(LOG_CATEGORIES[cat_name])
        return tuple(prefixes)
    except Exception:
        all_prefixes: list[str] = []
        for p in LOG_CATEGORIES.values():
            all_prefixes.extend(p)
        return tuple(all_prefixes)


def _get_logging_level() -> str:
    """
    Get current logging level string from user preferences.

    Returns:
        str: "off", "brief", "comprehensive", or "custom".
    """
    try:
        from src.user_preferences import get_user_preferences

        return get_user_preferences().get_logging_level()
    except Exception:
        return "brief"


# Map user preference strings to Python logging levels
_LEVEL_MAP = {
    "off": logging.WARNING,
    "brief": logging.INFO,
    "comprehensive": logging.DEBUG,
    "custom": logging.DEBUG,  # filter decides per-record
}

# ---------------------------------------------------------------------------
# File handler reference (for refresh / clear operations)
# ---------------------------------------------------------------------------

_file_handler: RotatingFileHandler | None = None
_category_filter: _CategoryFilter | None = None


def setup_logging() -> None:
    """
    Configure the root ``src`` logger.

    Must be called once at application startup (from main.py) before any
    modules log.  Sets up:

    - RotatingFileHandler → ``%APPDATA%/CasePrepd/logs/caseprepd.log``
      (5 MB max, 3 backups)
    - StreamHandler → stderr, WARNING+ only (console crash messages)
    - _CategoryFilter on the file handler for custom-mode filtering
    """
    global _file_handler, _category_filter

    from src.config import LOGS_DIR

    root = logging.getLogger("src")
    root.setLevel(logging.DEBUG)  # let handlers decide

    if root.handlers:
        return  # already configured

    fmt = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- File handler (rotates at 5 MB, keeps 3 backups) ---
    log_path = LOGS_DIR / "caseprepd.log"
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        level = _get_logging_level()
        fh.setLevel(_LEVEL_MAP.get(level, logging.INFO))
        fh.setFormatter(fmt)

        _category_filter = _CategoryFilter()
        fh.addFilter(_category_filter)

        root.addHandler(fh)
        _file_handler = fh
    except Exception as exc:
        import sys

        print(f"[WARNING] Could not set up file logging: {exc}", file=sys.stderr)

    # --- Console handler (WARNING+ only) ---
    import sys

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)


# ---------------------------------------------------------------------------
# Public helpers used by settings UI
# ---------------------------------------------------------------------------


def refresh_log_filter() -> None:
    """
    Invalidate the custom-prefix cache and update file handler level.

    Call after the user changes logging level or custom categories.
    """
    global _custom_enabled_prefixes
    _custom_enabled_prefixes = None

    if _file_handler is not None:
        level = _get_logging_level()
        _file_handler.setLevel(_LEVEL_MAP.get(level, logging.INFO))


# Alias used by logging_dialog.py
refresh_custom_log_filter = refresh_log_filter


def get_log_file_path() -> Path:
    """
    Get the path to the log file.

    Returns:
        Path: Path to caseprepd.log
    """
    from src.config import LOGS_DIR

    return LOGS_DIR / "caseprepd.log"


def get_log_file_size_mb() -> float:
    """
    Get the current log file size in megabytes.

    Returns:
        float: File size in MB, or 0.0 if file doesn't exist.
    """
    p = get_log_file_path()
    if p.exists():
        return p.stat().st_size / (1024 * 1024)
    return 0.0


def clear_log_file() -> bool:
    """
    Truncate the log file.

    Returns:
        bool: True if successful, False on error.
    """
    global _file_handler
    try:
        root = logging.getLogger("src")

        # Remove old handler
        if _file_handler is not None:
            root.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None

        # Truncate
        p = get_log_file_path()
        p.write_text("", encoding="utf-8")

        # Re-create handler

        fmt = logging.Formatter(
            "[%(levelname)s %(asctime)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        fh = RotatingFileHandler(
            p,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        level = _get_logging_level()
        fh.setLevel(_LEVEL_MAP.get(level, logging.INFO))
        fh.setFormatter(fmt)

        global _category_filter
        if _category_filter is None:
            _category_filter = _CategoryFilter()
        fh.addFilter(_category_filter)

        root.addHandler(fh)
        _file_handler = fh

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Timer context manager (unchanged public interface)
# ---------------------------------------------------------------------------

_timer_logger = logging.getLogger(__name__)


class Timer:
    """
    Context manager for timing code blocks with automatic logging.

    Usage:
        with Timer("FileParsing"):
            pass

    Attributes:
        operation_name: Name of the operation being timed.
        duration_ms: Duration in milliseconds (available after exit).
    """

    def __init__(self, operation_name: str, auto_log: bool = True):
        """
        Initialize the timer.

        Args:
            operation_name: Descriptive name for the operation.
            auto_log: If True, log start/end automatically.
        """
        self.operation_name = operation_name
        self.auto_log = auto_log
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.duration_ms: float | None = None

    def __enter__(self):
        if self.auto_log:
            _timer_logger.debug("Starting %s...", self.operation_name)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

        if self.auto_log:
            if self.duration_ms < 1000:
                duration_str = f"{self.duration_ms:.0f} ms"
            else:
                duration_str = f"{self.duration_ms / 1000:.1f} seconds"
            _timer_logger.debug("%s took %s", self.operation_name, duration_str)

        return False

    def get_duration_ms(self) -> float:
        """
        Get the measured duration in milliseconds.

        Returns:
            Duration in milliseconds.

        Raises:
            ValueError: If timer has not completed yet.
        """
        if self.duration_ms is None:
            raise ValueError("Timer has not been completed yet")
        return self.duration_ms


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "LOG_CATEGORIES",
    "Timer",
    "clear_log_file",
    "get_log_file_path",
    "get_log_file_size_mb",
    "refresh_custom_log_filter",
    "refresh_log_filter",
    "setup_logging",
]
