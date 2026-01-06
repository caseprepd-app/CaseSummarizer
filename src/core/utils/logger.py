"""
Logger Utilities Module (Backward Compatibility Wrapper)

This module provides backward compatibility for code importing from utils.logger.
All functionality has been consolidated into src/logging_config.py.

New code should import directly from logging_config:
    from src.logging_config import debug_log, info, warning, error, Timer

Existing imports will continue to work:
    from src.core.utils.logger import debug, info, Timer  # Still works
    from src.core.utils import debug, Timer  # Still works

See src/logging_config.py for full documentation on the unified logging system.
"""

# Re-export all logging functions from the unified module
from src.logging_config import (
    DEBUG_MODE,
    Timer,
    close_debug_log,
    critical,
    debug,
    debug_log,
    error,
    info,
    warning,
)

__all__ = [
    "debug_log",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "close_debug_log",
    "Timer",
    "DEBUG_MODE",
]
