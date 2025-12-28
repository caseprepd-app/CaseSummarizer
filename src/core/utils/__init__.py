"""
Utility Modules for LocalScribe

This package provides shared utility functions used across the application:
- Logging utilities (debug, info, warning, error, critical, Timer)
- Other utility functions as added

Note: Logging functionality is now centralized in src/logging_config.py.
      These re-exports maintain backward compatibility.
"""

from .logger import (
    DEBUG_MODE,
    Timer,
    critical,
    debug,
    debug_log,
    error,
    info,
    warning,
)

__all__ = [
    'debug',
    'debug_log',
    'info',
    'warning',
    'error',
    'critical',
    'Timer',
    'DEBUG_MODE',
]
