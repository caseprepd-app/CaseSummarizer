"""
File utilities for CasePrepd.

Shared helpers for safe file operations (atomic rename with retry, etc.).
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

_REPLACE_MAX_ATTEMPTS = 3
_REPLACE_RETRY_DELAY = 0.1  # seconds


def safe_replace(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    """
    Atomic file rename with retry for Windows/Dropbox file locking.

    os.replace() can raise PermissionError (WinError 5) when Dropbox or
    antivirus holds a brief lock on the destination file. Retrying after
    a short delay is sufficient to work around this.

    Args:
        src: Source file path (temp file)
        dst: Destination file path (final location)

    Raises:
        OSError: If all retry attempts fail
    """
    for attempt in range(1, _REPLACE_MAX_ATTEMPTS + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_MAX_ATTEMPTS:
                raise
            logger.debug(
                "os.replace failed (attempt %d/%d), retrying in %dms...",
                attempt,
                _REPLACE_MAX_ATTEMPTS,
                int(_REPLACE_RETRY_DELAY * 1000),
            )
            time.sleep(_REPLACE_RETRY_DELAY)
