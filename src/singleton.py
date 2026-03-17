"""
Thread-safe singleton holder.

Eliminates the boilerplate of double-checked locking that was
previously copy-pasted across user_preferences and ExportService.

Lives at src/ (not src/services/) to avoid circular imports when
user_preferences.py needs it before the services package initializes.

Usage:
    _holder = SingletonHolder(MyClass)

    def get_my_class() -> MyClass:
        return _holder.get()

    def reset_singleton() -> None:
        _holder.reset()
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class SingletonHolder:
    """Thread-safe container for a lazily-created singleton instance.

    Uses double-checked locking so the fast path (instance already
    created) never acquires the lock.

    Attributes:
        _factory: Callable that creates the instance on first access.
        _instance: The cached singleton (or None).
        _lock: Guards creation to prevent races.
    """

    def __init__(self, factory: Callable[..., T]):
        """Initialize with a factory callable.

        Args:
            factory: Called once (with any args forwarded from get())
                     to create the singleton.
        """
        self._factory = factory
        self._instance: T | None = None
        self._lock = threading.Lock()

    def get(self, *args, **kwargs) -> T:
        """Return the singleton, creating it on first call.

        Args:
            *args: Forwarded to factory on first call only.
            **kwargs: Forwarded to factory on first call only.

        Returns:
            The singleton instance.
        """
        if self._instance is not None:
            return self._instance
        with self._lock:
            if self._instance is None:
                self._instance = self._factory(*args, **kwargs)
        return self._instance

    def reset(self) -> None:
        """Clear the cached instance (for test isolation)."""
        with self._lock:
            self._instance = None
