"""Backward-compatibility shim — canonical module is src.services.base_worker."""

from src.services.base_worker import BaseWorker, CleanupWorker

__all__ = ["BaseWorker", "CleanupWorker"]
