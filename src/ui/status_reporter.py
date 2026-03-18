"""Backward-compatibility shim — canonical module is src.services.status_reporter."""

from src.services.status_reporter import StatusReporter

__all__ = ["StatusReporter"]
