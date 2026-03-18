"""Backward-compatibility shim — canonical module is src.services.silly_messages."""

from src.services.silly_messages import get_silly_message

__all__ = ["get_silly_message"]
