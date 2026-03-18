"""Backward-compatibility shim — canonical module is src.services.queue_messages."""

from src.services.queue_messages import MessageType, QueueMessage

__all__ = ["MessageType", "QueueMessage"]
