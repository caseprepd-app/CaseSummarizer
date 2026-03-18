"""
Status Reporter Module

Provides a standardized API for sending status bar updates from workers.
Wraps the ui_queue with clean methods for progress, errors, and silly messages.

Usage:
    from src.services.status_reporter import StatusReporter

    status = StatusReporter(ui_queue)
    status.update(50, "Processing documents...")
    status.error("Failed to extract file.pdf")
    status.silly(75)
"""

import logging
import random
from queue import Queue

from src.services.queue_messages import QueueMessage
from src.services.silly_messages import get_silly_message

logger = logging.getLogger(__name__)


class StatusReporter:
    """
    Standardized status bar reporting for workers.

    Provides a clean API that replaces scattered
    ``ui_queue.put(QueueMessage.progress(...))`` calls.

    Attributes:
        ui_queue: Queue for communication with the main UI thread.
    """

    def __init__(self, ui_queue: Queue):
        """
        Initialize the status reporter.

        Args:
            ui_queue: Queue for UI communication.
        """
        self.ui_queue = ui_queue

    # 1-in-25 chance any status update gets replaced with a silly message
    SILLY_ODDS = 25

    def update(self, percentage: int, message: str):
        """
        Send a progress update to the GUI status bar.

        Has a 1-in-25 chance of replacing the message with a random
        silly message instead, for unexpected humor.

        Args:
            percentage: Progress 0-100.
            message: Status message to display.
        """
        if random.randint(1, self.SILLY_ODDS) == 1:
            message = get_silly_message()
        self.ui_queue.put(QueueMessage.progress(percentage, message))

    def error(self, message: str):
        """
        Send a non-fatal error that displays in the status bar (orange text).

        Unlike QueueMessage.error() which shows a blocking modal dialog,
        this displays briefly in the status bar without interrupting the user.

        Args:
            message: Human-readable error description.
        """
        self.ui_queue.put(QueueMessage.status_error(message))

    def silly(self, percentage: int):
        """
        Send a random silly/fun message at the given percentage.

        Args:
            percentage: Progress 0-100.
        """
        self.ui_queue.put(QueueMessage.progress(percentage, get_silly_message()))
