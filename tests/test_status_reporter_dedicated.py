"""
Dedicated tests for src/services/status_reporter.py.

Covers StatusReporter message formatting, silly-message injection odds,
progress range handling, empty/special-character messages, and error routing.
"""

from queue import Queue
from unittest.mock import patch

from src.services.queue_messages import MessageType
from src.services.status_reporter import StatusReporter


def test_format_status_message_progress():
    """update() puts a progress tuple with correct type and payload."""
    q = Queue()
    reporter = StatusReporter(q)

    with patch("src.services.status_reporter.random.randint", return_value=2):
        reporter.update(50, "Processing documents...")

    msg_type, payload = q.get_nowait()
    assert msg_type == MessageType.PROGRESS
    assert payload == (50, "Processing documents...")


def test_format_status_message_error():
    """error() puts a status_error message (non-fatal, orange text)."""
    q = Queue()
    reporter = StatusReporter(q)

    reporter.error("File extraction failed")

    msg_type, payload = q.get_nowait()
    assert msg_type == MessageType.STATUS_ERROR
    assert payload == "File extraction failed"


def test_silly_message_injection():
    """When random hits 1-in-25, message is replaced with a silly message."""
    q = Queue()
    reporter = StatusReporter(q)

    # Force the 1-in-25 trigger
    with (
        patch("src.services.status_reporter.random.randint", return_value=1),
        patch(
            "src.services.status_reporter.get_silly_message",
            return_value="Herding digital cats...",
        ),
    ):
        reporter.update(30, "Normal message")

    _, payload = q.get_nowait()
    assert payload == (30, "Herding digital cats...")


def test_no_silly_when_odds_miss():
    """When random does not hit 1, original message is preserved."""
    q = Queue()
    reporter = StatusReporter(q)

    with patch("src.services.status_reporter.random.randint", return_value=5):
        reporter.update(80, "Almost done")

    _, payload = q.get_nowait()
    assert payload == (80, "Almost done")


def test_silly_method_always_sends_silly():
    """silly() always sends a silly message regardless of odds."""
    q = Queue()
    reporter = StatusReporter(q)

    with patch(
        "src.services.status_reporter.get_silly_message",
        return_value="Convincing electrons...",
    ):
        reporter.silly(75)

    msg_type, payload = q.get_nowait()
    assert msg_type == MessageType.PROGRESS
    assert payload == (75, "Convincing electrons...")


def test_progress_percentage_zero():
    """Progress at 0% is sent correctly."""
    q = Queue()
    reporter = StatusReporter(q)

    with patch("src.services.status_reporter.random.randint", return_value=2):
        reporter.update(0, "Starting...")

    _, payload = q.get_nowait()
    assert payload[0] == 0


def test_progress_percentage_hundred():
    """Progress at 100% is sent correctly."""
    q = Queue()
    reporter = StatusReporter(q)

    with patch("src.services.status_reporter.random.randint", return_value=2):
        reporter.update(100, "Complete")

    _, payload = q.get_nowait()
    assert payload[0] == 100


def test_empty_message():
    """Empty string message is sent without error."""
    q = Queue()
    reporter = StatusReporter(q)

    with patch("src.services.status_reporter.random.randint", return_value=2):
        reporter.update(50, "")

    _, payload = q.get_nowait()
    assert payload == (50, "")


def test_special_characters_in_message():
    """Unicode, em-dashes, and newlines in messages are preserved."""
    q = Queue()
    reporter = StatusReporter(q)

    unicode_msg = "Processing caf\u00e9 r\u00e9sum\u00e9 \u2014 100% done\nnext line"

    with patch("src.services.status_reporter.random.randint", return_value=2):
        reporter.update(99, unicode_msg)

    _, payload = q.get_nowait()
    assert payload == (99, unicode_msg)


def test_long_message():
    """Very long status message is passed through without truncation."""
    q = Queue()
    reporter = StatusReporter(q)
    long_msg = "A" * 5000

    with patch("src.services.status_reporter.random.randint", return_value=2):
        reporter.update(42, long_msg)

    _, payload = q.get_nowait()
    assert payload == (42, long_msg)
    assert len(payload[1]) == 5000
