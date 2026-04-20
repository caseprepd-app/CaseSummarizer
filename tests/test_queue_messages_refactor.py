"""
Tests for QueueMessage factory method additions and the worker_process.py
raw-tuple refactor.

These tests cover the DRY refactor that replaced raw result_queue.put((...))
tuples throughout src/worker_process.py with QueueMessage factory calls.
"""

import re
from pathlib import Path

from src.services.queue_messages import MessageType, QueueMessage


class TestNewFactoryMethods:
    """Verify new QueueMessage factory methods added for worker_process.py."""

    def test_worker_ready_produces_correct_tuple(self):
        """QueueMessage.worker_ready() returns ('worker_ready', None)."""
        msg = QueueMessage.worker_ready()
        assert msg == (MessageType.WORKER_READY, None)
        assert msg[0] == "worker_ready"
        assert msg[1] is None

    def test_command_ack_produces_correct_tuple(self):
        """QueueMessage.command_ack(cmd) returns ('command_ack', {'cmd': cmd})."""
        msg = QueueMessage.command_ack("extract")
        assert msg == (MessageType.COMMAND_ACK, {"cmd": "extract"})
        assert msg[0] == "command_ack"
        assert msg[1] == {"cmd": "extract"}

    def test_trigger_default_semantic_started_tuple(self):
        """QueueMessage.trigger_default_semantic_started() returns the correct tuple."""
        msg = QueueMessage.trigger_default_semantic_started()
        assert msg == (MessageType.TRIGGER_DEFAULT_SEMANTIC_STARTED, None)
        assert msg[0] == "trigger_default_semantic_started"
        assert msg[1] is None

    def test_key_sentences_error_produces_correct_tuple(self):
        """QueueMessage.key_sentences_error(err) returns ('key_sentences_error', err)."""
        msg = QueueMessage.key_sentences_error("Embeddings shape mismatch")
        assert msg == (MessageType.KEY_SENTENCES_ERROR, "Embeddings shape mismatch")
        assert msg[0] == "key_sentences_error"
        assert msg[1] == "Embeddings shape mismatch"


class TestMessageTypeConstants:
    """Verify new MessageType constants exist with expected string values."""

    def test_worker_ready_constant(self):
        """WORKER_READY constant matches wire string used across the codebase."""
        assert MessageType.WORKER_READY == "worker_ready"

    def test_command_ack_constant(self):
        """COMMAND_ACK constant matches wire string."""
        assert MessageType.COMMAND_ACK == "command_ack"

    def test_trigger_default_semantic_started_constant(self):
        """TRIGGER_DEFAULT_SEMANTIC_STARTED constant matches wire string."""
        assert MessageType.TRIGGER_DEFAULT_SEMANTIC_STARTED == "trigger_default_semantic_started"

    def test_key_sentences_error_constant(self):
        """KEY_SENTENCES_ERROR constant matches wire string."""
        assert MessageType.KEY_SENTENCES_ERROR == "key_sentences_error"


class TestFactoriesReturnTuples:
    """
    Factories must return plain tuples so existing consumer code that does
    `msg_type, data = msg` or `msg[0]`, `msg[1]` continues to work.
    """

    def test_worker_ready_is_tuple(self):
        """worker_ready() returns a 2-tuple usable with tuple unpacking."""
        msg_type, data = QueueMessage.worker_ready()
        assert msg_type == "worker_ready"
        assert data is None

    def test_command_ack_is_tuple(self):
        """command_ack() returns a 2-tuple usable with tuple unpacking."""
        msg_type, data = QueueMessage.command_ack("run_qa")
        assert msg_type == "command_ack"
        assert data == {"cmd": "run_qa"}

    def test_key_sentences_error_is_tuple(self):
        """key_sentences_error() returns a 2-tuple usable with tuple unpacking."""
        msg_type, data = QueueMessage.key_sentences_error("boom")
        assert msg_type == "key_sentences_error"
        assert data == "boom"


class TestWorkerProcessNoRawTuples:
    """
    Source-level guard: worker_process.py must not reintroduce raw-tuple
    result_queue.put((...)) or internal_queue.put((...)) calls with a string
    literal in position 0.

    The two intentional exceptions are the forwarder pass-through sites where
    msg_type is a dynamic variable (not a string literal): those puts look like
    `result_queue.put((msg_type, data))` and are explicitly allowed because
    the message was already produced by a QueueMessage factory upstream.
    """

    WORKER_PROCESS = Path(__file__).resolve().parents[1] / "src" / "worker_process.py"

    # Matches result_queue.put(("literal", ...)) or internal_queue.put(("literal", ...))
    # but NOT result_queue.put((msg_type, data)) where msg_type is an identifier.
    RAW_TUPLE_PATTERN = re.compile(r"""(?:result|internal)_queue\.put\(\(\s*["']""")

    def test_no_raw_string_literal_tuples_in_worker_process(self):
        """No result_queue/internal_queue raw-tuple puts with string literals remain."""
        source = self.WORKER_PROCESS.read_text(encoding="utf-8")
        matches = self.RAW_TUPLE_PATTERN.findall(source)
        assert matches == [], (
            f"Found {len(matches)} raw-string-literal tuple put(s) in worker_process.py. "
            "Use QueueMessage factory methods instead."
        )

    def test_pass_through_forwarder_puts_are_dynamic_msg_type(self):
        """The two allowed raw puts use the dynamic msg_type variable, not a string literal."""
        source = self.WORKER_PROCESS.read_text(encoding="utf-8")
        # Exactly two pass-through puts (ner_complete re-forward + else-branch catch-all)
        pattern = re.compile(r"result_queue\.put\(\(msg_type, data\)\)")
        assert len(pattern.findall(source)) == 2

    def test_semantic_ready_stripped_put_uses_message_type_constant(self):
        """
        The semantic_ready forwarding put intentionally bypasses the factory
        (stripped embeddings payload differs from the factory schema) but MUST
        still use MessageType.SEMANTIC_READY rather than a raw "semantic_ready"
        string literal, so the wire name stays centralized.
        """
        source = self.WORKER_PROCESS.read_text(encoding="utf-8")
        assert "result_queue.put((MessageType.SEMANTIC_READY, forwarded_data))" in source

    def test_queue_message_is_imported(self):
        """worker_process.py imports QueueMessage from the services module."""
        source = self.WORKER_PROCESS.read_text(encoding="utf-8")
        assert "from src.services.queue_messages import" in source
        assert "QueueMessage" in source
        assert "MessageType" in source
