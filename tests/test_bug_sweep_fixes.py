"""
Tests for bug sweep fixes (v1.0.20).

Covers:
1. Module relocation: base_worker, queue_messages, silly_messages, status_reporter
   moved from src/ui/ to src/services/ with backward-compat shims in src/ui/
2. key_sentences_error message type sent on extraction failure
3. find_violations.py services->UI check
"""

import threading
from queue import Queue
from unittest.mock import MagicMock, patch

# =========================================================================
# Bug #3: Module relocation — canonical imports from src.services
# =========================================================================


class TestCanonicalImports:
    """Verify modules are importable from their new canonical location."""

    def test_import_queue_messages_from_services(self):
        """QueueMessage and MessageType importable from src.services."""
        from src.services.queue_messages import MessageType, QueueMessage

        assert hasattr(QueueMessage, "progress")
        assert hasattr(MessageType, "PROGRESS")

    def test_import_base_worker_from_services(self):
        """BaseWorker importable from src.services."""
        from src.services.base_worker import BaseWorker, CleanupWorker

        assert hasattr(BaseWorker, "execute")
        assert issubclass(CleanupWorker, BaseWorker)

    def test_import_silly_messages_from_services(self):
        """get_silly_message importable from src.services."""
        from src.services.silly_messages import get_silly_message

        msg = get_silly_message()
        assert isinstance(msg, str)
        assert len(msg) > 5  # Messages are full sentences, not empty/trivial
        assert msg.endswith("...")  # All silly messages end with ellipsis

    def test_import_status_reporter_from_services(self):
        """StatusReporter importable from src.services."""
        from src.services.status_reporter import StatusReporter

        q = Queue()
        reporter = StatusReporter(q)
        assert hasattr(reporter, "update")
        assert hasattr(reporter, "error")


class TestBackwardCompatShims:
    """Verify old src.ui imports still work via re-export shims."""

    def test_shim_queue_messages(self):
        """src.ui.queue_messages re-exports from src.services."""
        from src.services.queue_messages import QueueMessage as Canonical
        from src.ui.queue_messages import QueueMessage as Shimmed

        assert Canonical is Shimmed

    def test_shim_base_worker(self):
        """src.ui.base_worker re-exports from src.services."""
        from src.services.base_worker import BaseWorker as Canonical
        from src.ui.base_worker import BaseWorker as Shimmed

        assert Canonical is Shimmed

    def test_shim_silly_messages(self):
        """src.ui.silly_messages re-exports from src.services."""
        from src.services.silly_messages import get_silly_message as canonical
        from src.ui.silly_messages import get_silly_message as shimmed

        assert canonical is shimmed

    def test_shim_status_reporter(self):
        """src.ui.status_reporter re-exports from src.services."""
        from src.services.status_reporter import StatusReporter as Canonical
        from src.ui.status_reporter import StatusReporter as Shimmed

        assert Canonical is Shimmed


class TestWorkerUsesCanonicalImports:
    """Verify worker modules use the canonical src.services path."""

    def test_workers_imports_base_worker_from_services(self):
        """Worker modules should import BaseWorker from src.services, not src.ui."""
        import inspect

        import src.services.processing_worker as mod

        source = inspect.getsource(mod)
        assert "from src.services.base_worker import BaseWorker" in source
        assert "from src.ui.base_worker import BaseWorker" not in source

    def test_workers_imports_queue_messages_from_services(self):
        """Worker modules should import QueueMessage from src.services, not src.ui."""
        import inspect

        import src.services.semantic_worker as mod

        source = inspect.getsource(mod)
        assert "from src.services.queue_messages import QueueMessage" in source
        assert "from src.ui.queue_messages import QueueMessage" not in source


# =========================================================================
# Bug #2: Key excerpts error reporting
# =========================================================================


class TestKeySentencesErrorMessage:
    """Verify extraction errors produce key_sentences_error, not silent empty list."""

    @patch("src.core.summarization.key_sentences.extract_key_passages")
    def test_error_sends_key_sentences_error_type(self, mock_extract):
        """Extraction failure sends ('key_sentences_error', error_string)."""
        from src.worker_process import _spawn_key_sentences

        mock_extract.side_effect = ValueError("Embeddings shape mismatch")
        state = {
            "embeddings": MagicMock(),
            "vector_store_path": "/tmp/vs",
            "chunk_scores": None,
            "chunk_texts": ["some text"],
            "chunk_metadata": [{"source_file": "doc.pdf", "chunk_num": 0}],
            "chunk_embeddings": [[0.1, 0.2]],
            "documents": [],
            "active_worker": None,
            "auto_semantic_worker": None,
            "ask_default_questions": True,
            "shutdown": threading.Event(),
            "worker_lock": threading.Lock(),
        }
        q = Queue()
        _spawn_key_sentences(state, q)

        msg = q.get(timeout=5)
        assert msg[0] == "key_sentences_error"
        assert "Embeddings shape mismatch" in msg[1]

    @patch("src.core.summarization.key_sentences.extract_key_passages")
    def test_error_message_is_string(self, mock_extract):
        """Error payload is a plain string (picklable for IPC)."""
        from src.worker_process import _spawn_key_sentences

        mock_extract.side_effect = RuntimeError("CUDA OOM")
        state = {
            "embeddings": MagicMock(),
            "vector_store_path": "/tmp/vs",
            "chunk_scores": None,
            "chunk_texts": ["text"],
            "chunk_metadata": [{"source_file": "f.pdf", "chunk_num": 0}],
            "chunk_embeddings": [[0.1]],
            "documents": [],
            "active_worker": None,
            "auto_semantic_worker": None,
            "ask_default_questions": True,
            "shutdown": threading.Event(),
            "worker_lock": threading.Lock(),
        }
        q = Queue()
        _spawn_key_sentences(state, q)

        msg = q.get(timeout=5)
        assert isinstance(msg[1], str)

    @patch("src.core.summarization.key_sentences.extract_key_passages")
    def test_success_still_sends_result(self, mock_extract):
        """Successful extraction still sends key_sentences_result."""
        from src.worker_process import _spawn_key_sentences

        mock_ks = MagicMock()
        mock_ks.text = "Important passage"
        mock_ks.source_file = "doc.pdf"
        mock_ks.position = 0
        mock_ks.score = 0.9
        mock_extract.return_value = [mock_ks]

        state = {
            "embeddings": MagicMock(),
            "vector_store_path": "/tmp/vs",
            "chunk_scores": None,
            "chunk_texts": ["Important passage"],
            "chunk_metadata": [{"source_file": "doc.pdf", "chunk_num": 0}],
            "chunk_embeddings": [[0.1]],
            "documents": [{"page_count": 5}],
            "active_worker": None,
            "auto_semantic_worker": None,
            "ask_default_questions": True,
            "shutdown": threading.Event(),
            "worker_lock": threading.Lock(),
        }
        q = Queue()
        _spawn_key_sentences(state, q)

        msg = q.get(timeout=5)
        assert msg[0] == "key_sentences_result"


# =========================================================================
# Bug #3 (continued): find_violations.py catches services->UI imports
# =========================================================================


class TestViolationChecker:
    """Verify find_violations.py detects services->UI imports."""

    def test_services_importing_ui_detected(self):
        """services->UI import is flagged as a violation."""
        from find_violations import check_services_ui_violation

        assert check_services_ui_violation("src.services.workers", "src.ui.base_worker")

    def test_services_importing_services_allowed(self):
        """services->services import is not flagged."""
        from find_violations import check_services_ui_violation

        assert not check_services_ui_violation("src.services.workers", "src.services.base_worker")

    def test_ui_importing_services_allowed(self):
        """UI->services import is not flagged by services checker."""
        from find_violations import check_services_ui_violation

        assert not check_services_ui_violation("src.ui.main_window", "src.services.queue_messages")

    def test_full_scan_passes(self):
        """Full project scan finds no violations after relocation."""
        from pathlib import Path

        from find_violations import find_all_violations

        project_root = Path(__file__).parent.parent
        violations = find_all_violations(project_root)
        total = sum(len(v) for v in violations.values())
        assert total == 0, f"Found {total} violations: {violations}"
