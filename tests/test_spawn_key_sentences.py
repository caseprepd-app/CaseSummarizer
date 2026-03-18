"""
Tests for _spawn_key_sentences() in worker_process.py.

Covers the fire-and-forget daemon thread that extracts key excerpts
after semantic search indexing completes.
"""

import threading
import time
from queue import Queue
from unittest.mock import MagicMock, patch

from src.worker_process import _spawn_key_sentences


def _make_state(**overrides):
    """Create a minimal subprocess state dict."""
    state = {
        "embeddings": None,
        "vector_store_path": None,
        "chunk_scores": None,
        "chunk_texts": None,
        "chunk_metadata": None,
        "chunk_embeddings": None,
        "documents": [],
        "active_worker": None,
        "auto_semantic_worker": None,
        "ask_default_questions": True,
        "shutdown": threading.Event(),
        "worker_lock": threading.Lock(),
    }
    state.update(overrides)
    return state


class TestSpawnKeySentences:
    """Tests for _spawn_key_sentences()."""

    def test_no_chunk_data_sends_empty_result(self):
        """When no chunk texts, sends empty key_sentences_result immediately."""
        state = _make_state()
        q = Queue()
        _spawn_key_sentences(state, q)
        # Give daemon thread a moment
        time.sleep(0.1)
        msg = q.get_nowait()
        assert msg == ("key_sentences_result", [])

    def test_no_chunk_embeddings_sends_empty_result(self):
        """When chunk_embeddings is None, sends empty result."""
        state = _make_state(
            chunk_texts=["Hello world"],
            chunk_metadata=[{"source_file": "doc.pdf", "chunk_num": 0}],
            chunk_embeddings=None,
        )
        q = Queue()
        _spawn_key_sentences(state, q)
        time.sleep(0.1)
        msg = q.get_nowait()
        assert msg == ("key_sentences_result", [])

    @patch("src.core.summarization.key_sentences.extract_key_passages")
    def test_successful_extraction(self, mock_extract):
        """Successful extraction sends serialized KeySentence dicts."""
        mock_ks = MagicMock()
        mock_ks.text = "Key passage"
        mock_ks.source_file = "doc.pdf"
        mock_ks.position = 0
        mock_ks.score = 0.95
        mock_extract.return_value = [mock_ks]

        state = _make_state(
            chunk_texts=["Key passage"],
            chunk_metadata=[{"source_file": "doc.pdf", "chunk_num": 0}],
            chunk_embeddings=[[0.1, 0.2]],
            documents=[{"page_count": 10}],
        )
        q = Queue()
        _spawn_key_sentences(state, q)

        msg = q.get(timeout=5)
        assert msg[0] == "key_sentences_result"
        assert len(msg[1]) == 1
        assert msg[1][0]["text"] == "Key passage"
        assert msg[1][0]["score"] == 0.95

    @patch("src.core.summarization.key_sentences.extract_key_passages")
    def test_extraction_error_sends_error_message(self, mock_extract):
        """Exception during extraction sends error message to GUI."""
        mock_extract.side_effect = RuntimeError("Model OOM")

        state = _make_state(
            chunk_texts=["text"],
            chunk_metadata=[{"source_file": "doc.pdf", "chunk_num": 0}],
            chunk_embeddings=[[0.1]],
        )
        q = Queue()
        _spawn_key_sentences(state, q)

        msg = q.get(timeout=5)
        assert msg[0] == "key_sentences_error"
        assert "Model OOM" in msg[1]

    def test_page_count_none_treated_as_zero(self):
        """Documents with page_count=None don't crash total_pages sum."""
        state = _make_state(
            chunk_texts=None,
            documents=[{"page_count": None}, {"page_count": 5}],
        )
        q = Queue()
        _spawn_key_sentences(state, q)
        time.sleep(0.1)
        # Should not crash; sends empty result because no chunk_texts
        msg = q.get_nowait()
        assert msg[0] == "key_sentences_result"

    @patch("src.core.summarization.key_sentences.extract_key_passages")
    def test_serializes_dataclass_to_dict(self, mock_extract):
        """KeySentence dataclasses are serialized to plain dicts."""
        mock_ks = MagicMock()
        mock_ks.text = "Passage"
        mock_ks.source_file = "a.pdf"
        mock_ks.position = 3
        mock_ks.score = 0.8
        mock_extract.return_value = [mock_ks]

        state = _make_state(
            chunk_texts=["Passage"],
            chunk_metadata=[{"source_file": "a.pdf", "chunk_num": 0}],
            chunk_embeddings=[[0.1]],
        )
        q = Queue()
        _spawn_key_sentences(state, q)

        msg = q.get(timeout=5)
        result = msg[1][0]
        assert isinstance(result, dict)
        assert set(result.keys()) == {"text", "source_file", "position", "score"}
