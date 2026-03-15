"""
Tests for ProgressiveExtractionWorker helper methods.

Covers the two methods added for title_page_handling support:
- _get_search_title_remover(): reads the setting and returns a remover or None
- _chunk_documents_for_search(): applies optional title removal before chunking
"""

from queue import Queue
from unittest.mock import MagicMock, patch


def _make_worker(documents=None):
    """
    Instantiate a ProgressiveExtractionWorker with minimal required args.

    Does not start the thread — only __init__ runs, which is lightweight.
    """
    from src.services.workers import ProgressiveExtractionWorker

    return ProgressiveExtractionWorker(
        documents=documents or [],
        combined_text="",
        ui_queue=Queue(),
    )


# ---------------------------------------------------------------------------
# _get_search_title_remover
# ---------------------------------------------------------------------------


class TestGetSearchTitleRemover:
    """_get_search_title_remover returns a TitlePageRemover or None based on the setting."""

    def test_vocab_only_returns_remover(self):
        """vocab_only: should return a TitlePageRemover instance."""
        from src.core.preprocessing.title_page_remover import TitlePageRemover
        from src.user_preferences import get_user_preferences

        get_user_preferences().set("title_page_handling", "vocab_only")
        worker = _make_worker()
        result = worker._get_search_title_remover()
        assert isinstance(result, TitlePageRemover)

    def test_exclude_all_returns_none(self):
        """exclude_all: pipeline already removed title pages — no remover needed here."""
        from src.user_preferences import get_user_preferences

        get_user_preferences().set("title_page_handling", "exclude_all")
        worker = _make_worker()
        assert worker._get_search_title_remover() is None

    def test_include_all_returns_none(self):
        """include_all: title pages kept everywhere — should return None."""
        from src.user_preferences import get_user_preferences

        get_user_preferences().set("title_page_handling", "include_all")
        worker = _make_worker()
        assert worker._get_search_title_remover() is None

    def test_exception_in_prefs_defaults_to_vocab_only(self):
        """If preferences cannot be read, fall back to vocab_only (safe default)."""
        from src.core.preprocessing.title_page_remover import TitlePageRemover

        worker = _make_worker()
        with patch("src.user_preferences.get_user_preferences", side_effect=Exception("fail")):
            result = worker._get_search_title_remover()
        assert isinstance(result, TitlePageRemover)


# ---------------------------------------------------------------------------
# _chunk_documents_for_search
# ---------------------------------------------------------------------------


class TestChunkDocumentsForSearch:
    """_chunk_documents_for_search chunks docs with optional title page removal."""

    def _mock_chunker(self, captured_texts):
        """Return a mock chunker that records texts it receives."""
        chunker = MagicMock()
        chunker.chunk_text.side_effect = lambda text, source_file: captured_texts.append(text) or []
        return chunker

    def test_no_remover_passes_text_through_unchanged(self):
        """With remover=None, text is passed to the chunker verbatim."""
        text = "Q. What happened?\nA. I fell."
        worker = _make_worker([{"filename": "doc.pdf", "preprocessed_text": text}])
        captured = []
        worker._chunk_documents_for_search(self._mock_chunker(captured), title_page_remover=None)
        assert captured == [text]

    def test_remover_strips_title_page_before_chunking(self):
        """When a remover is provided, title pages are removed before chunking."""
        title = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "COUNTY OF KINGS\n"
            "Index No. 123456/2024\n"
            "JOHN DOE, Plaintiff,\n"
            "-against-\n"
            "JANE SMITH, Defendant.\n"
            "DEPOSITION OF JOHN DOE\n"
        )
        content = "\fQ. What happened?\nA. I fell."
        worker = _make_worker([{"filename": "doc.pdf", "preprocessed_text": title + content}])

        from src.core.preprocessing.title_page_remover import TitlePageRemover

        captured = []
        worker._chunk_documents_for_search(
            self._mock_chunker(captured), title_page_remover=TitlePageRemover()
        )

        assert len(captured) == 1
        assert "SUPREME COURT" not in captured[0]
        assert "Q. What happened?" in captured[0]

    def test_skips_documents_with_no_text(self):
        """Documents with empty or whitespace-only text are silently skipped."""
        docs = [
            {"filename": "empty.pdf", "preprocessed_text": ""},
            {"filename": "blank.pdf", "preprocessed_text": "   "},
            {"filename": "good.pdf", "preprocessed_text": "Q. Name?\nA. John."},
        ]
        worker = _make_worker(docs)
        captured = []
        worker._chunk_documents_for_search(self._mock_chunker(captured), title_page_remover=None)
        assert len(captured) == 1
        assert "Q. Name?" in captured[0]

    def test_prefers_preprocessed_text_over_extracted_text(self):
        """preprocessed_text takes priority over extracted_text when both are present."""
        doc = {
            "filename": "doc.pdf",
            "preprocessed_text": "cleaned text",
            "extracted_text": "raw text",
        }
        worker = _make_worker([doc])
        captured = []
        worker._chunk_documents_for_search(self._mock_chunker(captured), title_page_remover=None)
        assert captured == ["cleaned text"]

    def test_falls_back_to_extracted_text_when_preprocessed_absent(self):
        """Falls back to extracted_text when preprocessed_text is missing."""
        doc = {"filename": "doc.pdf", "extracted_text": "raw text"}
        worker = _make_worker([doc])
        captured = []
        worker._chunk_documents_for_search(self._mock_chunker(captured), title_page_remover=None)
        assert captured == ["raw text"]

    def test_concatenates_chunks_from_all_documents(self):
        """All chunks from all documents are combined into one list."""
        docs = [
            {"filename": "a.pdf", "preprocessed_text": "text a"},
            {"filename": "b.pdf", "preprocessed_text": "text b"},
        ]
        chunker = MagicMock()
        chunker.chunk_text.side_effect = lambda text, source_file: [f"chunk:{text}"]
        worker = _make_worker(docs)
        result = worker._chunk_documents_for_search(chunker, title_page_remover=None)
        assert result == ["chunk:text a", "chunk:text b"]

    def test_passes_filename_as_source_file(self):
        """source_file kwarg is set to the document's filename."""
        doc = {"filename": "depo.pdf", "preprocessed_text": "some text"}
        worker = _make_worker([doc])
        chunker = MagicMock()
        chunker.chunk_text.return_value = []
        worker._chunk_documents_for_search(chunker, title_page_remover=None)
        chunker.chunk_text.assert_called_once_with("some text", source_file="depo.pdf")
