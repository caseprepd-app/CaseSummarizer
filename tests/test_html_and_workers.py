"""
Tests for HTML export builders and background worker classes.

Covers:
- html_builder.py: _escape, build_vocabulary_html, export_vocabulary_html,
  export_qa_html (with verification coloring)
- workers.py: ProcessingWorker, QAWorker,
  OllamaAIWorkerManager structure and initialization
"""

from queue import Queue
from unittest.mock import MagicMock

# ============================================================================
# html_builder._escape
# ============================================================================


class TestHtmlEscape:
    """Tests for the _escape() helper."""

    def test_escapes_html_entities(self):
        from src.core.export.html_builder import _escape

        assert "&amp;" in _escape("&")
        assert "&lt;" in _escape("<")
        assert "&gt;" in _escape(">")

    def test_none_returns_empty(self):
        from src.core.export.html_builder import _escape

        assert _escape(None) == ""

    def test_empty_returns_empty(self):
        from src.core.export.html_builder import _escape

        assert _escape("") == ""

    def test_plain_text_unchanged(self):
        from src.core.export.html_builder import _escape

        assert _escape("hello") == "hello"

    def test_non_string_coerced(self):
        from src.core.export.html_builder import _escape

        assert _escape(42) == "42"


# ============================================================================
# build_vocabulary_html
# ============================================================================


class TestBuildVocabularyHtml:
    """Tests for build_vocabulary_html()."""

    def _sample_vocab(self):
        return [
            {
                "Term": "plaintiff",
                "Score": "80",
                "Is Person": "No",
                "Found By": "NER",
                "Frequency": "5",
            },
            {
                "Term": "John Smith",
                "Score": "90",
                "Is Person": "Yes",
                "Found By": "NER",
                "Frequency": "3",
            },
        ]

    def test_returns_html_string(self):
        from src.core.export.html_builder import build_vocabulary_html

        html = build_vocabulary_html(self._sample_vocab())
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_contains_term_data(self):
        from src.core.export.html_builder import build_vocabulary_html

        html = build_vocabulary_html(self._sample_vocab())
        assert "plaintiff" in html
        assert "John Smith" in html

    def test_person_row_has_class(self):
        from src.core.export.html_builder import build_vocabulary_html

        html = build_vocabulary_html(self._sample_vocab())
        assert 'class="person"' in html

    def test_summary_has_counts(self):
        from src.core.export.html_builder import build_vocabulary_html

        html = build_vocabulary_html(self._sample_vocab())
        assert "2 entries" in html
        assert "1 persons" in html
        assert "1 terms" in html

    def test_default_visible_columns(self):
        from src.core.export.html_builder import build_vocabulary_html

        html = build_vocabulary_html(self._sample_vocab())
        # Default visible: Term, Score, Is Person, Found By
        # Term column should be visible (not hidden)
        assert "Term" in html

    def test_custom_visible_columns(self):
        from src.core.export.html_builder import build_vocabulary_html

        html = build_vocabulary_html(
            self._sample_vocab(),
            visible_columns=["Term", "Score"],
        )
        assert isinstance(html, str)
        assert "Term" in html

    def test_empty_vocab(self):
        from src.core.export.html_builder import build_vocabulary_html

        html = build_vocabulary_html([])
        assert "0 entries" in html

    def test_html_escaping_in_terms(self):
        from src.core.export.html_builder import build_vocabulary_html

        vocab = [
            {
                "Term": "<script>alert(1)</script>",
                "Score": "50",
                "Is Person": "No",
                "Found By": "NER",
                "Frequency": "1",
            }
        ]
        html = build_vocabulary_html(vocab)
        # Term data should be escaped (not raw <script> in table cells)
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    def test_javascript_included(self):
        from src.core.export.html_builder import build_vocabulary_html

        html = build_vocabulary_html(self._sample_vocab())
        assert "filterTable" in html
        assert "sortTable" in html

    def test_numeric_columns_json_embedded(self):
        from src.core.export.html_builder import build_vocabulary_html

        html = build_vocabulary_html(self._sample_vocab())
        assert "numericColumns" in html


# ============================================================================
# export_vocabulary_html (file I/O)
# ============================================================================


class TestExportVocabularyHtml:
    """Tests for export_vocabulary_html()."""

    def test_writes_file(self, tmp_path):
        from src.core.export.html_builder import export_vocabulary_html

        out = tmp_path / "vocab.html"
        vocab = [
            {"Term": "test", "Score": "50", "Is Person": "No", "Found By": "NER", "Frequency": "1"}
        ]
        result = export_vocabulary_html(vocab, str(out))
        assert result is True
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "test" in content

    def test_invalid_path_returns_false(self):
        from src.core.export.html_builder import export_vocabulary_html

        result = export_vocabulary_html([], "/nonexistent/dir/file.html")
        assert result is False


# ============================================================================
# export_qa_html
# ============================================================================


class TestExportQaHtml:
    """Tests for export_qa_html()."""

    def _mock_result(self, question="What?", answer="Something.", citation="p1"):
        result = MagicMock()
        result.question = question
        result.quick_answer = answer
        result.citation = citation
        result.source_summary = "doc.pdf"
        result.verification = None
        return result

    def test_writes_qa_file(self, tmp_path):
        from src.core.export.html_builder import export_qa_html

        out = tmp_path / "qa.html"
        results = [self._mock_result()]
        success = export_qa_html(results, str(out))
        assert success is True
        content = out.read_text(encoding="utf-8")
        assert "What?" in content
        assert "Something." in content

    def test_empty_results(self, tmp_path):
        from src.core.export.html_builder import export_qa_html

        out = tmp_path / "qa.html"
        success = export_qa_html([], str(out))
        assert success is True
        content = out.read_text(encoding="utf-8")
        assert "0 Q&amp;A pairs" in content

    def test_verification_coloring(self, tmp_path):
        from src.core.export.html_builder import export_qa_html

        out = tmp_path / "qa.html"
        result = self._mock_result()
        # Create mock verification
        span = MagicMock()
        span.text = "verified text"
        span.hallucination_prob = 0.1
        verification = MagicMock()
        verification.overall_reliability = 0.85
        verification.answer_rejected = False
        verification.spans = [span]
        result.verification = verification

        success = export_qa_html([result], str(out), include_verification=True)
        assert success is True
        content = out.read_text(encoding="utf-8")
        assert "verified" in content
        assert "HIGH" in content
        assert "legend" in content.lower()

    def test_rejected_answer_styling(self, tmp_path):
        from src.core.export.html_builder import export_qa_html

        out = tmp_path / "qa.html"
        result = self._mock_result()
        verification = MagicMock()
        verification.overall_reliability = 0.2
        verification.answer_rejected = True
        verification.spans = []
        result.verification = verification

        success = export_qa_html([result], str(out))
        assert success is True
        content = out.read_text(encoding="utf-8")
        assert "unreliable" in content
        assert "LOW" in content

    def test_no_verification(self, tmp_path):
        from src.core.export.html_builder import export_qa_html

        out = tmp_path / "qa.html"
        success = export_qa_html([self._mock_result()], str(out), include_verification=False)
        assert success is True
        content = out.read_text(encoding="utf-8")
        # No legend when verification is off
        assert "legend" not in content.lower() or "Verification" not in content

    def test_long_question_truncated_in_header(self, tmp_path):
        from src.core.export.html_builder import export_qa_html

        out = tmp_path / "qa.html"
        long_q = "A" * 100
        result = self._mock_result(question=long_q)
        export_qa_html([result], str(out))
        content = out.read_text(encoding="utf-8")
        assert "..." in content  # Should be truncated at 80 chars

    def test_invalid_path_returns_false(self):
        from src.core.export.html_builder import export_qa_html

        result = export_qa_html([], "/nonexistent/dir/qa.html")
        assert result is False


# ============================================================================
# ProcessingWorker init and structure
# ============================================================================


class TestProcessingWorker:
    """Tests for ProcessingWorker initialization and structure."""

    def test_init_with_defaults(self):
        from src.services.workers import ProcessingWorker

        q = Queue()
        worker = ProcessingWorker(["file1.txt"], q)
        assert worker.file_paths == ["file1.txt"]
        assert worker.jurisdiction == "ny"
        assert worker.processed_results == []

    def test_custom_jurisdiction(self):
        from src.services.workers import ProcessingWorker

        worker = ProcessingWorker(["f.txt"], Queue(), jurisdiction="ca")
        assert worker.jurisdiction == "ca"

    def test_injectable_strategy(self):
        from src.core.parallel import SequentialStrategy
        from src.services.workers import ProcessingWorker

        strategy = SequentialStrategy()
        worker = ProcessingWorker(["f.txt"], Queue(), strategy=strategy)
        assert worker.strategy is strategy

    def test_stop_sets_event(self):
        from src.services.workers import ProcessingWorker

        worker = ProcessingWorker(["f.txt"], Queue())
        assert not worker.is_stopped
        worker.stop()
        assert worker.is_stopped

    def test_empty_files_sends_finished(self):
        from src.services.workers import ProcessingWorker

        q = Queue()
        worker = ProcessingWorker([], q)
        worker.execute()
        msg = q.get(timeout=2)
        assert msg[0] == "processing_finished"


# ============================================================================
# ============================================================================
# OllamaAIWorkerManager
# ============================================================================


class TestOllamaAIWorkerManager:
    """Tests for OllamaAIWorkerManager."""

    def test_init_state(self):
        from src.services.workers import OllamaAIWorkerManager

        mgr = OllamaAIWorkerManager(Queue())
        assert mgr.is_running is False
        assert mgr.process is None

    def test_clear_queue_empties(self):
        from src.services.workers import OllamaAIWorkerManager

        q = Queue()
        q.put("a")
        q.put("b")
        OllamaAIWorkerManager._clear_queue(q)
        assert q.empty()

    def test_is_worker_alive_when_not_started(self):
        from src.services.workers import OllamaAIWorkerManager

        mgr = OllamaAIWorkerManager(Queue())
        assert mgr.is_worker_alive() is False

    def test_check_for_messages_empty(self):
        from src.services.workers import OllamaAIWorkerManager

        mgr = OllamaAIWorkerManager(Queue())
        assert mgr.check_for_messages() == []

    def test_stop_when_not_running(self):
        from src.services.workers import OllamaAIWorkerManager

        mgr = OllamaAIWorkerManager(Queue())
        mgr.stop_worker()  # Should not raise


# ============================================================================
# MultiDocSummaryWorker init
# ============================================================================


class TestMultiDocSummaryWorker:
    """Tests for MultiDocSummaryWorker initialization."""

    def test_init(self):
        from src.services.workers import MultiDocSummaryWorker

        docs = [{"filename": "test.txt", "extracted_text": "hello"}]
        params = {"summary_length": 200, "meta_length": 500}
        worker = MultiDocSummaryWorker(docs, Queue(), params)
        assert worker.documents == docs
        assert worker.ai_params == params

    def test_stop_sets_event(self):
        from src.services.workers import MultiDocSummaryWorker

        worker = MultiDocSummaryWorker([], Queue(), {})
        worker.stop()
        assert worker.is_stopped
