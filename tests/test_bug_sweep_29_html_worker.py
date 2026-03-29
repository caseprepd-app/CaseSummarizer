"""
Tests for recent bug sweep changes (v1.0.29) — HTML builder and worker.

Covers:
- html_builder.export_semantic_html() suppresses Answer div for empty quick_answer
- progressive_extraction_worker._wait_for_search_indexing cancellation vs failure
"""

import inspect
import threading
from dataclasses import dataclass
from queue import Queue

# =========================================================================
# html_builder.export_semantic_html — empty quick_answer guard
# =========================================================================


@dataclass
class _MockResult:
    """Minimal mock matching export_semantic_html attribute expectations."""

    question: str
    quick_answer: str
    citation: str
    source_summary: str = ""


class TestSemanticHtmlEmptyQuickAnswer:
    """export_semantic_html must NOT render an Answer div when quick_answer is empty."""

    def test_empty_quick_answer_omits_answer_div(self, tmp_path):
        """With quick_answer='', the HTML should not contain an Answer div."""
        from src.core.export.html_builder import export_semantic_html

        result = _MockResult(
            question="Who is the plaintiff?",
            quick_answer="",
            citation="John Smith filed the complaint.",
            source_summary="complaint.pdf",
        )
        output_file = tmp_path / "qa.html"

        export_semantic_html([result], str(output_file))
        content = output_file.read_text(encoding="utf-8")

        # No "Answer" label div should appear in the Q&A content
        assert ">Answer</div>" not in content.replace(" ", "").replace("\n", "")
        assert "Citation" in content
        assert "John Smith filed the complaint." in content

    def test_nonempty_quick_answer_renders_answer_div(self, tmp_path):
        """With a non-empty quick_answer, the Answer div should be present."""
        from src.core.export.html_builder import export_semantic_html

        result = _MockResult(
            question="Who is the plaintiff?",
            quick_answer="John Smith is the plaintiff.",
            citation="The plaintiff John Smith filed...",
            source_summary="complaint.pdf",
        )
        output_file = tmp_path / "qa.html"

        export_semantic_html([result], str(output_file))
        content = output_file.read_text(encoding="utf-8")

        assert "Answer" in content
        assert "John Smith is the plaintiff." in content

    def test_mixed_results_only_renders_answer_where_present(self, tmp_path):
        """Only results with non-empty quick_answer get an Answer div."""
        from src.core.export.html_builder import export_semantic_html

        results = [
            _MockResult(question="Q1?", quick_answer="", citation="Citation 1."),
            _MockResult(question="Q2?", quick_answer="Has answer.", citation="Citation 2."),
        ]
        output_file = tmp_path / "qa.html"

        export_semantic_html(results, str(output_file))
        content = output_file.read_text(encoding="utf-8")

        assert content.count("Has answer.") == 1
        assert "Citation 1." in content
        assert "Citation 2." in content


# =========================================================================
# ProgressiveExtractionWorker._wait_for_search_indexing
# =========================================================================


def _make_worker():
    """Create a ProgressiveExtractionWorker with minimal args (not started)."""
    from src.services.workers import ProgressiveExtractionWorker

    return ProgressiveExtractionWorker(
        documents=[],
        combined_text="",
        ui_queue=Queue(),
    )


class TestWaitForSearchCancellationVsFailure:
    """_wait_for_search_indexing must distinguish cancellation from failure."""

    def test_cancelled_does_not_report_failure(self):
        """When is_stopped is set, should not report a failure error."""
        worker = _make_worker()
        worker._stop_event.set()  # Simulate cancellation

        dead_thread = threading.Thread(target=lambda: None)
        dead_thread.start()
        dead_thread.join()

        worker._wait_for_search_indexing(dead_thread)

        # Drain queue and check no failure messages were sent
        messages = []
        while not worker.ui_queue.empty():
            messages.append(worker.ui_queue.get_nowait())

        error_msgs = [m for m in messages if hasattr(m, "__getitem__") and "fail" in str(m).lower()]
        assert len(error_msgs) == 0

    def test_failure_reports_error_when_not_cancelled(self):
        """Thread finished without success and not cancelled reports failure."""
        worker = _make_worker()

        dead_thread = threading.Thread(target=lambda: None)
        dead_thread.start()
        dead_thread.join()

        with worker._search_error_lock:
            worker._search_error_msg = "embedding model failed"

        worker._wait_for_search_indexing(dead_thread)

        messages = []
        while not worker.ui_queue.empty():
            messages.append(worker.ui_queue.get_nowait())

        all_text = " ".join(str(m) for m in messages)
        assert "fail" in all_text.lower() or "still available" in all_text.lower()

    def test_source_checks_is_stopped_before_failure_report(self):
        """_wait_for_search_indexing checks is_stopped before reporting failure."""
        from src.services.progressive_extraction_worker import (
            ProgressiveExtractionWorker,
        )

        source = inspect.getsource(ProgressiveExtractionWorker._wait_for_search_indexing)
        assert "self.is_stopped" in source
        stopped_pos = source.find("self.is_stopped")
        search_succeeded_pos = source.find("self._search_succeeded")
        assert stopped_pos < search_succeeded_pos

    def test_success_does_not_report_failure(self):
        """When _search_succeeded is set, no failure is reported."""
        worker = _make_worker()
        worker._search_succeeded.set()  # Simulate success

        dead_thread = threading.Thread(target=lambda: None)
        dead_thread.start()
        dead_thread.join()

        worker._wait_for_search_indexing(dead_thread)

        messages = []
        while not worker.ui_queue.empty():
            messages.append(worker.ui_queue.get_nowait())

        error_msgs = [
            m
            for m in messages
            if "fail" in str(m).lower() and "still available" not in str(m).lower()
        ]
        assert len(error_msgs) == 0
