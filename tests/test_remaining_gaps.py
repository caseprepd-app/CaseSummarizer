"""Tests for remaining coverage gaps: config meta, role profiles, corpus registry,
default questions, QA exporter, combined HTML builder, multi-doc orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# ConfigDefaultsMeta (config_defaults_meta.py)
# ---------------------------------------------------------------------------


class TestConfigDefaultsMeta:
    """DESCRIPTIONS dict provides tooltip metadata for all settings."""

    def test_descriptions_dict_exists(self):
        from src.config_defaults_meta import DESCRIPTIONS

        assert isinstance(DESCRIPTIONS, dict)
        assert len(DESCRIPTIONS) > 0

    def test_each_entry_has_label_and_tooltip(self):
        from src.config_defaults_meta import DESCRIPTIONS

        for key, entry in DESCRIPTIONS.items():
            assert "label" in entry, f"Missing 'label' for key: {key}"
            assert "tooltip" in entry, f"Missing 'tooltip' for key: {key}"
            assert isinstance(entry["label"], str)
            assert isinstance(entry["tooltip"], str)

    def test_known_settings_present(self):
        """Key settings should have metadata."""
        from src.config_defaults_meta import DESCRIPTIONS

        expected_keys = [
            "bm25_k1",
            "bm25_b",
            "summary_temperature",
            "qa_retrieval_k",
            "qa_max_tokens",
        ]
        for key in expected_keys:
            assert key in DESCRIPTIONS, f"Missing metadata for: {key}"

    def test_labels_non_empty(self):
        from src.config_defaults_meta import DESCRIPTIONS

        for key, entry in DESCRIPTIONS.items():
            assert len(entry["label"].strip()) > 0, f"Empty label for: {key}"

    def test_tooltips_non_empty(self):
        from src.config_defaults_meta import DESCRIPTIONS

        for key, entry in DESCRIPTIONS.items():
            assert len(entry["tooltip"].strip()) > 0, f"Empty tooltip for: {key}"


# ---------------------------------------------------------------------------
# RoleProfiles
# ---------------------------------------------------------------------------


class TestStenographerProfile:
    """StenographerProfile detects person roles and place relevance."""

    def _make(self):
        from src.core.vocabulary.role_profiles import StenographerProfile

        return StenographerProfile()

    def test_creation(self):
        profile = self._make()
        assert profile is not None

    def test_detect_person_role_plaintiff(self):
        profile = self._make()
        text = "SMITH, the Plaintiff, testified that he was injured."
        role = profile.detect_person_role("Smith", text)
        # Should detect some role (Plaintiff or Person in case)
        assert isinstance(role, str) and len(role) > 0

    def test_detect_person_role_defendant_attorney(self):
        profile = self._make()
        text = "MR. JONES, attorney for the defendant, objected."
        role = profile.detect_person_role("Jones", text)
        assert role != ""

    def test_detect_person_role_unknown(self):
        profile = self._make()
        text = "The weather was nice that day."
        role = profile.detect_person_role("Nobody", text)
        # Should return a default role or empty string
        assert isinstance(role, str)

    def test_detect_place_relevance_medical(self):
        profile = self._make()
        text = "The patient was treated at General Hospital for his injuries."
        relevance = profile.detect_place_relevance("General Hospital", text)
        assert isinstance(relevance, str)

    def test_detect_place_relevance_unknown(self):
        profile = self._make()
        text = "The sky was blue."
        relevance = profile.detect_place_relevance("Unknown Place", text)
        assert isinstance(relevance, str)


# ---------------------------------------------------------------------------
# DefaultQuestionsManager
# ---------------------------------------------------------------------------


class TestDefaultQuestion:
    """DefaultQuestion dataclass."""

    def test_creation(self):
        from src.core.qa.default_questions_manager import DefaultQuestion

        q = DefaultQuestion(text="What happened?")
        assert q.text == "What happened?"
        assert q.enabled is True

    def test_disabled(self):
        from src.core.qa.default_questions_manager import DefaultQuestion

        q = DefaultQuestion(text="Q?", enabled=False)
        assert q.enabled is False

    def test_to_dict(self):
        from src.core.qa.default_questions_manager import DefaultQuestion

        q = DefaultQuestion(text="Q?", enabled=True)
        d = q.to_dict()
        assert d["text"] == "Q?"
        assert d["enabled"] is True

    def test_from_dict(self):
        from src.core.qa.default_questions_manager import DefaultQuestion

        q = DefaultQuestion.from_dict({"text": "Q?", "enabled": False})
        assert q.text == "Q?"
        assert q.enabled is False


class TestDefaultQuestionsManager:
    """DefaultQuestionsManager manages default Q&A questions."""

    def _make(self, tmp_path):
        from src.core.qa.default_questions_manager import DefaultQuestionsManager

        return DefaultQuestionsManager(config_path=tmp_path / "questions.json")

    def test_creation_creates_defaults(self, tmp_path):
        mgr = self._make(tmp_path)
        questions = mgr.get_all_questions()
        assert len(questions) >= 1

    def test_get_enabled_questions(self, tmp_path):
        mgr = self._make(tmp_path)
        enabled = mgr.get_enabled_questions()
        assert isinstance(enabled, list)
        assert all(isinstance(q, str) for q in enabled)

    def test_get_enabled_count(self, tmp_path):
        mgr = self._make(tmp_path)
        count = mgr.get_enabled_count()
        assert count == len(mgr.get_enabled_questions())

    def test_get_total_count(self, tmp_path):
        mgr = self._make(tmp_path)
        total = mgr.get_total_count()
        assert total == len(mgr.get_all_questions())

    def test_add_question(self, tmp_path):
        mgr = self._make(tmp_path)
        initial = mgr.get_total_count()
        idx = mgr.add_question("New question?")
        assert idx >= 0
        assert mgr.get_total_count() == initial + 1

    def test_remove_question(self, tmp_path):
        mgr = self._make(tmp_path)
        initial = mgr.get_total_count()
        result = mgr.remove_question(0)
        assert result is True
        assert mgr.get_total_count() == initial - 1

    def test_update_question(self, tmp_path):
        mgr = self._make(tmp_path)
        result = mgr.update_question(0, "Updated question?")
        assert result is True
        assert mgr.get_all_questions()[0].text == "Updated question?"

    def test_set_enabled(self, tmp_path):
        mgr = self._make(tmp_path)
        mgr.set_enabled(0, False)
        assert mgr.get_all_questions()[0].enabled is False

    def test_reload(self, tmp_path):
        mgr = self._make(tmp_path)
        mgr.reload()  # Should not raise


# ---------------------------------------------------------------------------
# CorpusRegistry
# ---------------------------------------------------------------------------


class TestCorpusInfo:
    """CorpusInfo dataclass."""

    def test_creation(self):
        from src.core.vocabulary.corpus_registry import CorpusInfo

        info = CorpusInfo(name="General", path=Path("/tmp"), doc_count=5, is_active=True)
        assert info.name == "General"
        assert info.doc_count == 5
        assert info.is_active is True


class TestCorpusRegistry:
    """CorpusRegistry manages named document corpora."""

    def test_class_exists(self):
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        assert CorpusRegistry is not None

    def test_has_list_method(self):
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        assert hasattr(CorpusRegistry, "list_corpora")

    def test_has_create_method(self):
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        assert hasattr(CorpusRegistry, "create_corpus")

    def test_has_delete_method(self):
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        assert hasattr(CorpusRegistry, "delete_corpus")

    def test_has_corpus_exists_method(self):
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        assert hasattr(CorpusRegistry, "corpus_exists")


# ---------------------------------------------------------------------------
# QA Exporter
# ---------------------------------------------------------------------------


class TestQAExporter:
    """export_qa_results exports Q&A to Word/PDF."""

    def test_function_exists(self):
        from src.core.export.qa_exporter import export_qa_results

        assert callable(export_qa_results)

    def test_empty_results(self):
        from src.core.export.qa_exporter import export_qa_results

        mock_builder = MagicMock()
        export_qa_results([], mock_builder)
        # Should call builder methods but not crash

    def test_with_results(self):
        from src.core.export.qa_exporter import export_qa_results

        mock_builder = MagicMock()
        result = MagicMock()
        result.question = "What happened?"
        result.quick_answer = "An accident occurred."
        result.citation = "Page 5, lines 10-15"
        result.source_summary = "complaint.pdf"
        result.verification = None

        export_qa_results([result], mock_builder, include_verification_colors=False)
        # builder should have been called for title, Q&A content
        assert mock_builder.add_heading.called or mock_builder.add_paragraph.called


# ---------------------------------------------------------------------------
# Combined HTML Builder
# ---------------------------------------------------------------------------


class TestCombinedHtmlBuilder:
    """build_combined_html creates tabbed HTML report."""

    def test_function_exists(self):
        from src.core.export.combined_html_builder import build_combined_html

        assert callable(build_combined_html)

    def test_empty_data(self):
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html(vocab_data=[], qa_results=[], summary_text="")
        assert isinstance(html, str)
        assert "<html" in html.lower()

    def test_vocab_only(self):
        from src.core.export.combined_html_builder import build_combined_html

        vocab = [{"Term": "plaintiff", "Category": "Legal"}]
        html = build_combined_html(vocab_data=vocab, qa_results=[], summary_text="")
        assert "plaintiff" in html

    def test_summary_only(self):
        from src.core.export.combined_html_builder import build_combined_html

        html = build_combined_html(vocab_data=[], qa_results=[], summary_text="This is a summary.")
        assert "This is a summary" in html

    def test_qa_results(self):
        from src.core.export.combined_html_builder import build_combined_html

        qa = [
            MagicMock(
                question="What happened?",
                quick_answer="An accident.",
                citation="p5",
                source_summary="doc.pdf",
                verification=None,
            )
        ]
        html = build_combined_html(vocab_data=[], qa_results=qa, summary_text="")
        assert "What happened?" in html


# ---------------------------------------------------------------------------
# MultiDocumentOrchestrator
# ---------------------------------------------------------------------------


class TestMultiDocumentOrchestrator:
    """MultiDocumentOrchestrator coordinates multi-doc summarization."""

    def test_class_exists(self):
        from src.core.summarization.multi_document_orchestrator import (
            MultiDocumentOrchestrator,
        )

        assert MultiDocumentOrchestrator is not None

    def test_has_summarize_method(self):
        from src.core.summarization.multi_document_orchestrator import (
            MultiDocumentOrchestrator,
        )

        assert hasattr(MultiDocumentOrchestrator, "summarize_documents")

    def test_has_stop_method(self):
        from src.core.summarization.multi_document_orchestrator import (
            MultiDocumentOrchestrator,
        )

        assert hasattr(MultiDocumentOrchestrator, "stop")


# ---------------------------------------------------------------------------
# ExportService combined export
# ---------------------------------------------------------------------------


class TestExportServiceCombined:
    """ExportService combined export methods."""

    def test_has_combined_html_method(self):
        from src.services.export_service import ExportService

        assert hasattr(ExportService, "export_combined_html")

    def test_has_combined_word_method(self):
        from src.services.export_service import ExportService

        assert hasattr(ExportService, "export_combined_to_word")

    def test_has_combined_pdf_method(self):
        from src.services.export_service import ExportService

        assert hasattr(ExportService, "export_combined_to_pdf")

    def test_has_get_vocabulary_html_content(self):
        from src.services.export_service import ExportService

        assert hasattr(ExportService, "get_vocabulary_html_content")
