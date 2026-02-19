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

    def test_create_corpus(self, tmp_path, monkeypatch):
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        monkeypatch.setattr("src.core.vocabulary.corpus_registry.CORPORA_DIR", tmp_path)
        reg = CorpusRegistry()
        path = reg.create_corpus("TestCorpus")
        assert path.exists()
        assert reg.corpus_exists("TestCorpus")

    def test_list_corpora_returns_corpus_info(self, tmp_path, monkeypatch):
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        monkeypatch.setattr("src.core.vocabulary.corpus_registry.CORPORA_DIR", tmp_path)
        reg = CorpusRegistry()
        # Default corpus "General" should exist
        corpora = reg.list_corpora()
        assert len(corpora) >= 1
        assert any(c.name == "General" for c in corpora)

    def test_corpus_exists_false_for_missing(self, tmp_path, monkeypatch):
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        monkeypatch.setattr("src.core.vocabulary.corpus_registry.CORPORA_DIR", tmp_path)
        reg = CorpusRegistry()
        assert reg.corpus_exists("NonExistent") is False

    def test_delete_corpus_raises_for_last(self, tmp_path, monkeypatch):
        import pytest

        from src.core.vocabulary.corpus_registry import CorpusRegistry

        monkeypatch.setattr("src.core.vocabulary.corpus_registry.CORPORA_DIR", tmp_path)
        reg = CorpusRegistry()
        # Only "General" exists -- can't delete the last one
        with pytest.raises(ValueError, match="Cannot delete the last corpus"):
            reg.delete_corpus("General")

    def test_sanitize_name_removes_special_chars(self, tmp_path, monkeypatch):
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        monkeypatch.setattr("src.core.vocabulary.corpus_registry.CORPORA_DIR", tmp_path)
        reg = CorpusRegistry()
        safe = reg._sanitize_name('my<corpus>:name"here')
        assert "<" not in safe
        assert ">" not in safe
        assert ":" not in safe
        assert '"' not in safe


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

    def test_init_with_mocks(self):
        from src.core.summarization.multi_document_orchestrator import (
            MultiDocumentOrchestrator,
        )

        mock_summarizer = MagicMock()
        mock_model_manager = MagicMock()
        orch = MultiDocumentOrchestrator(
            document_summarizer=mock_summarizer,
            model_manager=mock_model_manager,
        )
        assert orch is not None

    def test_stop_sets_event(self):
        from src.core.summarization.multi_document_orchestrator import (
            MultiDocumentOrchestrator,
        )

        mock_summarizer = MagicMock()
        mock_model_manager = MagicMock()
        orch = MultiDocumentOrchestrator(
            document_summarizer=mock_summarizer,
            model_manager=mock_model_manager,
        )
        orch.stop()
        assert orch._stop_event.is_set()


# ---------------------------------------------------------------------------
# ExportService combined export
# ---------------------------------------------------------------------------


class TestExportServiceCombined:
    """ExportService combined export methods."""

    def test_combined_html_export(self, tmp_path):
        from unittest.mock import patch

        from src.services.export_service import ExportService

        svc = ExportService()
        out = tmp_path / "combined.html"
        with patch("src.services.export_service._auto_open_file"):
            result = svc.export_combined_html(
                vocab_data=[{"Term": "test", "Score": "50"}],
                qa_results=[],
                summary_text="A summary.",
                file_path=str(out),
            )
        assert result is True
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "test" in content

    def test_get_vocabulary_html_content_returns_string(self):
        from src.services.export_service import ExportService

        svc = ExportService()
        html = svc.get_vocabulary_html_content(
            [
                {
                    "Term": "plaintiff",
                    "Score": "80",
                    "Is Person": "No",
                    "Found By": "NER",
                    "Frequency": "1",
                }
            ]
        )
        assert isinstance(html, str)
        assert "plaintiff" in html

    def test_combined_word_export(self, tmp_path):
        from unittest.mock import patch

        from src.services.export_service import ExportService

        svc = ExportService()
        qa = MagicMock()
        qa.question = "Q?"
        qa.quick_answer = "A."
        qa.citation = "p1"
        qa.source_summary = "doc.pdf"
        qa.verification = None

        out = tmp_path / "combined.docx"
        with patch("src.services.export_service._auto_open_file"):
            result = svc.export_combined_to_word(
                vocab_data=[{"Term": "test"}],
                qa_results=[qa],
                file_path=str(out),
            )
        assert result is True
        assert out.exists()
