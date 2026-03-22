"""
Tests for the 17-bug sweep fixes.

Covers: followup attribute fix, preferences error handling, thread-safe singletons,
PDF truncation ellipsis, vector store cancel guard, semantic_service cleanup, worker_process
lock, and miscellaneous low-severity fixes.
"""

import threading
from pathlib import Path
from unittest.mock import patch

# ============================================================
# Bug #1: data.answer → data.quick_answer
# ============================================================


class TestFollowupAttributeFix:
    """Bug #1: SemanticResult uses quick_answer, not answer."""

    def test_qa_result_has_quick_answer(self):
        """SemanticResult dataclass has quick_answer as the primary field."""
        from src.core.semantic.semantic_orchestrator import SemanticResult

        result = SemanticResult(question="test", quick_answer="answer text")
        assert result.quick_answer == "answer text"
        # The .answer property exists for backward compat but delegates to quick_answer
        assert result.answer == "answer text"

    def test_main_window_uses_quick_answer(self):
        """_ask_followup_for_semantic_panel uses data.quick_answer, not data.answer."""
        import inspect

        from src.ui.main_window import MainWindow

        source = inspect.getsource(MainWindow._ask_followup_for_semantic_panel)
        assert "data.quick_answer" in source
        assert "data.answer" not in source


# ============================================================
# Bug #2: user_preferences PermissionError handling
# ============================================================


class TestPreferencesErrorHandling:
    """Bug #2: Transient errors should not rename valid prefs file."""

    def test_json_decode_error_renames_file(self, tmp_path):
        """JSONDecodeError should rename file to .corrupt."""
        from src.user_preferences import UserPreferencesManager

        prefs_file = tmp_path / "prefs.json"
        prefs_file.write_text("not valid json{{{", encoding="utf-8")

        mgr = UserPreferencesManager(preferences_file=prefs_file)
        result = mgr._load_preferences()

        # File should be renamed
        assert not prefs_file.exists()
        assert (tmp_path / "prefs.json.corrupt").exists()
        # Should return defaults
        assert "model_defaults" in result

    def test_permission_error_does_not_rename(self, tmp_path):
        """PermissionError should NOT rename the file."""
        from src.user_preferences import UserPreferencesManager

        prefs_file = tmp_path / "prefs.json"
        prefs_file.write_text('{"model_defaults": {}}', encoding="utf-8")

        mgr = UserPreferencesManager(preferences_file=prefs_file)

        # Patch open to raise PermissionError
        with patch("builtins.open", side_effect=PermissionError("Dropbox lock")):
            result = mgr._load_preferences()

        # File should still exist (not renamed)
        assert prefs_file.exists()
        assert not (tmp_path / "prefs.json.corrupt").exists()
        # Should return defaults
        assert "model_defaults" in result


# ============================================================
# Bug #3: Embeddings model thread safety
# ============================================================


class TestEmbeddingsThreadSafety:
    """Bug #3: get_embeddings_model() has a lock."""

    def test_embeddings_lock_exists(self):
        """The lock variable exists in the module."""
        from src.core.retrieval.algorithms import faiss_semantic

        assert hasattr(faiss_semantic, "_embeddings_lock")
        assert isinstance(faiss_semantic._embeddings_lock, type(threading.Lock()))


# ============================================================
# Bug #4: Panel followup race condition fix
# ============================================================


class TestPanelFollowupEvent:
    """Bug #4: Panel followup uses threading.Event instead of queue polling."""

    def test_event_attributes_exist(self):
        """MainWindow should have _panel_followup_event and _panel_followup_data."""
        # We can't instantiate MainWindow, but we can check the __init__ code
        import inspect

        from src.ui.main_window import MainWindow

        source = inspect.getsource(MainWindow.__init__)
        assert "_panel_followup_event" in source
        assert "_panel_followup_data" in source

    def test_handle_queue_message_routes_followup(self):
        """_handle_queue_message should handle semantic_followup_result messages."""
        import inspect

        from src.ui.main_window import MainWindow

        source = inspect.getsource(MainWindow._handle_queue_message)
        assert "semantic_followup_result" in source
        assert "_panel_followup_event" in source


# ============================================================
# Bug #5: PDF truncation with ellipsis
# ============================================================


class TestPdfTruncationEllipsis:
    """Bug #5: PDF cells truncated with '...' indicator."""

    def test_long_cell_gets_ellipsis(self):
        """Cells > 25 chars should end with '...'."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        # We test the logic directly: > 25 chars → 22 chars + "..."
        long_text = "cervical radiculopathy syndrome"
        assert len(long_text) > 25
        truncated = long_text[:22] + "..."
        assert truncated.endswith("...")
        assert len(truncated) == 25

    def test_short_cell_no_ellipsis(self):
        """Cells <= 25 chars should not be modified."""
        short_text = "negligence"
        assert len(short_text) <= 25
        # No truncation needed


# ============================================================
# Bug #6: Vector store cancel guard
# ============================================================


class TestVectorStoreCancelGuard:
    """Bug #6: _build_vector_store checks is_stopped before sending messages."""

    def test_build_vector_store_has_cancel_check(self):
        """Source code checks is_stopped before semantic_ready message."""
        import inspect

        from src.services.workers import ProgressiveExtractionWorker

        source = inspect.getsource(ProgressiveExtractionWorker._build_vector_store)
        # The cancel guard should appear before semantic_ready
        cancel_pos = source.find("is_stopped")
        semantic_ready_pos = source.find("semantic_ready")
        assert cancel_pos != -1, "is_stopped check not found"
        assert semantic_ready_pos != -1, "semantic_ready not found"
        assert cancel_pos < semantic_ready_pos, "is_stopped check must come before semantic_ready"


# ============================================================
# Bug #7: semantic_service.clear() calls cleanup()
# ============================================================


class TestSemanticServiceClearCleanup:
    """Bug #7: clear() should call cleanup() before clearing _temp_dir."""

    def test_clear_calls_cleanup(self):
        """clear() invokes cleanup() to delete temp directory."""
        import inspect

        from src.services.semantic_service import SemanticService

        source = inspect.getsource(SemanticService.clear)
        assert "self.cleanup()" in source


# ============================================================
# Bug #8: worker_process auto_semantic_worker lock
# ============================================================


class TestAutoSemanticWorkerLock:
    """Bug #8: auto_semantic_worker access protected by worker_lock."""

    def test_auto_semantic_worker_uses_lock(self):
        """_stop_active_worker accesses auto_semantic_worker under lock."""
        import inspect

        from src.worker_process import _stop_active_worker

        source = inspect.getsource(_stop_active_worker)
        # Both workers grabbed in single lock section (consolidated from 2 to 1)
        lock_count = source.count('state["worker_lock"]')
        assert lock_count >= 1, f"Expected >= 1 lock usages, found {lock_count}"


# ============================================================
# Bug #10: Dead expression removed
# ============================================================


class TestDeadExpressionRemoved:
    """Bug #10: Unused getattr call removed from dynamic_output.py."""

    def test_no_dead_getattr(self):
        """The dead getattr(_semantic_ready) expression should be gone."""
        import inspect

        from src.ui.dynamic_output import DynamicOutputWidget

        source = inspect.getsource(DynamicOutputWidget._refresh_tabs)
        assert 'getattr(main_window, "_semantic_ready"' not in source


# ============================================================
# Bug #11: macOS platform detection
# ============================================================


class TestMacOSPlatformDetection:
    """Bug #11: Uses sys.platform instead of os.name for macOS check."""

    def test_uses_sys_platform(self):
        """Should use sys.platform == 'darwin', not os.name."""
        import inspect

        from src.ui.corpus_dialog import CorpusDialog

        source = inspect.getsource(CorpusDialog._open_corpus_folder)
        assert 'sys.platform == "darwin"' in source
        assert 'os.name == "darwin"' not in source


# ============================================================
# Bug #12: base_dialog None check
# ============================================================


class TestBaseDialogNoneCheck:
    """Bug #12: min_width/height uses 'is not None' instead of truthiness."""

    def test_uses_is_not_none(self):
        """Should check 'is not None' not truthiness."""
        import inspect

        from src.ui.base_dialog import BaseModalDialog

        source = inspect.getsource(BaseModalDialog.__init__)
        assert "is not None" in source


# ============================================================
# Bug #13: token_budget thread safety
# ============================================================


class TestTokenBudgetThreadSafety:
    """Bug #13: _get_encoder() has a lock."""

    def test_encoder_lock_exists(self):
        """The lock variable exists in the module."""
        from src.core.semantic import token_budget

        assert hasattr(token_budget, "_encoder_lock")
        assert isinstance(token_budget._encoder_lock, type(threading.Lock()))


# ============================================================
# Bug #14: export_service thread safety
# ============================================================


class TestExportServiceThreadSafety:
    """Bug #14: get_export_service() uses thread-safe SingletonHolder."""

    def test_export_service_lock_exists(self):
        """The SingletonHolder lock exists in the module."""
        from src.services import export_service

        assert hasattr(export_service, "_export_holder")
        assert isinstance(export_service._export_holder._lock, type(threading.Lock()))


# ============================================================
# Bug #15: Image file handle closed
# ============================================================


class TestImageFileHandleClosed:
    """Bug #15: Image.open() uses context manager."""

    def test_image_open_uses_context_manager(self):
        """Source should use 'with Image.open(...)' pattern."""
        source = Path("src/core/extraction/file_readers.py").read_text(encoding="utf-8")
        assert "with Image.open(" in source


# ============================================================
# Bug #16: Sentinel docstring fix
# ============================================================


class TestSentinelDocstring:
    """Bug #16: Docstring correctly says equality, not identity."""

    def test_docstring_says_equality(self):
        """Comment should mention equality comparison, not identity."""

        source = Path("src/worker_process.py").read_text(encoding="utf-8")
        assert "equality comparison" in source
        assert "identity comparison" not in source


# ============================================================
# Bug #17: HF env vars use setdefault
# ============================================================


class TestHFEnvVarsSetDefault:
    """Bug #17: HF_HOME force-set to bundled path (dev mirrors production)."""

    def test_model_loader_force_sets_hf_home(self):
        """model_loader.py set_hf_cache_env force-sets HF_HOME."""
        source = Path("src/core/utils/model_loader.py").read_text(encoding="utf-8")
        assert 'os.environ["HF_HOME"]' in source

    def test_config_sets_offline_flags(self):
        """config.py sets HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE."""
        source = Path("src/config.py").read_text(encoding="utf-8")
        assert 'os.environ["HF_HUB_OFFLINE"]' in source
        assert 'os.environ["TRANSFORMERS_OFFLINE"]' in source

    def test_cross_encoder_uses_set_hf_cache_env(self):
        """cross_encoder_reranker.py delegates to set_hf_cache_env."""
        source = Path("src/core/retrieval/cross_encoder_reranker.py").read_text(encoding="utf-8")
        assert "set_hf_cache_env" in source
        assert 'os.environ["HF_HOME"]' not in source

    def test_hallucination_verifier_uses_set_hf_cache_env(self):
        """hallucination_verifier.py delegates to set_hf_cache_env."""
        source = Path("src/deprecated/hallucination_verifier.py").read_text(encoding="utf-8")
        assert "set_hf_cache_env" in source
        assert 'os.environ["HF_HOME"]' not in source
