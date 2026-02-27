"""
Tests for the "None" placeholder corpus dropdown and insufficient-corpus badge.

Covers:
  - CorpusRegistry: empty init, get_active_corpus returns None,
    get_active_corpus_path returns None, first-create auto-activation,
    second-create does NOT auto-activate
  - MainWindow._refresh_corpus_dropdown: 4-state badge logic
  - MainWindow._on_corpus_changed: early return for "None"
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# =========================================================================
# Corpus Registry — empty-state behaviour
# =========================================================================


class TestCorpusRegistryEmptyState:
    """Registry with no corpora should not auto-create 'General'."""

    @pytest.fixture()
    def fresh_registry(self, tmp_path, monkeypatch):
        """Create a CorpusRegistry pointing at an empty tmp dir."""
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        monkeypatch.setattr("src.core.vocabulary.corpus_registry.CORPORA_DIR", tmp_path)
        return CorpusRegistry()

    def test_init_has_no_corpora(self, fresh_registry):
        """Fresh registry should start with zero corpora."""
        assert fresh_registry.list_corpora() == []

    def test_get_active_corpus_returns_none(self, fresh_registry):
        """get_active_corpus() returns None when no corpora exist."""
        assert fresh_registry.get_active_corpus() is None

    def test_get_active_corpus_path_returns_none(self, fresh_registry):
        """get_active_corpus_path() returns None when no corpora exist."""
        assert fresh_registry.get_active_corpus_path() is None

    def test_corpus_exists_returns_false(self, fresh_registry):
        """corpus_exists() returns False for any name in empty registry."""
        assert fresh_registry.corpus_exists("General") is False
        assert fresh_registry.corpus_exists("Anything") is False


class TestCorpusRegistryFirstCreate:
    """Creating the first corpus should auto-activate it."""

    @pytest.fixture()
    def registry_with_first(self, tmp_path, monkeypatch):
        """Create a registry then add one corpus."""
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        monkeypatch.setattr("src.core.vocabulary.corpus_registry.CORPORA_DIR", tmp_path)
        reg = CorpusRegistry()
        reg.create_corpus("Criminal")
        return reg

    def test_first_corpus_auto_activated(self, registry_with_first):
        """The first corpus created becomes active automatically."""
        assert registry_with_first.get_active_corpus() == "Criminal"

    def test_first_corpus_path_not_none(self, registry_with_first):
        """Active corpus path is valid after first creation."""
        path = registry_with_first.get_active_corpus_path()
        assert path is not None
        assert path.exists()

    def test_list_corpora_shows_active(self, registry_with_first):
        """list_corpora() marks the auto-activated corpus as active."""
        corpora = registry_with_first.list_corpora()
        assert len(corpora) == 1
        assert corpora[0].is_active is True


class TestCorpusRegistrySecondCreate:
    """Creating a second corpus should NOT switch the active corpus."""

    @pytest.fixture()
    def registry_with_two(self, tmp_path, monkeypatch):
        """Create a registry with two corpora."""
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        monkeypatch.setattr("src.core.vocabulary.corpus_registry.CORPORA_DIR", tmp_path)
        reg = CorpusRegistry()
        reg.create_corpus("Criminal")
        reg.create_corpus("Civil")
        return reg

    def test_active_remains_first(self, registry_with_two):
        """Active corpus stays as the first one after second creation."""
        assert registry_with_two.get_active_corpus() == "Criminal"

    def test_second_corpus_exists(self, registry_with_two):
        """Second corpus exists but is not active."""
        corpora = registry_with_two.list_corpora()
        civil = next(c for c in corpora if c.name == "Civil")
        assert civil.is_active is False


class TestCorpusRegistryActivePrefValidation:
    """get_active_corpus falls back correctly when pref is stale."""

    def test_stale_pref_falls_back_to_first(self, tmp_path, monkeypatch):
        """If saved pref points to deleted corpus, fall back to first."""
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        monkeypatch.setattr("src.core.vocabulary.corpus_registry.CORPORA_DIR", tmp_path)
        reg = CorpusRegistry()
        reg.create_corpus("Alpha")
        reg.create_corpus("Beta")
        reg.set_active_corpus("Beta")

        # Simulate stale preference by directly removing Beta from registry
        safe = reg._sanitize_name("Beta")
        del reg._registry["corpora"][safe]
        beta_path = tmp_path / safe
        if beta_path.exists():
            shutil.rmtree(beta_path)

        # Should fall back to Alpha (first remaining)
        assert reg.get_active_corpus() == "Alpha"


# =========================================================================
# MainWindow dropdown — 4-state badge
# =========================================================================


class TestRefreshCorpusDropdown:
    """_refresh_corpus_dropdown() sets dropdown text and badge correctly."""

    @pytest.fixture()
    def mock_window(self):
        """Build a minimal mock of MainWindow with the dropdown widgets."""
        window = MagicMock()
        window.corpus_dropdown = MagicMock()
        window.corpus_doc_count_label = MagicMock()

        # Import and bind the real method
        from src.ui.main_window import MainWindow

        window._refresh_corpus_dropdown = MainWindow._refresh_corpus_dropdown.__get__(
            window, type(window)
        )
        return window

    def test_no_corpora_shows_none_placeholder(self, mock_window):
        """0 corpora → dropdown 'None' in reddish, blank badge."""
        mock_window.corpus_registry.list_corpora.return_value = []
        mock_window.corpus_registry.get_active_corpus.return_value = None

        mock_window._refresh_corpus_dropdown()

        mock_window.corpus_dropdown.configure.assert_any_call(values=["None"])
        mock_window.corpus_dropdown.set.assert_called_with("None")
        mock_window.corpus_dropdown.configure.assert_any_call(text_color="#e07070")
        mock_window.corpus_doc_count_label.configure.assert_called_with(text="")

    def test_corpus_with_zero_docs_shows_empty(self, mock_window):
        """Corpus with 0 docs → badge '(empty)' in text_secondary."""
        from src.core.vocabulary.corpus_registry import CorpusInfo
        from src.ui.theme import COLORS

        info = CorpusInfo(name="Criminal", path=Path("/tmp"), doc_count=0, is_active=True)
        mock_window.corpus_registry.list_corpora.return_value = [info]
        mock_window.corpus_registry.get_active_corpus.return_value = "Criminal"

        mock_window._refresh_corpus_dropdown()

        mock_window.corpus_dropdown.set.assert_called_with("Criminal")
        mock_window.corpus_doc_count_label.configure.assert_called_with(
            text="(empty)", text_color=COLORS["text_secondary"]
        )

    def test_corpus_with_1_to_4_docs_shows_warning(self, mock_window):
        """Corpus with 1-4 docs → warning badge with doc count."""
        from src.core.vocabulary.corpus_registry import CorpusInfo
        from src.ui.theme import COLORS

        info = CorpusInfo(name="Criminal", path=Path("/tmp"), doc_count=3, is_active=True)
        mock_window.corpus_registry.list_corpora.return_value = [info]
        mock_window.corpus_registry.get_active_corpus.return_value = "Criminal"

        mock_window._refresh_corpus_dropdown()

        mock_window.corpus_doc_count_label.configure.assert_called_with(
            text="(3/5+ docs required for corpus functionality)",
            text_color=COLORS["warning"],
        )

    def test_corpus_with_5_plus_docs_shows_bm25(self, mock_window):
        """Corpus with 5+ docs → BM25 active badge."""
        from src.core.vocabulary.corpus_registry import CorpusInfo
        from src.ui.theme import COLORS

        info = CorpusInfo(name="Criminal", path=Path("/tmp"), doc_count=10, is_active=True)
        mock_window.corpus_registry.list_corpora.return_value = [info]
        mock_window.corpus_registry.get_active_corpus.return_value = "Criminal"

        mock_window._refresh_corpus_dropdown()

        mock_window.corpus_doc_count_label.configure.assert_called_with(
            text="(10 docs · BM25 active)", text_color=COLORS["text_secondary"]
        )

    def test_corpus_with_exactly_5_docs_shows_bm25(self, mock_window):
        """Boundary: 5 docs → BM25 active (not warning)."""
        from src.core.vocabulary.corpus_registry import CorpusInfo
        from src.ui.theme import COLORS

        info = CorpusInfo(name="Criminal", path=Path("/tmp"), doc_count=5, is_active=True)
        mock_window.corpus_registry.list_corpora.return_value = [info]
        mock_window.corpus_registry.get_active_corpus.return_value = "Criminal"

        mock_window._refresh_corpus_dropdown()

        mock_window.corpus_doc_count_label.configure.assert_called_with(
            text="(5 docs · BM25 active)", text_color=COLORS["text_secondary"]
        )

    def test_corpus_with_exactly_4_docs_shows_warning(self, mock_window):
        """Boundary: 4 docs → warning (not BM25)."""
        from src.core.vocabulary.corpus_registry import CorpusInfo
        from src.ui.theme import COLORS

        info = CorpusInfo(name="Criminal", path=Path("/tmp"), doc_count=4, is_active=True)
        mock_window.corpus_registry.list_corpora.return_value = [info]
        mock_window.corpus_registry.get_active_corpus.return_value = "Criminal"

        mock_window._refresh_corpus_dropdown()

        mock_window.corpus_doc_count_label.configure.assert_called_with(
            text="(4/5+ docs required for corpus functionality)",
            text_color=COLORS["warning"],
        )

    def test_text_color_resets_after_none(self, mock_window):
        """Switching from None state to a real corpus resets text color."""
        from src.core.vocabulary.corpus_registry import CorpusInfo
        from src.ui.theme import COLORS

        # First call: no corpora
        mock_window.corpus_registry.list_corpora.return_value = []
        mock_window.corpus_registry.get_active_corpus.return_value = None
        mock_window._refresh_corpus_dropdown()

        # Second call: corpus exists
        info = CorpusInfo(name="Criminal", path=Path("/tmp"), doc_count=0, is_active=True)
        mock_window.corpus_registry.list_corpora.return_value = [info]
        mock_window.corpus_registry.get_active_corpus.return_value = "Criminal"
        mock_window.corpus_dropdown.configure.reset_mock()
        mock_window._refresh_corpus_dropdown()

        mock_window.corpus_dropdown.configure.assert_any_call(text_color=COLORS["text_primary"])

    def test_exception_shows_error(self, mock_window):
        """Registry exception → dropdown shows 'Error'."""
        mock_window.corpus_registry.list_corpora.side_effect = RuntimeError("fail")

        mock_window._refresh_corpus_dropdown()

        mock_window.corpus_dropdown.set.assert_called_with("Error")
        mock_window.corpus_doc_count_label.configure.assert_called_with(text="")


# =========================================================================
# MainWindow._on_corpus_changed — "None" guard
# =========================================================================


class TestOnCorpusChanged:
    """_on_corpus_changed() ignores the 'None' placeholder."""

    @pytest.fixture()
    def mock_window(self):
        """Build a minimal mock of MainWindow."""
        window = MagicMock()

        from src.ui.main_window import MainWindow

        window._on_corpus_changed = MainWindow._on_corpus_changed.__get__(window, type(window))
        window._refresh_corpus_dropdown = MagicMock()
        window.set_status = MagicMock()
        return window

    def test_none_is_ignored(self, mock_window):
        """Selecting 'None' should not call set_active_corpus."""
        mock_window._on_corpus_changed("None")

        mock_window.corpus_registry.set_active_corpus.assert_not_called()
        mock_window._refresh_corpus_dropdown.assert_not_called()

    def test_real_corpus_is_applied(self, mock_window):
        """Selecting a real corpus name calls set_active_corpus."""
        mock_window._on_corpus_changed("Criminal")

        mock_window.corpus_registry.set_active_corpus.assert_called_once_with("Criminal")
        mock_window._refresh_corpus_dropdown.assert_called_once()
        mock_window.set_status.assert_called_once_with("Active corpus: Criminal")
