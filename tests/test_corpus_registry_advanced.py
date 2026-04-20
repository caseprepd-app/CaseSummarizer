"""Tests for advanced CorpusRegistry operations not covered elsewhere.

Focus areas (delete / combine / listing details / naming):
- delete_corpus: removes registry entry, honors delete_files flag,
  rejects non-existent and last-remaining deletions, updates active
- combine_corpora: merges documents, handles filename collisions,
  rejects unknown sources
- list_corpora: reports doc_count and is_active correctly
- corpus_exists: presence check across sanitized names
- _sanitize_name: replaces unsafe filename characters

Fixtures monkeypatch CORPORA_DIR and user-preference storage to a tmp_path
so each test runs in isolation.
"""

from pathlib import Path

import pytest


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    """Build a CorpusRegistry rooted at tmp_path.

    Also isolates the user_preferences singleton so set_active_corpus()
    persists only for the current test.
    """
    from src.core.vocabulary.corpus_registry import CorpusRegistry

    monkeypatch.setattr("src.core.vocabulary.corpus_registry.CORPORA_DIR", tmp_path)
    # Redirect user preferences to tmp_path as well
    prefs_file = tmp_path / "user_preferences.json"

    from src.user_preferences import UserPreferencesManager

    # Swap the singleton factory so get_user_preferences returns our manager
    class _Stub:
        def __init__(self):
            self._data = {}

        def get(self, key, default=None):
            return self._data.get(key, default)

        def set(self, key, value):
            self._data[key] = value
            return True

        def get_custom_log_categories(self):
            return {}

        def get_logging_level(self):
            return "brief"

    stub = _Stub()
    monkeypatch.setattr("src.user_preferences.get_user_preferences", lambda: stub)
    # Also patch within the corpus_registry module namespace (late binding)
    monkeypatch.setattr("src.core.vocabulary.corpus_registry.get_user_preferences", lambda: stub)
    # Keep UserPreferencesManager referenced so unused-import linters are quiet
    _ = UserPreferencesManager
    _ = prefs_file
    return CorpusRegistry()


# ---------------------------------------------------------------------------
# _sanitize_name
# ---------------------------------------------------------------------------


class TestSanitizeName:
    """_sanitize_name replaces filesystem-unsafe characters with underscores."""

    def test_plain_name_unchanged(self, registry):
        """A normal alphanumeric name passes through unmodified."""
        assert registry._sanitize_name("Criminal") == "Criminal"

    def test_strips_surrounding_whitespace(self, registry):
        """Leading and trailing whitespace is trimmed."""
        assert registry._sanitize_name("  Criminal  ") == "Criminal"

    def test_replaces_slash(self, registry):
        """Forward slashes become underscores so the name is filesystem-safe."""
        assert registry._sanitize_name("a/b") == "a_b"

    def test_replaces_multiple_unsafe_chars(self, registry):
        """Each unsafe character in a string becomes an underscore."""
        # Characters: < > : " / \ | ? *
        result = registry._sanitize_name('<>:"/\\|?*')
        assert result == "_" * 9

    def test_preserves_inner_spaces(self, registry):
        """Spaces inside the name are kept (only outer whitespace stripped)."""
        assert registry._sanitize_name("New York") == "New York"


# ---------------------------------------------------------------------------
# corpus_exists
# ---------------------------------------------------------------------------


class TestCorpusExists:
    """corpus_exists reflects registry state, not filesystem state."""

    def test_returns_false_for_nonexistent(self, registry):
        """Unknown corpus names return False."""
        assert registry.corpus_exists("NotThere") is False

    def test_returns_true_after_create(self, registry):
        """After create_corpus, corpus_exists returns True."""
        registry.create_corpus("Criminal")
        assert registry.corpus_exists("Criminal") is True

    def test_respects_sanitized_lookup(self, registry):
        """Lookup matches after name sanitization (unsafe chars replaced)."""
        registry.create_corpus("a/b")
        # "a/b" is stored as "a_b" but corpus_exists sanitizes its input
        assert registry.corpus_exists("a/b") is True


# ---------------------------------------------------------------------------
# list_corpora
# ---------------------------------------------------------------------------


class TestListCorpora:
    """list_corpora returns CorpusInfo rows with accurate metadata."""

    def test_empty_registry_returns_empty_list(self, registry):
        """Fresh registry has no corpora to list."""
        assert registry.list_corpora() == []

    def test_reports_one_entry_per_corpus(self, registry):
        """Number of entries equals number of created corpora."""
        registry.create_corpus("Alpha")
        registry.create_corpus("Beta")
        assert len(registry.list_corpora()) == 2

    def test_doc_count_reflects_supported_files(self, registry, tmp_path):
        """doc_count counts .pdf/.txt/.rtf files in the corpus directory."""
        registry.create_corpus("Alpha")
        alpha_path = registry.get_corpus_path("Alpha")
        # Add two supported files and one unsupported one
        (alpha_path / "one.pdf").write_bytes(b"fake pdf")
        (alpha_path / "two.txt").write_text("text", encoding="utf-8")
        (alpha_path / "ignored.docx").write_bytes(b"unsupported")

        corpus = next(c for c in registry.list_corpora() if c.name == "Alpha")
        assert corpus.doc_count == 2

    def test_ignores_preprocessed_files(self, registry):
        """Files with '_preprocessed' in the stem are skipped by doc_count."""
        registry.create_corpus("Alpha")
        alpha_path = registry.get_corpus_path("Alpha")
        (alpha_path / "doc_preprocessed.txt").write_text("p", encoding="utf-8")
        (alpha_path / "real.txt").write_text("r", encoding="utf-8")

        corpus = next(c for c in registry.list_corpora() if c.name == "Alpha")
        assert corpus.doc_count == 1

    def test_marks_active_corpus(self, registry):
        """is_active is True on exactly one corpus (the active one)."""
        registry.create_corpus("Alpha")
        registry.create_corpus("Beta")
        # Alpha was auto-activated as the first corpus
        info_list = registry.list_corpora()
        actives = [c for c in info_list if c.is_active]
        assert len(actives) == 1
        assert actives[0].name == "Alpha"


# ---------------------------------------------------------------------------
# delete_corpus
# ---------------------------------------------------------------------------


class TestDeleteCorpus:
    """delete_corpus removes registry entries under various conditions."""

    def test_raises_on_unknown(self, registry):
        """Deleting a non-existent corpus raises ValueError."""
        registry.create_corpus("Alpha")
        registry.create_corpus("Beta")
        with pytest.raises(ValueError, match="does not exist"):
            registry.delete_corpus("Missing")

    def test_raises_when_only_one_remaining(self, registry):
        """Refuses to delete the last remaining corpus."""
        registry.create_corpus("Alpha")
        with pytest.raises(ValueError, match="Cannot delete the last"):
            registry.delete_corpus("Alpha")

    def test_removes_from_registry(self, registry):
        """After deletion, the corpus is no longer listed."""
        registry.create_corpus("Alpha")
        registry.create_corpus("Beta")
        registry.delete_corpus("Beta")
        names = [c.name for c in registry.list_corpora()]
        assert "Beta" not in names
        assert "Alpha" in names

    def test_returns_true_on_success(self, registry):
        """Successful delete returns True."""
        registry.create_corpus("Alpha")
        registry.create_corpus("Beta")
        assert registry.delete_corpus("Beta") is True

    def test_delete_files_removes_directory(self, registry):
        """With delete_files=True, the corpus directory is removed on disk."""
        registry.create_corpus("Alpha")
        registry.create_corpus("Beta")
        beta_path = registry.get_corpus_path("Beta")
        # Drop a file so the directory is non-empty
        (beta_path / "doc.txt").write_text("x", encoding="utf-8")
        assert beta_path.exists()

        registry.delete_corpus("Beta", delete_files=True)
        assert not beta_path.exists()

    def test_delete_files_false_preserves_directory(self, registry):
        """With delete_files=False (default), the directory is left on disk."""
        registry.create_corpus("Alpha")
        registry.create_corpus("Beta")
        beta_path = registry.get_corpus_path("Beta")

        registry.delete_corpus("Beta", delete_files=False)
        assert beta_path.exists()

    def test_deleting_active_switches_active(self, registry):
        """If the active corpus is deleted, activity shifts to a remaining one."""
        registry.create_corpus("Alpha")
        registry.create_corpus("Beta")
        # Alpha is active (first-created); delete it
        registry.delete_corpus("Alpha")
        active = registry.get_active_corpus()
        # Active must now be a remaining corpus, not the deleted one
        assert active != "Alpha"
        assert active in {"Beta"}


# ---------------------------------------------------------------------------
# combine_corpora
# ---------------------------------------------------------------------------


class TestCombineCorpora:
    """combine_corpora merges files from multiple corpora into a new one."""

    def test_raises_on_unknown_source(self, registry):
        """Combine rejects unknown source corpus names."""
        registry.create_corpus("Alpha")
        with pytest.raises(ValueError, match="does not exist"):
            registry.combine_corpora(["Alpha", "Nope"], "Combined")

    def test_raises_when_target_exists(self, registry):
        """Combine fails if the new name collides with an existing corpus."""
        registry.create_corpus("Alpha")
        registry.create_corpus("Beta")
        with pytest.raises(ValueError, match="already exists"):
            registry.combine_corpora(["Alpha"], "Beta")

    def test_creates_new_corpus_and_copies_files(self, registry):
        """Combined corpus exists and contains files from all sources."""
        registry.create_corpus("Alpha")
        registry.create_corpus("Beta")
        alpha_path = registry.get_corpus_path("Alpha")
        beta_path = registry.get_corpus_path("Beta")
        (alpha_path / "a.txt").write_text("alpha", encoding="utf-8")
        (beta_path / "b.txt").write_text("beta", encoding="utf-8")

        combined_path = registry.combine_corpora(["Alpha", "Beta"], "Combo")
        assert isinstance(combined_path, Path)
        names = {p.name for p in combined_path.iterdir() if p.is_file()}
        assert "a.txt" in names
        assert "b.txt" in names

    def test_prefixes_collision_filenames(self, registry):
        """When two sources share a filename, the second copy is prefixed."""
        registry.create_corpus("Alpha")
        registry.create_corpus("Beta")
        alpha_path = registry.get_corpus_path("Alpha")
        beta_path = registry.get_corpus_path("Beta")
        (alpha_path / "doc.txt").write_text("A", encoding="utf-8")
        (beta_path / "doc.txt").write_text("B", encoding="utf-8")

        combined_path = registry.combine_corpora(["Alpha", "Beta"], "Combo")
        names = {p.name for p in combined_path.iterdir() if p.is_file()}
        # First source wins the unprefixed slot; second is prefixed
        assert "doc.txt" in names
        # Either Alpha_doc.txt or Beta_doc.txt is present (depends on order)
        prefixed = {n for n in names if n.endswith("_doc.txt")}
        assert len(prefixed) == 1

    def test_combined_corpus_registered(self, registry):
        """The new combined corpus shows up in list_corpora()."""
        registry.create_corpus("Alpha")
        registry.combine_corpora(["Alpha"], "Combo")
        names = [c.name for c in registry.list_corpora()]
        assert "Combo" in names
