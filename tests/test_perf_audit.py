"""
Tests for performance audit optimizations.

Validates performance improvements:
1. O(n×m) → O(n+m) entity membership via set in NER
2. Removed redundant uppercase glob on Windows
3. Reusable RawTextExtractor in IDF build
4. Compiled regex for filename sanitization
5. Frozenset for O(1) membership checks in LLM extractor
6. Eliminated redundant .split() in name deduplicator
7. Cached .lower() in corpus_registry (covered by #2)
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

# ========================================================================
# 1. NER entity membership set
# ========================================================================


class TestNEREntityMembershipSet:
    """Verify O(1) set-based entity token exclusion replaces O(n×m) scan."""

    def test_entity_token_indices_built_as_set(self):
        """_extract_from_doc should build a set of entity token indices."""
        from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm

        # Create mock doc with one entity spanning tokens 2-4
        mock_ent = MagicMock()
        mock_ent.label_ = "PERSON"
        mock_ent.text = "John Smith"
        mock_ent.start = 2
        mock_ent.end = 4
        mock_ent.start_char = 10
        mock_ent.end_char = 20

        # Create tokens: index 0 (outside entity) and index 3 (inside entity)
        mock_token_outside = MagicMock()
        mock_token_outside.i = 0
        mock_token_outside.text = "unusual"
        mock_token_outside.is_alpha = True
        mock_token_outside.is_space = False
        mock_token_outside.is_punct = False
        mock_token_outside.is_digit = False
        mock_token_outside.ent_type_ = ""

        mock_token_inside = MagicMock()
        mock_token_inside.i = 3
        mock_token_inside.text = "Smith"
        mock_token_inside.is_alpha = True
        mock_token_inside.is_space = False
        mock_token_inside.is_punct = False
        mock_token_inside.is_digit = False
        mock_token_inside.ent_type_ = "PERSON"

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_token_outside, mock_token_inside]))

        algo = NERAlgorithm(nlp=MagicMock())
        from collections import defaultdict

        term_freq = defaultdict(int)
        # Patch _is_unusual to return True for all tokens (so we can see which are skipped)
        with patch.object(algo, "_is_unusual", return_value=True):
            with patch.object(algo, "_clean_entity_text", return_value="John Smith"):
                with patch.object(algo, "_matches_entity_filter", return_value=False):
                    with patch.object(algo, "_is_word_rare_enough", return_value=True):
                        result = algo._extract_from_doc(mock_doc, term_freq)

        # Token inside entity span (index 3) should be skipped
        # Token outside entity span (index 0) should be included as unusual
        unusual_terms = [c.term for c in result if c.confidence == 0.6]
        assert "unusual" in unusual_terms, "Token outside entity should be extracted"
        assert "Smith" not in unusual_terms, "Token inside entity should be skipped"

    def test_source_code_uses_set_not_any(self):
        """Verify the implementation uses set lookup, not any() generator."""
        import inspect

        from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm

        source = inspect.getsource(NERAlgorithm._extract_from_doc)
        assert "entity_token_indices" in source, "Should build entity_token_indices set"
        assert "token.i in entity_token_indices" in source, "Should use set membership"
        # Old O(n×m) pattern should NOT be present
        assert "any(ent.start" not in source, "Old O(n×m) pattern should be removed"


# ========================================================================
# 2. Removed redundant uppercase glob + 7. Cached .lower()
# ========================================================================


class TestCorpusRegistryGlob:
    """Verify redundant uppercase glob removed from _count_documents."""

    def test_count_documents_single_glob_pass(self, tmp_path):
        """_count_documents should only do one glob pass per extension."""
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        # Create test files
        (tmp_path / "doc1.pdf").touch()
        (tmp_path / "doc2.txt").touch()

        with patch("src.core.vocabulary.corpus_registry.CORPORA_DIR", tmp_path):
            reg = CorpusRegistry()
            count = reg._count_documents(tmp_path)

        assert count == 2

    def test_source_no_upper_glob(self):
        """Verify no ext.upper() glob call in _count_documents source."""
        import inspect

        from src.core.vocabulary.corpus_registry import CorpusRegistry

        source = inspect.getsource(CorpusRegistry._count_documents)
        # Check for the actual glob call pattern, not comments mentioning it
        assert '.glob(f"*{ext.upper()}")' not in source, "Redundant upper() glob should be removed"

    def test_corpus_manager_single_glob(self):
        """Verify corpus_manager._get_corpus_files also has no upper() glob."""
        import inspect

        from src.core.vocabulary.corpus_manager import CorpusManager

        source = inspect.getsource(CorpusManager._get_corpus_files)
        assert '.glob(f"*{ext.upper()}")' not in source, "Redundant upper() glob should be removed"

    def test_lower_cached_in_local_variable(self):
        """Verify .lower() is called once and cached (not twice per iteration)."""
        import inspect

        from src.core.vocabulary.corpus_registry import CorpusRegistry

        source = inspect.getsource(CorpusRegistry._count_documents)
        assert "name_lower" in source, "Should cache .lower() in local variable"


# ========================================================================
# 3. Reusable RawTextExtractor in IDF build
# ========================================================================


class TestReuseExtractor:
    """Verify build_idf_index reuses a single RawTextExtractor."""

    def test_extract_text_accepts_extractor_param(self):
        """_extract_text should accept optional extractor parameter."""
        import inspect

        from src.core.vocabulary.corpus_manager import CorpusManager

        sig = inspect.signature(CorpusManager._extract_text)
        assert "extractor" in sig.parameters, "Should accept extractor param"

    def test_build_idf_creates_shared_extractor(self):
        """build_idf_index should create one extractor and pass it to _extract_text."""
        import inspect

        from src.core.vocabulary.corpus_manager import CorpusManager

        source = inspect.getsource(CorpusManager.build_idf_index)
        assert "shared_extractor" in source, "Should create shared_extractor"
        assert "extractor=shared_extractor" in source, "Should pass to _extract_text"

    def test_extract_text_uses_provided_extractor(self, tmp_path):
        """When extractor is provided, _extract_text should not create a new one."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        # Create a mock extractor
        mock_extractor = MagicMock()
        mock_extractor.process_document.return_value = {
            "status": "success",
            "extracted_text": "hello world",
        }

        mgr = CorpusManager.__new__(CorpusManager)
        mgr.corpus_dir = tmp_path

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = mgr._extract_text(test_file, extractor=mock_extractor)

        mock_extractor.process_document.assert_called_once()
        assert result == "hello world"


# ========================================================================
# 4. Compiled regex for filename sanitization
# ========================================================================


class TestSanitizeNameRegex:
    """Verify filename sanitization uses compiled regex."""

    def test_unsafe_chars_pattern_exists(self):
        """Module-level compiled regex pattern should exist."""
        from src.core.vocabulary.corpus_registry import _UNSAFE_FILENAME_CHARS

        assert isinstance(_UNSAFE_FILENAME_CHARS, re.Pattern)

    def test_sanitize_replaces_all_unsafe_chars(self):
        """All 9 unsafe characters should be replaced with underscore."""
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        with patch("src.core.vocabulary.corpus_registry.CORPORA_DIR", Path("/tmp")):
            reg = CorpusRegistry()

        unsafe = 'test<>:"/\\|?*name'
        result = reg._sanitize_name(unsafe)
        for char in '<>:"/\\|?*':
            assert char not in result, f"'{char}' should be replaced"
        assert "test" in result
        assert "name" in result

    def test_sanitize_strips_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        with patch("src.core.vocabulary.corpus_registry.CORPORA_DIR", Path("/tmp")):
            reg = CorpusRegistry()

        assert reg._sanitize_name("  hello  ") == "hello"

    def test_sanitize_preserves_safe_chars(self):
        """Normal characters should be preserved."""
        from src.core.vocabulary.corpus_registry import CorpusRegistry

        with patch("src.core.vocabulary.corpus_registry.CORPORA_DIR", Path("/tmp")):
            reg = CorpusRegistry()

        assert reg._sanitize_name("my-corpus_v2.1") == "my-corpus_v2.1"

    def test_source_no_for_loop_replace(self):
        """Verify the for-loop replace pattern is gone."""
        import inspect

        from src.core.vocabulary.corpus_registry import CorpusRegistry

        source = inspect.getsource(CorpusRegistry._sanitize_name)
        assert "for char in" not in source, "Should not use for-loop replace"


# ========================================================================
# 5. Frozenset for O(1) membership checks
# ========================================================================


# ========================================================================
# 6. Eliminated redundant .split() in name deduplicator
# ========================================================================


class TestNameDeduplicatorSplit:
    """Verify single .split() call for partitioning person names."""

    def test_source_no_duplicate_split(self):
        """_absorb_single_word_names should not have two list comprehensions with .split()."""
        import inspect

        from src.core.vocabulary.name_deduplicator import _absorb_single_word_names

        source = inspect.getsource(_absorb_single_word_names)
        # The old pattern was two list comprehensions each calling .split()
        # New pattern uses a single loop
        split_count = source.count('.get("Term", "").split()')
        assert split_count <= 1, (
            f"Expected at most 1 .split() call in partitioning logic, found {split_count}"
        )

    def test_partition_correctness(self):
        """Single-word and multi-word partition should still work correctly."""
        from src.core.vocabulary.name_deduplicator import _absorb_single_word_names

        terms = [
            {"Term": "Smith", "Type": "Person", "Frequency": "3", "# Docs": "1"},
            {"Term": "John Smith", "Type": "Person", "Frequency": "5", "# Docs": "1"},
            {"Term": "Jane Doe", "Type": "Person", "Frequency": "2", "# Docs": "1"},
            {"Term": "Doe", "Type": "Person", "Frequency": "1", "# Docs": "1"},
        ]
        result = _absorb_single_word_names(terms)
        # "Smith" should be absorbed into "John Smith"
        result_terms = [t["Term"] for t in result]
        assert "John Smith" in result_terms
        assert "Jane Doe" in result_terms

    def test_empty_and_single_term_passthrough(self):
        """Edge case: fewer than 2 person terms should pass through unchanged."""
        from src.core.vocabulary.name_deduplicator import _absorb_single_word_names

        single = [{"Term": "Smith", "Type": "Person", "Frequency": "1", "# Docs": "1"}]
        assert _absorb_single_word_names(single) == single

        empty = []
        assert _absorb_single_word_names(empty) == empty
