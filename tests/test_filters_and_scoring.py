"""Tests for vocabulary filters, gibberish detection, and pattern matching."""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# FilterChain
# ---------------------------------------------------------------------------


class TestFilterChainStats:
    """FilterChainStats dataclass."""

    def test_defaults(self):
        from src.core.vocabulary.filters.filter_chain import FilterChainStats

        s = FilterChainStats()
        assert s.total_input == 0
        assert s.total_output == 0
        assert s.total_removed == 0
        assert s.total_time_ms == 0.0
        assert s.per_filter_stats == {}


class TestVocabularyFilterChain:
    """VocabularyFilterChain orchestrates filters."""

    def _make_filter(self, name, priority=50, enabled=True, keep_fn=None):
        """Create a mock filter for testing."""
        from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult

        class MockFilter(BaseVocabularyFilter):
            def filter(self, vocabulary):
                return FilterResult(vocabulary=vocabulary, removed_count=0)

        f = MockFilter()
        f.name = name
        f.priority = priority
        f.enabled = enabled

        if keep_fn:

            def do_filter(vocab):
                kept = [v for v in vocab if keep_fn(v)]
                return FilterResult(vocabulary=kept, removed_count=len(vocab) - len(kept))

            f.filter = do_filter
        else:
            f.filter = lambda vocab: FilterResult(vocabulary=vocab, removed_count=0)

        return f

    def test_empty_chain_returns_input(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        chain = VocabularyFilterChain([])
        vocab = [{"Term": "plaintiff"}, {"Term": "defendant"}]
        result = chain.run(vocab)
        assert len(result.vocabulary) == 2
        assert result.removed_count == 0

    def test_empty_vocabulary(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        chain = VocabularyFilterChain([self._make_filter("test")])
        result = chain.run([])
        assert result.vocabulary == []
        assert result.removed_count == 0

    def test_single_filter_removes_terms(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        f = self._make_filter("short_filter", keep_fn=lambda v: len(v.get("Term", "")) > 3)
        chain = VocabularyFilterChain([f])
        vocab = [{"Term": "a"}, {"Term": "plaintiff"}, {"Term": "bb"}]
        result = chain.run(vocab)
        assert len(result.vocabulary) == 1
        assert result.vocabulary[0]["Term"] == "plaintiff"
        assert result.removed_count == 2

    def test_filters_run_in_priority_order(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        order = []

        def make_ordered(name, prio):
            f = self._make_filter(name, priority=prio)
            original_filter = f.filter

            def tracking_filter(vocab):
                order.append(name)
                return original_filter(vocab)

            f.filter = tracking_filter
            return f

        chain = VocabularyFilterChain(
            [
                make_ordered("C", 30),
                make_ordered("A", 10),
                make_ordered("B", 20),
            ]
        )
        chain.run([{"Term": "test"}])
        assert order == ["A", "B", "C"]

    def test_disabled_filters_skipped(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        f_enabled = self._make_filter(
            "enabled", enabled=True, keep_fn=lambda v: v.get("Term") != "remove_me"
        )
        f_disabled = self._make_filter(
            "disabled", enabled=False, keep_fn=lambda v: False
        )  # Would remove everything

        chain = VocabularyFilterChain([f_enabled, f_disabled])
        vocab = [{"Term": "keep"}, {"Term": "remove_me"}]
        result = chain.run(vocab)
        assert len(result.vocabulary) == 1
        assert result.vocabulary[0]["Term"] == "keep"

    def test_add_filter(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        chain = VocabularyFilterChain()
        f = self._make_filter("test")
        returned = chain.add_filter(f)
        assert returned is chain  # Chaining
        assert len(chain.filters) == 1

    def test_remove_filter(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        f = self._make_filter("removable")
        chain = VocabularyFilterChain([f])
        assert chain.remove_filter("removable") is True
        assert len(chain.filters) == 0
        assert chain.remove_filter("nonexistent") is False

    def test_set_filter_enabled(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        f = self._make_filter("toggle", enabled=True)
        chain = VocabularyFilterChain([f])
        assert chain.set_filter_enabled("toggle", False) is True
        assert f.enabled is False
        assert chain.set_filter_enabled("nope", True) is False

    def test_get_filter(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        f = self._make_filter("findme")
        chain = VocabularyFilterChain([f])
        assert chain.get_filter("findme") is f
        assert chain.get_filter("nothere") is None

    def test_get_last_stats(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        chain = VocabularyFilterChain([self._make_filter("test")])
        assert chain.get_last_stats() is None

        chain.run([{"Term": "a"}])
        stats = chain.get_last_stats()
        assert stats is not None
        assert stats.total_input == 1

    def test_filter_error_does_not_crash_chain(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        f = self._make_filter("crasher")
        f.filter = MagicMock(side_effect=RuntimeError("boom"))

        chain = VocabularyFilterChain([f])
        vocab = [{"Term": "safe"}]
        result = chain.run(vocab)
        # Should pass through unchanged
        assert len(result.vocabulary) == 1
        stats = chain.get_last_stats()
        assert "error" in stats.per_filter_stats["crasher"]

    def test_repr(self):
        from src.core.vocabulary.filters.filter_chain import VocabularyFilterChain

        chain = VocabularyFilterChain([self._make_filter("A"), self._make_filter("B")])
        r = repr(chain)
        assert "A" in r
        assert "B" in r


# ---------------------------------------------------------------------------
# RegexExclusionFilter
# ---------------------------------------------------------------------------


class TestRegexExclusionFilter:
    """RegexExclusionFilter removes terms matching user-defined patterns."""

    def test_no_patterns_file_passes_through(self, tmp_path):
        from src.core.vocabulary.filters.regex_exclusion import RegexExclusionFilter

        f = RegexExclusionFilter(patterns_file=tmp_path / "nonexistent.txt")
        assert f.patterns == []
        vocab = [{"Term": "anything"}]
        result = f.filter(vocab)
        assert len(result.vocabulary) == 1

    def test_loads_patterns_from_file(self, tmp_path):
        from src.core.vocabulary.filters.regex_exclusion import RegexExclusionFilter

        pf = tmp_path / "patterns.txt"
        pf.write_text("^Q\\.\n^A\\.\n# comment\n\n^MR\\.", encoding="utf-8")

        f = RegexExclusionFilter(patterns_file=pf)
        assert len(f.patterns) == 3

    def test_filters_matching_terms(self, tmp_path):
        from src.core.vocabulary.filters.regex_exclusion import RegexExclusionFilter

        pf = tmp_path / "patterns.txt"
        pf.write_text("^Q\\.\n^A\\.", encoding="utf-8")

        f = RegexExclusionFilter(patterns_file=pf)
        vocab = [
            {"Term": "Q. Smith"},
            {"Term": "A. Jones"},
            {"Term": "plaintiff"},
            {"Term": "defendant"},
        ]
        result = f.filter(vocab)
        assert len(result.vocabulary) == 2
        assert result.removed_count == 2
        terms = [v["Term"] for v in result.vocabulary]
        assert "plaintiff" in terms
        assert "defendant" in terms

    def test_invalid_regex_skipped(self, tmp_path):
        from src.core.vocabulary.filters.regex_exclusion import RegexExclusionFilter

        pf = tmp_path / "patterns.txt"
        pf.write_text("valid_pattern\n[invalid\n", encoding="utf-8")

        f = RegexExclusionFilter(patterns_file=pf)
        assert len(f.patterns) == 1  # Only the valid one

    def test_case_insensitive_matching(self, tmp_path):
        from src.core.vocabulary.filters.regex_exclusion import RegexExclusionFilter

        pf = tmp_path / "patterns.txt"
        pf.write_text("^plaintiff$", encoding="utf-8")

        f = RegexExclusionFilter(patterns_file=pf)
        vocab = [{"Term": "PLAINTIFF"}, {"Term": "Plaintiff"}, {"Term": "defendant"}]
        result = f.filter(vocab)
        assert len(result.vocabulary) == 1

    def test_reload_patterns(self, tmp_path):
        from src.core.vocabulary.filters.regex_exclusion import RegexExclusionFilter

        pf = tmp_path / "patterns.txt"
        pf.write_text("^old$", encoding="utf-8")

        f = RegexExclusionFilter(patterns_file=pf)
        assert len(f.patterns) == 1

        pf.write_text("^new1$\n^new2$", encoding="utf-8")
        f.reload_patterns()
        assert len(f.patterns) == 2

    def test_attributes(self):
        from src.core.vocabulary.filters.regex_exclusion import RegexExclusionFilter

        assert RegexExclusionFilter.priority == 15
        assert RegexExclusionFilter.exempt_persons is False


# ---------------------------------------------------------------------------
# UnifiedPerTermFilter (combined_per_term)
# ---------------------------------------------------------------------------


class TestUnifiedPerTermFilter:
    """UnifiedPerTermFilter combines rarity + gibberish in single pass."""

    def test_attributes(self):
        from src.core.vocabulary.filters.combined_per_term import UnifiedPerTermFilter

        assert UnifiedPerTermFilter.name == "Combined Per-Term"
        assert UnifiedPerTermFilter.priority == 40
        assert UnifiedPerTermFilter.exempt_persons is True

    def test_empty_vocabulary(self):
        from src.core.vocabulary.filters.combined_per_term import UnifiedPerTermFilter

        f = UnifiedPerTermFilter()
        with (
            patch(
                "src.core.vocabulary.rarity_filter.should_filter_phrase",
                return_value=False,
            ),
            patch(
                "src.core.vocabulary.rarity_filter.should_passthrough_non_ner_term",
                return_value=False,
            ),
            patch("src.core.utils.gibberish_filter.is_gibberish", return_value=False),
            patch(
                "src.core.vocabulary.corpus_familiarity_filter.is_corpus_common_term",
                return_value=False,
            ),
        ):
            result = f.filter([])
        assert result.vocabulary == []
        assert result.removed_count == 0

    def test_keeps_valid_terms(self):
        from src.core.vocabulary.filters.combined_per_term import UnifiedPerTermFilter

        f = UnifiedPerTermFilter()
        vocab = [{"Term": "plaintiff", "Is Person": False}]

        with (
            patch(
                "src.core.vocabulary.rarity_filter.should_filter_phrase",
                return_value=False,
            ),
            patch(
                "src.core.vocabulary.rarity_filter.should_passthrough_non_ner_term",
                return_value=False,
            ),
            patch("src.core.utils.gibberish_filter.is_gibberish", return_value=False),
            patch(
                "src.core.vocabulary.corpus_familiarity_filter.is_corpus_common_term",
                return_value=False,
            ),
        ):
            result = f.filter(vocab)
        assert len(result.vocabulary) == 1

    def test_removes_gibberish(self):
        from src.core.vocabulary.filters.combined_per_term import UnifiedPerTermFilter

        f = UnifiedPerTermFilter()
        vocab = [{"Term": "xkjwqrmntplz", "Is Person": False}]

        with (
            patch(
                "src.core.vocabulary.rarity_filter.should_filter_phrase",
                return_value=False,
            ),
            patch(
                "src.core.vocabulary.rarity_filter.should_passthrough_non_ner_term",
                return_value=False,
            ),
            patch("src.core.utils.gibberish_filter.is_gibberish", return_value=True),
            patch(
                "src.core.vocabulary.corpus_familiarity_filter.is_corpus_common_term",
                return_value=False,
            ),
        ):
            result = f.filter(vocab)
        assert len(result.vocabulary) == 0
        assert result.metadata["gibberish_removed"] == 1

    def test_removes_common_rarity(self):
        from src.core.vocabulary.filters.combined_per_term import UnifiedPerTermFilter

        f = UnifiedPerTermFilter()
        vocab = [{"Term": "the", "Is Person": False}]

        with (
            patch(
                "src.core.vocabulary.rarity_filter.should_filter_phrase",
                return_value=True,
            ),
            patch(
                "src.core.vocabulary.rarity_filter.should_passthrough_non_ner_term",
                return_value=False,
            ),
            patch("src.core.utils.gibberish_filter.is_gibberish", return_value=False),
            patch(
                "src.core.vocabulary.corpus_familiarity_filter.is_corpus_common_term",
                return_value=False,
            ),
        ):
            result = f.filter(vocab)
        assert len(result.vocabulary) == 0
        assert result.metadata["rarity_removed"] == 1

    def test_adds_corpus_common_feature(self):
        from src.core.vocabulary.filters.combined_per_term import UnifiedPerTermFilter

        f = UnifiedPerTermFilter()
        vocab = [{"Term": "plaintiff", "Is Person": False}]

        with (
            patch(
                "src.core.vocabulary.rarity_filter.should_filter_phrase",
                return_value=False,
            ),
            patch(
                "src.core.vocabulary.rarity_filter.should_passthrough_non_ner_term",
                return_value=False,
            ),
            patch("src.core.utils.gibberish_filter.is_gibberish", return_value=False),
            patch(
                "src.core.vocabulary.corpus_familiarity_filter.is_corpus_common_term",
                return_value=True,
            ),
        ):
            result = f.filter(vocab)
        assert result.vocabulary[0]["corpus_common_term"] is True


# ---------------------------------------------------------------------------
# CorpusFamiliarityFilter
# ---------------------------------------------------------------------------


class TestCorpusFamiliarityFilter:
    """corpus_familiarity_filter module functions."""

    def test_is_corpus_common_term_returns_false_on_error(self):
        from src.core.vocabulary.corpus_familiarity_filter import is_corpus_common_term

        with patch(
            "src.core.vocabulary.corpus_manager.get_corpus_manager",
            side_effect=RuntimeError("no corpus"),
        ):
            assert is_corpus_common_term("plaintiff") is False

    def test_is_corpus_common_term_delegates(self):
        from src.core.vocabulary.corpus_familiarity_filter import is_corpus_common_term

        mock_mgr = MagicMock()
        mock_mgr.is_corpus_common_term.return_value = True

        with patch(
            "src.core.vocabulary.corpus_manager.get_corpus_manager",
            return_value=mock_mgr,
        ):
            assert is_corpus_common_term("plaintiff") is True

    def test_add_corpus_common_feature_empty(self):
        from src.core.vocabulary.corpus_familiarity_filter import add_corpus_common_feature

        result = add_corpus_common_feature([])
        assert result == []

    def test_add_corpus_common_feature_adds_key(self):
        from src.core.vocabulary.corpus_familiarity_filter import add_corpus_common_feature

        vocab = [{"Term": "test1"}, {"Term": "test2"}]
        with patch(
            "src.core.vocabulary.corpus_familiarity_filter.is_corpus_common_term",
            return_value=False,
        ):
            result = add_corpus_common_feature(vocab)
        assert all("corpus_common_term" in v for v in result)

    def test_backwards_compat_alias(self):
        from src.core.vocabulary.corpus_familiarity_filter import (
            add_corpus_common_feature,
            filter_corpus_familiar_terms,
        )

        vocab = [{"Term": "test"}]
        with patch(
            "src.core.vocabulary.corpus_familiarity_filter.is_corpus_common_term",
            return_value=False,
        ):
            r1 = add_corpus_common_feature(vocab.copy())
            r2 = filter_corpus_familiar_terms(vocab.copy())
        assert len(r1) == len(r2)


# ---------------------------------------------------------------------------
# GibberishFilter
# ---------------------------------------------------------------------------


class TestGibberishFilter:
    """GibberishFilter detects nonsense text."""

    def test_empty_not_gibberish(self):
        from src.core.utils.gibberish_filter import is_gibberish

        assert is_gibberish("") is False

    def test_short_text_not_gibberish(self):
        from src.core.utils.gibberish_filter import is_gibberish

        assert is_gibberish("cat") is False
        assert is_gibberish("the") is False

    def test_real_word_not_gibberish(self):
        from src.core.utils.gibberish_filter import is_gibberish

        assert is_gibberish("plaintiff") is False
        assert is_gibberish("defendant") is False
        assert is_gibberish("hospital") is False

    def test_obvious_gibberish_detected(self):
        from src.core.utils.gibberish_filter import is_gibberish

        assert is_gibberish("xkjwqrmntplz") is True
        assert is_gibberish("zzzzqqqxxx") is True

    def test_singleton_pattern(self):
        from src.core.utils.gibberish_filter import GibberishFilter

        g1 = GibberishFilter.get_instance()
        g2 = GibberishFilter.get_instance()
        assert g1 is g2

    def test_clean_for_check(self):
        from src.core.utils.gibberish_filter import GibberishFilter

        g = GibberishFilter()
        assert g._clean_for_check("Hello!") == "hello"
        assert g._clean_for_check("it's") == "its"
        assert g._clean_for_check("123abc") == "abc"

    def test_multiword_gibberish(self):
        from src.core.utils.gibberish_filter import is_gibberish

        # If any word is gibberish, the whole phrase is
        assert is_gibberish("modmess quanny") is True


# ---------------------------------------------------------------------------
# PatternFilter
# ---------------------------------------------------------------------------


class TestPatternFilter:
    """PatternFilter and pre-built filters."""

    def test_search_method(self):
        from src.core.utils.pattern_filter import MatchMethod, PatternFilter

        pf = PatternFilter(patterns=(r"\d+",), method=MatchMethod.SEARCH)
        assert pf.matches("abc 123 def") is True
        assert pf.matches("no digits here") is False

    def test_match_method(self):
        from src.core.utils.pattern_filter import MatchMethod, PatternFilter

        pf = PatternFilter(patterns=(r"\d+",), method=MatchMethod.MATCH)
        assert pf.matches("123 abc") is True
        assert pf.matches("abc 123") is False  # Doesn't match at start

    def test_fullmatch_method(self):
        from src.core.utils.pattern_filter import MatchMethod, PatternFilter

        pf = PatternFilter(patterns=(r"\d+",), method=MatchMethod.FULLMATCH)
        assert pf.matches("123") is True
        assert pf.matches("123 abc") is False

    def test_case_insensitive_default(self):
        from src.core.utils.pattern_filter import PatternFilter

        pf = PatternFilter(patterns=(r"hello",))
        assert pf.matches("HELLO") is True
        assert pf.matches("Hello") is True

    def test_case_sensitive(self):
        from src.core.utils.pattern_filter import PatternFilter

        pf = PatternFilter(patterns=(r"hello",), case_sensitive=True)
        assert pf.matches("hello") is True
        assert pf.matches("HELLO") is False

    def test_multiple_patterns_any_match(self):
        from src.core.utils.pattern_filter import PatternFilter

        pf = PatternFilter(patterns=(r"foo", r"bar"))
        assert pf.matches("foo") is True
        assert pf.matches("bar") is True
        assert pf.matches("baz") is False


class TestPreBuiltFilters:
    """Pre-built filter instances in pattern_filter module."""

    def test_address_filter(self):
        from src.core.utils.pattern_filter import ADDRESS_FILTER

        assert ADDRESS_FILTER.matches("123 Main Street") is True
        assert ADDRESS_FILTER.matches("5th Floor") is True
        assert ADDRESS_FILTER.matches("John Smith") is False

    def test_legal_boilerplate_filter(self):
        from src.core.utils.pattern_filter import LEGAL_BOILERPLATE_FILTER

        assert LEGAL_BOILERPLATE_FILTER.matches("Verified Complaint") is True
        assert LEGAL_BOILERPLATE_FILTER.matches("Notice of Commencement") is True
        assert LEGAL_BOILERPLATE_FILTER.matches("John Smith") is False

    def test_case_citation_filter(self):
        from src.core.utils.pattern_filter import CASE_CITATION_FILTER

        assert CASE_CITATION_FILTER.matches("Smith v. Jones") is True
        assert CASE_CITATION_FILTER.matches("Smith v Jones") is True
        assert CASE_CITATION_FILTER.matches("simple text") is False

    def test_geographic_code_filter(self):
        from src.core.utils.pattern_filter import GEOGRAPHIC_CODE_FILTER

        assert GEOGRAPHIC_CODE_FILTER.matches("12345") is True
        assert GEOGRAPHIC_CODE_FILTER.matches("12345-6789") is True
        assert GEOGRAPHIC_CODE_FILTER.matches("hello") is False

    def test_acronym_filter(self):
        from src.core.utils.pattern_filter import ACRONYM_FILTER

        assert ACRONYM_FILTER.matches("FBI") is True
        assert ACRONYM_FILTER.matches("CIA") is True
        assert ACRONYM_FILTER.matches("John") is False

    def test_ocr_error_filter(self):
        from src.core.utils.pattern_filter import OCR_ERROR_FILTER

        assert OCR_ERROR_FILTER.matches("3ohn5mith") is True
        assert OCR_ERROR_FILTER.matches("Joh3n") is True
        assert OCR_ERROR_FILTER.matches("John") is False


class TestEntityAndTokenFilters:
    """matches_entity_filter and matches_token_filter functions."""

    def test_entity_too_short(self):
        from src.core.utils.pattern_filter import matches_entity_filter

        assert matches_entity_filter("ab") is True  # < MIN_ENTITY_LENGTH

    def test_entity_too_long(self):
        from src.core.utils.pattern_filter import matches_entity_filter

        assert matches_entity_filter("x" * 61) is True  # > MAX_ENTITY_LENGTH

    def test_entity_valid_name(self):
        from src.core.utils.pattern_filter import matches_entity_filter

        assert matches_entity_filter("John Smith") is False

    def test_entity_address_filtered(self):
        from src.core.utils.pattern_filter import matches_entity_filter

        assert matches_entity_filter("123 Main Street") is True

    def test_is_valid_acronym(self):
        from src.core.utils.pattern_filter import is_valid_acronym

        assert is_valid_acronym("FBI") is True
        assert is_valid_acronym("DR") is False  # Title abbreviation
        assert is_valid_acronym("john") is False  # Lowercase
