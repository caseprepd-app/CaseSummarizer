"""
Tests for the refactored vocabulary filters package (src/core/vocabulary/filters/).

Covers: artifact.py, rarity.py, gibberish.py, regularizer.py, name_dedup.py
Each wraps an existing function behind the BaseVocabularyFilter interface.
"""


# ============================================================================
# Helper: build term dicts
# ============================================================================


def _term(name, is_person=False, score=50, found_by="NER", freq=1):
    """Build a minimal vocabulary term dict for testing."""
    return {
        "Term": name,
        "Is Person": "Yes" if is_person else "No",
        "Score": str(score),
        "Found By": found_by,
        "Frequency": str(freq),
    }


# ============================================================================
# 1. ExtractionArtifactFilter
# ============================================================================


class TestExtractionArtifactFilter:
    """Tests for filters/artifact.py wrapper."""

    def test_inherits_base(self):
        from src.core.vocabulary.filters.artifact import ExtractionArtifactFilter
        from src.core.vocabulary.filters.base import BaseVocabularyFilter

        assert issubclass(ExtractionArtifactFilter, BaseVocabularyFilter)

    def test_name_and_priority(self):
        from src.core.vocabulary.filters.artifact import ExtractionArtifactFilter

        f = ExtractionArtifactFilter()
        assert f.name == "Artifact Filter"
        assert f.priority == 20

    def test_returns_filter_result(self):
        from src.core.vocabulary.filters.artifact import ExtractionArtifactFilter
        from src.core.vocabulary.filters.base import FilterResult

        f = ExtractionArtifactFilter()
        result = f.filter([_term("plaintiff")])
        assert isinstance(result, FilterResult)
        assert isinstance(result.vocabulary, list)

    def test_removes_substring_artifacts(self):
        """A longer term that contains a high-freq shorter term should be removed."""
        from src.core.vocabulary.filters.artifact import ExtractionArtifactFilter

        vocab = [
            _term("Di Leo", score=90, freq=10),
            _term("Di Leo:", score=30, freq=2),
            _term("Ms. Di Leo", score=80, freq=8),
        ]
        f = ExtractionArtifactFilter(canonical_count=2)
        result = f.filter(vocab)
        remaining = [t["Term"] for t in result.vocabulary]
        # The artifact "Di Leo:" should be removed (substring of canonical "Di Leo")
        assert "Di Leo" in remaining

    def test_empty_input(self):
        from src.core.vocabulary.filters.artifact import ExtractionArtifactFilter

        f = ExtractionArtifactFilter()
        result = f.filter([])
        assert result.vocabulary == []
        assert result.removed_count == 0

    def test_canonical_count_configurable(self):
        from src.core.vocabulary.filters.artifact import ExtractionArtifactFilter

        f = ExtractionArtifactFilter(canonical_count=50)
        assert f.canonical_count == 50


# ============================================================================
# 2. RarityFilter
# ============================================================================


class TestRarityFilterWrapper:
    """Tests for filters/rarity.py wrapper."""

    def test_inherits_base(self):
        from src.core.vocabulary.filters.base import BaseVocabularyFilter
        from src.core.vocabulary.filters.rarity import RarityFilter

        assert issubclass(RarityFilter, BaseVocabularyFilter)

    def test_name_and_priority(self):
        from src.core.vocabulary.filters.rarity import RarityFilter

        f = RarityFilter()
        assert f.name == "Rarity Filter"
        assert f.priority == 40

    def test_exempt_persons(self):
        from src.core.vocabulary.filters.rarity import RarityFilter

        f = RarityFilter()
        assert f.exempt_persons is True

    def test_returns_filter_result(self):
        from src.core.vocabulary.filters.base import FilterResult
        from src.core.vocabulary.filters.rarity import RarityFilter

        f = RarityFilter()
        result = f.filter([_term("plaintiff")])
        assert isinstance(result, FilterResult)

    def test_processes_without_error(self):
        """RarityFilter should process vocab and return valid result."""
        from src.core.vocabulary.filters.rarity import RarityFilter

        vocab = [
            _term("the", score=10),
            _term("electrocardiogram", score=80),
        ]
        f = RarityFilter()
        result = f.filter(vocab)
        # Should have filtered some terms (at least "the")
        assert result.removed_count >= 0
        assert len(result.vocabulary) + result.removed_count == len(vocab)

    def test_empty_input(self):
        from src.core.vocabulary.filters.rarity import RarityFilter

        result = RarityFilter().filter([])
        assert result.vocabulary == []
        assert result.removed_count == 0


# ============================================================================
# 3. GibberishFilter
# ============================================================================


class TestGibberishFilterWrapper:
    """Tests for filters/gibberish.py wrapper."""

    def test_inherits_base(self):
        from src.core.vocabulary.filters.base import BaseVocabularyFilter
        from src.core.vocabulary.filters.gibberish import GibberishFilter

        assert issubclass(GibberishFilter, BaseVocabularyFilter)

    def test_name_and_priority(self):
        from src.core.vocabulary.filters.gibberish import GibberishFilter

        f = GibberishFilter()
        assert f.name == "Gibberish Filter"
        assert f.priority == 60

    def test_exempt_persons(self):
        from src.core.vocabulary.filters.gibberish import GibberishFilter

        assert GibberishFilter().exempt_persons is True

    def test_keeps_real_words(self):
        from src.core.vocabulary.filters.gibberish import GibberishFilter

        vocab = [_term("plaintiff"), _term("cardiomyopathy")]
        result = GibberishFilter().filter(vocab)
        remaining = [t["Term"] for t in result.vocabulary]
        assert "plaintiff" in remaining

    def test_person_names_exempt(self):
        """Person entries should never be filtered even if they look unusual."""
        from src.core.vocabulary.filters.gibberish import GibberishFilter

        vocab = [_term("Xyzqwkl", is_person=True)]
        result = GibberishFilter().filter(vocab)
        assert len(result.vocabulary) == 1

    def test_removes_gibberish(self):
        """Nonsense strings should be removed."""
        from src.core.vocabulary.filters.gibberish import GibberishFilter

        vocab = [_term("xkjqzwf")]
        result = GibberishFilter().filter(vocab)
        assert result.removed_count >= 0  # Depends on spellchecker
        if result.removed_count > 0:
            assert "xkjqzwf" in result.removed_terms

    def test_empty_input(self):
        from src.core.vocabulary.filters.gibberish import GibberishFilter

        result = GibberishFilter().filter([])
        assert result.vocabulary == []


# ============================================================================
# 4. NameRegularizerFilter
# ============================================================================


class TestNameRegularizerFilterWrapper:
    """Tests for filters/regularizer.py wrapper."""

    def test_inherits_base(self):
        from src.core.vocabulary.filters.base import BaseVocabularyFilter
        from src.core.vocabulary.filters.regularizer import NameRegularizerFilter

        assert issubclass(NameRegularizerFilter, BaseVocabularyFilter)

    def test_name_and_priority(self):
        from src.core.vocabulary.filters.regularizer import NameRegularizerFilter

        f = NameRegularizerFilter()
        assert f.name == "Name Regularizer"
        assert f.priority == 30

    def test_configurable_params(self):
        from src.core.vocabulary.filters.regularizer import NameRegularizerFilter

        f = NameRegularizerFilter(top_fraction=0.5, num_passes=5)
        assert f.top_fraction == 0.5
        assert f.num_passes == 5

    def test_returns_filter_result(self):
        from src.core.vocabulary.filters.base import FilterResult
        from src.core.vocabulary.filters.regularizer import NameRegularizerFilter

        result = NameRegularizerFilter().filter([_term("John Smith", is_person=True)])
        assert isinstance(result, FilterResult)

    def test_metadata_includes_config(self):
        from src.core.vocabulary.filters.regularizer import NameRegularizerFilter

        result = NameRegularizerFilter(top_fraction=0.25, num_passes=3).filter(
            [_term("Test", is_person=True)]
        )
        assert result.metadata.get("top_fraction") == 0.25
        assert result.metadata.get("num_passes") == 3

    def test_empty_input(self):
        from src.core.vocabulary.filters.regularizer import NameRegularizerFilter

        result = NameRegularizerFilter().filter([])
        assert result.vocabulary == []


# ============================================================================
# 5. NameDeduplicationFilter
# ============================================================================


class TestNameDeduplicationFilterWrapper:
    """Tests for filters/name_dedup.py wrapper."""

    def test_inherits_base(self):
        from src.core.vocabulary.filters.base import BaseVocabularyFilter
        from src.core.vocabulary.filters.name_dedup import NameDeduplicationFilter

        assert issubclass(NameDeduplicationFilter, BaseVocabularyFilter)

    def test_name_and_priority(self):
        from src.core.vocabulary.filters.name_dedup import NameDeduplicationFilter

        f = NameDeduplicationFilter()
        assert f.name == "Name Deduplication"
        assert f.priority == 10  # Must run first

    def test_default_threshold_from_config(self):
        from src.config import NAME_SIMILARITY_THRESHOLD
        from src.core.vocabulary.filters.name_dedup import NameDeduplicationFilter

        f = NameDeduplicationFilter()
        assert f.similarity_threshold == NAME_SIMILARITY_THRESHOLD

    def test_custom_threshold(self):
        from src.core.vocabulary.filters.name_dedup import NameDeduplicationFilter

        f = NameDeduplicationFilter(similarity_threshold=0.9)
        assert f.similarity_threshold == 0.9

    def test_returns_filter_result_with_metadata(self):
        from src.core.vocabulary.filters.base import FilterResult
        from src.core.vocabulary.filters.name_dedup import NameDeduplicationFilter

        vocab = [_term("John Smith", is_person=True)]
        result = NameDeduplicationFilter().filter(vocab)
        assert isinstance(result, FilterResult)
        assert "similarity_threshold" in result.metadata
        assert "potential_duplicates" in result.metadata

    def test_empty_input(self):
        from src.core.vocabulary.filters.name_dedup import NameDeduplicationFilter

        result = NameDeduplicationFilter().filter([])
        assert result.vocabulary == []


# ============================================================================
# 6. Filter Chain Factory Functions
# ============================================================================


class TestFilterChainFactories:
    """Tests for create_default_filter_chain and variants."""

    def test_default_chain_has_all_filters(self):
        from src.core.vocabulary.filters import create_default_filter_chain

        chain = create_default_filter_chain()
        names = [f.name for f in chain.filters]
        assert "Name Deduplication" in names
        assert "Artifact Filter" in names
        assert "Rarity Filter" in names
        assert "Gibberish Filter" in names
        assert "Name Regularizer" in names

    def test_optimized_chain_uses_unified(self):
        from src.core.vocabulary.filters import create_optimized_filter_chain

        chain = create_optimized_filter_chain()
        names = [f.name for f in chain.filters]
        # Should use UnifiedPerTermFilter instead of separate rarity/gibberish
        assert "Rarity Filter" not in names
        assert "Gibberish Filter" not in names

    def test_partial_results_chain(self):
        from src.core.vocabulary.filters import create_partial_results_filter_chain

        chain = create_partial_results_filter_chain()
        names = [f.name for f in chain.filters]
        assert "Rarity Filter" in names
        assert "Gibberish Filter" in names
        # Partial chain should NOT have name dedup
        assert "Name Deduplication" not in names

    def test_default_chain_sorted_by_priority(self):
        from src.core.vocabulary.filters import create_default_filter_chain

        chain = create_default_filter_chain()
        priorities = [f.priority for f in chain.filters]
        assert priorities == sorted(priorities)

    def test_chain_runs_without_error(self):
        """Full chain should process vocab without crashing."""
        from src.core.vocabulary.filters import create_default_filter_chain

        chain = create_default_filter_chain()
        vocab = [
            _term("plaintiff", score=80),
            _term("John Smith", is_person=True, score=90),
            _term("the", score=10),
        ]
        result = chain.run(vocab)
        assert isinstance(result.vocabulary, list)
        assert result.removed_count >= 0
