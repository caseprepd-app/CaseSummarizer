"""Tests for corpus_manager.py"""

from src.core.vocabulary.corpus_manager import (
    CorpusManager,
)


class TestEmptyCorpus:
    def test_empty_corpus_count(self, tmp_path):
        mgr = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        assert mgr.get_document_count() == 0

    def test_empty_corpus_not_ready(self, tmp_path):
        mgr = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        assert not mgr.is_corpus_ready()


class TestDocumentCount:
    def test_counts_supported_extensions(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc1.pdf").write_text("x")
        (corpus / "doc2.txt").write_text("x")
        (corpus / "doc3.csv").write_text("x")  # unsupported
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        assert mgr.get_document_count() == 2

    def test_counts_rtf(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc1.rtf").write_text("x")
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        assert mgr.get_document_count() == 1

    def test_is_ready_with_enough_docs(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(5):
            (corpus / f"doc{i}.pdf").write_text("x")
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        assert mgr.is_corpus_ready(min_docs=3)


class TestIdfLookup:
    def test_oov_returns_default(self, tmp_path):
        mgr = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        mgr._idf_index = {"hello": 1.5}
        mgr._doc_freq = {"hello": 3}
        assert mgr.get_idf("unknown_term") == 10.0

    def test_known_term_returns_idf(self, tmp_path):
        mgr = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        mgr._idf_index = {"hello": 1.5}
        mgr._doc_freq = {"hello": 3}
        assert mgr.get_idf("hello") == 1.5

    def test_doc_freq_oov(self, tmp_path):
        mgr = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        mgr._doc_freq = {"hello": 3}
        assert mgr.get_doc_freq("unknown") == 0

    def test_doc_freq_known(self, tmp_path):
        mgr = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        mgr._doc_freq = {"hello": 3}
        assert mgr.get_doc_freq("hello") == 3


class TestCacheInvalidation:
    def test_invalidate_clears_index(self, tmp_path):
        mgr = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        mgr._idf_index = {"test": 1.0}
        mgr._corpus_hash = "abc"
        mgr.invalidate_cache()
        assert mgr._idf_index == {}
        assert mgr._corpus_hash is None


class TestCorpusHash:
    def test_hash_changes_with_new_file(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc1.pdf").write_text("x")
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        hash1 = mgr._compute_corpus_hash()

        (corpus / "doc2.pdf").write_text("y")
        hash2 = mgr._compute_corpus_hash()
        assert hash1 != hash2

    def test_hash_stable_without_changes(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc1.pdf").write_text("x")
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        hash1 = mgr._compute_corpus_hash()
        hash2 = mgr._compute_corpus_hash()
        assert hash1 == hash2


class TestCorpusLimit:
    """Tests for the 25-document corpus limit."""

    def test_constants_exist(self):
        from src.core.vocabulary.corpus_manager import (
            CORPUS_COMMON_MIN_OCCURRENCES,
            CORPUS_COMMON_THRESHOLD,
            MAX_CORPUS_DOCS,
            MIN_CORPUS_DOCS,
        )

        assert MAX_CORPUS_DOCS == 25
        assert MIN_CORPUS_DOCS == 5
        assert CORPUS_COMMON_THRESHOLD == 0.64
        assert CORPUS_COMMON_MIN_OCCURRENCES == 5

    def test_corpus_not_disabled_under_limit(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(10):
            (corpus / f"doc{i}.pdf").write_text("x")
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        assert not mgr.is_corpus_disabled()
        assert mgr.get_disabled_reason() is None

    def test_corpus_disabled_over_limit(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(30):  # Over the 25 limit
            (corpus / f"doc{i}.pdf").write_text("x")
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        assert mgr.is_corpus_disabled()
        reason = mgr.get_disabled_reason()
        assert reason is not None
        assert "30" in reason
        assert "25" in reason
        assert "disabled" in reason.lower()

    def test_corpus_not_ready_when_disabled(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(30):  # Over the 25 limit but also >= 5
            (corpus / f"doc{i}.pdf").write_text("x")
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        # Even though we have >= 5 docs, corpus should not be "ready" if disabled
        assert not mgr.is_corpus_ready()

    def test_can_add_documents_under_limit(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(20):
            (corpus / f"doc{i}.pdf").write_text("x")
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        can_add, error = mgr.can_add_documents(5)
        assert can_add
        assert error is None

    def test_cannot_add_documents_over_limit(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(20):
            (corpus / f"doc{i}.pdf").write_text("x")
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        can_add, error = mgr.can_add_documents(10)  # Would go to 30
        assert not can_add
        assert error is not None
        assert "30" in error


class TestCorpusCommonTerm:
    """Tests for the corpus_common_term binary feature."""

    def test_common_term_false_when_disabled(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(30):  # Over limit
            (corpus / f"doc{i}.pdf").write_text("x")
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        assert mgr.is_corpus_disabled()
        assert not mgr.is_corpus_common_term("anything")

    def test_common_term_false_when_not_ready(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # Only 2 docs, need at least 5
        for i in range(2):
            (corpus / f"doc{i}.pdf").write_text("x")
        mgr = CorpusManager(corpus_dir=corpus, cache_dir=tmp_path / "cache")
        assert not mgr.is_corpus_common_term("anything")

    def test_common_term_false_when_below_min_occurrences(self, tmp_path):
        mgr = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        # Set up: 10 docs indexed, term appears in only 4 (below 5 min)
        mgr._doc_count = 10
        mgr._doc_freq = {"rare_term": 4}
        mgr._idf_index = {"rare_term": 1.0}
        assert not mgr.is_corpus_common_term("rare_term")

    def test_common_term_false_when_below_threshold(self, tmp_path):
        mgr = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        # Set up: 25 docs indexed, term appears in 10 (40%, below 64%)
        mgr._doc_count = 25
        mgr._doc_freq = {"uncommon_term": 10}
        mgr._idf_index = {"uncommon_term": 1.0}
        assert not mgr.is_corpus_common_term("uncommon_term")

    def test_common_term_true_when_above_threshold_and_min_occurrences(self, tmp_path):
        mgr = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        # Set up: 25 docs indexed, term appears in 16 (64%, exactly at threshold)
        mgr._doc_count = 25
        mgr._doc_freq = {"common_term": 16}
        mgr._idf_index = {"common_term": 0.5}
        assert mgr.is_corpus_common_term("common_term")

    def test_common_term_at_minimum_corpus_size(self, tmp_path):
        mgr = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        # Set up: 5 docs (minimum), term appears in 4 (80% > 64%, and >= 5 total)
        # But 4 < 5 min occurrences, so should be False
        mgr._doc_count = 5
        mgr._doc_freq = {"term": 4}
        mgr._idf_index = {"term": 1.0}
        assert not mgr.is_corpus_common_term("term")

        # Now with 5 occurrences (100%, meets both criteria)
        mgr._doc_freq = {"term": 5}
        assert mgr.is_corpus_common_term("term")
