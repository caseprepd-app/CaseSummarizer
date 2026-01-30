"""Tests for corpus_manager.py"""

from src.core.vocabulary.corpus_manager import CorpusManager


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
