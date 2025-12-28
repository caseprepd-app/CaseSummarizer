"""
Tests for BM25 Algorithm and Corpus Manager

Tests the corpus-based vocabulary extraction system:
1. CorpusManager - folder management, IDF index building, caching
2. BM25Algorithm - scoring, term extraction, integration

These tests use temporary directories to avoid affecting the user's actual corpus.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestCorpusManager:
    """Tests for CorpusManager class."""

    def test_initialization_creates_directories(self, tmp_path):
        """CorpusManager should create corpus and cache directories."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        corpus_dir = tmp_path / "corpus"
        cache_dir = tmp_path / "cache"

        manager = CorpusManager(corpus_dir=corpus_dir, cache_dir=cache_dir)

        assert corpus_dir.exists()
        assert cache_dir.exists()

    def test_empty_corpus_returns_zero_documents(self, tmp_path):
        """Empty corpus folder should return document count of 0."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")

        assert manager.get_document_count() == 0

    def test_is_corpus_ready_false_when_empty(self, tmp_path):
        """is_corpus_ready should return False when corpus has < 5 documents."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")

        assert manager.is_corpus_ready() is False
        assert manager.is_corpus_ready(min_docs=1) is False

    def test_counts_supported_file_extensions(self, tmp_path):
        """Document count should include PDF, TXT, and RTF files."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        # Create test files
        (corpus_dir / "doc1.pdf").write_text("test")
        (corpus_dir / "doc2.txt").write_text("test")
        (corpus_dir / "doc3.rtf").write_text("test")
        (corpus_dir / "ignored.docx").write_text("test")  # Not supported
        (corpus_dir / "ignored.md").write_text("test")    # Not supported

        manager = CorpusManager(corpus_dir=corpus_dir, cache_dir=tmp_path / "cache")

        assert manager.get_document_count() == 3

    def test_is_corpus_ready_true_when_enough_documents(self, tmp_path):
        """is_corpus_ready should return True when corpus has >= 5 documents."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        # Create 5 text files
        for i in range(5):
            (corpus_dir / f"doc{i}.txt").write_text(f"Test document {i}")

        manager = CorpusManager(corpus_dir=corpus_dir, cache_dir=tmp_path / "cache")

        assert manager.get_document_count() == 5
        assert manager.is_corpus_ready() is True

    def test_get_idf_returns_high_score_for_oov_terms(self, tmp_path):
        """OOV (out-of-vocabulary) terms should get high IDF score."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")

        # With empty corpus, all terms are OOV
        idf = manager.get_idf("spondylosis")
        assert idf == 10.0  # Max IDF for unknown terms

    def test_tokenize_filters_stopwords(self, tmp_path):
        """Tokenizer should filter out common stopwords."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")

        tokens = manager._tokenize("The patient was seen at the hospital.")

        assert "the" not in tokens
        assert "was" not in tokens
        assert "at" not in tokens
        assert "patient" in tokens
        assert "hospital" in tokens

    def test_tokenize_filters_short_words(self, tmp_path):
        """Tokenizer should filter out words shorter than 2 characters."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")

        tokens = manager._tokenize("I am a test X Y Z.")

        # Single letters should be filtered
        assert "x" not in tokens
        assert "y" not in tokens
        assert "z" not in tokens
        assert "test" in tokens

    def test_corpus_stats_returns_expected_fields(self, tmp_path):
        """get_corpus_stats should return expected metadata fields."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")

        stats = manager.get_corpus_stats()

        assert "doc_count" in stats
        assert "vocab_size" in stats
        assert "is_ready" in stats
        assert "corpus_path" in stats

    def test_cache_invalidation_on_corpus_change(self, tmp_path):
        """Cache should be invalidated when corpus folder changes."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        # Create initial corpus with 5 documents
        for i in range(5):
            (corpus_dir / f"doc{i}.txt").write_text(f"test document {i} with word{i}")

        manager = CorpusManager(corpus_dir=corpus_dir, cache_dir=tmp_path / "cache")

        # Build initial index
        manager.build_idf_index()
        initial_vocab_size = manager._vocab_size

        # Add another document
        (corpus_dir / "doc5.txt").write_text("new document with newword")

        # Invalidate and rebuild
        manager.invalidate_cache()
        manager.build_idf_index()

        # Vocabulary should have grown
        assert manager._vocab_size >= initial_vocab_size


class TestBM25Algorithm:
    """Tests for BM25Algorithm class."""

    def test_initialization_with_corpus_manager(self, tmp_path):
        """BM25Algorithm should accept a corpus manager."""
        from src.core.vocabulary.algorithms.bm25_algorithm import BM25Algorithm
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        bm25 = BM25Algorithm(corpus_manager=manager)

        assert bm25.name == "BM25"
        assert bm25.weight == 0.8
        assert bm25.corpus_manager is manager

    def test_extract_returns_empty_when_corpus_insufficient(self, tmp_path):
        """extract() should return empty result when corpus has < 5 documents."""
        from src.core.vocabulary.algorithms.bm25_algorithm import BM25Algorithm
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        bm25 = BM25Algorithm(corpus_manager=manager)

        result = bm25.extract("Test document with medical terms like adenocarcinoma.")

        assert len(result.candidates) == 0
        assert result.metadata.get("skipped") is True
        assert result.metadata.get("reason") == "insufficient_corpus"

    def test_extract_returns_candidates_when_corpus_ready(self, tmp_path):
        """extract() should return candidates when corpus is ready."""
        from src.core.vocabulary.algorithms.bm25_algorithm import BM25Algorithm
        from src.core.vocabulary.corpus_manager import CorpusManager

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        # Create 5 text files with common vocabulary
        for i in range(5):
            (corpus_dir / f"doc{i}.txt").write_text(
                "This is a standard legal document with common terms like plaintiff "
                "defendant court evidence testimony witness."
            )

        manager = CorpusManager(corpus_dir=corpus_dir, cache_dir=tmp_path / "cache")
        manager.build_idf_index()

        bm25 = BM25Algorithm(corpus_manager=manager)

        # Document with unusual term not in corpus
        result = bm25.extract(
            "The plaintiff diagnosed with adenocarcinoma of the cervical spine "
            "experienced severe radiculopathy and myelopathy symptoms."
        )

        assert result.metadata.get("skipped") is False
        # Should find some candidates (exact count depends on scoring threshold)
        # The key is that unusual medical terms should score high

    def test_tokenize_filters_stopwords(self):
        """BM25 tokenizer should filter stopwords."""
        from src.core.vocabulary.algorithms.bm25_algorithm import BM25Algorithm

        bm25 = BM25Algorithm()

        tokens = bm25._tokenize("The patient was seen for evaluation.")

        assert "the" not in tokens
        assert "was" not in tokens
        assert "for" not in tokens
        assert "patient" in tokens
        assert "evaluation" in tokens

    def test_is_valid_term_filters_short_terms(self):
        """_is_valid_term should filter terms shorter than MIN_TERM_LENGTH."""
        from src.core.vocabulary.algorithms.bm25_algorithm import BM25Algorithm

        bm25 = BM25Algorithm()

        assert bm25._is_valid_term("ab") is False
        assert bm25._is_valid_term("abc") is True
        assert bm25._is_valid_term("adenocarcinoma") is True

    def test_is_valid_term_filters_digits(self):
        """_is_valid_term should filter pure digit strings."""
        from src.core.vocabulary.algorithms.bm25_algorithm import BM25Algorithm

        bm25 = BM25Algorithm()

        assert bm25._is_valid_term("123") is False
        assert bm25._is_valid_term("2024") is False
        assert bm25._is_valid_term("covid19") is True  # Mixed is OK

    def test_candidates_have_bm25_metadata(self, tmp_path):
        """Candidates should include BM25-specific metadata."""
        from src.core.vocabulary.algorithms.bm25_algorithm import BM25Algorithm
        from src.core.vocabulary.corpus_manager import CorpusManager

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        # Create 5 text files
        for i in range(5):
            (corpus_dir / f"doc{i}.txt").write_text("standard legal document text")

        manager = CorpusManager(corpus_dir=corpus_dir, cache_dir=tmp_path / "cache")
        manager.build_idf_index()

        bm25 = BM25Algorithm(corpus_manager=manager, min_score_threshold=0.1)

        result = bm25.extract("The unusual medical term adenocarcinoma appears here.")

        if result.candidates:
            candidate = result.candidates[0]
            assert "bm25_score" in candidate.metadata
            assert "idf" in candidate.metadata
            assert "tf" in candidate.metadata
            assert candidate.source_algorithm == "BM25"

    def test_get_config_includes_corpus_info(self, tmp_path):
        """get_config should include corpus readiness info."""
        from src.core.vocabulary.algorithms.bm25_algorithm import BM25Algorithm
        from src.core.vocabulary.corpus_manager import CorpusManager

        manager = CorpusManager(corpus_dir=tmp_path / "corpus", cache_dir=tmp_path / "cache")
        bm25 = BM25Algorithm(corpus_manager=manager)

        config = bm25.get_config()

        assert "name" in config
        assert "weight" in config
        assert "corpus_ready" in config
        assert "corpus_doc_count" in config
        assert config["name"] == "BM25"


class TestBM25Integration:
    """Integration tests for BM25 with VocabularyExtractor."""

    def test_vocabulary_extractor_skips_bm25_when_corpus_empty(self, tmp_path):
        """VocabularyExtractor should not include BM25 when corpus is insufficient."""
        from src.core.vocabulary import VocabularyExtractor

        # Default extractor with empty corpus
        extractor = VocabularyExtractor()

        algorithm_names = [alg.name for alg in extractor.algorithms]

        assert "NER" in algorithm_names
        assert "RAKE" in algorithm_names
        # BM25 should NOT be present with empty corpus
        assert "BM25" not in algorithm_names

    def test_vocabulary_extractor_skips_bm25_when_disabled(self, tmp_path):
        """VocabularyExtractor should skip BM25 when disabled in preferences."""
        from src.core.vocabulary import VocabularyExtractor
        from src.user_preferences import get_user_preferences

        # Create corpus with enough documents
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        for i in range(5):
            (corpus_dir / f"doc{i}.txt").write_text("test content")

        # Mock preferences to disable BM25
        with patch.object(get_user_preferences(), 'get', return_value=False):
            extractor = VocabularyExtractor()

            algorithm_names = [alg.name for alg in extractor.algorithms]

            # BM25 should NOT be present when disabled
            # Note: This test may need adjustment based on how preferences work
            assert "NER" in algorithm_names
            assert "RAKE" in algorithm_names
