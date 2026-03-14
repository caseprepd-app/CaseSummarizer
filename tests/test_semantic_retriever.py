"""
Tests for SemanticRetriever.

Covers the hybrid retrieval layer that combines BM25+ and FAISS results:
- SourceInfo and RetrievalResult dataclasses
- Module-level helpers: _get_effective_semantic_context_window,
  _get_effective_algorithm_weights (fallback paths)
- SemanticRetriever.__init__: raises FileNotFoundError when index is missing
- _verify_integrity_hash: skips when no .hash file, passes on match,
  raises on tampered files
- get_relevant_sources_summary: formats source list correctly
- get_chunk_count: delegates to hybrid retriever
- get_algorithm_status: delegates to hybrid retriever
"""

from unittest.mock import MagicMock, patch

import pytest


class TestSourceInfo:
    """Tests for the SourceInfo dataclass."""

    def test_required_fields_stored(self):
        """SourceInfo stores all retrieval metadata."""
        from src.core.vector_store.semantic_retriever import SourceInfo

        info = SourceInfo(
            filename="complaint.pdf",
            chunk_num=3,
            section="Introduction",
            relevance_score=0.87,
            word_count=120,
        )

        assert info.filename == "complaint.pdf"
        assert info.chunk_num == 3
        assert info.section == "Introduction"
        assert info.relevance_score == 0.87
        assert info.word_count == 120

    def test_sources_defaults_to_none(self):
        """sources field defaults to None (no algorithm breakdown)."""
        from src.core.vector_store.semantic_retriever import SourceInfo

        info = SourceInfo(
            filename="doc.pdf", chunk_num=0, section="N/A", relevance_score=0.5, word_count=50
        )

        assert info.sources is None

    def test_sources_can_track_algorithms(self):
        """sources can record which algorithms found this chunk."""
        from src.core.vector_store.semantic_retriever import SourceInfo

        info = SourceInfo(
            filename="doc.pdf",
            chunk_num=1,
            section="Body",
            relevance_score=0.9,
            word_count=80,
            sources=["FAISS", "BM25+"],
        )

        assert "FAISS" in info.sources
        assert "BM25+" in info.sources


class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_fields_stored_correctly(self):
        """RetrievalResult stores context, sources, count, and timing."""
        from src.core.vector_store.semantic_retriever import RetrievalResult, SourceInfo

        sources = [
            SourceInfo(
                filename="a.pdf", chunk_num=0, section="N/A", relevance_score=0.8, word_count=50
            )
        ]
        result = RetrievalResult(
            context="Relevant passage here.",
            sources=sources,
            chunks_retrieved=1,
            retrieval_time_ms=42.5,
        )

        assert result.context == "Relevant passage here."
        assert len(result.sources) == 1
        assert result.chunks_retrieved == 1
        assert result.retrieval_time_ms == 42.5


class TestContextWindowHelper:
    """Tests for _get_effective_semantic_context_window."""

    def test_falls_back_to_config_when_prefs_unavailable(self):
        """Returns SEMANTIC_CONTEXT_WINDOW when user preferences cannot be loaded."""
        from src.config import SEMANTIC_CONTEXT_WINDOW
        from src.core.vector_store.semantic_retriever import _get_effective_semantic_context_window

        with patch(
            "src.user_preferences.get_user_preferences",
            side_effect=Exception("prefs unavailable"),
        ):
            result = _get_effective_semantic_context_window()

        assert result == SEMANTIC_CONTEXT_WINDOW

    def test_returns_value_from_preferences(self):
        """Returns the context size reported by user preferences."""
        from src.core.vector_store.semantic_retriever import _get_effective_semantic_context_window

        mock_prefs = MagicMock()
        mock_prefs.get_effective_context_size.return_value = 8192

        with patch(
            "src.user_preferences.get_user_preferences",
            return_value=mock_prefs,
        ):
            result = _get_effective_semantic_context_window()

        assert result == 8192


class TestAlgorithmWeightsHelper:
    """Tests for _get_effective_algorithm_weights."""

    def test_falls_back_to_config_weights_on_exception(self):
        """Returns RETRIEVAL_ALGORITHM_WEIGHTS when preferences raise."""
        from src.config import RETRIEVAL_ALGORITHM_WEIGHTS
        from src.core.vector_store.semantic_retriever import _get_effective_algorithm_weights

        with patch(
            "src.user_preferences.get_user_preferences",
            side_effect=Exception("no prefs"),
        ):
            weights = _get_effective_algorithm_weights()

        assert weights == RETRIEVAL_ALGORITHM_WEIGHTS

    def test_returns_weights_from_preferences(self):
        """Returns FAISS and BM25+ weights loaded from preferences."""
        from src.core.vector_store.semantic_retriever import _get_effective_algorithm_weights

        mock_prefs = MagicMock()
        mock_prefs.get.side_effect = lambda key, default=None: {
            "retrieval_weight_faiss": 0.7,
            "retrieval_weight_bm25": 0.3,
        }.get(key, default)

        with patch(
            "src.user_preferences.get_user_preferences",
            return_value=mock_prefs,
        ):
            weights = _get_effective_algorithm_weights()

        assert weights["FAISS"] == 0.7
        assert weights["BM25+"] == 0.3

    def test_falls_back_to_config_for_non_numeric_preference(self):
        """Non-numeric preference values are replaced with config defaults."""
        from src.config import RETRIEVAL_ALGORITHM_WEIGHTS
        from src.core.vector_store.semantic_retriever import _get_effective_algorithm_weights

        mock_prefs = MagicMock()
        # Simulate a corrupted preference returning a string
        mock_prefs.get.return_value = "not-a-number"

        with patch(
            "src.user_preferences.get_user_preferences",
            return_value=mock_prefs,
        ):
            weights = _get_effective_algorithm_weights()

        assert weights["FAISS"] == RETRIEVAL_ALGORITHM_WEIGHTS["FAISS"]
        assert weights["BM25+"] == RETRIEVAL_ALGORITHM_WEIGHTS["BM25+"]


class TestSemanticRetrieverInit:
    """Tests for SemanticRetriever.__init__ without loading real FAISS indexes."""

    def test_raises_file_not_found_when_index_missing(self, tmp_path):
        """SemanticRetriever raises FileNotFoundError if index.faiss is absent."""
        from src.core.vector_store.semantic_retriever import SemanticRetriever

        # tmp_path exists but has no index.faiss file
        with pytest.raises(FileNotFoundError, match="Vector store not found"):
            SemanticRetriever(vector_store_path=tmp_path, embeddings=MagicMock())


class TestVerifyIntegrityHash:
    """Tests for SemanticRetriever._verify_integrity_hash."""

    def _make_retriever(self):
        """Create a SemanticRetriever instance bypassing __init__."""
        from src.core.vector_store.semantic_retriever import SemanticRetriever

        return object.__new__(SemanticRetriever)

    def test_skips_verification_when_no_hash_file(self, tmp_path):
        """No .hash file → verification is skipped without error (legacy store)."""
        retriever = self._make_retriever()
        # Should not raise
        retriever._verify_integrity_hash(tmp_path)

    def test_passes_when_hash_matches(self, tmp_path):
        """Verification passes when the stored hash matches computed hash."""
        import hashlib

        retriever = self._make_retriever()

        # Write fake index files
        (tmp_path / "index.faiss").write_bytes(b"faiss content")
        (tmp_path / "index.pkl").write_bytes(b"pkl content")

        # Compute the expected hash manually
        hasher = hashlib.sha256()
        for fname in ["index.faiss", "index.pkl"]:
            with open(tmp_path / fname, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
        (tmp_path / ".hash").write_text(hasher.hexdigest())

        # Should not raise
        retriever._verify_integrity_hash(tmp_path)

    def test_raises_on_hash_mismatch(self, tmp_path):
        """Raises ValueError when stored hash does not match computed hash."""
        retriever = self._make_retriever()

        (tmp_path / "index.faiss").write_bytes(b"legitimate content")
        (tmp_path / "index.pkl").write_bytes(b"pkl content")
        # Tamper: write a wrong hash
        (tmp_path / ".hash").write_text("a" * 64)

        with pytest.raises(ValueError, match="integrity check failed"):
            retriever._verify_integrity_hash(tmp_path)


class TestGetRelevantSourcesSummary:
    """Tests for SemanticRetriever.get_relevant_sources_summary."""

    def _make_retriever(self):
        """Create a SemanticRetriever instance bypassing __init__."""
        from src.core.vector_store.semantic_retriever import SemanticRetriever

        return object.__new__(SemanticRetriever)

    def _make_result(self, sources):
        """Wrap a list of SourceInfo objects in a RetrievalResult."""
        from src.core.vector_store.semantic_retriever import RetrievalResult

        return RetrievalResult(context="", sources=sources, chunks_retrieved=0, retrieval_time_ms=0)

    def _make_source(self, filename, section="N/A"):
        """Create a minimal SourceInfo."""
        from src.core.vector_store.semantic_retriever import SourceInfo

        return SourceInfo(
            filename=filename, chunk_num=0, section=section, relevance_score=0.5, word_count=50
        )

    def test_returns_no_sources_found_for_empty_list(self):
        """Empty source list returns the 'No sources found' message."""
        retriever = self._make_retriever()
        result = self._make_result([])

        assert retriever.get_relevant_sources_summary(result) == "No sources found"

    def test_formats_single_source_without_section(self):
        """Single source with no section returns just the filename."""
        retriever = self._make_retriever()
        result = self._make_result([self._make_source("complaint.pdf")])

        summary = retriever.get_relevant_sources_summary(result)
        assert "complaint.pdf" in summary
        # No parentheses for N/A section
        assert "N/A" not in summary

    def test_formats_source_with_section(self):
        """Source with a named section includes it in parentheses."""
        retriever = self._make_retriever()
        result = self._make_result([self._make_source("depo.pdf", section="Direct Examination")])

        summary = retriever.get_relevant_sources_summary(result)
        assert "depo.pdf" in summary
        assert "Direct Examination" in summary

    def test_deduplicates_same_filename(self):
        """Multiple chunks from the same file appear only once."""
        retriever = self._make_retriever()
        sources = [
            self._make_source("brief.pdf"),
            self._make_source("brief.pdf"),  # duplicate
        ]
        result = self._make_result(sources)

        summary = retriever.get_relevant_sources_summary(result)
        # Filename should appear exactly once
        assert summary.count("brief.pdf") == 1

    def test_multiple_sources_comma_separated(self):
        """Multiple distinct files are joined with ', '."""
        retriever = self._make_retriever()
        sources = [
            self._make_source("a.pdf"),
            self._make_source("b.pdf"),
        ]
        result = self._make_result(sources)

        summary = retriever.get_relevant_sources_summary(result)
        assert "a.pdf" in summary
        assert "b.pdf" in summary
        assert ", " in summary


class TestGetChunkCount:
    """Tests for SemanticRetriever.get_chunk_count."""

    def _make_retriever_with_hybrid(self, count):
        """Create a retriever with a mocked hybrid retriever returning count."""
        from src.core.vector_store.semantic_retriever import SemanticRetriever

        retriever = object.__new__(SemanticRetriever)
        retriever._hybrid_retriever = MagicMock()
        retriever._hybrid_retriever.get_chunk_count.return_value = count
        return retriever

    def test_delegates_to_hybrid_retriever(self):
        """get_chunk_count returns whatever the hybrid retriever reports."""
        retriever = self._make_retriever_with_hybrid(123)
        assert retriever.get_chunk_count() == 123


class TestGetAlgorithmStatus:
    """Tests for SemanticRetriever.get_algorithm_status."""

    def test_delegates_to_hybrid_retriever(self):
        """get_algorithm_status returns the hybrid retriever's status dict."""
        from src.core.vector_store.semantic_retriever import SemanticRetriever

        retriever = object.__new__(SemanticRetriever)
        retriever._hybrid_retriever = MagicMock()
        retriever._hybrid_retriever.get_algorithm_status.return_value = {
            "FAISS": {"enabled": True},
            "BM25+": {"enabled": True},
        }

        status = retriever.get_algorithm_status()

        assert "FAISS" in status
        assert "BM25+" in status
