"""
High-priority coverage gap tests for 3 source modules.

Covers behavioral tests (not just import checks) for:
1. VectorStoreBuilder  — create_from_unified_chunks, cleanup_stale_stores, empty input
2. QARetriever         — retrieve_context, get_chunk_count, empty results
3. PDFExtractor        — _extract_pymupdf with mocked fitz

All external dependencies (fitz, pdfplumber, FAISS, Ollama, langchain, etc.)
are mocked so tests run without real models or connections.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers & fakes shared across test classes
# ---------------------------------------------------------------------------


@dataclass
class FakeUnifiedChunk:
    """Minimal stand-in for UnifiedChunk objects."""

    text: str
    chunk_num: int = 0
    token_count: int = 50
    word_count: int = 10
    section_name: str = "Introduction"
    source_file: str = "test.pdf"


@dataclass
class FakeMergedChunk:
    """Stand-in for MergedChunk returned by HybridRetriever.retrieve()."""

    chunk_id: str = "test.pdf_0"
    text: str = "The plaintiff filed a motion for summary judgment."
    combined_score: float = 0.85
    sources: list[str] = field(default_factory=lambda: ["FAISS", "BM25+"])
    filename: str = "test.pdf"
    chunk_num: int = 0
    section_name: str = "Parties"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeMergedRetrievalResult:
    """Stand-in for MergedRetrievalResult."""

    chunks: list[FakeMergedChunk] = field(default_factory=list)
    total_algorithms: int = 2
    processing_time_ms: float = 10.0
    query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# =========================================================================
# 1. VectorStoreBuilder
# =========================================================================


class TestVectorStoreBuilderCreateFromUnifiedChunks:
    """Tests for VectorStoreBuilder.create_from_unified_chunks()."""

    def test_empty_chunks_raises_value_error(self):
        """Passing an empty chunk list raises ValueError immediately."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        builder = VectorStoreBuilder()
        embeddings = MagicMock()

        with pytest.raises(ValueError, match="No chunks provided"):
            builder.create_from_unified_chunks(chunks=[], embeddings=embeddings)

    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_model_marker")
    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_integrity_hash")
    def test_creates_result_from_valid_chunks(self, mock_hash, mock_marker, tmp_path):
        """Valid unified chunks produce a VectorStoreResult with correct stats."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        builder = VectorStoreBuilder()
        embeddings = MagicMock()
        embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        mock_faiss_cls = MagicMock()
        mock_faiss_instance = MagicMock()
        mock_faiss_cls.from_embeddings.return_value = mock_faiss_instance

        chunks = [
            FakeUnifiedChunk(text="Plaintiff Jones filed a complaint.", chunk_num=0),
            FakeUnifiedChunk(text="Defendant Smith responded with an answer.", chunk_num=1),
        ]

        persist_dir = tmp_path / "test_store"
        persist_dir.mkdir()

        with (
            patch.dict("sys.modules", {}),
            patch(
                "src.core.vector_store.vector_store_builder.FAISS",
                mock_faiss_cls,
                create=True,
            ),
        ):
            # Patch the lazy imports inside the method
            import sys

            fake_faiss_mod = MagicMock()
            fake_faiss_mod.FAISS = mock_faiss_cls
            fake_utils_mod = MagicMock()
            fake_utils_mod.DistanceStrategy.MAX_INNER_PRODUCT = "max_ip"
            fake_doc_mod = MagicMock()

            with patch.dict(
                sys.modules,
                {
                    "langchain_community.vectorstores": fake_faiss_mod,
                    "langchain_community.vectorstores.utils": fake_utils_mod,
                    "langchain_core.documents": fake_doc_mod,
                },
            ):
                result = builder.create_from_unified_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    persist_dir=persist_dir,
                    case_id="test_case_001",
                )

        assert result.case_id == "test_case_001"
        assert result.chunk_count == 2
        assert result.persist_dir == persist_dir
        assert result.chunk_embeddings is not None
        assert len(result.chunk_embeddings) == 2
        assert result.creation_time_ms > 0

    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_model_marker")
    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_integrity_hash")
    def test_whitespace_only_chunks_are_skipped(self, mock_hash, mock_marker, tmp_path):
        """Chunks containing only whitespace are excluded from the index."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        builder = VectorStoreBuilder()
        embeddings = MagicMock()

        chunks = [
            FakeUnifiedChunk(text="   ", chunk_num=0),
            FakeUnifiedChunk(text="\n\t\n", chunk_num=1),
        ]

        with pytest.raises(ValueError, match="No valid chunks found"):
            import sys

            fake_faiss_mod = MagicMock()
            fake_utils_mod = MagicMock()
            fake_doc_mod = MagicMock()
            # Make Document a simple class that stores page_content/metadata
            fake_doc_mod.Document = lambda page_content, metadata: MagicMock(
                page_content=page_content, metadata=metadata
            )

            with patch.dict(
                sys.modules,
                {
                    "langchain_community.vectorstores": fake_faiss_mod,
                    "langchain_community.vectorstores.utils": fake_utils_mod,
                    "langchain_core.documents": fake_doc_mod,
                },
            ):
                builder.create_from_unified_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    persist_dir=tmp_path / "vs",
                    case_id="empty_case",
                )

    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_model_marker")
    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_integrity_hash")
    def test_progress_callback_is_invoked(self, mock_hash, mock_marker, tmp_path):
        """Progress callback receives (current, total) for each batch."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        builder = VectorStoreBuilder()
        embeddings = MagicMock()
        embeddings.embed_documents.return_value = [[0.1]]

        chunks = [
            FakeUnifiedChunk(text="First chunk of text here.", chunk_num=0),
            FakeUnifiedChunk(text="Second chunk of text here.", chunk_num=1),
            FakeUnifiedChunk(text="Third chunk of text here.", chunk_num=2),
        ]

        progress_calls = []

        def on_progress(current, total):
            progress_calls.append((current, total))

        import sys

        fake_faiss_mod = MagicMock()
        fake_utils_mod = MagicMock()
        fake_doc_mod = MagicMock()
        fake_doc_mod.Document = lambda page_content, metadata: MagicMock(
            page_content=page_content, metadata=metadata
        )

        persist_dir = tmp_path / "prog_store"
        persist_dir.mkdir()

        with patch.dict(
            sys.modules,
            {
                "langchain_community.vectorstores": fake_faiss_mod,
                "langchain_community.vectorstores.utils": fake_utils_mod,
                "langchain_core.documents": fake_doc_mod,
            },
        ):
            builder.create_from_unified_chunks(
                chunks=chunks,
                embeddings=embeddings,
                persist_dir=persist_dir,
                case_id="progress_test",
                progress_callback=on_progress,
            )

        # batch_size=2, 3 chunks => 2 batches: (2,3) and (3,3)
        assert len(progress_calls) == 2
        assert progress_calls[-1][0] == progress_calls[-1][1]  # last batch: current == total


class TestVectorStoreBuilderCleanupStaleStores:
    """Tests for VectorStoreBuilder.cleanup_stale_stores()."""

    def test_returns_zero_when_dir_missing(self, tmp_path):
        """Returns 0 when the vector store directory does not exist."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        nonexistent = tmp_path / "nonexistent"
        with patch("src.core.vector_store.vector_store_builder.VECTOR_STORE_DIR", nonexistent):
            deleted = VectorStoreBuilder.cleanup_stale_stores()

        assert deleted == 0

    def test_deletes_store_with_wrong_model(self, tmp_path):
        """Deletes stores whose .model file does not match the current model name."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        # Create a stale store directory
        store_dir = tmp_path / "stale_store"
        store_dir.mkdir()
        (store_dir / "index.faiss").write_bytes(b"fake")
        (store_dir / ".model").write_text("old-model-v1")

        with patch("src.core.vector_store.vector_store_builder.VECTOR_STORE_DIR", tmp_path):
            with patch("src.config.EMBEDDING_MODEL_NAME", "new-model-v2"):
                deleted = VectorStoreBuilder.cleanup_stale_stores()

        assert deleted == 1
        assert not store_dir.exists()

    def test_keeps_store_with_correct_model(self, tmp_path):
        """Stores matching the current model are not deleted."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        store_dir = tmp_path / "good_store"
        store_dir.mkdir()
        (store_dir / "index.faiss").write_bytes(b"fake")
        (store_dir / ".model").write_text("nomic-ai/nomic-embed-text-v1.5")

        with patch("src.core.vector_store.vector_store_builder.VECTOR_STORE_DIR", tmp_path):
            with patch(
                "src.config.EMBEDDING_MODEL_NAME",
                "nomic-ai/nomic-embed-text-v1.5",
            ):
                deleted = VectorStoreBuilder.cleanup_stale_stores()

        assert deleted == 0
        assert store_dir.exists()

    def test_deletes_store_without_model_marker(self, tmp_path):
        """Stores without a .model file are treated as stale and deleted."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        store_dir = tmp_path / "no_marker"
        store_dir.mkdir()
        (store_dir / "index.faiss").write_bytes(b"fake")
        # No .model file

        with patch("src.core.vector_store.vector_store_builder.VECTOR_STORE_DIR", tmp_path):
            with patch("src.config.EMBEDDING_MODEL_NAME", "any-model"):
                deleted = VectorStoreBuilder.cleanup_stale_stores()

        assert deleted == 1
        assert not store_dir.exists()


# =========================================================================
# 2. QARetriever
# =========================================================================


class TestQARetrieverRetrieveContext:
    """Tests for QARetriever.retrieve_context() with fully mocked internals."""

    def _make_retriever(self):
        """Build a QARetriever with all heavy dependencies mocked out."""
        from src.core.vector_store.qa_retriever import QARetriever

        # We'll bypass __init__ entirely and set attributes manually
        retriever = object.__new__(QARetriever)
        retriever.vector_store_path = Path("/fake/path")
        retriever.embeddings = MagicMock()
        retriever._faiss_store = MagicMock()
        retriever._documents = []
        retriever._hybrid_retriever = MagicMock()
        retriever._reranker = None  # disabled
        return retriever

    def test_retrieve_returns_context_and_sources(self):
        """retrieve_context returns formatted context with source attributions."""
        retriever = self._make_retriever()

        chunk = FakeMergedChunk(
            text="The plaintiff filed a complaint on January 15.",
            combined_score=0.9,
            filename="complaint.pdf",
            chunk_num=1,
            section_name="Parties",
        )
        retriever._hybrid_retriever.retrieve.return_value = FakeMergedRetrievalResult(
            chunks=[chunk]
        )
        retriever._hybrid_retriever.get_chunk_count.return_value = 5

        with patch(
            "src.core.vector_store.qa_retriever._get_effective_qa_context_window", return_value=4096
        ):
            result = retriever.retrieve_context("Who is the plaintiff?", k=5, min_score=0.0)

        assert result.chunks_retrieved == 1
        assert "complaint.pdf" in result.context
        assert "Parties" in result.context
        assert result.sources[0].filename == "complaint.pdf"
        assert result.sources[0].relevance_score == 0.9

    def test_retrieve_empty_results_when_all_below_min_score(self):
        """Chunks below min_score are excluded, yielding empty context."""
        retriever = self._make_retriever()

        chunk = FakeMergedChunk(combined_score=0.05)
        retriever._hybrid_retriever.retrieve.return_value = FakeMergedRetrievalResult(
            chunks=[chunk]
        )
        retriever._hybrid_retriever.get_chunk_count.return_value = 1

        with patch(
            "src.core.vector_store.qa_retriever._get_effective_qa_context_window", return_value=4096
        ):
            result = retriever.retrieve_context("test?", k=5, min_score=0.5)

        assert result.chunks_retrieved == 0
        assert result.context == ""

    def test_retrieve_deduplicates_same_chunk(self):
        """Duplicate chunks (same filename+chunk_num) keep only the best score."""
        retriever = self._make_retriever()

        chunk_a = FakeMergedChunk(
            chunk_id="file_0",
            combined_score=0.7,
            filename="file.pdf",
            chunk_num=0,
            text="Same chunk text.",
        )
        chunk_b = FakeMergedChunk(
            chunk_id="file_0",
            combined_score=0.9,
            filename="file.pdf",
            chunk_num=0,
            text="Same chunk text.",
        )
        retriever._hybrid_retriever.retrieve.return_value = FakeMergedRetrievalResult(
            chunks=[chunk_a, chunk_b]
        )
        retriever._hybrid_retriever.get_chunk_count.return_value = 2

        with patch(
            "src.core.vector_store.qa_retriever._get_effective_qa_context_window", return_value=4096
        ):
            result = retriever.retrieve_context("test?", k=5, min_score=0.0)

        assert result.chunks_retrieved == 1
        assert result.sources[0].relevance_score == 0.9

    def test_retrieve_no_chunks_from_hybrid_retriever(self):
        """When the hybrid retriever returns zero chunks, context is empty."""
        retriever = self._make_retriever()

        retriever._hybrid_retriever.retrieve.return_value = FakeMergedRetrievalResult(chunks=[])
        retriever._hybrid_retriever.get_chunk_count.return_value = 0

        with patch(
            "src.core.vector_store.qa_retriever._get_effective_qa_context_window", return_value=4096
        ):
            result = retriever.retrieve_context("anything?", k=5, min_score=0.0)

        assert result.chunks_retrieved == 0
        assert result.context == ""


class TestQARetrieverHelpers:
    """Tests for QARetriever helper methods."""

    def _make_retriever(self):
        """Build a QARetriever with mocked internals."""
        from src.core.vector_store.qa_retriever import QARetriever

        retriever = object.__new__(QARetriever)
        retriever._hybrid_retriever = MagicMock()
        retriever._reranker = None
        return retriever

    def test_get_chunk_count_delegates_to_hybrid_retriever(self):
        """get_chunk_count() proxies to the hybrid retriever's count."""
        retriever = self._make_retriever()
        retriever._hybrid_retriever.get_chunk_count.return_value = 42

        assert retriever.get_chunk_count() == 42

    def test_get_relevant_sources_summary_formats_sources(self):
        """get_relevant_sources_summary() returns a readable string of filenames."""
        from src.core.vector_store.qa_retriever import RetrievalResult, SourceInfo

        retriever = self._make_retriever()

        result = RetrievalResult(
            context="...",
            sources=[
                SourceInfo(
                    filename="complaint.pdf",
                    chunk_num=0,
                    section="Parties",
                    relevance_score=0.9,
                    word_count=50,
                ),
                SourceInfo(
                    filename="answer.pdf",
                    chunk_num=1,
                    section="N/A",
                    relevance_score=0.8,
                    word_count=30,
                ),
            ],
            chunks_retrieved=2,
            retrieval_time_ms=5.0,
        )

        summary = retriever.get_relevant_sources_summary(result)
        assert "complaint.pdf (Parties)" in summary
        assert "answer.pdf" in summary

    def test_get_relevant_sources_summary_no_sources(self):
        """Returns 'No sources found' when result has no sources."""
        from src.core.vector_store.qa_retriever import RetrievalResult

        retriever = self._make_retriever()

        result = RetrievalResult(context="", sources=[], chunks_retrieved=0, retrieval_time_ms=0.0)
        assert retriever.get_relevant_sources_summary(result) == "No sources found"


# =========================================================================
# 3. PDFExtractor
# =========================================================================


class TestPDFExtractorExtractPyMuPDF:
    """Tests for PDFExtractor._extract_pymupdf() with mocked fitz."""

    def _make_extractor(self):
        """Create a PDFExtractor with mocked dependencies."""
        mock_dict = MagicMock()

        with patch("src.core.extraction.pdf_extractor.LayoutAnalyzer"):
            from src.core.extraction.pdf_extractor import PDFExtractor

            extractor = PDFExtractor(dictionary=mock_dict)

        return extractor

    @patch("src.core.extraction.pdf_extractor.fitz")
    def test_successful_extraction(self, mock_fitz_module):
        """A normal PDF returns text, page count, and no error."""
        extractor = self._make_extractor()

        # Set up mock page that returns block tuples for extract_page_text
        def _make_mock_page():
            page = MagicMock()
            page.number = 0
            page.rect = MagicMock()
            page.rect.width = 612

            def mock_get_text(opt=None, clip=None, sort=False, flags=0):
                if opt == "blocks":
                    return [(50, 100, 200, 120, "Page one content.\n", 0, 0)]
                return "Page one content."

            page.get_text = mock_get_text
            return page

        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__iter__ = MagicMock(return_value=iter([_make_mock_page(), _make_mock_page()]))

        mock_fitz_module.open.return_value = mock_doc

        text, page_count, error = extractor._extract_pymupdf(Path("/fake/doc.pdf"))

        assert text is not None
        assert "Page one content." in text
        assert page_count == 2
        assert error is None

    @patch("src.core.extraction.pdf_extractor.fitz")
    def test_empty_pdf_returns_error(self, mock_fitz_module):
        """A zero-page PDF returns error type 'empty'."""
        extractor = self._make_extractor()

        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_doc.__len__ = MagicMock(return_value=0)

        mock_fitz_module.open.return_value = mock_doc

        text, page_count, error = extractor._extract_pymupdf(Path("/fake/empty.pdf"))

        assert text is None
        assert page_count == 0
        assert error == "empty"

    @patch("src.core.extraction.pdf_extractor.fitz")
    def test_password_protected_pdf(self, mock_fitz_module):
        """A password-protected PDF returns error type 'password'."""
        extractor = self._make_extractor()

        # Simulate fitz.FileDataError for encrypted PDF
        file_data_error = type("FileDataError", (Exception,), {})
        mock_fitz_module.FileDataError = file_data_error
        mock_fitz_module.open.side_effect = file_data_error("cannot open encrypted PDF")

        # We need to patch the module-level fitz reference too
        with patch(
            "src.core.extraction.pdf_extractor.fitz.FileDataError",
            file_data_error,
        ):
            text, page_count, error = extractor._extract_pymupdf(Path("/fake/locked.pdf"))

        assert text is None
        assert error == "password"

    @patch("src.core.extraction.pdf_extractor.fitz")
    def test_corrupted_pdf(self, mock_fitz_module):
        """A corrupted PDF returns error type 'corrupted'."""
        extractor = self._make_extractor()

        file_data_error = type("FileDataError", (Exception,), {})
        mock_fitz_module.FileDataError = file_data_error
        mock_fitz_module.open.side_effect = file_data_error("damaged file data")

        with patch(
            "src.core.extraction.pdf_extractor.fitz.FileDataError",
            file_data_error,
        ):
            text, page_count, error = extractor._extract_pymupdf(Path("/fake/corrupt.pdf"))

        assert text is None
        assert error == "corrupted"


class TestPDFExtractorExtractFullPipeline:
    """Tests for PDFExtractor.extract() top-level orchestration."""

    @patch("src.core.extraction.pdf_extractor.pdfplumber")
    @patch("src.core.extraction.pdf_extractor.fitz")
    def test_both_extractors_fail_returns_error(self, mock_fitz, mock_pdfplumber):
        """When both extractors fail, result has error and no text."""
        mock_dict = MagicMock()

        with patch("src.core.extraction.pdf_extractor.LayoutAnalyzer"):
            from src.core.extraction.pdf_extractor import PDFExtractor

            extractor = PDFExtractor(dictionary=mock_dict)

        # PyMuPDF fails
        mock_fitz.FileDataError = type("FileDataError", (Exception,), {})
        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_doc.__len__ = MagicMock(return_value=0)
        mock_fitz.open.return_value = mock_doc

        # pdfplumber also fails
        mock_pdfplumber.open.side_effect = Exception("pdfplumber failed")

        result = extractor.extract(Path("/fake/bad.pdf"))

        assert result["text"] is None
        assert result["method"] is None
        assert result["error"] is not None
