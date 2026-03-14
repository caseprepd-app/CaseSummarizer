"""
Tests for VectorStoreBuilder.

Covers FAISS vector store creation and file-based persistence:
- VectorStoreResult dataclass fields
- _generate_case_id: deterministic, order-independent, format-correct
- _convert_to_langchain_documents: dataclass chunks, dict chunks, empty text
- _save_integrity_hash: file creation, SHA256 validity, mismatch detection
- _save_model_marker: writes correct model name
- create_from_unified_chunks: delegates to FAISS, returns VectorStoreResult
- create_from_documents: raises on empty chunks
- cleanup_stale_stores, get_existing_stores, delete_store: filesystem ops
"""

from unittest.mock import MagicMock, patch

import pytest


class TestVectorStoreResult:
    """Tests for the VectorStoreResult dataclass."""

    def test_required_fields_stored(self, tmp_path):
        """VectorStoreResult stores path, case_id, chunk_count, and timing."""
        from src.core.vector_store.vector_store_builder import VectorStoreResult

        result = VectorStoreResult(
            persist_dir=tmp_path,
            case_id="abc12345_20260314",
            chunk_count=42,
            creation_time_ms=150.5,
        )

        assert result.persist_dir == tmp_path
        assert result.case_id == "abc12345_20260314"
        assert result.chunk_count == 42
        assert result.creation_time_ms == 150.5

    def test_chunk_embeddings_is_none_by_default(self, tmp_path):
        """chunk_embeddings defaults to None when not provided."""
        from src.core.vector_store.vector_store_builder import VectorStoreResult

        result = VectorStoreResult(
            persist_dir=tmp_path, case_id="x", chunk_count=1, creation_time_ms=0.0
        )

        assert result.chunk_embeddings is None

    def test_chunk_embeddings_can_be_set(self, tmp_path):
        """chunk_embeddings accepts a list of embedding vectors."""
        from src.core.vector_store.vector_store_builder import VectorStoreResult

        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        result = VectorStoreResult(
            persist_dir=tmp_path,
            case_id="x",
            chunk_count=2,
            creation_time_ms=0.0,
            chunk_embeddings=embeddings,
        )

        assert result.chunk_embeddings == embeddings


class TestGenerateCaseId:
    """Tests for VectorStoreBuilder._generate_case_id."""

    def test_format_is_hash_underscore_date(self):
        """Case ID has format <8-hex>_<YYYYMMDD>."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        builder = VectorStoreBuilder()
        case_id = builder._generate_case_id([{"filename": "complaint.pdf"}])

        parts = case_id.split("_")
        assert len(parts) == 2
        assert len(parts[0]) == 8
        assert all(c in "0123456789abcdef" for c in parts[0])
        assert len(parts[1]) == 8  # YYYYMMDD

    def test_same_filenames_produce_same_hash_prefix(self):
        """Same filename list always produces the same hash prefix."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        builder = VectorStoreBuilder()
        docs = [{"filename": "doc.pdf"}]
        id1 = builder._generate_case_id(docs)
        id2 = builder._generate_case_id(docs)

        assert id1.split("_")[0] == id2.split("_")[0]

    def test_different_filenames_produce_different_hash_prefix(self):
        """Different filename lists produce different hash prefixes."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        builder = VectorStoreBuilder()
        id_a = builder._generate_case_id([{"filename": "alpha.pdf"}])
        id_b = builder._generate_case_id([{"filename": "beta.pdf"}])

        assert id_a.split("_")[0] != id_b.split("_")[0]

    def test_filename_order_does_not_affect_hash(self):
        """Filenames are sorted before hashing — order is irrelevant."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        builder = VectorStoreBuilder()
        id1 = builder._generate_case_id([{"filename": "a.pdf"}, {"filename": "b.pdf"}])
        id2 = builder._generate_case_id([{"filename": "b.pdf"}, {"filename": "a.pdf"}])

        assert id1.split("_")[0] == id2.split("_")[0]

    def test_missing_filename_key_uses_unknown(self):
        """Documents without 'filename' key default to 'unknown'."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        builder = VectorStoreBuilder()
        case_id = builder._generate_case_id([{}])

        assert len(case_id.split("_")[0]) == 8  # Still produces a valid hash prefix


class TestConvertToLangchainDocuments:
    """Tests for VectorStoreBuilder._convert_to_langchain_documents."""

    def test_converts_dataclass_style_chunks(self):
        """Chunks with attribute access (.text, .chunk_num, etc.) are converted."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        chunk = MagicMock()
        chunk.text = "The plaintiff filed the complaint on March 1st."
        chunk.chunk_num = 1
        chunk.section_name = "Introduction"
        chunk.word_count = 8

        builder = VectorStoreBuilder()
        result = builder._convert_to_langchain_documents(
            [{"filename": "complaint.pdf", "chunks": [chunk]}]
        )

        assert len(result) == 1
        assert result[0].page_content == "The plaintiff filed the complaint on March 1st."
        assert result[0].metadata["filename"] == "complaint.pdf"
        assert result[0].metadata["chunk_num"] == 1
        assert result[0].metadata["section_name"] == "Introduction"

    def test_converts_dict_style_chunks(self):
        """Chunks as plain dicts are also handled correctly."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        chunk = {
            "text": "Defendant denies all allegations.",
            "chunk_num": 0,
            "section_name": "Response",
            "word_count": 4,
        }

        builder = VectorStoreBuilder()
        result = builder._convert_to_langchain_documents(
            [{"filename": "answer.pdf", "chunks": [chunk]}]
        )

        assert len(result) == 1
        assert result[0].page_content == "Defendant denies all allegations."
        assert result[0].metadata["section_name"] == "Response"

    def test_skips_empty_and_whitespace_chunk_text(self):
        """Chunks with empty or whitespace-only text are skipped."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        empty_chunk = MagicMock()
        empty_chunk.text = "   "
        empty_chunk.chunk_num = 0
        empty_chunk.section_name = "N/A"
        empty_chunk.word_count = 0

        builder = VectorStoreBuilder()
        result = builder._convert_to_langchain_documents(
            [{"filename": "empty.pdf", "chunks": [empty_chunk]}]
        )

        assert len(result) == 0

    def test_returns_empty_list_for_empty_input(self):
        """Empty document list returns an empty LangChain document list."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        assert VectorStoreBuilder()._convert_to_langchain_documents([]) == []

    def test_multiple_docs_and_chunks_all_converted(self):
        """Multiple documents with multiple chunks are all converted correctly."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        def make_chunk(text, num):
            c = MagicMock()
            c.text = text
            c.chunk_num = num
            c.section_name = "Body"
            c.word_count = len(text.split())
            return c

        docs = [
            {
                "filename": "doc1.pdf",
                "chunks": [make_chunk("First chunk.", 0), make_chunk("Second chunk.", 1)],
            },
            {"filename": "doc2.pdf", "chunks": [make_chunk("Third chunk.", 0)]},
        ]

        builder = VectorStoreBuilder()
        result = builder._convert_to_langchain_documents(docs)

        assert len(result) == 3
        filenames = [d.metadata["filename"] for d in result]
        assert filenames.count("doc1.pdf") == 2
        assert filenames.count("doc2.pdf") == 1

    def test_extracted_text_is_split_into_chunks(self):
        """Documents without pre-chunked chunks but with extracted_text are split."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        # Long text that RecursiveCharacterTextSplitter will split into multiple chunks
        long_text = "The defendant was present. " * 200  # ~5000 chars

        builder = VectorStoreBuilder()
        result = builder._convert_to_langchain_documents(
            [{"filename": "depo.pdf", "extracted_text": long_text}]
        )

        # Should produce multiple chunks from the long text
        assert len(result) > 1
        assert all(d.metadata["filename"] == "depo.pdf" for d in result)
        assert all(d.metadata["section_name"] == "Auto-chunked" for d in result)


class TestSaveIntegrityHash:
    """Tests for VectorStoreBuilder._save_integrity_hash (SEC-001)."""

    def test_creates_hash_file(self, tmp_path):
        """_save_integrity_hash creates a .hash file in the persist dir."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        (tmp_path / "index.faiss").write_bytes(b"fake faiss data")
        (tmp_path / "index.pkl").write_bytes(b"fake pkl data")

        VectorStoreBuilder()._save_integrity_hash(tmp_path)

        assert (tmp_path / ".hash").exists()

    def test_saved_hash_is_valid_sha256_hex(self, tmp_path):
        """The .hash file contains a valid 64-character lowercase hex SHA256 string."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        (tmp_path / "index.faiss").write_bytes(b"faiss")
        (tmp_path / "index.pkl").write_bytes(b"pkl")

        VectorStoreBuilder()._save_integrity_hash(tmp_path)

        saved = (tmp_path / ".hash").read_text().strip()
        assert len(saved) == 64
        assert all(c in "0123456789abcdef" for c in saved)

    def test_raises_when_vector_store_files_missing(self, tmp_path):
        """Raises FileNotFoundError if index.faiss or index.pkl are absent."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        with pytest.raises(FileNotFoundError):
            VectorStoreBuilder()._save_integrity_hash(tmp_path)

    def test_different_content_produces_different_hash(self, tmp_path):
        """Modifying index.faiss changes the stored hash."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        (tmp_path / "index.pkl").write_bytes(b"pkl")

        builder = VectorStoreBuilder()

        (tmp_path / "index.faiss").write_bytes(b"version1")
        builder._save_integrity_hash(tmp_path)
        hash1 = (tmp_path / ".hash").read_text().strip()

        (tmp_path / "index.faiss").write_bytes(b"version2")
        builder._save_integrity_hash(tmp_path)
        hash2 = (tmp_path / ".hash").read_text().strip()

        assert hash1 != hash2


class TestSaveModelMarker:
    """Tests for VectorStoreBuilder._save_model_marker."""

    def test_writes_model_name_to_dot_model_file(self, tmp_path):
        """_save_model_marker writes the configured embedding model name."""
        from src.config import EMBEDDING_MODEL_NAME
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        VectorStoreBuilder._save_model_marker(tmp_path)

        written = (tmp_path / ".model").read_text().strip()
        assert written == EMBEDDING_MODEL_NAME


class TestStaticStoreMethods:
    """Tests for get_existing_stores, cleanup_stale_stores, and delete_store."""

    def test_get_existing_stores_finds_faiss_dirs(self, tmp_path):
        """get_existing_stores returns directories containing index.faiss."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        store_dir = tmp_path / "case_abc"
        store_dir.mkdir()
        (store_dir / "index.faiss").write_bytes(b"")

        other_dir = tmp_path / "not_a_store"
        other_dir.mkdir()

        with patch("src.core.vector_store.vector_store_builder.VECTOR_STORE_DIR", tmp_path):
            stores = VectorStoreBuilder.get_existing_stores()

        assert store_dir in stores
        assert other_dir not in stores

    def test_get_existing_stores_empty_when_dir_does_not_exist(self, tmp_path):
        """get_existing_stores returns [] when VECTOR_STORE_DIR doesn't exist."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        missing = tmp_path / "nonexistent"
        with patch("src.core.vector_store.vector_store_builder.VECTOR_STORE_DIR", missing):
            stores = VectorStoreBuilder.get_existing_stores()

        assert stores == []

    def test_delete_store_removes_directory(self, tmp_path):
        """delete_store removes the given directory and returns True."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        store_dir = tmp_path / "to_delete"
        store_dir.mkdir()
        (store_dir / "index.faiss").write_bytes(b"")

        result = VectorStoreBuilder.delete_store(store_dir)

        assert result is True
        assert not store_dir.exists()

    def test_delete_store_returns_false_for_nonexistent_dir(self, tmp_path):
        """delete_store returns False if the directory doesn't exist."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        result = VectorStoreBuilder.delete_store(tmp_path / "ghost")

        assert result is False

    def test_cleanup_stale_stores_deletes_wrong_model(self, tmp_path):
        """cleanup_stale_stores removes stores with a different .model marker."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        stale_dir = tmp_path / "stale_case"
        stale_dir.mkdir()
        (stale_dir / "index.faiss").write_bytes(b"")
        (stale_dir / ".model").write_text("old-model-name-v0")

        with patch("src.core.vector_store.vector_store_builder.VECTOR_STORE_DIR", tmp_path):
            deleted = VectorStoreBuilder.cleanup_stale_stores()

        assert deleted == 1
        assert not stale_dir.exists()

    def test_cleanup_stale_stores_keeps_current_model(self, tmp_path):
        """cleanup_stale_stores preserves stores matching the current model."""
        from src.config import EMBEDDING_MODEL_NAME
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        current_dir = tmp_path / "current_case"
        current_dir.mkdir()
        (current_dir / "index.faiss").write_bytes(b"")
        (current_dir / ".model").write_text(EMBEDDING_MODEL_NAME)

        with patch("src.core.vector_store.vector_store_builder.VECTOR_STORE_DIR", tmp_path):
            deleted = VectorStoreBuilder.cleanup_stale_stores()

        assert deleted == 0
        assert current_dir.exists()

    def test_cleanup_stale_stores_treats_missing_marker_as_stale(self, tmp_path):
        """cleanup_stale_stores deletes stores with no .model file (pre-upgrade stores)."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        legacy_dir = tmp_path / "legacy_case"
        legacy_dir.mkdir()
        (legacy_dir / "index.faiss").write_bytes(b"")
        # No .model file

        with patch("src.core.vector_store.vector_store_builder.VECTOR_STORE_DIR", tmp_path):
            deleted = VectorStoreBuilder.cleanup_stale_stores()

        assert deleted == 1

    def test_cleanup_stale_stores_returns_zero_for_empty_dir(self, tmp_path):
        """cleanup_stale_stores returns 0 when VECTOR_STORE_DIR is empty."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        with patch("src.core.vector_store.vector_store_builder.VECTOR_STORE_DIR", tmp_path):
            deleted = VectorStoreBuilder.cleanup_stale_stores()

        assert deleted == 0


class TestCreateFromUnifiedChunks:
    """Tests for create_from_unified_chunks (FAISS and embeddings mocked)."""

    def _make_chunk(self, text, num=0):
        """Create a minimal mock UnifiedChunk."""
        c = MagicMock()
        c.text = text
        c.chunk_num = num
        c.token_count = len(text.split())
        c.word_count = len(text.split())
        c.section_name = "Body"
        c.source_file = "doc.pdf"
        return c

    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_model_marker")
    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_integrity_hash")
    @patch("langchain_community.vectorstores.FAISS")
    def test_returns_vector_store_result(self, mock_faiss_cls, mock_hash, mock_marker, tmp_path):
        """create_from_unified_chunks returns a VectorStoreResult with correct fields."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder, VectorStoreResult

        mock_faiss_cls.from_embeddings.return_value = MagicMock()
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]

        chunks = [self._make_chunk("First chunk.", 0), self._make_chunk("Second chunk.", 1)]
        builder = VectorStoreBuilder()
        result = builder.create_from_unified_chunks(
            chunks=chunks, embeddings=mock_emb, persist_dir=tmp_path / "idx"
        )

        assert isinstance(result, VectorStoreResult)
        assert result.chunk_count == 2

    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_model_marker")
    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_integrity_hash")
    @patch("langchain_community.vectorstores.FAISS")
    def test_returns_embeddings_in_result(self, mock_faiss_cls, mock_hash, mock_marker, tmp_path):
        """create_from_unified_chunks stores pre-computed embeddings in the result."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        mock_faiss_cls.from_embeddings.return_value = MagicMock()
        expected_vecs = [[float(i)] * 768 for i in range(2)]
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = expected_vecs

        chunks = [self._make_chunk("Chunk A.", 0), self._make_chunk("Chunk B.", 1)]
        builder = VectorStoreBuilder()
        result = builder.create_from_unified_chunks(
            chunks=chunks, embeddings=mock_emb, persist_dir=tmp_path / "idx"
        )

        assert result.chunk_embeddings == expected_vecs

    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_model_marker")
    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_integrity_hash")
    @patch("langchain_community.vectorstores.FAISS")
    def test_raises_on_empty_chunks(self, mock_faiss_cls, mock_hash, mock_marker, tmp_path):
        """create_from_unified_chunks raises ValueError when given no chunks."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        with pytest.raises(ValueError, match="No chunks provided"):
            VectorStoreBuilder().create_from_unified_chunks(
                chunks=[], embeddings=MagicMock(), persist_dir=tmp_path / "idx"
            )

    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_model_marker")
    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_integrity_hash")
    @patch("langchain_community.vectorstores.FAISS")
    def test_fires_progress_callback(self, mock_faiss_cls, mock_hash, mock_marker, tmp_path):
        """create_from_unified_chunks calls progress_callback after each embedding batch."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        mock_faiss_cls.from_embeddings.return_value = MagicMock()
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1] * 768]

        progress_calls = []
        chunks = [self._make_chunk("Text.", 0)]

        VectorStoreBuilder().create_from_unified_chunks(
            chunks=chunks,
            embeddings=mock_emb,
            persist_dir=tmp_path / "idx",
            progress_callback=lambda cur, tot: progress_calls.append((cur, tot)),
        )

        assert len(progress_calls) >= 1
        # Last call should report completion (current == total)
        assert progress_calls[-1][0] == progress_calls[-1][1]

    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_model_marker")
    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_integrity_hash")
    @patch("langchain_community.vectorstores.FAISS")
    def test_auto_generates_case_id_from_source_file(
        self, mock_faiss_cls, mock_hash, mock_marker, tmp_path
    ):
        """create_from_unified_chunks generates a case_id when none is provided."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        mock_faiss_cls.from_embeddings.return_value = MagicMock()
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1] * 768]

        chunks = [self._make_chunk("Content.", 0)]
        result = VectorStoreBuilder().create_from_unified_chunks(
            chunks=chunks,
            embeddings=mock_emb,
            source_file="complaint.pdf",
            persist_dir=tmp_path / "idx",
        )

        assert result.case_id != ""
        assert "_" in result.case_id


class TestCreateFromDocuments:
    """Tests for create_from_documents (FAISS mocked)."""

    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_model_marker")
    @patch("src.core.vector_store.vector_store_builder.VectorStoreBuilder._save_integrity_hash")
    @patch("langchain_community.vectorstores.FAISS")
    def test_raises_on_empty_documents(self, mock_faiss_cls, mock_hash, mock_marker, tmp_path):
        """create_from_documents raises ValueError when no valid chunks are found."""
        from src.core.vector_store.vector_store_builder import VectorStoreBuilder

        # Documents with no chunks and no extracted_text → nothing to index
        docs = [{"filename": "empty.pdf"}]

        with pytest.raises(ValueError, match="No valid chunks found"):
            VectorStoreBuilder().create_from_documents(
                documents=docs, embeddings=MagicMock(), persist_dir=tmp_path / "idx"
            )
