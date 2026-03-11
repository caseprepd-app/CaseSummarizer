"""
Vector Store Builder for CasePrepd Semantic Search System.

Creates FAISS-based vector stores from processed document chunks.
Uses file-based persistence for standalone Windows distribution.

Architecture:
- Converts document chunks to LangChain Documents with metadata
- Creates FAISS index using HuggingFaceEmbeddings (reuses existing)
- Saves index as files (index.faiss + index.pkl) - no database required
- Supports create_from_unified_chunks() for UnifiedChunk objects
- Single chunking pass serves both search indexing and vocabulary extraction

Integration:
- Called from worker subprocess after document extraction completes
- Runs in background thread to avoid UI blocking
- Signals UI via queue when vector store is ready for search
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import VECTOR_STORE_DIR

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings


@dataclass
class VectorStoreResult:
    """Result of vector store creation."""

    persist_dir: Path
    case_id: str
    chunk_count: int
    creation_time_ms: float
    chunk_embeddings: list[list[float]] | None = None


class VectorStoreBuilder:
    """
    Creates and manages FAISS vector stores for Q&A.

    Converts processed document chunks into a searchable vector index.
    Uses HuggingFaceEmbeddings (nomic-embed-text-v1.5) for embedding generation.

    Example:
        builder = VectorStoreBuilder()
        result = builder.create_from_documents(
            documents=extracted_docs,
            embeddings=embeddings_model,
            persist_dir=Path("./vector_stores/case_123")
        )
    """

    def create_from_documents(
        self,
        documents: list[dict],
        embeddings: "HuggingFaceEmbeddings",
        persist_dir: Path | None = None,
        case_id: str | None = None,
    ) -> VectorStoreResult:
        """
        Build vector store from processed documents.

        Args:
            documents: List of document dicts with keys:
                - 'filename': str
                - 'chunks': list[Chunk] (from ChunkingEngine)
                - 'extracted_text': str (optional, for fallback)
            embeddings: Already-initialized HuggingFace embeddings model
            persist_dir: Where to save vector store files (auto-generated if None)
            case_id: Unique identifier for this case (auto-generated if None)

        Returns:
            VectorStoreResult with persistence path, case ID, and stats

        Raises:
            ValueError: If no valid chunks found in documents
        """
        import time

        start_time = time.perf_counter()

        from langchain_community.vectorstores import FAISS

        # Generate case ID if not provided
        if case_id is None:
            case_id = self._generate_case_id(documents)

        # Generate persist directory if not provided
        if persist_dir is None:
            persist_dir = VECTOR_STORE_DIR / case_id

        # Ensure directory exists
        persist_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Building index for case: %s", case_id)
        logger.debug("Persist directory: %s", persist_dir)

        # Convert chunks to LangChain Documents with metadata
        lc_documents = self._convert_to_langchain_documents(documents)

        if not lc_documents:
            raise ValueError("No valid chunks found in documents")

        logger.debug("Converting %d chunks to embeddings...", len(lc_documents))

        # Create FAISS vector store with inner product (embeddings are L2-normalized,
        # so inner product = cosine similarity, giving scores in [0, 1])
        from langchain_community.vectorstores.utils import DistanceStrategy

        vector_store = FAISS.from_documents(
            documents=lc_documents,
            embedding=embeddings,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        # Save to disk as files (index.faiss + index.pkl)
        vector_store.save_local(str(persist_dir))

        # SEC-001: Save SHA256 hash for integrity verification on load
        self._save_integrity_hash(persist_dir)

        # Save embedding model name for stale-index detection on model upgrades
        self._save_model_marker(persist_dir)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug("Created index with %d chunks", len(lc_documents))
        logger.debug("Saved to: %s", persist_dir)
        logger.debug("Build time: %.1fms", elapsed_ms)

        return VectorStoreResult(
            persist_dir=persist_dir,
            case_id=case_id,
            chunk_count=len(lc_documents),
            creation_time_ms=elapsed_ms,
        )

    def create_from_unified_chunks(
        self,
        chunks: list,  # list[UnifiedChunk] - avoiding import for type hint
        embeddings: "HuggingFaceEmbeddings",
        source_file: str | None = None,
        persist_dir: Path | None = None,
        case_id: str | None = None,
        progress_callback=None,
    ) -> VectorStoreResult:
        """
        Build vector store from UnifiedChunk objects.

        This method uses the same chunks that LLM extraction uses,
        enabling efficient single-pass chunking for the entire pipeline.

        Args:
            chunks: List of UnifiedChunk objects from unified_chunker
            embeddings: Already-initialized HuggingFace embeddings model
            source_file: Source filename for metadata (optional)
            persist_dir: Where to save vector store files (auto-generated if None)
            case_id: Unique identifier for this case (auto-generated if None)
            progress_callback: Optional callback(current, total) called after
                              each batch of chunks is embedded

        Returns:
            VectorStoreResult with persistence path, case ID, and stats

        Raises:
            ValueError: If no valid chunks provided
        """
        import time

        start_time = time.perf_counter()

        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        if not chunks:
            raise ValueError("No chunks provided for vector store creation")

        # Generate case ID if not provided
        if case_id is None:
            # Use source file or hash of first chunk for ID
            hash_input = source_file or (chunks[0].text[:100] if chunks else "unknown")
            hash_prefix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            date_stamp = datetime.now().strftime("%Y%m%d")
            case_id = f"{hash_prefix}_{date_stamp}"

        # Generate persist directory if not provided
        if persist_dir is None:
            persist_dir = VECTOR_STORE_DIR / case_id

        # Ensure directory exists
        persist_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Building index from %d unified chunks", len(chunks))
        logger.debug("Case ID: %s", case_id)

        # Convert UnifiedChunk objects to LangChain Documents
        lc_documents = []
        for chunk in chunks:
            # Handle UnifiedChunk attributes
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            if not text.strip():
                continue

            chunk_num = getattr(chunk, "chunk_num", 0)
            token_count = getattr(chunk, "token_count", 0)
            word_count = getattr(chunk, "word_count", len(text.split()))
            section_name = getattr(chunk, "section_name", None) or "N/A"
            chunk_source = getattr(chunk, "source_file", None) or source_file or "unknown"

            lc_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "filename": chunk_source,
                        "chunk_num": chunk_num,
                        "section_name": section_name,
                        "word_count": word_count,
                        "token_count": token_count,
                    },
                )
            )

        if not lc_documents:
            raise ValueError("No valid chunks found after conversion")

        avg_tokens = sum(d.metadata.get("token_count", 0) for d in lc_documents) / len(lc_documents)
        logger.debug("Converting %d chunks (avg %.0f tokens)", len(lc_documents), avg_tokens)

        # Create FAISS vector store with inner product (embeddings are L2-normalized,
        # so inner product = cosine similarity, giving scores in [0, 1])
        from langchain_community.vectorstores.utils import DistanceStrategy

        # Embed texts in batches of 2 for progress reporting, then build
        # the FAISS index once from pre-computed embeddings. Each chunk's
        # embedding is independent — batching doesn't affect results.
        texts = [doc.page_content for doc in lc_documents]
        metadatas = [doc.metadata for doc in lc_documents]
        all_embeddings = []
        batch_size = 2
        total = len(texts)

        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_vecs = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_vecs)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        # Build FAISS index once from all pre-computed embeddings
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, all_embeddings)),
            embedding=embeddings,
            metadatas=metadatas,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        # Save to disk as files (index.faiss + index.pkl)
        vector_store.save_local(str(persist_dir))

        # SEC-001: Save SHA256 hash for integrity verification on load
        self._save_integrity_hash(persist_dir)

        # Save embedding model name for stale-index detection on model upgrades
        self._save_model_marker(persist_dir)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug("Created index with %d unified chunks", len(lc_documents))
        logger.debug("Saved to: %s", persist_dir)
        logger.debug("Build time: %.1fms", elapsed_ms)

        return VectorStoreResult(
            persist_dir=persist_dir,
            case_id=case_id,
            chunk_count=len(lc_documents),
            creation_time_ms=elapsed_ms,
            chunk_embeddings=all_embeddings,
        )

    def _convert_to_langchain_documents(self, documents: list[dict]) -> list:
        """
        Convert CasePrepd documents to LangChain Documents.

        Handles two scenarios:
        1. Documents with pre-computed chunks (from ChunkingEngine)
        2. Documents with only extracted_text (chunks on-demand)

        For raw text, we use RecursiveCharacterTextSplitter to create
        semantic chunks suitable for vector search.

        Args:
            documents: List of document dicts from extraction

        Returns:
            List of LangChain Document objects with metadata
        """
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        from src.config import RETRIEVAL_CHUNK_OVERLAP, RETRIEVAL_CHUNK_SIZE

        # Cache text splitter as class attribute
        if not hasattr(self, "_text_splitter"):
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=RETRIEVAL_CHUNK_SIZE,
                chunk_overlap=RETRIEVAL_CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        text_splitter = self._text_splitter

        lc_documents = []

        for doc in documents:
            filename = doc.get("filename", "unknown")
            chunks = doc.get("chunks", [])

            # If document has pre-computed chunks, use them
            if chunks:
                for chunk in chunks:
                    # Handle both Chunk dataclass and dict formats
                    if hasattr(chunk, "text"):
                        text = chunk.text
                        chunk_num = chunk.chunk_num
                        section_name = chunk.section_name or "N/A"
                        word_count = chunk.word_count
                    else:
                        text = chunk.get("text", "")
                        chunk_num = chunk.get("chunk_num", 0)
                        section_name = chunk.get("section_name", "N/A")
                        word_count = chunk.get("word_count", len(text.split()))

                    if not text.strip():
                        continue

                    lc_documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "filename": filename,
                                "chunk_num": chunk_num,
                                "section_name": section_name,
                                "word_count": word_count,
                            },
                        )
                    )
            # Otherwise, chunk the extracted_text
            elif doc.get("extracted_text"):
                text = doc["extracted_text"]
                if not text.strip():
                    continue

                # Apply preprocessing to remove line numbers, headers/footers, etc.
                # This improves search quality by removing noise from citations
                text = self._preprocess_text(text)
                if not text.strip():
                    continue

                # Split into chunks using LangChain text splitter
                split_texts = text_splitter.split_text(text)

                for i, chunk_text in enumerate(split_texts):
                    lc_documents.append(
                        Document(
                            page_content=chunk_text,
                            metadata={
                                "filename": filename,
                                "chunk_num": i,
                                "section_name": "Auto-chunked",
                                "word_count": len(chunk_text.split()),
                            },
                        )
                    )

                logger.debug("Split '%s' into %d chunks", filename, len(split_texts))

        return lc_documents

    def _generate_case_id(self, documents: list[dict]) -> str:
        """
        Generate unique case ID from document filenames.

        Format: <hash>_<date>
        Example: a1b2c3d4_20251129

        Args:
            documents: List of document dicts

        Returns:
            Unique case identifier string
        """
        # Combine filenames for hashing
        filenames = sorted([d.get("filename", "unknown") for d in documents])
        combined = "|".join(filenames)

        # Create MD5 hash (first 8 chars)
        hash_prefix = hashlib.md5(combined.encode()).hexdigest()[:8]

        # Add date stamp
        date_stamp = datetime.now().strftime("%Y%m%d")

        return f"{hash_prefix}_{date_stamp}"

    def _preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing pipeline to clean text before vectorization.

        Removes line numbers, headers/footers, and other noise that would
        degrade Q&A citation quality.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text suitable for vector search and citation display
        """
        try:
            from src.core.preprocessing import create_default_pipeline

            pipeline = create_default_pipeline()
            cleaned = pipeline.process(text)

            logger.debug("Preprocessing applied: %d changes", pipeline.total_changes)

            return cleaned
        except Exception as e:
            logger.error("Preprocessing error (using raw text): %s", e, exc_info=True)
            return text

    def _save_integrity_hash(self, persist_dir: Path) -> None:
        """
        Compute and save SHA256 hash of vector store files (SEC-001).

        Creates a .hash file containing the combined hash of index.faiss
        and index.pkl for integrity verification on load.

        Args:
            persist_dir: Directory containing the vector store files
        """
        faiss_file = persist_dir / "index.faiss"
        pkl_file = persist_dir / "index.pkl"
        hash_file = persist_dir / ".hash"

        # Verify both files exist before computing hash
        if not faiss_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(
                f"Vector store files missing after save. "
                f"FAISS: {faiss_file.exists()}, PKL: {pkl_file.exists()}"
            )

        # Compute combined hash of both files
        hasher = hashlib.sha256()

        for file_path in [faiss_file, pkl_file]:
            with open(file_path, "rb") as f:
                # Read in chunks for memory efficiency
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)

        # Save hash to file
        hash_file.write_text(hasher.hexdigest())

        logger.debug("Saved integrity hash: %s...", hasher.hexdigest()[:16])

    @staticmethod
    def _save_model_marker(persist_dir: Path) -> None:
        """
        Save embedding model name to .model file for stale-index detection.

        When the embedding model changes, old indexes produce wrong-dimension
        vectors and must be rebuilt. This marker lets us detect the mismatch.

        Args:
            persist_dir: Directory containing the vector store files
        """
        from src.config import EMBEDDING_MODEL_NAME

        model_file = persist_dir / ".model"
        model_file.write_text(EMBEDDING_MODEL_NAME)
        logger.debug("Saved model marker: %s", EMBEDDING_MODEL_NAME)

    @staticmethod
    def cleanup_stale_stores() -> int:
        """
        Delete vector stores built with a different embedding model.

        Stores without a .model marker file are assumed stale (pre-upgrade).

        Returns:
            Number of stale stores deleted.
        """
        import shutil

        from src.config import EMBEDDING_MODEL_NAME

        if not VECTOR_STORE_DIR.exists():
            return 0

        deleted = 0
        for d in VECTOR_STORE_DIR.iterdir():
            if not d.is_dir() or not (d / "index.faiss").exists():
                continue

            model_file = d / ".model"
            if not model_file.exists() or model_file.read_text().strip() != EMBEDDING_MODEL_NAME:
                logger.warning("Deleting stale vector store (wrong embedding model): %s", d.name)
                shutil.rmtree(d)
                deleted += 1

        if deleted:
            logger.warning("Deleted %d stale vector store(s)", deleted)
        return deleted

    @staticmethod
    def get_existing_stores() -> list[Path]:
        """
        List existing vector stores.

        Returns:
            List of paths to existing vector store directories
        """
        if not VECTOR_STORE_DIR.exists():
            return []

        return [
            d for d in VECTOR_STORE_DIR.iterdir() if d.is_dir() and (d / "index.faiss").exists()
        ]

    @staticmethod
    def delete_store(persist_dir: Path) -> bool:
        """
        Delete a vector store.

        Args:
            persist_dir: Path to the vector store directory

        Returns:
            True if deletion successful, False otherwise
        """
        import shutil

        try:
            if persist_dir.exists():
                shutil.rmtree(persist_dir)
                logger.debug("Deleted: %s", persist_dir)
                return True
            return False
        except Exception as e:
            logger.error("Error deleting %s: %s", persist_dir, e, exc_info=True)
            return False
