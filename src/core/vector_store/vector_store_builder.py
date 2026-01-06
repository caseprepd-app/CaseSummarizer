"""
Vector Store Builder for LocalScribe Q&A System (Session 45 Update).

Creates FAISS-based vector stores from processed document chunks.
Uses file-based persistence for standalone Windows distribution.

Architecture:
- Converts document chunks to LangChain Documents with metadata
- Creates FAISS index using HuggingFaceEmbeddings (reuses existing)
- Saves index as files (index.faiss + index.pkl) - no database required

Session 45 Update:
- Added create_from_unified_chunks() for UnifiedChunk objects
- Unified chunks are shared with LLM extraction for efficiency
- Single chunking pass serves both Q&A indexing and vocabulary extraction

Integration:
- Called from WorkflowOrchestrator after document extraction completes
- Runs in background thread to avoid UI blocking
- Signals UI via queue when vector store is ready for Q&A
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import DEBUG_MODE, VECTOR_STORE_DIR
from src.logging_config import debug_log

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings


@dataclass
class VectorStoreResult:
    """Result of vector store creation."""

    persist_dir: Path
    case_id: str
    chunk_count: int
    creation_time_ms: float


class VectorStoreBuilder:
    """
    Creates and manages FAISS vector stores for Q&A.

    Converts processed document chunks into a searchable vector index.
    Uses HuggingFaceEmbeddings (all-MiniLM-L6-v2) for embedding generation.

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

        if DEBUG_MODE:
            debug_log(f"[VectorStore] Building index for case: {case_id}")
            debug_log(f"[VectorStore] Persist directory: {persist_dir}")

        # Convert chunks to LangChain Documents with metadata
        lc_documents = self._convert_to_langchain_documents(documents)

        if not lc_documents:
            raise ValueError("No valid chunks found in documents")

        if DEBUG_MODE:
            debug_log(f"[VectorStore] Converting {len(lc_documents)} chunks to embeddings...")

        # Create FAISS vector store from documents
        # This embeds all chunks and builds the index
        vector_store = FAISS.from_documents(documents=lc_documents, embedding=embeddings)

        # Save to disk as files (index.faiss + index.pkl)
        vector_store.save_local(str(persist_dir))

        # SEC-001: Save SHA256 hash for integrity verification on load
        self._save_integrity_hash(persist_dir)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if DEBUG_MODE:
            debug_log(f"[VectorStore] Created index with {len(lc_documents)} chunks")
            debug_log(f"[VectorStore] Saved to: {persist_dir}")
            debug_log(f"[VectorStore] Build time: {elapsed_ms:.1f}ms")

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
    ) -> VectorStoreResult:
        """
        Build vector store from UnifiedChunk objects (Session 45).

        This method uses the same chunks that LLM extraction uses,
        enabling efficient single-pass chunking for the entire pipeline.

        Args:
            chunks: List of UnifiedChunk objects from unified_chunker
            embeddings: Already-initialized HuggingFace embeddings model
            source_file: Source filename for metadata (optional)
            persist_dir: Where to save vector store files (auto-generated if None)
            case_id: Unique identifier for this case (auto-generated if None)

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

        if DEBUG_MODE:
            debug_log(f"[VectorStore] Building index from {len(chunks)} unified chunks")
            debug_log(f"[VectorStore] Case ID: {case_id}")

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
                        "token_count": token_count,  # Session 45: include token count
                    },
                )
            )

        if not lc_documents:
            raise ValueError("No valid chunks found after conversion")

        if DEBUG_MODE:
            avg_tokens = sum(d.metadata.get("token_count", 0) for d in lc_documents) / len(
                lc_documents
            )
            debug_log(
                f"[VectorStore] Converting {len(lc_documents)} chunks (avg {avg_tokens:.0f} tokens)"
            )

        # Create FAISS vector store from documents
        vector_store = FAISS.from_documents(documents=lc_documents, embedding=embeddings)

        # Save to disk as files (index.faiss + index.pkl)
        vector_store.save_local(str(persist_dir))

        # SEC-001: Save SHA256 hash for integrity verification on load
        self._save_integrity_hash(persist_dir)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if DEBUG_MODE:
            debug_log(f"[VectorStore] Created index with {len(lc_documents)} unified chunks")
            debug_log(f"[VectorStore] Saved to: {persist_dir}")
            debug_log(f"[VectorStore] Build time: {elapsed_ms:.1f}ms")

        return VectorStoreResult(
            persist_dir=persist_dir,
            case_id=case_id,
            chunk_count=len(lc_documents),
            creation_time_ms=elapsed_ms,
        )

    def _convert_to_langchain_documents(self, documents: list[dict]) -> list:
        """
        Convert LocalScribe documents to LangChain Documents.

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

        # PERF-006: Cache text splitter as class attribute
        if not hasattr(self, "_text_splitter"):
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
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

                if DEBUG_MODE:
                    debug_log(f"[VectorStore] Split '{filename}' into {len(split_texts)} chunks")

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

        # Compute combined hash of both files
        hasher = hashlib.sha256()

        for file_path in [faiss_file, pkl_file]:
            if file_path.exists():
                with open(file_path, "rb") as f:
                    # Read in chunks for memory efficiency
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)

        # Save hash to file
        hash_file.write_text(hasher.hexdigest())

        if DEBUG_MODE:
            debug_log(f"[VectorStore] Saved integrity hash: {hasher.hexdigest()[:16]}...")

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
                debug_log(f"[VectorStore] Deleted: {persist_dir}")
                return True
            return False
        except Exception as e:
            debug_log(f"[VectorStore] Error deleting {persist_dir}: {e}")
            return False
