"""
Semantic Retriever for CasePrepd.

Retrieves the single best document chunk for user questions using hybrid
search combining BM25+ (lexical) and FAISS (semantic) algorithms, then
cross-encoder reranking to pick the top-1 result.

Architecture:
- Loads documents from FAISS index on disk (backward compatible)
- Builds BM25+ index on-the-fly for lexical search
- Retrieves k candidates via hybrid search (k scales with corpus size)
- Cross-encoder reranks candidates and returns the single best chunk
- Returns raw chunk text + source metadata for citation extraction

Integration:
- Used by SemanticWorker in background thread
- Provides context for search result citation extraction
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import (
    RETRIEVAL_ALGORITHM_WEIGHTS,
    RETRIEVAL_ENABLE_BM25,
    RETRIEVAL_ENABLE_FAISS,
    RETRIEVAL_MIN_SCORE,
    SEMANTIC_RETRIEVAL_K,
)

logger = logging.getLogger(__name__)


def _get_effective_algorithm_weights() -> dict[str, float]:
    """
    Get retrieval algorithm weights from user preferences.

    Falls back to config defaults if preferences unavailable.

    Returns:
        Dict mapping algorithm name to weight
    """
    try:
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        faiss_w = prefs.get("retrieval_weight_faiss", RETRIEVAL_ALGORITHM_WEIGHTS["FAISS"])
        bm25_w = prefs.get("retrieval_weight_bm25", RETRIEVAL_ALGORITHM_WEIGHTS["BM25+"])
        if not isinstance(faiss_w, (int, float)):
            faiss_w = RETRIEVAL_ALGORITHM_WEIGHTS["FAISS"]
        if not isinstance(bm25_w, (int, float)):
            bm25_w = RETRIEVAL_ALGORITHM_WEIGHTS["BM25+"]
        return {"FAISS": faiss_w, "BM25+": bm25_w}
    except Exception as e:
        logger.warning("Could not load retrieval algorithm weights: %s", e)
        return RETRIEVAL_ALGORITHM_WEIGHTS


if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings


@dataclass
class SourceInfo:
    """Source information for a retrieved chunk."""

    filename: str
    chunk_num: int
    section: str
    relevance_score: float
    word_count: int
    sources: list[str] | None = None  # Which algorithms found this chunk


@dataclass
class RetrievalResult:
    """Result of context retrieval."""

    context: str
    sources: list[SourceInfo]
    chunks_retrieved: int
    retrieval_time_ms: float


class SemanticRetriever:
    """
    Retrieves relevant context for semantic search using hybrid search.

    Combines BM25+ (lexical) and FAISS (semantic) search for comprehensive
    document retrieval. BM25+ handles exact terminology matching while
    FAISS can find conceptually related content.

    Example:
        retriever = SemanticRetriever(persist_dir, embeddings)
        result = retriever.retrieve_context("Who are the plaintiffs?")
        print(f"Context: {result.context}")
        print(f"Sources: {[s.filename for s in result.sources]}")
    """

    def __init__(self, vector_store_path: Path, embeddings: "HuggingFaceEmbeddings"):
        """
        Initialize retriever with existing vector store.

        Loads documents from FAISS index and builds BM25+ index.

        Args:
            vector_store_path: Path to directory containing index.faiss/index.pkl
            embeddings: HuggingFaceEmbeddings model for query encoding

        Raises:
            FileNotFoundError: If vector store files don't exist
        """
        from langchain_community.vectorstores import FAISS

        self.vector_store_path = Path(vector_store_path)
        self.embeddings = embeddings

        # Verify files exist
        faiss_file = self.vector_store_path / "index.faiss"
        if not faiss_file.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.vector_store_path}. "
                "Ensure documents have been processed first."
            )

        # SEC-001: Verify integrity hash before loading (if hash file exists)
        self._verify_integrity_hash(self.vector_store_path)

        # Load FAISS index from disk (for backward compatibility and document access)
        # allow_dangerous_deserialization=True is safe because we verify hash first
        self._faiss_store = FAISS.load_local(
            folder_path=str(self.vector_store_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        logger.debug("Loaded FAISS index from: %s", self.vector_store_path)

        # Extract documents from FAISS docstore for hybrid retrieval
        self._documents = self._extract_documents_from_faiss()

        # Initialize hybrid retriever
        self._hybrid_retriever = self._init_hybrid_retriever()

        # Initialize cross-encoder reranker (lazy-loaded on first use)
        self._reranker = self._init_reranker()

        logger.debug("Hybrid retriever initialized with %d chunks", len(self._documents))
        if self._reranker:
            logger.debug("Cross-encoder reranking enabled")

    def _verify_integrity_hash(self, persist_dir: Path) -> None:
        """
        Verify SHA256 hash of vector store files before loading (SEC-001).

        Computes hash of index.faiss and index.pkl files and compares
        against stored .hash file. Raises error if hash mismatch detected.

        For backward compatibility, skips verification if .hash file doesn't exist
        (older vector stores created before this security feature).

        Args:
            persist_dir: Directory containing the vector store files

        Raises:
            ValueError: If integrity check fails (hash mismatch)
        """
        hash_file = persist_dir / ".hash"

        # Skip verification for older stores without hash file
        if not hash_file.exists():
            logger.debug("No .hash file found - skipping integrity check (legacy store)")
            return

        # Compute current hash of files
        faiss_file = persist_dir / "index.faiss"
        pkl_file = persist_dir / "index.pkl"

        hasher = hashlib.sha256()
        for file_path in [faiss_file, pkl_file]:
            if file_path.exists():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)

        computed_hash = hasher.hexdigest()
        stored_hash = hash_file.read_text().strip()

        if computed_hash != stored_hash:
            raise ValueError(
                f"Vector store integrity check failed at {persist_dir}. "
                "Files may have been tampered with or corrupted. "
                "Please rebuild the vector store from source documents."
            )

        logger.debug("Integrity check passed: %s...", computed_hash[:16])

    def _extract_documents_from_faiss(self) -> list[dict]:
        """
        Extract document texts and metadata from FAISS docstore.

        Returns:
            List of document dicts with extracted_text and filename
        """
        documents = []

        # FAISS stores documents in docstore with index_to_docstore_id mapping
        docstore = self._faiss_store.docstore
        index_to_id = self._faiss_store.index_to_docstore_id

        # Group by filename for better organization
        chunks_by_file: dict[str, list[dict]] = {}

        for idx, doc_id in index_to_id.items():
            doc = docstore.search(doc_id)
            if doc is None:
                continue

            metadata = doc.metadata
            filename = metadata.get("filename", "unknown")

            chunk_info = {
                "text": doc.page_content,
                "chunk_num": metadata.get("chunk_num", idx),
                "section_name": metadata.get("section_name", "N/A"),
                "word_count": metadata.get("word_count", len(doc.page_content.split())),
            }

            if filename not in chunks_by_file:
                chunks_by_file[filename] = []
            chunks_by_file[filename].append(chunk_info)

        # Convert to document format expected by HybridRetriever
        for filename, chunks in chunks_by_file.items():
            documents.append(
                {
                    "filename": filename,
                    "chunks": chunks,
                }
            )

        return documents

    def _init_hybrid_retriever(self):
        """
        Initialize the hybrid retriever with extracted documents.

        Returns:
            HybridRetriever instance
        """
        from src.core.retrieval import HybridRetriever

        # Create hybrid retriever with user-configurable weights
        retriever = HybridRetriever(
            algorithm_weights=_get_effective_algorithm_weights(),
            embeddings=self.embeddings,
            enable_bm25=RETRIEVAL_ENABLE_BM25,
            enable_faiss=RETRIEVAL_ENABLE_FAISS,
        )

        # Index documents
        retriever.index_documents(self._documents)

        return retriever

    def _init_reranker(self):
        """
        Initialize the cross-encoder reranker for improved precision.

        Returns:
            CrossEncoderReranker instance or None if disabled
        """
        from src.config import RERANKING_ENABLED

        if not RERANKING_ENABLED:
            return None

        try:
            from src.core.retrieval import CrossEncoderReranker

            return CrossEncoderReranker()

        except ImportError as e:
            logger.debug("Reranker import failed: %s", e)
            return None
        except Exception as e:
            logger.debug("Reranker init failed: %s", e)
            return None

    def retrieve_context(
        self, question: str, k: int | None = None, min_score: float | None = None
    ) -> RetrievalResult:
        """
        Retrieve the single best chunk for a question.

        Uses hybrid search (BM25+ + FAISS) to find candidates, then
        cross-encoder reranking to select the top-1 result. Returns
        raw chunk text (no source prefix) for citation extraction.

        Args:
            question: The user's question
            k: Candidate pool size for hybrid retrieval (default: from config)
            min_score: Minimum relevance score (0-1) to include (default: from config)

        Returns:
            RetrievalResult with 0 or 1 sources and raw chunk text as context
        """
        import time

        start_time = time.perf_counter()
        min_score = min_score if min_score is not None else RETRIEVAL_MIN_SCORE

        # Determine candidate pool size (k controls how many the reranker sees)
        candidate_k = self._get_candidate_k(k)
        logger.debug(
            "Query: '%s...' (candidates=%d, min_score=%s)", question[:50], candidate_k, min_score
        )

        # Retrieve and deduplicate candidates
        sorted_chunks = self._retrieve_candidates(question, candidate_k)

        # Rerank to find the single best chunk
        if self._reranker and sorted_chunks:
            sorted_chunks = self._reranker.rerank(query=question, chunks=sorted_chunks, top_k=1)

        # Take the top-1 chunk if it passes the quality floor
        best = self._select_best_chunk(sorted_chunks, min_score)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if best:
            logger.debug(
                "Best chunk: %s #%d (score=%.3f) in %.1fms",
                best.filename,
                best.chunk_num,
                best.combined_score,
                elapsed_ms,
            )
        else:
            self._log_empty_result(sorted_chunks, min_score)

        return self._build_result(best, elapsed_ms)

    def _get_candidate_k(self, k: int | None) -> int:
        """
        Determine candidate pool size for hybrid retrieval.

        Adaptive: scales with corpus size. Config value is ceiling, not target.
        None means use all chunks.
        """
        if k is None:
            k = SEMANTIC_RETRIEVAL_K
        if k is None:
            return self.get_chunk_count()
        chunk_count = self.get_chunk_count()
        adaptive_k = min(k, max(5, chunk_count // 3))
        if adaptive_k != k:
            logger.debug("Adaptive k: %d -> %d (corpus=%d)", k, adaptive_k, chunk_count)
        return adaptive_k

    def _retrieve_candidates(self, question: str, k: int) -> list:
        """
        Run hybrid retrieval and deduplicate by chunk identity.

        Returns:
            List of MergedChunk objects sorted by score descending
        """
        merged_result = self._hybrid_retriever.retrieve(question, k=k)
        all_chunks = {}
        for chunk in merged_result.chunks:
            chunk_key = f"{chunk.filename}_{chunk.chunk_num}"
            if (
                chunk_key not in all_chunks
                or chunk.combined_score > all_chunks[chunk_key].combined_score
            ):
                all_chunks[chunk_key] = chunk
        return sorted(all_chunks.values(), key=lambda c: c.combined_score, reverse=True)[:k]

    def _select_best_chunk(self, sorted_chunks: list, min_score: float):
        """Return the top chunk if it passes min_score, else None."""
        if not sorted_chunks:
            return None
        best = sorted_chunks[0]
        if best.combined_score < min_score:
            return None
        return best

    def _log_empty_result(self, sorted_chunks: list, min_score: float) -> None:
        """Log diagnostic info when no chunk passed quality filters."""
        logger.warning("No chunk passed filters (min_score=%s)", min_score)
        if sorted_chunks:
            for i, chunk in enumerate(sorted_chunks[:3]):
                logger.debug(
                    "  [%d] %.4f | %s | %s",
                    i + 1,
                    chunk.combined_score,
                    chunk.sources,
                    chunk.filename,
                )

    def _build_result(self, best_chunk, elapsed_ms: float) -> RetrievalResult:
        """Build RetrievalResult from the single best chunk (or empty)."""
        if best_chunk is None:
            return RetrievalResult(
                context="", sources=[], chunks_retrieved=0, retrieval_time_ms=elapsed_ms
            )
        source = SourceInfo(
            filename=best_chunk.filename,
            chunk_num=best_chunk.chunk_num,
            section=best_chunk.section_name,
            relevance_score=best_chunk.combined_score,
            word_count=len(best_chunk.text.split()),
            sources=best_chunk.sources,
        )
        return RetrievalResult(
            context=best_chunk.text,
            sources=[source],
            chunks_retrieved=1,
            retrieval_time_ms=elapsed_ms,
        )

    def get_relevant_sources_summary(self, result: RetrievalResult) -> str:
        """
        Format source information for display.

        Creates a readable summary of sources used in the retrieval result.

        Args:
            result: RetrievalResult from retrieve_context()

        Returns:
            Formatted string like "complaint.pdf (Section Parties), answer.pdf"
        """
        if not result.sources:
            return "No sources found"

        summaries = []
        seen_files = set()

        for source in result.sources:
            if source.filename not in seen_files:
                if source.section and source.section != "N/A":
                    summaries.append(f"{source.filename} ({source.section})")
                else:
                    summaries.append(source.filename)
                seen_files.add(source.filename)

        return ", ".join(summaries)

    def get_chunk_count(self) -> int:
        """
        Get total number of chunks in the vector store.

        Returns:
            Number of indexed chunks
        """
        return self._hybrid_retriever.get_chunk_count()

    def get_algorithm_status(self) -> dict:
        """
        Get status of retrieval algorithms.

        Returns:
            Dictionary with algorithm name -> status info
        """
        return self._hybrid_retriever.get_algorithm_status()
