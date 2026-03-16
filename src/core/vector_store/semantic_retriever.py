"""
Semantic Retriever for CasePrepd.

Retrieves relevant document context for user questions using hybrid search
combining BM25+ (lexical) and FAISS (semantic) algorithms.

Architecture (Hybrid Retrieval):
- Loads documents from FAISS index on disk (backward compatible)
- Builds BM25+ index on-the-fly for lexical search
- Combines results from both algorithms using weighted merging
- Returns formatted context string with source attribution

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
    SEMANTIC_CONTEXT_WINDOW,
    SEMANTIC_RETRIEVAL_K,
)

logger = logging.getLogger(__name__)


def _get_effective_semantic_context_window() -> int:
    """
    Get semantic search context window (fixed token budget for retrieval).

    Returns:
        Context window size in tokens
    """
    return SEMANTIC_CONTEXT_WINDOW


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
        Retrieve top-k relevant chunks for a question.

        Uses query transformation (if available) to expand vague queries,
        then hybrid search (BM25+ + FAISS) to find the most relevant chunks.
        Filters by minimum relevance score if specified.

        Args:
            question: The user's question
            k: Number of chunks to retrieve. None = all chunks (from config or explicit)
            min_score: Minimum relevance score (0-1) to include (default: from config)

        Returns:
            RetrievalResult with formatted context and source information
        """
        import time

        start_time = time.perf_counter()

        # Use config default if not specified
        if k is None:
            k = SEMANTIC_RETRIEVAL_K

        # If k is still None (config says use all), use total chunk count
        if k is None:
            k = self.get_chunk_count()
            logger.debug("Using all %d chunks for retrieval", k)

        min_score = min_score if min_score is not None else RETRIEVAL_MIN_SCORE

        logger.debug("Query: '%s...' (k=%d, min_score=%s)", question[:50], k, min_score)

        # Retrieve chunks using hybrid retrieval
        merged_result = self._hybrid_retriever.retrieve(question, k=k)

        logger.debug("Received %d merged chunks from hybrid retriever", len(merged_result.chunks))

        all_chunks = {}  # chunk_id -> best chunk result (avoid duplicates)

        for chunk in merged_result.chunks:
            chunk_key = f"{chunk.filename}_{chunk.chunk_num}"

            # Keep the best score for each chunk
            if (
                chunk_key not in all_chunks
                or chunk.combined_score > all_chunks[chunk_key].combined_score
            ):
                all_chunks[chunk_key] = chunk

        # Sort all chunks by score descending and take top k
        sorted_chunks = sorted(all_chunks.values(), key=lambda c: c.combined_score, reverse=True)[
            :k
        ]

        # Rerank with cross-encoder if enabled (improves precision)
        if self._reranker and sorted_chunks:
            from src.config import RERANKER_TOP_K

            sorted_chunks = self._reranker.rerank(
                query=question,
                chunks=sorted_chunks,
                top_k=RERANKER_TOP_K,
            )
            logger.debug("Reranked to top %d chunks", len(sorted_chunks))

        # Filter by minimum score and build results
        # Track token count to stay within context window (approx 1 word = 1.3 tokens)
        context_parts = []
        sources = []
        estimated_tokens = 0
        # Semantic context scales with LLM context based on GPU VRAM
        semantic_context_window = _get_effective_semantic_context_window()
        # Reserve tokens for: system prompt (~200 tokens), question (~30 tokens),
        # formatting (~70 tokens), and LLM output (semantic_max_tokens from config).
        # Previous formula (80% for context) overflowed on small context windows,
        # causing prompt template overflow on small context windows.
        from src.config_defaults import get_default
        from src.core.semantic.semantic_constants import (
            COMPACT_PROMPT_THRESHOLD,
            COMPACT_SEMANTIC_PROMPT,
            FULL_SEMANTIC_PROMPT,
        )
        from src.core.semantic.token_budget import count_tokens as _count_tokens

        semantic_max_output_tokens = get_default("semantic_max_tokens")
        template = (
            COMPACT_SEMANTIC_PROMPT
            if semantic_context_window <= COMPACT_PROMPT_THRESHOLD
            else FULL_SEMANTIC_PROMPT
        )
        prompt_overhead_tokens = (
            _count_tokens(template.replace("{context}", "").replace("{question}", "")) + 30
        )  # ~30 tokens for a typical question
        max_context_tokens = max(
            200,  # minimum to avoid empty context
            semantic_context_window - semantic_max_output_tokens - prompt_overhead_tokens,
        )
        logger.debug(
            "Context window: %s, output reserve: %s, overhead: %s, max context tokens: %s",
            semantic_context_window,
            semantic_max_output_tokens,
            prompt_overhead_tokens,
            max_context_tokens,
        )
        chunks_included = 0
        chunks_skipped_score = 0
        chunks_skipped_limit = 0

        for chunk in sorted_chunks:
            # Skip low-relevance chunks
            if chunk.combined_score < min_score:
                chunks_skipped_score += 1
                continue

            # Format source citation for context
            source_cite = f"[{chunk.filename}"
            if chunk.section_name and chunk.section_name != "N/A":
                source_cite += f", {chunk.section_name}"
            source_cite += "]:"

            chunk_text = f"{source_cite}\n{chunk.text}"
            word_count = len(chunk.text.split())
            chunk_tokens = _count_tokens(chunk_text)

            # Check if adding this chunk would exceed context window
            if estimated_tokens + chunk_tokens > max_context_tokens:
                chunks_skipped_limit += 1
                if chunks_skipped_limit == 1:
                    logger.debug("Context window limit reached (%d tokens)", estimated_tokens)
                continue

            context_parts.append(chunk_text)
            estimated_tokens += chunk_tokens
            chunks_included += 1

            sources.append(
                SourceInfo(
                    filename=chunk.filename,
                    chunk_num=chunk.chunk_num,
                    section=chunk.section_name,
                    relevance_score=chunk.combined_score,
                    word_count=word_count,
                    sources=chunk.sources,  # Track which algorithms found this
                )
            )

        # Combine context parts with separator
        context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        # Always log retrieval summary when no results (helps diagnose issues)
        if chunks_included == 0:
            logger.warning(
                "No chunks passed filters! sorted_chunks=%d, skipped_score=%d, skipped_limit=%d, min_score=%s",
                len(sorted_chunks),
                chunks_skipped_score,
                chunks_skipped_limit,
                min_score,
            )
            # Log actual scores to diagnose why chunks are failing
            if sorted_chunks:
                logger.debug("Top chunk scores (combined_score | sources):")
                for i, chunk in enumerate(sorted_chunks[:5]):
                    logger.debug(
                        "  [%d] %.6f | %s | %s",
                        i + 1,
                        chunk.combined_score,
                        chunk.sources,
                        chunk.filename,
                    )
        elif chunks_skipped_score > 0 or chunks_skipped_limit > 0:
            logger.debug(
                "Chunks: %d included, %d below min_score, %d exceeded context limit",
                chunks_included,
                chunks_skipped_score,
                chunks_skipped_limit,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug("Retrieved %d chunks in %.1fms", len(sources), elapsed_ms)
        for src in sources:
            algo_info = " via %s" % src.sources if src.sources else ""
            logger.debug(
                "  - %s (chunk %d, score %.3f%s)",
                src.filename,
                src.chunk_num,
                src.relevance_score,
                algo_info,
            )

        return RetrievalResult(
            context=context,
            sources=sources,
            chunks_retrieved=len(sources),
            retrieval_time_ms=elapsed_ms,
        )

    def get_relevant_sources_summary(self, result: RetrievalResult) -> str:
        """
        Format source information for display.

        Creates a readable summary of sources used in the answer.

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
