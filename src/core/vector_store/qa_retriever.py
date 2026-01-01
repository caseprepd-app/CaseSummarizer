"""
Q&A Retriever for LocalScribe.

Retrieves relevant document context for user questions using hybrid search
combining BM25+ (lexical) and FAISS (semantic) algorithms.

Architecture (Session 31 - Hybrid Retrieval):
- Loads documents from FAISS index on disk (backward compatible)
- Builds BM25+ index on-the-fly for lexical search
- Combines results from both algorithms using weighted merging
- Returns formatted context string with source attribution

Integration:
- Used by QAWorker in background thread
- Provides context to Ollama for answer generation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import (
    DEBUG_MODE,
    QA_CONTEXT_WINDOW,
    QA_RETRIEVAL_K,
    QUERY_TRANSFORM_ENABLED,
    QUERY_TRANSFORM_TIMEOUT,
    QUERY_TRANSFORM_VARIANTS,
    RETRIEVAL_ALGORITHM_WEIGHTS,
    RETRIEVAL_ENABLE_BM25,
    RETRIEVAL_ENABLE_FAISS,
    RETRIEVAL_MIN_SCORE,
)
from src.logging_config import debug_log


def _get_effective_qa_context_window() -> int:
    """
    Get effective QA context window from user preferences (Session 67).

    QA context window now scales with LLM context window based on GPU VRAM.
    Falls back to config constant if preferences unavailable.

    Returns:
        Context window size in tokens
    """
    try:
        from src.user_preferences import get_user_preferences
        prefs = get_user_preferences()
        return prefs.get_effective_context_size()
    except Exception:
        return QA_CONTEXT_WINDOW

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS
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


class QARetriever:
    """
    Retrieves relevant context for Q&A using hybrid search.

    Combines BM25+ (lexical) and FAISS (semantic) search for comprehensive
    document retrieval. BM25+ handles exact terminology matching while
    FAISS can find conceptually related content.

    Example:
        retriever = QARetriever(persist_dir, embeddings)
        result = retriever.retrieve_context("Who are the plaintiffs?")
        print(f"Context: {result.context}")
        print(f"Sources: {[s.filename for s in result.sources]}")
    """

    def __init__(
        self,
        vector_store_path: Path,
        embeddings: "HuggingFaceEmbeddings"
    ):
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

        # Load FAISS index from disk (for backward compatibility and document access)
        # allow_dangerous_deserialization=True is safe because we control the data
        self._faiss_store = FAISS.load_local(
            folder_path=str(self.vector_store_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        if DEBUG_MODE:
            debug_log(f"[QARetriever] Loaded FAISS index from: {self.vector_store_path}")

        # Extract documents from FAISS docstore for hybrid retrieval
        self._documents = self._extract_documents_from_faiss()

        # Initialize hybrid retriever
        self._hybrid_retriever = self._init_hybrid_retriever()

        # Initialize query transformer (LlamaIndex + Ollama)
        self._query_transformer = self._init_query_transformer()

        if DEBUG_MODE:
            debug_log(f"[QARetriever] Hybrid retriever initialized with {len(self._documents)} chunks")
            if self._query_transformer:
                debug_log("[QARetriever] Query transformer enabled")

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
            documents.append({
                "filename": filename,
                "chunks": chunks,
            })

        return documents

    def _init_hybrid_retriever(self):
        """
        Initialize the hybrid retriever with extracted documents.

        Returns:
            HybridRetriever instance
        """
        from src.core.retrieval import HybridRetriever

        # Create hybrid retriever with config settings
        retriever = HybridRetriever(
            algorithm_weights=RETRIEVAL_ALGORITHM_WEIGHTS,
            embeddings=self.embeddings,
            enable_bm25=RETRIEVAL_ENABLE_BM25,
            enable_faiss=RETRIEVAL_ENABLE_FAISS,
        )

        # Index documents
        retriever.index_documents(self._documents)

        return retriever

    def _init_query_transformer(self):
        """
        Initialize the query transformer for expanding vague queries.

        Uses LlamaIndex + Ollama to generate query variants.

        Returns:
            QueryTransformer instance or None if disabled/unavailable
        """
        if not QUERY_TRANSFORM_ENABLED:
            return None

        try:
            from src.core.retrieval import QueryTransformer

            transformer = QueryTransformer(
                variant_count=QUERY_TRANSFORM_VARIANTS,
                timeout_seconds=QUERY_TRANSFORM_TIMEOUT,
                enabled=True,
            )

            # Check if LlamaIndex + Ollama is available
            if transformer.is_available():
                return transformer
            else:
                if DEBUG_MODE:
                    debug_log("[QARetriever] Query transformer not available (LlamaIndex/Ollama issue)")
                return None

        except ImportError as e:
            if DEBUG_MODE:
                debug_log(f"[QARetriever] Query transformer import failed: {e}")
            return None
        except Exception as e:
            if DEBUG_MODE:
                debug_log(f"[QARetriever] Query transformer init failed: {e}")
            return None

    def retrieve_context(
        self,
        question: str,
        k: int | None = None,
        min_score: float | None = None
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
            k = QA_RETRIEVAL_K

        # If k is still None (config says use all), use total chunk count
        if k is None:
            k = self.get_chunk_count()
            if DEBUG_MODE:
                debug_log(f"[QARetriever] Using all {k} chunks for retrieval")

        min_score = min_score if min_score is not None else RETRIEVAL_MIN_SCORE

        if DEBUG_MODE:
            debug_log(f"[QARetriever] Query: '{question[:50]}...' (k={k}, min_score={min_score})")

        # Transform query into variants if transformer is available
        queries_to_search = [question]
        if self._query_transformer:
            transform_result = self._query_transformer.transform(question)
            if transform_result.success and transform_result.expanded_queries:
                queries_to_search = transform_result.all_queries
                if DEBUG_MODE:
                    debug_log(f"[QARetriever] Query expanded to {len(queries_to_search)} variants")

        # Retrieve for all query variants and merge results
        all_chunks = {}  # chunk_id -> best chunk result (avoid duplicates)

        for query in queries_to_search:
            merged_result = self._hybrid_retriever.retrieve(query, k=k)

            for chunk in merged_result.chunks:
                chunk_key = f"{chunk.filename}_{chunk.chunk_num}"

                # Keep the best score for each chunk
                if chunk_key not in all_chunks or chunk.combined_score > all_chunks[chunk_key].combined_score:
                    all_chunks[chunk_key] = chunk

        # Sort all chunks by score descending and take top k
        sorted_chunks = sorted(all_chunks.values(), key=lambda c: c.combined_score, reverse=True)[:k]

        if DEBUG_MODE and len(queries_to_search) > 1:
            debug_log(f"[QARetriever] Merged {len(all_chunks)} unique chunks from {len(queries_to_search)} queries")

        # Filter by minimum score and build results
        # Track token count to stay within context window (approx 1 word = 1.3 tokens)
        context_parts = []
        sources = []
        estimated_tokens = 0
        # Session 67: QA context now scales with LLM context based on GPU VRAM
        qa_context_window = _get_effective_qa_context_window()
        max_context_tokens = int(qa_context_window * 0.8)  # Reserve 20% for prompt + answer
        if DEBUG_MODE:
            debug_log(f"[QARetriever] Using context window: {qa_context_window:,} "
                      f"(max context tokens: {max_context_tokens:,})")
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
            chunk_tokens = int(word_count * 1.3)  # Approximate token count

            # Check if adding this chunk would exceed context window
            if estimated_tokens + chunk_tokens > max_context_tokens:
                chunks_skipped_limit += 1
                if DEBUG_MODE and chunks_skipped_limit == 1:
                    debug_log(f"[QARetriever] Context window limit reached ({estimated_tokens} tokens)")
                continue

            context_parts.append(chunk_text)
            estimated_tokens += chunk_tokens
            chunks_included += 1

            sources.append(SourceInfo(
                filename=chunk.filename,
                chunk_num=chunk.chunk_num,
                section=chunk.section_name,
                relevance_score=chunk.combined_score,
                word_count=word_count,
                sources=chunk.sources,  # Track which algorithms found this
            ))

        # Combine context parts with separator
        context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        if DEBUG_MODE and (chunks_skipped_score > 0 or chunks_skipped_limit > 0):
            debug_log(f"[QARetriever] Chunks: {chunks_included} included, "
                      f"{chunks_skipped_score} below min_score, {chunks_skipped_limit} exceeded context limit")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if DEBUG_MODE:
            debug_log(f"[QARetriever] Retrieved {len(sources)} chunks in {elapsed_ms:.1f}ms")
            for src in sources:
                algo_info = f" via {src.sources}" if src.sources else ""
                debug_log(f"  - {src.filename} (chunk {src.chunk_num}, score {src.relevance_score:.3f}{algo_info})")

        return RetrievalResult(
            context=context,
            sources=sources,
            chunks_retrieved=len(sources),
            retrieval_time_ms=elapsed_ms
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
