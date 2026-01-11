"""
FAISS Semantic Retrieval Algorithm for CasePrepd Q&A.

Implements semantic (embedding-based) retrieval using FAISS vector store
with HuggingFace sentence transformers.

Semantic vs Lexical Search:
- Lexical (BM25): Matches exact words and terms
- Semantic (FAISS): Matches meaning/concepts using neural embeddings

Why Include Semantic Search:
- Can find conceptually related content even with different wording
- "Who are the parties?" may find "plaintiff and defendant" even without exact match
- Complements BM25 for comprehensive retrieval

Model Choice (Session 85):
- Uses BAAI/bge-base-en-v1.5 (~110MB) - trained specifically for retrieval tasks
- Significantly better than previous all-MiniLM-L6-v2 for legal/medical terminology
- Bundled locally for offline use (no download at runtime)

This algorithm has lower weight (0.5) compared to BM25+ (1.0) since
semantic search is secondary to exact term matching for legal documents.
"""

import time
from typing import TYPE_CHECKING, Any

from src.config import DEBUG_MODE, EMBEDDING_MODEL_LOCAL_PATH, EMBEDDING_MODEL_NAME
from src.core.retrieval.algorithms import register_algorithm
from src.core.retrieval.base import (
    AlgorithmRetrievalResult,
    BaseRetrievalAlgorithm,
    DocumentChunk,
    RetrievedChunk,
)
from src.logging_config import debug_log

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings


def _get_embedding_model_path() -> str:
    """
    Get the embedding model path, preferring local bundled model.

    Returns:
        Local path if bundled model exists, otherwise HuggingFace model name
        for automatic download.
    """
    if EMBEDDING_MODEL_LOCAL_PATH.exists():
        return str(EMBEDDING_MODEL_LOCAL_PATH)
    return EMBEDDING_MODEL_NAME


# Embedding model - uses bundled local model if available, otherwise downloads
DEFAULT_EMBEDDING_MODEL = _get_embedding_model_path()


@register_algorithm
class FAISSRetriever(BaseRetrievalAlgorithm):
    """
    FAISS semantic retrieval algorithm using embeddings.

    Uses sentence transformers to create vector embeddings and FAISS
    for efficient similarity search.

    Attributes:
        name: Algorithm identifier ("FAISS")
        weight: Default weight for merging (0.5 - secondary to BM25+)
        enabled: Whether this algorithm is active

    Example:
        retriever = FAISSRetriever()
        retriever.index_documents(chunks)
        results = retriever.retrieve("Who are the plaintiffs?", k=5)

    Note:
        This algorithm requires embeddings to be initialized. Use set_embeddings()
        or pass embeddings via kwargs to index_documents().
    """

    name: str = "FAISS"
    weight: float = 0.5  # Lower weight - semantic less reliable for legal docs
    enabled: bool = True

    def __init__(self, embeddings: "HuggingFaceEmbeddings | None" = None):
        """
        Initialize FAISS retriever.

        Args:
            embeddings: Pre-loaded HuggingFace embeddings model.
                       If None, will be created on first use.
        """
        self._embeddings = embeddings
        self._vector_store: FAISS | None = None
        self._chunks: list[DocumentChunk] = []

    def set_embeddings(self, embeddings: "HuggingFaceEmbeddings") -> None:
        """
        Set the embeddings model.

        Args:
            embeddings: HuggingFace embeddings model to use
        """
        self._embeddings = embeddings

    def _ensure_embeddings(self) -> "HuggingFaceEmbeddings":
        """
        Ensure embeddings model is loaded.

        Creates default embeddings if not set.

        Returns:
            HuggingFaceEmbeddings instance
        """
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            if DEBUG_MODE:
                debug_log(f"[FAISS] Loading embeddings model: {DEFAULT_EMBEDDING_MODEL}")

            self._embeddings = HuggingFaceEmbeddings(
                model_name=DEFAULT_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        return self._embeddings

    def index_documents(self, chunks: list[DocumentChunk], **kwargs) -> None:
        """
        Build FAISS vector index from document chunks.

        Creates embeddings for each chunk and builds the FAISS index.

        Args:
            chunks: List of DocumentChunk objects to index
            **kwargs: Optional parameters:
                - embeddings: Override embeddings model

        Raises:
            ValueError: If chunks is empty
        """
        start_time = time.perf_counter()

        if not chunks:
            raise ValueError("Cannot index empty chunk list")

        # Use provided embeddings or default
        if "embeddings" in kwargs:
            self._embeddings = kwargs["embeddings"]

        embeddings = self._ensure_embeddings()
        self._chunks = chunks

        # Convert to LangChain documents
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        lc_documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "filename": chunk.filename,
                    "chunk_num": chunk.chunk_num,
                    "section_name": chunk.section_name,
                    "word_count": chunk.word_count,
                },
            )
            for chunk in chunks
        ]

        if DEBUG_MODE:
            debug_log(f"[FAISS] Creating embeddings for {len(lc_documents)} chunks...")

        # Build FAISS index
        self._vector_store = FAISS.from_documents(documents=lc_documents, embedding=embeddings)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if DEBUG_MODE:
            debug_log(f"[FAISS] Indexed {len(chunks)} chunks in {elapsed_ms:.1f}ms")

    def retrieve(self, query: str, k: int = 5) -> AlgorithmRetrievalResult:
        """
        Retrieve top-k relevant chunks using semantic similarity.

        Args:
            query: The search query string
            k: Maximum number of chunks to retrieve

        Returns:
            AlgorithmRetrievalResult with ranked chunks

        Raises:
            RuntimeError: If index_documents() hasn't been called
        """
        start_time = time.perf_counter()

        if not self.is_indexed:
            raise RuntimeError("Index not built. Call index_documents() first.")

        if DEBUG_MODE:
            debug_log(f"[FAISS] Query: '{query[:50]}...'")

        # Perform similarity search with scores
        # Returns list of (Document, score) tuples
        # Note: FAISS returns cosine distance, so higher = more similar
        docs_and_scores = self._vector_store.similarity_search_with_relevance_scores(query, k=k)

        # Build result chunks
        retrieved_chunks = []
        for doc, score in docs_and_scores:
            metadata = doc.metadata

            # FAISS relevance scores can be negative or > 1 depending on distance metric
            # Clamp to 0-1 range
            normalized_score = max(0.0, min(1.0, score))

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=metadata.get("chunk_id", ""),
                    text=doc.page_content,
                    relevance_score=normalized_score,
                    raw_score=score,
                    source_algorithm=self.name,
                    filename=metadata.get("filename", "unknown"),
                    chunk_num=metadata.get("chunk_num", 0),
                    section_name=metadata.get("section_name", "N/A"),
                    metadata={
                        "word_count": metadata.get("word_count", 0),
                        "embedding_model": DEFAULT_EMBEDDING_MODEL,
                    },
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if DEBUG_MODE:
            debug_log(f"[FAISS] Retrieved {len(retrieved_chunks)} chunks in {elapsed_ms:.1f}ms")
            for i, chunk in enumerate(retrieved_chunks[:3]):
                debug_log(
                    f"  [{i + 1}] score={chunk.raw_score:.3f} -> {chunk.relevance_score:.3f} | {chunk.filename}"
                )

        return AlgorithmRetrievalResult(
            chunks=retrieved_chunks,
            processing_time_ms=elapsed_ms,
            query=query,
            metadata={
                "algorithm": self.name,
                "index_size": len(self._chunks),
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
            },
        )

    @property
    def is_indexed(self) -> bool:
        """Check if the FAISS index is built."""
        return self._vector_store is not None and len(self._chunks) > 0

    def get_config(self) -> dict[str, Any]:
        """Return FAISS configuration."""
        config = super().get_config()
        config.update(
            {
                "index_size": len(self._chunks) if self._chunks else 0,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "distance_metric": "cosine",
            }
        )
        return config
