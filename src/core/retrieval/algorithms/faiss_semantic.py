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

Model Choice:
- Uses nomic-ai/nomic-embed-text-v1.5 (137M params, 768 dims, 8192-token context)
- Downsized from modernbert-embed-large (1.58GB) — small embeddings + strong reranker
  performs equivalently, saving ~1.3GB for the installer
- Uses GPU when available, falls back to CPU
- Bundled locally for offline use (no download at runtime)

This algorithm has higher weight (0.8) compared to BM25+ (0.2) since
semantic search is primary for comprehensive retrieval of relevant content.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from src.config import EMBEDDING_MODEL_LOCAL_PATH, EMBEDDING_MODEL_NAME
from src.core.retrieval.algorithms import register_algorithm
from src.core.retrieval.base import (
    AlgorithmRetrievalResult,
    BaseRetrievalAlgorithm,
    DocumentChunk,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

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
    from src.core.utils.model_loader import resolve_model_path

    path, _ = resolve_model_path(EMBEDDING_MODEL_LOCAL_PATH, EMBEDDING_MODEL_NAME)
    return path


# Embedding model - uses bundled local model if available, otherwise downloads
DEFAULT_EMBEDDING_MODEL = _get_embedding_model_path()

# Module-level cached embeddings instance (shared across callers)
_shared_embeddings: "HuggingFaceEmbeddings | None" = None
_embeddings_lock = threading.Lock()


def _get_embedding_device() -> str:
    """
    Auto-detect best device for embedding model.

    Returns:
        "cuda" if NVIDIA GPU available, otherwise "cpu".
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_embeddings_model() -> "HuggingFaceEmbeddings":
    """
    Get a shared HuggingFaceEmbeddings model instance.

    Lazy-loads the embeddings model on first call and caches it.
    Uses GPU when available for faster embedding.
    Configures search_document/search_query prompt prefixes for nomic-embed-text.

    If the bundled model is present, uses local_files_only=True to prevent
    any network downloads. If not bundled, allows download but wraps in
    try/except for clear error messaging.

    Returns:
        HuggingFaceEmbeddings instance

    Raises:
        RuntimeError: If model cannot be loaded (missing bundled model + no network)
    """
    global _shared_embeddings
    if _shared_embeddings is not None:
        return _shared_embeddings

    with _embeddings_lock:
        # Double-check after acquiring lock
        if _shared_embeddings is not None:
            return _shared_embeddings

        from langchain_huggingface import HuggingFaceEmbeddings

        device = _get_embedding_device()
        is_local = EMBEDDING_MODEL_LOCAL_PATH.exists()
        logger.debug(
            "Loading shared embeddings model: %s (device=%s, local=%s)",
            DEFAULT_EMBEDDING_MODEL,
            device,
            is_local,
        )

        model_kwargs = {"device": device, "trust_remote_code": True}
        if is_local:
            model_kwargs["local_files_only"] = True

        try:
            _shared_embeddings = HuggingFaceEmbeddings(
                model_name=DEFAULT_EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True, "prompt": "search_document: "},
                query_encode_kwargs={"normalize_embeddings": True, "prompt": "search_query: "},
            )
        except Exception as e:
            logger.error(
                "Failed to load embedding model '%s': %s", DEFAULT_EMBEDDING_MODEL, e, exc_info=True
            )
            raise RuntimeError(
                f"Embedding model not available: {DEFAULT_EMBEDDING_MODEL}\n"
                f"Run: python scripts/download_models.py\n"
                f"Error: {e}"
            ) from e
    return _shared_embeddings


@register_algorithm
class FAISSRetriever(BaseRetrievalAlgorithm):
    """
    FAISS semantic retrieval algorithm using embeddings.

    Uses sentence transformers to create vector embeddings and FAISS
    for efficient similarity search.

    Attributes:
        name: Algorithm identifier ("FAISS")
        weight: Default weight for merging (0.8 - primary algorithm)
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
    weight: float = 0.8  # Primary weight - semantic for comprehensive retrieval
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
            self._embeddings = get_embeddings_model()

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

        logger.debug("Creating embeddings for %d chunks...", len(lc_documents))

        # Build FAISS index with inner product (embeddings are L2-normalized,
        # so inner product = cosine similarity, giving scores in [0, 1])
        from langchain_community.vectorstores.utils import DistanceStrategy

        self._vector_store = FAISS.from_documents(
            documents=lc_documents,
            embedding=embeddings,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug("Indexed %d chunks in %.1fms", len(chunks), elapsed_ms)

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

        logger.debug("Query: '%s...'", query[:50])

        # Use raw FAISS scores (inner product on normalized embeddings = cosine similarity)
        # Avoids LangChain's relevance_score transformations which invert the scores
        docs_and_scores = self._vector_store.similarity_search_with_score(query, k=k)

        # Build result chunks
        retrieved_chunks = []
        for doc, score in docs_and_scores:
            metadata = doc.metadata

            # Raw inner product on normalized vectors = cosine similarity ∈ [0, 1]
            # Clamp in case of numerical edge cases
            normalized_score = max(0.0, min(float(score), 1.0))

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

        logger.debug("Retrieved %d chunks in %.1fms", len(retrieved_chunks), elapsed_ms)
        for i, chunk in enumerate(retrieved_chunks[:3]):
            logger.debug(
                "  [%d] score=%.3f -> %.3f | %s",
                i + 1,
                chunk.raw_score,
                chunk.relevance_score,
                chunk.filename,
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
