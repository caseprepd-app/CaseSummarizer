"""
Hybrid Retriever for LocalScribe Q&A.

Coordinates multiple retrieval algorithms (BM25+, FAISS) and merges their
results for comprehensive document search. This is the main entry point for
the Q&A retrieval system.

Architecture:
- Manages multiple retrieval algorithm instances
- Indexes documents into all enabled algorithms
- Runs parallel retrieval and merges results
- Provides unified interface for QAOrchestrator

Why Hybrid Retrieval:
- BM25+: Reliable for exact terminology matching (primary)
- FAISS: Can find conceptually related content (secondary)
- Combined: Best of both worlds with weighted scoring

This mirrors the VocabularyExtractor pattern of using multiple algorithms
with weighted result merging.
"""

import time
from typing import TYPE_CHECKING, Any

from src.config import DEBUG_MODE
from src.logging_config import debug_log
from src.core.retrieval.base import AlgorithmRetrievalResult, DocumentChunk
from src.core.retrieval.chunk_merger import ChunkMerger, MergedRetrievalResult

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings

# Default algorithm weights - BM25+ is primary for legal documents
DEFAULT_ALGORITHM_WEIGHTS = {
    "BM25+": 1.0,   # Primary - reliable for exact legal terminology
    "FAISS": 0.5,   # Secondary - semantic can help but less reliable
}


class HybridRetriever:
    """
    Hybrid retrieval coordinator for multi-algorithm search.

    Manages BM25+ and FAISS retrievers, indexes documents into both,
    and merges their results for comprehensive retrieval.

    Attributes:
        algorithms: List of active retrieval algorithm instances
        merger: ChunkMerger for combining results
        algorithm_weights: Current weights for each algorithm

    Example:
        retriever = HybridRetriever()
        retriever.index_documents(documents)
        results = retriever.retrieve("Who are the plaintiffs?", k=5)

        for chunk in results.chunks:
            print(f"[{chunk.combined_score:.2f}] {chunk.text[:100]}...")
            print(f"  Sources: {chunk.sources}")
    """

    def __init__(
        self,
        algorithm_weights: dict[str, float] | None = None,
        embeddings: "HuggingFaceEmbeddings | None" = None,
        enable_bm25: bool = True,
        enable_faiss: bool = True,
    ):
        """
        Initialize hybrid retriever.

        Args:
            algorithm_weights: Custom weights for algorithms (default: BM25+=1.0, FAISS=0.5)
            embeddings: Pre-loaded embeddings for FAISS (optional, loaded on demand)
            enable_bm25: Whether to use BM25+ algorithm
            enable_faiss: Whether to use FAISS semantic algorithm
        """
        self.algorithm_weights = algorithm_weights or DEFAULT_ALGORITHM_WEIGHTS.copy()
        self._embeddings = embeddings
        self._enable_bm25 = enable_bm25
        self._enable_faiss = enable_faiss

        # Initialize algorithms
        self._algorithms = {}
        self._init_algorithms()

        # Initialize merger with weights
        self.merger = ChunkMerger(algorithm_weights=self.algorithm_weights)

        # Document storage for re-indexing
        self._chunks: list[DocumentChunk] = []

        if DEBUG_MODE:
            enabled = [name for name, algo in self._algorithms.items() if algo.enabled]
            debug_log(f"[HybridRetriever] Initialized with algorithms: {enabled}")

    def _init_algorithms(self) -> None:
        """Initialize retrieval algorithm instances."""
        # Import here to avoid circular imports
        from src.core.retrieval.algorithms.bm25_plus import BM25PlusRetriever
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        if self._enable_bm25:
            bm25 = BM25PlusRetriever()
            bm25.weight = self.algorithm_weights.get("BM25+", 1.0)
            self._algorithms["BM25+"] = bm25

        if self._enable_faiss:
            faiss = FAISSRetriever(embeddings=self._embeddings)
            faiss.weight = self.algorithm_weights.get("FAISS", 0.5)
            self._algorithms["FAISS"] = faiss

    def set_embeddings(self, embeddings: "HuggingFaceEmbeddings") -> None:
        """
        Set embeddings model for FAISS algorithm.

        Args:
            embeddings: HuggingFace embeddings model
        """
        self._embeddings = embeddings
        if "FAISS" in self._algorithms:
            self._algorithms["FAISS"].set_embeddings(embeddings)

    def index_documents(
        self,
        documents: list[dict],
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> int:
        """
        Index documents into all enabled algorithms.

        Converts documents to chunks and indexes into each algorithm.
        Documents can have pre-computed chunks or raw text.

        Args:
            documents: List of document dicts with keys:
                - 'filename': str
                - 'chunks': list[dict] (optional, pre-computed chunks)
                - 'extracted_text': str (optional, will be chunked)
            chunk_size: Characters per chunk for text splitting
            chunk_overlap: Overlap between chunks

        Returns:
            Number of chunks indexed

        Raises:
            ValueError: If no valid content found in documents
        """
        start_time = time.perf_counter()

        # Convert documents to DocumentChunks
        self._chunks = self._convert_to_chunks(documents, chunk_size, chunk_overlap)

        if not self._chunks:
            raise ValueError("No valid chunks found in documents")

        if DEBUG_MODE:
            debug_log(f"[HybridRetriever] Indexing {len(self._chunks)} chunks into {len(self._algorithms)} algorithms")

        # Index into each algorithm
        for name, algorithm in self._algorithms.items():
            if algorithm.enabled:
                try:
                    algorithm.index_documents(self._chunks)
                    if DEBUG_MODE:
                        debug_log(f"[HybridRetriever] {name} indexing complete")
                except Exception as e:
                    debug_log(f"[HybridRetriever] {name} indexing failed: {e}")
                    algorithm.enabled = False

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if DEBUG_MODE:
            debug_log(f"[HybridRetriever] Total indexing time: {elapsed_ms:.1f}ms")

        return len(self._chunks)

    def _convert_to_chunks(
        self,
        documents: list[dict],
        chunk_size: int,
        chunk_overlap: int
    ) -> list[DocumentChunk]:
        """
        Convert document dicts to DocumentChunk objects.

        Handles both pre-chunked documents and raw text.

        Args:
            documents: List of document dicts
            chunk_size: Characters per chunk for text splitting
            chunk_overlap: Overlap between chunks

        Returns:
            List of DocumentChunk objects
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = []
        chunk_counter = 0

        for doc in documents:
            filename = doc.get("filename", "unknown")
            pre_chunks = doc.get("chunks", [])

            # Use pre-computed chunks if available
            if pre_chunks:
                for chunk_data in pre_chunks:
                    # Handle both dict and object formats
                    if hasattr(chunk_data, "text"):
                        text = chunk_data.text
                        chunk_num = chunk_data.chunk_num
                        section_name = chunk_data.section_name or "N/A"
                    else:
                        text = chunk_data.get("text", "")
                        chunk_num = chunk_data.get("chunk_num", 0)
                        section_name = chunk_data.get("section_name", "N/A")

                    if not text.strip():
                        continue

                    chunks.append(DocumentChunk(
                        text=text,
                        chunk_id=f"{filename}_{chunk_counter}",
                        filename=filename,
                        chunk_num=chunk_num,
                        section_name=section_name,
                    ))
                    chunk_counter += 1

            # Otherwise, chunk the extracted_text
            elif doc.get("extracted_text"):
                text = doc["extracted_text"]
                if not text.strip():
                    continue

                split_texts = text_splitter.split_text(text)

                for i, chunk_text in enumerate(split_texts):
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        chunk_id=f"{filename}_{chunk_counter}",
                        filename=filename,
                        chunk_num=i,
                        section_name="Auto-chunked",
                    ))
                    chunk_counter += 1

        return chunks

    def retrieve(self, query: str, k: int = 5) -> MergedRetrievalResult:
        """
        Retrieve top-k relevant chunks using all enabled algorithms.

        Runs retrieval on each algorithm and merges results.

        Args:
            query: The search query string
            k: Maximum number of chunks to return

        Returns:
            MergedRetrievalResult with ranked chunks from all algorithms

        Raises:
            RuntimeError: If index_documents() hasn't been called
        """
        start_time = time.perf_counter()

        if not self.is_indexed:
            raise RuntimeError("Documents not indexed. Call index_documents() first.")

        if DEBUG_MODE:
            debug_log(f"[HybridRetriever] Query: '{query[:50]}...'")

        # Collect results from all enabled algorithms
        algorithm_results: list[AlgorithmRetrievalResult] = []

        for name, algorithm in self._algorithms.items():
            if not algorithm.enabled or not algorithm.is_indexed:
                continue

            try:
                # Request more chunks from each algorithm to allow merging
                result = algorithm.retrieve(query, k=k * 2)
                algorithm_results.append(result)

                if DEBUG_MODE:
                    debug_log(f"[HybridRetriever] {name}: {len(result)} chunks retrieved")

            except Exception as e:
                debug_log(f"[HybridRetriever] {name} retrieval failed: {e}")

        if not algorithm_results:
            # No algorithms returned results - return empty
            return MergedRetrievalResult(
                chunks=[],
                total_algorithms=0,
                processing_time_ms=0,
                query=query,
                metadata={"error": "No algorithms returned results"}
            )

        # Merge results
        merged = self.merger.merge(algorithm_results, k=k)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        merged.processing_time_ms = elapsed_ms

        if DEBUG_MODE:
            debug_log(f"[HybridRetriever] Merged {len(merged)} chunks in {elapsed_ms:.1f}ms")
            for i, chunk in enumerate(merged.chunks[:3]):
                debug_log(f"  [{i + 1}] score={chunk.combined_score:.3f} | sources={chunk.sources}")

        return merged

    @property
    def is_indexed(self) -> bool:
        """Check if at least one algorithm has indexed documents."""
        return any(
            algo.is_indexed
            for algo in self._algorithms.values()
            if algo.enabled
        )

    def get_algorithm_status(self) -> dict[str, dict[str, Any]]:
        """
        Get status of all algorithms.

        Returns:
            Dictionary with algorithm name -> status dict
        """
        return {
            name: {
                "enabled": algo.enabled,
                "indexed": algo.is_indexed,
                "weight": algo.weight,
                **algo.get_config()
            }
            for name, algo in self._algorithms.items()
        }

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Update algorithm weights.

        Updates both the internal weights and the merger.

        Args:
            new_weights: New weight mapping
        """
        self.algorithm_weights.update(new_weights)
        self.merger.update_weights(new_weights)

        # Update algorithm instances
        for name, weight in new_weights.items():
            if name in self._algorithms:
                self._algorithms[name].weight = weight

        if DEBUG_MODE:
            debug_log(f"[HybridRetriever] Updated weights: {self.algorithm_weights}")

    def get_chunk_count(self) -> int:
        """
        Get total number of indexed chunks.

        Returns:
            Number of chunks in the index
        """
        return len(self._chunks)
