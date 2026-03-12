"""
Base classes for document retrieval algorithms.

This module defines the abstract base class and data structures that all
retrieval algorithms must implement. The framework supports multiple algorithms
running in parallel, each contributing candidate chunks that are later merged.

Design Principles (mirroring vocabulary extraction):
- Single Responsibility: Each algorithm does one retrieval strategy
- Open/Closed: Add new algorithms without modifying existing code
- Dependency Injection: Configuration passed at construction

Example:
    class BM25PlusRetriever(BaseRetrievalAlgorithm):
        name = "BM25+"
        weight = 1.0

        def retrieve(self, query: str, k: int = 5) -> AlgorithmRetrievalResult:
            # BM25+ specific retrieval logic
            ...
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.core.base_component import BaseNamedComponent


@dataclass
class DocumentChunk:
    """
    A chunk of document text for indexing and retrieval.

    Attributes:
        text: The chunk text content
        chunk_id: Unique identifier for this chunk
        filename: Source document filename
        chunk_num: Sequential chunk number within the document
        section_name: Section or heading this chunk belongs to (if detected)
        word_count: Number of words in this chunk
        metadata: Additional metadata (page number, etc.)
    """

    text: str
    chunk_id: str
    filename: str
    chunk_num: int = 0
    section_name: str = "N/A"
    word_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate word count if not provided."""
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())


@dataclass
class RetrievedChunk:
    """
    A chunk retrieved by a single algorithm.

    Attributes:
        chunk_id: Reference to the original DocumentChunk
        text: The chunk text content
        relevance_score: Algorithm-specific relevance score (0.0-1.0 normalized)
        raw_score: Original algorithm score before normalization
        source_algorithm: Name of the algorithm that retrieved this
        filename: Source document filename
        chunk_num: Chunk number within the document
        section_name: Section name (if available)
        metadata: Algorithm-specific metadata for debugging/ML training
    """

    chunk_id: str
    text: str
    relevance_score: float
    raw_score: float
    source_algorithm: str
    filename: str
    chunk_num: int = 0
    section_name: str = "N/A"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Clamp relevance score to valid range."""
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))


@dataclass
class AlgorithmRetrievalResult:
    """
    Result from running a retrieval algorithm.

    Attributes:
        chunks: List of retrieved chunks, sorted by relevance
        processing_time_ms: Time taken to retrieve (for performance tracking)
        query: The original query string
        metadata: Algorithm-level statistics (index size, etc.)
    """

    chunks: list[RetrievedChunk]
    processing_time_ms: float = 0.0
    query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of chunks retrieved."""
        return len(self.chunks)


class BaseRetrievalAlgorithm(BaseNamedComponent):
    """
    Abstract base class for document retrieval algorithms.

    Inherits name, enabled, get_config(), and __repr__ from BaseNamedComponent.

    Class Attributes:
        weight: Relative weight for scoring when merging (default 1.0)

    Example:
        class BM25PlusRetriever(BaseRetrievalAlgorithm):
            name = "BM25+"
            weight = 1.0

            def index_documents(self, chunks: list[DocumentChunk]) -> None:
                ...

            def retrieve(self, query: str, k: int = 5) -> AlgorithmRetrievalResult:
                ...
    """

    name: str = "BaseAlgorithm"
    weight: float = 1.0

    @abstractmethod
    def index_documents(self, chunks: list[DocumentChunk]) -> None:
        """
        Build search index from document chunks.

        Args:
            chunks: List of DocumentChunk objects to index
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> AlgorithmRetrievalResult:
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query: The search query string
            k: Maximum number of chunks to retrieve

        Returns:
            AlgorithmRetrievalResult with ranked chunks and metadata
        """
        pass

    @property
    @abstractmethod
    def is_indexed(self) -> bool:
        """Check if the index is ready for retrieval."""
        pass

    def get_config(self) -> dict[str, Any]:
        """Return algorithm configuration for serialization/logging."""
        config = super().get_config()
        config["weight"] = self.weight
        return config

    def _repr_extras(self) -> dict[str, Any]:
        """Include weight in repr."""
        return {"weight": self.weight}
