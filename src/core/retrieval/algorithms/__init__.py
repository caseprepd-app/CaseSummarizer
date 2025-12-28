"""
Retrieval Algorithm Registry.

Provides registration and discovery of retrieval algorithms.
Mirrors the vocabulary extraction algorithm registry pattern.

Example:
    from src.core.retrieval.algorithms import get_all_algorithms, get_algorithm

    # Get all registered algorithms
    algorithms = get_all_algorithms()

    # Get a specific algorithm by name
    bm25 = get_algorithm("BM25+")
"""

from typing import Type

from src.core.retrieval.base import BaseRetrievalAlgorithm

# Algorithm registry - maps name to class
_ALGORITHM_REGISTRY: dict[str, Type[BaseRetrievalAlgorithm]] = {}


def register_algorithm(cls: Type[BaseRetrievalAlgorithm]) -> Type[BaseRetrievalAlgorithm]:
    """
    Decorator to register a retrieval algorithm.

    Example:
        @register_algorithm
        class BM25PlusRetriever(BaseRetrievalAlgorithm):
            name = "BM25+"
            ...
    """
    _ALGORITHM_REGISTRY[cls.name] = cls
    return cls


def get_all_algorithms() -> dict[str, Type[BaseRetrievalAlgorithm]]:
    """
    Get all registered retrieval algorithms.

    Returns:
        Dictionary mapping algorithm name to class
    """
    return _ALGORITHM_REGISTRY.copy()


def get_algorithm(name: str) -> Type[BaseRetrievalAlgorithm] | None:
    """
    Get a specific algorithm by name.

    Args:
        name: Algorithm name (e.g., "BM25+", "FAISS")

    Returns:
        Algorithm class or None if not found
    """
    return _ALGORITHM_REGISTRY.get(name)


# Import algorithms to trigger registration
# These imports must be at the bottom to avoid circular imports
from src.core.retrieval.algorithms.bm25_plus import BM25PlusRetriever  # noqa: E402, F401
from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever  # noqa: E402, F401

__all__ = [
    "register_algorithm",
    "get_all_algorithms",
    "get_algorithm",
    "BM25PlusRetriever",
    "FAISSRetriever",
]
