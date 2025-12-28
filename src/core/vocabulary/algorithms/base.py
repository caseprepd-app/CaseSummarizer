"""
Base classes for vocabulary extraction algorithms.

This module defines the abstract base class and data structures that all
extraction algorithms must implement. The framework supports multiple algorithms
running in parallel, each contributing candidate terms that are later merged.

Design Principles:
- Single Responsibility: Each algorithm does one extraction strategy
- Open/Closed: Add new algorithms without modifying existing code
- Dependency Injection: Configuration passed at construction

Example:
    @register_algorithm("RAKE")
    class RAKEAlgorithm(BaseExtractionAlgorithm):
        name = "RAKE"

        def extract(self, text: str, **kwargs) -> AlgorithmResult:
            # RAKE-specific extraction logic
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateTerm:
    """
    A term candidate extracted by an algorithm.

    This represents a single term found by one algorithm. Multiple algorithms
    may find the same term, which will be merged later by ResultMerger.

    Attributes:
        term: The extracted text (e.g., "John Smith", "cardiomyopathy")
        source_algorithm: Name of the algorithm that found this term
        confidence: Algorithm-specific confidence score (0.0-1.0)
        metadata: Algorithm-specific metadata for ML training/debugging
        suggested_type: Optional type hint (Person, Medical, Technical, etc.)
        frequency: Number of occurrences found in the text
    """
    term: str
    source_algorithm: str
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    suggested_type: str | None = None
    frequency: int = 1

    def __post_init__(self):
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class AlgorithmResult:
    """
    Result from running an extraction algorithm.

    Contains all candidate terms found by a single algorithm run,
    plus metadata about the processing.

    Attributes:
        candidates: List of candidate terms found
        processing_time_ms: Time taken to process (for performance tracking)
        metadata: Algorithm-level statistics (total entities processed, etc.)
    """
    candidates: list[CandidateTerm]
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of candidates found."""
        return len(self.candidates)


class BaseExtractionAlgorithm(ABC):
    """
    Abstract base class for vocabulary extraction algorithms.

    All algorithms must implement the `extract` method which analyzes text
    and returns candidate terms. The orchestrator will run multiple algorithms
    and merge their results.

    Class Attributes:
        name: Human-readable algorithm name (for logging and ML tracking)
        enabled: Whether this algorithm is active (can be toggled at runtime)
        weight: Relative weight for scoring when merging (default 1.0)

    Example:
        @register_algorithm("NER")
        class NERAlgorithm(BaseExtractionAlgorithm):
            name = "NER"

            def extract(self, text: str, **kwargs) -> AlgorithmResult:
                # NER-specific extraction using spaCy
                ...
    """

    name: str = "BaseAlgorithm"
    enabled: bool = True
    weight: float = 1.0

    @abstractmethod
    def extract(self, text: str, **kwargs) -> AlgorithmResult:
        """
        Extract candidate terms from text.

        This is the core method that each algorithm must implement. It should
        analyze the input text and return all candidate terms found.

        Args:
            text: Document text to analyze (may be combined from multiple docs)
            **kwargs: Algorithm-specific parameters. Common kwargs include:
                - nlp: Pre-loaded spaCy model (for NER algorithm)
                - doc_count: Number of source documents (for threshold adjustment)

        Returns:
            AlgorithmResult containing candidate terms and processing metadata

        Note:
            Implementations should be thread-safe if parallel execution is needed.
            Avoid storing state that could cause race conditions.
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Return algorithm configuration for serialization/logging.

        Override in subclass to include algorithm-specific settings
        (e.g., RAKE phrase length, TF-IDF max features).

        Returns:
            Dictionary of configuration values
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "weight": self.weight,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled}, weight={self.weight})"
