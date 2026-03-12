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

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.core.base_component import BaseNamedComponent


@dataclass
class CandidateTerm:
    """
    A term candidate extracted by an algorithm.

    This represents a single term found by one algorithm. Multiple algorithms
    may find the same term, which will be merged later by AlgorithmScoreMerger.

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


class BaseExtractionAlgorithm(BaseNamedComponent):
    """
    Abstract base class for vocabulary extraction algorithms.

    All algorithms must implement the `extract` method which analyzes text
    and returns candidate terms. The orchestrator will run multiple algorithms
    and merge their results.

    Inherits name, enabled, get_config(), and __repr__ from BaseNamedComponent.

    Class Attributes:
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
    weight: float = 1.0

    @abstractmethod
    def extract(self, text: str, **kwargs) -> AlgorithmResult:
        """
        Extract candidate terms from text.

        Args:
            text: Document text to analyze (may be combined from multiple docs)
            **kwargs: Algorithm-specific parameters.

        Returns:
            AlgorithmResult containing candidate terms and processing metadata
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """Return algorithm configuration for serialization/logging."""
        config = super().get_config()
        config["weight"] = self.weight
        return config

    def _repr_extras(self) -> dict[str, Any]:
        """Include weight in repr."""
        return {"weight": self.weight}
