"""
Base Vocabulary Filter Classes

Defines the abstract base class for vocabulary filters and result dataclass.
All vocabulary filters should inherit from BaseVocabularyFilter.

Design mirrors the preprocessing pipeline pattern from src/core/preprocessing/base.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FilterResult:
    """
    Result of a vocabulary filtering operation.

    Attributes:
        vocabulary: The filtered vocabulary list
        removed_count: Number of terms removed
        removed_terms: List of removed term strings (for debugging)
        processing_time_ms: Time taken to process in milliseconds
        metadata: Additional info about the filtering (e.g., per-sub-filter stats)
    """

    vocabulary: list[dict]
    removed_count: int = 0
    removed_terms: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVocabularyFilter(ABC):
    """
    Abstract base class for vocabulary filters.

    All filters must implement the `filter` method which takes a vocabulary
    list and returns a FilterResult with the filtered list and statistics.

    Attributes:
        name: Human-readable name for logging
        enabled: Whether this filter is active
        priority: Execution order (lower = runs first)
        exempt_persons: If True, person names bypass this filter

    Example:
        class MyFilter(BaseVocabularyFilter):
            name = "My Filter"
            priority = 50
            exempt_persons = True

            def filter(self, vocabulary: list[dict]) -> FilterResult:
                filtered = [t for t in vocabulary if self._keep(t)]
                return FilterResult(
                    vocabulary=filtered,
                    removed_count=len(vocabulary) - len(filtered)
                )
    """

    name: str = "Base Filter"
    enabled: bool = True
    priority: int = 50
    exempt_persons: bool = False

    @abstractmethod
    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """
        Filter the vocabulary list and return result with statistics.

        Args:
            vocabulary: List of term dictionaries

        Returns:
            FilterResult containing filtered vocabulary and statistics
        """
        pass

    def should_filter_term(self, term_data: dict) -> bool:
        """
        Check if a single term should be filtered.

        Override for per-term filters. List-level filters should override
        filter() directly.

        Args:
            term_data: Single term dictionary

        Returns:
            True if term should be REMOVED, False to keep
        """
        return False

    def get_config(self) -> dict[str, Any]:
        """Return filter configuration for logging."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "priority": self.priority,
            "exempt_persons": self.exempt_persons,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(enabled={self.enabled}, priority={self.priority})"
