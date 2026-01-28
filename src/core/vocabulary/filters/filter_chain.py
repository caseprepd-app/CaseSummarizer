"""
Vocabulary Filter Chain

Orchestrates multiple vocabulary filters in priority order with statistics tracking.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult

logger = logging.getLogger(__name__)


@dataclass
class FilterChainStats:
    """Statistics from a complete filter chain run."""

    total_input: int = 0
    total_output: int = 0
    total_removed: int = 0
    total_time_ms: float = 0.0
    per_filter_stats: dict[str, dict[str, Any]] = field(default_factory=dict)


class VocabularyFilterChain:
    """
    Orchestrates vocabulary filters in priority order.

    Filters are sorted by priority (lower = runs first) and executed sequentially.
    The chain tracks cumulative statistics and supports enabling/disabling filters.

    Example:
        chain = VocabularyFilterChain([
            NameDeduplicationFilter(),
            ExtractionArtifactFilter(),
            RarityFilter(),
        ])
        result = chain.run(vocabulary)
        print(f"Removed {result.removed_count} terms")
    """

    def __init__(self, filters: list[BaseVocabularyFilter] | None = None):
        """
        Initialize the filter chain.

        Args:
            filters: List of filter instances. If None, empty chain is created.
        """
        self._filters: list[BaseVocabularyFilter] = filters or []
        self._last_stats: FilterChainStats | None = None

    def add_filter(self, filter_: BaseVocabularyFilter) -> "VocabularyFilterChain":
        """Add a filter to the chain. Returns self for chaining."""
        self._filters.append(filter_)
        return self

    def remove_filter(self, name: str) -> bool:
        """Remove a filter by name. Returns True if found and removed."""
        for i, f in enumerate(self._filters):
            if f.name == name:
                self._filters.pop(i)
                return True
        return False

    def set_filter_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a filter by name. Returns True if found."""
        for f in self._filters:
            if f.name == name:
                f.enabled = enabled
                return True
        return False

    def get_filter(self, name: str) -> BaseVocabularyFilter | None:
        """Get a filter by name."""
        for f in self._filters:
            if f.name == name:
                return f
        return None

    def run(self, vocabulary: list[dict]) -> FilterResult:
        """
        Run all enabled filters in priority order.

        Args:
            vocabulary: Input vocabulary list

        Returns:
            FilterResult with final vocabulary and cumulative stats
        """
        if not vocabulary:
            return FilterResult(vocabulary=[], removed_count=0)

        # Sort filters by priority (lower = first)
        sorted_filters = sorted([f for f in self._filters if f.enabled], key=lambda f: f.priority)

        stats = FilterChainStats(total_input=len(vocabulary))
        current_vocab = vocabulary
        chain_start = time.time()

        logger.debug(
            "Starting with %s terms, %s enabled filters",
            len(current_vocab),
            len(sorted_filters),
        )

        for filter_ in sorted_filters:
            input_count = len(current_vocab)
            filter_start = time.time()

            try:
                result = filter_.filter(current_vocab)
                elapsed_ms = (time.time() - filter_start) * 1000

                # Session 80: Yield GIL between filters to keep GUI responsive
                time.sleep(0)

                stats.per_filter_stats[filter_.name] = {
                    "input_count": input_count,
                    "output_count": len(result.vocabulary),
                    "removed_count": result.removed_count,
                    "time_ms": elapsed_ms,
                    "metadata": result.metadata,
                }

                if result.removed_count > 0:
                    logger.debug(
                        "%s: %s removed in %.1fms",
                        filter_.name,
                        result.removed_count,
                        elapsed_ms,
                    )

                current_vocab = result.vocabulary

            except Exception as e:
                elapsed_ms = (time.time() - filter_start) * 1000
                logger.debug("Error in %s: %s", filter_.name, e)
                stats.per_filter_stats[filter_.name] = {
                    "input_count": input_count,
                    "output_count": input_count,
                    "removed_count": 0,
                    "error": str(e),
                    "time_ms": elapsed_ms,
                }
                # Continue with unchanged vocabulary on error

        stats.total_output = len(current_vocab)
        stats.total_removed = stats.total_input - stats.total_output
        stats.total_time_ms = (time.time() - chain_start) * 1000

        self._last_stats = stats

        logger.debug(
            "Complete: %s removed in %.1fms, %s remaining",
            stats.total_removed,
            stats.total_time_ms,
            stats.total_output,
        )

        return FilterResult(
            vocabulary=current_vocab,
            removed_count=stats.total_removed,
            processing_time_ms=stats.total_time_ms,
            metadata={"per_filter": stats.per_filter_stats},
        )

    def get_last_stats(self) -> FilterChainStats | None:
        """Get statistics from the last run."""
        return self._last_stats

    @property
    def filters(self) -> list[BaseVocabularyFilter]:
        """Get list of all filters."""
        return self._filters.copy()

    def __repr__(self) -> str:
        names = [f.name for f in self._filters]
        return f"VocabularyFilterChain({names})"
