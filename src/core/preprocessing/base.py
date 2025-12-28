"""
Base Preprocessor Classes

Defines the abstract base class for preprocessors and the pipeline orchestrator.
All custom preprocessors should inherit from BasePreprocessor.

Design Principles:
- Single Responsibility: Each preprocessor does one thing well
- Open/Closed: Add new preprocessors without modifying existing code
- Testable: Each preprocessor can be unit tested in isolation
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.logging_config import debug_log


@dataclass
class PreprocessingResult:
    """
    Result of a preprocessing operation.

    Attributes:
        text: The processed text
        changes_made: Number of changes/substitutions made
        metadata: Additional info about the processing (e.g., removed lines count)
        processing_time_ms: Time taken to process in milliseconds
    """
    text: str
    changes_made: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0


class BasePreprocessor(ABC):
    """
    Abstract base class for text preprocessors.

    All preprocessors must implement the `process` method which takes
    text input and returns a PreprocessingResult.

    Attributes:
        name: Human-readable name for logging
        enabled: Whether this preprocessor is active

    Example:
        class MyPreprocessor(BasePreprocessor):
            name = "My Preprocessor"

            def process(self, text: str) -> PreprocessingResult:
                cleaned = text.replace("foo", "bar")
                return PreprocessingResult(
                    text=cleaned,
                    changes_made=text.count("foo")
                )
    """

    name: str = "Base Preprocessor"
    enabled: bool = True

    @abstractmethod
    def process(self, text: str) -> PreprocessingResult:
        """
        Process the input text and return cleaned version.

        Args:
            text: Raw input text to process

        Returns:
            PreprocessingResult containing cleaned text and metadata
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(enabled={self.enabled})"


class PreprocessingPipeline:
    """
    Orchestrates multiple preprocessors in sequence.

    Preprocessors are executed in order, with each receiving the output
    of the previous. Disabled preprocessors are skipped.

    Attributes:
        preprocessors: List of preprocessor instances to execute
        total_changes: Cumulative changes made across all preprocessors

    Example:
        pipeline = PreprocessingPipeline([
            LineNumberRemover(),
            HeaderFooterRemover(),
        ])
        result = pipeline.process(raw_text)
        print(f"Made {pipeline.total_changes} total changes")
    """

    def __init__(self, preprocessors: list[BasePreprocessor] | None = None):
        """
        Initialize the pipeline with a list of preprocessors.

        Args:
            preprocessors: List of BasePreprocessor instances.
                          If None, an empty pipeline is created.
        """
        self.preprocessors: list[BasePreprocessor] = preprocessors or []
        self.total_changes: int = 0
        self._last_run_stats: dict[str, dict[str, Any]] = {}

    def add_preprocessor(self, preprocessor: BasePreprocessor) -> 'PreprocessingPipeline':
        """
        Add a preprocessor to the pipeline.

        Args:
            preprocessor: Preprocessor instance to add

        Returns:
            Self for method chaining
        """
        self.preprocessors.append(preprocessor)
        return self

    def remove_preprocessor(self, name: str) -> bool:
        """
        Remove a preprocessor by name.

        Args:
            name: Name of the preprocessor to remove

        Returns:
            True if removed, False if not found
        """
        for i, p in enumerate(self.preprocessors):
            if p.name == name:
                self.preprocessors.pop(i)
                return True
        return False

    def process(self, text: str) -> str:
        """
        Run all enabled preprocessors on the input text.

        Args:
            text: Raw input text to process

        Returns:
            Cleaned text after all preprocessing steps
        """
        if not text:
            return text

        self.total_changes = 0
        self._last_run_stats = {}
        current_text = text
        pipeline_start = time.time()

        enabled_count = sum(1 for p in self.preprocessors if p.enabled)
        debug_log(f"[PREPROCESSING] Starting pipeline with {enabled_count} "
                  f"enabled preprocessors on {len(text)//1024}KB text")

        for preprocessor in self.preprocessors:
            if not preprocessor.enabled:
                debug_log(f"[PREPROCESSING] Skipping disabled: {preprocessor.name}")
                continue

            start_time = time.time()
            try:
                result = preprocessor.process(current_text)
                elapsed_ms = (time.time() - start_time) * 1000

                self.total_changes += result.changes_made
                self._last_run_stats[preprocessor.name] = {
                    'changes': result.changes_made,
                    'time_ms': elapsed_ms,
                    'metadata': result.metadata,
                }

                debug_log(f"[PREPROCESSING] {preprocessor.name}: "
                          f"{result.changes_made} changes in {elapsed_ms:.1f}ms")

                current_text = result.text

            except Exception as e:
                debug_log(f"[PREPROCESSING] Error in {preprocessor.name}: {e}")
                # Continue with unchanged text on error
                self._last_run_stats[preprocessor.name] = {
                    'error': str(e),
                    'changes': 0,
                    'time_ms': (time.time() - start_time) * 1000,
                }

        total_time = (time.time() - pipeline_start) * 1000
        debug_log(f"[PREPROCESSING] Pipeline complete: {self.total_changes} total changes "
                  f"in {total_time:.1f}ms, output {len(current_text)//1024}KB")

        return current_text

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics from the last pipeline run.

        Returns:
            Dictionary mapping preprocessor names to their stats
        """
        return self._last_run_stats.copy()

    def __repr__(self) -> str:
        names = [p.name for p in self.preprocessors]
        return f"PreprocessingPipeline({names})"
