"""
Coreference Resolution Preprocessor (no-op stub).

fastcoref was removed (Mar 2026). This stub remains so the preprocessing
pipeline can instantiate CoreferenceResolver without branching.
"""

from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult


class CoreferenceResolver(BasePreprocessor):
    """No-op stub — fastcoref removed. Returns text unchanged."""

    name = "Coreference Resolver"

    def process(self, text: str) -> PreprocessingResult:
        """Return text unchanged."""
        return PreprocessingResult(text=text, changes_made=0)
