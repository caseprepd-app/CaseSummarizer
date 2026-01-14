"""
Prompt Parameters Configuration Loader
Loads and manages user-configurable AI prompt settings.

Moved from src/prompt_config.py to src/prompting/config.py in Session 33.

Note: Default values are imported from src/config.py as the single source of truth.
The JSON file allows user customization that overrides these defaults.
"""

import json
from pathlib import Path
from typing import Any, ClassVar

from src.config import (
    DEFAULT_SUMMARY_WORDS,
    LLM_TOKEN_BUFFER_MULTIPLIER,
    LLM_TOKENS_PER_WORD,
    LLM_TOP_P,
    MAX_SUMMARY_WORDS,
    MIN_SUMMARY_WORDS,
    SUMMARY_SLIDER_INCREMENT,
    SUMMARY_TEMPERATURE,
    SUMMARY_WORD_COUNT_TOLERANCE,
)

# Path to the prompt parameters file
PROMPT_PARAMS_FILE = (
    Path(__file__).parent.parent.parent.parent / "config" / "prompt_parameters.json"
)


class PromptConfig:
    """
    Loads and provides access to prompt parameters.

    This class reads the prompt_parameters.json file and provides
    easy access to settings with fallback defaults from src/config.py.
    """

    # Default values sourced from src/config.py (single source of truth)
    @classmethod
    def _get_defaults(cls) -> dict[str, dict[str, float | int]]:
        """Get defaults from central config."""
        return {
            "summary": {
                "word_count_tolerance": SUMMARY_WORD_COUNT_TOLERANCE,
                "slider_increment": SUMMARY_SLIDER_INCREMENT,
                "min_words": MIN_SUMMARY_WORDS,
                "max_words": MAX_SUMMARY_WORDS,
                "default_words": DEFAULT_SUMMARY_WORDS,
                "temperature": SUMMARY_TEMPERATURE,
            },
            "generation": {
                "top_p": LLM_TOP_P,
                "tokens_per_word_estimate": LLM_TOKENS_PER_WORD,
                "token_buffer_multiplier": LLM_TOKEN_BUFFER_MULTIPLIER,
            },
        }

    # Legacy class variable for backward compatibility
    DEFAULTS: ClassVar[dict[str, dict[str, float | int]]] = {
        "summary": {
            "word_count_tolerance": 20,
            "slider_increment": 50,
            "min_words": 100,
            "max_words": 500,
            "default_words": 200,
            "temperature": 0.3,
        },
        "generation": {
            "top_p": 0.9,
            "tokens_per_word_estimate": 1.5,
            "token_buffer_multiplier": 1.3,
        },
    }

    def __init__(self):
        """Initialize and load prompt parameters."""
        self._params = self._load_params()

    def _load_params(self) -> dict[str, Any]:
        """
        Load parameters from JSON file.

        Returns:
            dict: Loaded parameters, or defaults from src/config.py if file not found
        """
        defaults = self._get_defaults()
        try:
            if PROMPT_PARAMS_FILE.exists():
                with open(PROMPT_PARAMS_FILE, encoding="utf-8") as f:
                    params = json.load(f)

                    # Filter out comment keys (starting with _)
                    return self._filter_comments(params)
            else:
                from src.logging_config import debug_log

                debug_log(
                    f"[PROMPT CONFIG] Prompt parameters file not found at {PROMPT_PARAMS_FILE}"
                )
                debug_log("[PROMPT CONFIG] Using default values from config.py.")
                return defaults

        except json.JSONDecodeError as e:
            from src.logging_config import debug_log

            debug_log(f"[PROMPT CONFIG] Error parsing prompt parameters file: {e}")
            debug_log("[PROMPT CONFIG] Using default values from config.py.")
            return defaults
        except Exception as e:
            from src.logging_config import debug_log

            debug_log(f"[PROMPT CONFIG] Error loading prompt parameters: {e}")
            debug_log("[PROMPT CONFIG] Using default values from config.py.")
            return defaults

    def _filter_comments(self, data: Any) -> Any:
        """
        Recursively remove keys starting with '_' (comments).

        Args:
            data: Dictionary or other data structure

        Returns:
            Filtered data structure
        """
        if isinstance(data, dict):
            return {
                key: self._filter_comments(value)
                for key, value in data.items()
                if not key.startswith("_")
            }
        elif isinstance(data, list):
            return [self._filter_comments(item) for item in data]
        else:
            return data

    def get(self, *keys, default=None) -> Any:
        """
        Get a parameter value by nested keys.

        Args:
            *keys: Nested keys to traverse (e.g., 'summary', 'word_count_tolerance')
            default: Default value if key not found

        Returns:
            Parameter value or default

        Example:
            config.get('summary', 'word_count_tolerance')  # Returns 20
        """
        value = self._params
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    # Convenience properties for commonly used values

    @property
    def word_count_tolerance(self) -> int:
        """Get word count tolerance for summaries."""
        return self.get("summary", "word_count_tolerance", default=20)

    @property
    def slider_increment(self) -> int:
        """Get slider increment value."""
        return self.get("summary", "slider_increment", default=50)

    @property
    def min_summary_words(self) -> int:
        """Get minimum summary word count."""
        return self.get("summary", "min_words", default=100)

    @property
    def max_summary_words(self) -> int:
        """Get maximum summary word count."""
        return self.get("summary", "max_words", default=500)

    @property
    def default_summary_words(self) -> int:
        """Get default summary word count."""
        return self.get("summary", "default_words", default=200)

    @property
    def summary_temperature(self) -> float:
        """Get temperature for summary generation."""
        return self.get("summary", "temperature", default=0.3)

    @property
    def top_p(self) -> float:
        """Get top_p parameter for generation."""
        return self.get("generation", "top_p", default=0.9)

    @property
    def tokens_per_word(self) -> float:
        """Get tokens per word estimate."""
        return self.get("generation", "tokens_per_word_estimate", default=1.5)

    @property
    def token_buffer_multiplier(self) -> float:
        """Get token buffer multiplier to prevent mid-sentence cutoffs."""
        return self.get("generation", "token_buffer_multiplier", default=1.3)

    def get_word_count_range(self, target_words: int) -> tuple:
        """
        Calculate the acceptable word count range for a target.

        Args:
            target_words: Target word count

        Returns:
            tuple: (min_words, max_words) range

        Example:
            get_word_count_range(200) -> (180, 220) with tolerance=20
        """
        tolerance = self.word_count_tolerance
        return (target_words - tolerance, target_words + tolerance)


# Global instance for easy access
_prompt_config = None


def get_prompt_config() -> PromptConfig:
    """
    Get the global PromptConfig instance (singleton pattern).

    Returns:
        PromptConfig: The global configuration instance
    """
    global _prompt_config
    if _prompt_config is None:
        _prompt_config = PromptConfig()
    return _prompt_config
