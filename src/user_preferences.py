"""
User Preferences Manager for LocalScribe
Manages user-specific preferences like default prompts per model.
"""

import json
from pathlib import Path
from typing import Any


class UserPreferencesManager:
    """
    Manages user preferences stored in config/user_preferences.json.

    Handles default prompt selection per model with graceful fallbacks.
    """

    def __init__(self, preferences_file: Path):
        """
        Initialize the preferences manager.

        Args:
            preferences_file: Path to user_preferences.json
        """
        self.preferences_file = Path(preferences_file)
        self._preferences = self._load_preferences()

    def _load_preferences(self) -> dict[str, Any]:
        """
        Load preferences from JSON file.

        Returns:
            dict: User preferences, or default structure if file not found
        """
        default_structure = {
            "model_defaults": {},
            "last_used_model": None,
            "processing": {"cpu_fraction": 0.5},  # Default: 1/2 cores (0.25, 0.5, or 0.75)
            # Session 43: Experimental features and LLM extraction settings
            # Session 62b: vocab_use_llm changed to tri-state: "auto", "yes", "no"
            "experimental": {
                "briefing_enabled": False,  # Case Briefing (experimental)
                "vocab_use_llm": "auto",  # LLM extraction: "auto", "yes", or "no"
            },
        }

        try:
            if self.preferences_file.exists():
                with open(self.preferences_file, encoding="utf-8") as f:
                    prefs = json.load(f)
                    # Ensure structure exists
                    if "model_defaults" not in prefs:
                        prefs["model_defaults"] = {}
                    return prefs
            else:
                return default_structure

        except (json.JSONDecodeError, Exception):
            # If file is corrupted, return defaults
            return default_structure

    def _save_preferences(self) -> None:
        """Save preferences to JSON file."""
        try:
            # Ensure directory exists
            self.preferences_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.preferences_file, "w", encoding="utf-8") as f:
                json.dump(self._preferences, f, indent=2)

        except Exception as e:
            # Log error but don't crash
            from src.logging_config import debug_log

            debug_log(f"[PREFS] Could not save user preferences: {e}")

    def get_default_prompt(self, model_name: str) -> str | None:
        """
        Get the user's preferred default prompt for a model.

        Args:
            model_name: Name of the model (e.g., 'phi-3-mini')

        Returns:
            Preset ID string, or None if no default set
        """
        return self._preferences.get("model_defaults", {}).get(model_name)

    def set_default_prompt(self, model_name: str, preset_id: str) -> None:
        """
        Set the default prompt for a model.

        Args:
            model_name: Name of the model (e.g., 'phi-3-mini')
            preset_id: Preset identifier (e.g., 'factual-summary')
        """
        if "model_defaults" not in self._preferences:
            self._preferences["model_defaults"] = {}

        self._preferences["model_defaults"][model_name] = preset_id
        self._save_preferences()

    def get_last_used_model(self) -> str | None:
        """Get the last model the user loaded."""
        return self._preferences.get("last_used_model")

    def set_last_used_model(self, model_name: str) -> None:
        """
        Set the last used model.

        Args:
            model_name: Name of the model
        """
        self._preferences["last_used_model"] = model_name
        self._save_preferences()

    def clear_default_prompt(self, model_name: str) -> None:
        """
        Clear the default prompt for a model.

        Args:
            model_name: Name of the model
        """
        if "model_defaults" in self._preferences:
            self._preferences["model_defaults"].pop(model_name, None)
            self._save_preferences()

    def get_cpu_fraction(self) -> float:
        """
        Get the CPU fraction for parallel document processing.

        Returns:
            float: CPU fraction (0.25, 0.5, or 0.75). Defaults to 0.5
        """
        return self._preferences.get("processing", {}).get("cpu_fraction", 0.5)

    def set_cpu_fraction(self, cpu_fraction: float) -> None:
        """
        Set the CPU fraction for parallel document processing.

        Args:
            cpu_fraction: CPU fraction (0.25, 0.5, or 0.75)

        Raises:
            ValueError: If cpu_fraction is not 0.25, 0.5, or 0.75
        """
        valid_fractions = [0.25, 0.5, 0.75]
        if cpu_fraction not in valid_fractions:
            raise ValueError(f"CPU fraction must be one of {valid_fractions}, got {cpu_fraction}")

        if "processing" not in self._preferences:
            self._preferences["processing"] = {}

        self._preferences["processing"]["cpu_fraction"] = cpu_fraction
        self._save_preferences()

    # =========================================================================
    # Experimental Features (Session 43)
    # =========================================================================

    def is_experimental_briefing_enabled(self) -> bool:
        """
        Check if Case Briefing (experimental) is enabled.

        Returns:
            bool: True if Case Briefing should be shown in UI
        """
        return self._preferences.get("experimental", {}).get("briefing_enabled", False)

    def set_experimental_briefing_enabled(self, enabled: bool) -> None:
        """
        Enable or disable Case Briefing (experimental).

        Args:
            enabled: Whether to show Case Briefing in UI
        """
        if "experimental" not in self._preferences:
            self._preferences["experimental"] = {}
        self._preferences["experimental"]["briefing_enabled"] = enabled
        self._save_preferences()

    def get_vocab_llm_mode(self) -> str:
        """
        Get LLM extraction mode (Session 62b).

        Returns:
            str: "auto", "yes", or "no"
        """
        value = self._preferences.get("experimental", {}).get("vocab_use_llm", "auto")
        # Handle legacy boolean values from older preferences
        if value is True:
            return "yes"
        elif value is False:
            return "no"
        return value if value in ("auto", "yes", "no") else "auto"

    def is_vocab_llm_enabled(self) -> bool:
        """
        Check if LLM extraction is enabled for vocabulary (Session 62b).

        Resolves "auto" mode using GPU detection.

        Returns:
            bool: True if LLM should be used alongside NER
        """
        mode = self.get_vocab_llm_mode()
        if mode == "yes":
            return True
        elif mode == "no":
            return False
        else:  # "auto" - use GPU detection
            from src.core.utils.gpu_detector import has_dedicated_gpu

            return has_dedicated_gpu()

    def set_vocab_llm_mode(self, mode: str) -> None:
        """
        Set LLM extraction mode (Session 62b).

        Args:
            mode: "auto", "yes", or "no"
        """
        if mode not in ("auto", "yes", "no"):
            raise ValueError(f"Invalid mode: {mode}, must be 'auto', 'yes', or 'no'")
        if "experimental" not in self._preferences:
            self._preferences["experimental"] = {}
        self._preferences["experimental"]["vocab_use_llm"] = mode
        self._save_preferences()

    def set_vocab_llm_enabled(self, enabled: bool) -> None:
        """
        Legacy method - Enable or disable LLM for vocabulary extraction.

        Deprecated: Use set_vocab_llm_mode() instead.

        Args:
            enabled: Whether to use LLM alongside NER
        """
        # Convert boolean to string mode for backwards compatibility
        self.set_vocab_llm_mode("yes" if enabled else "no")

    # =========================================================================
    # LLM Context Window Settings (Session 64)
    # =========================================================================

    def get_context_size_mode(self) -> str | int:
        """
        Get context window size mode (Session 64).

        Returns:
            "auto" for automatic detection based on VRAM, or
            int for manual override (4000, 8000, 16000, 32000, 48000, 64000)
        """
        value = self._preferences.get("llm_context_size", "auto")
        if value == "auto":
            return "auto"
        # Ensure it's a valid int
        try:
            return int(value)
        except (ValueError, TypeError):
            return "auto"

    def set_context_size_mode(self, value: str | int) -> None:
        """
        Set context window size mode (Session 64).

        Args:
            value: "auto" or int (4000, 8000, 16000, 32000, 48000, 64000)
        """
        valid_sizes = [4000, 8000, 16000, 32000, 48000, 64000]
        if value != "auto" and value not in valid_sizes:
            raise ValueError(f"Context size must be 'auto' or one of {valid_sizes}, got {value}")
        self._preferences["llm_context_size"] = value
        self._save_preferences()

    def get_effective_context_size(self) -> int:
        """
        Get the actual context window size to use (Session 64).

        Resolves "auto" mode using GPU VRAM detection.

        Returns:
            int: Context window size (num_ctx) for Ollama API calls
        """
        mode = self.get_context_size_mode()
        if isinstance(mode, int):
            return mode
        # "auto" mode - use GPU detection
        from src.core.utils.gpu_detector import get_optimal_context_size

        return get_optimal_context_size()

    def get_effective_chunk_sizes(self) -> dict:
        """
        Get chunk sizes scaled to effective context window (Session 67).

        Chunk sizes scale proportionally with context window to utilize
        available VRAM more efficiently on high-memory GPUs.

        Returns:
            Dict with:
            - min_tokens: Minimum tokens per chunk
            - target_tokens: Target tokens per chunk
            - max_tokens: Maximum tokens per chunk
            - context_window: The context window used for calculation
        """
        from src.core.utils.gpu_detector import get_optimal_chunk_sizes

        context_size = self.get_effective_context_size()
        return get_optimal_chunk_sizes(context_size)

    # =========================================================================
    # Generic Get/Set Methods (for extensible settings system)
    # =========================================================================

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get any preference value by key.

        This generic method supports the extensible settings system,
        allowing new settings to be added without modifying this class.

        Args:
            key: The preference key to retrieve.
            default: Value to return if key doesn't exist.

        Returns:
            The stored value, or default if not found.
        """
        return self._preferences.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set any preference value by key with validation.

        This generic method supports the extensible settings system,
        allowing new settings to be added without modifying this class.

        Args:
            key: The preference key to set.
            value: The value to store.

        Raises:
            ValueError: If value fails validation for the given key.
        """
        # Validate known keys to prevent invalid configurations
        if key == "vocab_display_limit":
            if not isinstance(value, int) or value < 1 or value > 500:
                raise ValueError(f"vocab_display_limit must be 1-500, got {value}")
        elif key == "user_defined_max_workers":
            if not isinstance(value, int) or value < 1 or value > 8:
                raise ValueError(f"user_defined_max_workers must be 1-8, got {value}")
        elif key == "default_model_id":
            if not value or not isinstance(value, str):
                raise ValueError("default_model_id cannot be empty")
        elif key == "summary_words":
            if not isinstance(value, int) or value < 50 or value > 2000:
                raise ValueError(f"summary_words must be 50-2000, got {value}")
        elif key == "resource_usage_pct":
            if not isinstance(value, int) or value < 25 or value > 100:
                raise ValueError(f"resource_usage_pct must be 25-100, got {value}")
        # Session 59: Vocabulary filtering threshold validation
        elif key == "single_word_rarity_threshold":
            if not isinstance(value, (int, float)) or value < 0.1 or value > 0.9:
                raise ValueError(f"single_word_rarity_threshold must be 0.1-0.9, got {value}")
        elif key == "phrase_rarity_threshold":
            if not isinstance(value, (int, float)) or value < 0.1 or value > 0.9:
                raise ValueError(f"phrase_rarity_threshold must be 0.1-0.9, got {value}")
        # Session 62: New settings validation
        elif key == "ollama_model":
            if not isinstance(value, str) or not value:
                raise ValueError("ollama_model must be a non-empty string")
        elif key == "vocab_min_occurrences":
            if not isinstance(value, int) or value < 1 or value > 5:
                raise ValueError(f"vocab_min_occurrences must be 1-5, got {value}")
        elif key == "phrase_mean_rarity_threshold":
            if not isinstance(value, (int, float)) or value < 0.1 or value > 0.9:
                raise ValueError(f"phrase_mean_rarity_threshold must be 0.1-0.9, got {value}")
        # Session 68: Corpus familiarity threshold validation
        elif key == "corpus_familiarity_threshold":
            if not isinstance(value, (int, float)) or value < 0.25 or value > 1.0:
                raise ValueError(f"corpus_familiarity_threshold must be 0.25-1.0, got {value}")
        elif key == "corpus_familiarity_min_docs":
            if not isinstance(value, int) or value < 0 or value > 50:
                raise ValueError(f"corpus_familiarity_min_docs must be 0-50, got {value}")
        elif key == "corpus_familiarity_exempt_persons":
            if not isinstance(value, bool):
                raise ValueError(f"corpus_familiarity_exempt_persons must be boolean, got {value}")
        # Session 68: Corpus ready transition flag
        elif key == "corpus_was_ever_ready":
            if not isinstance(value, bool):
                raise ValueError(f"corpus_was_ever_ready must be boolean, got {value}")
        # Session 62b: LLM extraction mode validation
        elif key == "vocab_use_llm":
            # Accept both legacy boolean and new tri-state string
            if value not in ("auto", "yes", "no") and not isinstance(value, bool):
                raise ValueError(f"vocab_use_llm must be 'auto', 'yes', or 'no', got {value}")
        # Session 64: LLM context window size validation
        elif key == "llm_context_size":
            valid_sizes = [4000, 8000, 16000, 32000, 48000, 64000]
            if value != "auto" and value not in valid_sizes:
                raise ValueError(
                    f"llm_context_size must be 'auto' or one of {valid_sizes}, got {value}"
                )
        # Session 80: Column visibility validation
        elif key == "vocab_column_visibility":
            if not isinstance(value, dict):
                raise ValueError(
                    f"vocab_column_visibility must be a dict, got {type(value).__name__}"
                )
            # Validate that all keys are valid column names
            valid_columns = {
                "Term",
                "Score",
                "Is Person",
                "Found By",
                "# Docs",
                "Count",
                "Median Conf",
                "NER",
                "RAKE",
                "BM25",
                "Algo Count",
                "Freq Rank",
                "Keep",
                "Skip",
            }
            invalid = set(value.keys()) - valid_columns
            if invalid:
                raise ValueError(f"Invalid column names: {invalid}")
            # Validate that all values are boolean
            for col, visible in value.items():
                if not isinstance(visible, bool):
                    raise ValueError(
                        f"Column visibility value must be boolean, got {type(visible).__name__} for '{col}'"
                    )
            # Term column cannot be hidden
            if value.get("Term") is False:
                raise ValueError("'Term' column cannot be hidden")
        # Session 80: Column widths validation
        elif key == "vocab_column_widths":
            if not isinstance(value, dict):
                raise ValueError(f"vocab_column_widths must be a dict, got {type(value).__name__}")
            # Validate that all keys are valid column names
            valid_columns = {
                "Term",
                "Score",
                "Is Person",
                "Found By",
                "# Docs",
                "Count",
                "Median Conf",
                "NER",
                "RAKE",
                "BM25",
                "Algo Count",
                "Freq Rank",
                "Keep",
                "Skip",
            }
            invalid = set(value.keys()) - valid_columns
            if invalid:
                raise ValueError(f"Invalid column names: {invalid}")
            # Validate that all values are positive integers
            for col, width in value.items():
                if not isinstance(width, int) or width < 30 or width > 500:
                    raise ValueError(f"Column width must be int 30-500, got {width} for '{col}'")

        self._preferences[key] = value
        self._save_preferences()


# Global instance
_user_prefs = None


def get_user_preferences(preferences_file: Path | None = None) -> UserPreferencesManager:
    """
    Get the global UserPreferencesManager instance (singleton pattern).

    Args:
        preferences_file: Optional path to preferences file (only used on first call)

    Returns:
        UserPreferencesManager: The global preferences instance
    """
    global _user_prefs

    if _user_prefs is None:
        if preferences_file is None:
            from .config import CONFIG_DIR

            preferences_file = CONFIG_DIR / "user_preferences.json"

        _user_prefs = UserPreferencesManager(preferences_file)

    return _user_prefs
