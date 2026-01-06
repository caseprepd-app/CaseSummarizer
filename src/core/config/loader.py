"""
Unified configuration loading utilities.

Replaces 7+ duplicate _load_config() implementations across the codebase.
Provides consistent YAML/JSON loading with proper error handling and logging.
"""

from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import yaml

T = TypeVar("T")


def load_yaml(
    config_path: Union[str, Path],
    default: Optional[T] = None,
    raise_on_error: bool = True,
    log_prefix: str = "[Config]",
) -> Union[dict, T]:
    """
    Load a YAML configuration file with consistent error handling.

    Replaces duplicate _load_config() methods in:
    - chunking_engine.py
    - progressive_summarizer.py
    - vector_store/question_flow.py
    - qa/qa_orchestrator.py
    - ui/qa_question_editor.py (2 locations)
    - config.py (load_model_configs)

    Args:
        config_path: Path to the YAML file
        default: Value to return if file not found (only used if raise_on_error=False)
        raise_on_error: If True, raises exceptions. If False, returns default.
        log_prefix: Prefix for debug log messages

    Returns:
        Parsed YAML content as dict, or default if file not found

    Raises:
        FileNotFoundError: If file not found and raise_on_error=True
        yaml.YAMLError: If YAML parsing fails and raise_on_error=True
    """
    config_path = Path(config_path)

    # Import here to avoid circular imports during module initialization
    try:
        from src.logging_config import debug_log, error
    except ImportError:
        # Fallback if logging not available yet
        debug_log = lambda msg: None
        error = lambda msg: None

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        debug_log(f"{log_prefix} Loaded config from {config_path}")
        return config if config is not None else {}

    except FileNotFoundError:
        if raise_on_error:
            error(f"{log_prefix} Config file not found: {config_path}")
            raise
        debug_log(f"{log_prefix} Config not found, using default: {config_path}")
        return default if default is not None else {}

    except yaml.YAMLError as e:
        if raise_on_error:
            error(f"{log_prefix} YAML parse error in {config_path}: {e}")
            raise
        debug_log(f"{log_prefix} YAML parse error, using default: {e}")
        return default if default is not None else {}

    except Exception as e:
        if raise_on_error:
            error(f"{log_prefix} Failed to load config from {config_path}: {e}")
            raise
        debug_log(f"{log_prefix} Load error, using default: {e}")
        return default if default is not None else {}


def load_yaml_with_fallback(
    config_path: Union[str, Path], fallback: T, log_prefix: str = "[Config]"
) -> Union[dict, T]:
    """
    Convenience function that loads YAML with a fallback value.

    Never raises exceptions - always returns either parsed content or fallback.

    Args:
        config_path: Path to the YAML file
        fallback: Value to return if loading fails
        log_prefix: Prefix for debug log messages

    Returns:
        Parsed YAML content or fallback value
    """
    return load_yaml(config_path, default=fallback, raise_on_error=False, log_prefix=log_prefix)


def save_yaml(config_path: Union[str, Path], data: Any, log_prefix: str = "[Config]") -> bool:
    """
    Save data to a YAML file with consistent formatting.

    Args:
        config_path: Path to save the YAML file
        data: Data to serialize to YAML
        log_prefix: Prefix for debug log messages

    Returns:
        True if save succeeded, False otherwise
    """
    config_path = Path(config_path)

    try:
        from src.logging_config import debug_log, error
    except ImportError:
        debug_log = lambda msg: None
        error = lambda msg: None

    try:
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        debug_log(f"{log_prefix} Saved config to {config_path}")
        return True

    except Exception as e:
        error(f"{log_prefix} Failed to save config to {config_path}: {e}")
        return False
