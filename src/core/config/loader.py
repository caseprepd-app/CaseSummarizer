"""
Unified configuration loading utilities.

Replaces 7+ duplicate _load_config() implementations across the codebase.
Provides consistent YAML/JSON loading with proper error handling and logging.
"""

import logging
from pathlib import Path
from typing import Any, TypeVar

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


def load_yaml(
    config_path: str | Path,
    default: T | None = None,
    raise_on_error: bool = True,
    log_prefix: str = "[Config]",
) -> dict | T:
    """
    Load a YAML configuration file with consistent error handling.

    Replaces duplicate _load_config() methods in:
    - chunking_engine.py
    - vector_store/question_flow.py
    - semantic/semantic_orchestrator.py
    - ui/semantic_question_editor.py (2 locations)
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

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.debug("%s Loaded config from %s", log_prefix, config_path)
        return config if config is not None else {}

    except FileNotFoundError:
        if raise_on_error:
            logger.error("%s Config file not found: %s", log_prefix, config_path)
            raise
        logger.debug("%s Config not found, using default: %s", log_prefix, config_path)
        return default if default is not None else {}

    except yaml.YAMLError as e:
        if raise_on_error:
            logger.error("%s YAML parse error in %s: %s", log_prefix, config_path, e)
            raise
        logger.debug("%s YAML parse error, using default: %s", log_prefix, e)
        return default if default is not None else {}

    except Exception as e:
        if raise_on_error:
            logger.error("%s Failed to load config from %s: %s", log_prefix, config_path, e)
            raise
        logger.debug("%s Load error, using default: %s", log_prefix, e)
        return default if default is not None else {}


def load_yaml_with_fallback(
    config_path: str | Path, fallback: T, log_prefix: str = "[Config]"
) -> dict | T:
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


def save_yaml(config_path: str | Path, data: Any, log_prefix: str = "[Config]") -> bool:
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
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.debug("%s Saved config to %s", log_prefix, config_path)
        return True

    except Exception as e:
        logger.error("%s Failed to save config to %s: %s", log_prefix, config_path, e)
        return False
