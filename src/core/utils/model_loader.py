"""
Model Loading Utilities.

Shared helpers for loading bundled models. In frozen (installed) mode,
models MUST exist locally — no network downloads are attempted.
In dev mode, falls back to HuggingFace model name for convenience.

HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are set in config.py at import
time, before any model loading can occur.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_model_path(bundled_path: Path, hf_model_name: str) -> tuple[str, bool]:
    """
    Resolve a model path, requiring bundled local model in frozen mode.

    Args:
        bundled_path: Path where bundled model should be installed
        hf_model_name: HuggingFace model name (dev fallback only)

    Returns:
        Tuple of (model_path_string, is_local)

    Raises:
        RuntimeError: If frozen and bundled model is missing.
    """
    if bundled_path.exists():
        model_path = str(bundled_path)
        logger.debug("Using bundled model: %s", model_path)
        return model_path, True

    if getattr(sys, "frozen", False):
        raise RuntimeError(
            f"Required model not found: {bundled_path}\n"
            f"Please reinstall the application to restore model files."
        )

    logger.warning(
        "Bundled model not found at %s — using installed package: %s",
        bundled_path,
        hf_model_name,
    )
    return hf_model_name, False


def set_hf_cache_env(cache_dir: Path) -> None:
    """
    Set HuggingFace cache environment variables.

    Force-sets to bundled path so dev mode mirrors production exactly.

    Args:
        cache_dir: Directory for HuggingFace model cache
    """
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
