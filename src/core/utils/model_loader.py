"""
Model Loading Utilities.

Shared helpers for loading bundled models. In frozen (installed) mode,
models MUST exist locally — no network downloads are attempted.
In dev mode, falls back to HuggingFace model name for convenience.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Block all HuggingFace network access unconditionally.
# Models are bundled with the app; downloading is never intended.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


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

    In frozen mode, force-sets to bundled path so user env vars can't
    redirect model loading. In dev mode, uses setdefault to respect
    user-configured values.

    Args:
        cache_dir: Directory for HuggingFace model cache
    """
    if getattr(sys, "frozen", False):
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    else:
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
