"""
Model Loading Utilities.

Shared helpers for the bundled-model-first, HuggingFace-fallback loading
pattern used by faiss_semantic and cross_encoder_reranker.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_model_path(bundled_path: Path, hf_model_name: str) -> tuple[str, bool]:
    """
    Resolve a model path, preferring bundled local model.

    Args:
        bundled_path: Path where bundled model would be installed
        hf_model_name: HuggingFace model name for fallback download

    Returns:
        Tuple of (model_path_string, is_local)
    """
    if bundled_path.exists():
        model_path = str(bundled_path)
        logger.debug("Using bundled model: %s", model_path)
        return model_path, True

    logger.warning(
        "Bundled model not found at %s — falling back to HuggingFace: %s",
        bundled_path,
        hf_model_name,
    )
    return hf_model_name, False


def set_hf_cache_env(cache_dir: Path) -> None:
    """
    Set HuggingFace cache environment variables.

    Uses setdefault to avoid overwriting user-configured values.

    Args:
        cache_dir: Directory for HuggingFace model cache
    """
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
