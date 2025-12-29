"""
GPU Detection Utility for LocalScribe (Session 62b).

Detects whether the machine has a dedicated NVIDIA or AMD GPU.
Used to automatically enable/disable LLM vocabulary extraction.

Uses gpu-tracker library for cross-platform detection, with fallback
to checking for nvidia-smi/amd-smi command availability.
"""

import logging
import shutil
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def has_dedicated_gpu() -> bool:
    """
    Detect if the machine has a dedicated NVIDIA or AMD GPU.

    Returns:
        True if a dedicated GPU is detected, False otherwise.

    Note:
        Result is cached after first call (GPU doesn't change at runtime).
        Checks both gpu-tracker library and nvidia-smi/amd-smi availability.
    """
    try:
        # First check: nvidia-smi or amd-smi command availability
        # This is fast and doesn't require gpu-tracker to work
        nvidia_available = shutil.which('nvidia-smi') is not None
        amd_available = (
            shutil.which('amd-smi') is not None or
            shutil.which('rocm-smi') is not None
        )

        if nvidia_available or amd_available:
            logger.info(
                f"[GPU] Detected GPU tools: NVIDIA={nvidia_available}, AMD={amd_available}"
            )
            return True

        # Second check: use gpu-tracker for more thorough detection
        try:
            from gpu_tracker import Tracker

            # Create a minimal tracker just to check GPU availability
            tracker = Tracker()

            # Check if either NVIDIA or AMD GPU is available
            # gpu-tracker reports 0 for GPU metrics if no GPU found
            gpu_info = tracker.get_current()

            # If GPU memory is reported > 0, a GPU exists
            if gpu_info.get('gpu_memory_used', 0) > 0:
                logger.info("[GPU] Dedicated GPU detected via gpu-tracker")
                return True

        except ImportError:
            logger.debug("[GPU] gpu-tracker not installed, using command-only detection")
        except Exception as e:
            logger.debug(f"[GPU] gpu-tracker check failed: {e}")

        logger.info("[GPU] No dedicated GPU detected")
        return False

    except Exception as e:
        logger.warning(f"[GPU] Error detecting GPU: {e}, assuming no GPU")
        return False


def get_gpu_info() -> dict:
    """
    Get detailed GPU information for display/logging.

    Returns:
        Dict with GPU detection details:
        - nvidia_available: True if nvidia-smi command exists
        - amd_available: True if amd-smi/rocm-smi command exists
        - has_gpu: True if any dedicated GPU detected
    """
    try:
        nvidia_available = shutil.which('nvidia-smi') is not None
        amd_available = (
            shutil.which('amd-smi') is not None or
            shutil.which('rocm-smi') is not None
        )
        return {
            "nvidia_available": nvidia_available,
            "amd_available": amd_available,
            "has_gpu": has_dedicated_gpu(),
        }
    except Exception:
        return {
            "nvidia_available": False,
            "amd_available": False,
            "has_gpu": False,
        }


def get_gpu_status_text() -> str:
    """
    Get a human-readable GPU status string for display in UI.

    Returns:
        Status string like "NVIDIA GPU detected" or "No dedicated GPU detected"
    """
    try:
        info = get_gpu_info()
        if info.get("has_gpu"):
            if info.get("nvidia_available"):
                return "NVIDIA GPU detected"
            elif info.get("amd_available"):
                return "AMD GPU detected"
            return "GPU detected"
        return "No dedicated GPU detected"
    except Exception:
        return "GPU detection unavailable"
