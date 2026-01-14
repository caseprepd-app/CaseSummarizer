"""
GPU Detection Utility for CasePrepd.

Detects whether the machine has a dedicated NVIDIA or AMD GPU.
Used to automatically enable/disable LLM vocabulary extraction.

Session 62b: Initial implementation with gpu-tracker and CLI tools.
Session 64: Switched to PyTorch + WMI for reliable cross-vendor detection.
            - PyTorch CUDA for NVIDIA (fast, no CLI needed)
            - WMI for Windows (detects all GPUs including AMD)
            - CLI fallback for edge cases
"""

import logging
import platform
import shutil
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)

# Keywords indicating dedicated GPUs (case-insensitive matching)
NVIDIA_KEYWORDS = ["geforce", "rtx", "gtx", "quadro", "tesla"]
AMD_KEYWORDS = ["radeon rx", "radeon pro", "firepro"]

# Keywords indicating integrated GPUs to EXCLUDE
# "AMD Radeon Graphics" (no RX/Pro) is typically integrated in Ryzen APUs
INTEGRATED_KEYWORDS = ["intel", "uhd graphics", "iris", "integrated", "radeon graphics"]


def _is_dedicated_gpu(gpu_name: str) -> tuple[bool, str]:
    """
    Check if a GPU name indicates a dedicated (non-integrated) GPU.

    Args:
        gpu_name: The GPU name string from WMI or other source.

    Returns:
        Tuple of (is_dedicated, vendor) where vendor is 'nvidia', 'amd', or 'unknown'.
    """
    name_lower = gpu_name.lower()

    # Check for integrated GPU keywords first (exclude these)
    for keyword in INTEGRATED_KEYWORDS:
        # Exception: Some AMD integrated GPUs have "Radeon" in name
        # but we want dedicated Radeon RX/Pro cards
        if keyword in name_lower and not any(
            amd_kw in name_lower for amd_kw in ["radeon rx", "radeon pro"]
        ):
            return False, "integrated"

    # Check for NVIDIA dedicated GPU
    for keyword in NVIDIA_KEYWORDS:
        if keyword in name_lower:
            return True, "nvidia"

    # Check for AMD dedicated GPU
    for keyword in AMD_KEYWORDS:
        if keyword in name_lower:
            return True, "amd"

    # Unknown GPU type - be conservative and say not dedicated
    return False, "unknown"


def _detect_gpu_wmi() -> dict | None:
    """
    Detect GPUs using Windows Management Instrumentation (WMI).

    Returns:
        Dict with GPU info if dedicated GPU found, None otherwise.
        Keys: gpu_name, vendor ('nvidia' or 'amd'), vram_bytes
    """
    from src.config import GPU_DETECTION_TIMEOUT

    if platform.system() != "Windows":
        return None

    try:
        # Query WMI for video controllers
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-WmiObject Win32_VideoController | "
                "Select-Object Name, AdapterRAM | "
                "ConvertTo-Json",
            ],
            capture_output=True,
            text=True,
            timeout=GPU_DETECTION_TIMEOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,  # Hide console window
        )

        if result.returncode != 0 or not result.stdout.strip():
            logger.debug("[GPU] WMI query returned no results")
            return None

        # Parse JSON output
        import json

        output = result.stdout.strip()

        # Handle single GPU (returns object) vs multiple (returns array)
        data = json.loads(output)
        gpus = [data] if isinstance(data, dict) else data

        # Check each GPU for dedicated status
        for gpu in gpus:
            gpu_name = gpu.get("Name", "")
            vram = gpu.get("AdapterRAM", 0)

            is_dedicated, vendor = _is_dedicated_gpu(gpu_name)

            if is_dedicated:
                logger.info(f"[GPU] WMI detected dedicated GPU: {gpu_name} ({vendor})")
                return {
                    "gpu_name": gpu_name,
                    "vendor": vendor,
                    "vram_bytes": vram,
                }

        logger.debug("[GPU] WMI found no dedicated GPUs")
        return None

    except subprocess.TimeoutExpired:
        logger.debug("[GPU] WMI query timed out")
        return None
    except json.JSONDecodeError as e:
        logger.debug(f"[GPU] WMI JSON parse error: {e}")
        return None
    except Exception as e:
        logger.debug(f"[GPU] WMI detection failed: {e}")
        return None


def _detect_gpu_pytorch() -> dict | None:
    """
    Detect NVIDIA GPU using PyTorch CUDA.

    Returns:
        Dict with GPU info if NVIDIA GPU found, None otherwise.
    """
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_name = torch.cuda.get_device_name(0)
            # Get VRAM if available
            try:
                vram = torch.cuda.get_device_properties(0).total_memory
            except Exception:
                vram = 0

            logger.info(f"[GPU] PyTorch detected NVIDIA GPU: {gpu_name}")
            return {
                "gpu_name": gpu_name,
                "vendor": "nvidia",
                "vram_bytes": vram,
            }
    except ImportError:
        logger.debug("[GPU] PyTorch not available")
    except Exception as e:
        logger.debug(f"[GPU] PyTorch CUDA check failed: {e}")

    return None


def _detect_gpu_cli() -> dict | None:
    """
    Detect GPU using CLI tools (nvidia-smi, amd-smi).

    Returns:
        Dict with GPU info if found, None otherwise.
    """
    # Check NVIDIA
    if shutil.which("nvidia-smi") is not None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split("\n")[0]
                logger.info(f"[GPU] nvidia-smi detected: {gpu_name}")
                return {
                    "gpu_name": gpu_name,
                    "vendor": "nvidia",
                    "vram_bytes": 0,
                }
        except Exception as e:
            logger.debug(f"[GPU] nvidia-smi query failed: {e}")

    # Check AMD
    amd_cmd = shutil.which("amd-smi") or shutil.which("rocm-smi")
    if amd_cmd is not None:
        logger.info(f"[GPU] AMD CLI tool found: {amd_cmd}")
        return {
            "gpu_name": "AMD GPU",
            "vendor": "amd",
            "vram_bytes": 0,
        }

    return None


@lru_cache(maxsize=1)
def has_dedicated_gpu() -> bool:
    """
    Detect if the machine has a dedicated NVIDIA or AMD GPU.

    Returns:
        True if a dedicated GPU is detected, False otherwise.

    Note:
        Result is cached after first call (GPU doesn't change at runtime).

    Detection methods (in order):
        1. PyTorch CUDA - Fast check for NVIDIA GPUs
        2. WMI Query - Windows-specific, detects all GPU vendors
        3. CLI tools - nvidia-smi / amd-smi fallback
    """
    # Method 1: PyTorch CUDA (fast, NVIDIA only)
    if _detect_gpu_pytorch() is not None:
        return True

    # Method 2: WMI (Windows, all vendors)
    if _detect_gpu_wmi() is not None:
        return True

    # Method 3: CLI tools (fallback)
    if _detect_gpu_cli() is not None:
        return True

    logger.info("[GPU] No dedicated GPU detected")
    return False


def get_gpu_info() -> dict:
    """
    Get detailed GPU information for display/logging.

    Returns:
        Dict with GPU detection details:
        - has_gpu: True if any dedicated GPU detected
        - gpu_name: Name of the GPU if detected
        - vendor: 'nvidia', 'amd', or None
        - vram_bytes: VRAM in bytes (0 if unknown)
        - detection_method: How the GPU was detected
    """
    result = {
        "has_gpu": False,
        "gpu_name": None,
        "vendor": None,
        "vram_bytes": 0,
        "detection_method": None,
    }

    # Try each detection method in order
    gpu_info = _detect_gpu_pytorch()
    if gpu_info:
        result.update(gpu_info)
        result["has_gpu"] = True
        result["detection_method"] = "pytorch"
        return result

    gpu_info = _detect_gpu_wmi()
    if gpu_info:
        result.update(gpu_info)
        result["has_gpu"] = True
        result["detection_method"] = "wmi"
        return result

    gpu_info = _detect_gpu_cli()
    if gpu_info:
        result.update(gpu_info)
        result["has_gpu"] = True
        result["detection_method"] = "cli"
        return result

    return result


def get_gpu_status_text() -> str:
    """
    Get a human-readable GPU status string for display in UI.

    Returns:
        Status string like "NVIDIA GeForce RTX 3080 detected" or
        "No dedicated GPU detected"
    """
    try:
        info = get_gpu_info()
        if info.get("has_gpu"):
            gpu_name = info.get("gpu_name")
            if gpu_name:
                return f"{gpu_name} detected"
            vendor = info.get("vendor", "").upper()
            if vendor:
                return f"{vendor} GPU detected"
            return "GPU detected"
        return "No dedicated GPU detected"
    except Exception:
        return "GPU detection unavailable"


# Conservative context window sizes based on VRAM (with 1.5% safety buffer)
# These values are intentionally conservative to avoid VRAM overflow which
# causes severe performance degradation (5-20x slower when spilling to RAM)
VRAM_CONTEXT_TIERS = [
    # (min_vram_gb, num_ctx)
    (24, 64000),  # 24GB+ → 64K context
    (16, 48000),  # 16-24GB → 48K context
    (12, 32000),  # 12-16GB → 32K context
    (8, 16000),  # 8-12GB → 16K context
    (6, 8000),  # 6-8GB → 8K context
    (0, 4000),  # < 6GB or no GPU → 4K context (safe default)
]


def get_optimal_context_size() -> int:
    """
    Calculate optimal LLM context window (num_ctx) based on detected VRAM.

    Returns:
        Recommended num_ctx value for Ollama API calls.

    Note:
        Values are conservative (1.5% below theoretical max) to ensure
        stability. VRAM overflow causes 5-20x performance degradation.

    Session 64: Added for dynamic LLM configuration based on hardware.
    """
    info = get_gpu_info()
    vram_bytes = info.get("vram_bytes", 0)

    # Convert to GB
    vram_gb = vram_bytes / (1024**3) if vram_bytes > 0 else 0

    # Find appropriate tier
    for min_vram, context_size in VRAM_CONTEXT_TIERS:
        if vram_gb >= min_vram:
            logger.info(f"[GPU] VRAM: {vram_gb:.1f}GB → optimal context: {context_size:,} tokens")
            return context_size

    # Fallback (shouldn't reach here due to 0 tier)
    return 4000


def get_vram_gb() -> float:
    """
    Get detected VRAM in gigabytes.

    Returns:
        VRAM in GB, or 0.0 if no GPU or VRAM unknown.
    """
    info = get_gpu_info()
    vram_bytes = info.get("vram_bytes", 0)
    return vram_bytes / (1024**3) if vram_bytes > 0 else 0.0


# Fixed chunk sizes based on RAG research (Session 67 revised)
#
# Research findings (2024-2025):
# - Optimal chunk size is 400-1024 tokens regardless of context window
# - Chroma research: 200-400 tokens for best precision
# - Firecrawl/Pinecone: 400-512 tokens as starting point
# - arXiv study: 512-1024 tokens for analytical queries
# - NVIDIA benchmark: 1024 tokens with 15% overlap
#
# Key insight: Chunk size should NOT scale with context window.
# What scales is HOW MANY CHUNKS you retrieve, not chunk size.
# Larger context = more chunks included, not bigger chunks.
#
# Sources:
# - https://research.trychroma.com/evaluating-chunking
# - https://arxiv.org/html/2407.19794v2 (Context Window Utilization)
# - https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025
OPTIMAL_CHUNK_SIZES = {
    "min_tokens": 400,  # Minimum to prevent semantic fragmentation
    "target_tokens": 700,  # Optimal for mixed query types (research: 500-800)
    "max_tokens": 1000,  # Upper bound (research: >1024 hurts precision)
}


def get_optimal_chunk_sizes(context_size: int | None = None) -> dict:
    """
    Get optimal chunk sizes for RAG retrieval.

    Based on 2024-2025 research, chunk sizes should be FIXED at 400-1000 tokens
    regardless of context window size. What scales with larger context is how
    many chunks can be retrieved, not the chunk size itself.

    Args:
        context_size: Context window size in tokens (used for reference only).
                      If None, auto-detects from VRAM.

    Returns:
        Dict with:
        - min_tokens: Minimum tokens per chunk (400)
        - target_tokens: Target tokens per chunk (700)
        - max_tokens: Maximum tokens per chunk (1000)
        - context_window: The context window (for reference)

    Session 67: Fixed sizes based on RAG research. Chunk size does NOT scale
    with context window - only the number of retrieved chunks scales.
    """
    if context_size is None:
        context_size = get_optimal_context_size()

    logger.debug(
        f"[GPU] Fixed chunk sizes: min={OPTIMAL_CHUNK_SIZES['min_tokens']}, "
        f"target={OPTIMAL_CHUNK_SIZES['target_tokens']}, "
        f"max={OPTIMAL_CHUNK_SIZES['max_tokens']} "
        f"(context={context_size:,})"
    )

    return {
        "min_tokens": OPTIMAL_CHUNK_SIZES["min_tokens"],
        "target_tokens": OPTIMAL_CHUNK_SIZES["target_tokens"],
        "max_tokens": OPTIMAL_CHUNK_SIZES["max_tokens"],
        "context_window": context_size,
    }
