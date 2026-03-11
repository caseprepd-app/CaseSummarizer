"""
GPU Detection Utility for CasePrepd.

Detects whether the machine has a dedicated NVIDIA or AMD GPU.
Used to configure embedding model device selection and resource allocation.

Uses PyTorch + WMI for reliable cross-vendor detection:
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
            vram = gpu.get("AdapterRAM", 0) or 0

            # WMI AdapterRAM is a 32-bit uint, overflows for GPUs >4GB VRAM.
            # Detect overflow: vram <= 0 or suspiciously wrapped values.
            if vram <= 0 or vram > 2**32:
                vram = 0
                logger.debug("[GPU] WMI AdapterRAM overflow for %s, VRAM unknown", gpu_name)

            is_dedicated, vendor = _is_dedicated_gpu(gpu_name)

            if is_dedicated:
                logger.info("[GPU] WMI detected dedicated GPU: %s (%s)", gpu_name, vendor)
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
        logger.debug("[GPU] WMI JSON parse error: %s", e)
        return None
    except Exception as e:
        logger.debug("[GPU] WMI detection failed: %s", e)
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
            except Exception as e:
                logger.debug("[GPU] Could not read VRAM from PyTorch: %s", e)
                vram = 0

            logger.info("[GPU] PyTorch detected NVIDIA GPU: %s", gpu_name)
            return {
                "gpu_name": gpu_name,
                "vendor": "nvidia",
                "vram_bytes": vram,
            }
    except ImportError:
        logger.debug("[GPU] PyTorch not available")
    except Exception as e:
        logger.debug("[GPU] PyTorch CUDA check failed: %s", e)

    return None


def _detect_gpu_cli() -> dict | None:
    """
    Detect GPU using CLI tools (nvidia-smi, amd-smi).

    Returns:
        Dict with GPU info if found, None otherwise.
    """
    # Check NVIDIA
    creationflags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
    if shutil.which("nvidia-smi") is not None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=creationflags,
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if not parts:
                    return None
                gpu_name = parts[0]
                vram_mib = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                vram_bytes = vram_mib * 1024 * 1024
                logger.info("[GPU] nvidia-smi detected: %s (%d MiB)", gpu_name, vram_mib)
                return {
                    "gpu_name": gpu_name,
                    "vendor": "nvidia",
                    "vram_bytes": vram_bytes,
                }
        except Exception as e:
            logger.debug("[GPU] nvidia-smi query failed: %s", e)

    # Check AMD
    amd_cmd = shutil.which("amd-smi") or shutil.which("rocm-smi")
    if amd_cmd is not None:
        logger.info("[GPU] AMD CLI tool found: %s", amd_cmd)
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


_gpu_info_cache: dict | None = None


def get_gpu_info() -> dict:
    """
    Get detailed GPU information for display/logging.

    Result is cached after first call (GPU doesn't change at runtime).

    Returns:
        Dict with GPU detection details:
        - has_gpu: True if any dedicated GPU detected
        - gpu_name: Name of the GPU if detected
        - vendor: 'nvidia', 'amd', or None
        - vram_bytes: VRAM in bytes (0 if unknown)
        - detection_method: How the GPU was detected
    """
    global _gpu_info_cache
    if _gpu_info_cache is not None:
        return _gpu_info_cache.copy()

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
    elif gpu_info := _detect_gpu_wmi():
        result.update(gpu_info)
        result["has_gpu"] = True
        result["detection_method"] = "wmi"
        # WMI AdapterRAM is uint32 -- maxes out at 4GB (Microsoft limitation).
        # For NVIDIA GPUs, always prefer nvidia-smi for accurate VRAM.
        if result["vendor"] == "nvidia" or result["vram_bytes"] == 0:
            cli_info = _detect_gpu_cli()
            if cli_info and cli_info.get("vram_bytes", 0) > 0:
                result["vram_bytes"] = cli_info["vram_bytes"]
                result["detection_method"] = "wmi+cli"
    elif gpu_info := _detect_gpu_cli():
        result.update(gpu_info)
        result["has_gpu"] = True
        result["detection_method"] = "cli"

    _gpu_info_cache = result
    return _gpu_info_cache.copy()


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
    except Exception as e:
        logger.warning("[GPU] Status text retrieval failed: %s", e)
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
    (0, 2048),  # < 6GB or no GPU → 2K context (CPU-safe; 4K causes 149s+ per query)
]


def get_optimal_context_size() -> int:
    """
    Calculate optimal LLM context window (num_ctx) based on detected VRAM.

    Returns:
        Recommended num_ctx value for Ollama API calls.

    Note:
        Values are conservative (1.5% below theoretical max) to ensure
        stability. VRAM overflow causes 5-20x performance degradation.

    Enables dynamic LLM configuration based on hardware.
    """
    info = get_gpu_info()
    vram_bytes = info.get("vram_bytes", 0)

    # Convert to GB
    vram_gb = vram_bytes / (1024**3) if vram_bytes > 0 else 0

    # Find appropriate tier
    for min_vram, context_size in VRAM_CONTEXT_TIERS:
        if vram_gb >= min_vram:
            logger.info("[GPU] VRAM: %.1fGB -> optimal context: %d tokens", vram_gb, context_size)
            return context_size

    # Fallback (shouldn't reach here due to 0 tier)
    return 2048


def get_vram_gb() -> float:
    """
    Get detected VRAM in gigabytes.

    Returns:
        VRAM in GB, or 0.0 if no GPU or VRAM unknown.
    """
    info = get_gpu_info()
    vram_bytes = info.get("vram_bytes", 0)
    return vram_bytes / (1024**3) if vram_bytes > 0 else 0.0


# Fixed chunk sizes based on RAG research
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
    "min_tokens": 300,  # Minimum to prevent semantic fragmentation
    "target_tokens": 700,  # Optimal for mixed query types (research: 500-800)
    "max_tokens": 1000,  # Upper bound (research: >1024 hurts precision)
}


def get_optimal_chunk_sizes(context_size: int | None = None) -> dict:
    """
    Get optimal chunk sizes for RAG retrieval.

    Based on 2024-2025 research, chunk sizes should be FIXED at 300-1000 tokens
    regardless of context window size. What scales with larger context is how
    many chunks can be retrieved, not the chunk size itself.

    Args:
        context_size: Context window size in tokens (used for reference only).
                      If None, auto-detects from VRAM.

    Returns:
        Dict with:
        - min_tokens: Minimum tokens per chunk (300)
        - target_tokens: Target tokens per chunk (700)
        - max_tokens: Maximum tokens per chunk (1000)
        - context_window: The context window (for reference)

    Chunk size is fixed based on RAG research. Chunk size does NOT scale
    with context window - only the number of retrieved chunks scales.
    """
    if context_size is None:
        context_size = get_optimal_context_size()

    logger.debug(
        "[GPU] Fixed chunk sizes: min=%d, target=%d, max=%d (context=%d)",
        OPTIMAL_CHUNK_SIZES["min_tokens"],
        OPTIMAL_CHUNK_SIZES["target_tokens"],
        OPTIMAL_CHUNK_SIZES["max_tokens"],
        context_size,
    )

    return {
        "min_tokens": OPTIMAL_CHUNK_SIZES["min_tokens"],
        "target_tokens": OPTIMAL_CHUNK_SIZES["target_tokens"],
        "max_tokens": OPTIMAL_CHUNK_SIZES["max_tokens"],
        "context_window": context_size,
    }
