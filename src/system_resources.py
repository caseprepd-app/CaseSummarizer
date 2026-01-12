"""
System Resource Manager for CasePrepd.

Provides dynamic calculation of optimal worker counts based on available
system resources (CPU cores, RAM) and user-configurable resource usage
percentage.

The goal is to allow users to balance processing speed vs. system impact:
- 100%: Maximum speed, but computer may be slow during processing
- 75%: Good balance (recommended default)
- 50%: Moderate impact, computer remains responsive
- 25%: Minimal impact, processing takes longer

Usage:
    from src.system_resources import get_optimal_workers

    # For Case Briefing extraction (2GB per worker estimate)
    workers = get_optimal_workers(task_ram_gb=2.0)

    # For document extraction (0.5GB per worker estimate)
    workers = get_optimal_workers(task_ram_gb=0.5)
"""

import os
from typing import NamedTuple

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.logging_config import debug_log
from src.user_preferences import get_user_preferences


class ResourceInfo(NamedTuple):
    """System resource information."""

    cpu_count: int
    available_ram_gb: float
    total_ram_gb: float
    resource_usage_pct: int


def get_system_resources() -> ResourceInfo:
    """
    Get current system resource information.

    Returns:
        ResourceInfo with CPU count, available RAM, total RAM, and user's
        configured resource usage percentage.
    """
    prefs = get_user_preferences()
    resource_pct = prefs.get("resource_usage_pct", 75)

    cpu_count = os.cpu_count() or 4

    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        available_ram_gb = mem.available / (1024**3)
        total_ram_gb = mem.total / (1024**3)
    else:
        # Fallback: assume 16GB total, 8GB available (conservative defaults)
        debug_log("[Resources] psutil not available, using conservative defaults")
        available_ram_gb = 8.0
        total_ram_gb = 16.0

    return ResourceInfo(
        cpu_count=cpu_count,
        available_ram_gb=available_ram_gb,
        total_ram_gb=total_ram_gb,
        resource_usage_pct=resource_pct,
    )


def get_optimal_workers(
    task_ram_gb: float = 2.0,
    max_workers: int = 8,
    min_workers: int = 1,
) -> int:
    """
    Calculate optimal worker count based on system resources.

    The calculation considers:
    1. User's resource usage percentage (25-100%)
    2. Available CPU cores
    3. Available RAM (each Ollama request uses ~2GB)

    The final worker count is the minimum of:
    - CPU-based limit: cores * (resource_pct / 100)
    - RAM-based limit: available_ram / task_ram_gb
    - Hard maximum: max_workers

    Args:
        task_ram_gb: Estimated RAM per worker (default 2GB for Ollama)
        max_workers: Hard upper limit on workers
        min_workers: Minimum workers regardless of resources

    Returns:
        Optimal number of parallel workers (at least min_workers)
    """
    resources = get_system_resources()

    # Calculate CPU-based limit
    cpu_fraction = resources.resource_usage_pct / 100.0
    cpu_limited = int(resources.cpu_count * cpu_fraction)

    # Calculate RAM-based limit (leave some headroom)
    # Use 80% of available RAM to leave room for OS and other apps
    usable_ram = resources.available_ram_gb * 0.8
    ram_limited = int(usable_ram / task_ram_gb) if task_ram_gb > 0 else max_workers

    # Take the minimum of CPU and RAM limits
    optimal = min(cpu_limited, ram_limited, max_workers)

    # Ensure at least min_workers
    final_workers = max(min_workers, optimal)

    debug_log(
        f"[Resources] Calculated workers: {final_workers} "
        f"(CPU: {resources.cpu_count} cores x {resources.resource_usage_pct}% = {cpu_limited}, "
        f"RAM: {resources.available_ram_gb:.1f}GB avail / {task_ram_gb}GB per worker = {ram_limited}, "
        f"cap: {max_workers})"
    )

    return final_workers


def get_resource_summary() -> str:
    """
    Get a human-readable summary of system resources.

    Returns:
        String like "8 cores, 12.3 GB RAM available (75% usage)"
    """
    resources = get_system_resources()
    return (
        f"{resources.cpu_count} cores, "
        f"{resources.available_ram_gb:.1f} GB RAM available "
        f"({resources.resource_usage_pct}% usage setting)"
    )
