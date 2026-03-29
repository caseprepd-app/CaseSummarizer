"""
Centralized config/data/assets path resolution for CasePrepd.

ALL config, data, and asset file paths MUST use get_config_dir(),
get_data_dir(), or get_assets_dir() from this module. Do NOT compute
paths from __file__ — that pattern is fragile in PyInstaller frozen
builds where the directory layout differs from development.

In dev mode:
    Base dir = project root (parent of src/)
In frozen (PyInstaller onedir) mode:
    Base dir = sys._MEIPASS (e.g. dist/CasePrepd/_internal/)

This module is the SINGLE SOURCE OF TRUTH for base directory detection.
src/config.py imports from here for BUNDLED_BASE_DIR / BUNDLED_CONFIG_DIR.

Usage in src/core/ modules:
    from src.core.paths import get_config_dir, get_data_dir
    path = get_config_dir() / "categories.json"

Usage in src/services/ or src/ modules:
    Same import, or use BUNDLED_CONFIG_DIR from src/config.py.

NEVER do this:
    Path(__file__).parent.parent / "config" / "file.json"  # WRONG
"""

import sys
from pathlib import Path


def get_base_dir() -> Path:
    """Return the project root (dev) or _MEIPASS (frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    # In dev: this file is src/core/paths.py → .parent.parent.parent = project root
    return Path(__file__).parent.parent.parent


def get_config_dir() -> Path:
    """Return the bundled config/ directory path."""
    return get_base_dir() / "config"


def get_data_dir() -> Path:
    """Return the bundled data/ directory path."""
    return get_base_dir() / "data"


def get_assets_dir() -> Path:
    """Return the bundled assets/ directory path."""
    return get_base_dir() / "assets"
