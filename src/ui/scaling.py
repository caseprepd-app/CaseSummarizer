"""
Display Scaling Module for CasePrepd

Provides automatic and user-configurable UI scaling for different
monitor resolutions (1080p, QHD, 4K, etc.).

Two independent controls:
- Font size offset: integer point adjustment to all fonts (-4 to +10)
- UI scale: percentage scaling for widget dimensions (75% to 200%)

Called once at startup in main.py before MainWindow is created.
"""

import logging

import customtkinter as ctk

logger = logging.getLogger(__name__)


def get_effective_font_offset() -> int:
    """
    Read font_size_offset from user preferences.

    Returns:
        int: Point offset to apply to all fonts (default 0)
    """
    from src.user_preferences import get_user_preferences

    prefs = get_user_preferences()
    offset = prefs.get("font_size_offset", None)

    # Migration: if new key doesn't exist, check old font_size key
    if offset is None:
        old_key = prefs.get("font_size", "medium")
        migration_map = {"small": -2, "medium": 0, "large": 2}
        offset = migration_map.get(old_key, 0)

    try:
        return int(offset)
    except (TypeError, ValueError):
        return 0


def get_effective_ui_scale() -> float:
    """
    Read ui_scale_pct from user preferences and return as float multiplier.

    Returns:
        float: Scale factor (e.g. 1.0 for 100%, 1.25 for 125%)
    """
    from src.user_preferences import get_user_preferences

    prefs = get_user_preferences()
    pct = prefs.get("ui_scale_pct", 100)
    try:
        return max(0.75, min(2.0, int(pct) / 100.0))
    except (TypeError, ValueError):
        return 1.0


def apply_scaling() -> None:
    """
    Apply font and UI scaling at startup.

    Must be called once in main() after ctk.set_default_color_theme()
    and before MainWindow() is created.
    """
    font_offset = get_effective_font_offset()
    ui_scale = get_effective_ui_scale()

    logger.info("Applying display scaling: font_offset=%d, ui_scale=%.2f", font_offset, ui_scale)

    # Scale fonts (point offset)
    from src.ui.theme import scale_fonts

    scale_fonts(font_offset)

    # Scale CTk widgets (buttons, frames, padding)
    ctk.set_widget_scaling(ui_scale)

    # NOTE: Do NOT call initialize_all_styles() here.
    # ttk.Style() requires a Tk root, which doesn't exist until MainWindow().
    # MainWindow.__init__() calls initialize_all_styles() after super().__init__().

    # Scale vocabulary table column widths
    from src.config import scale_column_widths

    scale_column_widths(ui_scale)


def scale_value(pixels: int) -> int:
    """
    Scale a hardcoded pixel value by the effective UI scale.

    Use this for dialog geometry, window sizes, and other hardcoded
    dimensions that aren't handled by ctk.set_widget_scaling().

    Args:
        pixels: Original pixel value

    Returns:
        int: Scaled pixel value
    """
    return int(pixels * get_effective_ui_scale())
