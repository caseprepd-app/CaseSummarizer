"""
Vocabulary Table Column Configuration.

Contains:
- COLUMN_REGISTRY: Unified column configuration (width, visibility, etc.)
- COLUMN_ORDER: Display order for columns
- Display constants and helper functions
- compute_column_widths(): DPI-aware column width calculation
"""

import logging
import tkinter.font as tkfont

from src.config import (
    VF,
    VOCABULARY_BATCH_INSERT_DELAY_MS,
    VOCABULARY_BATCH_INSERT_SIZE,
    VOCABULARY_ROWS_PER_PAGE,
)

# Feedback icons (Unicode for cross-platform compatibility)
# Using checkmark and X for clearer approve/reject semantics
THUMB_UP_EMPTY = "\u2610"  # U+2610 Ballot Box (empty checkbox)
THUMB_UP_FILLED = "\u2713"  # U+2713 Check Mark (green via tag)
THUMB_DOWN_EMPTY = "\u2610"  # U+2610 Ballot Box (empty checkbox)
THUMB_DOWN_FILLED = "\u2717"  # U+2717 Ballot X (red via tag)

# Pagination settings (imported from config.py for centralized tuning)
ROWS_PER_PAGE = VOCABULARY_ROWS_PER_PAGE
BATCH_INSERT_SIZE = VOCABULARY_BATCH_INSERT_SIZE
BATCH_INSERT_DELAY_MS = VOCABULARY_BATCH_INSERT_DELAY_MS


# Unified column registry for configurable visibility
# Each column specifies: width, max_chars, default visibility, and whether it can be hidden
# User preferences override defaults; "Term" cannot be hidden
COLUMN_REGISTRY = {
    # Basic columns (default visible)
    VF.TERM: {"width": 180, "max_chars": 30, "default": True, "can_hide": False},
    "Score": {"width": 55, "max_chars": 5, "default": True, "can_hide": True},
    VF.IS_PERSON: {"width": 65, "max_chars": 4, "default": True, "can_hide": True},
    VF.FOUND_BY: {"width": 120, "max_chars": 20, "default": True, "can_hide": True},
    # TermSources columns (default visible)
    VF.OCCURRENCES: {"width": 65, "max_chars": 6, "default": True, "can_hide": True},
    VF.NUM_DOCS: {"width": 55, "max_chars": 4, "default": True, "can_hide": True},
    VF.OCR_CONFIDENCE: {"width": 80, "max_chars": 5, "default": True, "can_hide": True},
    # Algorithm detail columns (default hidden — 6 active algorithms)
    VF.NER: {"width": 45, "max_chars": 4, "default": False, "can_hide": True},
    VF.RAKE: {"width": 50, "max_chars": 4, "default": False, "can_hide": True},
    VF.BM25: {"width": 50, "max_chars": 4, "default": False, "can_hide": True},
    VF.TOPICRANK: {"width": 65, "max_chars": 4, "default": False, "can_hide": True},
    VF.MEDICALNER: {"width": 75, "max_chars": 4, "default": False, "can_hide": True},
    VF.YAKE: {"width": 50, "max_chars": 4, "default": False, "can_hide": True},
    VF.ALGO_COUNT: {"width": 55, "max_chars": 3, "default": False, "can_hide": True},
    # Additional columns (default hidden)
    VF.GOOGLE_RARITY_RANK: {"width": 80, "max_chars": 10, "default": False, "can_hide": True},
    # Feedback columns (default visible)
    VF.KEEP: {"width": 45, "max_chars": 3, "default": True, "can_hide": True},
    VF.SKIP: {"width": 45, "max_chars": 3, "default": True, "can_hide": True},
}

# Fixed column order (determines display sequence in table)
COLUMN_ORDER = [
    VF.TERM,
    "Score",
    VF.IS_PERSON,
    VF.FOUND_BY,
    VF.OCCURRENCES,
    VF.NUM_DOCS,
    VF.OCR_CONFIDENCE,
    VF.NER,
    VF.RAKE,
    VF.BM25,
    VF.TOPICRANK,
    VF.MEDICALNER,
    VF.YAKE,
    VF.ALGO_COUNT,
    VF.GOOGLE_RARITY_RANK,
    VF.KEEP,
    VF.SKIP,
]

# Backward compatibility: old column lists for reference
GUI_DISPLAY_COLUMNS = (VF.TERM, "Score", VF.IS_PERSON, VF.FOUND_BY, VF.KEEP, VF.SKIP)
GUI_DISPLAY_COLUMNS_EXTENDED = (
    VF.TERM,
    "Score",
    VF.IS_PERSON,
    VF.FOUND_BY,
    VF.NER,
    VF.RAKE,
    VF.BM25,
    VF.TOPICRANK,
    VF.MEDICALNER,
    VF.YAKE,
    VF.ALGO_COUNT,
    VF.KEEP,
    VF.SKIP,
)

# All columns available for export (includes ML feature columns)
ALL_EXPORT_COLUMNS = (
    VF.TERM,
    VF.QUALITY_SCORE,
    VF.IS_PERSON,
    VF.FOUND_BY,
    VF.NUM_DOCS,
    VF.OCR_CONFIDENCE,
    VF.NER,
    VF.RAKE,
    VF.BM25,
    VF.TOPICRANK,
    VF.MEDICALNER,
    VF.YAKE,
    VF.ALGO_COUNT,
    VF.OCCURRENCES,
    VF.GOOGLE_RARITY_RANK,
)

# UI-002: Centralized mapping from display column names to data field names
# "Score" in the GUI maps to "Quality Score" in the data dictionary
DISPLAY_TO_DATA_COLUMN = {
    "Score": VF.QUALITY_SCORE,
}

# Legacy COLUMN_CONFIG for backward compatibility
COLUMN_CONFIG = {
    name: {"width": cfg["width"], "max_chars": cfg["max_chars"]}
    for name, cfg in COLUMN_REGISTRY.items()
}


logger = logging.getLogger(__name__)

# Minimum column width in pixels (prevents columns from collapsing)
MIN_COLUMN_WIDTH = 40

# Sort indicator takes ~12px in the heading
_SORT_INDICATOR_PADDING = 12

# Cell padding (left + right) added to measured text width
_CELL_PADDING = 16


def truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text to prevent Treeview row overflow.

    Args:
        text: Text to truncate
        max_chars: Maximum characters before truncation

    Returns:
        Truncated text with ellipsis if needed
    """
    if not text:
        return ""
    text = str(text).replace("\n", " ").replace("\r", "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def compute_column_widths(
    visible_columns: list[str],
    font_spec: tuple,
    heading_font_spec: tuple,
    available_width: int,
    data_sample: list[dict] | None = None,
) -> dict[str, int]:
    """
    Calculate DPI-aware column widths using font metrics.

    Uses tkfont.Font.measure() which automatically accounts for DPI,
    following the same pattern as _get_rowheight() in styles.py.

    Args:
        visible_columns: List of visible column names.
        font_spec: Tuple like ("Segoe UI", 10) for content font.
        heading_font_spec: Tuple like ("Segoe UI", 10, "bold") for headings.
        available_width: Total available width in pixels.
        data_sample: Optional list of row dicts to measure content width.

    Returns:
        Dict of {column_name: width_in_pixels}.
    """
    content_font = tkfont.Font(font=font_spec)
    heading_font = tkfont.Font(font=heading_font_spec)

    widths = {}
    for col in visible_columns:
        # Heading width: text + sort indicator + padding
        heading_w = heading_font.measure(col) + _SORT_INDICATOR_PADDING + _CELL_PADDING

        # Content width: measure actual data if available
        content_w = 0
        if data_sample:
            content_w = _measure_content_width(col, data_sample, content_font)

        # Column width = max of heading and content, floored at MIN_COLUMN_WIDTH
        col_w = max(heading_w, content_w, MIN_COLUMN_WIDTH)

        # Apply max-width caps to prevent one column from dominating
        max_w = _max_width_for_column(col, available_width)
        widths[col] = min(col_w, max_w)

    return widths


def _measure_content_width(
    col_name: str, data_sample: list[dict], content_font: tkfont.Font
) -> int:
    """
    Measure the widest content in a column across up to 100 sample rows.

    Args:
        col_name: Column name (or mapped data key).
        data_sample: List of row dicts.
        content_font: Font used for content text.

    Returns:
        Width in pixels of the widest cell content + padding.
    """
    data_key = DISPLAY_TO_DATA_COLUMN.get(col_name, col_name)
    max_w = 0
    max_chars = COLUMN_REGISTRY.get(col_name, {}).get("max_chars", 30)

    for row in data_sample[:100]:
        value = row.get(data_key, "")
        text = truncate_text(str(value), max_chars)
        if text:
            w = content_font.measure(text)
            if w > max_w:
                max_w = w

    return max_w + _CELL_PADDING if max_w > 0 else 0


def _max_width_for_column(col_name: str, available_width: int) -> int:
    """
    Return max pixel width for a column based on its role.

    Args:
        col_name: Column name.
        available_width: Total available width.

    Returns:
        Maximum allowed width in pixels.
    """
    if col_name == VF.TERM:
        return int(available_width * 0.45)
    if col_name == VF.FOUND_BY:
        return int(available_width * 0.20)
    return int(available_width * 0.15)
