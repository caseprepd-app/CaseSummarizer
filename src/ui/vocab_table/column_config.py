"""
Vocabulary Table Column Configuration.

Session 82: Extracted from dynamic_output.py for modularity.

Contains:
- COLUMN_REGISTRY: Unified column configuration (width, visibility, etc.)
- COLUMN_ORDER: Display order for columns
- Display constants and helper functions
"""

from src.config import (
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


# Session 80: Unified column registry for configurable visibility
# Each column specifies: width, max_chars, default visibility, and whether it can be hidden
# User preferences override defaults; "Term" cannot be hidden
COLUMN_REGISTRY = {
    # Basic columns (default visible)
    "Term": {"width": 180, "max_chars": 30, "default": True, "can_hide": False},
    "Score": {"width": 55, "max_chars": 5, "default": True, "can_hide": True},
    "Is Person": {"width": 65, "max_chars": 4, "default": True, "can_hide": True},
    "Found By": {"width": 120, "max_chars": 20, "default": True, "can_hide": True},
    # TermSources columns (Session 80 - default visible)
    "# Docs": {"width": 55, "max_chars": 4, "default": True, "can_hide": True},
    "OCR Confidence": {"width": 80, "max_chars": 5, "default": True, "can_hide": True},
    # Algorithm detail columns (default hidden - formerly "Show Details")
    "NER": {"width": 45, "max_chars": 4, "default": False, "can_hide": True},
    "RAKE": {"width": 50, "max_chars": 4, "default": False, "can_hide": True},
    "BM25": {"width": 50, "max_chars": 4, "default": False, "can_hide": True},
    "Algo Count": {"width": 55, "max_chars": 3, "default": False, "can_hide": True},
    # Additional columns (default hidden)
    "Google Rarity Rank": {"width": 80, "max_chars": 10, "default": False, "can_hide": True},
    # Feedback columns (default visible)
    "Keep": {"width": 45, "max_chars": 3, "default": True, "can_hide": True},
    "Skip": {"width": 45, "max_chars": 3, "default": True, "can_hide": True},
}

# Fixed column order (determines display sequence in table)
COLUMN_ORDER = [
    "Term",
    "Score",
    "Is Person",
    "Found By",
    "# Docs",
    "OCR Confidence",
    "NER",
    "RAKE",
    "BM25",
    "Algo Count",
    "Google Rarity Rank",
    "Keep",
    "Skip",
]

# Backward compatibility: old column lists for reference
GUI_DISPLAY_COLUMNS = ("Term", "Score", "Is Person", "Found By", "Keep", "Skip")
GUI_DISPLAY_COLUMNS_EXTENDED = (
    "Term",
    "Score",
    "Is Person",
    "Found By",
    "NER",
    "RAKE",
    "BM25",
    "Algo Count",
    "Keep",
    "Skip",
)

# All columns available for export (includes ML feature columns)
ALL_EXPORT_COLUMNS = (
    "Term",
    "Quality Score",
    "Is Person",
    "Found By",
    "# Docs",
    "OCR Confidence",
    "NER",
    "RAKE",
    "BM25",
    "Algo Count",
    "Occurrences",
    "Google Rarity Rank",
)

# UI-002: Centralized mapping from display column names to data field names
# "Score" in the GUI maps to "Quality Score" in the data dictionary
DISPLAY_TO_DATA_COLUMN = {
    "Score": "Quality Score",
}

# Legacy COLUMN_CONFIG for backward compatibility
COLUMN_CONFIG = {
    name: {"width": cfg["width"], "max_chars": cfg["max_chars"]}
    for name, cfg in COLUMN_REGISTRY.items()
}


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
