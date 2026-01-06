"""
Shared Column Configuration for Vocabulary Table

Session 80: Single source of truth for column definitions used by both
GUI (dynamic_output.py) and HTML export (html_builder.py).

This module defines:
- Column metadata (width, visibility, hideability)
- Sort behavior (which columns trigger warnings)
- Numeric vs string sorting
- Display-to-data key mappings
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ColumnDefinition:
    """
    Definition of a vocabulary table column.

    Attributes:
        name: Display name shown in column header
        data_key: Key in vocabulary data dict (may differ from name)
        width: Default width in pixels
        max_chars: Max characters before truncation
        default_visible: Whether shown by default
        can_hide: Whether user can hide this column (False = required)
        triggers_sort_warning: Whether sorting by this column shows a warning
        is_numeric: Whether column uses numeric sorting (vs string)
    """

    name: str
    data_key: str
    width: int
    max_chars: int
    default_visible: bool
    can_hide: bool
    triggers_sort_warning: bool
    is_numeric: bool


# All column definitions in display order
# Note: Only "Score" has triggers_sort_warning=False because it's the quality
# ranking - sorting by Score shows best results first (intended behavior).
# All other columns trigger a warning since non-Score sorts show lower-quality first.
COLUMN_DEFINITIONS = [
    # Basic columns (default visible)
    ColumnDefinition("Term", "Term", 180, 30, True, False, False, False),
    ColumnDefinition("Score", "Quality Score", 55, 5, True, True, False, True),
    ColumnDefinition("Is Person", "Is Person", 65, 4, True, True, True, False),
    ColumnDefinition("Found By", "Found By", 120, 20, True, True, True, False),
    # TermSources columns (default visible)
    ColumnDefinition("# Docs", "# Docs", 55, 4, True, True, True, True),
    ColumnDefinition("Count", "Count", 60, 6, True, True, True, True),
    ColumnDefinition("Median Conf", "Median Conf", 80, 5, True, True, True, False),
    # Algorithm detail columns (default hidden)
    ColumnDefinition("NER", "NER", 45, 4, False, True, True, False),
    ColumnDefinition("RAKE", "RAKE", 50, 4, False, True, True, False),
    ColumnDefinition("BM25", "BM25", 50, 4, False, True, True, False),
    ColumnDefinition("Algo Count", "Algo Count", 55, 3, False, True, True, True),
    # Additional columns (default hidden)
    ColumnDefinition("Freq Rank", "Freq Rank", 80, 10, False, True, True, True),
    # Feedback columns (default visible) - Keep/Skip don't need sort warning
    # as they're action columns, not data columns
    ColumnDefinition("Keep", "Keep", 45, 3, True, True, False, False),
    ColumnDefinition("Skip", "Skip", 45, 3, True, True, False, False),
]

# ============================================================================
# Convenience lookups (derived from COLUMN_DEFINITIONS)
# ============================================================================

# Column names in display order
COLUMN_NAMES = tuple(c.name for c in COLUMN_DEFINITIONS)

# Columns that cannot be hidden (Term is required)
PROTECTED_COLUMNS = frozenset(c.name for c in COLUMN_DEFINITIONS if not c.can_hide)

# Columns that trigger sort warning (all except Score, Keep, Skip)
SORT_WARNING_COLUMNS = frozenset(c.name for c in COLUMN_DEFINITIONS if c.triggers_sort_warning)

# Columns that use numeric sorting
NUMERIC_COLUMNS = frozenset(c.name for c in COLUMN_DEFINITIONS if c.is_numeric)

# Display name to data key mapping (for columns where they differ)
DISPLAY_TO_DATA_KEY = {c.name: c.data_key for c in COLUMN_DEFINITIONS if c.name != c.data_key}


def get_column_by_name(name: str) -> Optional[ColumnDefinition]:
    """
    Get column definition by display name.

    Args:
        name: Column display name (e.g., "Score", "Term")

    Returns:
        ColumnDefinition if found, None otherwise
    """
    for col in COLUMN_DEFINITIONS:
        if col.name == name:
            return col
    return None


def get_data_key(display_name: str) -> str:
    """
    Get the data dictionary key for a column.

    Args:
        display_name: Column display name (e.g., "Score")

    Returns:
        Data key (e.g., "Quality Score" for "Score", or same as input)
    """
    return DISPLAY_TO_DATA_KEY.get(display_name, display_name)


def build_column_registry() -> dict[str, dict]:
    """
    Build COLUMN_REGISTRY dict for backward compatibility with dynamic_output.py.

    Returns:
        Dict mapping column name to {width, max_chars, default, can_hide}
    """
    return {
        c.name: {
            "width": c.width,
            "max_chars": c.max_chars,
            "default": c.default_visible,
            "can_hide": c.can_hide,
        }
        for c in COLUMN_DEFINITIONS
    }
