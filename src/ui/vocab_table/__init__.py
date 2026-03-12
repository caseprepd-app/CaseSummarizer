"""
Vocabulary Table Components.

This package contains the components for the vocabulary table display:
- column_config: Column registry and display constants
- vocab_treeview: Self-contained treeview widget with item-data mapping
"""

from src.ui.vocab_table.column_config import (
    ALL_EXPORT_COLUMNS,
    BATCH_INSERT_DELAY_MS,
    BATCH_INSERT_SIZE,
    COLUMN_CONFIG,
    COLUMN_ORDER,
    COLUMN_REGISTRY,
    DISPLAY_TO_DATA_COLUMN,
    GUI_DISPLAY_COLUMNS,
    GUI_DISPLAY_COLUMNS_EXTENDED,
    ROWS_PER_PAGE,
    THUMB_DOWN_EMPTY,
    THUMB_DOWN_FILLED,
    THUMB_UP_EMPTY,
    THUMB_UP_FILLED,
    truncate_text,
)

__all__ = [
    "ALL_EXPORT_COLUMNS",
    "BATCH_INSERT_DELAY_MS",
    "BATCH_INSERT_SIZE",
    "COLUMN_CONFIG",
    "COLUMN_ORDER",
    "COLUMN_REGISTRY",
    "DISPLAY_TO_DATA_COLUMN",
    "GUI_DISPLAY_COLUMNS",
    "GUI_DISPLAY_COLUMNS_EXTENDED",
    "ROWS_PER_PAGE",
    "THUMB_DOWN_EMPTY",
    "THUMB_DOWN_FILLED",
    "THUMB_UP_EMPTY",
    "THUMB_UP_FILLED",
    "VocabTreeview",
    "strip_display_prefix",
    "truncate_text",
]

from src.ui.vocab_table.vocab_treeview import VocabTreeview, strip_display_prefix
