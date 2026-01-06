"""
Vocabulary Table Components.

Session 82: Split from dynamic_output.py for modularity.

This package contains the components for the vocabulary table display:
- column_config: Column registry and display constants
- treeview_manager: Treeview display, sorting, filtering (mixin)
- export_handler: Export methods (TXT, CSV, Word, PDF, HTML) (mixin)
- feedback_handler: Feedback UI and term exclusion (mixin)
"""

from src.ui.vocab_table.column_config import (
    COLUMN_REGISTRY,
    COLUMN_ORDER,
    COLUMN_CONFIG,
    GUI_DISPLAY_COLUMNS,
    GUI_DISPLAY_COLUMNS_EXTENDED,
    ALL_EXPORT_COLUMNS,
    DISPLAY_TO_DATA_COLUMN,
    THUMB_UP_EMPTY,
    THUMB_UP_FILLED,
    THUMB_DOWN_EMPTY,
    THUMB_DOWN_FILLED,
    ROWS_PER_PAGE,
    BATCH_INSERT_SIZE,
    BATCH_INSERT_DELAY_MS,
    truncate_text,
)

__all__ = [
    "COLUMN_REGISTRY",
    "COLUMN_ORDER",
    "COLUMN_CONFIG",
    "GUI_DISPLAY_COLUMNS",
    "GUI_DISPLAY_COLUMNS_EXTENDED",
    "ALL_EXPORT_COLUMNS",
    "DISPLAY_TO_DATA_COLUMN",
    "THUMB_UP_EMPTY",
    "THUMB_UP_FILLED",
    "THUMB_DOWN_EMPTY",
    "THUMB_DOWN_FILLED",
    "ROWS_PER_PAGE",
    "BATCH_INSERT_SIZE",
    "BATCH_INSERT_DELAY_MS",
    "truncate_text",
]
