"""
CasePrepd - UI Utility Functions

Re-exports tooltip functionality from tooltip_helper for backward compatibility.
"""

# Re-export tooltip functions from the unified tooltip_helper module
from src.ui.tooltip_helper import create_tooltip, create_tooltip_for_frame

__all__ = ["create_tooltip", "create_tooltip_for_frame"]
