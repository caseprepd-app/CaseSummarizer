"""
Global Tooltip Manager for CasePrepd.

Session 62b: Centralized tooltip management to ensure only ONE tooltip
is visible at any time across the entire application.

All tooltip implementations should use this manager to coordinate:
- TooltipIcon (settings widgets)
- create_tooltip() helper functions
- Custom tooltips in various UI components

Usage:
    from src.ui.tooltip_manager import tooltip_manager

    # Register a tooltip window when showing
    tooltip_manager.register(my_tooltip_window)

    # Or use the convenience method to close any existing tooltip first
    tooltip_manager.close_active()
    # ... then create your new tooltip
"""

from typing import Optional

import customtkinter as ctk


class TooltipManager:
    """
    Singleton manager ensuring only one tooltip is visible at a time.

    When a new tooltip is about to be shown, any existing tooltip from
    ANY source (settings, vocabulary, main window, etc.) is closed first.

    This solves the problem of multiple tooltips stacking up when the
    user moves the mouse quickly between tooltip-triggering elements.
    """

    _instance: Optional["TooltipManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._active_tooltip = None
            cls._instance._active_owner = None
        return cls._instance

    def register(self, tooltip_window: ctk.CTkToplevel, owner: object = None) -> None:
        """
        Register a tooltip window as the active tooltip.

        Automatically closes any previously active tooltip first.

        Args:
            tooltip_window: The CTkToplevel tooltip window
            owner: Optional owner object (for debugging/tracking)
        """
        # Close any existing tooltip first
        self.close_active()

        # Store weak reference to avoid memory leaks
        self._active_tooltip = tooltip_window
        self._active_owner = owner

    def close_active(self) -> None:
        """
        Close the currently active tooltip if one exists.

        Safe to call even if no tooltip is active.
        """
        if self._active_tooltip is not None:
            try:
                if self._active_tooltip.winfo_exists():
                    self._active_tooltip.destroy()
            except Exception:
                pass  # Window may already be destroyed
            self._active_tooltip = None
            self._active_owner = None

    def unregister(self, tooltip_window: ctk.CTkToplevel) -> None:
        """
        Unregister a tooltip window (called when tooltip is hidden).

        Only clears the active reference if it matches the given window.

        Args:
            tooltip_window: The tooltip window being hidden
        """
        if self._active_tooltip is tooltip_window:
            self._active_tooltip = None
            self._active_owner = None

    def is_active(self, tooltip_window: ctk.CTkToplevel = None) -> bool:
        """
        Check if a tooltip is currently active.

        Args:
            tooltip_window: Optional - check if THIS specific tooltip is active

        Returns:
            True if a tooltip (or the specified tooltip) is active
        """
        if tooltip_window is not None:
            return self._active_tooltip is tooltip_window
        return self._active_tooltip is not None

    @property
    def active_tooltip(self) -> ctk.CTkToplevel | None:
        """Get the currently active tooltip window (if any)."""
        return self._active_tooltip


# Global singleton instance
tooltip_manager = TooltipManager()
