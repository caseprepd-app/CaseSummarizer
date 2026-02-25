"""
Tooltip Helper for CustomTkinter Widgets

Provides stable tooltips that appear near the mouse cursor without flickering.
Uses delayed display (500ms) and intelligent boundary detection for proper positioning.

Uses TooltipManager to ensure only one tooltip is visible at a time across
the entire application.

Best practices implemented:
- Tooltip appears near mouse cursor (not directly under to avoid enter/leave loops)
- Dynamic positioning calculated at show time (handles window resize/move)
- Multi-monitor aware using winfo_vrootx/vrooty
- Screen boundary detection prevents off-screen tooltips
- 500ms delay prevents flickering during mouse movement
- Global singleton ensures only one tooltip visible at a time
"""

import tkinter as tk

import customtkinter as ctk

from src.ui.theme import FONTS
from src.ui.tooltip_manager import tooltip_manager

# Common exceptions during tooltip operations (window destroyed, platform differences)
_TK_ERRORS = (tk.TclError, RuntimeError, AttributeError)


def create_tooltip(widget, text, delay_ms=500, offset_x=15, offset_y=10):
    """
    Create a stable tooltip that appears near the mouse cursor on hover.

    The tooltip appears below and to the right of the cursor by default,
    with automatic repositioning if it would go off-screen.

    Args:
        widget: The widget to attach the tooltip to
        text: The tooltip text to display
        delay_ms: Milliseconds to wait before showing tooltip (default 500)
        offset_x: Horizontal offset from cursor (default 15)
        offset_y: Vertical offset from cursor (default 10)
    """
    tooltip_window = None
    show_timer = None

    def schedule_show(event):
        """Schedule tooltip to appear after delay (prevents flickering)."""
        nonlocal show_timer
        cancel_show()
        # Store mouse position at time of enter for fallback
        show_timer = widget.after(delay_ms, lambda: show_tooltip_at_cursor())

    def cancel_show():
        """Cancel scheduled tooltip display."""
        nonlocal show_timer
        if show_timer:
            widget.after_cancel(show_timer)
            show_timer = None

    def show_tooltip_at_cursor():
        """Display tooltip near current mouse cursor position."""
        nonlocal tooltip_window, show_timer
        show_timer = None

        # Don't create duplicate tooltips
        if tooltip_window:
            return

        # Close any existing tooltip from anywhere in the app
        tooltip_manager.close_active()

        # Get current mouse position (dynamic - calculated at show time)
        try:
            mouse_x = widget.winfo_pointerx()
            mouse_y = widget.winfo_pointery()
        except _TK_ERRORS:
            # Fallback if pointer position unavailable
            mouse_x = widget.winfo_rootx() + widget.winfo_width()
            mouse_y = widget.winfo_rooty()

        # Create tooltip window
        try:
            tooltip_window = ctk.CTkToplevel(widget.winfo_toplevel())
        except _TK_ERRORS:
            return

        tooltip_window.wm_overrideredirect(True)
        tooltip_window.wm_attributes("-topmost", True)
        import contextlib

        with contextlib.suppress(_TK_ERRORS):
            tooltip_window.wm_attributes("-toolwindow", True)  # Windows-specific

        # Register with global manager
        tooltip_manager.register(tooltip_window, owner=widget)

        # Create tooltip label
        label = ctk.CTkLabel(
            tooltip_window,
            text=text,
            bg_color=("#333333", "#333333"),
            text_color=("white", "white"),
            corner_radius=5,
            wraplength=250,
            font=FONTS["small"],
            justify="left",
        )
        label.pack(padx=8, pady=6)

        # Calculate tooltip size
        tooltip_window.update_idletasks()
        tooltip_width = tooltip_window.winfo_width()
        tooltip_height = tooltip_window.winfo_height()

        # Get screen dimensions (accounting for multi-monitor via vroot)
        try:
            # Total virtual screen dimensions
            screen_width = widget.winfo_screenwidth()
            screen_height = widget.winfo_screenheight()
            # Offset for multi-monitor setups
            vroot_x = widget.winfo_vrootx()
            vroot_y = widget.winfo_vrooty()
        except _TK_ERRORS:
            screen_width = 1920
            screen_height = 1080
            vroot_x = 0
            vroot_y = 0

        # Calculate position: prefer below-right of cursor
        x = mouse_x + offset_x
        y = mouse_y + offset_y

        # Boundary checks with repositioning logic
        # Check right boundary
        if x + tooltip_width > screen_width + vroot_x:
            # Position to the left of cursor instead
            x = mouse_x - tooltip_width - offset_x

        # Check left boundary
        if x < vroot_x:
            x = vroot_x + 5

        # Check bottom boundary
        if y + tooltip_height > screen_height + vroot_y:
            # Position above cursor instead
            y = mouse_y - tooltip_height - offset_y

        # Check top boundary
        if y < vroot_y:
            y = vroot_y + 5

        # Final constraint to screen bounds
        x = max(vroot_x, min(x, screen_width + vroot_x - tooltip_width - 5))
        y = max(vroot_y, min(y, screen_height + vroot_y - tooltip_height - 5))

        # Position and display
        tooltip_window.wm_geometry(f"+{int(x)}+{int(y)}")
        tooltip_window.lift()

    def hide_tooltip(event=None):
        """Hide tooltip immediately."""
        nonlocal tooltip_window
        cancel_show()
        if tooltip_window:
            # Unregister from global manager
            tooltip_manager.unregister(tooltip_window)
            import contextlib

            with contextlib.suppress(_TK_ERRORS):
                tooltip_window.destroy()
            tooltip_window = None

    def on_enter(event):
        """Handle mouse enter on widget."""
        schedule_show(event)

    def on_leave(event):
        """Handle mouse leave from widget."""
        hide_tooltip(event)

    # Bind events
    widget.bind("<Enter>", on_enter, add="+")
    widget.bind("<Leave>", on_leave, add="+")

    # Return hide function for external control if needed
    return hide_tooltip
