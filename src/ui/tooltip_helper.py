"""
Tooltip Helper for CustomTkinter Widgets

Provides stable tooltips that appear near the mouse cursor without flickering.
Uses delayed display (500ms) and intelligent boundary detection for proper positioning.

Session 62b: Now uses TooltipManager to ensure only ONE tooltip is visible
at a time across the entire application.

Best practices implemented:
- Tooltip appears near mouse cursor (not directly under to avoid enter/leave loops)
- Dynamic positioning calculated at show time (handles window resize/move)
- Multi-monitor aware using winfo_vrootx/vrooty
- Screen boundary detection prevents off-screen tooltips
- 500ms delay prevents flickering during mouse movement
- Global singleton ensures only one tooltip visible at a time
"""

import customtkinter as ctk

from src.ui.theme import FONTS
from src.ui.tooltip_manager import tooltip_manager


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

        # Session 62b: Close any existing tooltip from anywhere in the app
        tooltip_manager.close_active()

        # Get current mouse position (dynamic - calculated at show time)
        try:
            mouse_x = widget.winfo_pointerx()
            mouse_y = widget.winfo_pointery()
        except Exception:
            # Fallback if pointer position unavailable
            mouse_x = widget.winfo_rootx() + widget.winfo_width()
            mouse_y = widget.winfo_rooty()

        # Create tooltip window
        try:
            tooltip_window = ctk.CTkToplevel(widget.winfo_toplevel())
        except Exception:
            return

        tooltip_window.wm_overrideredirect(True)
        tooltip_window.wm_attributes("-topmost", True)
        try:
            tooltip_window.wm_attributes("-toolwindow", True)  # Windows-specific
        except Exception:
            pass  # Not available on all platforms

        # Session 62b: Register with global manager
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
            justify="left"
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
        except Exception:
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
            # Session 62b: Unregister from global manager
            tooltip_manager.unregister(tooltip_window)
            try:
                tooltip_window.destroy()
            except Exception:
                pass
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


def create_tooltip_for_frame(frame, text, child_widgets=None, delay_ms=500, offset_x=15, offset_y=10):
    """
    Create a tooltip for a frame and optionally its child widgets.

    Useful when you want hovering over any part of a frame (or its children)
    to show the same tooltip.

    Args:
        frame: The parent frame widget
        text: The tooltip text to display
        child_widgets: Optional list of child widgets to also bind (default: None)
        delay_ms: Milliseconds to wait before showing tooltip
        offset_x: Horizontal offset from cursor
        offset_y: Vertical offset from cursor
    """
    tooltip_window = None
    show_timer = None
    hover_count = 0  # Track enter/leave across frame and children

    def schedule_show(event):
        nonlocal show_timer, hover_count
        hover_count += 1
        if show_timer is None:
            show_timer = frame.after(delay_ms, show_tooltip_at_cursor)

    def cancel_show():
        nonlocal show_timer
        if show_timer:
            frame.after_cancel(show_timer)
            show_timer = None

    def show_tooltip_at_cursor():
        nonlocal tooltip_window, show_timer
        show_timer = None

        if tooltip_window or hover_count <= 0:
            return

        # Session 62b: Close any existing tooltip from anywhere in the app
        tooltip_manager.close_active()

        try:
            mouse_x = frame.winfo_pointerx()
            mouse_y = frame.winfo_pointery()
        except Exception:
            mouse_x = frame.winfo_rootx() + frame.winfo_width()
            mouse_y = frame.winfo_rooty()

        try:
            tooltip_window = ctk.CTkToplevel(frame.winfo_toplevel())
        except Exception:
            return

        tooltip_window.wm_overrideredirect(True)
        tooltip_window.wm_attributes("-topmost", True)
        try:
            tooltip_window.wm_attributes("-toolwindow", True)
        except Exception:
            pass

        # Session 62b: Register with global manager
        tooltip_manager.register(tooltip_window, owner=frame)

        label = ctk.CTkLabel(
            tooltip_window,
            text=text,
            bg_color=("#333333", "#333333"),
            text_color=("white", "white"),
            corner_radius=5,
            wraplength=250,
            font=FONTS["small"],
            justify="left"
        )
        label.pack(padx=8, pady=6)

        tooltip_window.update_idletasks()
        tooltip_width = tooltip_window.winfo_width()
        tooltip_height = tooltip_window.winfo_height()

        try:
            screen_width = frame.winfo_screenwidth()
            screen_height = frame.winfo_screenheight()
            vroot_x = frame.winfo_vrootx()
            vroot_y = frame.winfo_vrooty()
        except Exception:
            screen_width, screen_height = 1920, 1080
            vroot_x, vroot_y = 0, 0

        x = mouse_x + offset_x
        y = mouse_y + offset_y

        if x + tooltip_width > screen_width + vroot_x:
            x = mouse_x - tooltip_width - offset_x
        if x < vroot_x:
            x = vroot_x + 5
        if y + tooltip_height > screen_height + vroot_y:
            y = mouse_y - tooltip_height - offset_y
        if y < vroot_y:
            y = vroot_y + 5

        x = max(vroot_x, min(x, screen_width + vroot_x - tooltip_width - 5))
        y = max(vroot_y, min(y, screen_height + vroot_y - tooltip_height - 5))

        tooltip_window.wm_geometry(f"+{int(x)}+{int(y)}")
        tooltip_window.lift()

    def hide_tooltip(event=None):
        nonlocal tooltip_window, hover_count
        hover_count = max(0, hover_count - 1)
        # Only hide if we've left all widgets
        if hover_count <= 0:
            cancel_show()
            if tooltip_window:
                # Session 62b: Unregister from global manager
                tooltip_manager.unregister(tooltip_window)
                try:
                    tooltip_window.destroy()
                except Exception:
                    pass
                tooltip_window = None

    def force_hide(event=None):
        """Force hide regardless of hover count."""
        nonlocal tooltip_window, hover_count
        hover_count = 0
        cancel_show()
        if tooltip_window:
            # Session 62b: Unregister from global manager
            tooltip_manager.unregister(tooltip_window)
            try:
                tooltip_window.destroy()
            except Exception:
                pass
            tooltip_window = None

    # Bind to frame
    frame.bind("<Enter>", schedule_show, add="+")
    frame.bind("<Leave>", hide_tooltip, add="+")

    # Bind to child widgets if provided
    if child_widgets:
        for child in child_widgets:
            child.bind("<Enter>", schedule_show, add="+")
            child.bind("<Leave>", hide_tooltip, add="+")

    return force_hide
