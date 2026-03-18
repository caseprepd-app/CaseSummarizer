"""
Tooltip Helper for CustomTkinter Widgets

Provides stable tooltips that appear near the mouse cursor without flickering.
Uses delayed display (500ms) and intelligent boundary detection for proper positioning.

Uses TooltipManager to ensure only one tooltip is visible at a time across
the entire application.

Best practices (based on IDLE, ttkbootstrap, CTkToolTip):
- Tooltip appears near mouse cursor (offset prevents enter/leave loops)
- Dynamic positioning calculated at show time (handles window resize/move)
- Multi-monitor aware using winfo_vrootx/vrooty
- Screen boundary detection prevents off-screen tooltips
- 500ms delay prevents flickering during mouse movement
- Global singleton ensures only one tooltip visible at a time
- <ButtonPress> hides tooltip (prevents obscuring click targets)
- <Destroy> cleans up tooltip when widget is destroyed
- All timers cancelled on every event handler entry (prevents stale callbacks)
"""

import contextlib
import tkinter as tk

import customtkinter as ctk

from src.ui.theme import FONTS
from src.ui.tooltip_manager import tooltip_manager

# Common exceptions during tooltip operations (window destroyed, platform differences)
_TK_ERRORS = (tk.TclError, RuntimeError, AttributeError)


def create_tooltip(widget, text, delay_ms=500, offset_x=15, offset_y=10):
    """
    Create a stable tooltip that appears near the mouse cursor on hover.

    Binds <Enter>, <Leave>, <ButtonPress>, and <Destroy> with add="+"
    so existing bindings are preserved.

    Args:
        widget: The widget to attach the tooltip to
        text: The tooltip text to display
        delay_ms: Milliseconds to wait before showing tooltip (default 500)
        offset_x: Horizontal offset from cursor (default 15)
        offset_y: Vertical offset from cursor (default 10)
    """
    tooltip_window = None
    show_timer = None

    def cancel_show():
        """Cancel scheduled tooltip display."""
        nonlocal show_timer
        if show_timer:
            with contextlib.suppress(_TK_ERRORS):
                widget.after_cancel(show_timer)
            show_timer = None

    def show_tooltip_at_cursor():
        """Display tooltip near current mouse cursor position."""
        nonlocal tooltip_window, show_timer
        show_timer = None

        if tooltip_window:
            return

        # Verify widget still exists before showing
        try:
            if not widget.winfo_exists():
                return
        except _TK_ERRORS:
            return

        tooltip_manager.close_active()

        try:
            mouse_x = widget.winfo_pointerx()
            mouse_y = widget.winfo_pointery()
        except _TK_ERRORS:
            mouse_x = widget.winfo_rootx() + widget.winfo_width()
            mouse_y = widget.winfo_rooty()

        try:
            tooltip_window = ctk.CTkToplevel(widget.winfo_toplevel())
        except _TK_ERRORS:
            return

        tooltip_window.wm_overrideredirect(True)
        tooltip_window.wm_attributes("-topmost", True)
        with contextlib.suppress(_TK_ERRORS):
            tooltip_window.wm_attributes("-toolwindow", True)

        tooltip_manager.register(tooltip_window, owner=widget)

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

        tooltip_window.update_idletasks()
        tooltip_width = tooltip_window.winfo_width()
        tooltip_height = tooltip_window.winfo_height()

        try:
            screen_width = widget.winfo_screenwidth()
            screen_height = widget.winfo_screenheight()
            vroot_x = widget.winfo_vrootx()
            vroot_y = widget.winfo_vrooty()
        except _TK_ERRORS:
            screen_width = 1920
            screen_height = 1080
            vroot_x = 0
            vroot_y = 0

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
        """Hide tooltip immediately and cancel any pending show."""
        nonlocal tooltip_window
        cancel_show()
        if tooltip_window:
            tooltip_manager.unregister(tooltip_window)
            with contextlib.suppress(_TK_ERRORS):
                tooltip_window.destroy()
            tooltip_window = None

    def on_enter(event):
        """Handle mouse enter — schedule tooltip after delay."""
        nonlocal show_timer
        cancel_show()
        show_timer = widget.after(delay_ms, show_tooltip_at_cursor)

    def on_leave(event):
        """Handle mouse leave — hide immediately."""
        hide_tooltip()

    # Bind all four events with add="+" to preserve existing bindings
    widget.bind("<Enter>", on_enter, add="+")
    widget.bind("<Leave>", on_leave, add="+")
    widget.bind("<ButtonPress>", on_leave, add="+")
    widget.bind("<Destroy>", hide_tooltip, add="+")

    return hide_tooltip
