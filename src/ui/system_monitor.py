"""
System Monitor Widget for LocalScribe

Real-time CPU and RAM monitoring with color-coded status indicators.
"""

import platform
import threading

import customtkinter as ctk
import psutil

from src.config import (
    SYSTEM_MONITOR_THRESHOLD_CRITICAL,
    SYSTEM_MONITOR_THRESHOLD_GREEN,
    SYSTEM_MONITOR_THRESHOLD_YELLOW,
)
from src.ui.theme import FONTS, COLORS
from src.ui.tooltip_manager import tooltip_manager


class SystemMonitor(ctk.CTkFrame):
    """
    Real-time system resource monitor with independent color-coded indicators for CPU and RAM.

    Color scheme - applied independently to CPU and RAM using thresholds from config.py:
    - 0 to SYSTEM_MONITOR_THRESHOLD_GREEN: Green (healthy)
    - GREEN to YELLOW threshold: Yellow (elevated)
    - YELLOW to CRITICAL threshold: Orange (high)
    - CRITICAL+: Red with exclamation mark indicator

    Thresholds can be customized in src/config.py.
    """

    def __init__(self, parent=None, update_interval_ms=2000):
        """
        Initialize the system monitor.

        Args:
            parent: Parent widget
            update_interval_ms: Update frequency in milliseconds (default 2000)
                               Increased from 1000ms to reduce main thread load
        """
        super().__init__(parent, fg_color="transparent")
        self.update_interval_ms = update_interval_ms
        self.monitoring = False
        self.tooltip_window = None
        self.show_timer = None
        self._metrics_updated = False
        self.current_cpu = 0
        self.current_ram_used = 0
        self.current_ram_total = 0
        self.current_ram_percent = 0

        # Get CPU info
        try:
            self.cpu_model = platform.processor() or "Unknown CPU"
        except Exception:
            self.cpu_model = "Unknown CPU"

        self.physical_cores = psutil.cpu_count(logical=False) or 1
        self.logical_cores = psutil.cpu_count(logical=True) or 1

        # Create separate frames for CPU and RAM with independent colors
        self.cpu_frame = ctk.CTkFrame(self, fg_color=COLORS["monitor_bg"], corner_radius=4)
        self.cpu_frame.pack(side="left", padx=(5, 2), pady=3)

        self.cpu_label = ctk.CTkLabel(
            self.cpu_frame,
            text="CPU: 0%",
            font=FONTS["tiny"],
            text_color=COLORS["success_light"]
        )
        self.cpu_label.pack(padx=6, pady=2)

        # Separator
        separator = ctk.CTkLabel(self, text="|", font=FONTS["tiny"], text_color=COLORS["text_disabled"])
        separator.pack(side="left", padx=2)

        self.ram_frame = ctk.CTkFrame(self, fg_color=COLORS["monitor_bg"], corner_radius=4)
        self.ram_frame.pack(side="left", padx=(2, 5), pady=3)

        self.ram_label = ctk.CTkLabel(
            self.ram_frame,
            text="RAM: 0%",
            font=FONTS["tiny"],
            text_color=COLORS["success_light"]
        )
        self.ram_label.pack(padx=6, pady=2)

        # Bind tooltip events to both frames
        for widget in [self.cpu_frame, self.cpu_label, self.ram_frame, self.ram_label]:
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)

        # Start monitoring thread
        self.start_monitoring()

    def start_monitoring(self):
        """Start the monitoring thread and main-thread update scheduler."""
        self.monitoring = True
        thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        thread.start()
        # Schedule main-thread update check (non-blocking)
        self._schedule_main_thread_update()

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.monitoring = False

    def _monitoring_loop(self):
        """Background thread that collects metrics."""
        import time
        # Initialize CPU percent tracking (first call returns 0)
        psutil.cpu_percent(interval=None)

        while self.monitoring:
            try:
                # Collect metrics (no GUI updates in background thread)
                self._collect_metrics()
                # Sleep for update interval (metrics collection is now non-blocking)
                time.sleep(self.update_interval_ms / 1000.0)
            except Exception:
                pass  # Silently ignore errors to avoid console spam

    def _collect_metrics(self):
        """Collect current system metrics (background thread safe)."""
        try:
            # NON-BLOCKING: Use interval=None to get CPU since last call
            # This doesn't block - it returns immediately with delta since last measurement
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            ram_used_gb = memory.used / (1024 ** 3)
            ram_total_gb = memory.total / (1024 ** 3)
            ram_percent = memory.percent  # psutil provides this directly

            # Store metrics in instance variables (thread-safe reads)
            # Main thread will periodically read these and update display
            self.current_cpu = cpu_percent
            self.current_ram_used = ram_used_gb
            self.current_ram_total = ram_total_gb
            self.current_ram_percent = ram_percent
            self._metrics_updated = True

        except Exception:
            pass  # Silently ignore errors

    def _schedule_main_thread_update(self):
        """Schedule the next display update check (main thread only)."""
        if not self.monitoring:
            return

        try:
            # Check if background thread has collected new metrics
            if self._metrics_updated:
                self._update_display()
                self._metrics_updated = False
        except Exception:
            pass  # Ignore display errors during heavy processing

        # Schedule next check - use after() which is resilient to busy main thread
        try:
            self.after(self.update_interval_ms, self._schedule_main_thread_update)
        except Exception:
            pass  # Widget may be destroyed

    def _update_display(self):
        """Update the display with stored metrics (main thread only)."""
        try:
            cpu_percent = self.current_cpu
            ram_percent = self.current_ram_percent

            # Format CPU display text with indicator at critical threshold
            cpu_indicator = "!" if cpu_percent >= SYSTEM_MONITOR_THRESHOLD_CRITICAL else ""
            cpu_text = f"CPU: {round(cpu_percent)}%{cpu_indicator}"

            # Format RAM display text with indicator at critical threshold
            ram_indicator = "!" if ram_percent >= SYSTEM_MONITOR_THRESHOLD_CRITICAL else ""
            ram_text = f"RAM: {round(ram_percent)}%{ram_indicator}"

            # Get colors independently for CPU and RAM
            cpu_bg, cpu_fg = self._get_colors(cpu_percent)
            ram_bg, ram_fg = self._get_colors(ram_percent)

            # Update CPU label and frame
            self.cpu_label.configure(text=cpu_text, text_color=cpu_fg)
            self.cpu_frame.configure(fg_color=cpu_bg)

            # Update RAM label and frame
            self.ram_label.configure(text=ram_text, text_color=ram_fg)
            self.ram_frame.configure(fg_color=ram_bg)

        except Exception as e:
            print(f"Error updating display: {e}")

    def _get_colors(self, percent: float) -> tuple:
        """
        Get background and foreground colors based on usage percentage.

        Uses thresholds from config.py (SYSTEM_MONITOR_THRESHOLD_*):
        - 0 to GREEN threshold: Green (healthy)
        - GREEN to YELLOW threshold: Yellow (elevated)
        - YELLOW to CRITICAL threshold: Orange (high)
        - CRITICAL+: Red (critical)

        Args:
            percent: Current usage percentage (CPU or RAM)

        Returns:
            tuple: (bg_color, fg_color)
        """
        if percent < SYSTEM_MONITOR_THRESHOLD_GREEN:
            # Green: healthy
            return ("#1a3a1a", "#90EE90")  # Dark green bg, light green text
        elif percent < SYSTEM_MONITOR_THRESHOLD_YELLOW:
            # Yellow: elevated
            return ("#3a3a1a", "#FFEB3B")  # Dark yellow bg, bright yellow text
        elif percent < SYSTEM_MONITOR_THRESHOLD_CRITICAL:
            # Orange: high
            return ("#3a2a1a", "#FFA500")  # Dark orange bg, bright orange text
        else:
            # Red: critical
            return ("#3a1a1a", "#FF4444")  # Dark red bg, bright red text

    def _on_enter(self, event):
        """Handle mouse enter - schedule tooltip display."""
        if self.show_timer:
            self.after_cancel(self.show_timer)
        self.show_timer = self.after(500, self._show_tooltip)

    def _on_leave(self, event):
        """Handle mouse leave - hide tooltip."""
        if self.show_timer:
            self.after_cancel(self.show_timer)
            self.show_timer = None
        self._hide_tooltip()

    def _show_tooltip(self):
        """Show detailed system information tooltip near the mouse cursor."""
        try:
            if self.tooltip_window:
                return

            # Session 62b: Close any existing tooltip first via global manager
            tooltip_manager.close_active()

            # Get current mouse position (dynamic positioning)
            try:
                mouse_x = self.winfo_pointerx()
                mouse_y = self.winfo_pointery()
            except Exception:
                # Fallback to widget position
                mouse_x = self.winfo_rootx() + self.winfo_width()
                mouse_y = self.winfo_rooty()

            # Get CPU frequency
            try:
                cpu_freq = psutil.cpu_freq()
                freq_text = f"Base: {cpu_freq.current:.1f} GHz"
                if cpu_freq.max:
                    freq_text += f" | Max: {cpu_freq.max:.1f} GHz"
            except Exception:
                freq_text = "Frequency: Unknown"

            # Build tooltip text with RAM percentage and GB breakdown
            ram_percent = round(self.current_ram_percent)
            tooltip_text = (
                f"{self.cpu_model}\n"
                f"{self.physical_cores} physical cores, {self.logical_cores} logical threads\n"
                f"{freq_text}\n"
                f"\n"
                f"Current CPU: {round(self.current_cpu)}%\n"
                f"Current RAM: {ram_percent}% ({self.current_ram_used:.1f} / {self.current_ram_total:.1f} GB)"
            )

            # Create tooltip window
            self.tooltip_window = ctk.CTkToplevel(self.winfo_toplevel())
            self.tooltip_window.wm_overrideredirect(True)
            self.tooltip_window.wm_attributes("-topmost", True)
            try:
                self.tooltip_window.wm_attributes("-toolwindow", True)
            except Exception:
                pass  # Not available on all platforms

            label = ctk.CTkLabel(
                self.tooltip_window,
                text=tooltip_text,
                bg_color=("#333333", "#333333"),
                text_color=("white", "white"),
                corner_radius=5,
                wraplength=280,
                font=FONTS["tiny"],
                justify="left"
            )
            label.pack(padx=8, pady=8)

            # Calculate tooltip size
            self.tooltip_window.update_idletasks()
            tooltip_width = self.tooltip_window.winfo_width()
            tooltip_height = self.tooltip_window.winfo_height()

            # Get screen dimensions (accounting for multi-monitor via vroot)
            try:
                screen_width = self.winfo_screenwidth()
                screen_height = self.winfo_screenheight()
                vroot_x = self.winfo_vrootx()
                vroot_y = self.winfo_vrooty()
            except Exception:
                screen_width, screen_height = 1920, 1080
                vroot_x, vroot_y = 0, 0

            # Position tooltip: prefer above and to the left of cursor (since monitor is usually bottom-right)
            offset_x = 15
            offset_y = 10
            x = mouse_x - tooltip_width - offset_x  # Left of cursor
            y = mouse_y - tooltip_height - offset_y  # Above cursor

            # Boundary checks with repositioning
            if x < vroot_x:
                # Position to the right of cursor instead
                x = mouse_x + offset_x

            if y < vroot_y:
                # Position below cursor instead
                y = mouse_y + offset_y

            # Final constraint to screen bounds
            if x + tooltip_width > screen_width + vroot_x:
                x = screen_width + vroot_x - tooltip_width - 5
            if y + tooltip_height > screen_height + vroot_y:
                y = screen_height + vroot_y - tooltip_height - 5

            x = max(vroot_x, x)
            y = max(vroot_y, y)

            self.tooltip_window.wm_geometry(f"+{int(x)}+{int(y)}")
            self.tooltip_window.lift()

            # Session 62b: Register with global manager
            tooltip_manager.register(self.tooltip_window, owner=self)

        except Exception as e:
            print(f"Tooltip error: {e}")

    def _hide_tooltip(self):
        """Hide the tooltip."""
        if self.tooltip_window:
            # Session 62b: Unregister from global manager
            tooltip_manager.unregister(self.tooltip_window)
            try:
                self.tooltip_window.destroy()
            except Exception:
                pass
            self.tooltip_window = None
