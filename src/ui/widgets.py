"""
CasePrepd - Custom UI Widgets

This module contains reusable CustomTkinter widget components for the main application:
- FileReviewTable: Displays document processing results in a table format
"""

import logging
from tkinter import ttk

import customtkinter as ctk

logger = logging.getLogger(__name__)

from src.ui.theme import COLORS, FILE_STATUS_TAGS, FONTS


class FileReviewTable(ctk.CTkFrame):
    """
    Custom table widget for displaying document processing results,
    refactored with CustomTkinter using a tkinter.ttk.Treeview.
    """

    def __init__(self, master, on_remove=None, on_select=None, **kwargs):
        """
        Args:
            master: Parent widget.
            on_remove: Optional callback(filename) invoked when the ✕ button is clicked.
            on_select: Optional callback(filename) invoked when a data column is clicked.
        """
        super().__init__(master, **kwargs)

        self._on_remove = on_remove
        self._on_select = on_select

        self.column_map = {
            "filename": ("Filename", 300),
            "status": ("Status", 100),
            "method": ("Method", 100),
            "confidence": ("Confidence", 100),
            "pages": ("Pages", 50),
            "size": ("Size", 80),
        }

        self._create_treeview()
        self.file_item_map = {}  # To map filename to treeview item ID
        self._result_data = {}  # Map filename -> full result dict (for hover previews)
        self._hovered_row = None  # Track currently hovered row for tooltip
        self._tooltip_window = None  # Single tooltip window for hover previews

        # Empty state overlay -- framed drop zone
        self._drop_zone = ctk.CTkFrame(
            self,
            fg_color=COLORS["drop_zone_idle_bg"],
            border_width=2,
            border_color=COLORS["drop_zone_idle_border"],
            corner_radius=10,
        )
        self._drop_zone_label = ctk.CTkLabel(
            self._drop_zone,
            text="\U0001f4c4  Drop files here or click + Add Files",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
        )
        self._drop_zone_label.pack(expand=True, pady=(0, 2))
        self._drop_zone_hint = ctk.CTkLabel(
            self._drop_zone,
            text="Supported: PDF, DOCX, TXT, RTF",
            font=FONTS["small"],
            text_color=COLORS["text_disabled"],
        )
        self._drop_zone_hint.pack()
        self._drop_zone.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.85, relheight=0.7)

    def _create_remove_icon(self):
        """Create an orange ✕ icon as a PhotoImage for the remove column."""
        import tkinter as tk

        # 9x9 XBM bitmap for ✕ shape
        xbm_data = (
            "#define x_width 9\n"
            "#define x_height 9\n"
            "static unsigned char x_bits[] = {\n"
            "   0x01, 0x01, 0x82, 0x00, 0x44, 0x00, 0x28, 0x00,\n"
            "   0x10, 0x00, 0x28, 0x00, 0x44, 0x00, 0x82, 0x00,\n"
            "   0x01, 0x01};"
        )
        # Use orange foreground on transparent background
        from src.ui.theme import get_color

        icon = tk.BitmapImage(data=xbm_data, foreground=get_color("remove_icon"), background="")
        return icon

    def _create_treeview(self):
        """Create the Treeview widget."""
        # Create orange ✕ icon (must keep reference to prevent GC)
        self._remove_icon = self._create_remove_icon()

        # Style configured centrally in src/ui/styles.py at app startup
        # show="tree headings" enables tree column (#0) for the ✕ icon
        self.tree = ttk.Treeview(self, columns=list(self.column_map.keys()), show="tree headings")

        # Tree column (#0) — narrow, holds the orange ✕ icon
        self.tree.heading("#0", text="", anchor="w")
        self.tree.column("#0", width=36, minwidth=36, stretch=False)

        for col_id, (text, width) in self.column_map.items():
            self.tree.heading(col_id, text=text, anchor="w")
            self.tree.column(col_id, width=width, anchor="w")

        self.tree.pack(expand=True, fill="both")

        # Click handler for remove column
        self.tree.bind("<ButtonRelease-1>", self._on_click)

        # Hover preview bindings — <ButtonPress> hides tooltip on click
        self.tree.bind("<Motion>", self._on_hover)
        self.tree.bind("<Leave>", self._on_leave)
        self.tree.bind("<ButtonPress>", lambda e: self._hide_tooltip(), add="+")

    def add_pending_file(self, filename, file_path):
        """
        Insert a placeholder row showing a file as 'Extracting...' in purple.

        Called immediately when files are selected, before the subprocess begins
        extraction. When add_result() is called later with the same filename,
        the existing row is updated in-place.

        Args:
            filename: Display name of the file (e.g. "report.pdf").
            file_path: Full path to the file, used to show file size.
        """
        import os

        # Hide empty state overlay on first file
        if not self.file_item_map and self._drop_zone.winfo_ismapped():
            self._drop_zone.place_forget()

        # Calculate file size for display
        try:
            size_bytes = os.path.getsize(file_path)
            size_display = self._format_file_size(size_bytes)
        except OSError:
            size_display = "—"
        except Exception as e:
            logger.error("Unexpected error getting file size: %s", e, exc_info=True)
            size_display = "—"

        values = (filename, "Extracting...", "—", "—", "—", size_display)
        tag = "extracting"

        if filename in self.file_item_map:
            item_id = self.file_item_map[filename]
            self.tree.item(item_id, values=values, image=self._remove_icon, tags=(tag,))
        else:
            item_id = self.tree.insert(
                "", "end", values=values, image=self._remove_icon, tags=(tag,)
            )
            self.file_item_map[filename] = item_id

        from src.ui.theme import resolve_tags

        for tag_name, tag_config in resolve_tags(FILE_STATUS_TAGS).items():
            self.tree.tag_configure(tag_name, **tag_config)

    def add_result(self, result):
        """Add or update a processing result in the table."""
        filename = result.get("filename", "Unknown")

        # Hide empty state overlay on first file
        if not self.file_item_map and self._drop_zone.winfo_ismapped():
            self._drop_zone.place_forget()

        # Store full result data for hover previews
        self._result_data[filename] = result

        values, status_color_tag = self._prepare_result_for_display(result)

        if filename in self.file_item_map:
            # Update existing item
            item_id = self.file_item_map[filename]
            self.tree.item(
                item_id, values=values, image=self._remove_icon, tags=(status_color_tag,)
            )
        else:
            # Insert new item with orange ✕ icon in tree column
            item_id = self.tree.insert(
                "", "end", values=values, image=self._remove_icon, tags=(status_color_tag,)
            )
            self.file_item_map[filename] = item_id

        # Configure colors for tags
        from src.ui.theme import resolve_tags

        for tag_name, tag_config in resolve_tags(FILE_STATUS_TAGS).items():
            self.tree.tag_configure(tag_name, **tag_config)

    def _prepare_result_for_display(self, result):
        """Prepares result data for display in the treeview."""
        status = result.get("status", "error")
        confidence = result.get("confidence", 0)

        status_text, status_color_tag = self._get_status_display(status, confidence)
        method_display = self._get_method_display(result.get("method", "unknown"))
        confidence_display = f"{confidence}%" if status != "error" and confidence > 0 else "—"
        pages_display = str(result.get("page_count", 0)) if result.get("page_count") else "—"
        size_display = (
            self._format_file_size(result.get("file_size", 0)) if status != "error" else "—"
        )

        values = (
            result.get("filename", "Unknown"),
            status_text,
            method_display,
            confidence_display,
            pages_display,
            size_display,
        )
        return values, status_color_tag

    def _get_status_display(self, status, confidence):
        if status == "error":
            return ("✗ Failed", "red")
        if status == "pending":
            return ("... Pending", "pending")
        if status == "success" and confidence >= 70:
            return ("✓ Ready", "green")
        return ("⚠ Low Quality", "yellow")

    def _get_method_display(self, method):
        """Format extraction method name for display."""
        if not method:
            return "Unknown"
        return method.replace("_", " ").title()

    def _format_file_size(self, size_bytes):
        if size_bytes == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB"]
        size = float(size_bytes)
        unit_index = 0
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        # Round to nearest integer for all units
        return f"{round(size)} {units[unit_index]}"

    def clear(self):
        """Clear all items and show empty state overlay."""
        self.file_item_map.clear()
        self._result_data.clear()
        self._hide_tooltip()
        for item in self.tree.get_children():
            self.tree.delete(item)
        # Show empty state overlay again
        self._drop_zone.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.85, relheight=0.7)

    def _on_click(self, event):
        """Handle click on the treeview — remove or select based on column."""
        row_id = self.tree.identify_row(event.y)
        if not row_id:
            return

        col = self.tree.identify_column(event.x)

        try:
            values = self.tree.item(row_id, "values")
            filename = values[0] if values else None
        except Exception:
            return

        if not filename:
            return

        if col == "#0" and self._on_remove:
            # Tree column (#0) holds the orange ✕ icon
            self._on_remove(filename)
        elif col != "#0" and self._on_select:
            # Any data column click triggers selection
            self.tree.selection_set(row_id)
            self._on_select(filename)

    def remove_result(self, filename):
        """
        Remove a single file from the table by filename.

        Args:
            filename: The filename to remove.
        """
        item_id = self.file_item_map.pop(filename, None)
        self._result_data.pop(filename, None)

        if item_id:
            self._hide_tooltip()
            try:
                self.tree.delete(item_id)
            except Exception:
                logger.debug("Tree item delete failed for %s", item_id)

        # Show drop zone if table is now empty
        if not self.file_item_map and not self._drop_zone.winfo_ismapped():
            self._drop_zone.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.85, relheight=0.7)

    def _on_hover(self, event):
        """Show a tooltip with file details when hovering over a row."""
        row_id = self.tree.identify_row(event.y)
        if not row_id or row_id == self._hovered_row:
            return

        self._hovered_row = row_id
        # Get filename from the row values (column index 0)
        try:
            values = self.tree.item(row_id, "values")
            filename = values[0] if values else None
        except Exception:
            return

        result = self._result_data.get(filename)
        if not result:
            return

        # Build tooltip text
        lines = []
        file_path = result.get("file_path", result.get("filepath", ""))
        if file_path:
            lines.append(f"Path: {file_path}")
        word_count = result.get("word_count", 0)
        if word_count:
            lines.append(f"Words: {word_count:,}")
        page_count = result.get("page_count", 0)
        if page_count:
            lines.append(f"Pages: {page_count}")
        method = result.get("method", "")
        if method:
            lines.append(f"Method: {method.replace('_', ' ').title()}")
        case_numbers = result.get("case_numbers", [])
        if case_numbers:
            lines.append(f"Case #: {', '.join(case_numbers[:3])}")

        if not lines:
            return

        tooltip_text = "\n".join(lines)
        self._show_tooltip(event.x_root + 15, event.y_root + 10, tooltip_text)

    def _on_leave(self, _event):
        """Hide tooltip when mouse leaves the treeview."""
        self._hovered_row = None
        self._hide_tooltip()

    def _show_tooltip(self, x, y, text):
        """Display a tooltip at the given screen coordinates."""
        self._hide_tooltip()

        # Close any tooltip from other UI components (e.g. settings, system monitor)
        from src.ui.tooltip_manager import tooltip_manager

        tooltip_manager.close_active()

        try:
            self._tooltip_window = ctk.CTkToplevel(self.winfo_toplevel())
        except Exception:
            return
        self._tooltip_window.wm_overrideredirect(True)
        self._tooltip_window.wm_attributes("-topmost", True)
        import contextlib

        with contextlib.suppress(Exception):
            self._tooltip_window.wm_attributes("-toolwindow", True)

        label = ctk.CTkLabel(
            self._tooltip_window,
            text=text,
            bg_color=COLORS["tooltip_bg"],
            text_color=COLORS["tooltip_fg"],
            corner_radius=5,
            wraplength=350,
            font=FONTS["small"],
            justify="left",
        )
        label.pack(padx=8, pady=6)
        self._tooltip_window.wm_geometry(f"+{x}+{y}")

        # Register with global tooltip manager
        tooltip_manager.register(self._tooltip_window, owner=self)

        # Cancel any previously-scheduled auto-dismiss before scheduling a new
        # one. Without this, rapid re-show (e.g. hover cycling) leaks after-IDs
        # that fire on whichever tooltip happens to be current.
        previous_dismiss = getattr(self, "_tooltip_dismiss_id", None)
        if previous_dismiss:
            with contextlib.suppress(Exception):
                self.after_cancel(previous_dismiss)
            self._tooltip_dismiss_id = None

        # Auto-dismiss after 15 seconds (handles minimized window, etc.)
        try:
            self._tooltip_dismiss_id = self.after(15000, self._hide_tooltip)
        except Exception:
            self._tooltip_dismiss_id = None

    def _hide_tooltip(self):
        """Destroy the tooltip window if it exists."""
        # Cancel auto-dismiss timer
        dismiss_id = getattr(self, "_tooltip_dismiss_id", None)
        if dismiss_id:
            import contextlib

            with contextlib.suppress(Exception):
                self.after_cancel(dismiss_id)
            self._tooltip_dismiss_id = None

        if self._tooltip_window:
            from src.ui.tooltip_manager import tooltip_manager

            tooltip_manager.unregister(self._tooltip_window)
            import contextlib

            with contextlib.suppress(Exception):
                self._tooltip_window.destroy()
            self._tooltip_window = None
