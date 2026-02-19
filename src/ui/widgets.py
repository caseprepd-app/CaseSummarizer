"""
CasePrepd - Custom UI Widgets

This module contains reusable CustomTkinter widget components for the main application:
- FileReviewTable: Displays document processing results in a table format

Note: System monitoring is handled by src/ui/system_monitor.py (not in this module).
"""

from tkinter import ttk

import customtkinter as ctk

from src.ui.theme import COLORS, FILE_STATUS_TAGS, FONTS


class FileReviewTable(ctk.CTkFrame):
    """
    Custom table widget for displaying document processing results,
    refactored with CustomTkinter using a tkinter.ttk.Treeview.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.column_map = {
            "include": ("Include", 50),
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

    def _create_treeview(self):
        """Create the Treeview widget."""
        # Style configured centrally in src/ui/styles.py at app startup
        self.tree = ttk.Treeview(self, columns=list(self.column_map.keys()), show="headings")

        for col_id, (text, width) in self.column_map.items():
            self.tree.heading(col_id, text=text, anchor="w")
            self.tree.column(col_id, width=width, anchor="w")

        self.tree.pack(expand=True, fill="both")

        # Hover preview bindings
        self.tree.bind("<Motion>", self._on_hover)
        self.tree.bind("<Leave>", self._on_leave)

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
            self.tree.item(item_id, values=values, tags=(status_color_tag,))
        else:
            # Insert new item
            item_id = self.tree.insert("", "end", values=values, tags=(status_color_tag,))
            self.file_item_map[filename] = item_id

        # Configure colors for tags
        for tag_name, tag_config in FILE_STATUS_TAGS.items():
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

        # Using a placeholder for the "Include" checkbox for now
        include_display = "✓" if status == "success" and confidence >= 70 else " "

        values = (
            include_display,
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

    def _on_hover(self, event):
        """Show a tooltip with file details when hovering over a row."""
        row_id = self.tree.identify_row(event.y)
        if not row_id or row_id == self._hovered_row:
            return

        self._hovered_row = row_id
        # Get filename from the row values (column index 1)
        try:
            values = self.tree.item(row_id, "values")
            filename = values[1] if values and len(values) > 1 else None
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
            bg_color=("#333333", "#333333"),
            text_color=("white", "white"),
            corner_radius=5,
            wraplength=350,
            font=FONTS["small"],
            justify="left",
        )
        label.pack(padx=8, pady=6)
        self._tooltip_window.wm_geometry(f"+{x}+{y}")

    def _hide_tooltip(self):
        """Destroy the tooltip window if it exists."""
        if self._tooltip_window:
            import contextlib

            with contextlib.suppress(Exception):
                self._tooltip_window.destroy()
            self._tooltip_window = None
