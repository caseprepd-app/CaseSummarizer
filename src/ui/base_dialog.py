"""
Base Modal Dialog for CasePrepd.

Provides a base class for modal dialogs with common boilerplate:
- Modal behavior (transient + grab_set)
- Window centering on parent
- Configurable size and minimum size

Session 82: Extracted from repeated patterns in corpus_dialog.py,
qa_question_editor.py, and settings_dialog.py to eliminate duplication.

Usage:
    class MyDialog(BaseModalDialog):
        def __init__(self, parent):
            super().__init__(
                parent=parent,
                title="My Dialog",
                width=600,
                height=400,
                min_width=400,
                min_height=300,
            )
            self._create_ui()

        def _create_ui(self):
            # Build your dialog UI here
            pass
"""

import customtkinter as ctk


class BaseModalDialog(ctk.CTkToplevel):
    """
    Base class for modal dialogs with centering behavior.

    Handles common dialog setup:
    - Window title and geometry
    - Modal behavior (blocks parent interaction)
    - Automatic centering on parent window
    - Configurable minimum size

    Subclasses should call super().__init__() then build their UI.

    Attributes:
        parent: The parent window this dialog is modal to.
    """

    def __init__(
        self,
        parent,
        title: str,
        width: int,
        height: int,
        min_width: int | None = None,
        min_height: int | None = None,
        resizable: bool = True,
    ):
        """
        Initialize the modal dialog.

        Args:
            parent: Parent window (dialog will be modal to this)
            title: Window title
            width: Initial window width in pixels
            height: Initial window height in pixels
            min_width: Minimum width (default: None, no minimum)
            min_height: Minimum height (default: None, no minimum)
            resizable: Whether the dialog can be resized (default: True)

        Example:
            super().__init__(
                parent=parent,
                title="Edit Settings",
                width=700,
                height=500,
                min_width=500,
                min_height=400,
            )
        """
        super().__init__(parent)
        self.parent = parent

        # Scale dimensions for high-DPI displays
        from src.ui.scaling import scale_value

        width = scale_value(width)
        height = scale_value(height)

        # Window configuration
        self.title(title)
        self.geometry(f"{width}x{height}")
        self.resizable(resizable, resizable)

        if min_width and min_height:
            self.minsize(scale_value(min_width), scale_value(min_height))

        # Make modal
        if parent:
            self.transient(parent)
        self.grab_set()

        # Center on parent
        self._center_on_parent(width, height)

    def _center_on_parent(self, width: int, height: int) -> None:
        """
        Center the dialog on its parent window.

        Falls back to screen center if no parent.

        Args:
            width: Dialog width for centering calculation
            height: Dialog height for centering calculation
        """
        self.update_idletasks()

        if self.parent:
            parent_x = self.parent.winfo_x()
            parent_y = self.parent.winfo_y()
            parent_w = self.parent.winfo_width()
            parent_h = self.parent.winfo_height()

            x = parent_x + (parent_w - width) // 2
            y = parent_y + (parent_h - height) // 2
        else:
            # Center on screen if no parent
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            x = (screen_w - width) // 2
            y = (screen_h - height) // 2

        self.geometry(f"+{x}+{y}")

    def close(self) -> None:
        """
        Close the dialog.

        Convenience method for subclasses to call on cancel/close.
        """
        self.destroy()
